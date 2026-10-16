import torch 
import torch.nn as nn 
from torch.cuda.amp import autocast, GradScaler
from contextlib import nullcontext
import time 
from pathlib import Path 
from src.core.utils import LRScheduler, save_checkpoint, load_checkpoint, compute_preplx

class Trainer:
    def __init__(self, model, train_loader, val_loader, config, device='cuda'):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config 
        self.device = device

        # usign AdamW(standar) will try more if needed 
        self.optimizer = torch.optim.AdamW(model.parameters(),lr = config.lr, weight_decay = config.weight_decay)

        steps_per_epoch = len(train_loader) // config.accumulation_steps
        total_steps = steps_per_epoch * config.epochs

        # warmup + cosine decay scheduler driven by optimzer step count 
        self.scheduler = LRScheduler(
            self.optimizer,
            peak_lr=config.lr,
            warmup_steps= config.warmup_steps,
            total_steps=total_steps
        )

        # GradScaler for mixed precision (only enabled on cuda)
        self.scaler = GradScaler(enabled=(device == 'cuda'))
        
        # global optimization step counter 
        self.step = 0
        self.epoch = 0

        # track best validation loss for model selection 
        self.best_val_loss = float('inf')

        # ensure checkpoint directory exists before training starts 
        Path(config.checkpoint_dir).mkdir(parents=True, exist_ok=True)

    def train_epoch(self):
        # enable dropout
        self.model.train()

        total_loss = 0 
        num_steps = 0
        # gradients accumulate across multiple forward passes
        self.optimizer.zero_grad()

        for i, batch in enumerate(self.train_loader):
            input_ids = batch['input_ids'].to(self.device)
            labels = batch['labels'].to(self.device)

            # mixed precision on cuda, plain context on cpu
            amp_ctx = autocast if self.device == 'cuda' else nullcontext
            with amp_ctx():
                outputs = self.model(input_ids, targets=labels)
                # dividing loss so accumulated gradientns math full-batch 

                loss = outputs['loss'] / self.config.accumulation_steps
            
            # backprop with scaled loss to preserve precision 
            self.scaler.scale(loss).backward()

            if (i + 1) % self.config.accumulation_steps == 0:
                # unsclae before clipping so norms are meaningful 
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.max_grad_norm
                )

                # optimizer step under GradScaler control
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()
                
                # advance lr schedule once per optimizer update
                lr = self.scheduler.step()
                self.step += 1
                num_steps += 1

                # accumulate loss for this optimizer step
                total_loss += loss.item() * self.config.accumulation_steps

                # lightweight training signal for miniitoring 
                if self.step % 10 == 0:
                    print(f"Step {self.step} | "f"Loss: {loss.item() * self.config.accumulation_steps:.4f} | "f"LR: {lr:.6f}")
                
                # predioc validation and best-model track
                if self.step % self.config.eval_every == 0:
                    val_loss = self.validate()
                    
                    if val_loss < self.best_val_loss:
                        self.best_val_loss = val_loss
                        save_checkpoint(
                            f"{self.config.checkpoint_dir}/best_model.pt",
                            self.model,
                            self.optimizer,
                            self.step,
                            self.epoch,
                            val_loss
                        )
                        print(f"saved best model ( val_loss : {val_loss:.4f})")

                if self.step % self.config.save_every == 0:
                    save_checkpoint(
                        f"{self.config.checkpoint_dir}/checkpoint_{self.step}.pt",
                        self.model,
                        self.optimizer,
                        self.step,
                        self.epoch,
                        loss.item() * self.config.accumulation_steps
                    )

        # avg loss
        return total_loss / num_steps if num_steps > 0 else 0
    
    @torch.no_grad()
    def validate(self):
        # disable dropout and gradient tracking 
        self.model.eval()
        total_loss = 0 

        for batch in self.val_loader:
            input_ids = batch['input_ids'].to(self.device)
            labels = batch['labels'].to(self.device)

            outputs = self.model(input_ids, targets = labels)
            total_loss += outputs['loss'].item()

        # mean validataion loss across batches 
        avg_loss = total_loss / len(self.val_loader)


        # preplexity is exp(cross-entropy)
        ppl = compute_preplx(avg_loss)
        print(f"Validation ---"f"Loss: {avg_loss:.4f} --- "f"Perplexity: {ppl:.2f}")

        # return model to train mode for next epoch 
        self.model.train()
        return avg_loss 
        
    def train(self):
        print(f"Training on {self.device}")
        print(f"Total steps: {self.scheduler.total_steps}")

        for epoch in range(self.config.epochs):
            self.epoch = epoch
            print(f"\n epoch {epoch + 1}/{self.config.epochs}")

            epoch_loss = self.train_epoch()
            print(
                f"epoch {epoch + 1} complete --- "
                f"avg loss: {epoch_loss:.4f}"
            )

        # Final evaluation after all epochs complete
        final_val_loss = self.validate()
        print("\nTraining complete!")
        print(f"best val loss: {self.best_val_loss:.4f}")
        print(f"final val loss: {final_val_loss:.4f}")


class FineTuner(Trainer):
    
    def __init__(self, model, train_loader, val_loader, config , device = 'cuda'):
        super().__init__(model, train_loader, val_loader, config, device)
        # counts consecutive epochs without validation improvement
        self.patience_counter = 0

    def train(self):
        # optional: initialize from a pretrained checkpoint
        if hasattr(self.config, 'pretrained'):
            print(f"Loading pretrained model from {self.config.pretrained}")
            checkpoint = load_checkpoint(self.config.pretrained, self.model)
            print("Loaded pretrained model")

        print(f"Fine-tuning on {self.device}")

        for epoch in range(self.config.epochs):
            self.epoch = epoch
            print(f"\nEpoch {epoch + 1}/{self.config.epochs}")

            epoch_loss = self.train_epoch()
            val_loss = self.validate()

            # early stopping logic to prevent overfitting
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.patience_counter = 0
            else:
                self.patience_counter += 1
                print(
                    f"NO.....IMprovvve "
                    f"({self.patience_counter}/{self.config.patience})"
                )

                if self.patience_counter >= self.config.patience:
                    print("Early stopping")
                    break

        save_checkpoint(
            self.config.output,
            self.model,
            self.optimizer,
            self.step,
            self.epoch,
            self.best_val_loss
        )
        print("\n fine-tuning complete!")
        print(f" model saved to {self.config.output}")
    
        
