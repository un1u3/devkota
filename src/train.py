# Training loop used for both pretraining (Nepali corpus) and fine-tuning (poems).
import math
import os
import time
import torch

from .devkota import Devkota
from .data import load_tokens, get_batch


def lr_at(step, max_steps, lr, warmup):
    # linear warmup, then cosine decay to 10% of lr
    if step < warmup:
        return lr * (step + 1) / warmup
    progress = (step - warmup) / max(1, max_steps - warmup)
    return lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress)))


@torch.no_grad()
def estimate_loss(model, splits, block_size, batch_size, device, iters=20):
    model.eval()
    out = {}
    for name, data in splits.items():
        losses = torch.zeros(iters)
        for i in range(iters):
            x, y = get_batch(data, block_size, batch_size, device)
            losses[i] = model(x, y)["loss"].item()
        out[name] = losses.mean().item()
    model.train()
    return out


def save_checkpoint(path, model, step, val_loss):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save({
        "config": model.config,
        "model_state_dict": model.state_dict(),
        "step": step,
        "val_loss": val_loss,
    }, path)


def load_model(path, device="cpu"):
    ckpt = torch.load(path, map_location=device)
    model = Devkota(**ckpt["config"]).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    return model, ckpt


def train(train_bin, val_bin, out_path, init_from=None, config=None,
          max_steps=5000, batch_size=16, lr=3e-4, warmup=200, weight_decay=0.1,
          grad_clip=1.0, eval_every=250, eval_iters=20, device="cpu"):
    train_data = load_tokens(train_bin)
    if val_bin:
        val_data = load_tokens(val_bin)
    else:
        # no separate val file (e.g. the poem corpus): hold out the last 10%
        n = int(0.9 * len(train_data))
        train_data, val_data = train_data[:n], train_data[n:]

    if init_from:
        model, _ = load_model(init_from, device)
        print(f"loaded weights from {init_from}")
    else:
        model = Devkota(**config).to(device)
    block_size = model.max_seq_len
    print(f"params: {model.count_parameters()['total'] / 1e6:.2f}M | train tokens: {len(train_data):,} | val tokens: {len(val_data):,}")

    # no weight decay on biases, layernorm gamma/beta (all 1-d params)
    decay = [p for p in model.parameters() if p.dim() >= 2]
    no_decay = [p for p in model.parameters() if p.dim() < 2]
    optimizer = torch.optim.AdamW(
        [{"params": decay, "weight_decay": weight_decay},
         {"params": no_decay, "weight_decay": 0.0}],
        lr=lr, betas=(0.9, 0.95),
    )

    best_val = float("inf")
    t0 = time.time()
    model.train()
    for step in range(max_steps + 1):
        for g in optimizer.param_groups:
            g["lr"] = lr_at(step, max_steps, lr, warmup)

        if step % eval_every == 0 or step == max_steps:
            losses = estimate_loss(model, {"train": train_data, "val": val_data},
                                   block_size, batch_size, device, eval_iters)
            print(f"step {step:6d} | train {losses['train']:.4f} | val {losses['val']:.4f} | {time.time() - t0:.0f}s")
            if losses["val"] < best_val:
                best_val = losses["val"]
                save_checkpoint(out_path, model, step, best_val)
            if step == max_steps:
                break

        x, y = get_batch(train_data, block_size, batch_size, device)
        loss = model(x, y)["loss"]
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

    print(f"best val loss {best_val:.4f}, saved to {out_path}")
    return model
