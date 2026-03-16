import torch
from main.devkota import Devkota
from src.core.config import ModelConfig
from src.core.train_spm import NepaliTokenizer
from src.core.utils import load_checkpoint
import os

def generate_text(prompt, model, tokenizer, max_tokens=80, temperature=0.7, top_k=40, top_p=0.85, repetition_penalty=1.1):
    # wrapper for generation 
    
    # encode prompt 
    input_ids = tokenizer.encode(prompt, add_bos=True, add_eos=False)
    input_ids = torch.tensor([input_ids], dtype=torch.long)
    
    # device thing 
    device = next(model.parameters()).device
    input_ids = input_ids.to(device)
    
    # generate 
    output_ids = model.generate(
        input_ids=input_ids,
        max_new_tokens=max_tokens,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        repetition_penalty=repetition_penalty
    )
    
    # decode back to text
    full = output_ids[0]
    new_tokens = full[input_ids.size(1):]  # decode only what was generated
    if len(new_tokens) == 0:
        new_tokens = full
    generated_text = tokenizer.decode(new_tokens.tolist())
    return generated_text

def build_poem_prompt(theme_or_prompt):
    # tiny helper to wrap a theme into a poem ask
    text = theme_or_prompt.strip()
    if len(text) == 0:
        return "कविता लेख्नुहोस्:"
    if " " not in text and len(text) < 20:
        return f"विषय: {text}\nकविता:"
    return text

def main():
    # config setup
    model_cfg = ModelConfig()
    
    # model init 
    model = Devkota(
        vocab_size=model_cfg.vocab_size,
        d_model=model_cfg.d_model,
        num_layers=model_cfg.num_layers,
        num_heads=model_cfg.num_heads,
        d_ff=model_cfg.d_ff,
        max_seq_len=model_cfg.max_seq_len,
        dropout=model_cfg.dropout,
        pad_idx=model_cfg.pad_idx
    )
    
    # loading checkpoint 
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    ft_path = "checkpoints/finetuned/devkota_poet.pt"
    checkpoint_path = ft_path if os.path.exists(ft_path) else "checkpoints/best_model.pt"
    
    if os.path.exists(checkpoint_path):
        print(f"loading from {checkpoint_path}")
        checkpoint = load_checkpoint(checkpoint_path, model)
        print("model loaded")
    else:
        print("checkpoint xina, using untrained model")

        
    model = model.to(device)
    model.eval()
    
    # tokenizer 
    tokenizer_path = "tokenizer/devkota_tokenizer.model"
    if not os.path.exists(tokenizer_path):
        print("tokenizer missing")
        return

    tokenizer = NepaliTokenizer(tokenizer_path)
    
    print("Devkota POeLM Generator")
    print("type 'quit' to exit")
    
    while True:
        try:
            prompt = input("Enter poem theme or prompt: ")
            if prompt.strip() == 'quit':
                break
            
            print("generating...")
            poem_prompt = build_poem_prompt(prompt)
            generated = generate_text(
                prompt=poem_prompt,
                model=model,
                tokenizer=tokenizer,
                max_tokens=80,
                temperature=0.7,
                top_k=40,
                top_p=0.85
            )
            
            print(f"\nResult: {generated}\n")
            print("-" * 50)
        except KeyboardInterrupt:
            # quit on ctrl c 
            break
        except Exception as e:
            print(f"error: {e}")

if __name__ == "__main__":
    main()
