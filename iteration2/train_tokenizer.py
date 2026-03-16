from src.core.train_spm import TokenizerTrainer
import os

def main():
    print("Training tokenizer...")
    print("-" * 50)

    output_dir = "tokenizer"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # settings vocab size 
    vocab_size = 16000
    trainer = TokenizerTrainer(vocab_size=vocab_size)
    
    # using small file for laptop 
    corpus_file = "preprocessed_data/train_small.txt" 
    
    if not os.path.exists(corpus_file):
        print(f"file xina : {corpus_file}")
        return

    print(f"training on {corpus_file}")
    
    # train command 
    trainer.train(
        corpus_file=corpus_file,
        output_dir=output_dir,
        model_name="devkota_tokenizer"
    )
    
    print("done training tokenizer")

if __name__ == "__main__":
    main()
