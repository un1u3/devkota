# Trains a SentencePiece subword tokenizer for Devkota and loads it back.

import sentencepiece as spm

def train():
    spm.SentencePieceTrainer.train(
        # plain-text corpus, one sentence per line
        input = '../data/tokenizer/train_tokenizer_corpus.txt',
        # output name: writes devkota_tokenizer.model and devkota_tokenizer.vocab
        model_prefix = 'devkota_tokenizer',
        vocab_size = 8000,
        # unigram LM segmentation 
        # fraction of characters the vocab must cover; set high so rare
        # Devanagari characters are kept instead of mapped to <unk>
        character_coverage = 0.9999,
        # unseen characters are split into UTF-8 bytes rather than becoming <unk>
        byte_fallback = True,
        # unicode nfkc normalization so visually identical text gets the same tokens
        normalization_rule_name = 'nfkc',
        # special token ids; pad is 0 so it can double as the padding index
        pad_id = 0,
        unk_id = 1,
        bos_id = 2,
        eos_id = 3
    )


def load(model_file='devkota_tokenizer.model'):
    # load the trained model for encoding/decoding
    return spm.SentencePieceProcessor(model_file=model_file)


if __name__ == "__main__":
    train()
    sp = load()
