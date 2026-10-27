# Trains a SentencePiece subword tokenizer for Devkota and loads it back.
import os

# paths are relative to the repo root so this works from any working directory
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOKENIZER_DIR = os.path.join(ROOT, 'devkota-v0.7', 'tokenizer')
MODEL_FILE = os.path.join(TOKENIZER_DIR, 'devkota_tokenizer.model')

def train(
    input_file=os.path.join(ROOT, 'data', 'merged', 'train_merged.txt'),
    model_prefix=os.path.join(TOKENIZER_DIR, 'devkota_tokenizer'),
    vocab_size=8000,
    character_coverage=0.9999,
):
    """Train a SentencePiece model from a plain-text corpus."""
    try:
        import sentencepiece as spm
    except ImportError as exc:
        raise ImportError("tokenizer commands require the 'sentencepiece' package") from exc
    spm.SentencePieceTrainer.train(
        input=str(input_file), model_prefix=str(model_prefix), vocab_size=vocab_size,
        character_coverage=character_coverage, byte_fallback=True,
        normalization_rule_name='nfkc', pad_id=0, unk_id=1, bos_id=2, eos_id=3,
    )


def load(model_file=MODEL_FILE):
    """Load a trained SentencePiece model."""
    try:
        import sentencepiece as spm
    except ImportError as exc:
        raise ImportError("loading a tokenizer requires the 'sentencepiece' package") from exc
    return spm.SentencePieceProcessor(model_file=model_file)


if __name__ == "__main__":
    train()
