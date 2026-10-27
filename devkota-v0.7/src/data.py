# Turns a text file into one long stream of token ids and samples random
# (input, target) windows from it, the same way gpt-2 / nanoGPT train.
import os
import re
import numpy as np
import torch

from .tokenizer import load

# stanza markers on their own line: "क.", "ख.", "१.", "2." ...
STANZA_MARKER = re.compile(r"^([क-ह]|[०-९0-9]+)\.$")


def build_token_file(text_path, out_path, sep="eos", tokenizer_path=None):
    """Encode every non-empty line of text_path and save as a uint16 .bin.

    sep="eos"     -> lines are separate documents (pretraining corpus)
    sep="newline" -> lines are verses of a poem; keep '\n' so the model learns line breaks
    """
    sp = load(tokenizer_path) if tokenizer_path else load()
    sep_id = sp.eos_id() if sep == "eos" else sp.piece_to_id("<0x0A>")

    ids = []
    try:
        with open(text_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                if sep == "newline" and STANZA_MARKER.match(line):
                    continue  # otherwise the model learns to write "क." "ख." between lines
                ids.extend(sp.encode(line))
                ids.append(sep_id)
    except UnicodeDecodeError as exc:
        # usually a file whose upload was cut off mid-character
        size = os.path.getsize(text_path)
        raise ValueError(f"{text_path} ({size:,} bytes) is not valid utf-8, it is probably "
                         f"truncated; re-upload it") from exc

    arr = np.array(ids, dtype=np.uint16)  # vocab is 8000, fits in uint16
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    arr.tofile(out_path)
    return len(arr)


def load_tokens(path):
    return torch.from_numpy(np.fromfile(path, dtype=np.uint16).astype(np.int64))


def get_batch(data, block_size, batch_size, device):
    # random windows; y is x shifted by one token
    ix = torch.randint(len(data) - block_size - 1, (batch_size,))
    x = torch.stack([data[i:i + block_size] for i in ix])
    y = torch.stack([data[i + 1:i + 1 + block_size] for i in ix])
    return x.to(device), y.to(device)
