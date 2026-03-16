#!/bin/bash
set -e

echo "Laptop Setup (8GB)"
echo "-------------------"

# check setup
echo "checking setup..."
if [ ! -d "preprocessed_data" ]; then
    echo "creating small dataset..."
    mkdir -p preprocessed_data
    head -n 100000 data/preprocessed/input.txt > preprocessed_data/train_small.txt
    tail -n 5000 data/preprocessed/input.txt > preprocessed_data/val_small.txt
fi

# train tokenizer
echo "training tokenizer..."
python3 train_tokenizer.py

# startup
echo "starting training..."
python3 train.py
