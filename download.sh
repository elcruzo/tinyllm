#!/bin/bash
mkdir -p models
echo "downloading stories15M model..."
curl -L -o models/stories15M.bin "https://huggingface.co/karpathy/tinyllamas/resolve/main/stories15M.bin"
echo "downloading tokenizer..."
curl -L -o tokenizer.bin "https://github.com/karpathy/llama2.c/raw/master/tokenizer.bin"
echo "done. run: ./tinyllm models/stories15M.bin \"Once upon a time\""
