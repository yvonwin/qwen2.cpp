#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=0

# Qwen-6B-Base
hf_model=Qwen/Qwen1.5-7B
ggml_model=qwen-base-ggml.bin

for dtype in f16; do
    python3 qwen_cpp/convert.py -i $hf_model -o $ggml_model -t $dtype
    echo "[perplexity] dtype=$dtype"
    ./build/bin/perplexity -m $ggml_model -f tests/data/wikitext-2-raw/wiki.test.raw -s 512 -l 2048
done