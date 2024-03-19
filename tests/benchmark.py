import os
import time
from typing import Any, cast

import tiktoken
import tiktoken_cpp


def benchmark_batch(documents: list[str]) -> None:
    # num_threads = int(os.environ["RAYON_NUM_THREADS"])
    num_threads = int(os.environ.get("RAYON_NUM_THREADS", "4"))
    num_bytes = sum(map(len, map(str.encode, documents)))
    print(f"num_threads: {num_threads}, num_bytes: {num_bytes}")

    enc = tiktoken_cpp.get_encoding("gpt2")
    enc.encode("warmup")

    start = time.perf_counter_ns()
    enc.encode_ordinary_batch(documents, num_threads=num_threads)
    end = time.perf_counter_ns()
    print(f"tiktoken.cpp \t{num_bytes / (end - start) * 1e9} bytes / s")

    enc = tiktoken.get_encoding("gpt2")
    enc.encode("warmup")

    start = time.perf_counter_ns()
    enc.encode_ordinary_batch(documents, num_threads=num_threads)
    end = time.perf_counter_ns()
    print(f"tiktoken \t{num_bytes / (end - start) * 1e9} bytes / s")

    import transformers

    # hf_enc = cast(Any, transformers).GPT2TokenizerFast.from_pretrained("/Users/wsj/project/hf/gpt2")
    hf_enc = cast(Any, transformers).GPT2TokenizerFast.from_pretrained("gpt2")
    hf_enc.model_max_length = 1e30  # silence!
    hf_enc.encode("warmup")

    start = time.perf_counter_ns()
    hf_enc(documents)
    end = time.perf_counter_ns()
    print(f"huggingface \t{num_bytes / (end - start) * 1e9} bytes / s")


filename = "lorem_ipsum_64k.txt"
if not os.path.exists(filename):
    import requests
    url = "https://raw.githubusercontent.com/MarshallCharles/Birthday-Attack-Crypto/master/lorem_ipsum_64k.txt"
    response = requests.get(url)
    with open(filename, 'wb') as f:
        f.write(response.content)
documents = []
with open(filename, 'r') as f:
    lines = f.readlines()
    for line in lines:
        documents.append(line)
benchmark_batch(documents)
