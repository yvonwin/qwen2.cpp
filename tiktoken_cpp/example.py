# import qwen_cpp
import tiktoken_cpp as tiktoken

# test tiktoken_cpp
enc = tiktoken.get_encoding("cl100k_base")
print(enc)
assert enc.decode(enc.encode("hello world")) == "hello world"
