import qwen_cpp
import tiktoken_cpp as tiktoken

# test tiktoken_cpp
enc = tiktoken.get_encoding("cl100k_base")
# enc = tiktoken.get_encoding("../qwen.tiktoken") # need regirester
assert enc.decode(enc.encode("hello world")) == "hello world"

Pipeline = qwen_cpp.Pipeline("../qwen2_4b-ggml.bin", "../qwen.tiktoken")
print(Pipeline.model.config.vocab_size)

# test tokenizer encoder, decoder
test_text = "hello world"
o = Pipeline.tokenizer.encode(test_text, 20)
print(o)
print(Pipeline.tokenizer.decode(o))

test_history = ["上海迪士尼的游玩攻略是什么"]
input_ids = Pipeline.tokenizer.encode_history(test_history, 2000)
print(input_ids)

# 1. test generate:
out = Pipeline._generate(input_ids=input_ids)
print(out)


# 2.1test chat
history = ["你好, 给我一个关于东京的游玩攻略"]
out = Pipeline.chat(history)
print(out)

# 2.2 test chat with stream
history = ["你好, 给我一个快速排序算法，我想要cpp和python的实现"]
out = Pipeline.chat(history, stream=True)
for i in out:
    print(i, sep="", end="", flush=True)


# 2.2 multithread test
# from concurrent.futures import ThreadPoolExecutor
# import time

# def chat(history):
#     out = Pipeline.chat(history, stream=True)
#     for i in out:
#         print(i, end="")
#     time.sleep(1)  # 模拟耗时操作

# history_list = [["你好, 给我一个多喝水的建议,500字"], ["你好, 给我一个多运动的建议,500字"], ["你好, 给我一个多读书的建议,500字"]]

# with ThreadPoolExecutor(max_workers=3) as executor:
#     executor.map(chat, history_list)