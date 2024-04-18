# quick test for codeqwen: use transformers tokenizer
from transformers import AutoTokenizer
import qwen_cpp

device = "cpu" # the device to load the model onto

pipeline = qwen_cpp.Pipeline("../codeqwen2_7b-ggml.bin", "../qwen.tiktoken", 2048)
tokenizer = AutoTokenizer.from_pretrained("Qwen/CodeQwen1.5-7B-Chat")

prompt = "Write a quicksort algorithm in python."
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": prompt}
]
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)
print(text)
model_inputs = tokenizer([text], return_tensors="pt").to(device)

input_ids = model_inputs.input_ids.tolist()[0]

print(input_ids)

gen_config = qwen_cpp._C.GenerationConfig(
    max_length=2048,
    # max_new_tokens=args.max_new_tokens,
    max_context_length=512,
    do_sample=False,
    top_k=1,
    top_p=1,
    temperature=1,
    repetition_penalty=0.9,
    num_threads = 0,
)

out_ids = pipeline._sync_generate_ids(input_ids, gen_config)
print(out_ids)

response = tokenizer.decode(out_ids, skip_special_tokens=True)

print(response)
