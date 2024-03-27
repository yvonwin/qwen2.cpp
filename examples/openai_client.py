"""
Need run openai_api.py.
python examples/openai_client.py 
"""
import argparse
from openai import OpenAI

parser = argparse.ArgumentParser()
parser.add_argument("--stream", action="store_true")
parser.add_argument("--prompt", default="你好", type=str)
args = parser.parse_args()

client = OpenAI(
    api_key = 'hello',
    base_url = "http://127.0.0.1:8000/v1"
)

messages = [{"role": "user", "content": args.prompt}]
if args.stream:
    response = client.chat.completions.create(model="default-model", messages=messages, stream=True)
    for chunk in response:
        content = chunk.choices[0].delta.content
        if content is not None:
            print(content, end="", flush=True)
    print()
else:
    response = client.chat.completions.create(model="default-model", messages=messages)
    print(response.choices[0].message.content)