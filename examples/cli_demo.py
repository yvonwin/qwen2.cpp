import argparse
from pathlib import Path
from typing import List

import qwen_cpp

DEFAULT_MODEL_PATH = Path(__file__).resolve().parent.parent / "qwen2_1.8b-ggml.bin"
DEFAULT_TIKTOKEN_PATH =   Path(__file__).resolve().parent.parent / "qwen.tiktoken"

BANNER = """
 ██████╗ ██╗    ██╗███████╗███╗   ██╗██████╗     ██████╗██████╗ ██████╗ 
██╔═══██╗██║    ██║██╔════╝████╗  ██║╚════██╗   ██╔════╝██╔══██╗██╔══██╗
██║   ██║██║ █╗ ██║█████╗  ██╔██╗ ██║ █████╔╝   ██║     ██████╔╝██████╔╝
██║▄▄ ██║██║███╗██║██╔══╝  ██║╚██╗██║██╔═══╝    ██║     ██╔═══╝ ██╔═══╝ 
╚██████╔╝╚███╔███╔╝███████╗██║ ╚████║███████╗██╗╚██████╗██║     ██║     
 ╚══▀▀═╝  ╚══╝╚══╝ ╚══════╝╚═╝  ╚═══╝╚══════╝╚═╝ ╚═════╝╚═╝     ╚═╝     
                                                                           
""".strip(
    "\n"
)
WELCOME_MESSAGE = "Welcome to Qwen.cpp! Ask whatever you want. Type 'clear' to clear context. Type 'stop' to exit."


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", default=DEFAULT_MODEL_PATH, type=str, help="model path")
    parser.add_argument("--tiktoken", default=DEFAULT_TIKTOKEN_PATH, type=str, help="tiktoken path")
    parser.add_argument("--mode", default="chat", type=str, choices=["chat", "generate"], help="inference mode")
    parser.add_argument("-p", "--prompt", default="你好", type=str, help="prompt to start generation with")
    parser.add_argument(
        "--pp", "--prompt_path", default=None, type=Path, help="path to the plain text file that stores the prompt"
    )
    parser.add_argument(
        "-s", "--system", default="You are a helpful assistant.", type=str, help="system message to set the behavior of the assistant"
    )
    parser.add_argument(
        "--sp",
        "--system_path",
        default=None,
        type=Path,
        help="path to the plain text file that stores the system message",
    )
    parser.add_argument("-i", "--interactive", action="store_true", help="run in interactive mode")
    parser.add_argument(
        "-l", "--max_length", default=4096, type=int, help="max total length including prompt and output"
    )
    # parser.add_argument(
    #     "--max_new_tokens",
    #     default=-1,
    #     type=int,
    #     help="max number of tokens to generate, ignoring the number of prompt tokens",
    # )
    parser.add_argument("-c", "--max_context_length", default=512, type=int, help="max context length")
    parser.add_argument("--top_k", default=0, type=int, help="top-k sampling")
    parser.add_argument("--top_p", default=0.7, type=float, help="top-p sampling")
    parser.add_argument("--temp", default=0.95, type=float, help="temperature")
    parser.add_argument("--repeat_penalty", default=1.0, type=float, help="penalize repeat sequence of tokens")
    parser.add_argument("-t", "--threads", default=0, type=int, help="number of threads for inference")
    args = parser.parse_args()

    prompt = args.prompt
    if args.pp:
        prompt = args.pp.read_text()

    system = args.system
    if args.sp:
        system = args.sp.read_text()

    pipeline = qwen_cpp.Pipeline(args.model, args.tiktoken, args.max_length)

    if args.mode != "chat" and args.interactive:
        print("interactive demo is only supported for chat mode, falling back to non-interactive one")
        args.interactive = False

    generation_kwargs = dict(
        max_length=args.max_length,
        # max_new_tokens=args.max_new_tokens,
        max_context_length=args.max_context_length,
        do_sample=args.temp > 0,
        top_k=args.top_k,
        top_p=args.top_p,
        temperature=args.temp,
        repetition_penalty=args.repeat_penalty,
        stream=True,
    )

    system_messages: List[qwen_cpp.ChatMessage] = []
    if system is not None:
        system_messages.append(qwen_cpp.ChatMessage(role="system", content=system))

    messages = system_messages.copy()

    if not args.interactive:
        if args.mode == "chat":
            messages.append(qwen_cpp.ChatMessage(role="user", content=prompt))
            for chunk in pipeline.chat(messages, **generation_kwargs):
                print(chunk.content, sep="", end="", flush=True)
        else:
            for chunk in pipeline.generate(prompt, **generation_kwargs):
                print(chunk, sep="", end="", flush=True)
        print()
        return

    print(BANNER)
    print()
    print(WELCOME_MESSAGE)
    print()

    # prompt_width = len(pipeline.model.config.model_type_name)
    prompt_width = len("qwen")

    if system:
        print(f"{'System':{prompt_width}} > {system}")

    while True:
        input_prompt = f"{'Prompt':{prompt_width}} > "
        role = "user"

        try:
            prompt = input(input_prompt)
        except EOFError:
            break

        if not prompt:
            continue
        if prompt == "stop":
            break
        if prompt == "clear":
            messages = system_messages
            continue

        messages.append(qwen_cpp.ChatMessage(role=role, content=prompt))
        chunks = []
        for chunk in pipeline.chat(messages, **generation_kwargs):
            print(chunk.content, sep="", end="", flush=True)
            chunks.append(chunk)
        print()
        messages.append(pipeline.merge_streaming_messages(chunks))

    print("Bye")


if __name__ == "__main__":
    main()
