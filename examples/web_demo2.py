import os
import gradio as gr
from typing import List, Optional, Tuple, Dict
import qwen_cpp
from pathlib import Path
import argparse

default_system = 'You are a helpful assistant.'

class Role:
    USER = 'user'
    SYSTEM = 'system'
    BOT = 'bot'
    ASSISTANT = 'assistant'
    ATTACHMENT = 'attachment'

DEFAULT_MODEL_PATH = Path(__file__).resolve().parent.parent / "qwen2_1.8b-ggml.bin"
DEFAULT_TIKTOKEN_PATH =   Path(__file__).resolve().parent.parent / "qwen.tiktoken"

parser = argparse.ArgumentParser()
parser.add_argument("-m", "--model", default=DEFAULT_MODEL_PATH, type=Path, help="model path")
parser.add_argument("--tiktoken", default=DEFAULT_TIKTOKEN_PATH, type=str, help="tiktoken path")
parser.add_argument("--mode", default="chat", type=str, choices=["chat", "generate"], help="inference mode")
parser.add_argument(
    "-s", "--system", default="You are a helpful assistant.", type=str, help="system message to set the behavior of the assistant"
)
parser.add_argument("-l", "--max_length", default=4096, type=int, help="max total length including prompt and output")
parser.add_argument("-c", "--max_context_length", default=512, type=int, help="max context length")
parser.add_argument("--top_k", default=0, type=int, help="top-k sampling")
parser.add_argument("--top_p", default=0.7, type=float, help="top-p sampling")
parser.add_argument("--temp", default=0.95, type=float, help="temperature")
parser.add_argument("--repeat_penalty", default=1.0, type=float, help="penalize repeat sequence of tokens")
parser.add_argument("-t", "--threads", default=0, type=int, help="number of threads for inference")
parser.add_argument("--plain", action="store_true", help="display in plain text without markdown support")
args = parser.parse_args()

pipeline = qwen_cpp.Pipeline(args.model, args.tiktoken)

History = List[Tuple[str, str]]
Messages = List[Dict[str, str]]

def clear_session() -> History:
    return '', []

def modify_system_session(system: str) -> str:
    if system is None or len(system) == 0:
        system = default_system
    return system, system, []

def history_to_messages(history: History, system: str) -> Messages:
    messages = [{'role': Role.SYSTEM, 'content': system}]
    for h in history:
        messages.append({'role': Role.USER, 'content': h[0]})
        messages.append({'role': Role.ASSISTANT, 'content': h[1]})
    return messages


def messages_to_history(messages: Messages) -> Tuple[str, History]:
    assert messages[0]['role'] == Role.SYSTEM
    system = messages[0]['content']
    history = []
    for q, r in zip(messages[1::2], messages[2::2]):
        history.append([q['content'], r['content']])
    return system, history


def model_chat(query: Optional[str], history: Optional[History], system: str
) -> Tuple[str, str, History]:
    if query is None:
        query = ''
    if history is None:
        history = []
    messages = history_to_messages(history, system)
    messages.append({'role': Role.USER, 'content': query})


    generation_kwargs = dict(
        max_length=args.max_length,
        # max_context_length=args.max_context_length,
        do_sample=args.temp > 0,
        top_k=args.top_k,
        top_p=args.top_p,
        temperature=args.temp,
        repetition_penalty=args.repeat_penalty,
        num_threads=args.threads,
        stream=True,
    )
    response = ""
    if args.mode == "chat":
        chunks = []
        for chunk in pipeline.chat(messages, **generation_kwargs):
            response += chunk.content
            chunks.append(chunk)
            role = Role.USER if not len(messages) % 2 else Role.SYSTEM
            system, history = messages_to_history(messages + [{'role': role, 'content': response}])
            yield '', history, system


with gr.Blocks() as demo:
    gr.Markdown("""<center><font size=8>Qwen1.5-Chat</center>""")
    gr.Markdown("""<center><font size=4>Qwen1.5 is a transformer-based model of the Qwen series.</center>""")

    with gr.Row():
        with gr.Column(scale=3):
            system_input = gr.Textbox(value=default_system, lines=1, label='System')
        with gr.Column(scale=1):
            modify_system = gr.Button("🛠️ Set system prompt and clear history.", scale=2)
        system_state = gr.Textbox(value=default_system, visible=False)
    chatbot = gr.Chatbot(label='Qwen1.5-Chat')
    textbox = gr.Textbox(lines=2, label='Input')

    with gr.Row():
        clear_history = gr.Button("🧹 Clear history")
        sumbit = gr.Button("🚀 Send")

    sumbit.click(model_chat,
                 inputs=[textbox, chatbot, system_state],
                 outputs=[textbox, chatbot, system_input])
    clear_history.click(fn=clear_session,
                        inputs=[],
                        outputs=[textbox, chatbot])
    modify_system.click(fn=modify_system_session,
                        inputs=[system_input],
                        outputs=[system_state, system_input, chatbot])

demo.queue(api_open=False).launch(max_threads=10,height=800, share=False)