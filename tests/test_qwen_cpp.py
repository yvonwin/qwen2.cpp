from pathlib import Path
import qwen_cpp
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent

QWEN_MODEL_PATH = PROJECT_ROOT / "qwen2_1.8b_f16.bin"
QWEN_TIKTOKEN_PAHT=  PROJECT_ROOT / "qwen.tiktoken"
LLAMA_MODEL_PATH = PROJECT_ROOT / "llama.bin"
LLAMA_TIKTOKEN_PATH = PROJECT_ROOT / "llama3.tiktoken"


def test_qwen_version():
    print(qwen_cpp.__version__)


def check_pipeline(model_path, tiktoken_path,  prompt, target, gen_kwargs={}):
    messages = [qwen_cpp.ChatMessage(role="system", content="You are a helpful assistant."), qwen_cpp.ChatMessage(role="user", content=prompt)]

    pipeline = qwen_cpp.Pipeline(model_path, tiktoken_path)
    output = pipeline.chat(messages, do_sample=False, **gen_kwargs).content
    assert output == target

    stream_output = pipeline.chat(messages, do_sample=False, stream=True, **gen_kwargs)
    stream_output = "".join([msg.content for msg in stream_output])
    assert stream_output == target


@pytest.mark.skipif(not QWEN_MODEL_PATH.exists(), reason="model file not found")
def test_pipeline_options():
    # check max_length option
    pipeline = qwen_cpp.Pipeline(QWEN_MODEL_PATH, QWEN_TIKTOKEN_PAHT)
    assert pipeline.model.config.max_length == 4096
    pipeline = qwen_cpp.Pipeline(QWEN_MODEL_PATH, QWEN_TIKTOKEN_PAHT, max_length=234)
    assert pipeline.model.config.max_length == 234

    # check if resources are properly released
    for _ in range(100):
        qwen_cpp.Pipeline(QWEN_MODEL_PATH, QWEN_TIKTOKEN_PAHT)


@pytest.mark.skipif(not QWEN_MODEL_PATH.exists(), reason="model file not found")
def test_qwen_pipeline():
    check_pipeline(
        model_path=QWEN_MODEL_PATH,
        tiktoken_path=QWEN_TIKTOKEN_PAHT,
        prompt="你好",
        target="你好！有什么我可以帮助你的吗？",
    )

@pytest.mark.skipif(not LLAMA_MODEL_PATH.exists(), reason="model file not found")
def test_llama_pipeline():
    check_pipeline(
        model_path=LLAMA_MODEL_PATH,
        tiktoken_path=LLAMA_TIKTOKEN_PATH,
        prompt="hello",
        target="Hello! It's nice to meet you. Is there something I can help you with, or would you like to chat? I'm here to assist you with any questions or topics you'd like to discuss.",
    )
