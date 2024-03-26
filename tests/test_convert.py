import sys
from pathlib import Path

import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent

QWEN2_0_5B_MODEL_PATH = Path("Qwen/Qwen1.5-0.5B").expanduser()

def make_data_qwen2_0_5b_wte():
    from transformers.models.qwen2.modeling_qwen2 import Qwen2Model
    from transformers import AutoConfig
    config = AutoConfig.from_pretrained(QWEN2_0_5B_MODEL_PATH, trust_remote_code=True)
    config.vocab_size = 48
    config.hidden_size = 256

    m = Qwen2Model(config).float().eval()
    m.embed_tokens.weight.data.normal_(mean=0.0, std=0.02)

    seq_len = 3
    x = torch.arange(seq_len, dtype=torch.int64)[None, :]
    with torch.no_grad():
        y = m.embed_tokens(x)

    with open(HERE / "data/qwe2_0_5b_wte.data", "wb") as f:
        m.embed_tokens.weight.data.numpy().tofile(f)
        x.int().numpy().tofile(f)
        y.numpy().tofile(f)

def make_data_qwen2_0_5b_mlp():
    from transformers.models.qwen2.modeling_qwen2 import Qwen2MLP
    from transformers import AutoConfig
    config = AutoConfig.from_pretrained(QWEN2_0_5B_MODEL_PATH, trust_remote_code=True)
    config.hidden_size = 32
    config.intermediate_size = config.hidden_size * 3
    config.num_hidden_layers = 1
    config.fp32 = True

    m = Qwen2MLP(config).float().eval()
    m.up_proj.weight.data.normal_(mean=0.0, std=0.02)
    m.gate_proj.weight.data.normal_(mean=0.0, std=0.02)
    m.down_proj.weight.data.normal_(mean=0.0, std=0.02)

    x = torch.randn(3, 32)
    with torch.no_grad():
        y = m(x)
    print(m.up_proj.weight.data.numpy().shape)
    print(m.down_proj.weight.data.numpy().shape)
    print(x.shape)
    print(y.shape)
    with open(HERE / "data/qwen2_0_5b_mlp.data", "wb") as f:
        m.up_proj.weight.data.numpy().tofile(f)
        m.gate_proj.weight.data.numpy().tofile(f)
        m.down_proj.weight.data.numpy().tofile(f)
        x.numpy().tofile(f)
        y.numpy().tofile(f)

def make_data_qwen2_0_5b_block():
    from transformers.models.qwen2.modeling_qwen2 import Qwen2Model
    from transformers import AutoConfig

    config = AutoConfig.from_pretrained(QWEN2_0_5B_MODEL_PATH, trust_remote_code=True)
    # config.hidden_size = 32
    # config.num_attention_heads = 8
    # config.intermediate_size = config.hidden_size * 3
    # config.num_hidden_layers = 1
    # config.torch_dtype = torch.float32
    # config.vocab_size = 5

    # config._attn_implementation = "eager"

    m = Qwen2Model(config).eval()
    # print(m)

    for param in m.parameters():
        param.data.uniform_(-0.5, 0.5)

    seq_len = 3
    
    # self attention
    x1 = torch.arange(seq_len, dtype=torch.int64)[None, :]
    attn_mask = torch.ones(1, seq_len, dtype=torch.int64)
    with torch.no_grad():
        out = m(x1, attention_mask=attn_mask, use_cache=True)
        y1 = out.last_hidden_state
        kv_cache = out.past_key_values

    # cross attention
    x2 = torch.tensor([[seq_len]], dtype=torch.int64)
    attn_mask = torch.ones(1, seq_len + 1, dtype=torch.int64)
    with torch.no_grad():
        out = m(x2, attention_mask=attn_mask, past_key_values=kv_cache, use_cache=True)
        y2 = out.last_hidden_state
        kv_cache = out.past_key_values

    # cross attention
    x3 = torch.tensor([[seq_len + 1]], dtype=torch.int64)
    attn_mask = torch.ones(1, seq_len + 2, dtype=torch.int64)
    with torch.no_grad():
        out = m(x3, attention_mask=attn_mask, past_key_values=kv_cache, use_cache=True)
        y3 = out.last_hidden_state
        kv_cache = out.past_key_values

    with open(HERE / "data/qweb2_0_5b_block.data", "wb") as f:
        m.embed_tokens.weight.data.numpy().tofile(f)
        m.layers[0].input_layernorm.weight.data.numpy().tofile(f)
        m.layers[0].self_attn.q_proj.weight.data.numpy().tofile(f)
        m.layers[0].self_attn.q_proj.bias.data.numpy().tofile(f)
        m.layers[0].self_attn.k_proj.weight.data.numpy().tofile(f)
        m.layers[0].self_attn.k_proj.bias.data.numpy().tofile(f)
        m.layers[0].self_attn.v_proj.weight.data.numpy().tofile(f)
        m.layers[0].self_attn.v_proj.bias.data.numpy().tofile(f)
        m.layers[0].self_attn.o_proj.weight.data.numpy().tofile(f)
        m.layers[0].post_attention_layernorm.weight.data.numpy().tofile(f)
        m.layers[0].mlp.gate_proj.weight.data.numpy().tofile(f)
        m.layers[0].mlp.up_proj.weight.data.numpy().tofile(f)
        m.layers[0].mlp.down_proj.weight.data.numpy().tofile(f)
        m.norm.weight.data.numpy().tofile(f)

        x1.int().numpy().tofile(f)
        y1.numpy().tofile(f)
        x2.int().numpy().tofile(f)
        y2.numpy().tofile(f)
        x3.int().numpy().tofile(f)
        y3.numpy().tofile(f)

def main():
    torch.manual_seed(0)
    (HERE / "data").mkdir(parents=True, exist_ok=True)
    sys.path.append(str(QWEN2_0_5B_MODEL_PATH))
    make_data_qwen2_0_5b_mlp()
    # make_data_qwen2_0_5b_block()
    make_data_qwen2_0_5b_wte()

if __name__ == "__main__":
    main()
