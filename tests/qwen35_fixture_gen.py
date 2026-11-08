#!/usr/bin/env python3
"""Write a frozen Qwen3.5 F32/eager fixture header from Transformers.

    python -I tests/qwen35_fixture_gen.py base tests/qwen35_fixture.h
    python -I tests/qwen35_fixture_gen.py gqa  tests/qwen35_gqa_fixture.h

"base" is the fixture test_qwen35 has carried since the family landed: one
key head and one KV head, so head grouping is never exercised. "gqa" has two
of each and two full-attention layers, so a head read from the wrong group or
a full-attention slot read at the wrong layer offset changes the numbers.
Weights are a fixed arithmetic pattern over the order Transformers registers
parameters in; only the reference forward is Transformers.
"""
import sys

import torch
import transformers
from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForCausalLM

SHAPES = {
    "base": dict(heads=2, kv=1, key_heads=1, value_heads=2,
                 layers=["linear_attention"] * 3 + ["full_attention"],
                 summary="E8 F12 H2 KV1 HD4 RoPE2; K-head1 V-head2 KD4 VD4 conv4; 3 delta+1 full."),
    "gqa": dict(heads=4, kv=2, key_heads=2, value_heads=4,
                layers=["linear_attention", "full_attention"] * 2,
                summary="E8 F12 H4 KV2 HD4 RoPE2; K-head2 V-head4 KD4 VD4 conv4; delta,full,delta,full."),
}
TOKENS = [0, 3, 5, 12, 2, 6, 7]


def hexf(v):
    return float(v).hex() + "f"


def main():
    if len(sys.argv) != 3 or sys.argv[1] not in SHAPES:
        sys.exit("usage: qwen35_fixture_gen.py {base|gqa} OUT.h")
    kind, out = sys.argv[1], sys.argv[2]
    s, p = SHAPES[kind], "" if kind == "base" else kind + "_"
    E, F, V, HD, ctx = 8, 12, 40, 4, 128
    cfg = Qwen3_5TextConfig(
        vocab_size=V, hidden_size=E, intermediate_size=F, num_hidden_layers=len(s["layers"]),
        num_attention_heads=s["heads"], num_key_value_heads=s["kv"], head_dim=HD,
        layer_types=s["layers"], linear_num_key_heads=s["key_heads"],
        linear_num_value_heads=s["value_heads"], linear_key_head_dim=4, linear_value_head_dim=4,
        linear_conv_kernel_dim=4, max_position_embeddings=ctx, rms_norm_eps=1e-6,
        rope_parameters={"rope_type": "default", "rope_theta": 10000.0, "partial_rotary_factor": 0.5},
        tie_word_embeddings=True, hidden_act="silu", attention_bias=False)
    cfg._attn_implementation = "eager"
    torch.manual_seed(0)
    model = Qwen3_5ForCausalLM(cfg).float().eval()

    with torch.no_grad():
        for idx, (name, param) in enumerate(model.named_parameters()):
            n = param.numel()
            v = ((torch.arange(n) * 7 + idx * 3) % 23 - 11).float() * 0.0175
            if name.endswith("linear_attn.norm.weight"):
                v = v + 1
            param.copy_(v.reshape(param.shape))
    state = {k.removeprefix("model."): t.detach().float().contiguous()
             for k, t in model.state_dict().items()}

    residual = []
    def keep(_m, _i, o):
        residual.append((o[0] if isinstance(o, tuple) else o)[0].detach().clone())
    for layer in model.model.layers:
        layer.register_forward_hook(keep)
    with torch.no_grad():
        logits = model(torch.tensor([TOKENS])).logits[0, -1]

    names = ["lm_head.weight"] + sorted(k for k in state if k != "lm_head.weight")
    full = [t == "full_attention" for t in s["layers"]]
    L = [f"/* Frozen independent Transformers {transformers.__version__} F32/eager reference.",
         f" * {s['summary']}",
         " * Parameters: ((arange(n)*7 + parameter_index*3)%23-11)*0.0175f;",
         f" * linear_attn.norm adds 1. Tokens: {','.join(map(str, TOKENS))}. */"]
    for i, k in enumerate(names):
        L.append(f"static const float {p}weight_{i}[]={{{','.join(hexf(x) for x in state[k].flatten().tolist())}}};")
    L.append(f"static int {p}fixture(const char *path,int mode){{")
    L.append(" gguf_writer *w=gguf_write_open(path);if(!w)return -1;")
    L.append(' gguf_write_kv_str(w,"general.architecture","qwen35");')
    L.append(' gguf_write_kv_str(w,"general.name","Qwen3.5 dense text, source tensor layout");')
    kv = [("source_layout_version", 1), ("block_count", len(full)), ("embedding_length", E),
          ("feed_forward_length", F), ("vocab_size", V), ("context_length", ctx),
          ("attention.head_count", s["heads"]), ("attention.head_count_kv", s["kv"]),
          ("attention.head_dim", HD), ("rope.dimension_count", HD // 2),
          ("linear.key_heads", s["key_heads"]), ("linear.value_heads", s["value_heads"]),
          ("linear.key_dim", 4), ("linear.value_dim", 4), ("linear.conv_kernel", 4)]
    kv += [(f"layer.{l}.full_attention", int(f)) for l, f in enumerate(full)]
    for key, val in kv:
        L.append(f' gguf_write_kv_u32(w,"qwen35.{key}",{val});')
    L.append(' gguf_write_kv_f32(w,"qwen35.rms_epsilon",1e-06);')
    L.append(' gguf_write_kv_f32(w,"qwen35.rope.freq_base",10000.0);')
    first_conv = next(f"layers.{l}.linear_attn.conv1d.weight" for l, f in enumerate(full) if not f)
    for k in names:
        dims = ",".join(str(d) for d in reversed(state[k].shape))
        guard = "if(mode!=1) " if k == first_conv else ""
        L.append(f' {guard}gguf_write_tensor_decl(w,"{k}",{state[k].dim()},(uint64_t[]){{{dims}}},GGUF_TYPE_F32);')
    for i, k in enumerate(names):
        guard = "if(mode!=1) " if k == first_conv else ""
        L.append(f' {guard}gguf_write_tensor_f32(w,"{k}",{p}weight_{i},{state[k].numel()});')
    L.append(" return gguf_write_close(w);")
    L.append("}")
    L.append(f"static const float {p}golden_residual[{len(full)}][{len(TOKENS) * E}]={{")
    for r in residual:
        L.append("{" + ",".join(hexf(x) for x in r.flatten().tolist()) + "},")
    L.append("};")
    L.append(f"static const float {p}golden_logits[{V}]={{{','.join(hexf(x) for x in logits.tolist())}}};")
    with open(out, "x") as f:
        f.write("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
