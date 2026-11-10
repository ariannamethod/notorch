# reference oracle, not an inference path; deps: transformers 5.22.2, torch 2.15.0, torchvision 0.29.2
"""Greedy answers of the official model (float32, CPU) for the e2e cases. Not a deliverable.
    python research/lfm3_omni_oracle_gen.py PKG_DIR SRC_DIR FIX_DIR [case ...]
Prints per case: name, generated text ids, decoded text."""
import importlib
import json
import sys
import time
import wave

import numpy as np
import torch
from PIL import Image

pkgdir, src, fix = sys.argv[1:4]
sys.path.insert(0, pkgdir.rstrip("/") + "/..")
pkg = pkgdir.rstrip("/").split("/")[-1]
om = importlib.import_module(pkg + ".modeling_lfm3_omni")
cfgmod = importlib.import_module(pkg + ".configuration_lfm3_omni")
proc_mod = importlib.import_module(pkg + ".processing_lfm3_omni")
ip_mod = importlib.import_module(pkg + ".image_processing_lfm3_vl")
fe_mod = importlib.import_module(pkg + ".feature_extraction_lfm3_omni")
vp_mod = importlib.import_module(pkg + ".video_processing_lfm3_vl")
try:
    from transformers import PreTrainedTokenizerFast as TokCls
except ImportError:
    from transformers import TokenizersBackend as TokCls

torch.set_grad_enabled(False)
tok = TokCls(tokenizer_file=src + "/tokenizer.json", bos_token="<|startoftext|>", eos_token="<|im_end|>", pad_token="<|pad|>")
ip = ip_mod.Lfm3VlImageProcessor(size={"longest_edge": 2097152, "shortest_edge": 65536}, patch_size=16,
                                 temporal_patch_size=2, merge_size=2, image_mean=[0.5] * 3, image_std=[0.5] * 3, resample=3)
fe = fe_mod.Lfm3OmniFeatureExtractor()
vcfg = {k: v for k, v in json.load(open(src + "/video_preprocessor_config.json")).items() if k not in ("video_processor_type", "auto_map")}
proc = proc_mod.Lfm3OmniProcessor(image_processor=ip, feature_extractor=fe, tokenizer=tok, video_processor=vp_mod.Lfm3VlVideoProcessor(**vcfg),
                                  chat_template=open(src + "/chat_template.jinja").read(),
                                  vision_start_token="<|vision_start|>", vision_end_token="<|vision_end|>")
cfg = cfgmod.Lfm3OmniConfig.from_pretrained(src)
cfg._attn_implementation = "eager"
for c in (cfg.vision_config, cfg.audio_encoder_config, cfg.audio_encoder_config.text_config, cfg.depth_config, cfg.text_config):
    c._attn_implementation = "eager"
t0 = time.time()
model = om.Lfm3OmniForConditionalGeneration.from_pretrained(src, config=cfg, dtype=torch.float32).eval()
print("loaded", round(time.time() - t0, 1), "s", file=sys.stderr)


def wav(path):
    with wave.open(path) as w:
        return np.frombuffer(w.readframes(w.getnframes()), dtype="<i2").astype(np.float32) / 32768.0


cases = {
    "text": ([{"role": "user", "content": "The capital of France is"}], None, None, 64),
    "image": ([{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "Describe this image in one sentence."}]}],
              [Image.open(fix + "/test-1-positive.png").convert("RGB")], None, 64),
    "asr": ([{"role": "system", "content": "Perform ASR in English."}, {"role": "user", "content": [{"type": "audio"}]}],
            None, [wav(fix + "/speech.wav")], 96),
}
want = sys.argv[4:] or list(cases)
for name in want:
    conv, images, audio, n = cases[name]
    text = proc.apply_chat_template(conv, add_generation_prompt=True, tokenize=False)
    kw = {}
    if images is not None: kw["images"] = images
    if audio is not None: kw["audio"], kw["sampling_rate"] = audio, 16000
    inputs = proc(text=text, return_tensors="pt", **kw)
    inputs.pop("mm_token_type_ids", None)
    t0 = time.time()
    out = model.generate(**inputs, max_new_tokens=n, do_sample=False)
    seq = out.sequences[0, inputs["input_ids"].shape[1]:int(out.sequence_lengths[0])].tolist()
    print(name, ",".join(map(str, seq)), flush=True)
    print(name, "text", json.dumps(tok.decode(seq, skip_special_tokens=True)), round(time.time() - t0, 1), "s", flush=True)
