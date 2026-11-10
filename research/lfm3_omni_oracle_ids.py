# reference oracle, not an inference path; deps: transformers 5.22.2, torch 2.15.0, torchvision 0.29.2
"""Prompt ids from the official processor for the CLI's prompt-assembly gate. Not a deliverable.
    python research/lfm3_omni_oracle_ids.py PKG_DIR SRC_DIR FIX_DIR
Prints one line per case: name, then the ids comma-separated."""
import importlib
import sys
import wave

import numpy as np
from PIL import Image
from transformers import AutoTokenizer

pkgdir, src, fix = sys.argv[1:4]
sys.path.insert(0, pkgdir.rstrip("/") + "/..")
pkg = pkgdir.rstrip("/").split("/")[-1]
proc_mod = importlib.import_module(pkg + ".processing_lfm3_omni")
ip_mod = importlib.import_module(pkg + ".image_processing_lfm3_vl")
fe_mod = importlib.import_module(pkg + ".feature_extraction_lfm3_omni")

import json
try:
    from transformers import PreTrainedTokenizerFast as TokCls
except ImportError:
    from transformers import TokenizersBackend as TokCls
tok = TokCls(tokenizer_file=src + "/tokenizer.json", bos_token="<|startoftext|>", eos_token="<|im_end|>", pad_token="<|pad|>")
ip = ip_mod.Lfm3VlImageProcessor(size={"longest_edge": 2097152, "shortest_edge": 65536}, patch_size=16,
                                 temporal_patch_size=2, merge_size=2, image_mean=[0.5] * 3, image_std=[0.5] * 3, resample=3)
fe = fe_mod.Lfm3OmniFeatureExtractor()
vp_mod = importlib.import_module(pkg + ".video_processing_lfm3_vl")
vcfg = json.load(open(src + "/video_preprocessor_config.json"))
vcfg = {k: v for k, v in vcfg.items() if k not in ("video_processor_type", "auto_map")}
vp = vp_mod.Lfm3VlVideoProcessor(**vcfg)
template = open(src + "/chat_template.jinja").read()
proc = proc_mod.Lfm3OmniProcessor(image_processor=ip, feature_extractor=fe, tokenizer=tok, video_processor=vp,
                                  chat_template=template, vision_start_token="<|vision_start|>", vision_end_token="<|vision_end|>")


def wav(path):
    with wave.open(path) as w:
        return np.frombuffer(w.readframes(w.getnframes()), dtype="<i2").astype(np.float32) / 32768.0


cases = {
    "text": ([{"role": "user", "content": "The capital of France is"}], None, None),
    "system": ([{"role": "system", "content": "Respond in text."}, {"role": "user", "content": "Name a famous bridge in San Francisco."}], None, None),
    "image": ([{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "Describe this image in one sentence."}]}],
              [Image.open(fix + "/test-1-positive.png").convert("RGB")], None),
    "audio": ([{"role": "system", "content": "Perform ASR in English."}, {"role": "user", "content": [{"type": "audio"}]}],
              None, [wav(fix + "/speech.wav")]),
}
for name, (conv, images, audio) in cases.items():
    text = proc.apply_chat_template(conv, add_generation_prompt=True, tokenize=False)
    kw = {}
    if images is not None: kw["images"] = images
    if audio is not None:
        kw["audio"] = audio
        kw["sampling_rate"] = 16000
    out = proc(text=text, **kw)
    ids = out["input_ids"]
    ids = ids[0] if isinstance(ids[0], list) else ids
    print(name, ",".join(str(int(i)) for i in ids))
