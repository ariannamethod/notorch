#!/usr/bin/env python3
"""Optional real-checkpoint gate: native C vs Transformers 5.15 eager/F32.
Python is an independent reference, never the production inference path.
Run with --snapshot PATH --weights MODEL.gguf --vision VISION.gguf --out DIR.
The fixed gates are declared before reading results. Missing dependencies fail
with an explicit SKIPPED message; they cannot produce a passing receipt.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import numpy as np
from PIL import Image

GATES = {"patches": (1e-6, 1e-6), "features": (2e-4, 2e-5), "logits": (2e-4, 2e-5)}
CASES = ("red", "blue", "layout", "small", "downsample")

def images(out):
    for name, size in zip(CASES, ((256,256),(256,256),(319,173),(19,43),(777,513))):
        w,h=size
        if name in ("red","blue"):
            a=np.empty((h,w,3),dtype=np.uint8); a[:]=(240,20,10) if name=="red" else (10,20,240)
        else:
            yy,xx=np.indices((h,w));a=np.stack(((xx*7+yy*13)%256,(xx*3+yy*5)%256,(xx+yy*11)%256),-1).astype(np.uint8)
        Image.fromarray(a).save(out/f"{name}.png")

def reference(args):
    try:
        import torch
        import transformers
        from safetensors import safe_open
        from transformers import AutoTokenizer, Qwen3_5Config, Qwen3_5ForConditionalGeneration
        from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5VisionModel
        from transformers.models.qwen2_vl.image_processing_pil_qwen2_vl import Qwen2VLImageProcessorPil
    except ImportError as e:
        raise SystemExit(f"SKIPPED: reference dependencies unavailable: {e}")
    torch.set_num_threads(2)
    if transformers.__version__!="5.15.0": raise SystemExit("reference requires transformers==5.15.0")
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True);images(out)
    config=Qwen3_5Config.from_pretrained(args.snapshot,local_files_only=True)
    config.vision_config._attn_implementation="eager"
    vc=Qwen3_5VisionModel(config.vision_config).float().eval()
    with safe_open(str(Path(args.snapshot)/"model.safetensors"),framework="pt") as sf:
        state={k[13:]:sf.get_tensor(k).float() for k in sf.keys() if k.startswith("model.visual.")}
    vc.load_state_dict(state,strict=True);del state
    processor_cfg=json.loads((Path(args.snapshot)/"processor_config.json").read_text())["image_processor"]
    processor=Qwen2VLImageProcessorPil(**processor_cfg)
    prepared={}
    with torch.no_grad():
        for name in CASES:
            px=processor(images=Image.open(out/f"{name}.png"),return_tensors="pt",max_pixels=262144,min_pixels=65536)
            result=vc(px["pixel_values"],grid_thw=px["image_grid_thw"])
            px["pixel_values"].numpy().tofile(out/f"{name}.ref.patches.f32")
            result.pooler_output.numpy().tofile(out/f"{name}.ref.features.f32")
            prepared[name]=px
            print("reference encoder",name,tuple(result.pooler_output.shape),flush=True)
    del vc
    if args.encoder_only:return
    tok=AutoTokenizer.from_pretrained(args.snapshot,local_files_only=True)
    model=Qwen3_5ForConditionalGeneration.from_pretrained(args.snapshot,dtype=torch.float32,attn_implementation="eager",local_files_only=True).eval()
    prompt="<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>\nIs the image predominantly red or blue?\nA: red\nB: blue\nAnswer with A or B only.<|im_end|>\n<|im_start|>assistant\n"
    for name in ("red","blue","layout","text"):
        px=None if name=="text" else prepared[name]
        text=prompt.replace("<|vision_start|><|image_pad|><|vision_end|>","") if px is None else prompt.replace("<|image_pad|>","<|image_pad|>"*(int(px["image_grid_thw"].prod())//4))
        ids=tok.encode(text,add_special_tokens=False);np.asarray(ids,dtype="<i4").tofile(out/f"{name}.ids.i32")
        inputs=torch.tensor([ids]); types=(inputs==config.image_token_id).int()
        positions,_=model.model.get_rope_index(inputs,types,image_grid_thw=None if px is None else px["image_grid_thw"])
        nxt=int(positions.max())+1
        kwargs={} if px is None else dict(pixel_values=px["pixel_values"],image_grid_thw=px["image_grid_thw"],mm_token_type_ids=types)
        with torch.no_grad():
            result=model(input_ids=inputs,position_ids=positions,use_cache=True,**kwargs)
            all_logits=[result.logits[0,-1].numpy().copy()];greedy=[]
            for step in range(3):
                token=int(result.logits[0,-1].argmax());greedy.append(token)
                result=model(input_ids=torch.tensor([[token]]),position_ids=torch.full((3,1,1),nxt+step,dtype=torch.long),past_key_values=result.past_key_values,use_cache=True)
                all_logits.append(result.logits[0,-1].numpy().copy())
        np.stack(all_logits).tofile(out/f"{name}.ref.logits.f32")
        (out/f"{name}.ref.json").write_text(json.dumps({"greedy":greedy,"next_position":nxt,"input_tokens":len(ids)}))
        print("reference decoder",name,greedy,all_logits[0][[32,33]].tolist(),flush=True)

def compare(name,kind,ref,actual):
    a=np.fromfile(ref,dtype="<f4");b=np.fromfile(actual,dtype="<f4")
    if a.shape!=b.shape or not a.size or not np.isfinite(a).all() or not np.isfinite(b).all():raise AssertionError(f"{name} {kind}: shape/finiteness")
    d=a.astype(np.float64)-b;mx=float(np.abs(d).max());rel=float(np.linalg.norm(d)/max(np.linalg.norm(a.astype(np.float64)),1e-30))
    atol,rtol=GATES[kind]
    print(json.dumps(dict(case=name,kind=kind,max_abs=mx,relative_l2=rel,pass_gate=mx<=atol and rel<=rtol)),flush=True)
    if mx>atol or rel>rtol:raise AssertionError(f"{name} {kind} outside fixed gate")
    # The same gate must reject disconnected image features / zero logits.
    if kind!="patches" and not np.any(np.abs(a)>atol):raise AssertionError("reference cannot detect zero-row red hand")

def check(args):
    out=Path(args.out);env={**os.environ,"OPENBLAS_NUM_THREADS":"1","OMP_NUM_THREADS":"1","NT_ATTN_THREADS":"2"}
    probe=Path(__file__).resolve().parents[1]/"qwen35_vision_probe"
    for name in CASES:
        subprocess.run([str(probe),args.vision,str(out/f"{name}.png"),"262144",str(out/f"{name}.c")],check=True,env=env)
        for kind in ("patches","features"):compare(name,kind,out/f"{name}.ref.{kind}.f32",out/f"{name}.c.{kind}.f32")
    if args.encoder_only:return
    for name in ("red","blue","layout","text"):
        image="-" if name=="text" else str(out/f"{name}.png")
        # Non-square layout exercises all 3 rotary axes and physical vs rotary offsets.
        for chunk in ((1,7,32) if name=="layout" else (32,)):
            prefix=out/f"{name}.c{chunk}"
            subprocess.run([str(probe),args.vision,image,"262144",str(prefix),args.weights,str(out/f"{name}.ids.i32"),str(chunk),"3"],check=True,env=env)
            compare(name,"logits",out/f"{name}.ref.logits.f32",str(prefix)+".logits.f32")
            result=json.loads(Path(str(prefix)+".json").read_text());ref=json.loads((out/f"{name}.ref.json").read_text())
            assert result["greedy"]==ref["greedy"]
            if name!="text":assert result["next_position"]==ref["next_position"]
    red=np.fromfile(out/"red.c32.logits.f32",dtype="<f4");blue=np.fromfile(out/"blue.c32.logits.f32",dtype="<f4")
    assert not np.array_equal(red,blue),"image substitution did not change logits"
    assert red[32]>red[33] and blue[33]>blue[32],"red/blue choice failed"
    print("PASS: real image encoder, full logits, greedy continuation, chunking and color counterfactual",flush=True)

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--snapshot",required=True);p.add_argument("--weights",required=True);p.add_argument("--vision",required=True);p.add_argument("--out",required=True);p.add_argument("--phase",choices=("all","reference","check"),default="all");p.add_argument("--encoder-only",action="store_true");args=p.parse_args()
    if args.phase=="all":
        # The full reference and native text models never coexist in RAM.
        subprocess.run([sys.executable,__file__,*sys.argv[1:],"--phase","reference"],check=True)
        check(args)
    elif args.phase=="reference":reference(args)
    else:check(args)
