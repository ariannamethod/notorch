# reference oracle, not an inference path; deps: transformers 5.22.2, torch 2.15.0, torchvision 0.29.2
"""Reference dumps for LFM3-Omni towers from the official remote code, in float64.

Not a deliverable. Runs the published modules (src/*.py at revision 21d75123) on the
fixtures and writes NTD1 dumps the C comparer reads. Usage:
    python research/lfm3_omni_oracle.py PKG_DIR SRC_DIR FIX_DIR OUT_DIR [what ...]
PKG_DIR is a copy of SRC_DIR/*.py with an __init__.py, so the relative imports resolve.
"""
import struct
import sys
import wave

import numpy as np
import torch
from PIL import Image
from safetensors import safe_open

pkgdir, src, fix, out = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
sys.path.insert(0, pkgdir.rstrip("/") + "/..")
pkg = pkgdir.rstrip("/").split("/")[-1]
import importlib

cfgmod = importlib.import_module(pkg + ".configuration_lfm3_omni")
om = importlib.import_module(pkg + ".modeling_lfm3_omni")
vl = importlib.import_module(pkg + ".modeling_lfm3_vl")
codecmod = importlib.import_module(pkg + ".modeling_liquid_codec")
codeccfg = importlib.import_module(pkg + ".configuration_liquid_codec")
fe_mod = importlib.import_module(pkg + ".feature_extraction_lfm3_omni")
ip_mod = importlib.import_module(pkg + ".image_processing_lfm3_vl")

torch.set_grad_enabled(False)
cfg = cfgmod.Lfm3OmniConfig.from_pretrained(src)
for c in (cfg, cfg.vision_config, cfg.audio_encoder_config, cfg.audio_encoder_config.text_config, cfg.depth_config, cfg.text_config):
    c._attn_implementation = "eager"


def put(f, tag, layer, x):
    x = x.detach().to(torch.float64).contiguous().cpu().numpy()
    if x.ndim == 1:
        x = x[None]
    x = x.reshape(x.shape[0], -1) if x.ndim > 2 else x
    t = tag.encode()
    f.write(struct.pack("<I", len(t)) + t + struct.pack("<iIII", layer, x.shape[0], x.shape[1], 1))
    f.write(x.astype("<f8").tobytes())


def load(module, path, prefix):
    sd = {}
    with safe_open(path, "pt") as st:
        for k in st.keys():
            if k.startswith(prefix):
                sd[k[len(prefix):]] = st.get_tensor(k)
    missing, unexpected = module.load_state_dict(sd, strict=False)
    missing = [m for m in missing if not m.endswith("inv_freq")]
    assert not missing and not unexpected, (prefix, missing[:5], unexpected[:5])
    return module.double().eval()


MODEL = src + "/model.safetensors"


class Lcg:
    def __init__(self, seed):
        self.x = seed

    def step(self):
        self.x = (self.x * 6364136223846793005 + 1442695040888963407) & ((1 << 64) - 1)
        return self.x

    def draw(self):
        return np.float32((self.step() >> 40) / 8388608.0 - 1.0)

    def draw_int(self, n):
        return (self.step() >> 33) % n


def vision(img_path, name):
    ip = ip_mod.Lfm3VlImageProcessor(
        size={"longest_edge": 2097152, "shortest_edge": 65536}, patch_size=16, temporal_patch_size=2, merge_size=2,
        image_mean=[0.5] * 3, image_std=[0.5] * 3, resample=3)
    img = Image.open(img_path).convert("RGB")
    feat = ip(images=[img], return_tensors="pt")
    pv, grid = feat["pixel_values"].to(torch.float64), feat["image_grid_thw"]
    _, gh, gw = [int(v) for v in grid[0]]
    # back to [3][H][W] (temporal slice 0) for the pixel comparison
    p = pv.reshape(gh // 2, gw // 2, 2, 2, 3, 2, 16, 16)[:, :, :, :, :, 0]
    img_chw = p.permute(4, 0, 2, 5, 1, 3, 6).reshape(3, gh * 16, gw * 16)
    tower = load(vl.Lfm3VlVisionModel(cfg.vision_config), MODEL, "model.vision_tower.")
    proj = load(om.Lfm3OmniMultiModalProjector(cfg), MODEL, "model.multi_modal_projector.")
    # merge-block order -> raster order of the patch grid
    # order[r] is the merge-block index of raster position r
    order = torch.arange(gh * gw).reshape(gh // 2, gw // 2, 2, 2).permute(0, 2, 1, 3).reshape(-1)
    recs = []
    tower.embedding_norm.register_forward_hook(lambda m, i, o: recs.append(("vision.embed", -1, o[order])))
    for li, layer in enumerate(tower.layers):
        layer.register_forward_hook(lambda m, i, o, li=li: recs.append(("vision.layer", li, o[order])))
    hs = tower(pv, grid_thw=grid).last_hidden_state
    rows = proj(hs)
    with open(f"{out}/oracle_{name}.ntd", "wb") as f:
        f.write(b"NTD1")
        put(f, "vision.pixels", -1, img_chw.reshape(3 * gh * 16, gw * 16))
        for tag, li, x in recs:
            put(f, tag, li, x)
        put(f, "vision.out", 0, rows)
    print(name, "grid", gh, gw, "rows", rows.shape[0])


def audio(wav_path, name):
    with wave.open(wav_path) as w:
        assert w.getsampwidth() == 2 and w.getnchannels() == 1 and w.getframerate() == 16000
        pcm = np.frombuffer(w.readframes(w.getnframes()), dtype="<i2").astype(np.float32) / 32768.0
    fe = fe_mod.Lfm3OmniFeatureExtractor()
    feats = fe(pcm, sampling_rate=16000, return_tensors="pt")
    mask = feats["feature_attention_mask"]
    # the extractor's _log_mel, in float64 end to end (window, transform, bank)
    x = torch.nn.functional.pad(torch.tensor(pcm, dtype=torch.float64), (0, fe.pad_length(len(pcm)) - len(pcm)))
    spec = torch.stft(x, n_fft=fe.n_fft, hop_length=fe.hop_length, win_length=fe.win_length,
                      window=torch.hann_window(fe.win_length, dtype=torch.float64), center=False, return_complex=True)
    lm = (torch.from_numpy(fe.mel_filters).T @ spec.abs().square()).log10().clamp(min=fe.log_floor)
    mel = ((lm + fe.normalization_offset) / fe.normalization_divisor)[None]
    f32 = feats["input_features"].to(torch.float64)
    print(name, "float32 extractor vs float64: rel_l2", float((f32 - mel).norm() / mel.norm()))
    enc = load(om.Lfm3OmniEncoder(cfg.audio_encoder_config), MODEL, "model.audio_encoder.")
    proj = load(om.Lfm3OmniProjector(cfg), MODEL, "model.audio_projector.")
    recs = []
    enc.downsample.register_forward_hook(lambda m, i, o: recs.append(("audio.downsample", -1, o[0].T)))
    for li, layer in enumerate(enc.lfm.layers):
        layer.register_forward_hook(lambda m, i, o, li=li: recs.append(("audio", li, o[0])))
    hs, lens = enc(mel, mask.sum(-1))
    rows = proj(hs[0, : int(lens[0])])
    with open(f"{out}/oracle_{name}.ntd", "wb") as f:
        f.write(b"NTD1")
        put(f, "audio.mel", -1, mel[0])
        for tag, li, x in recs:
            put(f, tag, li, x)
        put(f, "audio.encoder", -1, hs[0, : int(lens[0])])
        put(f, "audio.out", -1, rows)
    print(name, "mel", tuple(mel.shape), "rows", rows.shape[0])


def depth(seed, name):
    g = Lcg(seed)
    hid = torch.tensor([g.draw() for _ in range(cfg.text_config.hidden_size)], dtype=torch.float64)
    codes = torch.tensor([g.draw_int(cfg.codebook_size - 1) for _ in range(cfg.num_codebooks)])
    dd = load(om.Lfm3OmniDepthDecoder(cfg), MODEL, "model.depth_decoder.")
    emb = torch.nn.Embedding(cfg.num_codebooks * cfg.codebook_size, cfg.text_config.hidden_size)
    with safe_open(MODEL, "pt") as st:
        emb.weight.copy_(st.get_tensor("model.codebook_embedding.weight").to(torch.float64))
    emb = emb.double()
    with open(f"{out}/oracle_{name}.ntd", "wb") as f:
        f.write(b"NTD1")
        put(f, "depth.hidden", -1, hid)
        put(f, "depth.codes", -1, codes.to(torch.float64))
        for p in range(3):
            masked = torch.tensor([True if p == 0 else (k % 3 != 0) if p == 1 else False for k in range(cfg.num_codebooks)])
            lg = dd(hid[None], codes[None], masked[None])[0]
            put(f, "depth.logits", p, lg)
        offs = torch.arange(cfg.num_codebooks) * cfg.codebook_size
        put(f, "depth.frame_embed", -1, emb(codes + offs).sum(0))
    print(name, "codes", codes.tolist())


def codec(seed, frames, name):
    g = Lcg(seed)
    ccfg = codeccfg.LiquidCodecConfig.from_pretrained(src + "/audio_tokenizer")
    ccfg._attn_implementation = "eager"
    m = codecmod.LiquidCodecModel(ccfg)
    load(m, src + "/audio_tokenizer/model.safetensors", "")
    codes = torch.tensor([[g.draw_int(2016) for _ in range(cfg.num_codebooks)] for _ in range(frames)])
    recs = []
    for li, layer in enumerate(m.decoder.lfm.layers):
        layer.register_forward_hook(lambda mm, i, o, li=li: recs.append(("codec", li, o[0])))
    m.decoder.lin.register_forward_hook(lambda mm, i, o: recs.append(("codec.spec", -1, o[0])))
    pcm = m.decode(codes.T[None]).audio_values[0]
    with open(f"{out}/oracle_{name}.ntd", "wb") as f:
        f.write(b"NTD1")
        put(f, "codec.codes", -1, codes.to(torch.float64))
        for tag, li, x in recs:
            put(f, tag, li, x)
        put(f, "codec.pcm", -1, pcm)
    print(name, "pcm", pcm.shape[0])


what = sys.argv[5:] or ["vis_grad", "vis_pos", "aud_tone", "aud_speech", "depth", "codec"]
for w in what:
    if w == "vis_grad": vision(fix + "/grad.ppm", w)
    if w == "vis_pos": vision(fix + "/test-1-positive.png", w)
    if w == "aud_tone": audio(fix + "/tone.wav", w)
    if w == "aud_speech": audio(fix + "/speech.wav", w)
    if w == "depth": depth(1, w)
    if w == "codec": codec(2, 25, w)
