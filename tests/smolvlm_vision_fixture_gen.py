#!/usr/bin/env python3
"""Freeze a tiny SmolVLM/Idefics3 vision fixture from an independent numpy reference.

The reference is the path ocelli's vision.c runs and the notorch port must match:
bilinear idefics3 tiles plus a global frame, SigLIP pre-norm blocks with the f16-rounded
tanh GELU of llama.cpp's clip, the idefics3 pixel shuffle and the connector projection.
Transformers is not the oracle here on purpose: its GELU is exact f32, ocelli's is not.

The f16 GELU is a step function, so a value within an ulp of an f16 midpoint can round
differently under another BLAS summation order. The weight seed is chosen so that every
GELU input and output of the frozen frame stays 1e-5 (relative) away from a midpoint."""
import sys
import numpy as np

F = np.float32
D, LAYERS, HEADS, FFN, S, P, SCALE, TEXT, LONGEST = 8, 2, 2, 4, 8, 2, 2, 8, 16
EPS, MEAN, STD = F(1e-5), [0.4, 0.5, 0.6], [0.3, 0.5, 0.7]
G = S // P; NP = G * G; TOK = (G // SCALE) ** 2; HD = D // HEADS
H, W = 7, 11
RGB = np.array([(i * 7 + 3) % 256 for i in range(H * W * 3)], np.uint8).reshape(H, W, 3)
INSTRUCTION = 'hi'
MARGIN = 1e-5

def tensors():
    t = [('v.patch_embd.weight', [P, P, 3, D], 0), ('v.patch_embd.bias', [D], 0),
         ('v.position_embd.weight', [D, NP], 0)]
    for l in range(LAYERS):
        b = 'v.blk.%d.' % l
        t += [(b + 'ln1.weight', [D], 1), (b + 'ln1.bias', [D], 0)]
        for n in ('attn_q', 'attn_k', 'attn_v', 'attn_out'):
            t += [(b + n + '.weight', [D, D], 0), (b + n + '.bias', [D], 0)]
        t += [(b + 'ln2.weight', [D], 1), (b + 'ln2.bias', [D], 0)]
        fc1, fc2 = ('ffn_up', 'ffn_down') if l == 0 else ('ffn_down', 'ffn_up')
        t += [(b + fc1 + '.weight', [D, FFN], 0), (b + fc1 + '.bias', [FFN], 0),
              (b + fc2 + '.weight', [FFN, D], 0), (b + fc2 + '.bias', [D], 0)]
    t += [('v.post_ln.weight', [D], 1), ('v.post_ln.bias', [D], 0),
          ('mm.model.fc.weight', [D * SCALE * SCALE, TEXT], 0)]
    return t

def values(n, k, plus, seed):
    h = (np.arange(n, dtype=np.uint64) * 7919 + k * 104729 + seed * 15485863) % 4093
    return ((h % 23).astype(np.int64) - 11).astype(F) / F(64) + F(plus)

def weights(seed):
    w = {}
    for k, (name, shape, plus) in enumerate(tensors()):
        w[name] = values(int(np.prod(shape)), k, plus, seed).reshape(list(reversed(shape)))
    return w

def bilinear(src, tw, th):
    _, h, w = src.shape
    sx, sy = F(w) / F(tw), F(h) / F(th)
    out = np.empty((3, th, tw), F)
    for y in range(th):
        fy = (F(y) + F(0.5)) * sy - F(0.5); y0 = int(np.floor(fy)); y1 = y0 + 1; wy = fy - F(y0)
        y0 = max(y0, 0); y1 = min(y1, h - 1)
        for x in range(tw):
            fx = (F(x) + F(0.5)) * sx - F(0.5); x0 = int(np.floor(fx)); x1 = x0 + 1; wx = fx - F(x0)
            x0 = max(x0, 0); x1 = min(x1, w - 1)
            out[:, y, x] = (src[:, y0, x0] * (F(1) - wx) * (F(1) - wy) + src[:, y0, x1] * wx * (F(1) - wy)
                            + src[:, y1, x0] * (F(1) - wx) * wy + src[:, y1, x1] * wx * wy)
    return out

def prepare():
    img = RGB.transpose(2, 0, 1).astype(F) / F(255)
    h, w = H, W
    s = F(LONGEST) / F(max(w, h))
    nw, nh = max(int(F(w) * s + F(0.5)), 1), max(int(F(h) * s + F(0.5)), 1)
    if (nw, nh) != (w, h):
        img = bilinear(img, nw, nh); w, h = nw, nh
    cols, rows = (w + S - 1) // S, (h + S - 1) // S
    frames = []
    if rows * cols > 1:
        grid = bilinear(img, cols * S, rows * S)
        frames = [grid[:, r * S:(r + 1) * S, c * S:(c + 1) * S] for r in range(rows) for c in range(cols)]
    else:
        rows = cols = 1
    frames.append(bilinear(img, S, S))
    px = np.stack(frames)
    for c in range(3):
        px[:, c] = (px[:, c] - F(MEAN[c])) / F(STD[c])
    return px, rows, cols

def layernorm(x, w, b):
    mu = x.astype(np.float64).mean(1, keepdims=True)
    var = ((x.astype(np.float64) - mu) ** 2).mean(1, keepdims=True)
    inv = F(1) / np.sqrt(var.astype(F) + EPS)
    return ((x - mu.astype(F)) * inv) * w + b

def half_margin(v):
    v = v.astype(np.float64).ravel()
    h = v.astype(np.float16)
    up, dn = np.nextafter(h, np.float16(np.inf)), np.nextafter(h, np.float16(-np.inf))
    mid = np.minimum(np.abs(v - (h.astype(np.float64) + up) / 2), np.abs(v - (h.astype(np.float64) + dn) / 2))
    return (mid / np.maximum(np.abs(v), 2.0 ** -14)).min()

def gelu16(u, margins):
    c, a = F(0.79788456080286535588), F(0.044715)
    live = (u > F(-10)) & (u < F(10))
    margins.append(half_margin(u[live]))
    xr = u.astype(np.float16).astype(F)
    g = F(0.5) * xr * (F(1) + np.tanh(c * (xr + a * xr * xr * xr)))
    margins.append(half_margin(g[live]))
    out = g.astype(np.float16).astype(F)
    out[u <= F(-10)] = 0
    out[u >= F(10)] = u[u >= F(10)]
    return out

def tower(w, frame, margins):
    X = np.stack([frame[:, ph * P:(ph + 1) * P, pw * P:(pw + 1) * P].reshape(-1)
                  for ph in range(G) for pw in range(G)])
    x = X @ w['v.patch_embd.weight'].reshape(D, -1).T + w['v.patch_embd.bias']
    x = x + w['v.position_embd.weight']
    scale = F(1) / np.sqrt(F(HD))
    for l in range(LAYERS):
        b = 'v.blk.%d.' % l
        lin = lambda t, n: t @ w[b + n + '.weight'].T + w[b + n + '.bias']
        t = layernorm(x, w[b + 'ln1.weight'], w[b + 'ln1.bias'])
        q, k, v = lin(t, 'attn_q'), lin(t, 'attn_k'), lin(t, 'attn_v')
        att = np.empty_like(q)
        for hh in range(HEADS):
            sl = slice(hh * HD, (hh + 1) * HD)
            sc = (q[:, sl] @ k[:, sl].T) * scale
            e = np.exp(sc - sc.max(1, keepdims=True))
            att[:, sl] = (e / e.sum(1, keepdims=True)) @ v[:, sl]
        x = x + lin(att, 'attn_out')
        fc1, fc2 = ('ffn_up', 'ffn_down') if l == 0 else ('ffn_down', 'ffn_up')
        t = layernorm(x, w[b + 'ln2.weight'], w[b + 'ln2.bias'])
        x = x + lin(gelu16(lin(t, fc1), margins), fc2)
    return layernorm(x, w['v.post_ln.weight'], w['v.post_ln.bias'])

def connect(w, hidden):
    g = G // SCALE
    sh = np.empty((g * g, D * SCALE * SCALE), F)
    for hg in range(g):
        for wg in range(g):
            for hin in range(SCALE):
                for win in range(SCALE):
                    k = hin * SCALE + win
                    sh[hg * g + wg, k * D:(k + 1) * D] = hidden[(hg * SCALE + hin) * G + wg * SCALE + win]
    return sh @ w['mm.model.fc.weight'].T

SPECIALS = ['<|im_start|>', '<end_of_utterance>', '<fake_token_around_image>', '<global-img>',
            '<row_1_col_1>', '<row_1_col_2>', '<row_2_col_1>', '<row_2_col_2>', '<image>']

def prompt_ids(rows, cols):
    img = '<image>' * TOK
    s = '<|im_start|>User:\n'
    if rows * cols > 1:
        for r in range(rows):
            for c in range(cols):
                s += '<fake_token_around_image><row_%d_col_%d>' % (r + 1, c + 1) + img
            s += '\n'
        s += '\n<fake_token_around_image><global-img>' + img + '<fake_token_around_image>' + INSTRUCTION
    else:
        s += '<fake_token_around_image><global-img>' + img + '<fake_token_around_image>\n' + INSTRUCTION
    s += '<end_of_utterance>\nAssistant:'
    assert s.isascii()
    ids, i = [], 0
    while i < len(s):
        m = max((t for t in SPECIALS if s.startswith(t, i)), key=len, default=None)
        if m:
            ids.append(256 + SPECIALS.index(m)); i += len(m)
        else:
            ids.append(ord(s[i])); i += 1
    return ids

def hexf(a): return ','.join(float(x).hex() + 'f' for x in np.asarray(a, F).ravel())

px, rows, cols = prepare()
for seed in range(1, 100000):
    w = weights(seed); margins = []
    hidden = tower(w, px[-1], margins)
    if min(margins) > MARGIN:
        break
else:
    sys.exit('no seed keeps the f16 GELU off its midpoints')
out_rows = connect(w, hidden)
ids = prompt_ids(rows, cols)

L = ['/* Independent numpy %s reference. Generated by smolvlm_vision_fixture_gen.py; seed %d keeps '
     'every f16 GELU input and output %.0e (relative) off a rounding midpoint. */' % (np.__version__, seed, MARGIN),
     'static int smolvlm_vision_fixture(const char *path,int variant) {',
     'gguf_writer *w=gguf_write_open(path);if(!w)return 1;',
     'gguf_write_kv_str(w,"general.architecture","clip");',
     'gguf_write_kv_str(w,"clip.projector_type",variant==4?"mlp":"idefics3");',
     'gguf_write_kv_bool(w,"clip.has_vision_encoder",1);']
for key, val in (('embedding_length', D), ('block_count', LAYERS), ('attention.head_count', HEADS),
                 ('feed_forward_length', FFN), ('image_size', S), ('patch_size', P), ('projection_dim', TEXT)):
    L.append('gguf_write_kv_u32(w,"clip.vision.%s",%d);' % (key, val))
L += ['if(variant!=3)gguf_write_kv_u32(w,"clip.vision.projector.scale_factor",%d);' % SCALE,
      'if(variant!=2)gguf_write_kv_u32(w,"clip.vision.preproc_image_size",%d);' % LONGEST,
      'gguf_write_kv_f32(w,"clip.vision.attention.layer_norm_epsilon",%sf);' % float(EPS).hex(),
      'const float mean[3]={%s},std[3]={%s};' % (hexf(MEAN), hexf(STD)),
      'gguf_write_kv_f32_array(w,"clip.vision.image_mean",mean,3);gguf_write_kv_f32_array(w,"clip.vision.image_std",std,3);',
      'struct tensor { const char *name; int index,plus,rank; uint64_t shape[4]; } ts[]={']
for k, (name, shape, plus) in enumerate(tensors()):
    s = shape + [1] * (4 - len(shape))
    L.append('{"%s",%d,%d,%d,{%d,%d,%d,%d}},' % (name, k, plus, len(shape), *s))
L += ['};', 'const char *drop=variant==1?"v.blk.1.ln2.bias":"";',
      'for(size_t t=0;t<sizeof(ts)/sizeof(ts[0]);t++)if(strcmp(ts[t].name,drop))gguf_write_tensor_decl(w,ts[t].name,ts[t].rank,ts[t].shape,GGUF_TYPE_F32);',
      'for(size_t t=0;t<sizeof(ts)/sizeof(ts[0]);t++){struct tensor *a=&ts[t];if(!strcmp(a->name,drop))continue;',
      'size_t n=a->shape[0]*a->shape[1]*a->shape[2]*a->shape[3];float *x=malloc(n*sizeof(float));if(!x){gguf_write_abort(w);return 1;}',
      'for(size_t j=0;j<n;j++){uint64_t h=((uint64_t)j*7919u+(uint64_t)a->index*104729u+(uint64_t)%du*15485863u)%%4093u;x[j]=(float)((int)(h%%23u)-11)/64.0f+(float)a->plus;}' % seed,
      'int rc=gguf_write_tensor_f32(w,a->name,x,n);free(x);if(rc){gguf_write_abort(w);return 1;}',
      '}return gguf_write_close(w);}',
      'enum { smolvlm_rows=%d, smolvlm_cols=%d, smolvlm_frames=%d };' % (rows, cols, len(px)),
      'static const char smolvlm_instruction[]="%s";' % INSTRUCTION,
      'static const float smolvlm_pixels[]={%s};' % hexf(px),
      'static const float smolvlm_global_hidden[]={%s};' % hexf(hidden),
      'static const float smolvlm_global_rows[]={%s};' % hexf(out_rows),
      'static const int smolvlm_prompt_ids[]={%s};' % ','.join(map(str, ids))]
open(sys.argv[1], 'w').write('\n'.join(L) + '\n')
print('smolvlm fixture: seed', seed, 'grid %dx%d' % (rows, cols), 'frames', len(px), 'prompt ids', len(ids),
      'min GELU margin %.2e' % min(margins))
