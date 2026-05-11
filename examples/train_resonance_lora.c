/*
 * train_resonance_lora.c — Resonance 200M LoRA SFT, end-to-end through notorch.
 *
 * Architecture: 20 × ResonanceBlock (RMSNorm + Content QKV+RoPE+FlashAttn parallel
 *   with RRPRAM low-rank Wr_a×Wr_b, per-head sigmoid(gate) blend, output proj,
 *   RMSNorm, SwiGLU FFN) + final RMSNorm + out_head. dim=768 H=12 D=64 ffn=2048
 *   V=16384 R=48 ctx=2048.
 *
 * LoRA: rank=64, alpha=128, 7 targets × 20 layers = 140 pairs. Trainable ~18.7M.
 *   Targets: wq, wk, wv, wo, mlp_gate, mlp_up, mlp_down.
 *
 * Optimizer: Chuck (nt_tape_chuck_step). Verified stable at lr=1e-4 *after* the
 *   2026-05-11 NT_OP_MUL + NT_OP_SILU backward CPU-sync fix (without it, SwiGLU
 *   branch receives zero gradient and loss is flat regardless of lr).
 *
 * Gate blend: per-head sigmoid(gate) precomputed once into g_gate_sig /
 *   g_gate_one_minus tensors of shape [T*H*D], registered frozen each step.
 *   Allows the blend out = sig·c_out + (1−sig)·r_out via plain nt_mul + nt_add
 *   without a per-head broadcast primitive.
 *
 * Diagnostics: D1 per-target grad-norm dump at step 0 (verifies all 7 targets
 *   actually receive gradient — caught the SwiGLU bug). Periodic checkpoints
 *   every 250 steps + final. Each ckpt = 7 files (one per target class, since
 *   shapes differ: E×E vs E×M vs M×E), total ~75 MB.
 *
 * Build:
 *   cc -DUSE_CUDA -DUSE_BLAS -O2 -I<notorch> \
 *      train_resonance_lora.c <notorch>/notorch.c <notorch>/notorch_cuda.o \
 *      -L/usr/local/cuda/lib64 -lcudart -lcublas -lopenblas -lm \
 *      -o train_resonance_lora
 *
 * Run:
 *   ./train_resonance_lora smoke                  # forward sanity check
 *   ./train_resonance_lora train 2048 1500 1e-4   # T N_STEPS LR
 *
 * Reference run: 2026-05-11, A100 SXM 80GB, ~2 h, loss 3.5229 → 0.5927 (final),
 *   honest min 0.18, zero NaN. Phase 7 PASS 17/30 cells with Arianna voice
 *   markers. Adapter at ataeff/resonance/sft_v3_notorch/arianna_2026_05_11.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#include "notorch.h"

extern int gpu_init(void);
extern void gpu_shutdown(void);

/* ── Architecture constants (Resonance 200M production config) ───────────────── */
#define R_N_LAYER     20
#define R_N_EMBD      768
#define R_N_HEAD      12
#define R_HEAD_DIM    64
#define R_FFN_DIM     2048
#define R_VOCAB       16384
#define R_CTX_LEN     2048
#define R_RRPRAM_RANK 48

/* Per-block tensor indices into nt_load array */
typedef struct {
    int norm1, wq, wk, wv, wr_combined, gate, wo;
    int norm2, mlp_gate, mlp_up, mlp_down;
} BlockParams;

/* Per-block LoRA adapters (7 per block) */
typedef struct {
    nt_lora_pair lora_wq, lora_wk, lora_wv, lora_wo;
    nt_lora_pair lora_mlp_gate, lora_mlp_up, lora_mlp_down;
} BlockLoRA;

/* Globals */
static nt_tensor** g_params = NULL;
static int g_n_params = 0;
static BlockParams g_blocks[R_N_LAYER];
static BlockLoRA g_loras[R_N_LAYER];
static int g_tok_emb_param = -1, g_norm_f_param = -1, g_out_head_param = -1;

/* Precomputed per-head sigmoid(gate) and (1-sigmoid(gate)) expanded to [T*H*D]
 * for elementwise blend via nt_mul. Owned heap tensors, registered as frozen
 * params each step. */
static nt_tensor* g_gate_sig[R_N_LAYER];        /* [T*H*D] = sigmoid(gate[h]) replicated */
static nt_tensor* g_gate_one_minus[R_N_LAYER];  /* [T*H*D] = (1-sigmoid(gate[h])) */

static int precompute_gate_blends(int max_T) {
    int H = R_N_HEAD, D = R_HEAD_DIM;
    int len = max_T * H * D;
    for (int i = 0; i < R_N_LAYER; i++) {
        nt_tensor* gate = g_params[g_blocks[i].gate];  /* [H] */
        if (gate->len != H) {
            fprintf(stderr, "[gate] L%d shape mismatch: got %d want %d\n", i, gate->len, H);
            return -1;
        }
        nt_tensor* g_sig = nt_tensor_new(len);
        nt_tensor* g_one = nt_tensor_new(len);
        if (!g_sig || !g_one) return -1;
        for (int t = 0; t < max_T; t++) {
            for (int h = 0; h < H; h++) {
                float gate_h = gate->data[h];
                float sig_h = 1.0f / (1.0f + expf(-gate_h));
                for (int d = 0; d < D; d++) {
                    g_sig->data[t*H*D + h*D + d] = sig_h;
                    g_one->data[t*H*D + h*D + d] = 1.0f - sig_h;
                }
            }
        }
        g_gate_sig[i] = g_sig;
        g_gate_one_minus[i] = g_one;
    }
    fprintf(stderr, "[gate] precomputed %d × 2 tensors, T=%d H=%d D=%d\n",
            R_N_LAYER, max_T, H, D);
    return 0;
}

/* ── Load weights from native bin via nt_load and assign per-block indices ───── */
static int load_resonance_weights(const char* bin_path) {
    g_params = nt_load(bin_path, &g_n_params);
    if (!g_params || g_n_params != 11 * R_N_LAYER + 3) {
        fprintf(stderr, "[load] failed or wrong tensor count: %d (expected %d)\n",
                g_n_params, 11 * R_N_LAYER + 3);
        return -1;
    }
    /* Per-block: 11 tensors in order (norm1, wq, wk, wv, wr_combined, gate, wo, norm2, mlp_gate, mlp_up, mlp_down) */
    for (int i = 0; i < R_N_LAYER; i++) {
        int base = 11 * i;
        g_blocks[i].norm1       = base + 0;
        g_blocks[i].wq          = base + 1;
        g_blocks[i].wk          = base + 2;
        g_blocks[i].wv          = base + 3;
        g_blocks[i].wr_combined = base + 4;
        g_blocks[i].gate        = base + 5;
        g_blocks[i].wo          = base + 6;
        g_blocks[i].norm2       = base + 7;
        g_blocks[i].mlp_gate    = base + 8;
        g_blocks[i].mlp_up      = base + 9;
        g_blocks[i].mlp_down    = base + 10;
    }
    /* Globals */
    int g = 11 * R_N_LAYER;
    g_tok_emb_param  = g + 0;
    g_norm_f_param   = g + 1;
    g_out_head_param = g + 2;
    fprintf(stderr, "[load] %d tensors OK, %d blocks indexed\n", g_n_params, R_N_LAYER);
    return 0;
}

/* ── Init LoRA adapters (rank=64, alpha=128) ─────────────────────────────────── */
static int init_loras(int rank, float alpha) {
    int E = R_N_EMBD, M = R_FFN_DIM;
    for (int i = 0; i < R_N_LAYER; i++) {
        BlockLoRA* l = &g_loras[i];
        if (nt_lora_init(&l->lora_wq,       E, E, rank, alpha) < 0) return -1;
        if (nt_lora_init(&l->lora_wk,       E, E, rank, alpha) < 0) return -1;
        if (nt_lora_init(&l->lora_wv,       E, E, rank, alpha) < 0) return -1;
        if (nt_lora_init(&l->lora_wo,       E, E, rank, alpha) < 0) return -1;
        if (nt_lora_init(&l->lora_mlp_gate, E, M, rank, alpha) < 0) return -1;
        if (nt_lora_init(&l->lora_mlp_up,   E, M, rank, alpha) < 0) return -1;
        if (nt_lora_init(&l->lora_mlp_down, M, E, rank, alpha) < 0) return -1;
    }
    fprintf(stderr, "[lora] %d adapters init'd, rank=%d alpha=%.1f\n", 7 * R_N_LAYER, rank, alpha);
    return 0;
}

/* ── ResonanceBlock forward ──────────────────────────────────────────────────── */
static int block_forward(int x_idx, int layer_idx, int T, int use_lora) {
    BlockParams* b = &g_blocks[layer_idx];
    BlockLoRA* l = use_lora ? &g_loras[layer_idx] : NULL;
    int E = R_N_EMBD;
    int H = R_N_HEAD;
    int D = R_HEAD_DIM;

    /* Register base weights (frozen) */
    int norm1_idx = nt_tape_param_frozen(g_params[b->norm1]);
    int wq_idx    = nt_tape_param_frozen(g_params[b->wq]);
    int wk_idx    = nt_tape_param_frozen(g_params[b->wk]);
    int wv_idx    = nt_tape_param_frozen(g_params[b->wv]);
    int wo_idx    = nt_tape_param_frozen(g_params[b->wo]);
    int wr_idx    = nt_tape_param_frozen(g_params[b->wr_combined]);
    int norm2_idx = nt_tape_param_frozen(g_params[b->norm2]);
    int mg_idx    = nt_tape_param_frozen(g_params[b->mlp_gate]);
    int mu_idx    = nt_tape_param_frozen(g_params[b->mlp_up]);
    int md_idx    = nt_tape_param_frozen(g_params[b->mlp_down]);

    /* Pre-norm */
    int xn_idx = nt_seq_rmsnorm(x_idx, norm1_idx, T, E);
    if (xn_idx < 0) { fprintf(stderr, "[L%d] norm1 fail\n", layer_idx); return -1; }

    /* QKV (LoRA-wrapped if enabled) */
    int q_idx, k_idx, v_idx;
    if (l) {
        q_idx = nt_lora_forward(wq_idx, &l->lora_wq, xn_idx, T);
        k_idx = nt_lora_forward(wk_idx, &l->lora_wk, xn_idx, T);
        v_idx = nt_lora_forward(wv_idx, &l->lora_wv, xn_idx, T);
    } else {
        q_idx = nt_seq_linear(wq_idx, xn_idx, T);
        k_idx = nt_seq_linear(wk_idx, xn_idx, T);
        v_idx = nt_seq_linear(wv_idx, xn_idx, T);
    }
    if (q_idx < 0 || k_idx < 0 || v_idx < 0) { fprintf(stderr, "[L%d] qkv fail\n", layer_idx); return -1; }

    /* Apply RoPE to Q and K (default base 10000, even/odd interleave matches PyTorch) */
    q_idx = nt_rope(q_idx, T, D);
    k_idx = nt_rope(k_idx, T, D);
    if (q_idx < 0 || k_idx < 0) { fprintf(stderr, "[L%d] rope fail\n", layer_idx); return -1; }

    /* Content attention (multi-head causal SDPA) */
    int c_out_idx = nt_mh_causal_attention(q_idx, k_idx, v_idx, T, D);
    if (c_out_idx < 0) { fprintf(stderr, "[L%d] mh_attn fail\n", layer_idx); return -1; }

    /* RRPRAM low-rank attention */
    int r_out_idx = nt_rrpram_lowrank_attention(wr_idx, xn_idx, v_idx, T, E, H, D);
    if (r_out_idx < 0) { fprintf(stderr, "[L%d] rrpram fail\n", layer_idx); return -1; }

    /* Per-head sigmoid(gate) blend: out = sig_g · c_out + (1-sig_g) · r_out
     * sig_g and (1-sig_g) precomputed at load time (gate is frozen base param).
     * Pretrained gates skew content-favored (mean sigmoid ≈ 0.55-0.85 across layers). */
    int g_sig_idx = nt_tape_param_frozen(g_gate_sig[layer_idx]);
    int g_one_idx = nt_tape_param_frozen(g_gate_one_minus[layer_idx]);
    int c_scaled = nt_mul(c_out_idx, g_sig_idx);
    int r_scaled = nt_mul(r_out_idx, g_one_idx);
    if (c_scaled < 0 || r_scaled < 0) { fprintf(stderr, "[L%d] gate-mul fail\n", layer_idx); return -1; }
    int blended_idx = nt_add(c_scaled, r_scaled);
    if (blended_idx < 0) { fprintf(stderr, "[L%d] gate-blend add fail\n", layer_idx); return -1; }

    /* Output projection (LoRA-wrapped if enabled) + residual */
    int proj_idx;
    if (l) proj_idx = nt_lora_forward(wo_idx, &l->lora_wo, blended_idx, T);
    else   proj_idx = nt_seq_linear(wo_idx, blended_idx, T);
    if (proj_idx < 0) { fprintf(stderr, "[L%d] wo fail\n", layer_idx); return -1; }

    int x_after_attn = nt_add(x_idx, proj_idx);
    if (x_after_attn < 0) { fprintf(stderr, "[L%d] resid1 fail\n", layer_idx); return -1; }

    /* SwiGLU FFN: gate * silu(gate) * up → down */
    int xn2_idx = nt_seq_rmsnorm(x_after_attn, norm2_idx, T, E);
    if (xn2_idx < 0) { fprintf(stderr, "[L%d] norm2 fail\n", layer_idx); return -1; }

    int g_idx, u_idx;
    if (l) {
        g_idx = nt_lora_forward(mg_idx, &l->lora_mlp_gate, xn2_idx, T);
        u_idx = nt_lora_forward(mu_idx, &l->lora_mlp_up, xn2_idx, T);
    } else {
        g_idx = nt_seq_linear(mg_idx, xn2_idx, T);
        u_idx = nt_seq_linear(mu_idx, xn2_idx, T);
    }
    if (g_idx < 0 || u_idx < 0) { fprintf(stderr, "[L%d] ffn-gu fail\n", layer_idx); return -1; }

    int g_silu = nt_silu(g_idx);
    int gu_idx = nt_mul(g_silu, u_idx);
    if (g_silu < 0 || gu_idx < 0) { fprintf(stderr, "[L%d] silu/mul fail\n", layer_idx); return -1; }

    int down_idx;
    if (l) down_idx = nt_lora_forward(md_idx, &l->lora_mlp_down, gu_idx, T);
    else   down_idx = nt_seq_linear(md_idx, gu_idx, T);
    if (down_idx < 0) { fprintf(stderr, "[L%d] mlp_down fail\n", layer_idx); return -1; }

    int out_idx = nt_add(x_after_attn, down_idx);
    if (out_idx < 0) { fprintf(stderr, "[L%d] resid2 fail\n", layer_idx); return -1; }

    return out_idx;
}

extern void nt_tensor_sync_cpu(nt_tensor* t);

static void trace_dump(const char* label, int tape_idx) {
    nt_tape_entry* e = nt_tape_get()->entries + tape_idx;
    nt_tensor* t = e->output;
    nt_tensor_sync_cpu(t);
    printf("%s t0_first8=[", label);
    for (int i = 0; i < 8; i++) printf("%s%.4f", i?", ":"", t->data[i]);
    printf("] t3_first8=[");
    int E = R_N_EMBD;
    int T = t->len / E;
    int off = (T - 1) * E;
    for (int i = 0; i < 8; i++) printf("%s%.4f", i?", ":"", t->data[off + i]);
    printf("]\n");
    fflush(stdout);
}

/* ── Resonance forward: tokens → logits ──────────────────────────────────────── */
static int resonance_forward(int tokens_idx, int T, int use_lora) {
    int trace = getenv("NT_TRACE") != NULL;
    /* Token embedding lookup */
    int tok_emb_idx = nt_tape_param_frozen(g_params[g_tok_emb_param]);
    int h_idx = nt_seq_embedding(tok_emb_idx, -1, tokens_idx, T, R_N_EMBD);
    if (h_idx < 0) { fprintf(stderr, "[fwd] tok_emb fail\n"); return -1; }
    if (trace) trace_dump("tok_emb", h_idx);

    /* 20 blocks */
    for (int i = 0; i < R_N_LAYER; i++) {
        h_idx = block_forward(h_idx, i, T, use_lora);
        if (h_idx < 0) return -1;
        if (trace) {
            char buf[32]; snprintf(buf, sizeof(buf), "blk_%d", i);
            trace_dump(buf, h_idx);
        }
    }

    /* Final norm */
    int norm_f_idx = nt_tape_param_frozen(g_params[g_norm_f_param]);
    h_idx = nt_seq_rmsnorm(h_idx, norm_f_idx, T, R_N_EMBD);
    if (h_idx < 0) { fprintf(stderr, "[fwd] norm_f fail\n"); return -1; }

    /* Output head (frozen) */
    int out_head_idx = nt_tape_param_frozen(g_params[g_out_head_param]);
    int logits_idx = nt_seq_linear(out_head_idx, h_idx, T);
    if (logits_idx < 0) { fprintf(stderr, "[fwd] out_head fail\n"); return -1; }

    return logits_idx;
}

/* ── DIAG D1: per-target grad L2 norms (avg across layers) ─────────────────────
 * Called after nt_tape_backward, before nt_tape_adamw_step, on step 0 only.
 * Walks tape, finds entries whose ->output matches each persistent LoRA A/B,
 * reads ->grad on CPU, prints rollup. Diagnoses which target classes have
 * non-zero gradient flow. */
static void diag_grad_norms(void) {
    const char* names[7] = {"wq","wk","wv","wo","mlp_gate","mlp_up","mlp_down"};
    nt_tape* tp = nt_tape_get();

    for (int t = 0; t < 7; t++) {
        double sum_a = 0.0, sum_b = 0.0;
        int n_a = 0, n_b = 0;
        for (int L = 0; L < R_N_LAYER; L++) {
            BlockLoRA* l = &g_loras[L];
            nt_lora_pair* lp;
            switch (t) {
                case 0: lp = &l->lora_wq;       break;
                case 1: lp = &l->lora_wk;       break;
                case 2: lp = &l->lora_wv;       break;
                case 3: lp = &l->lora_wo;       break;
                case 4: lp = &l->lora_mlp_gate; break;
                case 5: lp = &l->lora_mlp_up;   break;
                case 6: lp = &l->lora_mlp_down; break;
                default: lp = NULL;
            }
            if (!lp) continue;
            int a_found = 0, b_found = 0;
            for (int i = 0; i < tp->count; i++) {
                nt_tape_entry* e = &tp->entries[i];
                if (!e->output) continue;
                if (e->output == lp->A && e->grad && !a_found) {
                    nt_tensor_sync_cpu(e->grad);
                    double s = 0.0;
                    int n = e->grad->len;
                    for (int j = 0; j < n; j++) {
                        float g = e->grad->data[j];
                        s += (double)g * (double)g;
                    }
                    sum_a += sqrt(s); n_a++; a_found = 1;
                }
                if (e->output == lp->B && e->grad && !b_found) {
                    nt_tensor_sync_cpu(e->grad);
                    double s = 0.0;
                    int n = e->grad->len;
                    for (int j = 0; j < n; j++) {
                        float g = e->grad->data[j];
                        s += (double)g * (double)g;
                    }
                    sum_b += sqrt(s); n_b++; b_found = 1;
                }
                if (a_found && b_found) break;
            }
        }
        fprintf(stderr, "  [D1] %-10s  n_gA=%2d avg|gA|=%.3e | n_gB=%2d avg|gB|=%.3e\n",
                names[t], n_a, n_a ? sum_a/(double)n_a : 0.0,
                          n_b, n_b ? sum_b/(double)n_b : 0.0);
    }
}

/* ── Main: smoke or train mode ───────────────────────────────────────────────── */
int main(int argc, char** argv) {
    const char* mode = (argc > 1) ? argv[1] : "smoke";
    if (gpu_init() != 0) { fprintf(stderr, "gpu_init failed\n"); return 1; }
    nt_set_gpu_mode(1);

    if (load_resonance_weights("/workspace/models/resonance/resonance_200m_native.bin") < 0)
        return 1;

    int max_T = (strcmp(mode, "smoke") == 0) ? 4 : 2048;
    if (precompute_gate_blends(max_T) < 0) {
        fprintf(stderr, "[init] gate precompute failed\n");
        return 1;
    }

    if (strcmp(mode, "smoke") == 0) {
        /* Smoke: 4-token forward, no LoRA, dump first few logits for PyTorch comparison */
        nt_tape_start();
        int T = 4;
        nt_tensor* tokens = nt_tensor_new(T);
        for (int i = 0; i < T; i++) tokens->data[i] = (float)(100 + i);  /* Arbitrary tokens */
        int tok_idx = nt_tape_param_frozen(tokens);

        int logits_idx = resonance_forward(tok_idx, T, /*use_lora=*/0);
        if (logits_idx < 0) { fprintf(stderr, "[smoke] forward failed\n"); return 1; }

        nt_tape_entry* pe = nt_tape_get()->entries + logits_idx;
        nt_tensor* logits = pe->output;
        /* Pull GPU → CPU mirror */
        nt_tensor_sync_cpu(logits);

        printf("[smoke] logits len=%d ndim=%d shape=[", logits->len, logits->ndim);
        for (int d = 0; d < logits->ndim; d++) printf("%s%d", d?",":"", logits->shape[d]);
        printf("]\n");
        printf("[smoke] first 8 logits at t=0:");
        for (int i = 0; i < 8; i++) printf(" %.4f", logits->data[i]);
        printf("\n");
        printf("[smoke] last 8 logits at t=T-1:");
        for (int i = 0; i < 8; i++) printf(" %.4f", logits->data[(T-1)*R_VOCAB + i]);
        printf("\n");
        printf("[smoke] OK\n");
    } else if (strcmp(mode, "train") == 0) {
        /* Training: load tokens, random T windows, forward+backward+chuck step. */
        const char* tokens_path = "/workspace/datasets/arianna/arianna_tokens.bin";
        FILE* tf = fopen(tokens_path, "rb");
        if (!tf) { fprintf(stderr, "[train] cannot open %s\n", tokens_path); return 1; }
        int n_tokens = 0;
        fread(&n_tokens, 4, 1, tf);
        int* tokens = (int*)malloc((size_t)n_tokens * sizeof(int));
        fread(tokens, 4, n_tokens, tf);
        fclose(tf);
        fprintf(stderr, "[train] loaded %d tokens\n", n_tokens);

        int T = (argc > 2) ? atoi(argv[2]) : 512;
        int N_STEPS = (argc > 3) ? atoi(argv[3]) : 200;
        float lr = (argc > 4) ? atof(argv[4]) : 1e-4f;
        fprintf(stderr, "[train] T=%d steps=%d lr=%.2e\n", T, N_STEPS, lr);

        if (init_loras(64, 128.0f) < 0) return 1;

        srand(42);
        for (int step = 0; step < N_STEPS; step++) {
            nt_tape_start();
            int start = rand() % (n_tokens - T - 1);
            nt_tensor* tok_in = nt_tensor_new(T);
            nt_tensor* tok_tgt = nt_tensor_new(T);
            nt_tensor* mask = nt_tensor_new(T);
            for (int i = 0; i < T; i++) {
                tok_in->data[i] = (float)tokens[start + i];
                tok_tgt->data[i] = (float)tokens[start + i + 1];
                mask->data[i] = 1.0f;
            }
            int tok_idx = nt_tape_param_frozen(tok_in);
            int tgt_idx = nt_tape_param(tok_tgt);
            int mask_idx = nt_tape_param(mask);

            int logits_idx = resonance_forward(tok_idx, T, /*use_lora=*/1);
            if (logits_idx < 0) { fprintf(stderr, "[train] step %d fwd fail\n", step); break; }

            int loss_idx = nt_seq_cross_entropy_masked(logits_idx, tgt_idx, mask_idx, T, R_VOCAB);
            if (loss_idx < 0) { fprintf(stderr, "[train] step %d ce fail\n", step); break; }

            nt_tape_entry* le = nt_tape_get()->entries + loss_idx;
            nt_tensor_sync_cpu(le->output);
            float loss_val = le->output->data[0];

            nt_tape_backward(loss_idx);
            if (step == 0) diag_grad_norms();
            /* Chuck — full notorch path. Was held off on prior session due to
             * "destabilizes on LoRA-scale" — but that was with half-broken
             * backward (NT_OP_MUL/SILU CPU-stale parent reads, fixed 2026-05-11).
             * Re-engaging Chuck as production optimizer; if instability
             * recurs, it's a different bug than the SwiGLU one. */
            nt_tape_chuck_step(lr, loss_val);

            if (step % 10 == 0 || step == N_STEPS - 1)
                fprintf(stderr, "  step %4d | loss %8.4f\n", step, loss_val);

            /* Periodic checkpoint every 250 steps + at final step. Flat layout:
             * 140 lora_pair tensors as one array (7 targets × 20 layers). */
            if ((step > 0 && step % 250 == 0) || step == N_STEPS - 1) {
                const char* target_names[7] = {
                    "wq", "wk", "wv", "wo", "mlp_gate", "mlp_up", "mlp_down"
                };
                nt_lora_pair flat[7 * R_N_LAYER];
                for (int L = 0; L < R_N_LAYER; L++) {
                    flat[L * 7 + 0] = g_loras[L].lora_wq;
                    flat[L * 7 + 1] = g_loras[L].lora_wk;
                    flat[L * 7 + 2] = g_loras[L].lora_wv;
                    flat[L * 7 + 3] = g_loras[L].lora_wo;
                    flat[L * 7 + 4] = g_loras[L].lora_mlp_gate;
                    flat[L * 7 + 5] = g_loras[L].lora_mlp_up;
                    flat[L * 7 + 6] = g_loras[L].lora_mlp_down;
                }
                char ckpt_path[256];
                if (step == N_STEPS - 1) {
                    snprintf(ckpt_path, sizeof(ckpt_path),
                             "/tmp/resonance_arianna_lora_final.bin");
                } else {
                    snprintf(ckpt_path, sizeof(ckpt_path),
                             "/tmp/resonance_arianna_lora_step%04d.bin", step);
                }
                /* nt_lora_save expects layout [layer*targets + target_idx],
                 * but the heterogeneous targets (E×E vs E×M vs M×E) break the
                 * single-shape contract. Save per-target group manually. */
                for (int t = 0; t < 7; t++) {
                    nt_lora_pair group[R_N_LAYER];
                    for (int L = 0; L < R_N_LAYER; L++) group[L] = flat[L * 7 + t];
                    char per_target[256];
                    snprintf(per_target, sizeof(per_target),
                             "%s.%s", ckpt_path, target_names[t]);
                    const char* one_name[1] = { target_names[t] };
                    int rc = nt_lora_save(group, R_N_LAYER, 1, one_name, per_target);
                    if (rc != 0) {
                        fprintf(stderr, "[ckpt] save %s failed rc=%d\n",
                                per_target, rc);
                    }
                }
                fprintf(stderr, "[ckpt] step %d → %s.{wq,wk,wv,wo,mlp_gate,mlp_up,mlp_down}\n",
                        step, ckpt_path);
            }
        }
    } else {
        fprintf(stderr, "[main] mode '%s' not implemented\n", mode);
        return 1;
    }

    return 0;
}
