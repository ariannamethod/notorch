/*
 * infer_janus.c — Universal Janus RRPRAM inference on notorch
 *
 * Auto-detects weight format:
 *   - DoE char:  [int32 n_params][weights...]  (V=256, MT=256)
 *   - BPE header: [V, E, H, D, BLK, FFN, MT][weights...]
 *   - ARIN:      "NIRA"[int32 n_params][pad...][weights...]
 *
 * Tested on all Janus weight variants (char, BPE, hybrid, resonance).
 *
 * Build: make infer
 * Run:   ./infer_janus_nt <weights.bin> [prompt] [max_tokens] [temp]
 */

#include "notorch.h"
#include <stdio.h>
#include <string.h>
#include <sys/time.h>

// ── Runtime config (filled from file header) ─────────────────────────────────

static int CFG_V, CFG_E, CFG_H, CFG_D, CFG_BLK, CFG_FFN, CFG_MT;
static int CFG_HAS_JANUS = 1;  // 1 = full Janus (wvr+wj+gate[H,3]), 0 = resonance (gate[H,1])

// ── Weight layout ────────────────────────────────────────────────────────────

#define MAX_BLK 24

typedef struct {
    float *tok_emb, *pos_emb;
    struct {
        float *rms1, *wq, *wk, *wv, *wr, *wvr, *wj, *gate, *wo;
        float *rms2, *wg, *wu, *wd;
    } b[MAX_BLK];
    float *rms_f, *head;
} Weights;

static int param_count(void) {
    int V = CFG_V, E = CFG_E, H = CFG_H, BLK = CFG_BLK, FFN = CFG_FFN, MT = CFG_MT;
    int s = V*E + MT*E;
    int gate_size = CFG_HAS_JANUS ? H*3 : H*1;
    int n_linear = CFG_HAS_JANUS ? 6 : 4;  // Janus: wq,wk,wv,wvr,wj,wo; Resonance: wq,wk,wv,wo
    for (int i = 0; i < BLK; i++)
        s += E + H*E*MT + gate_size + E*E*n_linear + E + FFN*E + FFN*E + E*FFN;
    s += E + V*E;
    return s;
}

static void assign_weights(Weights *w, float *p) {
    int V = CFG_V, E = CFG_E, H = CFG_H, BLK = CFG_BLK, FFN = CFG_FFN, MT = CFG_MT;
    w->tok_emb = p; p += V*E;
    w->pos_emb = p; p += MT*E;
    for (int i = 0; i < BLK; i++) {
        w->b[i].rms1 = p; p += E;
        w->b[i].wr = p;   p += H*E*MT;
        w->b[i].gate = p; p += CFG_HAS_JANUS ? H*3 : H*1;
        w->b[i].wq = p;   p += E*E;
        w->b[i].wk = p;   p += E*E;
        w->b[i].wv = p;   p += E*E;
        if (CFG_HAS_JANUS) {
            w->b[i].wvr = p;  p += E*E;
            w->b[i].wj = p;   p += E*E;
        } else {
            w->b[i].wvr = NULL;
            w->b[i].wj = NULL;
        }
        w->b[i].wo = p;   p += E*E;
        w->b[i].rms2 = p; p += E;
        w->b[i].wg = p;   p += FFN*E;
        w->b[i].wu = p;   p += FFN*E;
        w->b[i].wd = p;   p += E*FFN;
    }
    w->rms_f = p; p += E;
    w->head = p;
}

// ── BLAS matmul ──────────────────────────────────────────────────────────────

#ifdef USE_BLAS
  #ifdef ACCELERATE
    #include <Accelerate/Accelerate.h>
  #else
    #include <cblas.h>
  #endif
#endif

static void mm_t(float *C, const float *A, const float *B, int m, int k, int n) {
#ifdef USE_BLAS
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                m, n, k, 1.0f, A, k, B, k, 0.0f, C, n);
#else
    for (int i = 0; i < m; i++)
        for (int j = 0; j < n; j++) {
            float s = 0;
            for (int p = 0; p < k; p++) s += A[i*k+p] * B[j*k+p];
            C[i*n+j] = s;
        }
#endif
}

static void mm(float *C, const float *A, const float *B, int m, int k, int n) {
#ifdef USE_BLAS
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                m, n, k, 1.0f, A, k, B, n, 0.0f, C, n);
#else
    for (int i = 0; i < m; i++)
        for (int j = 0; j < n; j++) {
            float s = 0;
            for (int p = 0; p < k; p++) s += A[i*k+p] * B[p*n+j];
            C[i*n+j] = s;
        }
#endif
}

static void rmsnorm(float *out, const float *x, const float *w, int T, int dim) {
    for (int t = 0; t < T; t++) {
        float ss = 0;
        for (int i = 0; i < dim; i++) ss += x[t*dim+i] * x[t*dim+i];
        float inv = 1.0f / sqrtf(ss/dim + 1e-5f);
        for (int i = 0; i < dim; i++) out[t*dim+i] = w[i] * x[t*dim+i] * inv;
    }
}

static void softmax(float *x, int n) {
    float mx = x[0];
    for (int i = 1; i < n; i++) if (x[i] > mx) mx = x[i];
    float s = 0;
    for (int i = 0; i < n; i++) { x[i] = expf(x[i] - mx); s += x[i]; }
    for (int i = 0; i < n; i++) x[i] /= s;
}

static float siluf(float x) { return x > -20 ? x/(1+expf(-x)) : 0; }

// ── Forward pass ─────────────────────────────────────────────────────────────

static void forward(Weights *w, int *tok, int T, float *logits) {
    int V = CFG_V, E = CFG_E, H = CFG_H, D = CFG_D, BLK = CFG_BLK, FFN = CFG_FFN, MT = CFG_MT;
    float *x = (float*)calloc(T*E, sizeof(float));
    float *rn = (float*)calloc(T*E, sizeof(float));
    float sc = 1.0f / sqrtf((float)D);

    for (int t = 0; t < T; t++)
        for (int e = 0; e < E; e++)
            x[t*E+e] = w->tok_emb[tok[t]*E+e] + w->pos_emb[t*E+e];

    float *cat = (float*)calloc(T*E, sizeof(float));
    float *ao = (float*)calloc(T*E, sizeof(float));
    float *r1 = (float*)calloc(T*E, sizeof(float));
    float *mg = (float*)calloc(T*FFN, sizeof(float));
    float *mu = (float*)calloc(T*FFN, sizeof(float));
    float *mo = (float*)calloc(T*E, sizeof(float));

    for (int bl = 0; bl < BLK; bl++) {
        rmsnorm(rn, x, w->b[bl].rms1, T, E);

        float *qa = (float*)calloc(T*E, sizeof(float));
        float *ka = (float*)calloc(T*E, sizeof(float));
        float *va = (float*)calloc(T*E, sizeof(float));
        float *vra = NULL;
        mm_t(qa, rn, w->b[bl].wq, T, E, E);
        mm_t(ka, rn, w->b[bl].wk, T, E, E);
        mm_t(va, rn, w->b[bl].wv, T, E, E);
        if (CFG_HAS_JANUS) {
            vra = (float*)calloc(T*E, sizeof(float));
            mm_t(vra, rn, w->b[bl].wvr, T, E, E);
        }

        float *echo = NULL, *eback = NULL, *jsc = NULL, *jat = NULL;
        if (CFG_HAS_JANUS) {
            echo = (float*)calloc(T*E, sizeof(float));
            mm_t(echo, rn, w->b[bl].wj, T, E, E);
            eback = (float*)calloc(T*E, sizeof(float));
            mm(eback, echo, w->b[bl].wj, T, E, E);
            jsc = (float*)calloc(T, sizeof(float));
            for (int t = 0; t < T; t++) {
                float s = 0;
                for (int e = 0; e < E; e++) s += rn[t*E+e] * eback[t*E+e];
                jsc[t] = s / sqrtf((float)E);
            }
            jat = (float*)calloc(T*T, sizeof(float));
            for (int i = 0; i < T; i++) {
                for (int j = 0; j < T; j++)
                    jat[i*T+j] = (j > i) ? -1e9f : jsc[i] * jsc[j];
                softmax(jat + i*T, T);
            }
        }

        // Gates: Janus=[H,3] (QKV/RRPRAM/Janus), Resonance=[H,1] (QKV/RRPRAM blend)
        float gs[MAX_BLK][3];
        if (CFG_HAS_JANUS) {
            for (int h = 0; h < H; h++) {
                gs[h][0] = w->b[bl].gate[h*3+0];
                gs[h][1] = w->b[bl].gate[h*3+1];
                gs[h][2] = w->b[bl].gate[h*3+2];
                softmax(gs[h], 3);
            }
        } else {
            for (int h = 0; h < H; h++) {
                float g = 1.0f / (1.0f + expf(-w->b[bl].gate[h])); // sigmoid
                gs[h][0] = g;       // QKV weight
                gs[h][1] = 1.0f - g; // RRPRAM weight
                gs[h][2] = 0.0f;    // no Janus
            }
        }

        memset(cat, 0, T*E*sizeof(float));
        float *at = (float*)calloc(T*T, sizeof(float));
        float *ho = (float*)calloc(T*D, sizeof(float));

        for (int h = 0; h < H; h++) {
            float *q = (float*)calloc(T*D, sizeof(float));
            float *k = (float*)calloc(T*D, sizeof(float));
            float *v = (float*)calloc(T*D, sizeof(float));
            for (int t = 0; t < T; t++)
                for (int d = 0; d < D; d++) {
                    q[t*D+d] = qa[t*E + h*D + d];
                    k[t*D+d] = ka[t*E + h*D + d];
                    v[t*D+d] = va[t*E + h*D + d];
                }

            for (int i = 0; i < T; i++) {
                for (int j = 0; j < T; j++) {
                    if (j > i) { at[i*T+j] = -1e9f; continue; }
                    float s = 0;
                    for (int d = 0; d < D; d++) s += q[i*D+d] * k[j*D+d];
                    at[i*T+j] = s * sc;
                }
                softmax(at + i*T, T);
            }
            mm(ho, at, v, T, T, D);

            // RRPRAM
            float *wr_h = w->b[bl].wr + h*E*MT;
            float *rrp_sc = (float*)calloc(T, sizeof(float));
            for (int j = 0; j < T; j++) {
                float s = 0;
                for (int e = 0; e < E; e++) s += rn[j*E+e] * wr_h[e*MT+j];
                rrp_sc[j] = s * sc;
            }
            float *ra = (float*)calloc(T*T, sizeof(float));
            for (int i = 0; i < T; i++) {
                for (int j = 0; j < T; j++)
                    ra[i*T+j] = (j > i) ? -1e9f : rrp_sc[j];
                softmax(ra + i*T, T);
            }
            // RRPRAM values: use vra (Janus) or va (Resonance)
            float *rv_src = CFG_HAS_JANUS ? vra : va;
            float *rv = (float*)calloc(T*D, sizeof(float));
            for (int t = 0; t < T; t++)
                for (int d = 0; d < D; d++)
                    rv[t*D+d] = rv_src[t*E + h*D + d];
            float *ro = (float*)calloc(T*D, sizeof(float));
            mm(ro, ra, rv, T, T, D);

            // Janus attention (only if Janus mode)
            float *jo = NULL;
            if (CFG_HAS_JANUS) {
                float *jv = (float*)calloc(T*D, sizeof(float));
                for (int t = 0; t < T; t++)
                    for (int d = 0; d < D; d++)
                        jv[t*D+d] = echo[t*E + h*D + d];
                jo = (float*)calloc(T*D, sizeof(float));
                mm(jo, jat, jv, T, T, D);
                free(jv);
            }

            // Blend
            for (int t = 0; t < T; t++)
                for (int d = 0; d < D; d++) {
                    float val = gs[h][0]*ho[t*D+d] + gs[h][1]*ro[t*D+d];
                    if (jo) val += gs[h][2]*jo[t*D+d];
                    cat[t*E + h*D + d] = val;
                }
            free(q); free(k); free(v); free(rrp_sc);
            free(ra); free(rv); free(ro); free(jo);
        }

        mm_t(ao, cat, w->b[bl].wo, T, E, E);
        for (int i = 0; i < T*E; i++) r1[i] = x[i] + ao[i];

        rmsnorm(rn, r1, w->b[bl].rms2, T, E);
        mm_t(mg, rn, w->b[bl].wg, T, E, FFN);
        mm_t(mu, rn, w->b[bl].wu, T, E, FFN);
        for (int i = 0; i < T*FFN; i++) mg[i] = siluf(mg[i]) * mu[i];
        mm_t(mo, mg, w->b[bl].wd, T, FFN, E);
        for (int i = 0; i < T*E; i++) x[i] = r1[i] + mo[i];

        free(qa); free(ka); free(va); if (vra) free(vra);
        if (echo) free(echo); if (eback) free(eback);
        if (jsc) free(jsc); if (jat) free(jat);
        free(at); free(ho);
    }

    rmsnorm(rn, x, w->rms_f, T, E);
    mm_t(logits, rn, w->head, T, E, V);

    free(x); free(rn); free(cat); free(ao); free(r1);
    free(mg); free(mu); free(mo);
}

// ── Sampling ─────────────────────────────────────────────────────────────────

static int sample_top_p(float *logits, int n, float temp, float top_p) {
    for (int i = 0; i < n; i++) logits[i] /= temp;
    softmax(logits, n);
    // Greedy top-p: find cutoff
    float cum = 0;
    int best = 0;
    float best_p = -1;
    for (int i = 0; i < n; i++) if (logits[i] > best_p) { best_p = logits[i]; best = i; }
    // Simple sampling
    float r = (float)rand() / (float)RAND_MAX;
    cum = 0;
    for (int i = 0; i < n; i++) { cum += logits[i]; if (cum >= r) return i; }
    return best;
}

static double now_ms(void) {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec * 1000.0 + tv.tv_usec / 1000.0;
}

// ── Weight loading with format detection ─────────────────────────────────────

static float* load_weights(const char* path, int* n_params_out) {
    FILE *f = fopen(path, "rb");
    if (!f) { printf("cannot open %s\n", path); return NULL; }
    fseek(f, 0, SEEK_END);
    long fsize = ftell(f);
    fseek(f, 0, SEEK_SET);

    int32_t header[8];
    fread(header, sizeof(int32_t), 8, f);

    // Detect format
    if (header[0] == 0x4152494E || header[0] == 0x4E495241) {
        // ARIN / NIRA magic
        printf("format: ARIN (wrapped Janus BPE)\n");
        int np = header[1];
        // ARIN uses same architecture as BPE — detect from param count
        // 24,020,496 params = V=2048, E=384, H=4, D=96, BLK=12, FFN=768, MT=64
        CFG_V = 2048; CFG_E = 384; CFG_H = 4; CFG_D = 96;
        CFG_BLK = 12; CFG_FFN = 768; CFG_MT = 64;
        int expected = param_count();
        if (np != expected) {
            printf("ARIN param count %d != expected %d\n", np, expected);
            fclose(f); return NULL;
        }
        // Skip to offset 256 (ARIN has 256-byte header)
        fseek(f, 256, SEEK_SET);
        float *data = (float*)malloc(np * sizeof(float));
        fread(data, sizeof(float), np, f);
        fclose(f);
        *n_params_out = np;
        return data;
    }

    if (header[0] >= 1000 && header[0] <= 100000 &&
        header[1] >= 64 && header[1] <= 4096 &&
        header[2] >= 1 && header[2] <= 128) {
        // BPE config header: [V, E, H, D, BLK, FFN, MT]
        printf("format: BPE header\n");
        CFG_V = header[0];
        CFG_E = header[1];
        CFG_H = header[2];
        CFG_D = header[3];
        CFG_BLK = header[4];
        CFG_FFN = header[5];
        CFG_MT = header[6];
        int np = (int)((fsize - 7 * sizeof(int32_t)) / sizeof(float));
        // Try Janus first, then Resonance
        CFG_HAS_JANUS = 1;
        int expected = param_count();
        if (np != expected) {
            CFG_HAS_JANUS = 0;
            expected = param_count();
            if (np != expected) {
                printf("BPE param count %d != expected janus=%d or resonance=%d\n",
                       np, (CFG_HAS_JANUS=1, param_count()), expected);
                CFG_HAS_JANUS = 1; // reset
                fclose(f); return NULL;
            }
            printf("  → detected: resonance (no Janus echo)\n");
        } else {
            printf("  → detected: full Janus RRPRAM\n");
        }
        fseek(f, 7 * sizeof(int32_t), SEEK_SET);
        float *data = (float*)malloc(np * sizeof(float));
        fread(data, sizeof(float), np, f);
        fclose(f);
        *n_params_out = np;
        return data;
    }

    // DoE format: [int32 n_params][weights...]
    printf("format: DoE (flat)\n");
    int np = header[0];
    CFG_V = 256; CFG_E = 384; CFG_H = 4; CFG_D = 96;
    CFG_BLK = 12; CFG_FFN = 768; CFG_MT = 256;
    int expected = param_count();
    if (np != expected) {
        printf("DoE param count %d != expected %d\n", np, expected);
        fclose(f); return NULL;
    }
    fseek(f, 4, SEEK_SET);
    float *data = (float*)malloc(np * sizeof(float));
    fread(data, sizeof(float), np, f);
    fclose(f);
    *n_params_out = np;
    return data;
}

// ── Main ─────────────────────────────────────────────────────────────────────

int main(int argc, char **argv) {
    if (argc < 2) {
        printf("notorch inference — Janus RRPRAM (universal loader)\n");
        printf("usage: %s weights.bin [prompt] [max_tokens] [temp]\n", argv[0]);
        printf("\nSupports: DoE char (V=256), BPE header (V=2048), ARIN wrapped\n");
        return 1;
    }

    int np = 0;
    float *data = load_weights(argv[1], &np);
    if (!data) return 1;

    Weights w;
    assign_weights(&w, data);
    printf("config: V=%d E=%d H=%d D=%d BLK=%d FFN=%d MT=%d\n",
           CFG_V, CFG_E, CFG_H, CFG_D, CFG_BLK, CFG_FFN, CFG_MT);
    printf("loaded: %d params (%.1f MB)\n", np, np * 4.0f / 1048576.0f);

    // Loss verification
    if (CFG_V == 256) {
        FILE *df = fopen("/Users/ataeff/Downloads/janus-weights/leo_train.txt", "rb");
        if (!df) df = fopen("leo_train.txt", "rb");
        if (df) {
            fseek(df, 0, SEEK_END); long dsz = ftell(df); fseek(df, 0, SEEK_SET);
            unsigned char *dt = (unsigned char*)malloc(dsz);
            fread(dt, 1, dsz, df); fclose(df);
            int T = CFG_MT < 64 ? CFG_MT : 64;
            float loss_sum = 0; int n_win = 20;
            double t0 = now_ms();
            for (int wi = 0; wi < n_win; wi++) {
                int off = (int)((dsz - T - 1) * wi / n_win);
                int tok[256], tgt[256];
                for (int t = 0; t < T; t++) { tok[t] = dt[off+t]; tgt[t] = dt[off+t+1]; }
                float *lg = (float*)calloc(T * CFG_V, sizeof(float));
                forward(&w, tok, T, lg);
                float wloss = 0;
                for (int t = 0; t < T; t++) {
                    softmax(lg + t*CFG_V, CFG_V);
                    float p = lg[t*CFG_V + tgt[t]]; if (p < 1e-10f) p = 1e-10f;
                    wloss -= logf(p);
                }
                loss_sum += wloss / T;
                if (wi < 3) printf("  window %d: loss=%.4f\n", wi, wloss / T);
                free(lg);
            }
            printf("avg loss: %.4f (%d windows, %.0f ms)\n", loss_sum / n_win, n_win, now_ms() - t0);
            free(dt);
        }
    }

    // Generate
    const char *prompt = argc > 2 ? argv[2] : "Q: who are you?\nA: ";
    int max_tokens = argc > 3 ? atoi(argv[3]) : 200;
    float temp = argc > 4 ? atof(argv[4]) : 0.8f;
    int max_ctx = CFG_MT < 64 ? CFG_MT : 64;

    int ctx[512];
    int len = 0;
    if (CFG_V == 256) {
        // Char-level: each byte is a token
        for (int i = 0; prompt[i] && len < max_ctx; i++)
            ctx[len++] = (unsigned char)prompt[i];
    } else {
        // BPE: need tokenizer. For now, use byte-fallback
        printf("(BPE vocab=%d, using byte-fallback tokenizer)\n", CFG_V);
        for (int i = 0; prompt[i] && len < max_ctx; i++)
            ctx[len++] = (unsigned char)prompt[i] % CFG_V;
        if (len == 0) { ctx[len++] = 1; } // BOS fallback
    }

    printf("\n── generation (temp=%.2f) ──\n%s", temp, prompt);
    double gen_start = now_ms();
    for (int step = 0; step < max_tokens; step++) {
        int T = len < max_ctx ? len : max_ctx;
        int *tok = ctx + (len > max_ctx ? len - max_ctx : 0);
        float *lg = (float*)calloc(T * CFG_V, sizeof(float));
        forward(&w, tok, T, lg);
        float *last = lg + (T - 1) * CFG_V;
        int next = sample_top_p(last, CFG_V, temp, 0.9f);
        if (CFG_V == 256) printf("%c", (char)next);
        else printf("[%d]", next);
        fflush(stdout);
        if (len < 512) ctx[len++] = next;
        free(lg);
    }
    double gen_elapsed = now_ms() - gen_start;
    printf("\n── %.0f ms, %.1f tok/s ──\n", gen_elapsed, max_tokens * 1000.0 / gen_elapsed);

    free(data);
    return 0;
}
