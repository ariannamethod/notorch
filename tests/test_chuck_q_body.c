/* Q raw-logit oracle: ariannamethod/q@f5d00a36ecfcdb5e655e1576770f03d06d900e04
 * postgpt_q.c tf_load/tf_forward, compiled as C. The inference-only magnitude
 * multiplier was omitted to compare the training substrate's raw logits.
 * Inputs: nt_seed(42), fixture V256/D8/L1/NR1/NJ1/HD4/CTX8, tokens17*i+3.
 * Case1 scales wj by1e-5 to expose the native Janus epsilon1e-8. */
#define NT_CHUCK_Q_TRAINER
#define NT_CHUCK_ARCHITECT_TEST
#define main q_trainer_example_main
#include "../examples/chuck_architect_train.c"
#undef main

static const float native_logits[2][32] = {
    {
        0x1.660accp-8f,0x1.167074p-9f,-0x1.9b61p-6f,0x1.0361cp-6f,
        0x1.7ddafp-7f,0x1.7e9a24p-5f,-0x1.36aecap-5f,0x1.2f714ep-7f,
        -0x1.221f66p-5f,-0x1.e3e86p-6f,-0x1.3dc18p-11f,-0x1.b9b676p-7f,
        -0x1.0c52eep-5f,-0x1.8d3724p-6f,-0x1.f96598p-8f,-0x1.d440aap-7f,
        0x1.5d8adcp-7f,0x1.b97c84p-5f,-0x1.97b0f8p-5f,0x1.bd5f14p-7f,
        -0x1.923f4cp-5f,-0x1.2f8732p-5f,0x1.25d0ap-10f,-0x1.8e26cep-6f,
        -0x1.1c621cp-4f,-0x1.49627p-5f,-0x1.ee339p-8f,-0x1.12a4ecp-5f,
        0x1.5fe8ap-8f,0x1.586718p-5f,-0x1.a9b7ecp-6f,0x1.c1dd0ap-9f
    },
    {
        -0x1.3034cep-8f,0x1.634c4p-11f,-0x1.fb3e98p-6f,0x1.6b6c5cp-7f,
        -0x1.0046aep-5f,0x1.7bfb2ap-8f,-0x1.79bcccp-5f,-0x1.235b82p-9f,
        -0x1.f9c64p-5f,-0x1.dfbbfp-6f,-0x1.119f12p-6f,-0x1.4e59dap-6f,
        -0x1.35525p-4f,-0x1.8b6d3p-7f,-0x1.7e7312p-6f,-0x1.fb0718p-6f,
        -0x1.2bb978p-5f,0x1.065b3p-6f,-0x1.034234p-4f,-0x1.4e7c98p-9f,
        -0x1.13bd56p-4f,-0x1.20c54p-5f,-0x1.a97f2p-7f,-0x1.128d14p-5f,
        -0x1.584fd4p-4f,-0x1.21ae9p-5f,-0x1.b537a4p-7f,-0x1.6f90bp-5f,
        -0x1.b49556p-10f,0x1.435464p-5f,-0x1.dacb6p-6f,0x1.0302aap-8f
    }
};

/* Double reduction avoids subtracting two rounded float loss scalars when
 * checking small gradients. Model forward arithmetic remains on the C tape. */
static double precise_loss(body *m, const uint32_t *ids) {
    int loss = body_forward(m, ids, 0, 0);
    nt_tape *tape = nt_tape_get();
    const float *x = tape->entries[tape->entries[loss].parent1].output->data;
    double total = 0;
    for (int row = 0; row < CTX; ++row) {
        double maximum = x[row * m->vocab], sum = 0;
        for (int v = 1; v < m->vocab; ++v)
            if (x[row * m->vocab + v] > maximum) maximum = x[row * m->vocab + v];
        for (int v = 0; v < m->vocab; ++v) sum += exp((double)x[row * m->vocab + v] - maximum);
        total += log(sum) + maximum - x[row * m->vocab + ids[row + 1]];
    }
    return total / CTX;
}

int main(void) {
    nt_seed(42);
    body m; body_init_vocab(&m, "q", 256);
    uint32_t ids[CTX + 1];
    for (int i = 0; i <= CTX; ++i) ids[i] = (uint32_t)(17 * i + 3) % 256;
    int failed = 0;
    for (int slot = 0; slot < m.count; ++slot) {
        int loss = body_forward(&m, ids, 0, 1);
        nt_tape_backward(loss);
        int k = m.param[slot]->len / 3;
        float analytic = nt_tape_get()->entries[slot].grad->data[k];
        float original = m.param[slot]->data[k], epsilon = slot == 1 ? 1e-5f : 1e-4f;
        /* Smaller positional perturbation stays inside this fixture's ReLU region. */
        m.param[slot]->data[k] = original + epsilon;
        double plus = precise_loss(&m, ids);
        m.param[slot]->data[k] = original - epsilon;
        double minus = precise_loss(&m, ids);
        m.param[slot]->data[k] = original;
        double numerical = (plus - minus) / (2 * epsilon);
        if (!isfinite(analytic) || fabs(numerical - analytic) > 1e-4 + .02 * fabs(analytic)) {
            fprintf(stderr, "FAIL Q gradient slot%d: analytic%.9g numerical%.9g\n", slot, analytic, numerical);
            ++failed;
        }
    }
    const int columns[] = {0, 3, 83, 255};
    double worst = 0;
    for (int small = 0; small < 2; ++small) {
        if (small) for (int k = 0; k < m.param[4]->len; ++k) m.param[4]->data[k] *= 1e-5f;
        int loss = body_forward(&m, ids, 0, 0);
        nt_tape *tape = nt_tape_get();
        const float *actual = tape->entries[tape->entries[loss].parent1].output->data;
        for (int row = 0; row < CTX; ++row) for (int c = 0; c < 4; ++c) {
            double error = fabs(actual[row * m.vocab + columns[c]] - native_logits[small][row * 4 + c]);
            if (error > worst) worst = error;
            if (!isfinite(error) || error > 2e-6) ++failed;
        }
    }
    nt_tape_destroy();
    for (int i = 0; i < m.count; ++i) nt_tensor_free(m.param[i]);
    printf("Q native logits: 64 checks, max error%.9g; 10 tensor gradients; failures%d\n", worst, failed);
    return failed ? 1 : 0;
}
