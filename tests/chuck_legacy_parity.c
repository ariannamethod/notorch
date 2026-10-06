// Canonical Chuck trajectory writer: consumed by test_chuck_legacy_parity.sh.
// Writes each global/local state, weight, moment and timestep as binary.
#include "notorch.h"
#include <stdio.h>

int main(void) {
    nt_tape_start();
    nt_tensor *params[3];
    for (int slot = 0; slot < 3; slot++) {
        params[slot] = nt_tensor_new(5);
        for (int j = 0; j < 5; j++)
            params[slot]->data[j] = (float)(slot + 1) * (j + 1) * 0.17f;
        int index = nt_tape_param(params[slot]);
        nt_tape_get()->entries[index].grad = nt_tensor_new(5);
    }
    int noise_seen = 0, freeze_seen = 0, macro_decay_seen = 0;
    for (int step = 0; step < 6000; step++) {
        int phase = step % 200;
        float loss = phase < 80 ? 1.5f :
                     phase < 140 ? 1.5f + (phase - 80) * 0.09f : 0.3f;
        for (int slot = 0; slot < 3; slot++) {
            for (int j = 0; j < 5; j++) {
                nt_tape_get()->entries[slot].grad->data[j] =
                    slot == 2 && step > 16 ? 0.0001f :
                    (slot + 1) * (j + 1) * (0.02f + 0.005f * (step % 11));
            }
        }
        nt_tape_chuck_step(0.001f, loss);
        nt_tape *tape = nt_tape_get();
        noise_seen |= tape->chuck.noise > 0.0f;
        freeze_seen |= tape->chuck_params[2].frozen;
        macro_decay_seen |= tape->chuck.lr_scale < 1.0f;
        fwrite(&tape->chuck, sizeof(tape->chuck), 1, stdout);
        fwrite(tape->chuck_params, sizeof(tape->chuck_params[0]), 3, stdout);
        for (int slot = 0; slot < 3; slot++) {
            fwrite(params[slot]->data, sizeof(float), 5, stdout);
            fwrite(tape->adam[slot].m->data, sizeof(float), 5, stdout);
            fwrite(tape->adam[slot].v->data, sizeof(float), 5, stdout);
            fwrite(&tape->adam[slot].t, sizeof(int), 1, stdout);
        }
    }
    nt_tape_destroy();
    for (int slot = 0; slot < 3; slot++) nt_tensor_free(params[slot]);
    if (!noise_seen || !freeze_seen || !macro_decay_seen) {
        fprintf(stderr, "legacy fixture missed noise=%d freeze=%d macro_decay=%d\n",
                noise_seen, freeze_seen, macro_decay_seen);
        return 1;
    }
    return ferror(stdout) ? 1 : 0;
}
