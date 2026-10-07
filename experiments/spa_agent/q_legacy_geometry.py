#!/usr/bin/env python3
"""Probe the pinned Q C SPA geometry without modifying or importing Q."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shlex
import subprocess
import tempfile

SOURCE_SHA = "820287691af6bb6fde41d5e0a24d107574b5dcd0e35eb3834bdc39027ae4a135"
COMMIT = "f5d00a36ecfcdb5e655e1576770f03d06d900e04"
PROBE = r'''
static unsigned rng = 6138;
static float draw(void) {
    rng ^= rng << 13; rng ^= rng >> 17; rng ^= rng << 5;
    return (float)(rng & 65535u) / 32767.5f - 1.0f;
}
int main(void) {
    SPACtx *s = calloc(1, sizeof(*s));
    if (!s) return 2;
    spa_init(s, 1);
    float emb[12][32], scores[12], minimum = 2;
    unsigned triggers = 0;
    for (int trial = 0; trial < 4100; trial++) {
        memset(emb, 0, sizeof emb);
        for (int i = 0; i < 12; i++) {
            if (trial == 0) continue;
            if (trial == 1) { emb[i][0] = 1; continue; }
            if (trial == 2) { emb[i][0] = i == 0 ? -1 : 1; continue; }
            if (trial == 3) { emb[i][0] = i < 6 ? -1 : 1; continue; }
            float squared = 1e-8f;
            for (int d = 0; d < 32; d++) {
                emb[i][d] = draw(); squared += emb[i][d] * emb[i][d];
            }
            float inverse = 1.0f / sqrtf(squared);
            for (int d = 0; d < 32; d++) emb[i][d] *= inverse;
        }
        spa_cross_attend(s, emb, 12, scores);
        float sum = 0, low = scores[0];
        for (int i = 0; i < 12; i++) { sum += scores[i]; if (scores[i] < low) low = scores[i]; }
        float mean = sum / 12.0f, ratio = low / mean;
        if (ratio < minimum) minimum = ratio;
        for (int phase = 0; phase <= 2; phase++)
            if (low < mean * (.52f + .18f * (1.0f - .5f * phase))) triggers++;
    }
    free(s);
    printf("{\"fields\":4100,\"phase_values\":[0,0.5,1],\"triggers\":%u,"
           "\"minimum_observed_ratio\":%.9g}\n", triggers, minimum);
    return triggers ? 1 : 0;
}
'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("q_source", type=Path, help="pinned q/postgpt_q.c")
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()
    source = args.q_source.read_bytes()
    if hashlib.sha256(source).hexdigest() != SOURCE_SHA:
        raise SystemExit("refused: Q source differs from the registered commit")
    text = source.decode()
    begin = text.index("#define SPA_DIM  32")
    end = text.index("/* \u2500\u2500 chain \u2500\u2500 */", begin)
    unit = "#include <math.h>\n#include <stdio.h>\n#include <stdlib.h>\n#include <string.h>\n"
    unit += "#define MAX_VOCAB 1280\n#define CHAIN_STEPS 12\n" + text[begin:end] + PROBE
    compiler = shlex.split(os.environ.get("CC", "cc"))
    flags = ["-O2", "-std=c11"]
    with tempfile.TemporaryDirectory(prefix="spa-q-geometry-") as directory:
        temporary = Path(directory)
        (temporary / "probe.c").write_text(unit)
        subprocess.run(compiler + flags + [str(temporary / "probe.c"), "-lm", "-o", str(temporary / "probe")], check=True)
        run = subprocess.run([str(temporary / "probe")], check=True, text=True, capture_output=True)
    minimum_edge = math.exp(-1 / math.sqrt(32) + 0.1 / 12)
    maximum_edge = math.exp(1 / math.sqrt(32) + 0.1 / 2)
    lower = 12 * 11 * minimum_edge / (2 * 11 * minimum_edge + 110 * maximum_edge)
    report = {
        "source": {"repository": "ariannamethod/q", "commit": COMMIT, "path": "postgpt_q.c", "sha256": SOURCE_SHA},
        "method": "Extract the exact pinned spa_cross_attend implementation into a temporary probe; no Q source edits.",
        "compiler": subprocess.check_output(compiler + ["--version"], text=True).splitlines()[0],
        "flags": flags,
        "probe_sha256": hashlib.sha256(unit.encode()).hexdigest(),
        "numeric_probe": json.loads(run.stdout),
        "analytical_bound": {
            "assumptions": "12 finite sentence vectors of Euclidean norm at most 1, dimension 32, distance bias 0.1/(1+distance).",
            "edge_minimum": minimum_edge, "edge_maximum": maximum_edge,
            "formula": "12*11*m/(2*11*m+110*M); symmetry counts the selected row twice in total degree.",
            "minimum_score_to_mean_lower_bound": lower,
            "maximum_trigger_ratio": 0.70,
            "conclusion": "The registered 12-sentence Q geometry stays above the reseed trigger. Synthetic low-score fixtures exercise the decision branch separately."
        }
    }
    assert lower > .70
    if args.json:
        args.json.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
