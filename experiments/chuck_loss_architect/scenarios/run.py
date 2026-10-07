#!/usr/bin/env python3
"""Fixed common-state HOLD/BRAKE/PUSH comparisons and restoration gates."""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
spec = importlib.util.spec_from_file_location("chuck_v1_receipts", HERE.parent / "run.py")
v1 = importlib.util.module_from_spec(spec)
assert spec and spec.loader
spec.loader.exec_module(v1)


def strict_json(text: str):
    def invalid(value):
        raise ValueError("non-JSON numeric value: " + value)
    return json.loads(text, parse_constant=invalid)


def events(path: Path) -> list[dict]:
    return [strict_json(line) for line in path.read_text().splitlines()]


def build(binary: Path, source: Path | None = None) -> list[str]:
    command = ["cc", "-O2", "-std=gnu11", "-DUSE_SIMD", "-march=native", "-pthread", "-I", str(ROOT),
               str(source or ROOT / "examples/chuck_architect_train.c"), str(ROOT / "notorch.c"), "-lm", "-o", str(binary)]
    subprocess.run(command, check=True, cwd=ROOT)
    return command


def execute(command: list[str], log: Path, env: dict) -> int:
    with log.open("w") as output:
        result = subprocess.run(command, cwd=ROOT, env=env, stdout=output, stderr=subprocess.STDOUT)
    return result.returncode


def verification(output: Path, env: dict, token_file: Path, config: Path, valid_binary: Path) -> dict:
    directory = output / "verification"
    directory.mkdir(exist_ok=True)
    original = (ROOT / "examples/chuck_architect_train.c").read_text()
    original_helper = (ROOT / "examples/chuck_architect_scenarios.h").read_text()
    result = {}
    short_tokens = directory / "short.tokens.u32"
    short_tokens.write_bytes(bytes(640 * 4))
    short_prefix = directory / "short-corpus"
    short_exit = execute([str(valid_binary), "--scenarios", "simple", str(short_tokens), str(short_prefix),
                          "4", "42", "0.0003", str(config), "0"], directory / "short-corpus.log", env)
    assert short_exit != 0 and "corpus split too small" in (directory / "short-corpus.log").read_text()
    assert not Path(str(short_prefix) + ".jsonl").exists()
    result["short_corpus_refusal"] = {"tokens": 640, "exit": short_exit, "status": "PASS",
                                      "log_sha256": v1.digest(directory / "short-corpus.log")}
    # Restoring a snapshot must restore its body, not keep the last branch's body.
    restore = directory / "restore-mutant"; restore.mkdir(exist_ok=True)
    source = restore / "runner.c"; source.write_text(original)
    needle = "memcpy(m->param[i]->data, s->parameters[i]->data, (size_t)m->param[i]->len * sizeof(float));"
    assert original_helper.count(needle) == 1
    helper = restore / "chuck_architect_scenarios.h"
    helper.write_text(original_helper.replace(needle, "(void)i; /* deliberate lost body restore */"))
    binary = restore / "runner"; build(binary, source)
    prefix = restore / "receipt"
    exit_code = execute([str(binary), "--scenarios", "simple", str(token_file), str(prefix), "4", "42", "0.0003", str(config), "1"],
                        restore / "run.log", env)
    log = (restore / "run.log").read_text()
    assert exit_code != 0 and "snapshot_restore_exact failed" in log, "restore mutation was not caught"
    result["restore_mutation"] = {"defect": "parameter_restoration_suppressed", "exit": exit_code,
                                  "caught": True, "mutant_sha256": v1.digest(helper), "log_sha256": v1.digest(restore / "run.log")}
    # An actual post-action NaN must become a credited negative consequence.
    failure = directory / "nonfinite-mutant"; failure.mkdir(exist_ok=True)
    needle = "int after_idx = body_forward(&m, data, offset, 0);"
    assert original.count(needle) == 1
    source = failure / "runner.c"
    source.write_text(original.replace(needle, "m.param[m.count - 1]->data[0] = NAN; /* deliberate post-action divergence */\n        " + needle))
    (failure / "chuck_architect_scenarios.h").write_text(original_helper)
    binary = failure / "runner"; build(binary, source)
    prefix = failure / "receipt"
    exit_code = execute([str(binary), "simple", "learned", str(token_file), str(prefix), "4", "42", "0.0003", str(config)],
                        failure / "run.log", env)
    rows = events(failure / "receipt.jsonl")
    assert exit_code == 3 and [row["type"] for row in rows] == ["run", "evaluation", "step", "failure"]
    step = rows[2]; credit = step["architect"]["consequence"]
    assert step["step"] == 1 and step["loss_after_same_window"] is None and step["improvement"] is None
    assert credit["after_loss"] is None and credit["loss_delta"] is None
    assert credit["reward"] == -1 and credit["learned"] == 1 and credit["nonfinite"] == 1
    assert rows[3]["stopped"] is True and rows[3]["learned_feedback_recorded"] is True
    for suffix in (".final.bin", ".moments.final.bin", ".optimizer.final.json", ".policy.final.bin"):
        assert Path(str(prefix) + suffix).is_file(), suffix
    inspector_source = failure / "inspect.c"
    inspector_source.write_text('''#include "chuck_architect.h"
#include <stdio.h>
#include <inttypes.h>
int main(int argc,char **argv) {
    nt_chuck_architect a;
    if(argc!=2 || nt_chuck_architect_load(&a,argv[1])) return 1;
    printf("{\\"decisions\\":%" PRIu64 ",\\"updates\\":%" PRIu64 ",\\"pending\\":%d,\\"hash\\":\\"%016" PRIx64 "\\"}\\n",
           a.decisions,a.updates,a.pending,nt_chuck_architect_hash(&a));
    return 0;
}
''')
    inspector = failure / "inspect"; build(inspector, inspector_source)
    state = strict_json(subprocess.check_output([str(inspector), str(prefix) + ".policy.final.bin"], text=True, env=env))
    assert state == {"decisions": 1, "updates": 1, "pending": 0, "hash": step["architect"]["policy_post"]}
    result["nonfinite_consequence"] = {"exit": exit_code, "caught": True, "policy_state": state,
                                       "mutant_sha256": v1.digest(source), "receipts_sha256": v1.digest(failure / "receipt.jsonl"),
                                       "policy_sha256": v1.digest(failure / "receipt.policy.final.bin")}
    v1.write_json(directory / "gates.json", result)
    return result


def compare_hosts(control: list[dict], diagnostic: list[dict], a: Path, b: Path, steps: int) -> dict:
    left = [row for row in control if row["type"] == "host_step"]
    right = [row for row in diagnostic if row["type"] == "host_step"]
    assert len(left) == len(right) == steps
    for x, y in zip(left, right):
        assert x == y, f"host continuation differs at step {x['step']}"
    identities = {}
    for suffix in (".final.bin", ".moments.final.bin", ".optimizer.final.json", ".policy.final.bin"):
        pa, pb = Path(str(a) + suffix), Path(str(b) + suffix)
        assert pa.read_bytes() == pb.read_bytes(), "final host state differs: " + suffix
        identities[suffix] = v1.digest(pa)
    assert control[-1]["final_heldout"] == diagnostic[-1]["final_heldout"]
    return {"status": "PASS", "host_steps_compared": steps, "final_state_sha256": identities}


def comparisons(rows: list[dict], body: str, seed: int, protocol: dict, steps: int) -> list[dict]:
    expected = [c for c in protocol["checkpoints_before_updates"] if c <= steps]
    forks = {r["step"]: r for r in rows if r["type"] == "fork"}
    restores = {r["step"]: r for r in rows if r["type"] == "restore"}
    assert sorted(forks) == sorted(restores) == expected
    actions = protocol["actions"]
    result = []
    for checkpoint in expected:
        assert restores[checkpoint]["exact"] is True
        assert restores[checkpoint]["state_hash"] == forks[checkpoint]["state_hash"]
        trajectory = [r for r in rows if r["type"] == "branch_step" and r["checkpoint"] == checkpoint]
        for action in actions:
            branch = [r for r in trajectory if r["intervention"] == action]
            assert len(branch) == max(protocol["horizons_total_updates"])
            for i, row in enumerate(branch):
                assert row["update"] == i + 1 and row["offset"] == forks[checkpoint]["offsets"][i]
                assert row["executed_action"] == (action if i == 0 else "hold")
        for horizon in protocol["horizons_total_updates"]:
            arms = {r["action"]: r for r in rows if r["type"] == "comparison" and
                    r["checkpoint"] == checkpoint and r["horizon"] == horizon}
            assert set(arms) == set(actions)
            assert {r["initial_hash"] for r in arms.values()} == {forks[checkpoint]["state_hash"]}
            for field in ("immediate_before", "future_before", "heldout_before"):
                assert len({r[field] for r in arms.values()}) == 1
            losses = ("immediate_after", "origin_after_horizon", "future_after", "heldout_after")
            winners = {field: [action for action in actions if arms[action][field] == min(r[field] for r in arms.values())]
                       for field in losses}
            advantage = {action: {field: arms["hold"][field] - arms[action][field] for field in losses}
                         for action in actions}
            result.append({"body": body, "seed": seed, "checkpoint": checkpoint, "horizon": horizon,
                           "winners": winners, "advantage_vs_hold": advantage, "arms": arms})
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--references", type=Path)
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--allow-dirty", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    output = args.output.resolve()
    if output == ROOT or ROOT in output.parents: parser.error("output must be outside Git")
    output.mkdir(parents=True, exist_ok=True)
    if (output / "results.json").exists(): parser.error("results already exist; choose a new output directory")
    protocol = strict_json((HERE / "protocol.json").read_text())
    source_status = v1.command(["git", "status", "--porcelain"])
    if source_status and not args.allow_dirty: parser.error("final runs require clean committed source")
    if args.smoke and not args.allow_dirty: parser.error("smoke is explicitly a development mode; add --allow-dirty")
    bodies = ["simple"] if args.smoke else protocol["bodies"]
    seeds = [42] if args.smoke else protocol["seeds"]
    steps = 8 if args.smoke else protocol["host_steps"]
    datasets = {body: v1.prepare(body, output, args.references) for body in bodies}
    v1.write_json(output / "datasets.json", datasets)
    sources = ["notorch.c", "notorch.h", "notorch_simd.h", "chuck_architect.h", "chuck_architect_impl.h",
               "examples/chuck_architect_train.c", "examples/chuck_architect_scenarios.h",
               "examples/chuck-loss-architect.json", "experiments/chuck_loss_architect/run.py",
               "experiments/chuck_loss_architect/scenarios/run.py", "experiments/chuck_loss_architect/scenarios/protocol.json"]
    hashes = {name: v1.digest(ROOT / name) for name in sources}
    binary = output / "scenario_runner"; build(binary)
    config = ROOT / "examples/chuck-loss-architect.json"
    env = dict(os.environ, NT_SIMD_THREADS=str(protocol["threads"]), OMP_NUM_THREADS=str(protocol["threads"]),
               OPENBLAS_NUM_THREADS=str(protocol["threads"]))
    manifest = {"source_commit": v1.command(["git", "rev-parse", "HEAD"]), "source_status": source_status,
                "source_sha256": hashes, "binary_sha256": v1.digest(binary), "protocol": protocol,
                "smoke": args.smoke, "executed_bodies": bodies, "executed_seeds": seeds, "executed_host_steps": steps,
                "machine": v1.machine(), "compiler": v1.command(["cc", "--version"]).splitlines()[0],
                "compile_flags": "-O2 -std=gnu11 -DUSE_SIMD -march=native -pthread", "datasets": datasets}
    v1.write_json(output / "manifest.json", manifest)
    shutil.copyfile(HERE / "protocol.json", output / "protocol.json")
    shutil.copyfile(config, output / "architect-config.json")
    gates = verification(output, env, output / "simple.tokens.u32", config, binary) if args.verify else {}
    runs, grouped = [], []
    for body in bodies:
        for seed in seeds:
            traces, prefixes = {}, {}
            for probes, label in ((0, "control"), (1, "diagnostic")):
                name = f"{body}-s{seed}-{label}"; prefix = output / name
                invocation = [str(binary), "--scenarios", body, str(output / f"{body}.tokens.u32"), str(prefix),
                              str(steps), str(seed), str(protocol["lr"]), str(config), str(probes)]
                print("RUN", name, flush=True)
                exit_code = execute(invocation, output / (name + ".log"), env)
                if exit_code: raise RuntimeError(f"{name} failed with exit{exit_code}; see its saved log")
                traces[label] = events(output / (name + ".jsonl")); prefixes[label] = prefix
            parity = compare_hosts(traces["control"], traces["diagnostic"], prefixes["control"], prefixes["diagnostic"], steps)
            groups = comparisons(traces["diagnostic"], body, seed, protocol, steps)
            grouped.extend(groups)
            row = {"body": body, "seed": seed, "parity": parity, "comparison_groups": len(groups),
                   "control_summary": traces["control"][-1], "diagnostic_summary": traces["diagnostic"][-1],
                   "control_receipts_sha256": v1.digest(Path(str(prefixes["control"]) + ".jsonl")),
                   "diagnostic_receipts_sha256": v1.digest(Path(str(prefixes["diagnostic"]) + ".jsonl"))}
            runs.append(row); print(json.dumps(row), flush=True)
    for name, digest in hashes.items():
        if v1.digest(ROOT / name) != digest: raise RuntimeError("source changed during experiment: " + name)
    artifacts = {str(p.relative_to(output)): {"bytes": p.stat().st_size, "sha256": v1.digest(p)} for p in sorted(output.rglob("*"))
                 if p.is_file() and p.suffix in (".bin", ".json", ".jsonl", ".log", ".u32", ".corpus", ".c", ".h")}
    result = {"manifest": manifest, "verification": gates, "runs": runs, "comparisons": grouped,
              "source_hashes_rechecked": True, "artifacts": artifacts}
    v1.write_json(output / "results.json", result)
    print(f"SCENARIOS_OK cohorts={len(runs)} comparison_groups={len(grouped)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
