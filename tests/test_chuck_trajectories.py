#!/usr/bin/env python3
"""Check source-state retention and named continuation weights on a real body.

The synthetic lives have different history-sensitive action readouts. A third
saved life shares the source's weights but has a different config and RNG, so
aliasing must inspect weights and grafting must preserve the visited source.
The isolated wrong-weight mutation must reach the named native invariant.
"""
from __future__ import annotations

import argparse
import atexit
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shlex
import struct
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[1]
NUMERICAL = ("notorch.c", "notorch.h", "notorch_simd.h", "chuck_architect.h", "chuck_architect_impl.h",
             "examples/chuck_architect_train.c", "examples/chuck_architect_scenarios.h",
             "examples/chuck_architect_rollout.h", "examples/chuck_architect_lived.h")
FIXTURE = r'''
#include "chuck_architect.h"
#include <inttypes.h>
#include <stdio.h>
#include <string.h>
static void weights(nt_chuck_architect *to, const nt_chuck_architect *from) {
    memcpy(to->w1, from->w1, sizeof(to->w1)); memcpy(to->b1, from->b1, sizeof(to->b1));
    memcpy(to->w2, from->w2, sizeof(to->w2)); memcpy(to->b2, from->b2, sizeof(to->b2));
}
static int equal_weights(const nt_chuck_architect *a, const nt_chuck_architect *b) {
    return !memcmp(a->w1,b->w1,sizeof(a->w1)) && !memcmp(a->b1,b->b1,sizeof(a->b1)) &&
           !memcmp(a->w2,b->w2,sizeof(a->w2)) && !memcmp(a->b2,b->b2,sizeof(a->b2));
}
int main(int argc, char **argv) {
    if (argc != 5) return 2;
    nt_chuck_architect a, b, c;
    if (!strcmp(argv[1], "emit")) {
        nt_chuck_architect_config cfg;
        if (nt_chuck_architect_config_parse_json(&cfg, "{\"mode\":\"learned\",\"exploration\":0}", NULL, 0) ||
            nt_chuck_architect_init(&a, &cfg)) return 3;
        memset(a.w1, 0, sizeof a.w1); memset(a.b1, 0, sizeof a.b1);
        memset(a.w2, 0, sizeof a.w2); memset(a.b2, 0, sizeof a.b2);
        a.w1[0][15] = 16; a.w2[1][0] = -1; a.w2[2][0] = 1;
        a.b2[0] = -.2f; a.b2[1] = .01f; a.b2[2] = -.01f;
        cfg.seed = 991; cfg.learning_rate = .125f; strcpy(cfg.life_id, "continuation_fixture");
        if (nt_chuck_architect_init(&b, &cfg)) return 4;
        weights(&b, &a);
        b.w2[1][0] = 1; b.w2[2][0] = -1; b.b2[1] = -.01f; b.b2[2] = .01f;
        cfg.seed = 997; strcpy(cfg.life_id, "alias_fixture");
        if (nt_chuck_architect_init(&c, &cfg)) return 5;
        weights(&c, &a);
        if (nt_chuck_architect_save(&a, argv[2]) || nt_chuck_architect_save(&b, argv[3]) ||
            nt_chuck_architect_save(&c, argv[4])) return 6;
    } else if (!strcmp(argv[1], "graft")) {
        if (nt_chuck_architect_load(&a, argv[2]) || nt_chuck_architect_load(&b, argv[3]) ||
            nt_chuck_architect_load(&c, argv[4])) return 7;
        if (!equal_weights(&b, &c)) { fputs("saved graft has wrong continuation weights\n", stderr); return 8; }
        nt_chuck_architect restored = b; weights(&restored, &a);
        if (memcmp(&restored, &a, sizeof a)) { fputs("saved graft changed non-weight state\n", stderr); return 9; }
    } else return 2;
    printf("{\"source\":\"%016" PRIx64 "\",\"student\":\"%016" PRIx64 "\",\"continuation\":\"%016" PRIx64 "\","
           "\"decisions\":%" PRIu64 ",\"updates\":%" PRIu64 ",\"has_history\":%d,\"rng\":%" PRIu32 "}\n",
           nt_chuck_architect_hash(&a), nt_chuck_architect_hash(&b), nt_chuck_architect_hash(&c),
           a.decisions, a.updates, a.has_history, a.rng);
    return 0;
}
'''


def digest(data):
    return hashlib.sha256(data).hexdigest()


def invoke(command, env):
    result = subprocess.run(list(map(str, command)), cwd=ROOT, env=env, capture_output=True,
                            text=True, timeout=240)
    return {"command": list(map(str, command)), "returncode": result.returncode,
            "stdout": result.stdout, "stderr": result.stderr}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", type=Path)
    parser.add_argument("--old-host", type=Path, help="optional pre-change host for exact lived compatibility")
    args = parser.parse_args()
    originals = {name: (ROOT / name).read_bytes() for name in NUMERICAL}
    env = dict(os.environ, NT_SIMD_THREADS="2", OMP_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2")
    compiler = shlex.split(os.environ.get("CC", "cc"))
    report = {"source_sha256": {name: digest(data) for name, data in originals.items()},
              "fixture_source_sha256": digest(FIXTURE.encode()), "passed": False,
              "fixture": {"body": "simple", "parameters": 450688, "steps": 32, "seed": 42,
                          "corpus": "6400 synthetic tokens; token[i] = i mod 23",
                          "purpose": "source history retention, named continuation weights, alias noninterference"}}

    def save_report():
        if args.json:
            args.json.parent.mkdir(parents=True, exist_ok=True)
            args.json.write_text(json.dumps(report, indent=2) + "\n")
    atexit.register(save_report)

    spec = importlib.util.spec_from_file_location("trajectory_gate_lived", ROOT / "experiments/chuck_loss_architect/lived/run.py")
    lived = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(lived)
    legacy_protocol = json.loads((ROOT / "experiments/chuck_loss_architect/lived/protocol.json").read_text())
    policy_protocol = dict(legacy_protocol, checkpoints_before_updates=[1, 17, 33, 49, 65, 97, 129, 193], continuations=["policy"])

    with tempfile.TemporaryDirectory(prefix="notorch-trajectory-gates-") as temporary:
        temp = Path(temporary)
        source = temp / "source"
        for name, data in originals.items():
            path = source / name; path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes(data)
        fixture = source / "fixture.c"; fixture.write_text(FIXTURE)
        base_command = compiler + ["-O2", "-std=gnu11", "-DUSE_SIMD", "-march=native", "-pthread", "-I", str(source)]
        host, checker = temp / "host", temp / "fixture"
        report["builds"] = []
        for unit, binary in ((source / "examples/chuck_architect_train.c", host), (fixture, checker)):
            result = invoke(base_command + [unit, source / "notorch.c", "-lm", "-o", binary], env)
            report["builds"].append(result)
            assert result["returncode"] == 0, json.dumps(result)
        parent, student, alias = (temp / (name + ".policy.bin") for name in ("parent", "student", "alias"))
        emitted = invoke([checker, "emit", parent, student, alias], env)
        assert emitted["returncode"] == 0, json.dumps(emitted)
        report["fixture"]["emitter"] = emitted
        report["fixture"]["lives"] = {name: {"sha256": digest(path.read_bytes()), "hex": path.read_bytes().hex()}
                                         for name, path in (("parent", parent), ("student", student), ("alias", alias))}
        tokens = temp / "synthetic.u32"
        tokens.write_bytes(b"".join(struct.pack("<I", i % 23) for i in range(6400)))
        config = temp / "config.json"; config.write_text('{"mode":"learned","exploration":0}\n')

        def run(binary, label, probes, continuation=None, mode="--trajectory", steps=32):
            prefix = temp / label
            command = [binary, mode, "simple", tokens, prefix, steps, 42, ".0003", config, parent, int(probes)]
            if mode == "--trajectory": command.append(continuation)
            result = invoke(command, env)
            report.setdefault("runs", {})[label] = result
            assert result["returncode"] == 0, json.dumps(result)
            return prefix, lived.scenarios.events(Path(str(prefix) + ".jsonl"))

        control, control_rows = run(host, "control", False, student)
        report["control"] = lived.validate_cohort(control_rows, "simple", 42, policy_protocol, 32, False)
        report["cohorts"] = {}
        for label, continuation, is_alias in (("distinct", student, False), ("alias", alias, True)):
            prefix, rows = run(host, label, True, continuation)
            metadata = rows[0]
            assert metadata["diagnostics"] == "trajectory"
            assert metadata["source_saved_policy_hash"] == metadata["sealed_policy_hash"]
            assert metadata["source_saved_policy_hash"] != metadata["continuation_saved_policy_hash"]
            assert metadata["executed_continuations"] == (["policy"] if is_alias else ["policy", "student"])
            assert metadata["continuation_aliases"] == ({"student": "policy"} if is_alias else {})
            policy_rows = [row for row in rows if row.get("continuation") != "student"]
            validated = lived.validate_cohort(policy_rows, "simple", 42, policy_protocol, 32, True)
            parity = lived.compare_hosts(control_rows, rows, control, prefix, 32)
            forks = [row for row in rows if row["type"] == "fork"]
            checks, action_changes = [], 0
            for fork in forks:
                checkpoint = fork["step"]
                source_policy = Path(str(prefix) + f".fork-{checkpoint}.policy.bin")
                student_policy = Path(str(prefix) + f".fork-{checkpoint}.student.policy.bin")
                checked = invoke([checker, "graft", source_policy, student_policy, continuation], env)
                assert checked["returncode"] == 0, json.dumps(checked)
                saved = json.loads(checked["stdout"])
                assert saved["source"] == fork["policy_hash"] and saved["student"] == fork["student_policy_hash"]
                assert saved["continuation"] == metadata["continuation_saved_policy_hash"]
                assert saved["decisions"] == saved["updates"] == checkpoint - 1
                assert saved["has_history"] == int(checkpoint > 1)
                assert fork["executed_continuations"] == metadata["executed_continuations"]
                assert fork["continuation_aliases"] == metadata["continuation_aliases"]
                assert (source_policy.read_bytes() == student_policy.read_bytes()) is is_alias
                assert (fork["state_hash"] == fork["student_state_hash"]) is is_alias
                checks.append({"checkpoint": checkpoint, "native": checked, "source_hex": source_policy.read_bytes().hex(),
                               "student_hex": student_policy.read_bytes().hex()})
                for continuation_name in metadata["executed_continuations"]:
                    policy_hash = fork["policy_hash"] if continuation_name == "policy" else fork["student_policy_hash"]
                    initial_hash = fork["state_hash"] if continuation_name == "policy" else fork["student_state_hash"]
                    for action_index, action in enumerate(lived.ACTIONS, 1):
                        branch = [r for r in rows if r["type"] == "branch_step" and
                                  (r["checkpoint"], r["continuation"], r["intervention"]) == (checkpoint, continuation_name, action)]
                        assert [r["update"] for r in branch] == list(range(1, 17))
                        for index, row in enumerate(branch):
                            policy, previous = row["architect"], branch[index - 1] if index else None
                            assert row["source_hash"] == fork["state_hash"]
                            assert row["initial_policy_hash"] == policy_hash and row["branch_initial_hash"] == initial_hash
                            assert policy["policy_pre"] == (previous["architect"]["policy_post"] if previous else policy_hash)
                            assert row["forced"] is (index == 0)
                            assert policy["intervention"] is (index == 0)
                            assert row["offset"] == fork["offsets"][index]
                            assert policy["sequence"] == checkpoint + index
                            assert policy["consequence"]["learned"] == policy["consequence"]["nonfinite"] == 0
                            assert policy["consequence"]["after_loss"] == row["after_same_window"]
                            assert policy["features"] == fork["features"] if not index else policy["features"][15] == previous["architect"]["consequence"]["reward"]
                            assert row["action"] == action_index if not index else row["action"] == 1 + max(range(3), key=lambda k: policy["scores"][k])
                            if continuation_name == "student" and index:
                                source_row = next(r for r in rows if r["type"] == "branch_step" and
                                                  (r["checkpoint"], r["continuation"], r["intervention"], r["update"]) == (checkpoint, "policy", action, index + 1))
                                action_changes += row["action"] != source_row["action"]
                        measurements = [r for r in rows if r["type"] == "comparison" and
                                        (r["checkpoint"], r["continuation"], r["action"]) == (checkpoint, continuation_name, action)]
                        assert [r["horizon"] for r in measurements] == [1, 4, 16]
                        for measured in measurements:
                            assert measured["initial_hash"] == fork["state_hash"]
                            assert measured["initial_policy_hash"] == policy_hash and measured["branch_initial_hash"] == initial_hash
                            assert measured["state_hash"] == branch[measured["horizon"] - 1]["state_hash"]
            assert action_changes > 0 if not is_alias else action_changes == 0
            expected_updates = 96 if is_alias else 192
            assert sum(row["type"] == "branch_step" for row in rows) == expected_updates
            report["cohorts"][label] = {"source_policy_validation": validated, "host_parity": parity,
                                         "grafts": checks, "student_action_changes": action_changes,
                                         "branch_updates": expected_updates, "events": rows}
        print("PASS distinct continuation, acquired source state, weight-only alias, host parity", flush=True)

        legacy_prefix, legacy_rows = run(host, "legacy", True, mode="--lived")
        report["legacy"] = lived.validate_cohort(legacy_rows, "simple", 42, legacy_protocol, 32, True)
        if args.old_host:
            old_prefix, old_rows = run(args.old_host.resolve(), "prechange", True, mode="--lived")
            assert old_rows[:-1] == legacy_rows[:-1], "existing lived trace changed"
            old_summary, summary = dict(old_rows[-1]), dict(legacy_rows[-1])
            for field in ("seconds", "max_rss_kib"): old_summary.pop(field); summary.pop(field)
            assert old_summary == summary
            files = {}
            for path in sorted(temp.glob("legacy.*")):
                suffix = path.name.removeprefix("legacy")
                if suffix == ".jsonl": continue
                previous = Path(str(old_prefix) + suffix)
                assert path.read_bytes() == previous.read_bytes(), "existing lived artifact changed: " + suffix
                files[suffix] = digest(path.read_bytes())
            report["legacy_compatibility"] = {"status": "PASS", "prechange_binary_sha256": digest(args.old_host.read_bytes()),
                                                "artifacts": files, "events_compared": len(legacy_rows)}
        print("PASS existing lived trajectory", flush=True)

        labels = {}
        for label in ("student", "refit-parent", "refit-self"):
            prefix = temp / ("rollout-" + label)
            result = invoke([host, "--rollout", "simple", tokens, prefix, 1, 42, ".0003", config, label, parent, 0], env)
            assert result["returncode"] == 0, json.dumps(result)
            rows = lived.scenarios.events(Path(str(prefix) + ".jsonl"))
            assert rows[0]["arm"] == label
            assert [r for r in rows if r["type"] == "rollout_step"] == [r for r in control_rows if r["type"] == "rollout_step"][:1]
            labels[label] = result
        report["rollout_labels"] = labels

        target = source / "examples/chuck_architect_lived.h"
        text = target.read_text()
        anchor = "rollout_weights(architect, continuation); // NT_TRAJECTORY_STUDENT_WEIGHTS"
        assert text.count(anchor) == 1
        mutant = text.replace(anchor, "rollout_weights(architect, &saved->architect); // NT_TRAJECTORY_STUDENT_WEIGHTS")
        target.write_text(mutant)
        mutated_host = temp / "wrong-continuation"
        built = invoke(base_command + [source / "examples/chuck_architect_train.c", source / "notorch.c", "-lm", "-o", mutated_host], env)
        report["wrong_weight_mutation"] = {"build": built, "mutant_sha256": digest(mutant.encode())}
        assert built["returncode"] == 0, json.dumps(built)
        result = invoke([mutated_host, "--trajectory", "simple", tokens, temp / "mutant", 1, 42, ".0003", config, parent, 1, student], env)
        caught = result["returncode"] == 1 and "trajectory branch weights differ from named continuation" in result["stderr"]
        report["wrong_weight_mutation"].update(run=result, caught=caught)
        assert caught, json.dumps(result)
        print("PASS caught wrong-continuation weights", flush=True)

    report["working_sources_unchanged"] = all((ROOT / name).read_bytes() == data for name, data in originals.items())
    assert report["working_sources_unchanged"]
    report["passed"] = True
    save_report()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
