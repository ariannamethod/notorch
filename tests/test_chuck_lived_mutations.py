#!/usr/bin/env python3
"""Exercise lived-world diagnostics and their actual terminal receipt verifier.

Private builds use an explicit synthetic token corpus and history-sensitive
policy. No experiment evaluation seeds or corpora are loaded. Compiler failures
and crashes do not count as detected defects; a named invariant must reject the
compiled mutation, or the production semantic verifier must reject its trace.
"""
from __future__ import annotations

import argparse
import atexit
import copy
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
             "examples/chuck_architect_rollout.h", "examples/chuck_architect_lived.h", "tests/test_chuck_lived.c")
VERIFIERS = ("experiments/chuck_loss_architect/lived/run.py", "experiments/chuck_loss_architect/lived/protocol.json",
             "experiments/chuck_loss_architect/conditional/run.py", "experiments/chuck_loss_architect/future/run.py",
             "experiments/chuck_loss_architect/scenarios/run.py", "experiments/chuck_loss_architect/run.py")


def digest(data):
    return hashlib.sha256(data).hexdigest()


def invoke(command, env=None):
    result = subprocess.run([str(x) for x in command], cwd=ROOT, capture_output=True,
                            text=True, timeout=240, env=env)
    return {"command": [str(x) for x in command], "returncode": result.returncode,
            "stdout": result.stdout, "stderr": result.stderr}


def rejected(call):
    try:
        call()
    except (AssertionError, ValueError, OSError) as exc:
        return {"caught": True, "exception": type(exc).__name__, "reason": str(exc)}
    return {"caught": False, "reason": "mutated receipt accepted"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()
    originals = {name: (ROOT / name).read_bytes() for name in (*NUMERICAL, *VERIFIERS)}
    spec = importlib.util.spec_from_file_location("chuck_lived_test_runner", ROOT / VERIFIERS[0])
    lived = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(lived)
    protocol = json.loads(originals[VERIFIERS[1]])
    compiler = shlex.split(os.environ.get("CC", "cc"))
    env = dict(os.environ, NT_SIMD_THREADS="2", OMP_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2")
    report = {"source_sha256": {n: digest(b) for n, b in originals.items()}, "compiled_mutations": [],
              "semantic_mutations": [], "terminal_mutations": [], "passed": False,
              "fixture": {"body": "simple", "parameters": 450688, "seed": 42, "steps": 32,
                          "corpus": "6400 synthetic tokens, token[i] = i mod 23",
                          "policy": "synthetic 163-parameter life; BRAKE/PUSH scores depend on native previous reward",
                          "purpose": "native history, matched continuation and noninterference contracts"}}
    def save_report():
        if args.json:
            args.json.parent.mkdir(parents=True, exist_ok=True)
            args.json.write_text(json.dumps(report, indent=2) + "\n")
    atexit.register(save_report)  # Retain a failed build/baseline too.
    steps, seed = 32, 42
    with tempfile.TemporaryDirectory(prefix="notorch-lived-gates-") as temporary:
        temp = Path(temporary)
        source = temp / "source"
        for name in NUMERICAL:
            path = source / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(originals[name])

        def build(label, changed=None, api=False):
            directory = temp / label
            directory.mkdir()
            for name in NUMERICAL:
                path = directory / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(changed.get(name, originals[name]) if changed else originals[name])
            binary = directory / "gate"
            unit = "tests/test_chuck_lived.c" if api else "examples/chuck_architect_train.c"
            command = compiler + ["-O2", "-std=gnu11", "-DUSE_SIMD", "-march=native", "-pthread",
                                  "-I", str(directory), str(directory / unit), str(directory / "notorch.c"),
                                  "-lm", "-o", str(binary)]
            return binary, invoke(command, env)

        api, built = build("api", api=True)
        report["api"] = {"build": built}
        if built["returncode"] == 0:
            report["api"]["test"] = invoke([api], env)
        assert report["api"].get("test", {}).get("returncode") == 0, json.dumps(report["api"])
        parent = temp / "fixture.policy.bin"
        report["fixture"]["emitter"] = invoke([api, "--write-life", parent], env)
        assert report["fixture"]["emitter"]["returncode"] == 0
        report["fixture"]["policy_sha256"] = digest(parent.read_bytes())
        tokens = temp / "synthetic.tokens.u32"
        tokens.write_bytes(b"".join(struct.pack("<I", i % 23) for i in range(6400)))
        config = temp / "config.json"
        config.write_text('{"mode":"learned","exploration":0}\n')

        def host(binary, prefix, probes):
            return invoke([binary, "--lived", "simple", tokens, prefix, steps, seed,
                           "0.0003", config, parent, int(probes)], env)

        binary, built = build("baseline")
        report["baseline"] = {"build": built}
        assert built["returncode"] == 0, json.dumps(built)
        stem = temp / "fixture"
        control, diagnostic = Path(str(stem) + "-control"), Path(str(stem) + "-diagnostic")
        report["baseline"]["control_run"] = host(binary, control, False)
        report["baseline"]["diagnostic_run"] = host(binary, diagnostic, True)
        assert report["baseline"]["control_run"]["returncode"] == report["baseline"]["diagnostic_run"]["returncode"] == 0, json.dumps(report["baseline"])
        left = lived.scenarios.events(Path(str(control) + ".jsonl"))
        right = lived.scenarios.events(Path(str(diagnostic) + ".jsonl"))
        valid_left = lived.validate_cohort(left, "simple", seed, protocol, steps, False)
        valid_right = lived.validate_cohort(right, "simple", seed, protocol, steps, True)
        parity = lived.compare_hosts(left, right, control, diagnostic, steps)
        report["baseline"].update(control=valid_left, diagnostic=valid_right, parity=parity,
                                  diagnostic_events=right)
        policy_steps = [r for r in right if r["type"] == "branch_step" and r["continuation"] == "policy" and r["update"] > 1]
        assert any(r["action"] != 1 for r in policy_steps), "fixture has no learned-versus-HOLD distinction"
        assert any(abs(r["architect"]["features"][15]) > .001 for r in policy_steps), "fixture has no acquired reward in next readout"
        # The exact synthetic first hidden unit is tanh(16 * prior reward).
        # At zero prior reward this policy selects BRAKE; its real positive
        # consequences must produce at least one subsequent PUSH.
        assert any(r["action"] == 3 and r["architect"]["features"][15] > .001 for r in policy_steps), "history does not change fixture action preference"
        print("PASS API and real-body source/continuation baseline", flush=True)

        mutations = [
            {"name": "learned_continuation_replaced_by_hold", "file": "examples/chuck_architect_lived.h",
             "anchor": "int forced = h == 1 || ci == 0; // NT_LIVED_CONTINUATION",
             "replacement": "int forced = 1; // NT_LIVED_CONTINUATION",
             "expected_refusal": "lived selected continuation differs from host"},
            {"name": "native_intervention_reward_discarded", "file": "chuck_architect_impl.h",
             "anchor": "a.prev_reward = r.reward; // NT_CA_INTERVENTION_HISTORY",
             "replacement": "a.prev_reward = 0; // NT_CA_INTERVENTION_HISTORY", "api": True,
             "expected_refusal": "FAIL selected_equivalence:"},
            {"name": "diagnostic_history_leaks_into_source", "file": "examples/chuck_architect_lived.h",
             "anchor": "scenario_restore(saved, m, architect, windows); // NT_LIVED_HOST_RESTORE",
             "replacement": "architect->prev_reward = .75f; // NT_LIVED_HOST_RESTORE",
             "expected_refusal": "lived host restore identity differs"},
            {"name": "comparison_outcome_action_swapped", "file": "examples/chuck_architect_lived.h",
             "anchor": "step, continuations[ci], names[ai], h, saved_hash, branch_hash, ok ? \"true\" : \"false\", before, // NT_LIVED_COMPARISON_ACTION",
             "replacement": "step, continuations[ci], names[ai == 1 ? 2 : ai == 2 ? 1 : 0], h, saved_hash, branch_hash, ok ? \"true\" : \"false\", before, // NT_LIVED_COMPARISON_ACTION"},
        ]
        for m in mutations:
            text = originals[m["file"]].decode()
            assert text.count(m["anchor"]) == 1, "mutation anchor is not unique: " + m["name"]
            changed = text.replace(m["anchor"], m["replacement"], 1).encode()
            candidate, built = build(m["name"], {m["file"]: changed}, m.get("api", False))
            record = {"name": m["name"], "source": m["file"], "mutant_sha256": digest(changed),
                      "build": built, "caught": False}
            if built["returncode"] == 0:
                prefix = temp / (m["name"] + "-trace")
                result = invoke([candidate, "selected_equivalence"], env) if m.get("api") else host(candidate, prefix, True)
                record["run"] = result
                if "expected_refusal" in m:
                    record["caught"] = result["returncode"] == 1 and m["expected_refusal"] in result["stderr"]
                elif result["returncode"] == 0:
                    rows = lived.scenarios.events(Path(str(prefix) + ".jsonl"))
                    record["semantic_refusal"] = rejected(lambda: lived.validate_cohort(rows, "simple", seed, protocol, steps, True))
                    record["caught"] = record["semantic_refusal"]["caught"]
            report["compiled_mutations"].append(record)
            print(("PASS caught " if record["caught"] else "FAIL escaped/invalid ") + m["name"], flush=True)

        def semantic(name, modify):
            rows = copy.deepcopy(right)
            modify(rows)
            result = {"name": name, **rejected(lambda: lived.validate_cohort(rows, "simple", seed, protocol, steps, True))}
            report["semantic_mutations"].append(result)
            print(("PASS caught " if result["caught"] else "FAIL escaped ") + name, flush=True)

        def first(rows, kind):
            return next(r for r in rows if r["type"] == kind)

        semantic("missing_branch_step", lambda rows: rows.remove(first(rows, "branch_step")))
        semantic("duplicated_branch_step", lambda rows: rows.insert(rows.index(first(rows, "branch_step")), copy.deepcopy(first(rows, "branch_step"))))
        semantic("missing_action_outcome", lambda rows: rows.remove(first(rows, "comparison")))
        semantic("altered_outcome_state_identity", lambda rows: first(rows, "comparison").__setitem__("state_hash", "0000000000000000"))
        semantic("missing_native_history", lambda rows: next(r for r in rows if r["type"] == "branch_step" and r["update"] == 2)["architect"]["features"].__setitem__(15, 0))
        semantic("altered_restore_identity", lambda rows: first(rows, "restore").__setitem__("state_hash", "0000000000000000"))
        semantic("missing_continuation_check", lambda rows: rows.remove(first(rows, "continuation_check")))
        semantic("missing_terminal_summary", lambda rows: rows.pop())

        files = {path.relative_to(temp).as_posix(): lived.identity(path)
                 for prefix in (control, diagnostic) for path in temp.glob(prefix.name + ".*") if path.is_file()}
        anchors = {"protocol": protocol, "steps": steps, "files": files,
                   "cohorts": [{"prefix": stem.name, "body": "simple", "seed": seed,
                                "control": valid_left, "diagnostic": valid_right, "parity": parity}],
                   "fits": {}, "readouts": {}, "rollouts": [], "continuation": [], "source_deployment": [], "lives": {}}
        report["terminal_baseline"] = lived.verify_terminal_artifacts(temp, anchors)
        trace = Path(str(diagnostic) + ".jsonl")
        original_trace = trace.read_bytes()
        terminal_changes = [
            ("persisted_trace_truncated", trace, original_trace[:len(original_trace) // 2]),
            ("persisted_trace_missing", trace, None),
            ("persisted_trace_altered", trace, original_trace.replace(b'"forced":true', b'"forced":false', 1)),
        ]
        policy = Path(str(diagnostic) + ".policy.final.bin")
        original_policy = policy.read_bytes()
        terminal_changes.append(("persisted_policy_changed", policy, original_policy[:-1] + bytes([original_policy[-1] ^ 1])))
        for name, path, corrupt in terminal_changes:
            saved = path.read_bytes()
            if corrupt is None: path.unlink()
            else: path.write_bytes(corrupt)
            result = {"name": name, **rejected(lambda: lived.verify_terminal_artifacts(temp, anchors))}
            report["terminal_mutations"].append(result)
            path.write_bytes(saved)
            assert lived.verify_terminal_artifacts(temp, anchors)["status"] == "PASS", "restored baseline rejected"
            print(("PASS caught " if result["caught"] else "FAIL escaped ") + name, flush=True)
    report["working_sources_unchanged"] = all((ROOT / n).read_bytes() == data for n, data in originals.items())
    report["passed"] = (len(report["compiled_mutations"]) == 4 and len(report["semantic_mutations"]) == 8 and
                        len(report["terminal_mutations"]) == 4 and all(r["caught"] for r in
                        report["compiled_mutations"] + report["semantic_mutations"] + report["terminal_mutations"]) and
                        report["working_sources_unchanged"])
    save_report()
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
