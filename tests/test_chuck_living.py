#!/usr/bin/env python3
"""Bounded native gates for consequences acquired inside a Chuck training life.

The private test build uses the explicit 8-token fixture body, synthetic tokens,
and an emitted history-sensitive policy. It never reads experiment corpora or
saved production policies. A compiled credit defect must be rejected by its
actual measured-action association; crashes and build failures are not evidence.
"""
from __future__ import annotations

import argparse
import atexit
import hashlib
import json
import math
import os
from pathlib import Path
import shlex
import struct
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[1]
NUMERICAL = (
    "notorch.c", "notorch.h", "notorch_simd.h", "chuck_architect.h",
    "chuck_architect_impl.h", "examples/chuck_architect_train.c",
    "examples/chuck_architect_scenarios.h", "examples/chuck_architect_rollout.h",
    "examples/chuck_architect_lived.h", "examples/chuck_architect_living.h",
)
ACTIONS = ("hold", "brake", "push")
FINAL_FILES = (".final.bin", ".moments.final.bin", ".optimizer.final.json", ".policy.final.bin")
FIXTURE = r'''
#include "chuck_architect.h"
#include <inttypes.h>
#include <stdio.h>
#include <string.h>
static void weights(nt_chuck_architect *to, const nt_chuck_architect *from) {
    memcpy(to->w1, from->w1, sizeof to->w1); memcpy(to->b1, from->b1, sizeof to->b1);
    memcpy(to->w2, from->w2, sizeof to->w2); memcpy(to->b2, from->b2, sizeof to->b2);
}
static float read_float(FILE *f) {
    unsigned char b[4];
    if (fread(b, 1, 4, f) != 4) { fputs("truncated replay sample\n", stderr); exit(10); }
    uint32_t bits = (uint32_t)b[0] | (uint32_t)b[1]<<8 | (uint32_t)b[2]<<16 | (uint32_t)b[3]<<24;
    float v; memcpy(&v, &bits, sizeof v); return v;
}
int main(int argc, char **argv) {
    nt_chuck_architect a, b;
    if (argc == 3 && !strcmp(argv[1], "emit")) {
        nt_chuck_architect_config c;
        if (nt_chuck_architect_config_parse_json(&c,
            "{\"mode\":\"learned\",\"exploration\":0,\"enabled_actions\":[\"hold\",\"brake\",\"push\"]}", NULL, 0) ||
            nt_chuck_architect_init(&a, &c)) return 2;
        memset(a.w1, 0, sizeof a.w1); memset(a.b1, 0, sizeof a.b1);
        memset(a.w2, 0, sizeof a.w2); memset(a.b2, 0, sizeof a.b2);
        a.w1[0][15] = 16; a.w2[1][0] = -1; a.w2[2][0] = 1;
        a.b2[0] = -.2f; a.b2[1] = .01f; a.b2[2] = -.01f;
        if (nt_chuck_architect_save(&a, argv[2])) return 3;
    } else if (argc == 4 && !strcmp(argv[1], "weight-only")) {
        if (nt_chuck_architect_load(&a, argv[2]) || nt_chuck_architect_load(&b, argv[3])) return 4;
        nt_chuck_architect restored = b; weights(&restored, &a);
        if (memcmp(&restored, &a, sizeof a)) { fputs("fit changed non-weight life state\n", stderr); return 5; }
        printf("{\"before\":\"%016" PRIx64 "\",\"after\":\"%016" PRIx64 "\",\"changed\":%s}\n",
               nt_chuck_architect_hash(&a), nt_chuck_architect_hash(&b), memcmp(&a, &b, sizeof a) ? "true" : "false");
        return 0;
    } else if (argc == 5 && !strcmp(argv[1], "replay")) {
        if (nt_chuck_architect_load(&a, argv[2]) || nt_chuck_architect_load(&b, argv[4])) return 6;
        FILE *f = fopen(argv[3], "rb"); if (!f) return 7;
        nt_chuck_architect_comparison s;
        for (int i = 0; i < NT_CHUCK_ARCHITECT_FEATURES; ++i) s.features[i] = read_float(f);
        for (int i = 0; i < NT_CHUCK_ARCHITECT_HEADS; ++i) s.future_loss[i] = read_float(f);
        if (fgetc(f) != EOF || ferror(f) || fclose(f)) return 8;
        for (int i = 0; i < 32; ++i) if (nt_chuck_architect_fit_conditioned(&a, &s, NULL)) return 9;
        if (memcmp(&a, &b, sizeof a)) { fputs("saved life differs from actual assigned consequence replay\n", stderr); return 11; }
    } else return 2;
    printf("{\"hash\":\"%016" PRIx64 "\"}\n", nt_chuck_architect_hash(&a));
    return 0;
}
'''


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def f32(value: float) -> float:
    return struct.unpack("<f", struct.pack("<f", value))[0]


def events(path: Path) -> list[dict]:
    def bad(value):
        raise ValueError("nonfinite JSON literal " + value)
    return [json.loads(line, parse_constant=bad) for line in path.read_text().splitlines()]


def invoke(command, env):
    command = list(map(str, command))
    result = subprocess.run(command, cwd=ROOT, env=env, capture_output=True,
                            text=True, timeout=240)
    return {"command": command, "returncode": result.returncode,
            "stdout": result.stdout, "stderr": result.stderr}


def selected(rows: list[dict], kind: str) -> list[dict]:
    return [row for row in rows if row["type"] == kind]


def verify_targets(assigned: list[float], targets: list[float], scale: float) -> None:
    values = list(map(f32, assigned))
    expected_scale = max(1e-6 * (abs(values[0]) + 1), *(abs(values[0] - value) for value in values))
    assert scale == expected_scale, "recorded conditioned scale differs from measured assigned losses"
    expected = [f32((values[0] - value) / expected_scale) for value in values]
    assert list(map(f32, targets)) == expected, "conditioned reward differs from actual assigned consequence"


def final_parity(left: Path, right: Path, policy=True) -> dict:
    identities = {}
    for suffix in FINAL_FILES if policy else FINAL_FILES[:-1]:
        a, b = Path(str(left) + suffix).read_bytes(), Path(str(right) + suffix).read_bytes()
        assert a == b, "final continuation differs: " + suffix
        identities[suffix] = digest(a)
    return identities


def validate(rows: list[dict], arm: str, steps: int, probes: bool) -> dict:
    """Independent event/action joins; the production runner has its own gate."""
    assert rows and rows[0]["type"] == "living_run" and rows[-1]["type"] == "living_summary", "incomplete living trace"
    metadata, summary = rows[0], rows[-1]
    assert len(selected(rows, "living_run")) == len(selected(rows, "living_summary")) == 1
    assert metadata["arm"] == arm and metadata["steps"] == steps and metadata["probes"] is probes
    assert metadata["horizon"] == 16 and metadata["feedback_regression"] is False
    assert f32(metadata["policy_learning_rate"]) == f32(.03)
    assert metadata["fits_per_acquisition"] == (32 if arm in ("living", "rotated") else 0)
    assert summary["steps"] == steps and summary["failed"] is False
    allowed = {"living_run", "living_summary", "acquisition", "branch_step", "comparison", "restore",
               "fit", "acquisition_result", "rollout_step", "continuation_check", "policy_resume", "evaluation"}
    assert all(row["type"] in allowed for row in rows), "unexpected failure or event"
    checkpoints = list(range(1, min(steps, 449) + 1, 64)) if probes else []
    acquisition = selected(rows, "acquisition")
    results = selected(rows, "acquisition_result")
    restores = selected(rows, "restore")
    hosts = selected(rows, "rollout_step")
    branches = selected(rows, "branch_step")
    fits = selected(rows, "fit")
    comparisons = selected(rows, "comparison")
    continuation = selected(rows, "continuation_check")
    assert [row["checkpoint"] for row in acquisition] == checkpoints
    assert [row["checkpoint"] for row in results] == checkpoints
    assert [row["checkpoint"] for row in restores] == checkpoints
    assert [row["step"] for row in hosts] == list(range(1, steps + 1))
    assert len(branches) == len(checkpoints) * 3 * 16 and len(comparisons) == len(checkpoints) * 3
    expected_fits = len(checkpoints) * (32 if arm in ("living", "rotated") else 0)
    assert len(fits) == expected_fits and [row["fit_step"] for row in fits] == list(range(1, expected_fits + 1))
    acq_by_step = {row["checkpoint"]: row for row in acquisition}
    result_by_step = {row["checkpoint"]: row for row in results}
    positions = {id(row): index for index, row in enumerate(rows)}

    def argmax(values):
        assert len(values) == 3 and all(math.isfinite(value) for value in values)
        return 1 + max(range(3), key=lambda k: values[k])

    def feedback(row, sequence, prior=None, forced=False):
        actual = row["architect"]
        consequence = actual["consequence"]
        before, after = row["before"], row.get("after_same_window", row.get("after"))
        assert math.isfinite(before) and math.isfinite(after)
        assert actual["sequence"] == consequence["decision"] == sequence
        assert actual["explored"] == 0 and consequence["learned"] == consequence["nonfinite"] == 0
        assert consequence["error"] == 0
        assert consequence["before_loss"] == before and consequence["after_loss"] == after
        delta = f32(before) - f32(after)
        reward = f32(max(-1, min(1, delta / (abs(f32(before)) + 1e-6))))
        assert consequence["loss_delta"] == delta and f32(consequence["reward"]) == reward, "feedback missed its actual same-window consequence"
        assert actual["action"]["kind"] == row["action"] and actual["action"]["value"] == 0
        assert actual["action"]["type"] == ACTIONS[row["action"] - 1]
        assert consequence["predicted"] == actual["scores"][row["action"] - 1]
        assert len(actual["features"]) == 16 and all(math.isfinite(x) and -1 <= x <= 1 for x in actual["features"])
        assert actual["observation"]["loss"] == before and actual["observation"]["step"] == sequence
        if not forced:
            assert row["action"] == argmax(actual["scores"]), "actual action differs from acquired argmax"
        if prior is not None:
            previous = prior["architect"]
            expected_delta = f32(max(-1, min(1, (f32(before) - f32(prior["before"])) / (1 + abs(f32(prior["before"]))))))
            assert f32(actual["features"][13]) == expected_delta, "loss history was discarded"
            assert f32(actual["features"][14]) == f32(max(-1, min(1, previous["observation"]["loss_trend"])))
            assert actual["features"][15] == previous["consequence"]["reward"], "acquired reward was discarded"
        elif sequence == 1:
            assert actual["features"][13:] == [0, 0, 0]

    for checkpoint, start, end, restored in zip(checkpoints, acquisition, results, restores):
        assert start["arm"] == end["arm"] == arm
        assert positions[id(start)] < positions[id(restored)] < positions[id(end)] < positions[id(hosts[checkpoint - 1])]
        assert restored["exact"] is True
        assert start["source_hash"] == restored["source_hash"] == restored["restored_hash"] == end["source_hash"] == end["restored_hash"]
        assert end["non_weight_fields_unchanged"] is True and end["source_world_unchanged"] is True
        assert end["features"] == start["features"] and end["scores_before"] == start["scores_before"]
        assert end["policy_before"] == start["policy_hash"]
        assert start["action_before"] == end["action_before"] == argmax(start["scores_before"])
        assert end["action_after"] == argmax(end["scores_after"])
        assert len(start["offsets"]) == 20
        measured = []
        for action_index, action in enumerate(ACTIONS, 1):
            branch = [row for row in branches if (row["checkpoint"], row["intervention"]) == (checkpoint, action)]
            assert [row["update"] for row in branch] == list(range(1, 17)), "incomplete or duplicated measured branch"
            policy_hash = start["policy_hash"]
            for index, row in enumerate(branch):
                actual = row["architect"]
                assert positions[id(start)] < positions[id(row)] < positions[id(restored)]
                assert row["continuation"] == "current" and row["source_hash"] == start["source_hash"]
                assert row["forced"] is (index == 0) and actual["intervention"] is (index == 0)
                assert row["offset"] == start["offsets"][index]
                assert actual["policy_pre"] == policy_hash, "branch did not retain current life history"
                if not index:
                    assert row["action"] == action_index and actual["features"] == start["features"]
                    assert actual["observation"] == start["observation"] and actual["scores"] == start["scores_before"]
                feedback(row, checkpoint + index, branch[index - 1] if index else None, index == 0)
                policy_hash = actual["policy_post"]
            records = [row for row in comparisons if (row["checkpoint"], row["action"]) == (checkpoint, action)]
            assert len(records) == 1, "missing or duplicated named consequence"
            outcome = records[0]
            assert positions[id(branch[-1])] < positions[id(outcome)] < positions[id(restored)]
            assert outcome["action_kind"] == action_index and outcome["horizon"] == 16 and outcome["finite"] is True
            assert outcome["source_hash"] == start["source_hash"] and outcome["policy_hash"] == start["policy_hash"]
            assert outcome["state_hash"] == branch[-1]["state_hash"]
            assert outcome["immediate_after"] == branch[0]["after_same_window"]
            assert all(math.isfinite(outcome[key]) for key in ("future_after", "heldout_after"))
            measured.append(outcome["future_after"])
        assigned = [measured[2], measured[0], measured[1]] if arm == "rotated" else measured
        assert end["measured_future_loss"] == measured
        assert end["assigned_future_loss"] == assigned, "action-to-outcome association differs from named arm"
        assert end["assignment"] == ("rotated-PHB" if arm == "rotated" else "measured-HBP")
        own_fits = [row for row in fits if row["checkpoint"] == checkpoint]
        assert end["fit_iterations"] == len(own_fits) == (0 if arm == "frozen" else 32)
        assert end["fit_updates"] == sum(row["checkpoint"] <= checkpoint for row in fits)
        previous_hash, previous_scores = start["policy_hash"], start["scores_before"]
        for iteration, row in enumerate(own_fits, 1):
            assert positions[id(restored)] < positions[id(row)] < positions[id(end)]
            assert row["iteration"] == iteration and row["source_hash"] == start["source_hash"]
            assert row["hash_before"] == previous_hash and row["predicted_before"] == previous_scores
            assert row["measured_future_loss"] == measured
            assert row["assigned_future_loss"] == assigned, "action-to-outcome association differs from named arm"
            verify_targets(assigned, row["target"], row["scale"])
            assert row["online_decisions"] == row["online_updates"] == checkpoint - 1
            assert all(f32(error) == f32(f32(predicted) - f32(target)) for error, predicted, target in zip(row["error_before"], row["predicted_before"], row["target"]))
            previous_hash, previous_scores = row["hash_after"], row["predicted_after"]
        assert end["policy_after"] == previous_hash and end["scores_after"] == previous_scores
        if arm == "frozen":
            assert end["post_fit_state_hash"] == start["source_hash"]

    previous_policy = metadata["initial_policy_hash"]
    actions = [0, 0, 0, 0]
    brake_changed = brake_bound = 0
    for index, row in enumerate(hosts):
        step = index + 1
        assert row["action"] in ((0,) if arm == "canonical" else (1, 2, 3))
        actions[row["action"]] += 1
        if arm != "canonical":
            if step in result_by_step:
                start, end = acq_by_step[step], result_by_step[step]
                assert start["policy_hash"] == previous_policy
                assert row["architect"]["features"] == start["features"]
                assert row["architect"]["observation"] == start["observation"]
                assert row["architect"]["scores"] == end["scores_after"]
                assert row["action"] == end["action_after"]
                previous_policy = end["policy_after"]
            assert row["architect"]["policy_pre"] == previous_policy
            feedback(row, step, hosts[index - 1] if index else None)
            previous_policy = row["architect"]["policy_post"]
        if row["action"] == 2:
            before_dampen = row["pre_chuck"]["dampen"] if row["pre_chuck"]["initialized"] else 1
            if row["post_chuck"]["dampen"] < before_dampen:
                brake_changed += 1
            else:
                brake_bound += 1
    expected_checks = sum(min(16, steps - checkpoint + 1) for checkpoint in checkpoints) if arm == "frozen" else 0
    assert len(continuation) == expected_checks
    expected_pairs = [(checkpoint, step) for checkpoint in checkpoints
                      for step in range(checkpoint, min(checkpoint + 16, steps + 1))] if arm == "frozen" else []
    assert [(row["checkpoint"], row["step"]) for row in continuation] == expected_pairs, "missing or duplicated frozen continuation"
    for row in continuation:
        checkpoint, step, update = row["checkpoint"], row["step"], row["update"]
        assert step == checkpoint + update - 1 and 1 <= update <= 16 and row["exact"] is True
        action = ACTIONS[acq_by_step[checkpoint]["action_before"] - 1]
        branch = next(value for value in branches if (value["checkpoint"], value["intervention"], value["update"]) == (checkpoint, action, update))
        host = hosts[step - 1]
        assert row["state_hash"] == row["expected_hash"] == host["state_hash"] == branch["state_hash"]
        assert all(host[key] == branch[key] for key in ("offset", "window_rng", "before", "action", "pre_chuck", "post_chuck"))
        assert host["after"] == branch["after_same_window"]
        assert all(host["architect"][key] == branch["architect"][key] for key in ("policy_pre", "policy_post", "observation", "features", "scores", "consequence"))
    assert summary["acquisitions"] == len(checkpoints) and summary["branch_updates"] == len(branches)
    assert summary["fit_updates"] == expected_fits and summary["continuation_checks"] == expected_checks
    assert summary["action_counts"] == actions
    assert summary["brake_changed_dampen"] == brake_changed and summary["brake_at_bound"] == brake_bound
    assert summary["history_updates"] == (0 if arm == "canonical" else steps)
    assert summary["policy_hash"] == previous_policy
    evaluations = selected(rows, "evaluation")
    assert [row["step"] for row in evaluations] == [0, *range(128, steps + 1, 128), *([] if steps % 128 == 0 else [steps])]
    assert all(math.isfinite(row["heldout_loss"]) for row in evaluations)
    resumes = selected(rows, "policy_resume")
    assert len(resumes) == int(metadata["resume_step"] != 0)
    for row in resumes:
        assert row["step"] == metadata["resume_step"] and row["exact"] is True
        assert row["state_hash"] == hosts[row["step"] - 1]["state_hash"]
    return {"status": "PASS", "host_updates": steps, "acquisitions": len(checkpoints),
            "branch_updates": len(branches), "fit_updates": expected_fits,
            "continuation_checks": expected_checks, "action_counts": actions,
            "brake_changed_dampen": brake_changed, "brake_at_bound": brake_bound}


def acquisition_lives(prefix: Path, row: dict) -> tuple[Path, Path]:
    return tuple(Path(str(prefix) + f'.acquire-{row["checkpoint"]}.{phase}.policy.bin') for phase in ("pre", "post"))


def outcome_mutation() -> tuple[str, str]:
    anchor = "sample.future_loss[k] = measured[rotated ? (k + 2) % NT_CHUCK_ARCHITECT_HEADS : k]; // NT_LIVING_ASSIGNED_OUTCOMES"
    return anchor, "sample.future_loss[k] = measured[(k + 2) % NT_CHUCK_ARCHITECT_HEADS]; // NT_LIVING_ASSIGNED_OUTCOMES"


def fixture_choice(rows: list[dict]) -> dict:
    assert rows[0]["fixture"] is True
    changes = [row for row in selected(rows, "acquisition_result") if row["action_before"] != row["action_after"]]
    assert changes, "fixture acquired scores but did not change its next action"
    first = changes[0]
    host = next(row for row in selected(rows, "rollout_step") if row["step"] == first["checkpoint"])
    assert host["action"] == first["action_after"] and host["architect"]["features"] == first["features"]
    return {"checkpoint": first["checkpoint"], "before": first["action_before"], "after": first["action_after"],
            "source_hash": first["source_hash"], "policy_before": first["policy_before"],
            "policy_after": first["policy_after"], "actual_host_action": host["action"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()
    originals = {name: (ROOT / name).read_bytes() for name in NUMERICAL}
    env = dict(os.environ, NT_SIMD_THREADS="2", OMP_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2")
    compiler = shlex.split(os.environ.get("CC", "cc"))
    report = {"source_sha256": {name: digest(data) for name, data in originals.items()},
              "fixture_source_sha256": digest(FIXTURE.encode()), "passed": False,
              "fixture": {"body": "simple", "build_define": "NT_CHUCK_ARCHITECT_TEST",
                          "context": 8, "dimension": 8, "head_dimension": 4,
                          "layers": 1, "hidden": 16, "vocabulary": 8,
                          "steps": 80, "seed": 42, "resume_after": 40,
                          "corpus": "6400 synthetic tokens; token[i] = i mod 8"},
              "builds": [], "runs": {}, "policy_checks": []}

    def save_report():
        if args.json:
            args.json.parent.mkdir(parents=True, exist_ok=True)
            args.json.write_text(json.dumps(report, indent=2) + "\n")
    atexit.register(save_report)

    with tempfile.TemporaryDirectory(prefix="notorch-living-gates-") as temporary:
        temp = Path(temporary)
        source = temp / "source"
        for name, data in originals.items():
            path = source / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
        fixture = source / "fixture.c"
        fixture.write_text(FIXTURE)
        base = compiler + ["-O2", "-std=gnu11", "-DUSE_SIMD", "-march=native", "-pthread",
                           "-DNT_CHUCK_ARCHITECT_TEST", "-I", str(source)]
        host, checker = temp / "host", temp / "fixture"
        for unit, binary in ((source / "examples/chuck_architect_train.c", host), (fixture, checker)):
            result = invoke(base + [unit, source / "notorch.c", "-lm", "-o", binary], env)
            report["builds"].append(result)
            assert result["returncode"] == 0, json.dumps(result)
        life = temp / "fixture.policy.bin"
        emitted = invoke([checker, "emit", life], env)
        report["fixture"]["emitter"] = emitted
        assert emitted["returncode"] == 0, json.dumps(emitted)
        report["fixture"]["policy_hex"] = life.read_bytes().hex()
        tokens = temp / "synthetic.tokens.u32"
        tokens.write_bytes(b"".join(struct.pack("<I", i % 8) for i in range(6400)))
        config = temp / "config.json"
        config.write_text('{"mode":"learned","exploration":0,"learning_rate":0.03,'
                          '"enabled_actions":["hold","brake","push"]}\n')

        def run(label, arm, probes, resume=0, steps=80, binary=host):
            prefix = temp / label
            command = [binary, "--living", "simple", tokens, prefix, steps, 42,
                       ".0003", config, arm, "-" if arm == "canonical" else life, resume, int(probes)]
            result = invoke(command, env)
            report["runs"][label] = result
            assert result["returncode"] == 0, json.dumps(result)
            trace = Path(str(prefix) + ".jsonl")
            rows = events(trace)
            result.update(trace_sha256=digest(trace.read_bytes()), events=rows)
            result["validation"] = validate(rows, arm, steps, probes)
            return prefix, rows

        control, control_rows = run("frozen-control", "frozen", False)
        frozen, frozen_rows = run("frozen-probes", "frozen", True)
        assert selected(control_rows, "rollout_step") == selected(frozen_rows, "rollout_step"), "probes changed host transitions"
        assert selected(control_rows, "evaluation") == selected(frozen_rows, "evaluation"), "probes changed held-out evaluations"
        report["frozen_parity"] = {"transitions": 80, "final": final_parity(control, frozen)}
        print("PASS frozen probe-on/off full source continuation", flush=True)

        living, living_rows = run("living", "living", True)
        report["first_acquired_choice"] = fixture_choice(living_rows)
        resumed, resumed_rows = run("living-resumed", "living", True, resume=40)
        ignored = {"living_run", "living_summary", "policy_resume"}
        assert [row for row in living_rows if row["type"] not in ignored] == [row for row in resumed_rows if row["type"] not in ignored], "save/load changed acquired continuation"
        resumes = selected(resumed_rows, "policy_resume")
        assert len(resumes) == 1 and resumes[0]["step"] == 40 and resumes[0]["exact"], "save/load was not executed"
        report["living_resume"] = {"transitions": 80, "acquisitions": 2, "final": final_parity(living, resumed)}
        print("PASS deterministic learning and saved-life continuation", flush=True)

        rotated, rotated_rows = run("rotated", "rotated", True)
        report["rotated_control"] = {"correctly_labelled": True,
                                      "outcome_association_changes_final_policy": Path(str(rotated) + ".policy.final.bin").read_bytes() != Path(str(living) + ".policy.final.bin").read_bytes()}
        assert report["rotated_control"]["outcome_association_changes_final_policy"], "fixture does not distinguish the assigned consequences"

        for prefix, rows in ((frozen, frozen_rows), (living, living_rows), (rotated, rotated_rows)):
            for row in selected(rows, "acquisition_result"):
                before, after = acquisition_lives(prefix, row)
                checked = invoke([checker, "weight-only", before, after], env)
                entry = {"arm": rows[0]["arm"], "checkpoint": row["checkpoint"], "weight_only": checked}
                report["policy_checks"].append(entry)
                assert checked["returncode"] == 0, json.dumps(checked)
                checked_identity = json.loads(checked["stdout"])
                assert checked_identity["changed"] is (rows[0]["arm"] != "frozen"), "fixture fit did not change the expected weights"
                if rows[0]["arm"] != "frozen":
                    sample = temp / (prefix.name + "-" + str(row["checkpoint"]) + ".sample")
                    sample.write_bytes(b"".join(struct.pack("<f", value) for value in row["features"] + row["assigned_future_loss"]))
                    replayed = invoke([checker, "replay", before, sample, after], env)
                    entry["actual_credit_replay"] = replayed
                    assert replayed["returncode"] == 0, json.dumps(replayed)
        print("PASS measured action credit, weight-only acquisition and native replay", flush=True)

        canonical, canonical_rows = run("canonical", "canonical", False, steps=2)
        legacy = temp / "old-canonical"
        old = invoke([host, "--rollout", "simple", tokens, legacy, 2, 42,
                      ".0003", config, "canonical", "-", 0], env)
        report["canonical"] = {"old_run": old}
        assert old["returncode"] == 0, json.dumps(old)
        old_rows = events(Path(str(legacy) + ".jsonl"))
        assert selected(canonical_rows, "rollout_step") == selected(old_rows, "rollout_step"), "canonical dispatch changed"
        report["canonical"]["final"] = final_parity(canonical, legacy, policy=False)

        target = source / "examples/chuck_architect_living.h"
        text = target.read_text()
        anchor, replacement = outcome_mutation()
        assert text.count(anchor) == 1, "credit mutation anchor is not unique"
        mutant = text.replace(anchor, replacement, 1)
        target.write_text(mutant)
        wrong = temp / "wrong-credit"
        built = invoke(base + [source / "examples/chuck_architect_train.c", source / "notorch.c", "-lm", "-o", wrong], env)
        report["wrong_credit_mutation"] = {"build": built, "mutant_sha256": digest(mutant.encode()), "caught": False}
        assert built["returncode"] == 0, json.dumps(built)
        prefix = temp / "wrong-credit-output"
        executed = invoke([wrong, "--living", "simple", tokens, prefix, 1, 42,
                           ".0003", config, "living", life, 0, 1], env)
        report["wrong_credit_mutation"]["run"] = executed
        assert executed["returncode"] == 0, "mutation must reach the semantic gate, not crash: " + json.dumps(executed)
        rows = events(Path(str(prefix) + ".jsonl"))
        report["wrong_credit_mutation"]["events"] = rows
        try:
            validate(rows, "living", 1, True)
        except AssertionError as exc:
            reason = str(exc)
            assert "action-to-outcome" in reason, "mutation failed an unrelated invariant: " + reason
            report["wrong_credit_mutation"].update(caught=True, reason=reason)
        assert report["wrong_credit_mutation"]["caught"], "deliberately wrong living credit escaped"
        print("PASS caught compiled wrong action-to-outcome wiring", flush=True)

    report["working_sources_unchanged"] = all((ROOT / name).read_bytes() == data for name, data in originals.items())
    assert report["working_sources_unchanged"]
    report["passed"] = True
    save_report()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
