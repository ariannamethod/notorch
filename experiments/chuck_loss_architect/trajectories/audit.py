#!/usr/bin/env python3
"""Independent interpretation of the completed, fixed student-trajectory run."""
import argparse
import collections
import csv
import hashlib
import json
import math
from pathlib import Path
import struct
import traceback

PROTOCOL_SHA = "86673afd0abc04b5f389aeb4400765287fbc9a0951963639d961fc70a9c6c121"
ACTIONS = ("hold", "brake", "push")
FLOATS = ("loss", "loss_ema", "loss_trend", "macro_ema", "best_macro", "dampen", "lr_scale", "noise", "grad_norm", "grad_trend", "frozen_fraction")
INTS = ("step", "stag", "macro_stag", "history_len")
CHECKS = 0


def need(condition, label):
    global CHECKS
    CHECKS += 1
    if not condition:
        raise AssertionError(label)


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def identity(path):
    return {"bytes": path.stat().st_size, "sha256": digest(path)}


def read_json(path):
    return json.loads(path.read_text(), parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))


def f32(value):
    return struct.unpack("<f", struct.pack("<f", value))[0]


def fnv(data):
    value = 14695981039346656037
    for byte in data:
        value = ((value ^ byte) * 1099511628211) & ((1 << 64) - 1)
    return f"{value:016x}"


def life(data):
    need(len(data) == 1016 and data[:8] == b"NTCALIFE", "saved-life envelope")
    version, size, checksum = struct.unpack_from("<IIQ", data, 8)
    need(version == 1 and size == 992 and checksum == int(fnv(data[24:]), 16), "saved-life checksum")
    return data[24:]


def table_bytes(samples):
    parts = [b"NTCAFT01", struct.pack("<I", len(samples))]
    for sample in samples:
        obs = sample["observation"]
        parts.append(struct.pack("<IIIQQ", 1 if sample["body"] == "simple" else 2, sample["seed"], sample["checkpoint"], int(sample["state_hash"], 16), int(sample["policy_hash"], 16)))
        parts.append(struct.pack("<11f4I", *(obs[name] for name in FLOATS), *(obs[name] for name in INTS)))
        parts.append(struct.pack("<16f3f", *sample["features"], *sample["future_loss"]))
        name = sample["source_policy"].encode("ascii")
        parts.extend([struct.pack("<I", len(name)), name])
    return b"".join(parts)


def csv_file(path, rows):
    need(bool(rows), "nonempty CSV " + path.name)
    with path.open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def audit(run, out, expected_source, expected_result):
    need(digest(run / "results.json") == expected_result, "externally pinned results identity")
    result = read_json(run / "results.json")
    need(identity(run / "terminal_anchors.json") == result["artifacts"]["terminal_anchors.json"], "original terminal anchors")
    anchors = read_json(run / "terminal_anchors.json")
    authenticated = {}

    def raw(name):
        path = run / name
        if name not in authenticated:
            actual = identity(path)
            need(actual == result["artifacts"][name], "result identity " + name)
            if name in anchors["files"]:
                need(actual == anchors["files"][name], "original anchor " + name)
            authenticated[name] = actual
        return path

    def events(name):
        return [json.loads(line) for line in raw(name).read_text().splitlines()]

    protocol = read_json(raw("protocol.json"))
    manifest = read_json(raw("manifest.json"))
    sealed = read_json(raw("sealed_lives.json"))
    need(digest(raw("protocol.json")) == PROTOCOL_SHA, "unchanged preregistration")
    need(manifest == result["manifest"] and manifest["source_commit"] == expected_source, "clean source identity")
    need(not manifest["source_status"] and not manifest["smoke"], "clean full run")
    need(manifest["protocol"] == protocol == anchors["protocol"], "protocol copies")
    need(manifest["executed_seeds"] == {"development": [42, 73], "evaluation": [1013, 1217]}, "fixed seeds")
    need(sealed == result["sealed_lives"] and not sealed["evaluation_generation_started"], "saved seal")
    lives = {}
    for label, record in sealed["lives"].items():
        path = raw(record["file"])
        need(identity(path) == {key: record[key] for key in ("bytes", "sha256")}, "sealed life " + label)
        lives[label] = life(path.read_bytes())
    for label, record in protocol["input_lives"].items():
        need(sealed["lives"][label]["sha256"] == record["sha256"], "pinned input " + label)
    timeline = events("timeline.jsonl")
    need([row["phase"] for row in timeline] == ["acquisition_started", "lives_sealed", "evaluation_generation_started", "closed_loop_started", "evaluation_complete"], "phase order")
    need(all(a["time_ns"] <= b["time_ns"] for a, b in zip(timeline, timeline[1:])), "chronological phase receipts")
    need(timeline[1]["sealed_lives_sha256"] == timeline[2]["sealed_lives_sha256"] == digest(raw("sealed_lives.json")), "evaluation follows exact sealed lives")
    counts = collections.Counter()
    worlds, host_traces, initial_bodies = {}, {}, {}
    cohorts = anchors["cohorts"]
    expected_cohorts = {(split, source, body, seed) for split, sources, seeds in (("development", ["parent", "student"], [42, 73]), ("evaluation", ["student"], [1013, 1217])) for source in sources for body in protocol["body_order"] for seed in seeds}
    need({(c["split"], c["source"], c["body"], c["seed"]) for c in cohorts} == expected_cohorts and len(cohorts) == 12, "complete source cohorts")
    for cohort in cohorts:
        key = cohort["split"], cohort["source"], cohort["body"], cohort["seed"]
        traces = {}
        for variant in ("control", "diagnostic"):
            prefix = cohort["prefix"] + "-" + variant
            rows = events(prefix + ".jsonl")
            traces[variant] = rows
            steps = [row for row in rows if row["type"] == "rollout_step"]
            need([row["step"] for row in steps] == list(range(1, 257)), "complete source updates")
            need(rows[-1]["type"] == "rollout_summary" and not rows[-1]["failed"], "source completes")
            need(rows[0]["source_saved_policy_hash"] == fnv(lives[cohort["source"]]) and rows[0]["continuation_saved_policy_hash"] == fnv(lives["student"]), "actual source/continuation lives")
            need(all(row["architect"]["consequence"]["learned"] == 0 for row in steps), "frozen source history")
            counts["paired_host_processes"] += 1
            counts["host_updates"] += len(steps)
            initial = identity(raw(prefix + ".initial.bin"))["sha256"]
            body_key = cohort["body"], cohort["seed"]
            need(initial_bodies.setdefault(body_key, initial) == initial, "matched source body initialization")
        select = lambda rows: [row for row in rows if row["type"] in ("rollout_step", "evaluation")]
        need(select(traces["control"]) == select(traces["diagnostic"]), "diagnostics preserve actual host")
        host_traces[key] = traces["control"]
        rows = traces["diagnostic"]
        forks = [row for row in rows if row["type"] == "fork"]
        need([row["step"] for row in forks] == protocol["checkpoints_before_updates"], "fixed source worlds")
        checks = [row for row in rows if row["type"] == "continuation_check"]
        need(len(checks) == 128 and all(row["exact"] for row in checks), "complete natural continuation checks")
        counts["source_continuation_transitions"] += len(checks)
        branches = [row for row in rows if row["type"] == "branch_step"]
        need(all(row["architect"]["consequence"]["learned"] == 0 for row in branches), "frozen branch history")
        counts["branch_updates"] += len(branches)
        for fork in forks:
            prefix = cohort["prefix"] + "-diagnostic" + f".fork-{fork['step']}"
            source = life(raw(prefix + ".policy.bin").read_bytes())
            graft = life(raw(prefix + ".student.policy.bin").read_bytes())
            need(fnv(source) == fork["policy_hash"] and fnv(graft) == fork["student_policy_hash"], "source/graft hashes")
            need(source[:140] == graft[:140] and source[792:] == graft[792:], "graft preserves actual history/config/counters")
            need(graft[140:792] == lives["student"][140:792], "graft carries sealed student weights")
            continuation = "policy" if cohort["source"] == "student" else "student"
            need(fork["continuation_aliases"] == ({"student": "policy"} if continuation == "policy" else {}), "explicit continuation alias")
            outcome = [row for row in rows if row["type"] == "comparison" and row["checkpoint"] == fork["step"] and row["continuation"] == continuation and row["horizon"] == 16]
            need(len(outcome) == 3 and {row["action"] for row in outcome} == set(ACTIONS), "three measured H16 outcomes")
            arm = {row["action"]: row for row in outcome}
            worlds[(*key, fork["step"])] = (fork, arm, prefix + ".policy.bin")
            counts["acquisition_worlds" if key[0] == "development" else "evaluation_worlds"] += 1
    samples_by_group = {}
    for split, source in (("development", "parent"), ("development", "student"), ("evaluation", "student")):
        name = f"{split}-{source}.samples"
        samples = read_json(raw(name + ".json"))
        need(len(samples) == 32 and raw(name + ".bin").read_bytes() == table_bytes(samples), "exact canonical sample table")
        keys = []
        for sample in samples:
            key = split, source, sample["body"], sample["seed"], sample["checkpoint"]
            fork, arms, source_path = worlds[key]
            keys.append(key)
            need(sample["source"] == source and sample["continuation"] == "student", "sample source and shared continuation")
            need(sample["state_hash"] == fork["state_hash"] and sample["policy_hash"] == fork["policy_hash"] and sample["source_policy"] == source_path, "sample history identity")
            need(sample["observation"] == fork["observation"] and sample["features"] == fork["features"], "captured sample features")
            need(sample["arms"] == arms and list(map(f32, sample["future_loss"])) == [f32(arms[a]["future_after"]) for a in ACTIONS], "raw action/outcome association")
        need(len(set(keys)) == 32 and keys == sorted(keys, key=lambda key: (protocol["body_order"].index(key[2]), key[3], key[4])), "unique fixed sample order")
        samples_by_group[split, source] = samples
    fit_summaries = {}
    for label, spec in protocol["students"].items():
        samples = samples_by_group["development", spec["source"]]
        need(read_json(raw(label + ".samples.json")) == samples and raw(label + ".samples.bin").read_bytes() == table_bytes(samples), "matched refit source table")
        rows = events(label + ".fit.jsonl")
        need(rows[0]["type"] == "fit_run" and rows[-1]["type"] == "fit_summary" and len(rows) == 16386, "complete fit trace")
        initial = bytearray(lives["student"])
        initial[8:72] = spec["life_id"].encode().ljust(64, b"\0")
        need(rows[0]["initial_hash"] == fnv(initial), "same student initialization with named identity")
        need(lives[label][:140] == initial[:140] and lives[label][792:] == initial[792:], "fit changes weights only")
        prior = fnv(initial)
        for index, row in enumerate(rows[1:-1]):
            sample = samples[index % 32]
            losses = list(map(f32, sample["future_loss"]))
            scale = max(1e-6 * (abs(losses[0]) + 1), *(abs(losses[0] - loss) for loss in losses))
            target = [f32((losses[0] - loss) / scale) for loss in losses]
            need(row["type"] == "fit" and row["fit_step"] == index + 1 and row["epoch"] == index // 32 + 1 and row["sample_index"] == index % 32, "fixed fitting chronology")
            need((row["body"], row["seed"], row["checkpoint"], row["state_hash"], row["source_policy_hash"]) == (sample["body"], sample["seed"], sample["checkpoint"], sample["state_hash"], sample["policy_hash"]), "fit attached to actual source")
            need(list(map(f32, row["future_loss"])) == losses and list(map(f32, row["target"])) == target and row["scale"] == scale, "independent F32 H16 targets")
            need(row["hash_before"] == prior and row["online_decisions"] == row["online_updates"] == 0, "fit hash chain and unchanged online chronology")
            prior = row["hash_after"]
        need(prior == rows[-1]["final_hash"] == fnv(lives[label]), "fit ends in sealed life")
        counts["fit_updates"] += len(rows) - 2
        fit_summaries[label] = {"source": spec["source"], "samples": 32, "fits": len(rows) - 2, "final_hash": prior}
    common, fresh_readouts = [], []
    for (split, source), samples in samples_by_group.items():
        lookup = {(s["body"], s["seed"], s["checkpoint"]): s for s in samples}
        for label in protocol["evaluation"]["readouts"]:
            rows = events(f"{split}-{source}-{label}.readout.jsonl")
            need(len(rows) == 32 and len({(r["body"], r["seed"], r["checkpoint"]) for r in rows}) == 32, "complete common-state readouts")
            for row in rows:
                sample = lookup[row["body"], row["seed"], row["checkpoint"]]
                loss = list(map(f32, sample["future_loss"]))
                selected = row["action"] - 1
                need(row["label"] == label and row["state_hash"] == sample["state_hash"] and row["source_policy_hash"] == sample["policy_hash"], "readout source identity")
                need(row["regret"] == loss[selected] - min(loss) and row["advantage"] == loss[0] - loss[selected] and row["optimal"] == (loss[selected] == min(loss)), "independent regret/advantage/ties")
                need(f32(row["future_loss"]) == loss[selected] and f32(row["hold_loss"]) == loss[0], "readout actual consequence")
                if label in ACTIONS:
                    need(selected == ACTIONS.index(label) and row["forced"], "fixed reference action")
                else:
                    need(row["model_hash"] == fnv(lives[label]) and not row["forced"] and selected == max(range(3), key=lambda k: row["scores"][k]), "sealed readout selection")
                if split == "evaluation":
                    fresh_readouts.append({**row, "tied_optimal_actions": [ACTIONS[i] for i, value in enumerate(loss) if value == min(loss)]})
            for body in protocol["body_order"]:
                group = [r for r in rows if r["body"] == body]
                common.append({"split": split, "source": source, "body": body, "label": label, "states": len(group), "optimal": sum(r["optimal"] for r in group), "mean_regret": sum(r["regret"] for r in group) / len(group), "mean_advantage": sum(r["advantage"] for r in group) / len(group), **{a + "_choices": sum(r["action"] == i + 1 for r in group) for i, a in enumerate(ACTIONS)}})
            counts[split + "_readouts"] += len(rows)
    curves, deployment, rollout_rows = [], [], {}
    expected_runs = {(body, seed, arm) for body in protocol["body_order"] for seed in [1013, 1217] for arm in protocol["closed_loop"]["arms"]}
    primary = [r for r in anchors["rollouts"] if r["resume_step"] == 0]
    need(len(primary) == 24 and {(r["body"], r["seed"], r["arm"]) for r in primary} == expected_runs, "complete deployment arms")
    for record in anchors["rollouts"]:
        rows = events(record["prefix"] + ".jsonl")
        updates = [r for r in rows if r["type"] == "rollout_step"]
        evals = [r for r in rows if r["type"] == "evaluation"]
        need([r["step"] for r in updates] == list(range(1, 513)) and [r["step"] for r in evals] == [0, 128, 256, 384, 512], "complete deployment curve")
        need(rows[-1]["type"] == "rollout_summary" and all(math.isfinite(r["heldout_loss"]) for r in evals), "finite completed deployment")
        if record["arm"] in lives:
            need(rows[0]["sealed_policy_hash"] == fnv(lives[record["arm"]]) and all(r["architect"]["consequence"]["learned"] == 0 for r in updates), "frozen sealed deployment")
            final = life(raw(record["prefix"] + ".policy.final.bin").read_bytes())
            need(final[140:792] == lives[record["arm"]][140:792], "deployment keeps all163weights")
        counts["deployment_updates"] += len(updates)
        counts["resume_processes" if record["resume_step"] else "primary_deployment_processes"] += 1
        if record["resume_step"]:
            baseline = rollout_rows[record["body"], record["seed"], record["arm"]]
            need([r for r in rows if r["type"] in ("rollout_step", "evaluation")] == [r for r in baseline if r["type"] in ("rollout_step", "evaluation")], "exact saved-life continuation")
            continue
        key = record["body"], record["seed"], record["arm"]
        rollout_rows[key] = rows
        curves.extend({"body": key[0], "seed": key[1], "arm": key[2], "step": r["step"], "heldout_loss": r["heldout_loss"]} for r in evals)
    for (body, seed, arm), rows in rollout_rows.items():
        updates = [r for r in rows if r["type"] == "rollout_step"]
        student_rows = rollout_rows[body, seed, "student"]
        student_steps = [r for r in student_rows if r["type"] == "rollout_step"]
        different = [(a, b) for a, b in zip(updates, student_steps) if a["action"] != b["action"]]
        first, reference = different[0] if different else (None, None)
        base = {"body": body, "seed": seed, "arm": arm, "final_heldout": [r for r in rows if r["type"] == "evaluation"][-1]["heldout_loss"], "first_divergence_vs_student": first["step"] if first else None, "first_action": first["action"] if first else None, "student_action": reference["action"] if first else None, "first_before_loss_equal": first["before"] == reference["before"] if first else None, "first_pre_chuck_equal": first["pre_chuck"] == reference["pre_chuck"] if first else None, "first_features_equal": first["architect"]["features"] == reference["architect"]["features"] if first and "architect" in first else None, "first_observation_equal": first["architect"]["observation"] == reference["architect"]["observation"] if first and "architect" in first else None, "first_dampening_floor": next((r["step"] for r in updates if f32(r["post_chuck"]["dampen"]) == f32(.3)), None)}
        for limit, label in ((128, "early128"), (512, "all512")):
            for index, name in enumerate(("legacy", *ACTIONS)):
                base[label + "_" + name] = sum(r["action"] == index for r in updates[:limit])
        deployment.append(base)
        if arm == "student":
            source = host_traces["evaluation", "student", body, seed]
            need([r for r in source if r["type"] == "rollout_step"] == updates[:256], "source/deployment exact first256steps")
    pairs = []
    for body in protocol["body_order"]:
        for seed in [1013, 1217]:
            by_arm = {r["arm"]: r for r in deployment if r["body"] == body and r["seed"] == seed}
            pairs.append({"body": body, "seed": seed, "student": by_arm["student"]["final_heldout"], "refit_parent": by_arm["refit-parent"]["final_heldout"], "refit_self": by_arm["refit-self"]["final_heldout"], "self_minus_parent": by_arm["refit-self"]["final_heldout"] - by_arm["refit-parent"]["final_heldout"], "self_minus_student": by_arm["refit-self"]["final_heldout"] - by_arm["student"]["final_heldout"], "parent_minus_student": by_arm["refit-parent"]["final_heldout"] - by_arm["student"]["final_heldout"]})
    counts["executed_branches"] = counts["branch_updates"] // 16
    counts["readouts"] = counts["development_readouts"] + counts["evaluation_readouts"]
    counts["all_body_updates"] = counts["branch_updates"] + counts["host_updates"] + counts["deployment_updates"]
    need(dict(counts) == protocol["fixed_counts"] == result["executed_counts"], "independently derived fixed counts")
    for name, rows in (("common_state.csv", common), ("heldout_curves.csv", curves), ("deployment.csv", deployment), ("paired_final.csv", pairs)):
        csv_file(out / name, rows)
    report = {"status": "PASS", "checks": CHECKS, "source_commit": expected_source, "results_identity": identity(run / "results.json"), "protocol_sha256": PROTOCOL_SHA, "script_sha256": digest(Path(__file__)), "counts": dict(counts), "seal_chronology": timeline, "fits": fit_summaries, "fresh_common_state": [r for r in common if r["split"] == "evaluation"], "paired_final": pairs, "deployment": deployment, "authenticated_files": authenticated, "csv_identities": {p.name: identity(p) for p in out.glob("*.csv")}}
    (out / "audit.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"status": "PASS", "checks": CHECKS, "counts": dict(counts), "paired_final": pairs}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--results-sha256", required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    try:
        audit(args.run, args.out, args.source_commit, args.results_sha256)
    except Exception:
        (args.out / "failure.json").write_text(json.dumps({"status": "FAIL", "checks": CHECKS, "script_sha256": digest(Path(__file__)), "traceback": traceback.format_exc()}, indent=2) + "\n")
        raise


if __name__ == "__main__":
    main()
