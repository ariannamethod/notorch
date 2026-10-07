#!/usr/bin/env python3
"""Diagnose retained SPA future-credit outcomes without generation or fitting.

Both original splits are exposed diagnostic data here: training seeds 42/73 and
the previously evaluated seeds 101/211. The script reads the pinned parents,
rebuilds their source associations, and writes only the requested JSON report.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import gzip
import hashlib
import importlib.util
import io
import json
import math
from pathlib import Path
import statistics
import sys

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[3]
FUTURE = ROOT / "experiments/spa_agent/future"
SCENARIOS = ROOT / "experiments/spa_agent/scenarios"
TOLERANCE = 1e-7
PINS = {
    "experiments/spa_agent/future/protocol.json": "134460b76ff7fecc9934eea8187d8aac4f6d0154921db5989de86cc482ee1861",
    "experiments/spa_agent/future/receipts.json": "b2caeead7ca5c3a02234f60a37d28e81dec2f6feb6f694a0d069bbf79d4c8dbf",
    "experiments/spa_agent/future/raw_manifest.json": "7c42b5039fffbd97e9d553eb06d74474e7bcc2d8671021efcb3bcb88e72fd86c",
    "experiments/spa_agent/future/result_audit.json": "4e9024918b0aa7535e13f02d2ac076e8f495246dd7ed8b93cb304b8879998eba",
    "experiments/spa_agent/future/run.py": "6131057ffe1f25c42f4888af644012a91f21ad6adfb93b32f5be5ada223a7df1",
    "experiments/spa_agent/scenarios/protocol.json": "25aeed3248bcde9920a4a4eed29fd5ef06e5373da670c26fbb87696d819edcf4",
    "experiments/spa_agent/scenarios/raw_traces.jsonl.gz": "dcee2f733f8a7a4063a44f0654b574dad55ce6b57ea258ee81a7582d39370d22",
    "experiments/spa_agent/scenarios/run.py": "3da78ceb56bc50a0bb341db029373c9941003ad0a8a60ed422d2db091ef29195",
    "experiments/spa_agent/run.py": "4dc7152c421cd5d1812669f117673ca721a6800d3ee15020c6581d4cfeda85f7",
    "tests/test_spa_trace_io.py": "f4cc9893b03d641653a5ca8e4efb7ca37e90fa0717d9b98db2f3980df18c288d",
}


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def mean(values):
    return statistics.mean(values) if values else 0.0


def signs(values):
    return {"positive": sum(x > TOLERANCE for x in values),
            "negative": sum(x < -TOLERANCE for x in values),
            "neutral": sum(abs(x) <= TOLERANCE for x in values)}


def distribution(values):
    require(bool(values), "empty distribution")
    return {"count": len(values), "mean": mean(values), "minimum": min(values),
            "maximum": max(values), "population_stdev": statistics.pstdev(values)}


def variance_by_target(rows, field):
    values = [row[field] for row in rows]
    center = mean(values)
    groups = defaultdict(list)
    for row in rows:
        groups[row["target"]].append(row[field])
    total = sum((x-center)**2 for x in values)
    between = sum(len(v)*(mean(v)-center)**2 for v in groups.values())
    within = sum(sum((x-mean(v))**2 for x in v) for v in groups.values())
    require(abs(total-between-within) <= 1e-12, "variance decomposition")
    return {"population_stdev": statistics.pstdev(values),
            "fraction_variance_between_targets": between/total if total else 0,
            "within_target_population_stdev": math.sqrt(within/len(values))}


def load_parents():
    for name, expected in PINS.items():
        require(digest((ROOT/name).read_bytes()) == expected, "pinned parent changed: " + name)
    spec = importlib.util.spec_from_file_location("spa_replicate_parent", FUTURE/"run.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    receipts = json.loads((FUTURE/"receipts.json").read_text())
    manifest = json.loads((FUTURE/"raw_manifest.json").read_text())
    chunks = []
    for part in manifest["parts"]:
        require(Path(part["path"]).name == part["path"], "part path")
        raw = (FUTURE/part["path"]).read_bytes()
        require(len(raw) == part["bytes"] and digest(raw) == part["sha256"], "part identity")
        chunks.append(raw)
    compressed = b"".join(chunks)
    require(len(compressed) == manifest["archive"]["bytes"]
            and digest(compressed) == manifest["archive"]["sha256"], "complete archive identity")
    streams = defaultdict(bytearray)
    counts = Counter()
    with gzip.GzipFile(fileobj=io.BytesIO(compressed)) as archive:
        for line in archive:
            require(line.endswith(b"\n"), "truncated archive")
            row = json.loads(line, object_pairs_hook=module.SCENARIOS.unique_object)
            require(set(row) == {"seed", "stream", "raw"}, "archive envelope")
            require(row["raw"].endswith("\n"), "truncated raw record")
            key = row["seed"], row["stream"]
            counts[key] += 1
            if key != (0, "fit"):
                streams[key].extend(row["raw"].encode())
    require(counts == {(0,"fit"):73786, (0,"training_readout"):723,
        (0,"evaluation_readout"):723, (101,"ordinary_off"):128,
        (101,"ordinary_on"):128, (101,"scenarios"):253,
        (211,"ordinary_off"):128, (211,"ordinary_on"):128, (211,"scenarios"):253},
        "complete nine-stream coverage")
    scenario_protocol = json.loads((SCENARIOS/"protocol.json").read_text())
    train_streams = module.archive_streams(SCENARIOS/"raw_traces.jsonl.gz",
        PINS["experiments/spa_agent/scenarios/raw_traces.jsonl.gz"])
    samples = {
        "training": module.samples_from_streams(train_streams, [42,73], scenario_protocol),
        "evaluation_exposed": module.samples_from_streams(
            {key: bytes(value) for key, value in streams.items() if key[0] != 0},
            [101,211], scenario_protocol),
    }
    readouts = {}
    for split, stream in (("training","training_readout"), ("evaluation_exposed","evaluation_readout")):
        raw_rows, _ = module.parse_raw(bytes(streams[0, stream]))
        rows = [row for row in raw_rows if row["type"] == "readout" and row["horizon"] == 4]
        require(len(rows) == 336, "H4 readout count")
        index = {}
        for row in rows:
            source = samples[split][row["source"]["index"]]
            require(row["source"] == module.source_identity(source), "readout/source association")
            require(row["model_hash"] == receipts["executions"][0]["fit"]["life_hashes"].get(row["label"], receipts["executions"][0]["fit"]["life_hashes"]["initial"]), "sealed policy identity")
            require(row["scores_before"] == row["scores_after"] and row["life_unchanged"], "frozen readout")
            selected = next(o for o in source["outcomes"][4] if o["action"] == row["action"])
            require(selected["raw_sha256"] == row["selected_outcome_sha256"], "selected branch identity")
            key = row["source"]["index"], row["label"]
            require(key not in index, "duplicate readout")
            index[key] = row
        require(set(index) == {(i, label) for i in range(48) for label in module.LABELS}, "readout coverage")
        readouts[split] = index
    return receipts, samples, readouts, manifest["archive"]


def policy_summary(samples, readouts, label):
    counts = Counter()
    regret = missed = harm = total_advantage = 0.0
    for i, sample in enumerate(samples):
        rewards = {o["action"]["kind"]: o["reward"] for o in sample["outcomes"][4]}
        chosen = readouts[i,label]["action"]["kind"]
        advantage = rewards[chosen]-rewards[0]
        best = max(rewards.values())-rewards[0]
        counts["states_with_profitable_action"] += best > TOLERANCE
        counts["interventions"] += chosen != 0
        counts["beneficial_interventions"] += chosen != 0 and advantage > TOLERANCE
        counts["harmful_interventions"] += chosen != 0 and advantage < -TOLERANCE
        counts["neutral_interventions"] += chosen != 0 and abs(advantage) <= TOLERANCE
        counts["kept_profitable_states"] += chosen == 0 and best > TOLERANCE
        counts["harmful_choice_despite_profitable_alternative"] += advantage < -TOLERANCE and best > TOLERANCE
        counts["oracle_choices"] += best-advantage <= TOLERANCE
        regret += best-advantage
        missed += best-max(advantage, 0)
        harm += max(-advantage, 0)
        total_advantage += advantage
    require(abs(regret-missed-harm) < 1e-12, "regret decomposition")
    require(counts["interventions"] == counts["beneficial_interventions"] + counts["harmful_interventions"] + counts["neutral_interventions"] == 48-sum(readouts[i,label]["action"]["kind"] == 0 for i in range(48)), "intervention count decomposition")
    return {"label": label, **counts, "mean_advantage_over_keep": total_advantage/48,
            "mean_oracle_regret": regret/48, "missed_opportunity_regret_component": missed/48,
            "harmful_intervention_regret_component": harm/48,
            "mean_h4_huber_loss": mean([readouts[i,label]["loss_before"] for i in range(48)])}


def split_analysis(split, samples, readouts, vocabulary):
    opportunities = []
    for i, sample in enumerate(samples):
        outcomes = {o["action"]["kind"]:o for o in sample["outcomes"][4]}
        original_chain = next(o for o in sample["outcomes"][0] if o["action"]["kind"] == 0)["chain"]
        predicted = readouts[i,"h4"]["scores_before"]
        chosen = readouts[i,"h4"]["action"]["kind"]
        for kind, outcome in outcomes.items():
            if kind == 0:
                continue
            source_sentence = original_chain[outcome["action"]["source"]]
            suffix = source_sentence["tokens"][-3:]
            opportunities.append({"split": split, "seed": sample["seed"],
                "snapshot": sample["snapshot"], "sample_index": i, "target": sample["target"],
                "action_kind": kind, "action_mask": sample["action_mask"],
                "source_index": outcome["action"]["source"],
                "source_suffix_tokens": suffix,
                "source_suffix_text": "".join(vocabulary[t] for t in suffix),
                "source_sentence_length": source_sentence["length"],
                "source_sentence_terminated": source_sentence["terminated"],
                "source_sentence_tokens": source_sentence["tokens"],
                "target_sentence_tokens": original_chain[sample["target"]]["tokens"],
                "host_rng": sample["host_rng"],
                "measured_advantage": outcome["reward"]-outcomes[0]["reward"],
                "predicted_advantage": predicted[kind]-predicted[0],
                "h4_selected": chosen == kind,
                "keep_reward": outcomes[0]["reward"], "action_reward": outcome["reward"],
                "snapshot_sha256": sample["snapshot_raw_sha256"],
                "action_outcome_sha256": outcome["raw_sha256"],
                "keep_outcome_sha256": outcomes[0]["raw_sha256"]})
    require(len(opportunities) == 72, "non-KEEP opportunity count")
    by_target = []
    for target in range(4):
        indices = [i for i,s in enumerate(samples) if s["target"] == target]
        row = {"target": target, "states": len(indices),
            "h4_action_counts": dict(Counter(readouts[i,"h4"]["action"]["kind"] for i in indices)), "actions": []}
        for kind in (1,2):
            selected = [o for o in opportunities if o["target"] == target and o["action_kind"] == kind]
            if not selected:
                continue
            measured = [o["measured_advantage"] for o in selected]
            predictions = [o["predicted_advantage"] for o in selected]
            row["actions"].append({"kind": kind, "signs": signs(measured),
                "measured_advantage": distribution(measured),
                "predicted_advantage": distribution(predictions),
                "prediction_rmse": math.sqrt(mean([(a-b)**2 for a,b in zip(measured,predictions)]))})
        by_target.append(row)
    by_mask = []
    for mask in (3,5,7):
        group = [o for o in opportunities if o["action_mask"] == mask]
        indices = [i for i,s in enumerate(samples) if s["action_mask"] == mask]
        by_mask.append({"action_mask": mask, "states": len(indices),
            "non_keep_outcomes": len(group), "signs": signs([o["measured_advantage"] for o in group]),
            "h4_action_counts": dict(Counter(readouts[i,"h4"]["action"]["kind"] for i in indices))})
    selected = [o for o in opportunities if o["h4_selected"]]
    target3 = [s for s in samples if s["target"] == 3]
    policies = [policy_summary(samples, readouts, label) for label in ("initial","h4","h0","shuffled_h4","keep","left","right")]
    feature_ranges = [{"index": i, **distribution([s["features"][i] for s in samples])} for i in range(29)]
    return {"states": len(samples), "seeds": sorted({s["seed"] for s in samples}),
        "policies": policies, "non_keep_outcome_signs": signs([o["measured_advantage"] for o in opportunities]),
        "targets": by_target, "boundary_masks": by_mask,
        "selected_prediction_margins": distribution([o["predicted_advantage"] for o in selected]),
        "selected_measured_advantages": distribution([o["measured_advantage"] for o in selected]),
        "position_variance": [{"action_kind": kind,
            "measured": variance_by_target([o for o in opportunities if o["action_kind"] == kind], "measured_advantage"),
            "predicted": variance_by_target([o for o in opportunities if o["action_kind"] == kind], "predicted_advantage")}
            for kind in (1,2)],
        "feature_ranges": feature_ranges,
        "constant_feature_indices": [row["index"] for row in feature_ranges if row["minimum"] == row["maximum"]],
        "unique_feature_vectors": len({tuple(s["features"]) for s in samples}),
        "unique_host_rngs": len({s["host_rng"] for s in samples}),
        "execution_context": {"non_keep_actions": len(opportunities),
            "unique_three_token_suffixes": len({tuple(o["source_suffix_tokens"]) for o in opportunities}),
            "unique_full_source_sentences": len({tuple(o["source_sentence_tokens"]) for o in opportunities}),
            "source_terminated_count": sum(o["source_sentence_terminated"] for o in opportunities),
            "source_length_counts": dict(Counter(o["source_sentence_length"] for o in opportunities))},
        "target3_identical_h4_chains": sum(s["outcomes"][4][0]["chain"] == s["outcomes"][4][1]["chain"] for s in target3),
        "target3_identical_h4_raw_axes": sum(s["outcomes"][4][0]["after"] == s["outcomes"][4][1]["after"] for s in target3),
        "opportunities": opportunities}


def report():
    receipts, samples, readouts, archive = load_parents()
    vocabulary = receipts["metadata"]["inputs"]["vocabulary.u32"]["characters"]
    require(len(vocabulary) == 94, "frozen vocabulary")
    splits = {name: split_analysis(name, ss, readouts[name], vocabulary) for name,ss in samples.items()}
    training = splits["training"]["opportunities"]
    evaluation = splits["evaluation_exposed"]["opportunities"]
    all_opportunities = training+evaluation
    train_suffixes = {tuple(o["source_suffix_tokens"]) for o in training}
    grouped = defaultdict(list)
    for o in all_opportunities:
        grouped[o["target"],o["action_kind"],tuple(o["source_suffix_tokens"])].append(o)
    duplicate = [rows for rows in grouped.values() if len(rows) > 1]
    mixed = [rows for rows in duplicate if signs([o["measured_advantage"] for o in rows])["positive"]
             and signs([o["measured_advantage"] for o in rows])["negative"]]
    full_context_groups = {(o["target"],o["action_kind"],tuple(o["source_sentence_tokens"])) for o in all_opportunities}
    outlier = next(o for o in training if (o["seed"],o["snapshot"],o["action_kind"]) == (42,11,1))
    paired = next(o for o in training if (o["seed"],o["snapshot"],o["action_kind"]) == (42,11,2))
    peers = [o for o in training if o["target"] == 1 and o["action_kind"] == 1]
    require(outlier["measured_advantage"] == max(o["measured_advantage"] for o in peers), "named largest target1 LEFT realization")
    for split, summary in splits.items():
        original_name = "evaluation" if split == "evaluation_exposed" else split
        old_rows = receipts["executions"][0][original_name]["summaries"]
        for policy in summary["policies"]:
            original = next(row for row in old_rows if row["group"] == "combined" and row["horizon"] == 4 and row["label"] == policy["label"])
            require(abs(policy["mean_advantage_over_keep"]-original["mean_advantage_over_keep"]) < 1e-12
                    and abs(policy["mean_oracle_regret"]-original["mean_oracle_regret"]) < 1e-12,
                    "diagnosis must reproduce registered primary metrics")
    return {"schema": 1, "status": "PASS", "name": "Retained future-credit failure diagnosis before paired stochastic replicates",
        "command": "python3 experiments/spa_agent/replicates/diagnosis.py --json experiments/spa_agent/replicates/diagnosis.json",
        "script_sha256": digest(Path(__file__).read_bytes()),
        "parents_sha256": PINS, "future_archive": archive,
        "native_source_sha256_from_pinned_parent": {name: value for name,value in receipts["metadata"]["source_files_sha256"].items() if name in ("spa_agent.c","spa_agent.h","examples/spa_agent_demo.c","examples/spa_agent_scenarios.h")},
        "data_exposure": {"states": 96, "training_seeds": [42,73], "previous_evaluation_seeds_now_exposed": [101,211],
            "purpose": "Descriptive diagnosis and design of the next frozen control. Both seed sets are exposed diagnostic data.",
            "model_training_runs": 0, "model_generation_runs": 0, "old_artifacts_modified": False},
        "definitions": {"advantage": "Recorded reward(action,H4) minus recorded reward(KEEP,H4) at the identical captured state and paired stream.",
            "profitable": "advantage > 1e-7; harmful advantage < -1e-7; neutral otherwise.",
            "regret_decomposition": "best_advantage - selected_advantage = [best_advantage - max(selected_advantage,0)] + max(-selected_advantage,0)",
            "variance": "Population variance of the finite retained outcomes. Between-target sum of squares divided by total; within-target variation retains state and RNG variation together.",
            "prediction_margin": "H4 policy action-head score minus its KEEP score, from the sealed readout on the original 29 features.",
            "loss": "Original mean Huber over valid heads; readout at H4, no updates."},
        "splits": splits,
        "single_state_influence": {"source": outlier, "same_state_right": paired,
            "target1_left_outcomes": len(peers), "target1_left_sum_advantage": sum(o["measured_advantage"] for o in peers),
            "target1_left_mean_advantage": mean([o["measured_advantage"] for o in peers]),
            "other_11_left_mean_advantage": mean([o["measured_advantage"] for o in peers if o is not outlier]),
            "operation": "Descriptive decomposition of the existing 12 outcomes. No row is removed from training or reported primary measurements."},
        "context_coverage": {"evaluation_action_contexts": len(evaluation),
            "evaluation_suffixes_seen_in_training": sum(tuple(o["source_suffix_tokens"]) in train_suffixes for o in evaluation),
            "distinct_suffix_overlap": len(train_suffixes & {tuple(o["source_suffix_tokens"]) for o in evaluation}),
            "same_target_action_suffix_groups": len(grouped), "duplicate_groups": len(duplicate),
            "mixed_sign_duplicate_groups": len(mixed),
            "mixed_sign_group_sources": [[{"seed":o["seed"],"snapshot":o["snapshot"],"target":o["target"],"kind":o["action_kind"],"suffix":o["source_suffix_text"],"advantage":o["measured_advantage"],"host_rng":o["host_rng"]} for o in rows] for rows in mixed],
            "distinct_target_action_full_source_groups": len(full_context_groups),
            "available_within_identical_state_rng_replications": 1},
        "representation_execution_boundary": {
            "native_observation": "spa_agent.c nt_spa_agent_perceive projects only the current target sentence's 128-dimensional embedding into four bucket means followed by tanh. Its neighbors enter through left/right cosine, global connectedness/coherence/novelty summaries; features() appends position, masks and ten memory values.",
            "actual_initial_reseed_input": "examples/spa_agent_demo.c generate and examples/spa_agent_scenarios.h scenario_execute pass the selected neighbor's last min(3,length) token IDs to the frozen body, plus an independent 64-bit host RNG. Every retained source has at least three tokens.",
            "absent_explicit_inputs": ["selected source suffix token identities", "selected source sentence embedding", "source sentence length/termination", "host initial/future RNG draws"],
            "constant_features": ["phase_lock=0.5", "reseed_count=0", "temperature/16=0.0500000007"],
            "target3_h4": "Continuation target0 reads sentence1, then target1 reads0, target2 reads1, target3 reads2. The initial target3 intervention is overwritten before any continuation reads it. All24 retained target3 H4 final chains and axes are identical across actions; cost remains action-dependent."},
        "next_control_question": "At identical captured states, are the state/action mean paired H4 advantages reproducible across independent host RNG replicates, and does fitting those means change useful action selection relative to fitting the original single realization?",
        "next_control_constraints": ["Keep the same 267-parameter model, captured 29 features, valid-action masks, reward, H4 continuation and optimizer budget.",
            "Retain r0 exactly; pair each new replicate's initial and each future-hop RNG across all valid actions.",
            "Compute each replicate's reward and KEEP-relative advantage before averaging; retain all raw axes and costs.",
            "Use exposed101/211 for diagnosis only. Freeze policies and protocol before any new evaluation-state generation.",
            "Measure within-state paired variance/sign stability first; existing context-plus-RNG heterogeneity is not a within-state variance estimate."],
        "checks": {"all_parent_hashes": True, "complete_archive": True, "source_and_selected_action_joins": True,
            "registered_primary_metrics_reproduced": True, "regret_and_variance_decompositions": True}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", required=True, type=Path)
    args = parser.parse_args()
    result = report()
    # Check anchors again after analysis; never change a parent artifact.
    for name, expected in PINS.items():
        require(digest((ROOT/name).read_bytes()) == expected, "parent changed during diagnosis: " + name)
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print("PASS retained96 diagnosis; generation=0 training=0")
    for split, value in result["splits"].items():
        policy = next(p for p in value["policies"] if p["label"] == "h4")
        print(split, "beneficial", policy["beneficial_interventions"], "harmful", policy["harmful_interventions"],
              "kept_profitable", policy["kept_profitable_states"], "mean_advantage", policy["mean_advantage_over_keep"])
    print("receipt_sha256", digest(args.json.read_bytes()))


if __name__ == "__main__":
    main()
