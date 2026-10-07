#!/usr/bin/env python3
"""Run preregistered sentence-action forks from an immutable native-C snapshot."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
PROTOCOL = HERE / "protocol.json"
FROZEN_PROTOCOL_SHA256 = "25aeed3248bcde9920a4a4eed29fd5ef06e5373da670c26fbb87696d819edcf4"
SPEC = importlib.util.spec_from_file_location("spa_parent_run", HERE.parent / "run.py")
V1 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(V1)
AXES, WEIGHTS = V1.AXES, V1.WEIGHTS
NAMES = ("KEEP", "RESEED_LEFT", "RESEED_RIGHT")
MASK64 = (1 << 64) - 1
INCREMENT = 0x9E3779B97F4A7C15
SOURCE_FILES = tuple(dict.fromkeys((*V1.SOURCE_FILES,
    "examples/spa_agent_scenarios.h", "experiments/spa_agent/scenarios/run.py",
    "experiments/spa_agent/scenarios/protocol.json", "tests/test_spa_scenarios.py")))


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def stream(seed: int, domain: int, episode: int, slot: int) -> int:
    value = (domain ^ seed ^ (episode << 32) ^ (slot << 48)) + INCREMENT
    value &= MASK64
    value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & MASK64
    value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & MASK64
    return value ^ (value >> 31)


def hex_rng(value: str) -> int:
    require(isinstance(value, str) and len(value) == 16, "invalid hexadecimal RNG")
    result = int(value, 16)
    require(0 <= result <= MASK64, "RNG out of range")
    return result


def finite_tree(value: object) -> None:
    if isinstance(value, float):
        require(math.isfinite(value), "non-finite record value")
    elif isinstance(value, dict):
        for child in value.values():
            finite_tree(child)
    elif isinstance(value, list):
        for child in value:
            finite_tree(child)


def unique_object(items: list[tuple]) -> dict:
    result = {}
    for key, value in items:
        require(key not in result, f"duplicate JSON key {key}")
        result[key] = value
    return result


def read_events(path: Path) -> list[dict]:
    rows = [json.loads(line, object_pairs_hook=unique_object) for line in path.read_text().splitlines()]
    for row in rows:
        require(isinstance(row, dict), "record must be an object")
        finite_tree(row)
    return rows


def check_axes(value: dict) -> None:
    require(set(value) == set(AXES), "seven metric axes required")
    require(all(isinstance(value[a], (float, int)) and math.isfinite(value[a]) and 0 <= value[a] <= 1
                for a in AXES), "metric axis outside [0,1]")


def repetition(sentence: dict) -> float:
    tokens = sentence["tokens"]
    triples = list(zip(tokens, tokens[1:], tokens[2:]))
    return (len(triples) - len(set(triples))) / len(triples) if triples else 0.0


def check_chain(chain: list, count: int, cap: int) -> None:
    require(isinstance(chain, list) and len(chain) == count, "wrong chain size")
    for sentence in chain:
        require(type(sentence["length"]) is int and 0 < sentence["length"] <= cap, "invalid sentence length")
        require(sentence["length"] == len(sentence["tokens"]), "token count differs from sentence length")
        require(type(sentence["terminated"]) is bool, "invalid termination flag")
        require(all(type(t) is int and 0 <= t < 94 for t in sentence["tokens"]), "invalid character token")


def valid_actions(target: int, count: int) -> set[int]:
    return {0} | ({1} if target > 0 else set()) | ({2} if target + 1 < count else set())


def check_action(action: dict, target: int, count: int) -> None:
    require(action["kind"] in valid_actions(target, count) and action["target"] == target, "invalid typed action")
    expected = None if action["kind"] == 0 else target + (-1 if action["kind"] == 1 else 1)
    require(action["source"] == expected, "typed action has wrong neighbor")


def reward(before: dict, after: dict, cost: float) -> float:
    return max(-1.0, min(1.0, sum(w * (after[a] - before[a]) for a, w in zip(AXES, WEIGHTS)) - .05 * cost))


def validate_seed(events: list[dict], ordinary: list[dict], protocol: dict) -> dict:
    """Validate complete common-state forks independently of C assertions."""
    for row in events:
        finite_tree(row)
    count = protocol["sentences_per_chain"]
    cap = protocol["interventions"]["generation"]["maximum_generated_characters"]
    episodes, per_episode = protocol["episodes"], protocol["snapshots_per_episode"]
    horizons = protocol["continuation"]["horizons"]
    expected_snapshots = episodes * per_episode
    kinds = {"snapshot", "measurement", "restoration", "selected_initial", "summary"}
    require(all(row.get("type") in kinds for row in events), "unknown scenario record")
    snapshots = [r for r in events if r["type"] == "snapshot"]
    measurements = [r for r in events if r["type"] == "measurement"]
    restorations = [r for r in events if r["type"] == "restoration"]
    confirmations = [r for r in events if r["type"] == "selected_initial"]
    summaries = [r for r in events if r["type"] == "summary"]
    require(len(snapshots) == expected_snapshots and len(restorations) == expected_snapshots,
            "incomplete snapshots or restoration records")
    require(len(summaries) == 1 and events[-1]["type"] == "summary", "one final summary required")
    summary = summaries[0]
    seed = summary["seed"]
    require(seed in protocol["seeds"], "unregistered seed")
    require(all(row["seed"] == seed for row in events), "mixed seeds")
    main_rows = {(r["episode"], r["step"]): r for r in ordinary
                 if r["type"] == "decision" and r["arm"] == "learned"}
    bases = {r["episode"]: r for r in ordinary if r["type"] == "base"}
    body = next(r for r in ordinary if r["type"] == "body")
    require(len(main_rows) == expected_snapshots and len(bases) == episodes, "ordinary join incomplete")
    index = {}
    for row in measurements:
        key = (row["snapshot"], row["action"]["kind"], row["horizon"])
        require(key not in index, "duplicate action/horizon measurement")
        index[key] = row
    restoration_index = {r["snapshot"]: r for r in restorations}
    require(len(restoration_index) == expected_snapshots, "duplicate restoration")
    confirmation_index = {r["snapshot"]: r for r in confirmations}
    require(len(confirmations) == len(confirmation_index) == expected_snapshots, "incomplete selected-action confirmations")
    snapshot_index = {r["snapshot"]: r for r in snapshots}
    require(set(snapshot_index) == set(range(expected_snapshots)), "snapshot coverage")
    used = set()
    forks = 0
    diagnostic_forwards = 0
    results = []
    for sid in range(expected_snapshots):
        snap = snapshot_index[sid]
        ep, step = divmod(sid, per_episode)
        target = (ep + step) % count
        require((snap["episode"], snap["step"], snap["target"]) == (ep, step, target), "snapshot position")
        main = main_rows[(ep, step)]
        original = bases[ep]["chain"] if step == 0 else main_rows[(ep, step - 1)]["chain"]
        require(snap["chain"] == original, "snapshot chain differs from ordinary pre-decision state")
        check_chain(snap["chain"], count, cap)
        counters = [0] * count
        for prior in range(step):
            action = main_rows[(ep, prior)]["action"]
            counters[action["target"]] += action["kind"] != 0
        require(snap["reseeds"] == counters, "snapshot reseed history differs")
        check_axes(snap["before"])
        require(snap["before"] == main["before"], "snapshot measurements differ from ordinary perception")
        require(snap["body_hash"] == body["weights_fnv1a"], "snapshot body changed")
        require(snap["policy_action"] == main["action"], "snapshot preview differs from ordinary action")
        require(snap["features"] == main["features"] and snap["scores"] == main["scores"], "snapshot policy inputs differ")
        require(snap["policy_rng_before"] == main["policy_rng_before"] == snap["agent_rng"], "snapshot policy RNG differs")
        initial_rng = stream(seed, 0x7370615F6163746E, ep, step)
        require(hex_rng(snap["host_rng_before"]) == initial_rng, "initial RNG differs from preregistered stream")
        action_set = valid_actions(target, count)
        forks += len(action_set)
        before = snap["before"]
        action_results = {}
        for action_kind in sorted(action_set):
            previous_continuation = []
            action_results[action_kind] = {}
            for horizon in horizons:
                key = (sid, action_kind, horizon)
                require(key in index, "missing valid action/horizon")
                used.add(key)
                row = index[key]
                require((row["episode"], row["step"], row["target"]) == (ep, step, target), "measurement target drift")
                for field in ("life_hash", "body_hash", "agent_rng", "host_rng_before"):
                    require(row[field] == snap[field], f"fork starts from different {field}")
                require(row["before"] == before, "fork starts from different measurements")
                check_action(row["action"], target, count)
                check_axes(row["after"])
                check_chain(row["chain"], count, cap)
                require(abs(row["after"]["repetition"] - repetition(row["chain"][target])) <= 5e-8,
                        "repetition metric uses wrong target or tokens")
                generated = row["initial_generated"]
                require(type(generated) is int and 0 <= generated <= cap, "initial generation count")
                require((generated == 0) == (action_kind == 0), "initial action/generation mismatch")
                require(row["initial_cost"] == generated / cap, "initial raw cost mismatch")
                require(hex_rng(row["initial_rng_before"]) == initial_rng, "initial fork RNG mismatch")
                require(hex_rng(row["initial_rng_after"]) == (initial_rng + generated * INCREMENT) & MASK64,
                        "initial RNG consumption mismatch")
                require(len(row["continuation"]) == horizon, "continuation horizon mismatch")
                require(row["continuation"][:len(previous_continuation)] == previous_continuation,
                        "later horizon did not continue the earlier branch")
                previous_continuation = row["continuation"]
                branch_counters = counters.copy()
                branch_counters[target] += action_kind != 0
                total = generated
                for hop, future in enumerate(row["continuation"], 1):
                    future_target = (target + hop) % count
                    require(future["hop"] == hop, "future hop order")
                    check_action(future["action"], future_target, count)
                    require(future["action"]["kind"] == (1 if future_target else 2), "future action differs from fixed continuation")
                    rng = stream(seed, 0x7370615F66757472, sid, hop)
                    require(hex_rng(future["rng_before"]) == rng, "future RNG pairing mismatch")
                    chars = future["generated"]
                    require(type(chars) is int and 0 < chars <= cap and future["cost"] == chars / cap, "future raw cost mismatch")
                    require(hex_rng(future["rng_after"]) == (rng + chars * INCREMENT) & MASK64, "future RNG consumption mismatch")
                    total += chars
                    branch_counters[future_target] += 1
                require(row["cumulative_generated"] == total and row["branch_forwards"] == total, "unaccounted generated tokens or forward calls")
                cost = total / (cap * (horizon + 1))
                require(row["cost_denominator"] == cap * (horizon + 1), "horizon cost denominator mismatch")
                require(abs(row["cost_normalized"] - cost) <= 5e-8, "normalized horizon cost mismatch")
                require(row["reseeds"] == branch_counters, "fork reseed counters mismatch")
                require(-1 <= row["reward"] <= 1 and abs(row["reward"] - reward(before, row["after"], cost)) <= 2e-7,
                        "horizon reward differs from frozen raw-axis formula")
                if horizon == 0 and action_kind == 0:
                    require(row["chain"] == original and row["before"] == row["after"] and row["reward"] == 0,
                            "initial KEEP changed original chain/measurement")
                if horizon == 0 and action_kind == main["action"]["kind"]:
                    require(row["chain"] == main["chain"] and row["after"] == main["after"]
                            and row["initial_cost"] == main["cost"] and row["reward"] == main["reward"]
                            and row["initial_rng_after"] == main["host_rng_after"], "selected initial fork differs from actual ordinary execution")
                action_results[action_kind][horizon] = row
        restore = restoration_index[sid]
        require((restore["episode"], restore["step"]) == (ep, step), "restoration position")
        for field in ("chain_unchanged", "reseeds_unchanged", "life_unchanged", "body_unchanged", "rng_unchanged", "forwards_restored"):
            require(restore.get(field) is True, f"main state leaked: {field}")
        require(restore["life_hash_before"] == restore["life_hash_after"] == snap["life_hash"], "restored life identity differs")
        require(restore["body_hash_before"] == restore["body_hash_after"] == snap["body_hash"], "restored body identity differs")
        require(restore["agent_rng"] == snap["agent_rng"] and restore["host_rng_before"] == snap["host_rng_before"], "restored RNG differs")
        require(restore["host_forwards"] == snap["host_forwards"], "ordinary forward counter leaked")
        require(restore["alternatives"] == len(action_set), "restoration fork counter differs")
        expected_forwards = sum(action_results[k][max(horizons)]["cumulative_generated"] for k in action_set)
        require(restore["diagnostic_forwards"] == expected_forwards, "diagnostic forwards unaccounted")
        diagnostic_forwards += expected_forwards
        confirm = confirmation_index[sid]
        require((confirm["episode"], confirm["step"], confirm["action"]) == (ep, step, main["action"]), "selected confirmation differs")
        for field in ("chain_equal", "reseeds_equal", "cost_equal", "rng_equal", "metrics_equal", "policy_input_equal"):
            require(confirm.get(field) is True, f"selected-action confirmation failed: {field}")
        comparisons = []
        for horizon in horizons:
            rows = {kind: action_results[kind][horizon] for kind in sorted(action_set)}
            best = max(row["reward"] for row in rows.values())
            tolerance = protocol["comparison"]["tie_tolerance"]
            winners = [kind for kind, row in rows.items() if best - row["reward"] <= tolerance]
            keep = rows[0]
            selected = rows[main["action"]["kind"]]
            comparisons.append({"horizon": horizon, "winner_kinds": winners, "winner_names": [NAMES[k] for k in winners],
                "positive_opportunity": best > keep["reward"] + tolerance,
                "selected_regret": best - selected["reward"],
                "selected_regret_positive": best - selected["reward"] > tolerance,
                "actions": [{"kind": kind, "name": NAMES[kind], "reward": row["reward"],
                    "reward_vs_keep": row["reward"] - keep["reward"], "before": row["before"], "after": row["after"],
                    "axis_delta_vs_keep": {a: row["after"][a] - keep["after"][a] for a in AXES},
                    "initial_generated": row["initial_generated"], "cumulative_generated": row["cumulative_generated"],
                    "cost_normalized": row["cost_normalized"]} for kind, row in rows.items()]})
        results.append({"snapshot": sid, "episode": ep, "step": step, "target": target,
                        "selected_kind": main["action"]["kind"], "comparisons": comparisons})
    require(used == set(index), "extra unavailable action/horizon")
    require(forks == protocol["interventions"]["forks_per_seed"], "fork count differs from protocol")
    require(len(index) == protocol["continuation"]["measurements_per_seed"], "measurement count differs from protocol")
    require(summary["snapshots"] == expected_snapshots and summary["alternatives"] == forks
            and summary["measurements"] == len(index) and summary["matched_actual_actions"] == expected_snapshots,
            "terminal counters incomplete")
    require(summary["diagnostic_forwards"] == diagnostic_forwards, "terminal diagnostic forward count differs")
    aggregate = []
    for hi, horizon in enumerate(horizons):
        comparisons = [s["comparisons"][hi] for s in results]
        winner_sets = {}
        for row in comparisons:
            name = "+".join(row["winner_names"])
            winner_sets[name] = winner_sets.get(name, 0) + 1
        aggregate.append({"horizon": horizon, "positive_opportunities": sum(r["positive_opportunity"] for r in comparisons),
            "positive_selected_regrets": sum(r["selected_regret_positive"] for r in comparisons),
            "mean_selected_regret": sum(r["selected_regret"] for r in comparisons) / expected_snapshots,
            "max_selected_regret": max(r["selected_regret"] for r in comparisons), "winner_sets": winner_sets,
            "winner_set_changed_from_h0": sum(s["comparisons"][0]["winner_kinds"] != s["comparisons"][hi]["winner_kinds"] for s in results)})
    return {"seed": seed, "snapshots": expected_snapshots, "alternatives": forks, "measurements": len(index),
            "diagnostic_forwards": summary["diagnostic_forwards"], "horizons": aggregate, "comparisons": results,
            "checks": {"complete": True, "common_state": True, "initial_rng_and_future_pairing": True,
                       "raw_costs_and_reward_recomputed": True, "selected_initial_matches_actual": True,
                       "original_target_repetition": True, "restoration": True}}


def run_command(command: list[str], cwd: Path, output: Path, label: str) -> dict:
    started = time.monotonic()
    result = subprocess.run(command, cwd=cwd, text=True, capture_output=True)
    stdout, stderr = output / f"{label}.stdout.txt", output / f"{label}.stderr.txt"
    stdout.write_text(result.stdout); stderr.write_text(result.stderr)
    record = {"command": command, "cwd": str(cwd), "returncode": result.returncode,
              "wall_seconds": time.monotonic() - started,
              "stdout": {"file": stdout.name, "sha256": V1.digest(stdout), "text": result.stdout},
              "stderr": {"file": stderr.name, "sha256": V1.digest(stderr), "text": result.stderr}}
    V1.write_json(output / f"{label}.command.json", record)
    require(result.returncode == 0, f"{label} failed ({result.returncode}); see {stderr}")
    return record


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--inputs", type=Path, help="reuse already verified public simple.weights/dracula.txt")
    args = parser.parse_args()
    output = args.output.resolve()
    require(output != ROOT and ROOT not in output.parents, "output must be outside the repository")
    require(not output.exists() or not any(output.iterdir()), "output must be new/empty; preserve prior attempts")
    output.mkdir(parents=True, exist_ok=True)
    require(V1.digest(PROTOCOL) == FROZEN_PROTOCOL_SHA256, "scenario protocol changed after preregistration")
    protocol = json.loads(PROTOCOL.read_text())
    parent_protocol = json.loads(V1.PROTOCOL.read_text())
    require(V1.digest(V1.PROTOCOL) == protocol["parent_protocol"]["sha256"], "parent protocol changed")
    historical = json.loads((HERE.parent / "receipts.json").read_text())
    if args.inputs:
        for name in ("simple.weights", "dracula.txt"):
            shutil.copyfile(args.inputs / name, output / name)
    identities, vocabulary = V1.prepare(output, args.reference, parent_protocol)
    source_bytes = {name: (ROOT / name).read_bytes() for name in SOURCE_FILES}
    source_hashes = {name: hashlib.sha256(data).hexdigest() for name, data in source_bytes.items()}
    require(all(V1.digest(ROOT / name) == value for name, value in source_hashes.items()), "source changed while snapshotting")
    snapshot = output / "source_snapshot"
    for name, data in source_bytes.items():
        target = snapshot / name; target.parent.mkdir(parents=True, exist_ok=True); target.write_bytes(data)
    command_records = []
    command_records.append(run_command([sys.executable, str(snapshot / "tests/test_spa_scenarios.py"),
                                       "--json", str(output / "gates.json")], snapshot, output, "gates"))
    binary = output / "spa_agent_demo"
    build = ["cc", "-std=c11", "-O2", "-DUSE_SIMD", "-march=native", "-pthread", "-I.",
             "examples/spa_agent_demo.c", "spa_agent.c", "notorch.c", "-lm", "-o", str(binary)]
    command_records.append(run_command(build, snapshot, output, "build"))
    metadata = {"schema": 1, "protocol_sha256": V1.digest(PROTOCOL), "protocol": protocol,
        "parent_protocol_sha256": V1.digest(V1.PROTOCOL), "base_commit": V1.git("rev-parse", "HEAD"),
        "source_status": V1.git("status", "--porcelain"), "source_files_sha256": source_hashes,
        "compiled_from_immutable_snapshot": True, "binary_sha256": V1.digest(binary), "inputs": identities,
        "compiler": subprocess.check_output(["cc", "--version"], text=True).splitlines()[0], "machine": V1.machine()}
    metadata["narrow_gates"] = json.loads((output / "gates.json").read_text())
    V1.write_json(output / "metadata.json", metadata)
    runs, artifacts, raw_records = [], {"gates.json": {"sha256": V1.digest(output / "gates.json"),
                                                     "bytes": (output / "gates.json").stat().st_size}}, []
    for seed in protocol["seeds"]:
        off, on = output / f"off-s{seed}", output / f"on-s{seed}"
        scenario_path = output / f"scenarios-s{seed}.jsonl"
        base = [str(binary), str(output / "simple.weights"), str(output / "corpus.tokens.u32"), str(output / "vocabulary.u32")]
        command_records.append(run_command(base + [str(off), str(seed)], snapshot, output, f"off-s{seed}"))
        command_records.append(run_command(base + [str(on), str(seed), "--scenarios", str(scenario_path)], snapshot, output, f"on-s{seed}"))
        for suffix in ("jsonl", *(f"{arm}.life.bin" for arm in parent_protocol["arms"])):
            off_file, on_file = Path(f"{off}.{suffix}"), Path(f"{on}.{suffix}")
            require(off_file.read_bytes() == on_file.read_bytes(), f"off/on state leak seed{seed} {suffix}")
            expected = historical["artifacts"][f"simple-s{seed}.{suffix}"]["sha256"]
            require(V1.digest(off_file) == expected, f"parent identity changed seed{seed} {suffix}")
        ordinary = read_events(Path(f"{off}.jsonl"))
        parent_result = V1.summarize(ordinary, vocabulary, parent_protocol)
        result = validate_seed(read_events(scenario_path), ordinary, protocol)
        learned = next(a for a in parent_result["arms"] if a["name"] == "learned")
        result["parent_acquired_weight_changes"] = learned["weights_changed_choices"]
        result["parent_learned_mean_reward"] = learned["mean_reward"]
        result["checks"].update({"off_on_original_trace_and_five_lives_identical": True,
                                 "original_parent_trace_and_life_hashes_reproduced": True})
        runs.append(result)
        for kind, path in (("ordinary_off", Path(f"{off}.jsonl")), ("ordinary_on", Path(f"{on}.jsonl")), ("scenarios", scenario_path)):
            for line in path.read_bytes().splitlines(keepends=True):
                raw_records.append(json.dumps({"stream": kind, "seed": seed, "raw": line.decode("utf-8")},
                                              separators=(",", ":"), allow_nan=False).encode() + b"\n")
        for path in sorted(output.glob(f"*-s{seed}*")):
            if path.is_file(): artifacts[path.name] = {"sha256": V1.digest(path), "bytes": path.stat().st_size}
        V1.write_json(output / "receipts.partial.json", {"metadata": metadata, "runs": runs, "artifacts": artifacts})
        print(json.dumps({"seed": seed, "snapshots": result["snapshots"], "horizons": result["horizons"]}), flush=True)
    require(sum(r["parent_acquired_weight_changes"] for r in runs) == 0, "registered parent 0/48 changed")
    archive = output / "raw_traces.jsonl.gz"
    archive.write_bytes(gzip.compress(b"".join(raw_records), mtime=0))
    artifacts[archive.name] = {"sha256": V1.digest(archive), "bytes": archive.stat().st_size}
    metadata["commands"] = command_records
    receipts = {"schema": 1, "metadata": metadata, "runs": runs, "artifacts": artifacts,
                "historical_result": {"acquired_weight_changes": 0, "decisions": 48}}
    # Exact machine paths remain in per-command files; committed receipts use
    # explicit reproducible placeholders for this invocation's output/snapshot.
    encoded = json.dumps(receipts, ensure_ascii=False, allow_nan=False)
    encoded = encoded.replace(str(snapshot), "<source_snapshot>").replace(str(output), "<output>")
    encoded = re.sub(r"/[^\s\"\\]*notorch-spa-scenarios-[^/\s\"\\]*", "<gate_tmp>", encoded)
    V1.write_json(output / "receipts.json", json.loads(encoded))
    print(f"Completed: {output / 'receipts.json'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
