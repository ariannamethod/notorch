#!/usr/bin/env python3
"""Cheap synthetic repeated-host and actual import-SPA acquisition gates."""
from __future__ import annotations

import argparse
import copy
import io
import json
from pathlib import Path
import subprocess
import tempfile
import importlib.util

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("spa_replicates", ROOT / "experiments/spa_agent/replicates/run.py")
RUN = importlib.util.module_from_spec(SPEC); SPEC.loader.exec_module(RUN)
POLICY = RUN.POLICY


def invoke(command, cwd=ROOT):
    command = list(map(str, command))
    result = subprocess.run(command, cwd=cwd, capture_output=True, text=True, timeout=240)
    return {"command": command, "returncode": result.returncode, "stdout": result.stdout, "stderr": result.stderr}


def passed(record):
    RUN.require(record["returncode"] == 0, "command failed: " + record["stderr"])
    return record


def reject(name, operation, expected):
    try: operation()
    except (AssertionError, ValueError, KeyError) as error:
        RUN.require(expected in str(error), name + ": wrong named failure: " + str(error))
        return {"name": name, "caught": True, "failure": str(error)}
    raise AssertionError(name + ": deliberate defect escaped")


def fixture_inputs(prefix):
    rows, _ = RUN.parse_raw(Path(str(prefix) + ".sources.jsonl").read_bytes())
    samples = [RUN.source_from_row(row) for row in rows]
    raw = Path(str(prefix) + ".expected_r0.jsonl").read_bytes()
    measurements, _ = RUN.parse_raw(raw)
    references = {(row["seed"], row["snapshot"], row["action"]["kind"], row["horizon"]): line
                  for row, line in zip(measurements, raw.splitlines(keepends=True))}
    return samples, references


def synthetic_samples(projected):
    result = []
    by_target = {sample["target"]: sample for sample in projected}
    for seed in (42, 73):
        for sid in range(24):
            episode, step = divmod(sid, 4); target = (episode + step) % 4
            sample = copy.deepcopy(by_target[target]); sample.update(ordinal=len(result), seed=seed,
                episode=episode, step=step, snapshot=sid,
                host_rng=f"{RUN.SCENARIOS.stream(seed, 0x7370615f6163746e, episode, step):016x}")
            # Distinct invented source identities preserve the field values and
            # valid compact history while exercising strict source association.
            sample["life_hash"] = f"{1000 + len(result):016x}"
            sample["snapshot_raw_sha256"] = RUN.sha256(f"synthetic source {len(result)}".encode())
            sample["ordinary_decision_raw_sha256"] = RUN.sha256(f"synthetic main {len(result)}".encode())
            sample["feature_witness"] = RUN.FUTURE.feature_witness(sample)
            sample["snapshot_witness"] = RUN.snapshot_witness(sample)
            result.append(sample)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", type=Path)
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()
    commands, failures = [], []
    source_names = ("tests/spa_replicates_fixture.c", "examples/spa_agent_replicates.c", "examples/spa_agent_scenarios.h",
                    "examples/spa_agent_demo.c", "spa_agent.c", "spa_agent.h", "spa_binding.c", "spa_binding.h",
                    "python/SPA.py", "python/notorch.py", "experiments/spa_agent/replicates/policy.py",
                    "experiments/spa_agent/replicates/run.py", "tests/test_spa_replicates.py")
    source_before = {name: RUN.V1.digest(ROOT / name) for name in source_names}
    with tempfile.TemporaryDirectory(prefix="spa-replicates-test-") as directory:
        directory = Path(directory)
        fixture = directory / "fixture"
        command = ["cc", "-std=c11", "-O2", "-I.", "tests/spa_replicates_fixture.c", "spa_agent.c", "notorch.c", "-lm", "-o", fixture]
        commands.append(passed(invoke(command)))
        prefix = directory / "source"
        commands.append(passed(invoke([fixture, "write", prefix])))
        commands.append(passed(invoke([fixture, "cases", directory / "refusal"])))
        native_cases = json.loads(commands[-1]["stdout"].splitlines()[-1])
        trace = directory / "baseline.jsonl"
        commands.append(passed(invoke([fixture, "run", Path(str(prefix) + ".sources.txt"), trace])))
        samples, references = fixture_inputs(prefix)
        raw = trace.read_bytes()
        projected, host = RUN.validate_replicates(raw, samples, references,
            dataset_fnv=RUN.fnv(Path(str(prefix) + ".sources.txt").read_bytes()))
        # Exporting the imported source again must preserve every source float,
        # token and witness; this exercises the actual production text importer.
        exported = directory / "exported.txt"
        header = RUN.parse_raw(raw)[0][0]
        RUN.write_sources(exported, samples, header["vocabulary_fnv1a"])
        reexport = directory / "reexport.jsonl"
        commands.append(passed(invoke([fixture, "run", exported, reexport])))
        replay, _ = RUN.validate_replicates(reexport.read_bytes(), samples, references, dataset_fnv=RUN.fnv(exported.read_bytes()))
        RUN.require(projected == replay, "source export/import changed measurements")
        rows, _ = RUN.parse_raw(raw)
        def mutate(name, change, expected):
            altered = copy.deepcopy(rows); change(altered)
            # Preserve original native bytes for every unaffected line so r0's
            # byte gate remains a real independent reference.
            altered_raw = b"".join(original if old == new else
                (json.dumps(new, separators=(",", ":")) + "\n").encode()
                for original, old, new in zip(raw.splitlines(keepends=True), rows, altered))
            failures.append(reject(name, lambda: RUN.validate_replicates(altered_raw, samples, references), expected))
        def find(rows, kind, **values):
            return next(row for row in rows if row["type"] == kind and all(row.get(k) == v for k, v in values.items()))
        begin_index = next(i for i, row in enumerate(rows) if row["type"] == "replicate_begin" and row["replicate"] == 1)
        h1 = begin_index + 2; h4 = begin_index + 3
        RUN.require(rows[h1]["type"] == "measurement" and rows[h1]["horizon"] == 1, "fixture mutation coordinate")
        mutate("wrong_replicate", lambda r: r[begin_index].update(replicate=2), "bracket index/count/order")
        mutate("paired_rng", lambda r: r[h1]["continuation"][0].update(rng_before="0000000000000001"), "future RNG pairing")
        mutate("untouched_future_chain", lambda r: r[h1]["chain"][2]["tokens"].__setitem__(0, (r[h1]["chain"][2]["tokens"][0] + 1) % 94), "untouched future sentence changed")
        def change_length(r):
            r[h4]["chain"][3]["tokens"].pop(0); r[h4]["chain"][3]["length"] -= 1
        mutate("future_token_accounting", change_length, "future token accounting")
        mutate("reward_sign", lambda r: r[h1].update(reward=-r[h1]["reward"] + .1), "reward sign/raw axes")
        mutate("source_witness", lambda r: find(r, "source").update(feature_witness="0000000000000000"), "imported source changed")
        failures.append(reject("truncated_stream", lambda: RUN.validate_replicates(b"".join(raw.splitlines(keepends=True)[:-1]), samples, references), "incomplete/reordered"))
        # Extra/doubled records and missing a complete bracket are rejected even
        # when a terminal summary still claims the correct counts.
        failures.append(reject("duplicate_record", lambda: RUN.validate_replicates(raw.splitlines(keepends=True)[0] + raw, samples, references), "incomplete/reordered"))
        library = args.library.resolve() if args.library else directory / "libnotorch.so"
        if not args.library:
            commands.append(passed(invoke(["cc", "-std=c11", "-O2", "-fPIC", "-shared", "-I.",
                "spa_binding.c", "spa_agent.c", "notorch.c", "-lm", "-o", library])))
        native = POLICY.SPA.Native(library)
        invented = synthetic_samples(projected)
        dataset = directory / "synthetic.json"
        RUN.write_json(dataset, {"schema": 1, "protocol_sha256": RUN.FROZEN_PROTOCOL_SHA256,
                               "replicates": 8, "samples": invented})
        loaded = POLICY.read_dataset(dataset, RUN.FROZEN_PROTOCOL_SHA256, training=True)
        RUN.require(loaded == invented, "helper imported source changed")
        bad = copy.deepcopy(invented[0]); bad["features"][0] += .125
        failures.append(reject("feature_association", lambda: POLICY.validate_source(bad, 0), "feature-source witness"))
        bad_witness = copy.deepcopy(invented[0]); bad_witness["snapshot_witness"] = "bad"
        failures.append(reject("malformed_snapshot_witness", lambda: POLICY.validate_source(bad_witness, 0), "snapshot witness"))
        failures.append(reject("wrong_donor_coordinates", lambda: POLICY.comparisons(invented[0], invented[1], 8), "donor coordinates differ"))
        bad_r0 = copy.deepcopy(invented[0])
        bad_r0["repeats"][0]["outcomes"] = copy.deepcopy(bad_r0["repeats"][0]["outcomes"])
        bad_r0["repeats"][0]["outcomes"]["4"][0]["cost"] += .01
        failures.append(reject("r0_association", lambda: POLICY.comparisons(bad_r0, bad_r0, 8), "r0 differs"))
        bad_count = copy.deepcopy(invented[0]); bad_count["repeats"].pop()
        failures.append(reject("wrong_replica_count", lambda: POLICY.comparisons(bad_count, bad_count, 8), "repeat order"))
        policies = directory / "policies"
        seal = POLICY.fit_policies(invented, native, policies, epochs=2,
            protocol_sha=RUN.FROZEN_PROTOCOL_SHA256, dataset_identity=RUN.identity(dataset), require_parent_parity=False)
        fit, _ = RUN.parse_raw((policies / "fit.jsonl").read_bytes())
        fitted = RUN.validate_fit(fit, invented, epochs=2)
        agents = {arm: POLICY.SPA.Agent.from_file(policies / (arm + ".life"), native=native) for arm in RUN.ARMS}
        before = {arm: bytes(agent.state) for arm, agent in agents.items()}
        clean, poisoned = io.StringIO(), io.StringIO()
        POLICY.score_samples(invented, agents, clean)
        deleted = copy.deepcopy(invented)
        for sample in deleted:
            del sample["outcomes"]; del sample["repeats"]
        POLICY.score_samples(deleted, agents, poisoned)
        RUN.require(clean.getvalue() == poisoned.getvalue(), "readout leaked measured outcomes")
        RUN.require(before == {arm: bytes(agent.state) for arm, agent in agents.items()}, "readout changed full life bytes")
        readouts, _ = RUN.parse_raw(clean.getvalue().encode())
        choices = RUN.validate_readouts(readouts, invented, {arm: f"{agent.hash:016x}" for arm, agent in agents.items()})
        summary = RUN.summarize(invented, choices, "synthetic")
        RUN.require(len(summary["state_comparisons"]) == 48 * 3 * 7, "measured action join coverage")
        # A self-consistent replacement seal cannot impersonate the registered
        # count-one parent control, even if it declares the full fit budget.
        seal["fits_per_arm"] = 24576; seal["count_one_parent_parity"] = "byte-identical canonical checkpoint"
        (policies / "seal.json").write_text(json.dumps(seal) + "\n")
        failures.append(reject("replaced_resealed_parent_control", lambda: POLICY.verify_seal(policies, native, RUN.FROZEN_PROTOCOL_SHA256), "parent"))
    RUN.require({name: RUN.V1.digest(ROOT / name) for name in source_names} == source_before, "test source changed during synthetic preflight")
    result = {"pass": True, "body_runs": 0, "synthetic_host": host, "native_import_refusals": native_cases,
        "synthetic_fit": fitted, "pure_readouts": 192, "poisoned_outcomes_unused": True,
        "full_life_bytes_unchanged": True, "named_red_hands": failures, "source_sha256": source_before,
        "commands": commands}
    if args.json: RUN.V1.write_json(args.json, result)
    print(json.dumps({key: result[key] for key in ("pass", "body_runs", "synthetic_host", "synthetic_fit", "pure_readouts")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
