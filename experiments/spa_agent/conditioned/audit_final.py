#!/usr/bin/env python3
"""Run the frozen audit with its one disclosed outcome-reference schema repair.

The original checker, experiment sources, protocol and outputs remain intact.
Only the expected JSON representation of three action-indexed outcome hashes
changes from a dictionary to the helper's existing three-slot list.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import textwrap
import traceback

HERE = Path(__file__).resolve().parent
ORIGINAL_SHA = "5eee0345a4309415be064ea0cf6b930cd87d1317e6016b8477eb3df0f0bb8af6"
FAILURE_SHA = "4e7c9e5a83ce50a42626692cd034e11d352a93690bb8121be5ec9f0b5b4ded13"
BEFORE = '''                expected = [{"replicate": r, "outcome_sha256": {str(o["action"]["kind"]): o["raw_sha256"]
                    for o in donor["repeats"][r]["outcomes"]["4"]}} for r in range(8)]'''
AFTER = '''                expected = [{"replicate": r, "outcome_sha256": [next((o["raw_sha256"]
                    for o in donor["repeats"][r]["outcomes"]["4"] if o["action"]["kind"] == kind), None)
                    for kind in range(3)]} for r in range(8)]'''


def require(value, message):
    if not value:
        raise AssertionError(message)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def identity(path):
    raw = path.read_bytes()
    return {"bytes": len(raw), "sha256": sha(raw)}


def publish(path, value):
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def correction():
    original = (HERE / "audit.py").read_bytes()
    failure = (HERE / "audit_failure.json").read_bytes()
    require(sha(original) == ORIGINAL_SHA and sha(failure) == FAILURE_SHA,
            "frozen original audit/failure identity")
    source = original.decode()
    require(source.count(BEFORE) == 1, "unique exact schema repair site")
    repaired = source.replace(BEFORE, AFTER)
    require(repaired.replace(AFTER, BEFORE) == source, "only registered textual correction")
    namespace = {"__name__": "spa_conditioned_corrected_independent_audit",
                 "__file__": str(HERE / "audit.py")}
    exec(compile(repaired, "<frozen-audit-with-explicit-schema-repair>", "exec"), namespace)
    donor = {"repeats": [{"outcomes": {"4": [
        {"action": {"kind": kind}, "raw_sha256": sha(f"draw{r}:action{kind}".encode())}
        for kind in (0, 2)]}} for r in range(8)]}
    fixture = {"donor": donor}
    exec(compile(textwrap.dedent(AFTER), "<exact-repaired-reference-expression>", "exec"), fixture)
    expected = fixture["expected"]
    actual = [{"replicate": r, "outcome_sha256": [donor["repeats"][r]["outcomes"]["4"][0]["raw_sha256"],
        None, donor["repeats"][r]["outcomes"]["4"][1]["raw_sha256"]]} for r in range(8)]
    require(actual == expected, "valid three-slot helper references")
    defects = []
    for name, bad in (("swapped_action_hashes", copy.deepcopy(actual)),
                      ("populated_invalid_action_slot", copy.deepcopy(actual))):
        if name == "swapped_action_hashes":
            bad[0]["outcome_sha256"][0], bad[0]["outcome_sha256"][2] = bad[0]["outcome_sha256"][2], bad[0]["outcome_sha256"][0]
        else:
            bad[0]["outcome_sha256"][1] = sha(b"invalid action witness")
        try:
            namespace["require"](bad == expected, "comparison raw outcome association")
        except AssertionError as error:
            defects.append({"name": name, "caught": True, "message": str(error)})
        else:
            raise AssertionError("reference corruption escaped: " + name)
    receipt = {"schema": 1, "status": "PASS", "scope": "Audit-schema correction only",
        "original_audit": identity(HERE / "audit.py"), "retained_failure": identity(HERE / "audit_failure.json"),
        "wrapper": identity(Path(__file__)),
        "exact_patch": {"before": BEFORE, "after": AFTER, "matches": 1,
                        "before_sha256": sha(BEFORE.encode()), "after_sha256": sha(AFTER.encode())},
        "derived_checker": {"bytes": len(repaired.encode()), "sha256": sha(repaired.encode())},
        "valid_fixture": {"draws": 8, "valid_kinds": [0, 2], "invalid_slot": 1},
        "named_red_hands": defects,
        "unchanged_checks": ["source and donor identity", "native consequence input bytes",
            "all valid action coverage", "every raw per-action outcome hash", "reward/target/scale",
            "policy hash chain and saved weights", "raw outcome metrics and summary"],
        "experiment_source_or_data_changes": False, "generation_runs": 0, "native_fit_runs": 0,
        "original_failure_preserved": True}
    return namespace, receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execution-dir", type=Path, required=True)
    parser.add_argument("--correction", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    namespace, repaired = correction()
    publish(args.correction, repaired)
    try:
        report = namespace["audit"](args.execution_dir)
    except Exception:
        failure = {"schema": 1, "status": "FAIL", "correction": identity(args.correction),
            "traceback": traceback.format_exc().replace(str(HERE.parents[2]), "<repo>"),
            "data_or_source_changes": False, "generation_runs": 0, "native_fit_runs": 0}
        publish(args.output.with_name("audit_after_correction_failure.json"), failure)
        raise
    report["frozen_audit"] = report.pop("script")
    report["script"] = identity(Path(__file__))
    report["schema_correction"] = repaired
    report["schema_correction_receipt"] = identity(args.correction)
    report["command"] = "python3 experiments/spa_agent/conditioned/audit_final.py --execution-dir <completed-run> --correction experiments/spa_agent/conditioned/audit_correction.json --output experiments/spa_agent/conditioned/result_audit.json"
    publish(args.output, report)
    print(json.dumps({"status": report["status"], "gates": len(report["gates"]),
        "schema_corrections": 1, "output": identity(args.output)}, sort_keys=True))


if __name__ == "__main__":
    main()
