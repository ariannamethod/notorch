#!/usr/bin/env python3
"""Compile future-credit defects and compare freshly rebuilt v1 saved lives.

The checked-out production sources are read-only. Compiler errors, signals and
unnamed failures never count as caught defects. Receipts use portable paths.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
BASELINE = "47cb6f8207b8ca2c3fc02be898eeb19d0f6bbad7"
SOURCES = ("spa_agent.c", "spa_agent.h", "notorch.c", "notorch.h",
           "notorch_simd.h", "chuck_architect.h", "chuck_architect_impl.h")
NATIVE_TEST = "tests/test_spa_agent_future.c"
HOST = "examples/spa_agent_future.c"

V1_EMITTER = r'''#include "spa_agent.h"
#include <stdio.h>
#include <string.h>
static int check(int status, const char *operation) {
    if (status != NT_SPA_OK) {
        fprintf(stderr,"FAIL v1_compatibility: %s status=%d\n",operation,status);
        return 0;
    }
    return 1;
}
static int save(const nt_spa_agent *a,const char *directory,const char *name) {
    char path[4096];
    int n=snprintf(path,sizeof(path),"%s/%s",directory,name);
    return n>0 && (size_t)n<sizeof(path) && check(nt_spa_agent_save(a,path),name);
}
int main(int argc,char **argv) {
    nt_spa_agent a;
    nt_spa_agent_config c;
    nt_spa_observation o;
    nt_spa_decision d;
    nt_spa_receipt r;
    nt_spa_consequence consequence;
    float loss;
    int i;
    if(argc!=2) return 2;
    nt_spa_agent_config_default(&c);
    c.mode=NT_SPA_AGENT_LEARNED;c.seed=991;c.exploration=.2f;
    memset(&o,0,sizeof(o));
    o.embedding[0]=-.2f;o.embedding[1]=.3f;o.embedding[2]=.17f;o.embedding[3]=-.08f;
    o.connectedness=.6f;o.left_similarity=.3f;o.right_similarity=.7f;
    o.coherence=.65f;o.novelty=.31f;o.repetition=.21f;o.phase_lock=.5f;
    o.sentence_score=.2f;o.mean_sentence_score=.4f;o.temperature=.8f;
    o.sentence_index=1;o.sentence_count=4;
    memset(&consequence,0,sizeof(consequence));
    consequence.before=(nt_spa_metrics){.4f,.3f,.5f,.4f,.2f,.1f,.6f};
    consequence.after=(nt_spa_metrics){.6f,.5f,.7f,.6f,.1f,.05f,.7f};
    consequence.regeneration_cost=.25f;
    if(!check(nt_spa_agent_init(&a,&c),"init") || !save(&a,argv[1],"initial.bin")) return 1;
    if(!check(nt_spa_agent_choose(&a,&o,&d),"choose") || !save(&a,argv[1],"pending.bin")) return 1;
    if(!check(nt_spa_agent_observe(&a,d.sequence,&d.action,&consequence,&r),"observe")) return 1;
    for(i=1;i<12;++i) {
        o.embedding[0]+=.01f;o.reseed_count=(unsigned)i;
        consequence.after.novelty=.4f+.01f*i;
        if(!check(nt_spa_agent_choose(&a,&o,&d),"choose history") ||
           !check(nt_spa_agent_observe(&a,d.sequence,&d.action,&consequence,&r),"observe history")) return 1;
    }
    if(!save(&a,argv[1],"completed.bin")) return 1;
    if(!check(nt_spa_agent_imitate(&a,&o,NT_SPA_RESEED_RIGHT,&loss),"imitate") ||
       !save(&a,argv[1],"imitation.bin")) return 1;
    puts("PASS v1_compatibility: initial, pending, completed and imitation lives");
    return 0;
}
'''


def digest(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def invoke(command, cwd=ROOT, timeout=180):
    result = subprocess.run([str(value) for value in command], cwd=cwd,
                            capture_output=True, text=True, timeout=timeout,
                            check=False)
    return {"command": [str(value) for value in command],
            "returncode": result.returncode, "stdout": result.stdout,
            "stderr": result.stderr}


def portable(value, temporary):
    if isinstance(value, str):
        return value.replace(str(temporary), "<temporary>").replace(str(ROOT), "<source>")
    if isinstance(value, list):
        return [portable(item, temporary) for item in value]
    if isinstance(value, dict):
        return {key: portable(item, temporary) for key, item in value.items()}
    return value


def write_receipt(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as stream:
        json.dump(value, stream, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def comparison_build(compiler, temporary, name, sources, emitter):
    directory = temporary / name
    directory.mkdir()
    for path, raw in sources.items():
        destination = directory / path
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(raw)
    test = directory / "gate.c"
    test.write_bytes(emitter)
    binary = directory / "gate"
    command = compiler + ["-O2", "-std=gnu11", "-pthread", "-I", directory,
                          "-I", ROOT, test, directory / "spa_agent.c",
                          directory / "notorch.c", "-lm", "-o", binary]
    return binary, invoke(command)


def v1_compatibility(compiler, temporary, originals):
    old = {}
    for name in SOURCES:
        result = subprocess.run(["git", "show", f"{BASELINE}:{name}"], cwd=ROOT,
                                capture_output=True, timeout=30, check=False)
        if result.returncode:
            raise ValueError(f"pinned v1 source unavailable: {name}")
        old[name] = result.stdout
    report = {"baseline_commit": BASELINE,
              "baseline_source_sha256": {name: digest(raw) for name, raw in old.items()},
              "emitter_sha256": digest(V1_EMITTER.encode()), "files": []}
    binaries = {}
    for name, sources in (("v1-old", old), ("v1-current", originals)):
        binary, built = comparison_build(compiler, temporary, name, sources, V1_EMITTER.encode())
        report[name] = {"build": built}
        if built["returncode"]:
            continue
        output = temporary / (name + "-output")
        output.mkdir()
        report[name]["run"] = invoke([binary, output])
        binaries[name] = output
    if all(report[name].get("run", {}).get("returncode") == 0
           for name in ("v1-old", "v1-current")):
        for name in ("initial.bin", "pending.bin", "completed.bin", "imitation.bin"):
            prior = (binaries["v1-old"] / name).read_bytes()
            current = (binaries["v1-current"] / name).read_bytes()
            report["files"].append({"name": name, "bytes": len(current),
                                     "old_sha256": digest(prior),
                                     "new_sha256": digest(current),
                                     "identical": bool(current) and current == prior})
    report["passed"] = len(report["files"]) == 4 and all(x["identical"] for x in report["files"])
    print(("PASS" if report["passed"] else "FAIL") + " v1_compatibility (four rebuilt lives)")
    return report


def native_mutations(compiler, temporary, originals):
    test = (ROOT / NATIVE_TEST).read_bytes()
    source = originals["spa_agent.c"].decode()
    binary, built = comparison_build(compiler, temporary, "native-baseline", originals, test)
    report = {"test_sha256": digest(test), "baseline": {"build": built}, "mutations": []}
    if built["returncode"] == 0:
        report["baseline"]["run"] = invoke([binary])
    baseline_ok = report["baseline"].get("run", {}).get("returncode") == 0
    if not baseline_ok:
        report["passed"] = False
        print("FAIL unmodified native future gate")
        return report
    print("PASS unmodified native future gate")
    target = "r.targets[i]=r.rewards[i]-r.rewards[NT_SPA_KEEP];"
    update = "if(rate>0) train_policy(&next,e->features,hidden,gradient,rate);"
    mutations = (
        ("reversed_comparison_credit", target,
         "r.targets[i]=r.rewards[NT_SPA_KEEP]-r.rewards[i];", "credit"),
        ("suppressed_comparison_update", update,
         "if(rate>0) train_policy(&next,e->features,hidden,gradient,0);", "credit"),
        ("left_right_targets_swapped", target,
         "r.targets[i]=r.rewards[i==1 ? 2 : i==2 ? 1 : 0]-r.rewards[NT_SPA_KEEP];", "credit"),
    )
    for name, anchor, replacement, gate in mutations:
        if source.count(anchor) != 1:
            raise ValueError(f"{name}: native mutation anchor is not unique")
        mutant = source.replace(anchor, replacement, 1).encode()
        changed = dict(originals, **{"spa_agent.c": mutant})
        executable, compilation = comparison_build(compiler, temporary, name, changed, test)
        entry = {"name": name, "gate": gate, "mutant_source_sha256": digest(mutant),
                 "build": compilation, "caught": False}
        if compilation["returncode"] == 0:
            entry["run"] = invoke([executable, gate])
            entry["caught"] = (entry["run"]["returncode"] == 1 and
                                f"FAIL {gate}:" in entry["run"]["stderr"])
        report["mutations"].append(entry)
        print(("PASS caught " if entry["caught"] else "FAIL escaped or invalid ") + name)
    report["working_test_unchanged"] = (ROOT / NATIVE_TEST).read_bytes() == test
    report["passed"] = (len(report["mutations"]) == 3 and
                        all(item["caught"] for item in report["mutations"]) and
                        report["working_test_unchanged"])
    return report


FIT_VERIFIER = r"""import importlib.util,json,sys
from pathlib import Path
root,gate,trace,dataset=Path(sys.argv[1]),sys.argv[2],Path(sys.argv[3]),Path(sys.argv[4])
spec=importlib.util.spec_from_file_location('future_fixture',root/'tests/test_spa_future.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
future=module.FUTURE
try:
    rows,_=future.parse_raw(trace.read_bytes())
    summary=future.validate_fit(rows,module.synthetic_samples(),dataset,json.loads(future.PROTOCOL.read_text()),'2'*64)
except (AssertionError,ValueError,KeyError) as error:
    print(f'FAIL {gate}: {error}',file=sys.stderr)
    sys.exit(1)
print(f'PASS {gate}: '+json.dumps({k:summary[k] for k in ('fits','samples','epochs_per_arm','life_hashes','checks')},sort_keys=True))
"""


def artifact(path):
    hasher = hashlib.sha256()
    count = 0
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            hasher.update(chunk)
            count += len(chunk)
    return {"bytes": count, "sha256": hasher.hexdigest()}


def host_mutations(compiler, temporary, originals):
    helper_names = ("tests/test_spa_future.py", "experiments/spa_agent/future/run.py",
                    "experiments/spa_agent/future/protocol.json",
                    "experiments/spa_agent/scenarios/run.py", "experiments/spa_agent/run.py",
                    "tests/test_spa_trace_io.py")
    helpers = {name: (ROOT / name).read_bytes() for name in helper_names}
    source_raw = (ROOT / HOST).read_bytes()
    source = source_raw.decode()
    spec = importlib.util.spec_from_file_location("spa_future_fixture", ROOT / "tests/test_spa_future.py")
    fixture_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fixture_module)
    samples = fixture_module.synthetic_samples()
    dataset = temporary / "synthetic.txt"
    fixture_module.FUTURE.write_dataset(dataset, samples, fixture_module.FUTURE.FROZEN_PROTOCOL_SHA256, "2" * 64)
    verifier = temporary / "verify_fit.py"
    verifier.write_text(FIT_VERIFIER)
    report = {"host_sha256": digest(source_raw), "helper_source_sha256": {n: digest(b) for n, b in helpers.items()},
              "fixture": {"kind": "48 invented states; no sentence generation", "dataset": artifact(dataset)},
              "verifier_sha256": digest(FIT_VERIFIER.encode()), "mutations": []}

    def execute(name, raw, gate):
        binary, build = comparison_build(compiler, temporary, name, originals, raw)
        result = {"build": build}
        if build["returncode"]:
            return result
        prefix = temporary / (name + "-output")
        result["run"] = invoke([binary, "train", dataset, prefix], timeout=300)
        trace = prefix.with_suffix(".fit.jsonl")
        if trace.exists():
            result["trace"] = artifact(trace)
        if result["run"]["returncode"] == 0:
            result["gate"] = invoke([sys.executable, verifier, ROOT, gate, trace, dataset], timeout=300)
        return result

    report["baseline"] = execute("host-baseline", source_raw, "future_fit_baseline")
    baseline_ok = report["baseline"].get("gate", {}).get("returncode") == 0
    if not baseline_ok:
        report["passed"] = False
        print("FAIL unmodified future host gate")
        return report
    print("PASS unmodified future host gate")
    horizon = "unsigned horizon_index = arm == 1 ? 0u : 1u;"
    feature_anchor = "// SPA_FUTURE_MUTATE_SOURCE_FEATURES: provenance must follow the features actually used."
    action_anchor = "// SPA_FUTURE_MUTATE_ACTION_TARGETS: independent gates join each typed outcome to its source."
    mutations = (
        ("immediate_substituted_for_future", horizon, "unsigned horizon_index = 0u;",
         "future_horizon", "fit horizon differs from registered arm", False),
        ("host_action_outcomes_swapped", action_anchor,
         action_anchor + "\n            if ((comparison.action_mask & 6u) == 6u) {\n"
         "                nt_spa_consequence exchanged = comparison.alternatives[1].consequence;\n"
         "                comparison.alternatives[1].consequence = comparison.alternatives[2].consequence;\n"
         "                comparison.alternatives[2].consequence = exchanged;\n            }",
         "future_action_targets", "fit rewards differs from raw outcome", False),
        ("source_features_reassociated", feature_anchor,
         feature_anchor + "\n            experience.features[0] = experience.features[0] == .125f ? -.125f : .125f;",
         "source_feature_witness", "source-feature witness mismatch", True),
    )
    for name, anchor, replacement, gate, expected_failure, native_refusal in mutations:
        if source.count(anchor) != 1:
            raise ValueError(f"{name}: host mutation anchor is not unique")
        mutant = source.replace(anchor, replacement, 1).encode()
        result = execute(name, mutant, gate)
        result.update(name=name, named_gate=gate, expected_failure=expected_failure,
                      mutant_source_sha256=digest(mutant), caught=False)
        if result["build"]["returncode"] == 0:
            checked = result.get("run" if native_refusal else "gate", {})
            result["caught"] = (checked.get("returncode") == 1 and
                                expected_failure in checked.get("stderr", ""))
            if not native_refusal:
                result["caught"] = (result["caught"] and result.get("run", {}).get("returncode") == 0
                                    and f"FAIL {gate}:" in checked["stderr"])
        report["mutations"].append(result)
        print(("PASS caught " if result["caught"] else "FAIL escaped or invalid ") + name)
    report["working_host_and_helpers_unchanged"] = ((ROOT / HOST).read_bytes() == source_raw and
        all((ROOT / name).read_bytes() == raw for name, raw in helpers.items()))
    if not report["working_host_and_helpers_unchanged"]:
        print("FAIL host or independent helpers changed during gate")
    report["passed"] = (len(report["mutations"]) == 3 and
                        all(item["caught"] for item in report["mutations"]) and
                        report["working_host_and_helpers_unchanged"])
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", type=Path)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--v1-only", action="store_true", help="run only the narrow old-API byte gate")
    mode.add_argument("--native-only", action="store_true", help="run native mutants and old-API bytes")
    mode.add_argument("--host-only", action="store_true", help="run host mutants and old-API bytes")
    args = parser.parse_args()
    compiler = shlex.split(os.environ.get("CC", "cc"))
    report = {"schema": 1, "passed": False}
    originals = {name: (ROOT / name).read_bytes() for name in SOURCES}
    report["source_sha256"] = {name: digest(raw) for name, raw in originals.items()}
    with tempfile.TemporaryDirectory(prefix="notorch-spa-future-") as temporary_name:
        temporary = Path(temporary_name)
        try:
            report["v1_compatibility"] = v1_compatibility(compiler, temporary, originals)
            if not args.v1_only:
                if not args.host_only:
                    report["native"] = native_mutations(compiler, temporary, originals)
                if not args.native_only:
                    report["host"] = host_mutations(compiler, temporary, originals)
            report["changed_sources"] = [name for name, raw in originals.items()
                                         if (ROOT / name).read_bytes() != raw]
            report["working_sources_unchanged"] = not report["changed_sources"]
            if report["changed_sources"]:
                print("FAIL working sources changed during gate: " + ", ".join(report["changed_sources"]))
            report["passed"] = (report["v1_compatibility"]["passed"] and report["working_sources_unchanged"]
                                and (args.v1_only or args.host_only or report["native"]["passed"])
                                and (args.v1_only or args.native_only or report["host"]["passed"]))
        except (OSError, ValueError, subprocess.TimeoutExpired) as error:
            report["error"] = str(error)
            print("FAIL", error)
        report = portable(report, temporary)
    if args.json:
        write_receipt(args.json, report)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
