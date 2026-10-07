#!/usr/bin/env python3
"""Independent trace checks and compiled red-hand common-state fork tests.

The native fixture supplies a small deterministic generator and embedding table.
It exercises production snapshot/action/measurement/restore code without loading
the public model or changing its frozen experiment. Mutants live in temporary
copies. Compile errors, crashes and unrelated failures never count as caught.
"""
from __future__ import annotations

import argparse
import copy
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
SPEC = importlib.util.spec_from_file_location("spa_scenarios", ROOT / "experiments/spa_agent/scenarios/run.py")
SCENARIOS = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SCENARIOS)
PROTOCOL = json.loads(SCENARIOS.PROTOCOL.read_text())

FIXTURE = r'''
#define SPA_SCENARIO_GENERATOR fixture_generate
#define main spa_real_body_main
#include "examples/spa_agent_demo.c"
#undef main

static void fixture_generate(sentence_body *body, const int *prompt, int length,
                             uint64_t *rng, sentence *out) {
    if (length < 1) fail("fixture empty prompt");
    memset(out,0,sizeof(*out));
    out->length = MIN_GENERATED + prompt[length-1] % 5;
    out->stopped = 1;
    unsigned offset=(unsigned)((*rng >> 17) % VOCAB);
    for (int i=0;i<out->length;i++) {
        (void)next_random(rng);
        /* Generated targets repeat their 1..3-token prompt; untouched base
         * targets have unique trigrams. This gives the fixed-target gate a
         * distinct observable at horizon1, independently of reward tuning. */
        out->ids[i]=(int)(((unsigned)prompt[i%length]+offset) % VOCAB);
        body->forwards++;
    }
}

int main(int argc,char **argv) {
    if(argc==4 && !strcmp(argv[1],"--path-guard")) {
        scenario_sink path_sink;
        scenario_open(&path_sink,argv[2],42);
        scenario_require_separate(&path_sink,argv[3]);
        scenario_close(&path_sink);
        puts("SPA_SCENARIO_PATH_OK");
        return 0;
    }
    if(argc!=4)fail("fixture requires scenario/main trace paths and seed");
    unsigned seed=(unsigned)strtoul(argv[3],NULL,10);
    sentence_body body={0};
    nt_tensor *parameters[PARAMS];
    body.parameters=parameters;
    for(int i=0;i<PARAMS;i++) {
        int len=i ? 1 : VOCAB*DIM;
        parameters[i]=nt_tensor_new(len);
        if(!parameters[i])fail("fixture tensor allocation");
        body.count+=len;
        for(int j=0;j<len;j++) parameters[i]->data[j]=((j*17+i*7)%43-21)*.0078125f;
    }
    nt_train_mode(0);
    experiment_arm arm;
    arm_init(&arm,4,seed);
    scenario_sink sink;
    scenario_open(&sink,argv[1],seed);
    FILE *main_trace=fopen(argv[2],"w");
    if(!main_trace)fail("fixture ordinary trace");
    fprintf(main_trace,"{\"type\":\"body\",\"seed\":%u,\"parameters\":%ld,\"weights_fnv1a\":\"%016" PRIx64 "\"}\n",seed,body.count,body_hash(&body));
    for(unsigned episode=0;episode<EPISODES;episode++) {
        sentence chain[SENTENCES]={0};
        unsigned reseeds[SENTENCES]={0};
        for(int i=0;i<SENTENCES;i++) {
            chain[i].length=MIN_GENERATED+i;
            chain[i].stopped=1;
            for(int j=0;j<chain[i].length;j++)chain[i].ids[j]=(episode*7+i*13+j*3)%VOCAB;
        }
        fprintf(main_trace,"{\"type\":\"base\",\"seed\":%u,\"episode\":%u,\"chain\":",seed,episode);
        write_chain(main_trace,chain); fputs("}\n",main_trace);
        for(unsigned step=0;step<SENTENCES;step++) {
            unsigned target=(episode+step)%SENTENCES;
            nt_spa_observation obs,after_obs;
            nt_spa_metrics before,after;
            observation(&body,chain,target,reseeds[target],&obs,&before);
            scenario_snapshot(&sink,&body,&arm,chain,reseeds,seed,episode,step,target,&obs,&before);
            nt_spa_decision decision;
            check(nt_spa_agent_choose(&arm.life,&obs,&decision),"fixture choose");
            uint64_t rng=stream(seed,UINT64_C(0x7370615f6163746e),episode,step);
            scenario_execution executed=scenario_execute(&body,chain,reseeds,decision.action,rng,0);
            observation(&body,chain,target,reseeds[target],&after_obs,&after);
            scenario_match_actual(&sink,&decision,chain,reseeds,executed.cost,executed.rng_after,&after);
            nt_spa_consequence consequence={before,after,executed.cost};
            nt_spa_receipt receipt;
            check(nt_spa_agent_observe(&arm.life,decision.sequence,&decision.action,&consequence,&receipt),"fixture observe");
            fprintf(main_trace,"{\"type\":\"decision\",\"arm\":\"learned\",\"seed\":%u,\"episode\":%u,\"step\":%u,\"action\":",seed,episode,step);
            write_action(main_trace,&decision.action);
            fprintf(main_trace,",\"policy_rng_before\":%u,\"host_rng_before\":\"%016" PRIx64 "\",\"host_rng_after\":\"%016" PRIx64 "\",\"cost\":%.9g,\"reward\":%.9g,\"before\":",decision.rng_before,rng,executed.rng_after,executed.cost,receipt.reward);
            write_metrics(main_trace,&before); fputs(",\"after\":",main_trace);write_metrics(main_trace,&after);
            fputs(",\"features\":",main_trace);write_floats(main_trace,decision.features,NT_SPA_AGENT_FEATURES);
            fputs(",\"scores\":",main_trace);write_floats(main_trace,decision.scores,NT_SPA_AGENT_ACTIONS);
            fputs(",\"chain\":",main_trace);write_chain(main_trace,chain);fputs("}\n",main_trace);
        }
    }
    scenario_close(&sink);
    if(fclose(main_trace))fail("fixture ordinary close");
    for(int i=0;i<PARAMS;i++)nt_tensor_free(parameters[i]);
    nt_tape_destroy();
    puts("SPA_SCENARIO_FIXTURE_OK snapshots=24 forks=60 measurements=180");
    return 0;
}
'''


def invoke(command, cwd=ROOT):
    command = [str(item) for item in command]
    try:
        completed = subprocess.run(command, cwd=cwd, text=True, capture_output=True, timeout=180)
        return {"command": command, "returncode": completed.returncode,
                "stdout": completed.stdout, "stderr": completed.stderr}
    except subprocess.TimeoutExpired:
        return {"command": command, "returncode": None, "stdout": "", "stderr": "TIMEOUT"}


def validate_files(scenarios: Path, ordinary: Path):
    return SCENARIOS.validate_seed(SCENARIOS.read_events(scenarios), SCENARIOS.read_events(ordinary), PROTOCOL)


def parser_gate(name, rows, ordinary, alter, expected):
    changed = copy.deepcopy(rows)
    alter(changed)
    try:
        SCENARIOS.validate_seed(changed, ordinary, PROTOCOL)
    except (AssertionError, ValueError, KeyError) as error:
        message = str(error)
        return {"name": name, "caught": expected in message, "failure": message}
    return {"name": name, "caught": False, "failure": "defect escaped"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", type=Path)
    parser.add_argument("--validate", nargs=2, type=Path, metavar=("SCENARIOS", "ORDINARY"))
    args = parser.parse_args()
    if args.validate:
        try:
            result = validate_files(*args.validate)
        except (AssertionError, ValueError, KeyError) as error:
            print(f"FAIL scenario_trace: {error}", file=sys.stderr)
            return 1
        print(f"SPA_SCENARIO_TRACE_OK snapshots={result['snapshots']} forks={result['alternatives']} measurements={result['measurements']}")
        return 0
    names = ("notorch.c", "notorch.h", "chuck_architect.h", "chuck_architect_impl.h",
             "spa_agent.c", "spa_agent.h", "examples/spa_agent_demo.c", "examples/spa_agent_scenarios.h",
             "experiments/spa_agent/scenarios/run.py", "experiments/spa_agent/scenarios/protocol.json",
             "experiments/spa_agent/run.py", "experiments/spa_agent/protocol.json", "tests/test_spa_scenarios.py")
    originals = {name: (ROOT / name).read_bytes() for name in names}
    report = {"fixture": "Synthetic embedding rows and deterministic generated tokens; production native action/snapshot/metric/restore code.",
        "source_sha256": {name: hashlib.sha256(data).hexdigest() for name, data in originals.items()},
        "protocol_sha256": SCENARIOS.V1.digest(SCENARIOS.PROTOCOL), "parser_gates": [], "path_gates": [], "mutations": [], "passed": False}
    compiler = shlex.split(os.environ.get("CC", "cc"))
    flags = shlex.split(os.environ.get("SPA_SCENARIO_TEST_CFLAGS", "-O2 -std=gnu11 -pthread"))
    with tempfile.TemporaryDirectory(prefix="notorch-spa-scenarios-") as directory:
        temp = Path(directory)
        for name, raw in originals.items():
            path = temp / name; path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes(raw)
        fixture = temp / "fixture.c"; fixture.write_text(FIXTURE)
        report["core_build"] = invoke(compiler + flags + ["-I", temp, "-c", temp / "notorch.c", "-o", temp / "notorch.o"])
        report["agent_build"] = invoke(compiler + flags + ["-I", temp, "-c", temp / "spa_agent.c", "-o", temp / "spa_agent.o"])
        if report["core_build"]["returncode"] or report["agent_build"]["returncode"]:
            print("FAIL scenario fixture dependency compilation")
            print(report["core_build"]["stderr"] + report["agent_build"]["stderr"])
        else:
            header_path = temp / "examples/spa_agent_scenarios.h"
            header = originals["examples/spa_agent_scenarios.h"].decode()

            def trial(name, changed, seed=42):
                header_path.write_text(changed)
                binary = temp / name
                build = invoke(compiler + flags + ["-I", temp, fixture, temp / "spa_agent.o", temp / "notorch.o", "-lm", "-o", binary])
                row = {"build": build}
                scenario_path, main_path = temp / f"{name}.jsonl", temp / f"{name}-ordinary.jsonl"
                if build["returncode"] == 0:
                    row["run"] = invoke([binary, scenario_path, main_path, str(seed)])
                    if row["run"]["returncode"] == 0:
                        row["validation"] = invoke([sys.executable, temp / "tests/test_spa_scenarios.py", "--validate", scenario_path, main_path])
                return row, scenario_path, main_path

            baseline, scenario_path, main_path = trial("baseline", header)
            report["baseline"] = baseline
            baseline_ok = (baseline["build"]["returncode"] == 0 and baseline.get("run", {}).get("returncode") == 0
                           and baseline.get("validation", {}).get("returncode") == 0)
            if not baseline_ok:
                print("FAIL unmodified scenario baseline")
                print(json.dumps(baseline, indent=2))
            else:
                print("PASS unmodified native fixture: 24 snapshots,60 forks,180 measurements")
                existing_scenario, existing_main = scenario_path.read_bytes(), main_path.read_bytes()
                collision = invoke([temp / "baseline", scenario_path, main_path, "42"])
                collision["caught"] = (collision["returncode"] == 1
                    and "cannot create new scenario trace" in collision["stderr"]
                    and scenario_path.read_bytes() == existing_scenario and main_path.read_bytes() == existing_main)
                report["existing_sink_refusal"] = collision
                print("PASS existing sink preserved" if collision["caught"] else "FAIL existing sink overwritten")
                def path_gate(name, scenario, host, expected, preserve=()):
                    saved = [(path, path.read_bytes(), path.is_symlink()) for path in preserve]
                    record = invoke([temp / "baseline", "--path-guard", scenario, host])
                    record["name"] = name
                    record["caught"] = (record["returncode"] == (1 if expected else 0)
                        and (expected in record["stderr"] if expected else "SPA_SCENARIO_PATH_OK" in record["stdout"])
                        and all(path.read_bytes() == raw and path.is_symlink() == symlink for path, raw, symlink in saved))
                    report["path_gates"].append(record)
                    print(("PASS path guard " if record["caught"] else "FAIL path guard ") + name)
                host_sentinel = temp / "host-sentinel.jsonl"
                host_sentinel.write_bytes(b"ordinary host output must survive\n")
                path_gate("distinct_outputs", temp / "distinct-scenario.jsonl", host_sentinel, "", (host_sentinel,))
                path_gate("dot_trace_alias", f"{temp}/./dot-trace.jsonl", temp / "dot-trace.jsonl",
                          "scenario trace aliases a host output")
                path_gate("future_life_alias", f"{temp}/./future.learned.life.bin", temp / "future.learned.life.bin",
                          "scenario trace aliases a host output")
                existing_link = temp / "existing-scenario-link.jsonl"
                existing_link.symlink_to(host_sentinel)
                path_gate("existing_symlink_sink", existing_link, temp / "unused-host.jsonl",
                          "cannot create new scenario trace", (existing_link, host_sentinel))
                new_scenario = temp / "new-scenario.jsonl"
                future_host_link = temp / "future-host-link.jsonl"
                future_host_link.symlink_to(new_scenario)
                path_gate("symlink_host_alias", new_scenario, future_host_link,
                          "scenario trace aliases a host output")
                rows, ordinary = SCENARIOS.read_events(scenario_path), SCENARIOS.read_events(main_path)
                def measurement(value):
                    return next(r for r in value if r["type"] == "measurement" and r["horizon"] == 4)
                def future(value):
                    return next(r for r in value if r["type"] == "measurement" and r["horizon"] == 1)["continuation"][0]
                parser_specs = [
                    ("missing_measurement", lambda x: x.remove(measurement(x)), "missing valid action/horizon"),
                    ("duplicate_measurement", lambda x: x.insert(1, copy.deepcopy(measurement(x))), "duplicate action/horizon"),
                    ("nonfinite_axis", lambda x: measurement(x)["after"].__setitem__("coherence", float("nan")), "non-finite"),
                    ("cost_denominator", lambda x: measurement(x).__setitem__("cost_denominator", 64), "denominator"),
                    ("future_rng", lambda x: future(x).__setitem__("rng_before", "0000000000000001"), "future RNG"),
                    ("reward", lambda x: measurement(x).__setitem__("reward", .75), "reward differs"),
                    ("body_identity", lambda x: measurement(x).__setitem__("body_hash", "0000000000000001"), "body_hash"),
                    ("restoration", lambda x: next(r for r in x if r["type"] == "restoration").__setitem__("chain_unchanged", False), "main state leaked"),
                ]
                for name, alter, expected in parser_specs:
                    record = parser_gate(name, rows, ordinary, alter, expected)
                    report["parser_gates"].append(record)
                    print(("PASS caught " if record["caught"] else "FAIL escaped/wrong ") + name + ": " + record["failure"])
                mutants = [
                    {"name": "host_chain_leak", "anchor": "// SPA_SCENARIO_MUTATE_HOST_CHAIN:",
                     "injection": "chain[0].ids[0] ^= 1;\n        ", "needle": "scenario leaked host chain", "failure_stage": "run"},
                    {"name": "future_rng_pairing", "anchor": "// SPA_SCENARIO_MUTATE_FUTURE_RNG:",
                     "injection": "paired_rng ^= 1;\n            ", "needle": "diagnostic future RNG pairing mismatch", "failure_stage": "run"},
                    {"name": "consequence_sign", "anchor": "// SPA_SCENARIO_MUTATE_REWARD:",
                     "injection": "reward = -reward;\n    ", "needle": "reward differs", "failure_stage": "validation"},
                    {"name": "future_metric_target", "anchor": "if (hop == 1 || hop == 4) {\n                observation(body, branch, target, counts[target]",
                     "replacement": "if (hop == 1 || hop == 4) {\n                observation(body, branch, future_target, counts[future_target]",
                     "needle": "repetition metric uses wrong target", "failure_stage": "validation"},
                ]
                for spec in mutants:
                    matches = header.count(spec["anchor"])
                    if matches != 1:
                        record = {"name": spec["name"], "caught": False, "error": f"expected one anchor,found{matches}"}
                    else:
                        replacement = spec.get("replacement", spec.get("injection", "") + spec["anchor"])
                        changed = header.replace(spec["anchor"], replacement, 1)
                        record, _, _ = trial(spec["name"], changed)
                        failure = record.get(spec["failure_stage"], {})
                        record.update({"name": spec["name"], "mutant_sha256": hashlib.sha256(changed.encode()).hexdigest(),
                            "caught": record["build"]["returncode"] == 0 and failure.get("returncode") == 1
                                      and spec["needle"] in failure.get("stderr", "")})
                    report["mutations"].append(record)
                    print(("PASS caught " if record["caught"] else "FAIL escaped/invalid ") + spec["name"])
                    if not record["caught"]: print(json.dumps(record, indent=2))
                report["passed"] = collision["caught"] and all(r["caught"] for r in report["parser_gates"] + report["path_gates"] + report["mutations"])
    report["working_sources_unchanged"] = all((ROOT / name).read_bytes() == raw for name, raw in originals.items())
    report["passed"] = report["passed"] and report["working_sources_unchanged"]
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(report, indent=2) + "\n")
    if report["passed"]: print("SPA_SCENARIO_GATES_OK parser_mutations=8 compiled_mutations=4 path_gates=5")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
