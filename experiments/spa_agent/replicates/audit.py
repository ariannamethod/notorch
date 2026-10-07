#!/usr/bin/env python3
"""Independent synthetic audit; no real-model generation or experiment fitting.

Builds isolated native defects, checks the complete cheap host fixture, and
compares freshly rebuilt canonical v1 lives with the preregistered parent.
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
import traceback

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
PROTOCOL_SHA = "c53b00f02ebbc8d7bd1144839719741887f0a0e3ac411767a3fde7359a59db75"
BASELINE = "b2891195abd0a318e50d7a445e48d3ed2e6dbcf9"
COMPILE_FILES = ("notorch.c", "notorch.h", "notorch_simd.h", "spa_agent.c", "spa_agent.h",
    "chuck_architect.h", "chuck_architect_impl.h", "examples/spa_agent_demo.c",
    "examples/spa_agent_scenarios.h", "examples/spa_agent_replicates.c",
    "tests/spa_replicates_fixture.c")
AUDITED_FILES = (*COMPILE_FILES, "experiments/spa_agent/replicates/protocol.json",
    "experiments/spa_agent/replicates/run.py", "experiments/spa_agent/replicates/policy.py",
    "tests/test_spa_replicates.py", "python/SPA.py", "python/notorch.py",
    "spa_binding.c", "spa_binding.h", "tests/test_spa_agent_repeated.c",
    "tests/test_spa_future_mutations.py", "experiments/spa_agent/replicates/audit.py")

PRECISION_GATE = r'''#include "spa_agent.h"
#include <math.h>
#include <stdio.h>
#include <string.h>
int main(void) {
    nt_spa_agent_config cfg; nt_spa_agent a,original;
    nt_spa_observation o; nt_spa_experience e;
    nt_spa_comparison c[64]; nt_spa_comparison_receipt result;
    unsigned counts[]={1,2,3,8,64},i,j,n;
    nt_spa_agent_config_default(&cfg); cfg.mode=NT_SPA_AGENT_LEARNED;
    memset(&cfg.reward_weights,0,sizeof(cfg.reward_weights)); cfg.reward_weights.coherence=1; cfg.cost_weight=0;
    memset(&o,0,sizeof(o)); o.temperature=1; o.sentence_index=1; o.sentence_count=2;
    if(nt_spa_agent_init(&a,&cfg) || nt_spa_agent_capture_experience(&a,&o,&e)) return 2;
    original=a; memset(c,0,sizeof(c));
    for(i=0;i<64;i++) {
        c[i].source_life_hash=e.source_life_hash;c[i].action_mask=3;c[i].horizon=4;
        c[i].alternatives[0].action=(nt_spa_action){NT_SPA_KEEP,1,NT_SPA_AGENT_NO_SOURCE};
        c[i].alternatives[1].action=(nt_spa_action){NT_SPA_RESEED_LEFT,1,0};
        c[i].alternatives[1].consequence.after.coherence=i ? 0x1p-24f : 1.0f;
    }
    for(j=0;j<5;j++) {
        float expected; n=counts[j]; expected=(float)((1.0+(n-1)*0x1p-24)/n);
        if(nt_spa_agent_fit_repeated(&a,&e,c,n,0,&result)) return 2;
        if(memcmp(&a,&original,sizeof(a))) return 2;
        if(result.rewards[1]!=expected || result.targets[1]!=expected) {
            fprintf(stderr,"FAIL native_double_accumulation: count=%u expected=%a actual=%a\n",n,(double)expected,(double)result.rewards[1]);return 1;
        }
    }
    puts("PASS native_double_accumulation: counts1/2/3/8/64, exact mean bits, rate0 life unchanged");return 0;
}
'''


def sha(raw): return hashlib.sha256(raw).hexdigest()
def require(condition, message):
    if not condition: raise AssertionError(message)


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec); spec.loader.exec_module(value)
    return value


def invoke(command, cwd, timeout=180):
    result = subprocess.run(list(map(str, command)), cwd=cwd, capture_output=True,
                            text=True, timeout=timeout, check=False)
    return {"command": list(map(str, command)), "returncode": result.returncode,
            "stdout": result.stdout, "stderr": result.stderr}


def passed(record, label):
    require(record["returncode"] == 0, label + " failed: " + record["stderr"])


def portable(value, temporary):
    if isinstance(value, dict): return {k: portable(v, temporary) for k,v in value.items()}
    if isinstance(value, list): return [portable(v, temporary) for v in value]
    if isinstance(value, str):
        return value.replace(str(temporary), "<temporary>").replace(str(ROOT), "<source>").replace(sys.executable, "python3")
    return value


def fixture_inputs(runner, prefix):
    samples = [runner.source_from_row(json.loads(line)) for line in Path(str(prefix)+".sources.jsonl").read_bytes().splitlines()]
    refs = {}
    for line in Path(str(prefix)+".expected_r0.jsonl").read_bytes().splitlines(keepends=True):
        row = json.loads(line); refs[row["seed"],row["snapshot"],row["action"]["kind"],row["horizon"]] = line
    return samples, refs


def validate_host_child(prefix, raw_path, gate):
    runner = module("spa_independent_host_child", HERE/"run.py")
    samples, refs = fixture_inputs(runner, Path(prefix))
    try:
        runner.validate_replicates(Path(raw_path).read_bytes(), samples, refs)
    except (AssertionError, ValueError) as error:
        print("FAIL " + gate + ": " + str(error), file=sys.stderr)
        return 1
    print("PASS " + gate)
    return 0


def chain_corruptions(runner, raw, samples, refs):
    results = []
    for name, horizon, failure in (("untouched_h1_sentence",1,"replicate untouched future sentence changed"),
                                    ("h4_generation_length",4,"replicate future token accounting")):
        lines = raw.splitlines(keepends=True); replica = None; changed = False
        for i, line in enumerate(lines):
            row = json.loads(line)
            if row["type"] == "replicate_begin": replica = row["replicate"]
            if row["type"] == "measurement" and replica == 1 and row["snapshot"] == 0 and row["action"]["kind"] == 0 and row["horizon"] == horizon:
                if horizon == 1: row["chain"][2]["tokens"][0] = (row["chain"][2]["tokens"][0]+1)%94
                else: row["chain"][3]["tokens"].pop(0); row["chain"][3]["length"] -= 1
                lines[i] = json.dumps(row, separators=(",", ":")).encode()+b"\n"; changed = True; break
        require(changed, "corruption target missing")
        try: runner.validate_replicates(b"".join(lines), samples, refs)
        except (AssertionError, ValueError) as error:
            require(failure in str(error), "unexpected corruption failure: " + str(error))
            results.append({"name":name, "caught":True, "failure":str(error)})
        else: raise AssertionError("uncaught trace corruption: " + name)
    return results


def perform(temporary, report):
    originals = {name:(ROOT/name).read_bytes() for name in AUDITED_FILES}
    require(sha(originals["experiments/spa_agent/replicates/protocol.json"]) == PROTOCOL_SHA, "registered protocol changed")
    report["source_sha256"] = {name:sha(raw) for name,raw in originals.items()}
    report["baseline_commit"] = BASELINE
    compiler = shlex.split(os.environ.get("CC", "cc"))
    base = temporary/"base"
    for name in COMPILE_FILES:
        path = base/name; path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes(originals[name])
    build = invoke([*compiler,"-std=c11","-O2","-pthread","-I",base,"-c",base/"notorch.c","-o",base/"notorch.o"], ROOT)
    passed(build,"shared native arithmetic compile"); report["notorch_object_build"] = build
    flags = [*compiler,"-std=c11","-O2","-Wall","-Wextra","-Werror","-pedantic","-pthread","-I",base]
    def host_build(name, text):
        directory = temporary/name
        for path in ("examples/spa_agent_replicates.c","examples/spa_agent_demo.c","examples/spa_agent_scenarios.h","tests/spa_replicates_fixture.c"):
            target=directory/path;target.parent.mkdir(parents=True,exist_ok=True)
            target.write_bytes(text if path=="examples/spa_agent_replicates.c" else originals[path])
        binary=directory/"fixture"
        result=invoke([*flags,directory/"tests/spa_replicates_fixture.c",base/"spa_agent.c",base/"notorch.o","-lm","-o",binary],ROOT)
        passed(result,"host "+name+" compile")
        return binary,result
    host_source=originals["examples/spa_agent_replicates.c"].decode()
    binary,built=host_build("host-baseline",host_source.encode())
    prefix=temporary/"valid"
    written=invoke([binary,"write",prefix],ROOT);passed(written,"baseline source construction")
    output=temporary/"valid.output.jsonl"
    baseline=invoke([binary,"run",str(prefix)+".sources.txt",output],ROOT);passed(baseline,"baseline synthetic host")
    cases=invoke([binary,"cases",temporary/"cases"],ROOT);passed(cases,"strict source/output fixture")
    runner=module("spa_independent_repeated_runner",HERE/"run.py")
    samples,refs=fixture_inputs(runner,prefix);raw=output.read_bytes()
    _,checked=runner.validate_replicates(raw,samples,refs)
    report["host_baseline"]={"build":built,"write":written,"run":baseline,"strict_cases":cases,
        "raw_sha256":sha(raw),"source_bytes_sha256":sha(Path(str(prefix)+".sources.txt").read_bytes()),
        "validation":checked}
    report["fixed_discovered_trace_gaps"]={"before_fix":"Both corruptions were accepted during the independent pre-freeze review on the same genuine cheap fixture.",
        "current_red_hands":chain_corruptions(runner,raw,samples,refs)}
    print("PASS independent_host_baseline_and_chain_boundaries",flush=True)
    mutations=[
        ("initial_rng_pairing","uint64_t initial_rng = starts[0];","uint64_t initial_rng = starts[0] + 1;","replicate initial RNG pairing mismatch"),
        ("future_rng_pairing","uint64_t future_rng = starts[hop];","uint64_t future_rng = starts[hop] + 1;","replicate future RNG pairing mismatch"),
        ("source_restoration","// SPA_REPLICATES_MUTATE_SOURCE:","((rep_source *)source)->experience.features[19] += 0.01f;\n                // SPA_REPLICATES_MUTATE_SOURCE:","replicate branch leaked into its fixed source"),
        ("output_preservation",'fopen(path, "wx")','fopen(path, "w")',"fixture expected a named normal-exit refusal"),
        ("measurement_target","// SPA_REPLICATES_MUTATE_TARGET: measure the original intervention target.\n                        observation(body, branch, target, counts[target], &observed, &after);",
         "// SPA_REPLICATES_MUTATE_TARGET: deliberate wrong measurement target.\n                        observation(body, branch, (target+1)%SENTENCES, counts[(target+1)%SENTENCES], &observed, &after);",None)]
    report["compiled_host_mutations"]=[]
    for name,anchor,replacement,expected in mutations:
        require(host_source.count(anchor)==1,"mutation anchor moved: "+name)
        changed=host_source.replace(anchor,replacement).encode()
        mutant,build=host_build(name,changed)
        trace=temporary/(name+".jsonl")
        command=[mutant,"cases",temporary/(name+"-cases")] if name=="output_preservation" else [mutant,"run",str(prefix)+".sources.txt",trace]
        run=invoke(command,ROOT)
        record={"name":name,"mutant_source_sha256":sha(changed),"build":build,"run":run}
        if expected:
            require(run["returncode"]==1 and expected in run["stderr"],"compiled defect escaped named native gate: "+name)
            record["gate"]=expected
        else:
            passed(run,"wrong-target mutant must execute normally")
            gate=invoke([sys.executable,Path(__file__),"--validate-host",prefix,trace,"host_measurement_target"],ROOT)
            require(gate["returncode"]==1 and "FAIL host_measurement_target:" in gate["stderr"]
                and ("replicate repetition metric uses wrong target" in gate["stderr"] or "replicate r0 parent raw-line parity" in gate["stderr"]),"wrong target escaped independent gate")
            record["gate"]="host_measurement_target";record["independent_gate"]=gate
        record["caught"]=True;report["compiled_host_mutations"].append(record)
        print("PASS compiled_"+name,flush=True)
    driver=temporary/"precision.c";driver.write_text(PRECISION_GATE)
    core_source=originals["spa_agent.c"].decode()
    anchor="double sums[NT_SPA_AGENT_ACTIONS]={0};"
    require(core_source.count(anchor)==1,"native sum anchor moved")
    precision=[]
    for name,source in (("baseline",core_source),("float_sum",core_source.replace(anchor,"float sums[NT_SPA_AGENT_ACTIONS]={0};"))):
        path=temporary/("precision-"+name+".c");path.write_text(source);out=temporary/("precision-"+name)
        built=invoke([*flags,driver,path,base/"notorch.o","-lm","-o",out],ROOT);passed(built,"precision compile "+name)
        ran=invoke([out],ROOT)
        require((ran["returncode"]==0 and "PASS native_double_accumulation" in ran["stdout"]) if name=="baseline" else
                (ran["returncode"]==1 and "FAIL native_double_accumulation:" in ran["stderr"]),"precision gate "+name)
        precision.append({"name":name,"source_sha256":sha(source.encode()),"build":built,"run":ran})
    report["native_double_accumulation"]={"driver_sha256":sha(PRECISION_GATE.encode()),"executions":precision,"float_sum_mutant_caught":True}
    print("PASS native_double_accumulation_and_mutant",flush=True)
    legacy=module("spa_independent_v1",ROOT/"tests/test_spa_future_mutations.py")
    legacy.BASELINE=BASELINE
    report["v1_compatibility"]=legacy.v1_compatibility(compiler,temporary,{name:originals[name] for name in legacy.SOURCES})
    require(report["v1_compatibility"]["passed"],"fresh v1 canonical compatibility")
    report["sources_unchanged"] = all((ROOT/name).read_bytes()==raw for name,raw in originals.items())
    require(report["sources_unchanged"],"audited sources changed during preflight")
    report["review"]={"mean_native_reward":"All members validate before mutation; each reward clips before double accumulation, one float mean and one simultaneous update; count1 delegates unchanged comparison API.",
        "source_only_readout":"policy.score_samples reads features and source identities. Parent preflight separately poisons/removes outcomes, verifies unchanged complete lives and all192 readouts.",
        "barrier":"Runner anchors both training datasets, both complete fit traces, both seals and all8 life files in an in-memory barrier before any new-seed process; it rechecks the barrier and sources/binaries between stages.",
        "fixed_helper_gaps":["snapshot_witness now requires 16 lowercase hexadecimal characters", "verify_seal now pins the historical single_h4 SHA and native life hash", "immutable binary identities now participate in runner stable()"],
        "sample_unit":"48 captured source states; eight paired realizations per state remain nested outcomes."}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json",type=Path)
    parser.add_argument("--validate-host",nargs=3,metavar=("FIXTURE_PREFIX","RAW_TRACE","GATE"))
    args=parser.parse_args()
    if args.validate_host:return validate_host_child(*args.validate_host)
    require(args.json is not None,"--json is required")
    report={"schema":1,"status":"RUNNING","operation":"Independent synthetic repeated-consequence preflight",
        "protocol_sha256":PROTOCOL_SHA,"body_generation_runs":0,"experiment_training_runs":0,
        "command":"python3 experiments/spa_agent/replicates/audit.py --json experiments/spa_agent/replicates/audit.json",
        "script_sha256":sha(Path(__file__).read_bytes())}
    code=0
    with tempfile.TemporaryDirectory(prefix="spa-replicate-audit-") as tmp:
        temporary=Path(tmp)
        try:perform(temporary,report);report["status"]="PASS"
        except Exception as error:
            report["status"]="FAIL";report["error"]={"type":type(error).__name__,"message":str(error)}
            traceback.print_exc();code=1
        report=portable(report,temporary)
    args.json.parent.mkdir(parents=True,exist_ok=True)
    args.json.write_text(json.dumps(report,indent=2,ensure_ascii=False)+"\n")
    print("independent repeated audit",report["status"],"receipt_sha256",sha(args.json.read_bytes()),flush=True)
    return code


if __name__=="__main__":raise SystemExit(main())
