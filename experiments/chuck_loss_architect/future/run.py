#!/usr/bin/env python3
"""Fixed future-credit replay, independent body adaptation, and sealed evaluation."""
from __future__ import annotations
import argparse
import collections
import importlib.util
import json
import os
from pathlib import Path
import shutil
import struct
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
spec = importlib.util.spec_from_file_location("chuck_scenarios", HERE.parent / "scenarios/run.py")
scenarios = importlib.util.module_from_spec(spec)
assert spec and spec.loader
spec.loader.exec_module(scenarios)
v1 = scenarios.v1
FLOAT_FIELDS = ("loss", "loss_ema", "loss_trend", "macro_ema", "best_macro", "dampen", "lr_scale", "noise", "grad_norm", "grad_trend", "frozen_fraction")
INT_FIELDS = ("step", "stag", "macro_stag", "history_len")
ACTIONS = ("hold", "brake", "push")


def authenticate_previous(previous: Path, result: dict) -> int:
    for name, recorded in result["artifacts"].items():
        path = previous / name
        assert path.stat().st_size == recorded["bytes"] and v1.digest(path) == recorded["sha256"], "previous artifact identity mismatch: " + name
    return len(result["artifacts"])


def execute(command: list[str], log: Path, env: dict) -> None:
    code = scenarios.execute(command, log, env)
    if code: raise RuntimeError(f"command failed with exit {code}; see {log.name}")


def old_association(current: list[dict], old: list[dict], prefix: Path, old_prefix: Path, steps: int) -> dict:
    for kind in ("host_step", "fork", "branch_step", "comparison", "restore"):
        left = [r for r in old if r["type"] == kind and r.get("step", r.get("checkpoint", 0)) <= steps]
        right = [r for r in current if r["type"] == kind]
        assert len(left) == len(right), (kind, len(left), len(right))
        for a, b in zip(left, right):
            assert a == {key: b[key] for key in a}, f"old {kind} association mismatch"
    compared = []
    if steps == 512:
        for suffix in (".final.bin", ".moments.final.bin", ".optimizer.final.json", ".policy.final.bin"):
            assert Path(str(prefix) + suffix).read_bytes() == Path(str(old_prefix) + suffix).read_bytes(), suffix
            compared.append(suffix)
    for r in current:
        if r["type"] != "fork": continue
        for suffix in (".body.bin", ".gradients.bin", ".moments.final.bin", ".optimizer.final.json", ".policy.bin"):
            part = f".fork-{r['step']}" + suffix
            assert Path(str(prefix) + part).read_bytes() == Path(str(old_prefix) + part).read_bytes(), part
    return {"status": "PASS", "old_host_steps_compared": steps, "forks_compared": sum(r["type"] == "fork" for r in current),
            "final_files_compared": compared, "old_trace_sha256": v1.digest(Path(str(old_prefix) + ".jsonl"))}


def write_samples(path: Path, samples: list[dict]) -> None:
    """Canonical LE table: magic, count; IIIQQ, 11f, 4I, 16f, 3f, path length/path."""
    with path.open("wb") as f:
        f.write(b"NTCAFT01" + struct.pack("<I", len(samples)))
        for s in samples:
            o = s["observation"]
            f.write(struct.pack("<IIIQQ", 1 if s["body"] == "simple" else 2, s["seed"], s["checkpoint"], int(s["state_hash"],16), int(s["policy_hash"],16)))
            f.write(struct.pack("<11f4I", *(o[k] for k in FLOAT_FIELDS), *(o[k] for k in INT_FIELDS)))
            f.write(struct.pack("<16f3f", *s["features"], *s["future_loss"]))
            name = s["source_policy"].encode("ascii")
            f.write(struct.pack("<I", len(name)) + name)
    v1.write_json(path.with_suffix(".json"), samples)


def gather_samples(rows: list[dict], body: str, seed: int, prefix: Path, output: Path, outcome_rows: list[dict] | None = None) -> list[dict]:
    outcomes = rows if outcome_rows is None else outcome_rows
    result = []
    for fork in [r for r in rows if r["type"] == "fork"]:
        arms = {r["action"]: r for r in outcomes if r["type"] == "comparison" and r["checkpoint"] == fork["step"] and r["horizon"] == 16}
        assert set(arms) == set(ACTIONS)
        assert {r["initial_hash"] for r in arms.values()} == {fork["state_hash"]}
        result.append({"body": body, "seed": seed, "checkpoint": fork["step"], "state_hash": fork["state_hash"],
                       "policy_hash": fork["policy_hash"], "observation": fork["observation"], "features": fork["features"],
                       "future_loss": [arms[a]["future_after"] for a in ACTIONS],
                       "source_policy": Path(str(prefix)+f".fork-{fork['step']}.policy.bin").relative_to(output).as_posix(),
                       "outcome_source": "verified_old_measurement" if outcome_rows is not None else "fresh_common_state_branches",
                       "arms": arms})
    return result


def readouts(binary: Path, table: Path, samples: list[dict], lives: dict[str, Path], output: Path, split: str, env: dict) -> list[dict]:
    results = []
    lookup = {(s["body"],s["seed"],s["checkpoint"]):s for s in samples}
    for label in ("initial", "simple", "adapted", "source", "hold", "push"):
        trace = output / f"{split}-{label}.readout.jsonl"
        execute([str(binary), "eval", str(table), str(lives[label]) if label in lives else "-", label, str(trace)],
                output / f"{split}-{label}.readout.log", env)
        rows = scenarios.events(trace)
        assert len(rows) == len(samples)
        for r in rows:
            s = lookup[(r["body"],r["seed"],r["checkpoint"])]
            assert r["state_hash"] == s["state_hash"] and r["source_policy_hash"] == s["policy_hash"]
            action = ACTIONS[r["action"] - 1]
            as_f32 = lambda value: struct.unpack("<f", struct.pack("<f",value))[0]
            assert as_f32(r["future_loss"]) == as_f32(s["arms"][action]["future_after"])
            r["split"] = split
            r["immediate_after"] = s["arms"][action]["immediate_after"]
            r["origin_after_horizon"] = s["arms"][action]["origin_after_horizon"]
            r["heldout_after"] = s["arms"][action]["heldout_after"]
            results.append(r)
    return results


def summarize(rows: list[dict]) -> list[dict]:
    result = []
    indexed = {(r["split"],r["body"],r["seed"],r["checkpoint"],r["label"]):r for r in rows}
    for split in sorted({r["split"] for r in rows}):
        for body in ("simple","hevlm"):
            for label in ("initial","simple","adapted","source","hold","push"):
                group = [r for r in rows if (r["split"],r["body"],r["label"]) == (split,body,label)]
                if not group: continue
                changes = {}
                for baseline in ("initial","simple"):
                    changes[baseline] = sum(r["action"] != indexed[(split,body,r["seed"],r["checkpoint"],baseline)]["action"] for r in group)
                result.append({"split":split,"body":body,"label":label,"states":len(group),
                               "actions":dict(collections.Counter(ACTIONS[r["action"]-1] for r in group)),
                               "optimal":sum(r["optimal"] for r in group), "mean_regret":sum(r["regret"] for r in group)/len(group),
                               "mean_advantage_vs_hold":sum(r["advantage"] for r in group)/len(group),
                               "choice_changes":changes})
    return result


def verify_cohort_traces(cohorts: list[dict], output: Path, steps: int) -> dict:
    """Re-read persisted receipts after fitting/readout, before final success."""
    for cohort in cohorts:
        prefix = output / "cohorts" / f"{cohort['body']}-s{cohort['seed']}"
        control = Path(str(prefix) + "-control")
        diagnostic = Path(str(prefix) + "-diagnostic")
        control_path = Path(str(control) + ".jsonl")
        diagnostic_path = Path(str(diagnostic) + ".jsonl")
        assert v1.digest(diagnostic_path) == cohort["trace_sha256"], "persisted diagnostic trace identity changed: " + diagnostic_path.name
        if "control_trace_sha256" in cohort:
            assert v1.digest(control_path) == cohort["control_trace_sha256"], "persisted control trace identity changed: " + control_path.name
        left, right = scenarios.events(control_path), scenarios.events(diagnostic_path)
        assert left[-1] == cohort["control_summary"] and right[-1] == cohort["diagnostic_summary"], "persisted cohort summary changed"
        for rows in (left, right):
            assert [r["step"] for r in rows if r["type"] == "host_step"] == list(range(1, steps + 1)), "incomplete persisted host trajectory"
        assert scenarios.compare_hosts(left, right, control, diagnostic, steps) == cohort["parity"], "persisted host parity changed"
    return {"cohorts_verified":len(cohorts), "host_steps_compared":len(cohorts)*steps,
            "summaries_and_recorded_trace_hashes_match":True, "final_state_bytes_match":True}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--previous-run",type=Path,required=True,help="expanded previous common-state raw run, containing results.json")
    parser.add_argument("--references",type=Path)
    parser.add_argument("--allow-dirty",action="store_true")
    parser.add_argument("--smoke",action="store_true")
    args = parser.parse_args()
    output=args.output.resolve(); previous=args.previous_run.resolve()
    if output == ROOT or ROOT in output.parents: parser.error("output must be outside Git")
    output.mkdir(parents=True,exist_ok=True)
    if (output/"manifest.json").exists(): parser.error("choose a fresh output directory")
    protocol=scenarios.strict_json((HERE/"protocol.json").read_text())
    if v1.digest(previous/"results.json") != protocol["old_results_sha256"]: parser.error("previous results identity mismatch")
    old_result=scenarios.strict_json((previous/"results.json").read_text())
    previous_artifacts_verified=authenticate_previous(previous,old_result)
    status=v1.command(["git","status","--porcelain"])
    if status and not args.allow_dirty: parser.error("final run requires clean committed source")
    if args.smoke and not args.allow_dirty: parser.error("smoke requires explicit --allow-dirty")
    steps=8 if args.smoke else protocol["host_steps"]
    old_seeds=[42] if args.smoke else protocol["training"]["seeds"]
    eval_seeds=[protocol["future_smoke_evaluation_seed"]] if args.smoke else protocol["evaluation"]["seeds"]
    datasets={body:v1.prepare(body,output,args.references) for body in ("simple","hevlm")}
    for body in datasets: assert datasets[body] == old_result["manifest"]["datasets"][body]
    source_files=["notorch.c","notorch.h","notorch_simd.h","chuck_architect.h","chuck_architect_impl.h",
                  "examples/chuck_architect_train.c","examples/chuck_architect_scenarios.h","examples/chuck_architect_future.c",
                  "examples/chuck-loss-architect.json","experiments/chuck_loss_architect/run.py",
                  "experiments/chuck_loss_architect/scenarios/run.py","experiments/chuck_loss_architect/scenarios/protocol.json",
                  "experiments/chuck_loss_architect/future/run.py","experiments/chuck_loss_architect/future/protocol.json"]
    hashes={name:v1.digest(ROOT/name) for name in source_files}
    host_binary=output/"host_runner"; future_binary=output/"future_runner"
    commands=[scenarios.build(host_binary),scenarios.build(future_binary,ROOT/"examples/chuck_architect_future.c")]
    manifest={"source_commit":v1.command(["git","rev-parse","HEAD"]),"source_status":status,"source_sha256":hashes,
              "protocol":protocol,"protocol_sha256":v1.digest(HERE/"protocol.json"),"datasets":datasets,"machine":v1.machine(),
              "compiler":v1.command(["cc","--version"]).splitlines()[0],"compile_flags":"-O2 -std=gnu11 -DUSE_SIMD -march=native -pthread",
              "binaries_sha256":{"host":v1.digest(host_binary),"future":v1.digest(future_binary)},
              "smoke":args.smoke,"executed_host_steps":steps,"executed_old_seeds":old_seeds,"executed_eval_seeds":eval_seeds,
              "previous_results_sha256":v1.digest(previous/"results.json"),"previous_source_commit":old_result["manifest"]["source_commit"]}
    manifest["previous_artifacts_verified"]=previous_artifacts_verified
    manifest["prior_smoke_exposure"]=protocol["prior_smoke_exposure"]
    v1.write_json(output/"manifest.json",manifest); shutil.copyfile(HERE/"protocol.json",output/"protocol.json")
    config=ROOT/"examples/chuck-loss-architect.json"; shutil.copyfile(config,output/"architect-config.json")
    env=dict(os.environ,NT_SIMD_THREADS="2",OMP_NUM_THREADS="2",OPENBLAS_NUM_THREADS="2")
    cohort_dir=output/"cohorts";cohort_dir.mkdir()
    scenario_protocol=scenarios.strict_json((HERE.parent/"scenarios/protocol.json").read_text())
    cohorts=[]; training=[]; evaluation=[]; all_comparisons=[]
    timeline=output/"timeline.jsonl"
    def phase(name, **fields):
        with timeline.open("a") as f: f.write(json.dumps({"phase":name,"time_ns":time.time_ns(),**fields})+"\n")
    def cohort(body,seed,replay):
        prefixes={};traces={}
        for probes,label in ((0,"control"),(1,"diagnostic")):
            name=f"{body}-s{seed}-{label}";prefix=cohort_dir/name
            print("RUN",name,flush=True)
            execute([str(host_binary),"--scenarios",body,str(output/f"{body}.tokens.u32"),str(prefix),str(steps),str(seed),
                     str(protocol["body_lr"]),str(config),str(probes)],cohort_dir/(name+".log"),env)
            prefixes[label]=prefix;traces[label]=scenarios.events(Path(str(prefix)+".jsonl"))
        parity=scenarios.compare_hosts(traces["control"],traces["diagnostic"],prefixes["control"],prefixes["diagnostic"],steps)
        comparisons=scenarios.comparisons(traces["diagnostic"],body,seed,scenario_protocol,steps)
        all_comparisons.extend(comparisons)
        old_rows=None;association=None
        if replay:
            old_prefix=previous/f"{body}-s{seed}-diagnostic"
            old_rows=scenarios.events(Path(str(old_prefix)+".jsonl"))
            association=old_association(traces["diagnostic"],old_rows,prefixes["diagnostic"],old_prefix,steps)
        samples=gather_samples(traces["diagnostic"],body,seed,prefixes["diagnostic"],output,old_rows)
        cohorts.append({"body":body,"seed":seed,"replay":replay,"parity":parity,"old_association":association,
                        "control_summary":traces["control"][-1],"diagnostic_summary":traces["diagnostic"][-1],
                        "control_trace_sha256":v1.digest(Path(str(prefixes["control"])+".jsonl")),
                        "trace_sha256":v1.digest(Path(str(prefixes["diagnostic"])+".jsonl"))})
        print("COHORT_OK",body,seed,"old_association="+str(replay),flush=True)
        return samples
    phase("old_replay_started")
    for body in ("simple","hevlm"):
        for seed in old_seeds: training.extend(cohort(body,seed,True))
    phase("old_replay_complete",states=len(training))
    lives={name:output/(name+".policy.bin") for name in ("initial","simple","adapted")}
    execute([str(future_binary),"init",str(protocol["policy_seed"]),str(lives["initial"]),protocol["training"]["life_id"]],output/"initial.log",env)
    fit_receipts={}
    for body,label,parent,identity in (("simple","simple","initial",protocol["training"]["life_id"]),
                                       ("hevlm","adapted","simple",protocol["adaptation"]["life_id"])):
        samples=sorted((s for s in training if s["body"]==body),key=lambda s:(s["seed"],s["checkpoint"]))
        table=output/(label+".samples.bin");write_samples(table,samples)
        trace=output/(label+".fit.jsonl")
        execute([str(future_binary),"fit",str(lives[parent]),str(table),str(lives[label]),str(trace),str(protocol["epochs_per_stage"]),identity],output/(label+".fit.log"),env)
        rows=scenarios.events(trace);fits=[r for r in rows if r["type"]=="fit"]
        assert len(fits)==protocol["epochs_per_stage"]*len(samples)
        assert rows[-1]["non_weight_fields_unchanged"]
        for i,row in enumerate(fits):
            expected=samples[i%len(samples)]
            assert (row["body"],row["seed"],row["checkpoint"])==(expected["body"],expected["seed"],expected["checkpoint"])
            assert row["fit_step"]==i+1 and row["sample_index"]==i%len(samples)
            assert row["online_decisions"]==row["online_updates"]==0
        fit_receipts[label]={"fit_steps":len(fits),"epochs":protocol["epochs_per_stage"],"samples":len(samples),
                             "initial_hash":rows[0]["initial_hash"],"final_hash":rows[-1]["final_hash"],
                             "sample_table_sha256":v1.digest(table),"fit_trace_sha256":v1.digest(trace)}
    sealed={label:{"file":p.name,"sha256":v1.digest(p)} for label,p in lives.items()}
    seal={"protocol_sha256":manifest["protocol_sha256"],"lives":sealed,"fit_receipts":fit_receipts,
          "evaluation_generation_started":False}
    v1.write_json(output/"sealed_lives.json",seal);phase("lives_sealed",sealed_lives_sha256=v1.digest(output/"sealed_lives.json"))
    # A wrongly associated feature vector must fail before a fit output is saved.
    invalid=bytearray((output/"simple.samples.bin").read_bytes());struct.pack_into("<f",invalid,100,0.0)
    bad_table=output/"wrong-feature.samples.bin";bad_table.write_bytes(invalid)
    bad_life=output/"wrong-feature.policy.bin"
    exit_code=scenarios.execute([str(future_binary),"fit",str(lives["initial"]),str(bad_table),str(bad_life),str(output/"wrong-feature.fit.jsonl"),"1",protocol["training"]["life_id"]],output/"wrong-feature.log",env)
    assert exit_code!=0 and not bad_life.exists() and "source observation/features mismatch" in (output/"wrong-feature.log").read_text()
    gate={"feature_association_fault":{"caught":True,"exit":exit_code,"log_sha256":v1.digest(output/"wrong-feature.log")}}
    dev_table=output/"development.samples.bin";write_samples(dev_table,training)
    outputs=readouts(future_binary,dev_table,training,lives,output,"development",env)
    phase("evaluation_generation_started",sealed_lives_sha256=v1.digest(output/"sealed_lives.json"))
    for body in ("simple","hevlm"):
        for seed in eval_seeds: evaluation.extend(cohort(body,seed,False))
    eval_table=output/"evaluation.samples.bin";write_samples(eval_table,evaluation)
    outputs.extend(readouts(future_binary,eval_table,evaluation,lives,output,"evaluation",env))
    for label,p in lives.items(): assert v1.digest(p)==sealed[label]["sha256"],"evaluation changed a fitted life"
    phase("evaluation_complete",states=len(evaluation),lives_unchanged=True)
    final_trace_recheck=verify_cohort_traces(cohorts,output,steps)
    for name,digest in hashes.items(): assert v1.digest(ROOT/name)==digest,"source changed during experiment: "+name
    result={"manifest":manifest,"fit_receipts":fit_receipts,"sealed_lives":seal,"cohorts":cohorts,"gates":gate,
            "summary":summarize(outputs),"readouts":outputs,"comparisons":all_comparisons,
            "untouched_seed211_summary":summarize([r for r in outputs if r["split"]=="evaluation" and r["seed"]==211]),
            "final_trace_recheck":final_trace_recheck,
            "source_hashes_rechecked":True,"fitted_lives_unchanged_by_evaluation":True,
            "artifacts":{str(p.relative_to(output)):{"bytes":p.stat().st_size,"sha256":v1.digest(p)} for p in sorted(output.rglob("*"))
                         if p.is_file() and p.suffix in (".json",".jsonl",".log",".bin",".u32",".corpus")}}
    v1.write_json(output/"results.json",result)
    print("FUTURE_OK",json.dumps({"development_states":len(training),"evaluation_states":len(evaluation),"fit_steps":sum(r["fit_steps"] for r in fit_receipts.values())}),flush=True)
    return 0

if __name__=="__main__": sys.exit(main())
