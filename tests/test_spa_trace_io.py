#!/usr/bin/env python3
"""Verify complete v1 body traces in the runner's parent and on later reads.

    python3 tests/test_spa_trace_io.py --self-test
    python3 tests/test_spa_trace_io.py --check TRACE.jsonl --seed 42
    python3 tests/test_spa_trace_io.py --run SANITIZED_BINARY --inputs DIR \
        --output NEW_DIR --source IMMUTABLE_SOURCE --seed 42
    python3 tests/test_spa_trace_io.py --recheck NEW_DIR/trace_io.json
    python3 tests/test_spa_trace_io.py --emit-stdio-probe /tmp/spa_stdio_probe.c

The production writer, body, action law and reward remain unchanged. A zero
process exit is recorded separately from the complete-artifact gate.
"""
from __future__ import annotations

import argparse
import collections
import gzip
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


ARMS = ("disabled", "legacy", "random", "frozen", "learned")
EPISODES, DECISIONS = 6, 4
SOURCE_FILES = ("examples/spa_agent_demo.c", "spa_agent.c", "spa_agent.h",
                "notorch.c", "notorch.h", "notorch_simd.h",
                "chuck_architect.h", "chuck_architect_impl.h",
                "experiments/spa_agent/protocol.json")


# Linux /proc plus GNU ld --wrap diagnostic only; the production writer is unchanged.
STDIO_PROBE = r'''#define _POSIX_C_SOURCE 200809L
#include <stdio.h>
#include <stdint.h>
#include <inttypes.h>
#include <unistd.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <string.h>
int __real_fflush(FILE *stream);
int __real_fclose(FILE *stream);
static FILE *trace_file;
static char trace_name[4096];
static void identify(FILE *f) {
 if(trace_file||!f)return;
 char proc[80],path[4096];snprintf(proc,sizeof proc,"/proc/self/fd/%d",fileno(f));
 ssize_t n=readlink(proc,path,sizeof path-1);if(n<0)return;path[n]=0;
 char *end=strstr(path,".jsonl");if(!end)return;end[6]=0;
 trace_file=f;snprintf(trace_name,sizeof trace_name,"%s",path);
}
static void inspect(FILE *f,const char *kind,int result) {
 struct stat st,ps;off_t pos=ftello(f);int stat_rc=fstat(fileno(f),&st),path_rc=stat(trace_name,&ps);uint64_t hash=UINT64_C(14695981039346656037);size_t size=0;
 char proc[80],link[4096];snprintf(proc,sizeof proc,"/proc/self/fd/%d",fileno(f));ssize_t ln=readlink(proc,link,sizeof link-1);if(ln<0)strcpy(link,"READLINK_FAILED");else link[ln]=0;
 int fd=open(proc,O_RDONLY);unsigned char bytes[16384];ssize_t n=0;
 if(fd>=0){while((n=read(fd,bytes,sizeof bytes))>0){for(ssize_t i=0;i<n;++i)hash=(hash^bytes[i])*UINT64_C(1099511628211);size+=(size_t)n;}close(fd);}
 fprintf(stderr,"TRACE_IO kind=%s result=%d ferror=%d pos=%jd fd=%d fd_size=%jd fd_ino=%ju path_size=%jd path_ino=%ju reread_fd=%zu fnv=%016" PRIx64 " read_end=%zd link=%s\n",kind,result,ferror(f),(intmax_t)pos,fileno(f),stat_rc?-1:(intmax_t)st.st_size,stat_rc?0:(uintmax_t)st.st_ino,path_rc?-1:(intmax_t)ps.st_size,path_rc?0:(uintmax_t)ps.st_ino,size,hash,n,link);
}
int __wrap_fflush(FILE *f){int rc=__real_fflush(f);identify(f);if(f==trace_file)inspect(f,"fflush",rc);return rc;}
int __wrap_fclose(FILE *f){identify(f);int trace=f==trace_file;if(trace){int rc=__real_fflush(f);inspect(f,"pre-fclose",rc);}int rc=__real_fclose(f);if(trace){struct stat st;int sr=stat(trace_name,&st);fprintf(stderr,"TRACE_IO kind=post-fclose result=%d path_size=%jd path_ino=%ju\n",rc,sr?-1:(intmax_t)st.st_size,sr?0:(uintmax_t)st.st_ino);}return rc;}
'''


class TraceError(ValueError):
    pass


def sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def validate(raw: bytes, seed: int) -> dict:
    """Require one complete, unique record for every registered v1 decision."""
    rows = []
    for number, line in enumerate(raw.splitlines(), 1):
        try:
            row = json.loads(line)
        except (ValueError, UnicodeError) as error:
            raise TraceError(f"JSONL line {number}: {error}") from error
        if not isinstance(row, dict) or row.get("seed") != seed:
            raise TraceError(f"JSONL line {number}: wrong record type or seed")
        rows.append(row)
    if not raw.endswith(b"\n"):
        raise TraceError("trace lacks its final newline")
    counts = dict(collections.Counter(row.get("type") for row in rows))
    wanted = {"body": 1, "base": EPISODES,
              "decision": EPISODES * DECISIONS * len(ARMS), "summary": 1}
    if counts != wanted or rows[0]["type"] != "body" or rows[-1]["type"] != "summary":
        raise TraceError(f"incomplete event stream: observed {counts}, expected {wanted}")
    bases = [row.get("episode") for row in rows if row["type"] == "base"]
    if sorted(bases) != list(range(EPISODES)):
        raise TraceError("base episodes are missing or duplicated")
    decisions = [(row.get("episode"), row.get("arm"), row.get("step"))
                 for row in rows if row["type"] == "decision"]
    expected = {(episode, arm, step) for episode in range(EPISODES)
                for arm in ARMS for step in range(DECISIONS)}
    if len(set(decisions)) != len(decisions) or set(decisions) != expected:
        raise TraceError("decision coordinates are missing, duplicated or unregistered")
    summary = rows[-1]
    if summary.get("body_unchanged") is not True:
        raise TraceError("summary does not record unchanged body parameters")
    arms = summary.get("arms")
    if not isinstance(arms, list) or [arm.get("name") for arm in arms] != list(ARMS):
        raise TraceError("summary arm identities differ from the registered protocol")
    for arm in arms:
        count = 0 if arm["name"] == "disabled" else EPISODES * DECISIONS
        updates = EPISODES * DECISIONS if arm["name"] == "learned" else 0
        if (arm.get("verified_resumes") != EPISODES or arm.get("decisions") != count
                or arm.get("observations") != count or arm.get("updates") != updates):
            raise TraceError(f"incomplete life counters in {arm['name']}")
    return {"bytes": len(raw), "sha256": sha256(raw), "records": len(rows),
            "record_types": counts, "complete": True, "seed": seed}


def inspect(path: Path, seed: int) -> tuple[dict, bytes]:
    with path.open("rb") as stream:
        stat = os.fstat(stream.fileno())
        raw = stream.read()
    result = validate(raw, seed)
    result["stat"] = {"bytes": stat.st_size, "device": stat.st_dev,
                      "inode": stat.st_ino, "mtime_ns": stat.st_mtime_ns}
    if stat.st_size != len(raw):
        raise TraceError("trace size changed during parent read")
    return result, raw


def durable_write(path: Path, raw: bytes) -> None:
    with path.open("wb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())


def run(args: argparse.Namespace) -> dict:
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    inputs, binary = args.inputs.resolve(), args.run.resolve()
    prefix = output / f"body-s{args.seed}"
    trace = prefix.with_suffix(".jsonl")
    command = [str(binary), *(str(inputs / name) for name in
               ("simple.weights", "corpus.tokens.u32", "vocabulary.u32")),
               str(prefix), str(args.seed)]
    env = dict(os.environ, ASAN_OPTIONS="detect_leaks=0",
               UBSAN_OPTIONS="halt_on_error=1:print_stacktrace=1")
    source_files = list(SOURCE_FILES)
    if args.source and (args.source / "examples/spa_agent_scenarios.h").exists():
        source_files.append("examples/spa_agent_scenarios.h")
    source_hashes = ({name: sha256((args.source / name).read_bytes())
                      for name in source_files} if args.source else {})
    start = time.monotonic()
    process = subprocess.run(command, cwd=args.source, env=env, capture_output=True,
                             timeout=args.timeout, check=False)
    durable_write(output / "stdout.log", process.stdout)
    durable_write(output / "stderr.log", process.stderr)
    result = {"schema": 1, "command": command, "binary_sha256": sha256(binary.read_bytes()),
              "environment": {key: env[key] for key in ("ASAN_OPTIONS", "UBSAN_OPTIONS")},
              "source_files_sha256": source_hashes, "returncode": process.returncode,
              "wall_seconds": time.monotonic() - start, "seed": args.seed,
              "trace": trace.name, "mirrors": ["verified.jsonl", "verified.jsonl.gz"]}
    try:
        result["same_parent"], raw = inspect(trace, args.seed)
        if trace.read_bytes() != raw:
            raise TraceError("trace changed on immediate second parent read")
        durable_write(output / "verified.jsonl", raw)
        durable_write(output / "verified.jsonl.gz", gzip.compress(raw, mtime=0))
        result["immediate_second_read_equal"] = True
        if source_hashes and any(sha256((args.source / name).read_bytes()) != value
                                 for name, value in source_hashes.items()):
            raise TraceError("source snapshot changed during the diagnostic")
        if process.returncode:
            raise TraceError(f"body process exited {process.returncode}")
        result["status"] = "PASS"
    except (OSError, TraceError) as error:
        result["status"], result["error"] = "FAIL", str(error)
    durable_write(output / "trace_io.json", (json.dumps(result, indent=2) + "\n").encode())
    return result


def recheck(receipt_path: Path) -> dict:
    receipt = json.loads(receipt_path.read_text())
    if receipt.get("status") != "PASS":
        raise TraceError("earlier process/artifact gate did not pass")
    checks = []
    for name in [receipt["trace"], *receipt["mirrors"]]:
        raw = (receipt_path.parent / name).read_bytes()
        if name.endswith(".gz"):
            raw = gzip.decompress(raw)
        current = validate(raw, receipt["seed"])
        for field in ("bytes", "sha256", "records", "record_types"):
            if current[field] != receipt["same_parent"][field]:
                raise TraceError(f"later read differs from immediate parent: {name}, {field}")
        checks.append({"artifact": name, **current})
    return {"status": "PASS", "checks": checks}


def fixture() -> bytes:
    rows = [{"type": "body", "seed": 42}]
    for episode in range(EPISODES):
        rows.append({"type": "base", "seed": 42, "episode": episode})
        for arm in ARMS:
            for step in range(DECISIONS):
                rows.append({"type": "decision", "seed": 42, "episode": episode,
                             "arm": arm, "step": step})
    rows.append({"type": "summary", "seed": 42, "body_unchanged": True,
                 "arms": [{"name": arm, "verified_resumes": EPISODES,
                           "decisions": 0 if arm == "disabled" else EPISODES * DECISIONS,
                           "observations": 0 if arm == "disabled" else EPISODES * DECISIONS,
                           "updates": EPISODES * DECISIONS if arm == "learned" else 0}
                          for arm in ARMS]})
    return b"".join(json.dumps(row).encode() + b"\n" for row in rows)


def self_test() -> dict:
    raw = fixture()
    validate(raw, 42)
    lines = raw.splitlines(keepends=True)
    mutants = {
        "empty": b"",
        "partial JSON write": raw[:len(raw) // 2 + 3],
        "missing final summary": b"".join(lines[:-1]),
        "success summary with missing decision": b"".join(lines[:3] + lines[4:]),
        "duplicate decision with unchanged total count": b"".join(lines[:3] + [lines[2]] + lines[4:]),
        "missing final newline": raw[:-1],
        "summary falsely declares incomplete acquired life": raw.replace(b'"updates": 24', b'"updates": 23'),
    }
    caught = {}
    for name, mutant in mutants.items():
        try:
            validate(mutant, 42)
        except TraceError as error:
            caught[name] = str(error)
        else:
            raise AssertionError(f"trace gate failed to catch {name}")
    return {"status": "PASS", "complete_fixture_records": 128,
            "caught_mutations": caught}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--self-test", action="store_true")
    mode.add_argument("--check", type=Path)
    mode.add_argument("--run", type=Path)
    mode.add_argument("--recheck", type=Path)
    mode.add_argument("--emit-stdio-probe", type=Path)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--inputs", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--json", type=Path, help="also write the gate result to this path")
    parser.add_argument("--timeout", type=float, default=300)
    args = parser.parse_args()
    if args.run and (not args.inputs or not args.output):
        parser.error("--run requires --inputs and a new --output directory")
    try:
        if args.self_test:
            result = self_test()
        elif args.check:
            result, _ = inspect(args.check, args.seed)
            result["status"] = "PASS"
        elif args.recheck:
            result = recheck(args.recheck)
        elif args.emit_stdio_probe:
            raw = STDIO_PROBE.encode()
            with args.emit_stdio_probe.open("xb") as stream:
                stream.write(raw)
                stream.flush()
                os.fsync(stream.fileno())
            result = {"status": "PASS", "path": str(args.emit_stdio_probe),
                      "sha256": sha256(raw), "bytes": len(raw)}
        else:
            result = run(args)
    except (OSError, ValueError, AssertionError, subprocess.TimeoutExpired) as error:
        result = {"status": "FAIL", "error": str(error)}
        if args.json:
            args.json.parent.mkdir(parents=True, exist_ok=True)
            durable_write(args.json, (json.dumps(result, indent=2) + "\n").encode())
        print(json.dumps(result, indent=2))
        return 1
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        durable_write(args.json, (json.dumps(result, indent=2) + "\n").encode())
    print(json.dumps(result, indent=2))
    return 0 if result.get("status") == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
