#!/usr/bin/env python3
"""Fast synthetic IO and immutable-source checks for the conditioned runner.

No native policy fit, model load, or sentence generation occurs here.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tarfile
import tempfile
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / "experiments/spa_agent/conditioned/run.py"
SPEC = importlib.util.spec_from_file_location("spa_conditioned_runner_preflight", PATH)
RUN = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RUN)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()
    original = PATH.read_bytes()
    declared_names = RUN.source_files()
    self_receipt = "experiments/spa_agent/conditioned/runner_preflight.json"
    # The runner pins this completed receipt when it later snapshots sources.
    # A receipt cannot contain its own final hash; every other source is pinned.
    names = [name for name in declared_names if name != self_receipt]
    identities = {name: RUN.identity(ROOT / name) for name in names}
    tests, failures, commands = [], [], []

    def check(name, assertion):
        if not assertion:
            raise AssertionError(name)
        tests.append(name)

    def reject(name, operation, fragment=None):
        try:
            operation()
        except (AssertionError, ValueError, OSError, KeyError) as error:
            if fragment is not None and fragment not in str(error):
                raise AssertionError(name + ": wrong failure: " + str(error)) from error
            failures.append({"name": name, "exception": type(error).__name__, "message": str(error)})
        else:
            raise AssertionError(name + ": deliberate defect escaped")

    with tempfile.TemporaryDirectory(prefix="spa-conditioned-runner-") as directory:
        work = Path(directory)

        def portable(value):
            if isinstance(value, str):
                return value.replace(str(work), "<temporary>").replace(str(ROOT), "<source>").replace(sys.executable, "python3")
            if isinstance(value, list):
                return [portable(x) for x in value]
            if isinstance(value, dict):
                return {k: portable(v) for k, v in value.items()}
            return value

        published = work / "record.json"
        RUN.write_json(published, {"z": "фонон", "a": [1, 2]})
        expected = b'{"a":[1,2],"z":"\\u0444\\u043e\\u043d\\u043e\\u043d"}\n'
        check("canonical_json_bytes", published.read_bytes() == expected)
        reject("existing_record_refused", lambda: RUN.write_bytes(published, b"replacement"))
        check("existing_record_preserved", published.read_bytes() == expected)
        pending = published.with_name(published.name + ".pending")
        check("failed_publish_retains_forensic_pending", pending.read_bytes() == b"replacement")
        stage = work / "stages.json"
        RUN.write_json(stage, {"stage": 1})
        RUN.write_json(stage, {"stage": 2}, replace=True)
        check("explicit_stage_replacement", stage.read_bytes() == b'{"stage":2}\n')
        reject("nonfinite_json_refused", lambda: RUN.write_json(work / "nonfinite.json", {"x": float("nan")}))
        check("nonfinite_json_unpublished", not (work / "nonfinite.json").exists())

        original_read = Path.read_bytes
        damaged = work / "write-mismatch"

        def bad_read(path):
            raw = original_read(path)
            return raw + b"corruption" if path == damaged.with_name(damaged.name + ".pending") else raw

        with mock.patch.object(Path, "read_bytes", bad_read):
            reject("write_readback_corruption_refused", lambda: RUN.write_bytes(damaged, b"intact"), "write readback differs")
        check("readback_failure_unpublished", not damaged.exists())

        fixture = {"metadata.json": b'{"body":"synthetic"}\n',
                   "fit-1/fit.jsonl.gz": gzip.compress(b'{"type":"synthetic-fit-receipt"}\n', mtime=0),
                   "fit-2/seal.json": b'{"sealed":true}\n',
                   "evaluation-repeats-s509.jsonl": b'{"seed":509,"fixture":true}\n',
                   "training.json": b'{"samples":[]}\n',
                   "evaluation.readout-1.jsonl": b'{"kind":"KEEP"}\n',
                   "evaluation.summary.json": b'{"fixture":true}\n',
                   "commands.json": b'[]\n', "empty.stderr.txt": b""}
        archives = []
        for index in (1, 2):
            output = work / ("archive-" + str(index))
            output.mkdir()
            for name, raw in fixture.items():
                path = output / name
                path.parent.mkdir(exist_ok=True)
                RUN.write_bytes(path, raw)
            closed = {name: RUN.identity(output / name) for name in fixture}
            order = list(fixture) if index == 1 else list(reversed(fixture))
            result = RUN.pack_records(output, order, closed)
            archives.append((output / result["file"]).read_bytes())
            with tarfile.open(output / result["file"], "r:gz") as archive:
                members = archive.getmembers()
                check(f"archive_{index}_exact_sorted_members", [m.name for m in members] == sorted(fixture))
                check(f"archive_{index}_exact_bytes", all(archive.extractfile(m).read() == fixture[m.name] for m in members))
                check(f"archive_{index}_portable_headers", all(m.isfile() and m.uid == 0 and m.gid == 0 and
                    m.uname == "" and m.gname == "" and m.mtime == 0 and m.mode == 0o644 for m in members))
            check(f"archive_{index}_manifest_has_every_record", result["members"] == closed)
            if index == 1:
                reject("existing_archive_refused", lambda: RUN.pack_records(output, order, closed))
                check("existing_archive_preserved", (output / result["file"]).read_bytes() == archives[0])
        check("archive_bytes_deterministic", archives[0] == archives[1])

        truncated = work / "truncated"
        truncated.mkdir()
        record = truncated / "evaluation.json"
        RUN.write_bytes(record, b'{"samples":[1,2,3,4]}\n')
        retained = {record.name: RUN.identity(record)}
        record.write_bytes(record.read_bytes()[:8])
        reject("closed_identity_rejects_later_truncation", lambda: RUN.pack_records(truncated, [record.name], retained),
               "record changed before archive")
        check("truncated_input_no_published_archive", not (truncated / "raw_records.tar.gz").exists())

        mismatch = work / "archive-readback"
        mismatch.mkdir()
        RUN.write_bytes(mismatch / "record.json", b'{"value":1}\n')
        closed = {"record.json": RUN.identity(mismatch / "record.json")}
        true_open = tarfile.open

        def corrupt_archive(path=None, mode="r", *arguments, **keywords):
            if path == mismatch / "raw_records.tar.gz.pending" and mode == "r:gz":
                raw = gzip.decompress(path.read_bytes())
                assert raw.count(b'{"value":1}\n') == 1
                path.write_bytes(gzip.compress(raw.replace(b'{"value":1}\n', b'{"value":2}\n'), mtime=0))
            return true_open(path, mode, *arguments, **keywords)

        with mock.patch.object(RUN.tarfile, "open", corrupt_archive):
            reject("archive_readback_corruption_refused", lambda: RUN.pack_records(mismatch, ["record.json"], closed),
                   "archive readback mismatch")
        check("archive_readback_failure_unpublished", not (mismatch / "raw_records.tar.gz").exists())
        missing = work / "unclosed"
        missing.mkdir()
        RUN.write_bytes(missing / "unclosed.txt", b"not registered")
        reject("unregistered_archive_record_refused", lambda: RUN.pack_records(missing, ["unclosed.txt"], {}))

        for label, paths in (("absolute", [str(work / "outside.txt")]),
                             ("parent_traversal", ["../outside.txt"]),
                             ("nested_parent_traversal", ["nested/../outside.txt"]),
                             ("noncanonical", ["./record.json"]),
                             ("backslash", ["nested\\record.json"]),
                             ("duplicate", ["record.json", "record.json"])):
            boundary = work / ("boundary-" + label)
            boundary.mkdir()
            reject("archive_" + label + "_refused", lambda b=boundary, p=paths: RUN.pack_records(b, p, {}),
                   "duplicate archive member" if label == "duplicate" else "portable relative path")
            check("archive_" + label + "_no_temporary", not list(boundary.iterdir()))

        inventory_root = work / "inputs" / "source_snapshot" / "run"
        entries = {"metadata.json": b'{}\n', "fit-1/seal.json": b'{}\n',
            "fit-1/initial.life": b"synthetic life", "off-s509.initial.life.bin": b"synthetic host life",
            "libnotorch.so": b"synthetic binary", "inputs/simple.weights": b"synthetic weights",
            "source_snapshot/runner.py": b"synthetic frozen source"}
        for name, raw in entries.items():
            path = inventory_root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            RUN.write_bytes(path, raw)
        closed = {name: RUN.identity(inventory_root / name) for name in entries
                  if name not in ("libnotorch.so", "inputs/simple.weights", "source_snapshot/runner.py")}
        artifacts, records = RUN.record_inventory(inventory_root, {"libnotorch.so": {}}, closed)
        check("reserved_ancestor_names_do_not_omit_artifacts", artifacts == closed)
        check("inventory_retains_life_identities_excludes_binary_archive_members",
              records == ["fit-1/seal.json", "metadata.json"])
        (inventory_root / "metadata.json").write_bytes(b"{")
        reject("inventory_rejects_closed_record_truncation", lambda: RUN.record_inventory(inventory_root, {"libnotorch.so": {}}, closed),
               "closed artifact differs at final inventory")
        unregistered = work / "unregistered-output"
        unregistered.mkdir()
        RUN.write_bytes(unregistered / "extra.txt", b"unknown result")
        reject("inventory_refuses_omitted_output", lambda: RUN.record_inventory(unregistered, {}, {}),
               "unregistered closed artifact")

        command_output = work / "command-output"
        command_output.mkdir()
        source = work / "source-cwd"
        source.mkdir()
        code = "import sys; print(sys.argv[1]); print(sys.argv[2], file=sys.stderr)"
        command = RUN.run_command([sys.executable, "-I", "-c", code, source, command_output], source, command_output, "portable")
        check("command_paths_portable", str(work) not in json.dumps(command))
        check("command_streams_closed_exact", all(RUN.identity(command_output / command[k]["file"]) ==
              {field: command[k][field] for field in ("bytes", "sha256")} for k in ("stdout", "stderr")))
        check("command_original_stream_hashes_retained", command["stdout"]["original_sha256"] == sha((str(source) + "\n").encode()) and
              command["stderr"]["original_sha256"] == sha((str(command_output) + "\n").encode()))
        reject("failed_command_receipt_recorded", lambda: RUN.run_command([sys.executable, "-I", "-c", "raise SystemExit(7)"],
            source, command_output, "failed"), "failed exited 7")
        check("failed_command_has_all_three_records", all((command_output / ("failed." + suffix)).is_file()
              for suffix in ("command.json", "stdout.txt", "stderr.txt")))

        snapshot = work / "snapshot"
        for name in names:
            target = snapshot / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes((ROOT / name).read_bytes())

        def invoke(argv):
            process = subprocess.run(list(map(str, argv)), cwd=snapshot, capture_output=True, text=True, timeout=30)
            commands.append({"argv": list(map(str, argv)), "returncode": process.returncode,
                             "stdout": process.stdout, "stderr": process.stderr})
            check("snapshot_command_" + str(len(commands)), process.returncode == 0)
            return process.stdout

        invoke([sys.executable, "-I", "-B", snapshot / "experiments/spa_agent/conditioned/run.py", "--help"])
        compiler = shlex.split(os.environ.get("CC", "cc"))
        needed = set()
        for source_name in ("examples/spa_agent_replicates.c", "examples/spa_agent_demo.c", "spa_binding.c", "spa_agent.c", "notorch.c"):
            output = invoke([*compiler, "-std=c11", "-DUSE_SIMD", "-march=native", "-pthread", "-I.", "-MM", source_name])
            for name in output.replace("\\\n", " ").split(":", 1)[1].split():
                needed.add(str((snapshot / name).resolve().relative_to(snapshot)))
        check("compiler_local_header_closure", needed <= set(names))
        check("snapshot_source_identities", all(RUN.identity(snapshot / name) == identities[name] for name in names))
        check("runner_declares_preflight_sources", self_receipt in declared_names and "tests/test_spa_conditioned_runner.py" in declared_names)
        check("working_sources_unchanged", PATH.read_bytes() == original and
              all(RUN.identity(ROOT / name) == identities[name] for name in names))
        report = portable({"schema": 1, "operation": "synthetic conditioned runner IO preflight",
            "passed": True, "test_count": len(tests), "checks": tests, "named_refusals": failures,
            "source_files": identities, "runner_sha256": sha(original), "test_sha256": sha(Path(__file__).read_bytes()),
            "archive": {"bytes": len(archives[0]), "sha256": sha(archives[0]), "members": sorted(fixture), "deterministic": True},
            "snapshot": {"files": len(names), "declared_files": len(declared_names),
                "compiler_dependencies": sorted(needed), "isolated_import": True,
                "self_receipt": {"path": self_receipt,
                    "identity": "Excluded only from its own hash table; the experiment runner pins the completed receipt."}},
            "commands": commands, "native_fits": 0, "model_generations": 0})
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(f"PASS conditioned runner: {len(tests)} checks, {len(failures)} named refusals, zero native fits/generation")


if __name__ == "__main__":
    main()
