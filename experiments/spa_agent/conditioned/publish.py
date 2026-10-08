#!/usr/bin/env python3
"""Publish closed raw evidence as exact, bounded, lossless Git archive parts."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile

HERE = Path(__file__).resolve().parent
PART_BYTES = 8011776


def identity(raw):
    return {"bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def write(path, raw):
    with path.open("xb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    require(path.read_bytes() == raw, "publication write differs: " + path.name)


def encode(value):
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execution-dir", type=Path, required=True)
    args = parser.parse_args()
    source = args.execution_dir.resolve()
    receipt_raw = (source / "receipts.json").read_bytes()
    receipts = json.loads(receipt_raw)
    audited = json.loads((HERE / "result_audit.json").read_bytes())
    require(audited["status"] == "PASS" and audited["identities"]["receipt"] == identity(receipt_raw),
            "independent audit does not authenticate this receipt")
    for name, expected in receipts["artifacts"].items():
        require(identity((source / name).read_bytes()) == expected, "closed artifact changed: " + name)
    archive = source / receipts["archive"]["file"]
    raw = archive.read_bytes()
    archive_id = identity(raw)
    require(all(archive_id[k] == receipts["archive"][k] for k in archive_id), "raw archive differs")
    seen = set()
    with tarfile.open(archive, "r:gz") as stream:
        for entry in stream:
            require(entry.isfile() and entry.name not in seen and entry.name in receipts["archive"]["members"],
                    "unexpected archive member")
            require(not Path(entry.name).is_absolute() and ".." not in Path(entry.name).parts and
                    Path(entry.name).suffix not in (".life", ".bin"), "invalid archive member")
            require(identity(stream.extractfile(entry).read()) == receipts["archive"]["members"][entry.name],
                    "archive member differs")
            seen.add(entry.name)
    require(seen == set(receipts["archive"]["members"]), "incomplete archive")
    parts = []
    for index, offset in enumerate(range(0, len(raw), PART_BYTES)):
        chunk = raw[offset:offset + PART_BYTES]
        name = f"raw_records.tar.gz.part-{index:02d}"
        write(HERE / name, chunk)
        parts.append({"path": name, **identity(chunk),
            "git_blob_sha": hashlib.sha1(f"blob {len(chunk)}\0".encode() + chunk).hexdigest()})
    write(HERE / "receipts.json", receipt_raw)
    write(HERE / "recovery.json", (source / "recovery.json").read_bytes())
    manifest = {"schema": 1, "archive": {"file": "raw_records.tar.gz", **archive_id},
        "parts": parts, "receipt": identity(receipt_raw), "member_count": len(seen)}
    write(HERE / "raw_manifest.json", encode(manifest))
    with tempfile.TemporaryDirectory(prefix="spa-conditioned-publish-") as directory:
        target = Path(directory) / "reassembled.tar.gz"
        process = subprocess.run([sys.executable, str(HERE / "traces.py"), "--output", str(target)],
                                 capture_output=True, text=True)
        require(process.returncode == 0, "real archive reassembly failed: " + process.stderr)
        require(identity(target.read_bytes()) == archive_id, "real archive reassembly differs")
    write(HERE / "publication.json", encode({"schema": 1, "status": "PASS",
        "command": "python3 experiments/spa_agent/conditioned/publish.py --execution-dir <completed-run>",
        "source": identity(Path(__file__).read_bytes()), "raw_receipt_unchanged": True,
        "archive": archive_id, "members": len(seen), "parts": len(parts),
        "reassembly": {"command": "python3 experiments/spa_agent/conditioned/traces.py --output <new-archive>",
                       "returncode": process.returncode, "stdout": process.stdout},
        "numerical_recomputation": "none; exact byte publication only"}))
    print(json.dumps({"passed": True, "bytes": len(raw), "members": len(seen), "parts": len(parts)}))


if __name__ == "__main__":
    main()
