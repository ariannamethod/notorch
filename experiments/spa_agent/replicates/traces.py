#!/usr/bin/env python3
"""Reassemble the exact paired-continuation archive from lossless parts."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile

HERE = Path(__file__).resolve().parent


def require(condition, message):
    if not condition:
        raise ValueError(message)


def rebuild(manifest_path, destination):
    manifest_path, destination = Path(manifest_path), Path(destination)
    manifest = json.loads(manifest_path.read_bytes())
    require(manifest["schema"] == 1, "unsupported archive manifest")
    parts, archive = manifest["parts"], manifest["archive"]
    require(parts and len({p["path"] for p in parts}) == len(parts), "missing or duplicate parts")
    require(sum(p["bytes"] for p in parts) == archive["bytes"], "archive length disagrees with parts")
    require(not destination.exists() and not destination.is_symlink(), "destination already exists")
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(prefix=".spa-traces-", dir=destination.parent,
                                         delete=False) as output:
            temporary = Path(output.name)
            total, complete = 0, hashlib.sha256()
            for part in parts:
                name = part["path"]
                require(Path(name).name == name and name not in (".", ".."), "part must be a basename")
                measured, sha = 0, hashlib.sha256()
                git = hashlib.sha1(f"blob {part['bytes']}\0".encode())
                with (manifest_path.parent / name).open("rb") as stream:
                    while chunk := stream.read(1024 * 1024):
                        output.write(chunk)
                        measured += len(chunk)
                        sha.update(chunk)
                        git.update(chunk)
                        complete.update(chunk)
                require(measured == part["bytes"], "part length mismatch: " + name)
                require(sha.hexdigest() == part["sha256"], "part SHA256 mismatch: " + name)
                require(git.hexdigest() == part["git_blob_sha"], "part Git identity mismatch: " + name)
                total += measured
            require(total == archive["bytes"] and complete.hexdigest() == archive["sha256"],
                    "complete archive identity mismatch")
            output.flush()
            os.fsync(output.fileno())
        # Linking installs the checked file atomically and refuses a destination
        # created concurrently. The temporary file lives on the same filesystem.
        os.link(temporary, destination)
        return {"bytes": total, "sha256": complete.hexdigest(), "parts": len(parts)}
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=HERE / "raw_traces.jsonl.gz")
    args = parser.parse_args()
    result = rebuild(HERE / "raw_manifest.json", args.output)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
