#!/usr/bin/env python3
"""Reconstruct the original measured archive from its lossless transport parts."""
import argparse
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent


def reconstruct(output):
    manifest = json.loads((HERE / "raw_manifest.json").read_text())
    if manifest["schema"] != 1:
        raise ValueError("unknown trace manifest schema")
    paths = []
    whole = hashlib.sha256()
    total = 0
    # Authenticate the complete input before creating the destination.
    for part in manifest["parts"]:
        name = part["path"]
        if Path(name).name != name:
            raise ValueError("trace part must be a sibling filename")
        path = HERE / name
        raw = path.read_bytes()
        if len(raw) != part["bytes"] or hashlib.sha256(raw).hexdigest() != part["sha256"]:
            raise ValueError("trace part identity mismatch: " + name)
        whole.update(raw)
        total += len(raw)
        paths.append((path, part))
    expected = manifest["archive"]
    if total != expected["bytes"] or whole.hexdigest() != expected["sha256"]:
        raise ValueError("complete trace archive identity mismatch")
    # Exclusive creation preserves any existing destination, including symlinks.
    with Path(output).open("xb") as target:
        copied = hashlib.sha256()
        for path, part in paths:
            raw = path.read_bytes()
            if len(raw) != part["bytes"] or hashlib.sha256(raw).hexdigest() != part["sha256"]:
                raise ValueError("trace part changed while copying: " + path.name)
            target.write(raw)
            copied.update(raw)
        if copied.hexdigest() != expected["sha256"]:
            raise ValueError("reconstructed archive identity mismatch")
    return expected


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(reconstruct(args.output), sort_keys=True))
