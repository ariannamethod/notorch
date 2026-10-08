#!/usr/bin/env python3
"""Reassemble this experiment's exact archive with the verified parent helper."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
PARENT = HERE.parent / "replicates/traces.py"
PARENT_SHA256 = "b9881a0857790ad0cbcde7d5d12bfb40264b67a21cc4b5bc95ee66228e75d611"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=HERE / "raw_records.tar.gz")
    args = parser.parse_args()
    if hashlib.sha256(PARENT.read_bytes()).hexdigest() != PARENT_SHA256:
        raise ValueError("parent reassembly helper changed")
    spec = importlib.util.spec_from_file_location("spa_reassemble_parent", PARENT)
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    result = helper.rebuild(HERE / "raw_manifest.json", args.output)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
