"""Verify the ASL and NUS frozen manifests before running experiments."""

import argparse
import hashlib
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("split_root", type=Path)
    parser.add_argument("--checksums", type=Path,
                        default=Path("configs/asl_alphabet/split_checksums.json"))
    args = parser.parse_args()
    checksums = json.loads(args.checksums.read_text(encoding="utf8"))
    for dataset, files in checksums.items():
        for filename, digest in files.items():
            path = args.split_root / dataset / filename
            if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
                raise SystemExit(f"SHA-256 mismatch: {path}")
            print(f"OK {dataset}/{filename}")


if __name__ == "__main__":
    main()
