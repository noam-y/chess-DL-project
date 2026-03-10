#!/usr/bin/env python3
"""Download the labeled dataset into assets/labeled from Google Drive."""

from __future__ import annotations

import argparse
import importlib.util
import subprocess
import sys
from pathlib import Path

DATASET_URL = "https://drive.google.com/drive/folders/1EJ7cAeCuZvlHRBsLF3m7UgcJXwnb0lYB?usp=sharing"
DEFAULT_OUTPUT_DIR = Path("assets/labeled")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download the labeled dataset into assets/labeled."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Target directory for downloaded data (default: assets/labeled).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Download even if the target directory already contains files.",
    )
    return parser.parse_args()


def ensure_gdown_installed() -> None:
    if importlib.util.find_spec("gdown") is None:
        raise SystemExit(
            "Missing dependency: gdown\n"
            "Install it with: pip install gdown"
        )


def download_folder(target_dir: Path) -> None:
    cmd = [
        sys.executable,
        "-m",
        "gdown",
        "--folder",
        DATASET_URL,
        "-O",
        str(target_dir),
    ]
    subprocess.run(cmd, check=True)


def main() -> None:
    args = parse_args()
    target_dir = args.output_dir
    target_dir.mkdir(parents=True, exist_ok=True)

    if not args.force and any(target_dir.iterdir()):
        print(
            f"Target directory '{target_dir}' is not empty. "
            "Use --force to re-download."
        )
        return

    ensure_gdown_installed()
    print(f"Downloading dataset to: {target_dir}")
    download_folder(target_dir)
    print("Dataset download completed.")


if __name__ == "__main__":
    main()
