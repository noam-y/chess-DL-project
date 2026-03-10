#!/usr/bin/env python3
"""Download the labeled dataset into assets/labeled from Google Drive."""

from __future__ import annotations

import argparse
import importlib.util
import shutil
import subprocess
import sys
from pathlib import Path

DATASET_ZIP_URL = (
    "https://drive.google.com/open?id=153dCfplp8GHfXtujWlLaA5EdhhOy_P3o&usp=drive_copy"
)
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
        "--source-url",
        type=str,
        default=DATASET_ZIP_URL,
        help=(
            "Google Drive source URL. Supports both folder links and single file links "
            "(for example, a .zip file)."
        ),
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Download even if the target directory already contains files.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help=(
            "Fail if gdown cannot retrieve all files in large folders. "
            "By default, partial folder retrieval is allowed."
        ),
    )
    return parser.parse_args()


def ensure_gdown_installed() -> None:
    if importlib.util.find_spec("gdown") is None:
        raise SystemExit(
            "Missing dependency: gdown\n"
            "Install it with: pip install gdown"
        )


def is_drive_folder_url(url: str) -> bool:
    return "/folders/" in url


def clear_directory(path: Path) -> None:
    for item in path.iterdir():
        if item.is_dir():
            shutil.rmtree(item)
        else:
            item.unlink()


def normalize_extracted_dataset_layout(target_dir: Path) -> None:
    if any(target_dir.glob("game*_per_frame")):
        return

    candidate_dirs = [p for p in target_dir.iterdir() if p.is_dir()]
    if len(candidate_dirs) != 1:
        return

    extracted_root = candidate_dirs[0]
    if not any(extracted_root.glob("game*_per_frame")):
        return

    for child in extracted_root.iterdir():
        shutil.move(str(child), str(target_dir / child.name))
    extracted_root.rmdir()


def download_folder(source_url: str, target_dir: Path, strict: bool) -> None:
    cmd = [
        sys.executable,
        "-m",
        "gdown",
        "--folder",
        source_url,
        "-O",
        str(target_dir),
    ]
    if not strict:
        # Some shared Drive folders exceed gdown's listing limit.
        # --remaining-ok keeps downloaded files instead of failing the script.
        cmd.append("--remaining-ok")
    subprocess.run(cmd, check=True)


def download_zip_and_extract(source_url: str, target_dir: Path) -> None:
    zip_path = target_dir.parent / "_dataset_download.zip"
    cmd = [
        sys.executable,
        "-m",
        "gdown",
        "--fuzzy",
        source_url,
        "-O",
        str(zip_path),
    ]
    subprocess.run(cmd, check=True)
    try:
        shutil.unpack_archive(str(zip_path), str(target_dir))
        normalize_extracted_dataset_layout(target_dir)
    finally:
        if zip_path.exists():
            zip_path.unlink()


def main() -> None:
    args = parse_args()
    target_dir = args.output_dir
    target_dir.mkdir(parents=True, exist_ok=True)

    if any(target_dir.iterdir()):
        if not args.force:
            print(
                f"Target directory '{target_dir}' is not empty. "
                "Use --force to re-download."
            )
            return
        print(f"Cleaning existing contents in: {target_dir}")
        clear_directory(target_dir)

    ensure_gdown_installed()
    print(f"Downloading dataset to: {target_dir}")
    if is_drive_folder_url(args.source_url):
        download_folder(args.source_url, target_dir, strict=args.strict)
    else:
        download_zip_and_extract(args.source_url, target_dir)
    print("Dataset download completed.")


if __name__ == "__main__":
    main()
