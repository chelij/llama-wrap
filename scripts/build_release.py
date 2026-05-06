from __future__ import annotations

import argparse
import os
import platform
import shutil
import subprocess
import sys
import tarfile
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
APP_NAME = "llama-wrap"
ENTRYPOINT = ROOT / "llamawrap.py"
DIST_DIR = ROOT / "dist"
BUILD_DIR = ROOT / "build"


def run(command: list[str]) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def platform_tag() -> str:
    system = platform.system().lower()
    machine = platform.machine().lower().replace("amd64", "x86_64")

    if system == "darwin":
        return f"macos-{machine}"
    if system == "windows":
        return f"windows-{machine}"
    if system == "linux":
        return f"linux-{machine}"
    return f"{system}-{machine}"


def archive(output_dir: Path, tag: str) -> Path:
    if platform.system().lower() == "windows":
        archive_path = DIST_DIR / f"{APP_NAME}-{tag}.zip"
        with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            for path in output_dir.rglob("*"):
                zf.write(path, path.relative_to(output_dir.parent))
        return archive_path

    archive_path = DIST_DIR / f"{APP_NAME}-{tag}.tar.gz"
    with tarfile.open(archive_path, "w:gz") as tf:
        tf.add(output_dir, arcname=output_dir.name)
    return archive_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a llama-wrap release bundle for this OS.")
    parser.add_argument("--tag", default=platform_tag(), help="Artifact platform tag.")
    args = parser.parse_args()

    if not ENTRYPOINT.exists():
        print(f"Missing entrypoint: {ENTRYPOINT}", file=sys.stderr)
        return 1

    shutil.rmtree(BUILD_DIR, ignore_errors=True)
    output_dir = DIST_DIR / APP_NAME
    shutil.rmtree(output_dir, ignore_errors=True)
    DIST_DIR.mkdir(exist_ok=True)

    command = [
        sys.executable,
        "-m",
        "PyInstaller",
        "--noconfirm",
        "--clean",
        "--name",
        APP_NAME,
        "--windowed",
        "--onedir",
        str(ENTRYPOINT),
    ]

    if platform.system().lower() == "darwin":
        command.extend(["--osx-bundle-identifier", "com.chelib.llama-wrap"])

    run(command)

    readme = ROOT / "README.md"
    if readme.exists():
        shutil.copy2(readme, output_dir / "README.md")

    archive_path = archive(output_dir, args.tag)
    print(f"Built {archive_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
