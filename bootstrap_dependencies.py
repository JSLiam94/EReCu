"""Check and install EReCu runtime dependencies in the active Python environment.

This is intentionally dependency-free so it can run in a newly-created server
environment.  Torch is checked but never auto-installed: choosing a PyTorch
wheel without knowing the server CUDA driver can silently replace a working GPU
build.  All other missing runtime packages are installed through pip.
"""
from __future__ import annotations

import argparse
import importlib
import subprocess
import sys
from dataclasses import dataclass


PYPI = "https://pypi.org/simple"


@dataclass(frozen=True)
class Dependency:
    module: str
    package: str
    required_for_metrics: bool = False


DEPENDENCIES = (
    Dependency("numpy", "numpy"),
    Dependency("PIL", "Pillow"),
    Dependency("cv2", "opencv-python-headless"),
    Dependency("yaml", "PyYAML"),
    Dependency("tqdm", "tqdm"),
    Dependency("scipy", "scipy", required_for_metrics=True),
    Dependency("py_sod_metrics", "pysodmetrics", required_for_metrics=True),
)


def available(module: str) -> bool:
    try:
        importlib.import_module(module)
    except Exception:
        return False
    return True


def pip_install(requirement: str, index_url: str) -> None:
    command = [
        sys.executable,
        "-m",
        "pip",
        "install",
        "--disable-pip-version-check",
        "--no-input",
        "--index-url",
        index_url,
        requirement,
    ]
    print("$ " + " ".join(command), flush=True)
    subprocess.run(command, check=True)


def main() -> None:
    parser = argparse.ArgumentParser("Check/install EReCu dependencies for the active Python environment")
    parser.add_argument("--check", action="store_true", help="only report missing packages; do not install")
    parser.add_argument("--metrics-only", action="store_true", help="only check/install evaluation dependencies")
    parser.add_argument("--index-url", default=PYPI, help=f"PyPI-compatible package index (default: {PYPI})")
    args = parser.parse_args()

    if not args.metrics_only and (not available("torch") or not available("torchvision")):
        raise RuntimeError(
            "Missing torch/torchvision. They are intentionally not auto-installed because CUDA wheel selection is server-specific. "
            "Install a CUDA-compatible pair in this environment, then rerun this script."
        )

    wanted = [item for item in DEPENDENCIES if not args.metrics_only or item.required_for_metrics]
    missing = [item for item in wanted if not available(item.module)]
    if not missing:
        print("PASS: all requested EReCu dependencies are available")
        return
    print("Missing: " + ", ".join(item.module for item in missing), flush=True)
    if args.check:
        raise SystemExit(2)

    for item in missing:
        if item.module == "py_sod_metrics":
            pip_install("pysodmetrics", args.index_url)
        else:
            pip_install(item.package, args.index_url)
        if not available(item.module):
            raise RuntimeError(f"Installation finished but import still failed: {item.module}")
    print("PASS: missing EReCu dependencies were installed and verified")


if __name__ == "__main__":
    main()
