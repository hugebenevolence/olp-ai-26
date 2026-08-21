"""Executed by the first cell of each Colab notebook template."""

import importlib.util
import subprocess
import sys
from pathlib import Path

IN_COLAB = "google.colab" in sys.modules
PROJECT_ROOT = Path(globals().get("PROJECT_ROOT", "/content/olp-ai-26" if IN_COLAB else Path.cwd()))
if not (PROJECT_ROOT / "pyproject.toml").exists():
    raise FileNotFoundError(
        f"Project not found at {PROJECT_ROOT}. Upload or clone it, then change PROJECT_ROOT."
    )

source_root = str(PROJECT_ROOT / "src")
if source_root not in sys.path:
    sys.path.insert(0, source_root)

if IN_COLAB:
    required_modules = ("timm", "sacrebleu", "rouge_score", "seqeval")
    if any(importlib.util.find_spec(module) is None for module in required_modules):
        subprocess.run(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "--quiet",
                "--disable-pip-version-check",
                "--requirement",
                str(PROJECT_ROOT / "requirements-colab.txt"),
            ],
            check=True,
        )
