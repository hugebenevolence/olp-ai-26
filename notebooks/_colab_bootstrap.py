"""Executed by the first cell of each Colab notebook template."""

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
    from olp_ai_26.core.colab import install_missing_dependencies

    installed = install_missing_dependencies(
        PROJECT_ROOT / "requirements-colab.txt",
        required_modules=globals().get("REQUIRED_MODULES"),
    )
    if installed:
        print("Installed missing notebook dependencies:", installed)
