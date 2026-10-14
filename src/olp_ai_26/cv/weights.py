"""Require staged weight files instead of implicit CV model downloads."""

from pathlib import Path
from typing import Any


def local_checkpoint(pretrained_allowed: bool, checkpoint_path: Path | str | None) -> str:
    """Resolve a staged checkpoint; pretrained opt-in requires an explicit permitted local file.

    A checkpoint supplied with pretrained disabled can be the team's own trained state. The
    caller must record provenance; a local path alone does not establish competition permission.
    """
    if pretrained_allowed and checkpoint_path is None:
        raise ValueError("Stage the exact permitted weights locally and set checkpoint_path")
    if checkpoint_path is None:
        return ""
    path = Path(checkpoint_path)
    if not path.is_file():
        raise FileNotFoundError(path)
    return str(path)


def timm_weight_options(
    pretrained_allowed: bool, checkpoint_path: Path | str | None
) -> dict[str, Any]:
    """Load allowed backbone weights from a local file with timm's classifier-head adaptation.

    Empty remote/source fields force the local-file loading path. With pretrained disabled, a
    local file is treated as an exact matching checkpoint produced by the team.
    """
    checkpoint = local_checkpoint(pretrained_allowed, checkpoint_path)
    if pretrained_allowed:
        return {
            "pretrained": True,
            "pretrained_cfg_overlay": {
                "file": checkpoint,
                "source": "",
                "url": None,
                "hf_hub_id": None,
            },
        }
    return {"pretrained": False, "checkpoint_path": checkpoint}
