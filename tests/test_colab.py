from __future__ import annotations

import zipfile

from olp_ai_26.core.colab import (
    ColabPaths,
    dataloader_kwargs,
    hf_precision_flags,
    stage_data,
    sync_artifacts,
)


def test_colab_paths_and_artifact_sync(tmp_path):
    paths = ColabPaths.create(
        runtime_dir=tmp_path / "runtime",
        persistent_dir=tmp_path / "drive",
    )
    checkpoint = paths.output_dir / "run" / "best.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"weights")
    ignored = paths.output_dir / "training.log"
    ignored.write_bytes(b"ignore")
    copied = sync_artifacts(paths.output_dir, paths.persistent_dir)
    assert copied == [paths.persistent_dir / "run" / "best.pt"]
    assert copied[0].read_bytes() == b"weights"


def test_stage_zip_archive_locally(tmp_path):
    archive = tmp_path / "dataset.zip"
    with zipfile.ZipFile(archive, "w") as stream:
        stream.writestr("train.csv", "id,label\n1,a\n")
    destination = stage_data(archive, tmp_path / "runtime" / "data")
    assert (destination / "train.csv").read_text(encoding="utf-8").startswith("id,label")


def test_cpu_precision_and_loader_flags():
    assert hf_precision_flags("cpu") == {"bf16": False, "fp16": False}
    assert dataloader_kwargs("cuda", 2) == {
        "num_workers": 2,
        "pin_memory": True,
        "persistent_workers": True,
    }
