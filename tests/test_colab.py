from __future__ import annotations

import zipfile
from pathlib import Path
from types import SimpleNamespace

import pytest

from olp_ai_26.core import colab
from olp_ai_26.core.colab import (
    ColabPaths,
    dataloader_kwargs,
    hf_precision_flags,
    install_missing_dependencies,
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


def test_missing_only_bootstrap_constrains_existing_stack(tmp_path, monkeypatch):
    requirements = tmp_path / "requirements.txt"
    requirements.write_text("timm>=1,<2\ntransformers>=4,<6\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(colab.importlib.util, "find_spec", lambda name: None)
    protected = {
        "torch": "2.9.0",
        "torchvision": "0.24.0",
        "numpy": "2.0.0",
        "nvidia-cuda-runtime-cu12": "12.6.0",
    }
    monkeypatch.setattr(
        colab.importlib.metadata,
        "distributions",
        lambda: [
            SimpleNamespace(metadata={"Name": name}, version=version)
            for name, version in protected.items()
        ],
    )
    monkeypatch.setattr(colab.importlib.metadata, "version", protected.__getitem__)
    calls = []

    def install(command, **kwargs):
        constraint = command[command.index("--constraint") + 1]
        contents = Path(constraint).read_text(encoding="utf-8")
        assert "torch==2.9.0" in contents and "numpy==2.0.0" in contents
        assert "nvidia-cuda-runtime-cu12==12.6.0" in contents
        calls.append(command)

    monkeypatch.setattr(colab.subprocess, "run", install)
    assert install_missing_dependencies(requirements, required_modules=("timm",)) == ["timm>=1,<2"]
    assert calls[0][-1] == "timm>=1,<2"
    assert not any("transformers" in value for value in calls[0])
    assert not list(tmp_path.glob(".colab-stack-*"))


def test_complete_artifact_sync_and_nested_destination_rejection(tmp_path):
    source, destination = tmp_path / "output", tmp_path / "drive"
    source.mkdir()
    for name in ("pipeline.pkl", "vocab.txt", "submission.zip", "solution.ipynb"):
        (source / name).write_bytes(b"artifact")
    copied = sync_artifacts(source, destination)
    assert {path.name for path in copied} == {
        "pipeline.pkl",
        "vocab.txt",
        "submission.zip",
        "solution.ipynb",
    }
    with pytest.raises(ValueError):
        sync_artifacts(source, source / "backup")
