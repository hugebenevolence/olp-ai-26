from __future__ import annotations

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from olp_ai_26.core.config import TrainerConfig
from olp_ai_26.core.metrics import evaluate_metric
from olp_ai_26.core.trainer import Trainer, oom_recovery_batch_size, resolve_amp


def test_tiny_training_loop(tmp_path):
    torch.manual_seed(1)
    inputs = torch.cat([torch.randn(20, 2) - 2, torch.randn(20, 2) + 2])
    labels = torch.tensor([0] * 20 + [1] * 20)
    loader = DataLoader(TensorDataset(inputs, labels), batch_size=8, shuffle=True)
    model = nn.Linear(2, 2)
    config = TrainerConfig(epochs=3, learning_rate=0.05, scheduler="none", patience=3)
    backup_dir = tmp_path / "backup"
    trainer = Trainer(
        model,
        nn.CrossEntropyLoss(),
        config,
        output_dir=tmp_path,
        device="cpu",
        metric_fn=lambda y, p: evaluate_metric("accuracy", y, p),
        backup_dir=backup_dir,
    )
    history = trainer.fit(loader, loader)
    assert history[-1]["metric"] > 0.9
    assert trainer.checkpoint_path.exists()
    assert (backup_dir / "best.pt").exists()
    assert oom_recovery_batch_size(8) == 4
    assert resolve_amp(torch.device("cpu"), "auto") == (False, None, False)
