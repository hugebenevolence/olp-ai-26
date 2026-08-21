"""Compact task-agnostic PyTorch training with AMP, scheduling, and atomic checkpoints."""

from __future__ import annotations

import math
import shutil
from collections.abc import Callable, Iterable, Mapping
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.optim import Optimizer

from olp_ai_26.core.config import TimeBudget, TrainerConfig
from olp_ai_26.core.experiment import ExperimentLogger


def resolve_amp(
    device: torch.device,
    preference: str | bool,
) -> tuple[bool, torch.dtype | None, bool]:
    """Return autocast enabled, dtype, and GradScaler enabled for the actual GPU."""
    if device.type != "cuda" or preference in {False, "none"}:
        return False, None, False
    normalized = "auto" if preference is True else str(preference)
    if normalized == "bf16":
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    elif normalized == "fp16":
        dtype = torch.float16
    elif normalized == "auto":
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    else:
        raise ValueError(f"Unsupported mixed precision mode: {preference}")
    return True, dtype, dtype == torch.float16


def build_optimizer(model: nn.Module, config: TrainerConfig) -> Optimizer:
    """Create the configured optimizer over trainable model parameters."""
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not parameters:
        raise ValueError("Model has no trainable parameters")
    if config.optimizer == "adamw":
        return torch.optim.AdamW(
            parameters, lr=config.learning_rate, weight_decay=config.weight_decay
        )
    if config.optimizer == "adam":
        return torch.optim.Adam(
            parameters, lr=config.learning_rate, weight_decay=config.weight_decay
        )
    if config.optimizer == "sgd":
        return torch.optim.SGD(
            parameters,
            lr=config.learning_rate,
            momentum=0.9,
            nesterov=True,
            weight_decay=config.weight_decay,
        )
    if config.optimizer == "adafactor":
        from transformers.optimization import Adafactor

        return Adafactor(
            parameters,
            lr=config.learning_rate,
            relative_step=False,
            scale_parameter=False,
            weight_decay=config.weight_decay,
        )
    raise ValueError(f"Unsupported optimizer: {config.optimizer}")


def build_scheduler(
    optimizer: Optimizer,
    config: TrainerConfig,
    total_steps: int,
) -> Any | None:
    """Create the configured step- or metric-driven learning-rate scheduler."""
    if config.scheduler == "none":
        return None
    if total_steps < 1:
        raise ValueError("total_steps must be positive")
    if config.scheduler == "cosine":
        warmup_steps = round(total_steps * config.warmup_ratio)

        def schedule(step: int) -> float:
            if warmup_steps and step < warmup_steps:
                return max(1e-8, step / warmup_steps)
            progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
            return 0.5 * (1 + math.cos(math.pi * min(1.0, progress)))

        return torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)
    if config.scheduler == "onecycle":
        return torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=config.learning_rate, total_steps=total_steps
        )
    if config.scheduler == "plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max" if config.maximize_metric else "min", patience=1
        )
    raise ValueError(f"Unsupported scheduler: {config.scheduler}")


def _move(value: Any, device: torch.device) -> Any:
    if isinstance(value, torch.Tensor):
        return value.to(device, non_blocking=True)
    if isinstance(value, Mapping):
        return {key: _move(item, device) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_move(item, device) for item in value)
    if isinstance(value, list):
        return [_move(item, device) for item in value]
    return value


def _split_batch(batch: Any, device: torch.device) -> tuple[Any, torch.Tensor]:
    batch = _move(batch, device)
    if isinstance(batch, Mapping):
        inputs = dict(batch)
        try:
            targets = inputs.pop("labels")
        except KeyError as error:
            raise KeyError("Dictionary batches must contain a 'labels' tensor") from error
        return inputs, targets
    if isinstance(batch, (tuple, list)) and len(batch) >= 2:
        return batch[0], batch[1]
    raise TypeError("Batch must be (inputs, targets) or a dictionary containing labels")


def _forward(model: nn.Module, inputs: Any) -> Any:
    output = model(**inputs) if isinstance(inputs, Mapping) else model(inputs)
    if hasattr(output, "logits"):
        return output.logits
    if isinstance(output, Mapping) and "logits" in output:
        return output["logits"]
    return output


def default_prediction_decoder(logits: torch.Tensor) -> torch.Tensor:
    """Decode binary logits at zero or multiclass logits by argmax."""
    if logits.ndim == 1 or logits.shape[-1] == 1:
        return (logits.reshape(-1) >= 0).long()
    return logits.argmax(dim=-1)


class Trainer:
    """Compact PyTorch trainer for classifier-like competition tasks."""

    def __init__(
        self,
        model: nn.Module,
        criterion: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        config: TrainerConfig,
        *,
        output_dir: Path | str,
        device: str = "auto",
        metric_fn: Callable[[np.ndarray, np.ndarray], float] | None = None,
        prediction_decoder: Callable[[torch.Tensor], torch.Tensor] = default_prediction_decoder,
        time_budget: TimeBudget | None = None,
        logger: ExperimentLogger | None = None,
        backup_dir: Path | str | None = None,
    ) -> None:
        resolved = "cuda" if device == "auto" and torch.cuda.is_available() else device
        self.device = torch.device("cpu" if resolved == "auto" else resolved)
        self.model = model.to(self.device)
        self.criterion = criterion
        self.config = config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.metric_fn = metric_fn
        self.prediction_decoder = prediction_decoder
        self.time_budget = time_budget
        self.logger = logger
        self.backup_dir = Path(backup_dir) if backup_dir else None
        if self.backup_dir:
            self.backup_dir.mkdir(parents=True, exist_ok=True)
        self.history: list[dict[str, float]] = []
        self.checkpoint_path = self.output_dir / config.checkpoint_name
        self.amp_enabled, self.amp_dtype, scaler_enabled = resolve_amp(
            self.device, config.mixed_precision
        )
        self.scaler = torch.amp.GradScaler("cuda", enabled=scaler_enabled)

    def fit(
        self, train_loader: Iterable[Any], valid_loader: Iterable[Any]
    ) -> list[dict[str, float]]:
        """Train with early stopping and reload the best saved checkpoint."""
        optimizer = build_optimizer(self.model, self.config)
        batches_per_epoch = len(train_loader)  # type: ignore[arg-type]
        update_steps = math.ceil(batches_per_epoch / self.config.gradient_accumulation_steps)
        scheduler = build_scheduler(optimizer, self.config, update_steps * self.config.epochs)
        best = -math.inf if self.config.maximize_metric else math.inf
        stale_epochs = 0

        for epoch in range(1, self.config.epochs + 1):
            if self.time_budget and self.time_budget.should_stop:
                break
            train_loss = self._train_epoch(train_loader, optimizer, scheduler)
            valid = self.evaluate(valid_loader)
            score = valid.get(
                "metric", -valid["loss"] if self.config.maximize_metric else valid["loss"]
            )
            improved = score > best if self.config.maximize_metric else score < best
            record = {"epoch": float(epoch), "train_loss": train_loss, **valid}
            self.history.append(record)
            if self.logger:
                self.logger.log("epoch_end", **record)
            if improved:
                best = score
                stale_epochs = 0
                self.save_checkpoint(optimizer, epoch, score)
            else:
                stale_epochs += 1
            if scheduler is not None and self.config.scheduler == "plateau":
                scheduler.step(score)
            if stale_epochs >= self.config.patience:
                break
        if self.checkpoint_path.exists():
            self.load_checkpoint()
        return self.history

    def _train_epoch(
        self,
        loader: Iterable[Any],
        optimizer: Optimizer,
        scheduler: Any | None,
    ) -> float:
        self.model.train()
        optimizer.zero_grad(set_to_none=True)
        total_loss = 0.0
        count = 0
        for step, batch in enumerate(loader, start=1):
            inputs, targets = _split_batch(batch, self.device)
            with torch.autocast(
                device_type=self.device.type,
                dtype=self.amp_dtype,
                enabled=self.amp_enabled,
            ):
                logits = _forward(self.model, inputs)
                loss = self.criterion(logits, targets) / self.config.gradient_accumulation_steps
            self.scaler.scale(loss).backward()
            total_loss += float(loss.detach()) * self.config.gradient_accumulation_steps
            count += 1
            should_update = step % self.config.gradient_accumulation_steps == 0
            if should_update:
                if self.config.max_grad_norm is not None:
                    self.scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.config.max_grad_norm
                    )
                self.scaler.step(optimizer)
                self.scaler.update()
                optimizer.zero_grad(set_to_none=True)
                if scheduler is not None and self.config.scheduler != "plateau":
                    scheduler.step()
            if self.time_budget and self.time_budget.should_stop:
                break
        if count and count % self.config.gradient_accumulation_steps:
            self.scaler.step(optimizer)
            self.scaler.update()
            optimizer.zero_grad(set_to_none=True)
        return total_loss / max(1, count)

    @torch.inference_mode()
    def evaluate(self, loader: Iterable[Any]) -> dict[str, float]:
        """Return average loss and, when configured, decoded validation metric."""
        self.model.eval()
        losses: list[float] = []
        predictions: list[np.ndarray] = []
        targets_all: list[np.ndarray] = []
        for batch in loader:
            inputs, targets = _split_batch(batch, self.device)
            with torch.autocast(
                device_type=self.device.type,
                dtype=self.amp_dtype,
                enabled=self.amp_enabled,
            ):
                logits = _forward(self.model, inputs)
                losses.append(float(self.criterion(logits, targets)))
            predictions.append(self.prediction_decoder(logits).detach().cpu().numpy())
            targets_all.append(targets.detach().cpu().numpy())
        result = {"loss": float(np.mean(losses)) if losses else math.nan}
        if self.metric_fn and predictions:
            result["metric"] = float(
                self.metric_fn(np.concatenate(targets_all), np.concatenate(predictions))
            )
        return result

    def save_checkpoint(self, optimizer: Optimizer, epoch: int, score: float) -> Path:
        """Atomically save model/optimizer state and optionally copy it to durable storage."""
        temporary = self.checkpoint_path.with_suffix(self.checkpoint_path.suffix + ".tmp")
        torch.save(
            {
                "model": self.model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
                "score": score,
            },
            temporary,
        )
        temporary.replace(self.checkpoint_path)
        if self.backup_dir:
            backup = self.backup_dir / self.checkpoint_path.name
            backup_temporary = backup.with_suffix(backup.suffix + ".tmp")
            shutil.copy2(self.checkpoint_path, backup_temporary)
            backup_temporary.replace(backup)
        return self.checkpoint_path

    def load_checkpoint(self) -> dict[str, Any]:
        """Load the best checkpoint into the model and return the full saved payload."""
        source = self.checkpoint_path
        if not source.exists() and self.backup_dir:
            source = self.backup_dir / self.checkpoint_path.name
        checkpoint = torch.load(source, map_location=self.device, weights_only=True)
        self.model.load_state_dict(checkpoint["model"])
        return checkpoint


def oom_recovery_batch_size(current_batch_size: int) -> int:
    """Halve a batch size after CUDA OOM while refusing to go below one."""
    if current_batch_size <= 1:
        raise RuntimeError("CUDA out of memory even at batch size 1")
    return max(1, current_batch_size // 2)
