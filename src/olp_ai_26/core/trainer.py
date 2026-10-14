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

from olp_ai_26.core.config import TimeBudget, TrainerConfig, resolve_device
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


def _split_batch(batch: Any, device: torch.device) -> tuple[Any, Any]:
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
        criterion: Callable[[Any, Any], torch.Tensor],
        config: TrainerConfig,
        *,
        output_dir: Path | str,
        device: str = "auto",
        metric_fn: Callable[[np.ndarray, np.ndarray], float] | None = None,
        prediction_decoder: Callable[[torch.Tensor], torch.Tensor] = default_prediction_decoder,
        target_decoder: Callable[[Any], torch.Tensor] | None = None,
        time_budget: TimeBudget | None = None,
        logger: ExperimentLogger | None = None,
        backup_dir: Path | str | None = None,
    ) -> None:
        self.device = torch.device(resolve_device(device))
        self.model = model.to(self.device)
        self.criterion = criterion
        self.config = config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.metric_fn = metric_fn
        self.prediction_decoder = prediction_decoder
        self.target_decoder = target_decoder or (lambda targets: targets)
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
        self.history = []
        optimizer = build_optimizer(self.model, self.config)
        batches_per_epoch = len(train_loader)  # type: ignore[arg-type]
        if batches_per_epoch == 0:
            raise ValueError("Training loader must not be empty")
        update_steps = math.ceil(batches_per_epoch / self.config.gradient_accumulation_steps)
        scheduler = build_scheduler(optimizer, self.config, update_steps * self.config.epochs)
        best = -math.inf if self.config.maximize_metric else math.inf
        stale_epochs = 0

        saved_this_run = False
        for epoch in range(1, self.config.epochs + 1):
            if self.time_budget and self.time_budget.should_stop:
                break
            train_loss = self._train_epoch(train_loader, optimizer, scheduler)
            if self.time_budget and self.time_budget.should_stop:
                break
            valid = self.evaluate(valid_loader)
            score = valid.get(
                "metric", -valid["loss"] if self.config.maximize_metric else valid["loss"]
            )
            if not math.isfinite(score):
                raise ValueError("Validation score is not finite; check data, loss, and metric")
            improved = score > best if self.config.maximize_metric else score < best
            record = {"epoch": float(epoch), "train_loss": train_loss, **valid}
            self.history.append(record)
            if self.logger:
                self.logger.log("epoch_end", **record)
            if improved:
                best = score
                stale_epochs = 0
                self.save_checkpoint(optimizer, epoch, score)
                saved_this_run = True
            else:
                stale_epochs += 1
            if scheduler is not None and self.config.scheduler == "plateau":
                scheduler.step(score)
            if stale_epochs >= self.config.patience:
                break
        if saved_this_run:
            self.load_checkpoint()
        if not self.history:
            raise TimeoutError(
                "No epoch completed before the time reserve; use a shorter baseline run"
            )
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
        examples = 0
        pending_examples = 0
        pending_batches = 0

        def update() -> None:
            self.scaler.unscale_(optimizer)
            # Each microbatch contributes its sample-weighted loss. Normalize even a short tail.
            factor = self.config.gradient_accumulation_steps / pending_examples
            for parameter in self.model.parameters():
                if parameter.grad is not None:
                    parameter.grad.mul_(factor)
            if self.config.max_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
            previous_scale = self.scaler.get_scale()
            self.scaler.step(optimizer)
            self.scaler.update()
            optimizer.zero_grad(set_to_none=True)
            # GradScaler skips an optimizer update on overflow; the scheduler must skip it too.
            if (
                scheduler is not None
                and self.config.scheduler != "plateau"
                and self.scaler.get_scale() >= previous_scale
            ):
                scheduler.step()

        for batch in loader:
            if self.time_budget and self.time_budget.should_stop:
                break
            inputs, targets = _split_batch(batch, self.device)
            batch_size = len(self.target_decoder(targets))
            with torch.autocast(
                device_type=self.device.type,
                dtype=self.amp_dtype,
                enabled=self.amp_enabled,
            ):
                logits = _forward(self.model, inputs)
                loss = self.criterion(logits, targets)
            self.scaler.scale(
                loss * batch_size / self.config.gradient_accumulation_steps
            ).backward()
            total_loss += float(loss.detach()) * batch_size
            examples += batch_size
            pending_examples += batch_size
            pending_batches += 1
            if pending_batches == self.config.gradient_accumulation_steps:
                update()
                pending_examples = pending_batches = 0
        if pending_examples:
            update()
        return total_loss / max(1, examples)

    @torch.inference_mode()
    def evaluate(self, loader: Iterable[Any]) -> dict[str, float]:
        """Return average loss and, when configured, decoded validation metric."""
        self.model.eval()
        loss_total = 0.0
        examples = 0
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
                batch_size = len(self.target_decoder(targets))
                loss_total += float(self.criterion(logits, targets)) * batch_size
                examples += batch_size
            predictions.append(self.prediction_decoder(logits).detach().cpu().numpy())
            targets_all.append(self.target_decoder(targets).detach().cpu().numpy())
        if not examples:
            raise ValueError("Validation loader must not be empty")
        result = {"loss": loss_total / examples}
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


def hf_time_budget_callback(budget: TimeBudget) -> Any:
    """Create a lazy HF callback that stops training and requests final evaluation/save."""
    from transformers import TrainerCallback

    class BudgetCallback(TrainerCallback):
        def on_step_end(self, args: Any, state: Any, control: Any, **kwargs: Any) -> Any:
            if budget.should_stop:
                control.should_training_stop = True
                control.should_evaluate = True
                control.should_save = True
            return control

    return BudgetCallback()


def local_hf_trainer_class(*, generation: bool = False) -> type[Any]:
    """Create an HF trainer that reloads its own best checkpoint through model key conversion.

    Some Transformers versions rename saved encoder keys. Loading the raw state dictionary can
    silently miss those keys. ``from_pretrained`` applies the model's conversion and tied-weight
    handling, then copies complete state into the existing (possibly accelerator-wrapped) model.
    This loads only the checkpoint produced by this run, never a remote pretrained model.
    """
    from transformers import Seq2SeqTrainer
    from transformers import Trainer as HFTrainer

    base = Seq2SeqTrainer if generation else HFTrainer

    class LocalBestTrainer(base):
        def _load_best_model(self) -> None:
            path = self.state.best_model_checkpoint
            if path is None:
                raise RuntimeError("No evaluated HF checkpoint is available")
            restored, info = type(self.model).from_pretrained(
                path, local_files_only=True, output_loading_info=True
            )
            if (
                info.get("missing_keys")
                or info.get("unexpected_keys")
                or info.get("mismatched_keys")
            ):
                raise RuntimeError(f"Best HF checkpoint does not match the model: {info}")
            self.model.load_state_dict(restored.state_dict(), strict=True)

    return LocalBestTrainer
