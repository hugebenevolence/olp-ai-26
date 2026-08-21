from __future__ import annotations

import torch
from torch import nn


def fgsm(
    model: nn.Module,
    images: torch.Tensor,
    labels: torch.Tensor,
    *,
    epsilon: float,
    loss_fn: nn.Module | None = None,
    clip_min: float = 0.0,
    clip_max: float = 1.0,
) -> torch.Tensor:
    loss_fn = loss_fn or nn.CrossEntropyLoss()
    attacked = images.detach().clone().requires_grad_(True)
    loss = loss_fn(model(attacked), labels)
    gradient = torch.autograd.grad(loss, attacked)[0]
    return (attacked + epsilon * gradient.sign()).clamp(clip_min, clip_max).detach()


def pgd(
    model: nn.Module,
    images: torch.Tensor,
    labels: torch.Tensor,
    *,
    epsilon: float,
    step_size: float,
    steps: int = 10,
    random_start: bool = True,
    loss_fn: nn.Module | None = None,
    clip_min: float = 0.0,
    clip_max: float = 1.0,
) -> torch.Tensor:
    loss_fn = loss_fn or nn.CrossEntropyLoss()
    original = images.detach()
    if random_start:
        attacked = original + torch.empty_like(original).uniform_(-epsilon, epsilon)
        attacked = attacked.clamp(clip_min, clip_max)
    else:
        attacked = original.clone()
    for _ in range(steps):
        attacked.requires_grad_(True)
        loss = loss_fn(model(attacked), labels)
        gradient = torch.autograd.grad(loss, attacked)[0]
        attacked = attacked.detach() + step_size * gradient.sign()
        delta = (attacked - original).clamp(-epsilon, epsilon)
        attacked = (original + delta).clamp(clip_min, clip_max).detach()
    return attacked


def perturbation_statistics(original: torch.Tensor, attacked: torch.Tensor) -> dict[str, float]:
    difference = (attacked - original).detach().float()
    return {
        "linf": float(difference.abs().max()),
        "l2_mean": float(difference.flatten(1).norm(p=2, dim=1).mean()),
        "changed_fraction": float((difference != 0).float().mean()),
    }
