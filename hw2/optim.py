"""Caffe's SGD update and the learning-rate schedules used in the experiments."""

from collections.abc import Iterable
from math import ceil
from typing import Any

import torch


class CaffeSGD(torch.optim.Optimizer):
    """Store momentum in update units, including each step's learning rate."""

    def __init__(self, param_groups: Iterable[dict[str, Any]]) -> None:
        super().__init__(param_groups, {"lr_mult": 1.0, "weight_decay": 0.0})
        for group in self.param_groups:
            for parameter in group["params"]:
                self.state[parameter]["momentum_buffer"] = torch.zeros_like(parameter)

    @torch.no_grad()
    def step(self, learning_rate: float, momentum: float) -> None:
        for group in self.param_groups:
            rate = learning_rate * group["lr_mult"]
            for parameter in group["params"]:
                if parameter.grad is None:
                    continue
                gradient = parameter.grad.add(parameter, alpha=group["weight_decay"])
                velocity = self.state[parameter]["momentum_buffer"]
                velocity.mul_(momentum).add_(gradient, alpha=rate)
                parameter.add_(velocity, alpha=-1)


def learning_rate(experiment: str, step: int) -> tuple[float, float]:
    """Return learning rate and momentum before a zero-based optimizer update."""
    if experiment in {"mnist_baseline", "mnist_short"}:
        return 0.01 * (1 + 0.0001 * step) ** -0.75, 0.9

    if experiment == "mnist_cycle":
        epoch = min(step * 512 / 60_000, 12.0)
        # Include the assumed terminal LR in the last of the 1,407 updates.
        if step >= ceil(12 * 60_000 / 512) - 1:
            epoch = 12.0
        if epoch <= 5:
            return 0.01 + 0.09 * epoch / 5, 0.95 - 0.15 * epoch / 5
        if epoch <= 10:
            return 0.1 - 0.09 * (epoch - 5) / 5, 0.8 + 0.15 * (epoch - 5) / 5
        # The paper omits the tail endpoint: use a linear decline to 1e-5.
        return 0.01 + (0.00001 - 0.01) * (epoch - 10) / 2, 0.95

    if experiment == "cifar_baseline":
        drops = int(step >= 50_000) + int(step >= 70_000)
        return 0.35 * 0.1**drops, 0.9

    if experiment == "cifar_cycle":
        progress = min(step, 10_000) / 5_000
        return 0.1 + 2.9 * (1 - abs(progress - 1)), 0.9

    raise ValueError(f"Unknown experiment: {experiment}")
