"""Checks for optimizer semantics and the experimental schedule boundaries."""

from copy import deepcopy

import pytest
import torch

from hw2.optim import CaffeSGD, learning_rate


def test_caffe_momentum_keeps_the_previous_learning_rate() -> None:
    parameter = torch.nn.Parameter(torch.tensor(2.0))
    optimizer = CaffeSGD(
        [{"params": [parameter], "lr_mult": 2.0, "weight_decay": 0.25}]
    )
    parameter.grad = torch.tensor(3.0)
    optimizer.step(learning_rate=0.1, momentum=0.5)
    assert parameter.item() == pytest.approx(1.3)
    parameter.grad = torch.tensor(-1.0)
    optimizer.step(learning_rate=0.01, momentum=0.8)
    # Previous displacement 0.7 retains its scale after a tenfold LR decrease.
    assert parameter.item() == pytest.approx(0.7535)
    assert parameter.grad.item() == -1.0


def test_optimizer_checkpoint_preserves_the_next_update() -> None:
    parameter = torch.nn.Parameter(torch.tensor([2.0, -1.0]))
    optimizer = CaffeSGD([{"params": [parameter], "weight_decay": 0.1}])
    parameter.square().sum().backward()
    optimizer.step(0.1, 0.9)

    resumed_parameter = torch.nn.Parameter(parameter.detach().clone())
    resumed_optimizer = CaffeSGD([{"params": [resumed_parameter]}])
    resumed_optimizer.load_state_dict(deepcopy(optimizer.state_dict()))
    for weights, solver in (
        (parameter, optimizer),
        (resumed_parameter, resumed_optimizer),
    ):
        solver.zero_grad()
        weights.square().sum().backward()
        solver.step(0.01, 0.8)
    torch.testing.assert_close(parameter, resumed_parameter)


@pytest.mark.parametrize(
    ("step", "expected"),
    [(0, 0.35), (49_999, 0.35), (50_000, 0.035), (69_999, 0.035), (70_000, 0.0035)],
)
def test_cifar_baseline_drops_at_the_published_updates(
    step: int, expected: float
) -> None:
    assert learning_rate("cifar_baseline", step) == pytest.approx((expected, 0.9))


def test_cifar_cycle_has_no_extra_annealing_or_momentum_cycle() -> None:
    assert learning_rate("cifar_cycle", 0) == pytest.approx((0.1, 0.9))
    assert learning_rate("cifar_cycle", 5_000) == pytest.approx((3.0, 0.9))
    assert learning_rate("cifar_cycle", 9_999) == pytest.approx((0.10058, 0.9))
    assert learning_rate("cifar_cycle", 10_000) == pytest.approx((0.1, 0.9))


def test_mnist_cycle_has_five_epoch_peak_and_a_two_epoch_tail() -> None:
    rates, momenta = zip(*(learning_rate("mnist_cycle", step) for step in range(1407)))
    assert rates[0] == pytest.approx(0.01)
    assert max(range(len(rates)), key=rates.__getitem__) == 586
    assert rates[586] == pytest.approx(0.1, abs=1e-5)
    assert momenta[586] == pytest.approx(0.8, abs=2e-5)
    assert rates[1171] > 0.01 > rates[1172]
    assert rates[-1] == pytest.approx(0.00001)
    assert momenta[-1] == pytest.approx(0.95)
    assert learning_rate("mnist_cycle", 1407) == pytest.approx((0.00001, 0.95))


def test_short_mnist_control_is_the_same_baseline_at_equal_budget() -> None:
    for step in (0, 586, 1406):
        assert learning_rate("mnist_short", step) == learning_rate(
            "mnist_baseline", step
        )
    assert learning_rate("mnist_baseline", 10_000) == pytest.approx(
        (0.005946035575, 0.9)
    )
