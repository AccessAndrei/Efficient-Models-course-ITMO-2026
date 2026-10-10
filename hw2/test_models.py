"""Regression checks for details that differ from common PyTorch model ports."""

import math

import torch
from torch import nn

from hw2.models import CaffeBatchNorm2d, LeNet, ResidualBlock, ResNet56


def test_lenet_contract_and_gradients() -> None:
    torch.manual_seed(7)
    model = LeNet()
    assert sum(p.numel() for p in model.parameters()) == 431_080
    for layer in (model.conv1, model.conv2, model.fc1, model.fc2):
        bound = math.sqrt(3 / layer.weight[0].numel())
        assert layer.weight.detach().abs().max() <= bound
        assert torch.count_nonzero(layer.bias) == 0
    logits = model(torch.randn(2, 1, 28, 28))
    assert logits.shape == (2, 10)
    nn.functional.cross_entropy(logits, torch.tensor([0, 1])).backward()
    assert all(
        p.grad is not None and p.grad.isfinite().all() for p in model.parameters()
    )


def test_lenet_convolutions_do_not_clip_negative_values() -> None:
    model = LeNet()
    nn.init.zeros_(model.conv1.weight)
    nn.init.constant_(model.conv1.bias, -1)
    inputs = []
    handle = model.conv2.register_forward_pre_hook(
        lambda module, args: inputs.append(args[0].detach())
    )
    model(torch.zeros(1, 1, 28, 28))
    handle.remove()
    torch.testing.assert_close(inputs[0], -torch.ones(1, 20, 12, 12))


def test_resnet_author_topology_and_gradients() -> None:
    torch.manual_seed(7)
    model = ResNet56(batch_norm_decay=0.95)
    assert sum(p.numel() for p in model.parameters()) == 808_682
    assert sum(isinstance(layer, nn.Conv2d) for layer in model.modules()) == 55
    assert sum(isinstance(layer, CaffeBatchNorm2d) for layer in model.modules()) == 55
    assert [len(model.stage1), len(model.stage2), len(model.stage3)] == [9, 9, 9]
    assert model.stage2[0].conv2.out_channels == 16
    assert model.stage3[0].conv2.out_channels == 32
    logits = model(torch.randn(2, 3, 32, 32))
    assert logits.shape == (2, 10)
    nn.functional.cross_entropy(logits, torch.tensor([0, 1])).backward()
    assert all(
        p.grad is not None and p.grad.isfinite().all() for p in model.parameters()
    )


def test_transition_pools_then_appends_zero_channels() -> None:
    block = ResidualBlock(2, batch_norm_decay=0.999, downsample=True).eval()
    for layer in (block.conv1, block.conv2):
        nn.init.zeros_(layer.weight)
        nn.init.zeros_(layer.bias)
    image = torch.arange(16.0).reshape(1, 1, 4, 4).expand(1, 2, 4, 4)
    output = block(image)
    pooled = torch.tensor([[5.0, 6.5], [11.0, 12.5]]).expand(1, 2, 2, 2)
    torch.testing.assert_close(output[:, :2], pooled)
    torch.testing.assert_close(output[:, 2:], torch.zeros_like(pooled))


def test_caffe_batch_norm_running_statistics_and_resume() -> None:
    norm = CaffeBatchNorm2d(1, decay=0.95)
    first = torch.tensor([1.0, 3.0, 5.0, 7.0]).reshape(1, 1, 2, 2)
    output = norm(first)
    torch.testing.assert_close(output, (first - 4) / math.sqrt(5 + 1e-5))
    torch.testing.assert_close(norm.running_var, torch.tensor([20 / 3]))

    resumed = CaffeBatchNorm2d(1, decay=0.95)
    resumed.load_state_dict(norm.state_dict())
    norm(first * 2)
    resumed(first * 2)
    expected_mean = torch.tensor([(0.95 * 4 + 8) / 1.95])
    expected_var = torch.tensor([(0.95 * 20 / 3 + 80 / 3) / 1.95])
    torch.testing.assert_close(norm.running_mean, expected_mean)
    torch.testing.assert_close(norm.running_var, expected_var)
    torch.testing.assert_close(resumed.running_var, norm.running_var)
    torch.testing.assert_close(resumed.running_mean, norm.running_mean)
    norm.eval()
    torch.testing.assert_close(
        norm(first), (first - expected_mean) / torch.sqrt(expected_var + 1e-5)
    )


def test_non_root_microbatch_uses_local_statistics_without_updating_buffers() -> None:
    norm = CaffeBatchNorm2d(1, decay=0.95)
    norm(torch.arange(4.0).reshape(1, 1, 2, 2))
    norm.update_stats = False
    before = {name: value.clone() for name, value in norm.named_buffers()}
    mass = norm.running_mass
    image = torch.arange(4.0).reshape(1, 1, 2, 2) + 100
    output = norm(image)
    torch.testing.assert_close(output.mean(), torch.tensor(0.0), atol=1e-5, rtol=0)
    assert norm.running_mass == mass
    for name, value in norm.named_buffers():
        torch.testing.assert_close(value, before[name])


def test_grouped_resnet_matches_eight_interleaved_microbatches() -> None:
    # Float64 isolates mathematical equivalence from batch-dependent rounding.
    torch.manual_seed(123)
    grouped = ResNet56(0.95, batch_norm_groups=8).double()
    serial = ResNet56(0.95).double()
    serial.load_state_dict(grouped.state_dict())
    images = torch.randn(16, 3, 32, 32, dtype=torch.float64)
    labels = torch.arange(16) % 10
    logits = grouped(images)
    loss = nn.functional.cross_entropy(logits, labels)
    loss.backward()
    serial_loss = 0.0
    for rank in range(8):
        for layer in serial.modules():
            if isinstance(layer, CaffeBatchNorm2d):
                layer.update_stats = rank == 0
        prediction = serial(images[rank::8])
        partial_loss = nn.functional.cross_entropy(prediction, labels[rank::8]) / 8
        partial_loss.backward()
        serial_loss += partial_loss.item()
        torch.testing.assert_close(logits[rank::8], prediction, atol=1e-10, rtol=1e-8)
    assert math.isclose(loss.item(), serial_loss, abs_tol=1e-10)
    for actual, expected in zip(grouped.parameters(), serial.parameters()):
        torch.testing.assert_close(actual.grad, expected.grad, atol=1e-10, rtol=1e-8)
    for actual, expected in zip(grouped.buffers(), serial.buffers()):
        torch.testing.assert_close(actual, expected, atol=1e-10, rtol=1e-8)
    grouped.eval()
    serial.eval()
    with torch.no_grad():
        torch.testing.assert_close(
            grouped(images), serial(images), atol=1e-9, rtol=1e-8
        )
