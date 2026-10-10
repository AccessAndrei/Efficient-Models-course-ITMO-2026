"""PyTorch ports of the Caffe LeNet and authors' Resnet56Cifar models.

Sources: BVLC/caffe/examples/mnist/lenet_train_test.prototxt and
lnsmith54/super-convergence/architectures/Resnet56Cifar.prototxt.
"""

import math

import torch
from torch import nn
from torch.nn import functional as F


def caffe_xavier(layer: nn.Conv2d | nn.Linear) -> None:
    """Caffe's default Xavier uses fan_in, not PyTorch's fan_in/fan_out mean."""
    bound = math.sqrt(3 / layer.weight[0].numel())
    nn.init.uniform_(layer.weight, -bound, bound)
    nn.init.zeros_(layer.bias)


class LeNet(nn.Module):
    """The Caffe MNIST example: its only ReLU is after the first dense layer."""

    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(1, 20, 5)
        self.conv2 = nn.Conv2d(20, 50, 5)
        self.fc1 = nn.Linear(50 * 4 * 4, 500)
        self.fc2 = nn.Linear(500, 10)
        for layer in (self.conv1, self.conv2, self.fc1, self.fc2):
            caffe_xavier(layer)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.max_pool2d(self.conv1(x), 2)
        x = F.max_pool2d(self.conv2(x), 2)
        x = F.relu(self.fc1(x.flatten(1)))
        return self.fc2(x)


class CaffeBatchNorm2d(nn.BatchNorm2d):
    """Caffe BatchNorm + Scale, including normalized running-statistic sums.

    groups>1 normalizes interleaved replicas together; only group 0 saves stats.
    Set update_stats=False on non-root microbatches when simulating Caffe's
    multiple GPUs: they normalize locally but do not change saved root stats.
    """

    def __init__(self, channels: int, decay: float, groups: int = 1) -> None:
        super().__init__(channels, eps=1e-5)
        self.decay = decay
        self.groups = groups
        self.running_mass = 0.0
        self.update_stats = True
        self.running_var.zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training and self.update_stats:
            self.running_mass = self.decay * self.running_mass + 1
            self.momentum = 1 / self.running_mass
        if self.training and self.groups > 1:
            return self._grouped_forward(x)
        if self.training and not self.update_stats:
            return F.batch_norm(
                x, None, None, self.weight, self.bias, True, 0, self.eps
            )
        return super().forward(x)

    def _grouped_forward(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize interleaved replicas x[rank::groups] in one fused call."""
        batch, channels, height, width = x.shape
        grouped = x.reshape(batch // self.groups, channels * self.groups, height, width)
        mean = self.running_mean.repeat(self.groups) if self.update_stats else None
        variance = self.running_var.repeat(self.groups) if self.update_stats else None
        output = F.batch_norm(
            grouped,
            mean,
            variance,
            self.weight.repeat(self.groups),
            self.bias.repeat(self.groups),
            True,
            self.momentum,
            self.eps,
        )
        if self.update_stats:
            self.running_mean.copy_(mean[:channels])
            self.running_var.copy_(variance[:channels])
            self.num_batches_tracked.add_(1)
        return output.reshape(x.shape)

    def get_extra_state(self) -> float:
        """Keep the Python scalar in checkpoints without a GPU synchronization."""
        return self.running_mass

    def set_extra_state(self, state: float) -> None:
        self.running_mass = state


class ResidualBlock(nn.Module):
    """Author block; downsampling appends zero channels AFTER the residual sum."""

    def __init__(
        self, channels: int, batch_norm_decay: float, downsample: bool = False
    ) -> None:
        super().__init__()
        self.downsample = downsample
        self.conv1 = nn.Conv2d(channels, channels, 3, 2 if downsample else 1, 1)
        self.bn1 = CaffeBatchNorm2d(channels, batch_norm_decay)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn2 = CaffeBatchNorm2d(channels, batch_norm_decay)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = F.relu(self.bn1(self.conv1(x)))
        residual = self.bn2(self.conv2(residual))
        shortcut = F.avg_pool2d(x, 3, 2, ceil_mode=True) if self.downsample else x
        x = F.relu(residual + shortcut)
        if self.downsample:
            x = torch.cat((x, torch.zeros_like(x)), dim=1)
        return x


class ResNet56(nn.Module):
    """Exact layer topology of the authors' CIFAR-10 prototxt (27 blocks)."""

    def __init__(
        self, batch_norm_decay: float = 0.999, batch_norm_groups: int = 1
    ) -> None:
        super().__init__()
        self.conv = nn.Conv2d(3, 16, 3, padding=1)
        self.bn = CaffeBatchNorm2d(16, batch_norm_decay)
        self.stage1 = nn.Sequential(
            *[ResidualBlock(16, batch_norm_decay) for _ in range(9)]
        )
        self.stage2 = nn.Sequential(
            ResidualBlock(16, batch_norm_decay, downsample=True),
            *[ResidualBlock(32, batch_norm_decay) for _ in range(8)],
        )
        self.stage3 = nn.Sequential(
            ResidualBlock(32, batch_norm_decay, downsample=True),
            *[ResidualBlock(64, batch_norm_decay) for _ in range(8)],
        )
        self.fc = nn.Linear(64, 10)
        for layer in self.modules():
            if isinstance(layer, nn.Conv2d):
                nn.init.kaiming_normal_(
                    layer.weight, mode="fan_in", nonlinearity="relu"
                )
                nn.init.zeros_(layer.bias)
            elif isinstance(layer, CaffeBatchNorm2d):
                layer.groups = batch_norm_groups
        caffe_xavier(self.fc)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.bn(self.conv(x)))
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        return self.fc(F.avg_pool2d(x, 8).flatten(1))
