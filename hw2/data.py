"""Download the standard datasets and prepare the Caffe input conventions."""

from pathlib import Path

import torch
from torchvision.datasets import CIFAR10, MNIST

DATA_DIR = Path(__file__).parent / "data"


class CIFAR10Mirror(CIFAR10):
    """Unmodified archive; torchvision verifies the original CIFAR-10 MD5."""

    url = (
        "https://huggingface.co/datasets/MIT-OL-AI-D/cifar-10-python/resolve/"
        "a48007227f9e2cd0af96175f4afb5ac4965e261b/cifar-10-python.tar.gz"
    )


def load_data(
    name: str, device: torch.device, data_dir: Path = DATA_DIR
) -> tuple[torch.Tensor, ...]:
    dataset = MNIST if name == "mnist" else CIFAR10Mirror
    train = dataset(data_dir, train=True, download=True)
    test = dataset(data_dir, train=False, download=True)
    if name == "mnist":
        train_images = train.data.unsqueeze(1).float() / 256
        test_images = test.data.unsqueeze(1).float() / 256
    else:
        train_images = torch.from_numpy(train.data).permute(0, 3, 1, 2).float()
        test_images = torch.from_numpy(test.data).permute(0, 3, 1, 2).float()
        mean_image = train_images.mean(dim=0, keepdim=True)
        train_images -= mean_image
        test_images -= mean_image
    tensors = (
        train_images.contiguous(),
        torch.as_tensor(train.targets, dtype=torch.long),
        test_images.contiguous(),
        torch.as_tensor(test.targets, dtype=torch.long),
    )
    return tuple(tensor.to(device) for tensor in tensors)
