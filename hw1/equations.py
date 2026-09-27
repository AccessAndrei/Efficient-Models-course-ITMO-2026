import numpy as np
from numpy.typing import ArrayLike

BYTES_PER_NUMBER = 4
WEIGHT_BYTES = 4_181_312
ACTIVATIONS_PER_S2 = 47
ACTIVATIONS_CONST = 868

# name, FLOPs = flops_s2 * B * S^2 + flops_b * B, Bytes = bytes_s2 * B * S^2 + bytes_b * B + bytes_const
LAYERS = [
    ("conv1", 2352, 0, 44, 0, 18_816),
    ("bn1", 16, 0, 64, 0, 512),
    ("relu1", 8, 0, 64, 0, 0),
    ("maxpool", 16, 0, 40, 0, 0),
    ("conv2", 6400, 0, 24, 0, 204_800),
    ("bn2", 8, 0, 32, 0, 1_024),
    ("relu2", 4, 0, 32, 0, 0),
    ("conv3", 2304, 0, 24, 0, 294_912),
    ("bn3", 4, 0, 16, 0, 2_048),
    ("relu3", 2, 0, 16, 0, 0),
    ("conv4", 1024, 0, 24, 0, 131_072),
    ("bn4", 8, 0, 32, 0, 4_096),
    ("relu4", 4, 0, 32, 0, 0),
    ("conv5", 4608, 0, 20, 0, 2_359_296),
    ("bn5", 2, 0, 8, 0, 4_096),
    ("relu5", 1, 0, 8, 0, 0),
    ("conv6", 1024, 0, 12, 0, 524_288),
    ("bn6", 4, 0, 16, 0, 8_192),
    ("relu6", 2, 0, 16, 0, 0),
    ("avgpool", 2, 0, 8, 2_048, 0),
    ("fc1", 0, 262_144, 0, 3_072, 525_312),
    ("relu7", 0, 256, 0, 2_048, 0),
    ("fc2", 0, 51_200, 0, 1_424, 102_800),
]


def to_arrays(image_size: ArrayLike, batch: ArrayLike) -> tuple[np.ndarray, np.ndarray]:
    return np.asarray(image_size, dtype=float), np.asarray(batch, dtype=float)


def layer_flops(layer: tuple, s: np.ndarray, b: np.ndarray) -> np.ndarray:
    _, flops_s2, flops_b, _, _, _ = layer
    return flops_s2 * b * s**2 + flops_b * b


def layer_bytes(layer: tuple, s: np.ndarray, b: np.ndarray) -> np.ndarray:
    _, _, _, bytes_s2, bytes_b, bytes_const = layer
    return bytes_s2 * b * s**2 + bytes_b * b + bytes_const


def flops(image_size: ArrayLike, batch: ArrayLike) -> np.ndarray:
    s, b = to_arrays(image_size, batch)
    return sum(layer_flops(layer, s, b) for layer in LAYERS)


def bytes_moved(image_size: ArrayLike, batch: ArrayLike) -> np.ndarray:
    s, b = to_arrays(image_size, batch)
    return sum(layer_bytes(layer, s, b) for layer in LAYERS)


def memory(image_size: ArrayLike, batch: ArrayLike) -> np.ndarray:
    s, b = to_arrays(image_size, batch)
    activations = ACTIVATIONS_PER_S2 * s**2 + ACTIVATIONS_CONST
    return WEIGHT_BYTES + BYTES_PER_NUMBER * b * activations


def latency(image_size: ArrayLike, batch: ArrayLike, theta: tuple) -> np.ndarray:
    t_launch, flops_per_second, bytes_per_second = theta
    s, b = to_arrays(image_size, batch)
    total = 0.0
    for layer in LAYERS:
        compute_time = layer_flops(layer, s, b) / flops_per_second
        memory_time = layer_bytes(layer, s, b) / bytes_per_second
        total = total + t_launch + np.maximum(compute_time, memory_time)
    return total


def energy(image_size: ArrayLike, batch: ArrayLike, theta_energy: tuple) -> np.ndarray:
    e_flop, e_byte, p_static, t_launch, flops_per_second, bytes_per_second = (
        theta_energy
    )
    theta = (t_launch, flops_per_second, bytes_per_second)
    return (
        e_flop * flops(image_size, batch)
        + e_byte * bytes_moved(image_size, batch)
        + p_static * latency(image_size, batch, theta)
    )


if __name__ == "__main__":
    print("FLOPs ", flops(224, 1))
    print("Bytes ", bytes_moved(224, 1))
    print("Memory", memory(224, 1))
