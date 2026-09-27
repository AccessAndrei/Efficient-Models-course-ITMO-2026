import csv
import random
import statistics
import time
from pathlib import Path

import pynvml
import torch

from models import Net

BASE_SIZES = [32, 64, 128, 224, 256, 384, 512]
BASE_BATCHES = [1, 2, 4, 8, 16, 32, 64, 128, 256]
SEED = 0
WARMUP_RUNS = 5
TIMED_RUNS = 20
ENERGY_SECONDS = 1.0
OUTPUT = Path(__file__).parent / "results" / "measurements.csv"


def sample_extra(seed: int) -> tuple[list[int], list[int]]:
    rng = random.Random(seed)
    sizes = [s for s in range(32, 513, 16) if s not in BASE_SIZES]
    batches = [b for b in range(1, 257) if b not in BASE_BATCHES]
    return sorted(rng.sample(sizes, 4)), sorted(rng.sample(batches, 3))


def measure_latency(model: torch.nn.Module, x: torch.Tensor) -> float:
    times = []
    for _ in range(TIMED_RUNS):
        torch.cuda.synchronize()
        start = time.perf_counter()
        model(x)
        torch.cuda.synchronize()
        times.append(time.perf_counter() - start)
    return statistics.median(times)


def measure_memory(model: torch.nn.Module, x: torch.Tensor) -> int:
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    model(x)
    torch.cuda.synchronize()
    return torch.cuda.max_memory_allocated()


def measure_energy(model: torch.nn.Module, x: torch.Tensor, gpu: object) -> float:
    torch.cuda.synchronize()
    start_energy = pynvml.nvmlDeviceGetTotalEnergyConsumption(gpu)
    start_time = time.perf_counter()
    runs = 0
    while time.perf_counter() - start_time < ENERGY_SECONDS:
        model(x)
        torch.cuda.synchronize()
        runs += 1
    end_energy = pynvml.nvmlDeviceGetTotalEnergyConsumption(gpu)
    return (end_energy - start_energy) / 1000 / runs


def measure_config(model: torch.nn.Module, size: int, batch: int, gpu: object) -> list:
    torch.cuda.empty_cache()
    try:
        x = torch.randn(batch, 3, size, size, device="cuda")
        with torch.inference_mode():
            for _ in range(WARMUP_RUNS):
                model(x)
            latency = measure_latency(model, x)
            memory = measure_memory(model, x)
            energy = measure_energy(model, x, gpu)
        return [latency, memory, energy]
    except torch.cuda.OutOfMemoryError:
        return ["OOM", "OOM", "OOM"]


def print_environment() -> None:
    print("GPU:", torch.cuda.get_device_name())
    print("driver:", pynvml.nvmlSystemGetDriverVersion())
    print("torch:", torch.__version__)
    print("CUDA:", torch.version.cuda)
    print("cuDNN:", torch.backends.cudnn.version())


def main() -> None:
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    model = Net().cuda().eval()

    pynvml.nvmlInit()
    gpu = pynvml.nvmlDeviceGetHandleByIndex(0)
    print_environment()

    extra_sizes, extra_batches = sample_extra(SEED)
    sizes = sorted(BASE_SIZES + extra_sizes)
    batches = sorted(BASE_BATCHES + extra_batches)

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["S", "B", "latency", "memory", "energy", "is_validation"])
        for size in sizes:
            for batch in batches:
                is_validation = size in extra_sizes or batch in extra_batches
                row = [
                    size,
                    batch,
                    *measure_config(model, size, batch, gpu),
                    is_validation,
                ]
                writer.writerow(row)
                file.flush()
                print(row)

    pynvml.nvmlShutdown()


if __name__ == "__main__":
    main()
