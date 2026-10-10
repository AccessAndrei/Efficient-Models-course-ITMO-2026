"""Run the published MNIST and CIFAR-10 comparisons, recording every evaluation."""

import argparse
import csv
import hashlib
import json
import logging
import math
import platform
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torchvision
from torch import nn

if __package__:
    from .data import DATA_DIR, load_data
    from .models import LeNet, ResNet56
    from .optim import CaffeSGD, learning_rate
else:
    from data import DATA_DIR, load_data
    from models import LeNet, ResNet56
    from optim import CaffeSGD, learning_rate

ROOT = Path(__file__).parent
EXPERIMENTS = {
    "mnist_baseline": {"steps": 10_000, "batch": 512, "eval_every": 118},
    "mnist_cycle": {"steps": 1_407, "batch": 512, "eval_every": 118},
    "mnist_short": {"steps": 1_407, "batch": 512, "eval_every": 118},
    "cifar_baseline": {"steps": 80_000, "batch": 1_000, "eval_every": 100},
    "cifar_cycle": {"steps": 10_000, "batch": 1_000, "eval_every": 100},
}
COLUMNS = (
    "step",
    "epoch",
    "lr",
    "momentum",
    "train_loss",
    "train_accuracy",
    "test_loss",
    "test_accuracy",
    "training_seconds",
    "evaluation_seconds",
    "utc",
)


def optimizer_for(model: nn.Module, dataset: str) -> CaffeSGD:
    groups = []
    decay = 0.0005 if dataset == "mnist" else 0.0001
    for layer in model.modules():
        for name, parameter in layer.named_parameters(recurse=False):
            weighted_bias = name == "bias" and isinstance(layer, (nn.Conv2d, nn.Linear))
            conv_bias = name == "bias" and isinstance(layer, nn.Conv2d)
            groups.append(
                {
                    "params": [parameter],
                    "lr_mult": 2.0 if weighted_bias else 1.0,
                    "weight_decay": 0.0 if dataset == "cifar" and conv_bias else decay,
                }
            )
    return CaffeSGD(groups)


def synchronize(device: torch.device) -> None:
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize(device)


def experiment_config(args: argparse.Namespace, device: torch.device) -> dict:
    """Fields that must stay identical when continuing a saved experiment."""
    dataset = args.experiment.split("_")[0]
    config = {
        **EXPERIMENTS[args.experiment],
        "experiment": args.experiment,
        "seed": args.seed,
        "dataset": dataset,
        "device": str(device),
        "pilot_steps": args.pilot_steps,
        "batch_norm_groups": 8 if dataset == "cifar" else 1,
        "batch_norm_decay": (0.95 if args.experiment == "cifar_cycle" else 0.999)
        if dataset == "cifar"
        else None,
        "torch_version": str(torch.__version__),
        "torchvision_version": str(torchvision.__version__),
        "numpy_version": np.__version__,
        "precision": "float32",
        "amp": False,
        "tf32": False,
        "cudnn_deterministic": True,
        "cudnn_benchmark": True,
        "source_sha256": {
            name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
            for name in ("train.py", "models.py", "optim.py", "data.py")
        },
    }
    if args.pilot_steps:
        config["steps"] = args.pilot_steps
    if device.type == "cuda":
        gpu = torch.cuda.get_device_properties(device)
        config.update(
            gpu_name=gpu.name,
            gpu_total_memory_bytes=gpu.total_memory,
            cuda_version=torch.version.cuda,
            cudnn_version=torch.backends.cudnn.version(),
        )
    return config


def atomic_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def config_hash(config: dict) -> str:
    return hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()


@torch.inference_mode()
def evaluate(
    model: nn.Module, images: torch.Tensor, labels: torch.Tensor, batch: int
) -> tuple[float, float]:
    model.eval()
    loss_sum = torch.zeros((), device=images.device)
    correct = torch.zeros((), device=images.device)
    for start in range(0, len(images), batch):
        target = labels[start : start + batch]
        logits = model(images[start : start + batch])
        loss_sum += nn.functional.cross_entropy(logits, target, reduction="sum")
        correct += (logits.argmax(1) == target).sum()
    return loss_sum.item() / len(images), correct.item() / len(images)


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: CaffeSGD,
    progress: dict,
    device: torch.device,
) -> None:
    state = {
        **progress,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "torch_rng": torch.get_rng_state(),
    }
    if device.type == "mps":
        state["device_rng"] = torch.mps.get_rng_state()
    elif device.type == "cuda":
        state["device_rng"] = torch.cuda.get_rng_state(device)
    temporary = path.with_suffix(".tmp")
    torch.save(state, temporary)
    temporary.replace(path)


def restore_checkpoint(
    state: dict, model: nn.Module, optimizer: CaffeSGD, device: torch.device
) -> None:
    model.load_state_dict(state["model"])
    optimizer.load_state_dict(state["optimizer"])
    torch.set_rng_state(state["torch_rng"].cpu())
    if device.type == "mps":
        torch.mps.set_rng_state(state["device_rng"].cpu())
    elif device.type == "cuda":
        torch.cuda.set_rng_state(state["device_rng"].cpu(), device)


def rollback_metrics(run_dir: Path, state: dict) -> None:
    """Keep committed rows; archive the original CSV before replacing it."""
    path = run_dir / "metrics.csv"
    if not path.exists():
        rows = []
    else:
        with path.open(newline="") as stream:
            rows = list(csv.DictReader(stream))
    committed = [
        row
        for row in rows
        if row.get("step", "").isdigit()
        and int(row["step"]) < state["step"]
        and all(value is not None for value in row.values())
    ]
    if state["last_metric"] is not None:
        committed.append(
            {key: str(value) for key, value in state["last_metric"].items()}
        )
    if rows == committed and path.exists():
        return
    if path.exists():
        archive = run_dir / f"metrics.before_resume_{time.time_ns()}.csv"
        shutil.copy2(path, archive)
        logging.info("Archived pre-resume metrics: %s", archive.name)
    temporary = path.with_suffix(".tmp")
    with temporary.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(committed)
    temporary.replace(path)


def write_summary(run_dir: Path, config: dict, progress: dict) -> None:
    if progress["step"] == config["steps"]:
        atomic_json(
            run_dir / "summary.json",
            {
                **progress["last_metric"],
                "experiment": config["experiment"],
                "seed": config["seed"],
                "status": "complete",
            },
        )


def train(args: argparse.Namespace) -> None:
    if args.pilot_steps < 0:
        raise ValueError("--pilot-steps must be nonnegative")
    device = torch.device(args.device)
    name = f"{args.experiment}_seed{args.seed}"
    run_dir = args.output_dir / (f"pilot_{name}" if args.pilot_steps else name)
    config_path = run_dir / "config.json"
    if config_path.exists() and not args.resume:
        raise FileExistsError(f"Run already exists: {run_dir}. Use --resume.")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    requested = experiment_config(args, device)
    state = None
    if args.resume:
        config = json.loads(config_path.read_text())
        changed = [key for key, value in requested.items() if config.get(key) != value]
        if changed:
            raise ValueError(f"Cannot resume with changed configuration: {changed}")
        state = torch.load(
            run_dir / "checkpoint.pt", map_location="cpu", weights_only=True
        )
        if state["config_sha256"] != config_hash(config):
            raise ValueError("Checkpoint and config.json do not belong to the same run")
        if state["step"] == config["steps"]:
            rollback_metrics(run_dir, state)
            if not (run_dir / "summary.json").exists():
                write_summary(run_dir, config, state)
            print(f"Already complete: {run_dir}")
            return
    run_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(message)s",
        force=True,
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(run_dir / "training.log"),
        ],
    )
    torch.manual_seed(args.seed)
    tensors = load_data(requested["dataset"], device, data_dir=args.data_dir)
    model = (
        LeNet()
        if requested["dataset"] == "mnist"
        else ResNet56(requested["batch_norm_decay"], batch_norm_groups=8)
    )
    model = model.to(device)
    optimizer = optimizer_for(model, requested["dataset"])
    if state is not None:
        restore_checkpoint(state, model, optimizer, device)
        rollback_metrics(run_dir, state)
        logging.info("RESUME from committed step %d", state["step"])
        progress = {
            key: state[key]
            for key in (
                "step",
                "training_seconds",
                "evaluation_seconds",
                "last_metric",
                "config_sha256",
            )
        }
    else:
        config = {
            **requested,
            "started_utc": datetime.now(timezone.utc).isoformat(),
            "python_version": platform.python_version(),
            "platform": platform.platform(),
            "parameters": sum(p.numel() for p in model.parameters()),
            "train_size": len(tensors[0]),
            "test_size": len(tensors[2]),
            "output_dir": str(args.output_dir),
            "data_dir": str(args.data_dir),
            "timing": "Synchronized training/evaluation; excludes data loading, logging and checkpoints.",
        }
        progress = {
            "step": 0,
            "training_seconds": 0.0,
            "evaluation_seconds": 0.0,
            "last_metric": None,
            "config_sha256": config_hash(config),
        }
        save_checkpoint(run_dir / "checkpoint.pt", model, optimizer, progress, device)
        atomic_json(config_path, config)
        rollback_metrics(run_dir, progress)
    logging.info(
        "%s seed=%d steps=%d device=%s",
        args.experiment,
        args.seed,
        config["steps"],
        device,
    )
    run_steps(model, optimizer, tensors, config, run_dir, progress)


def run_steps(
    model: nn.Module,
    optimizer: CaffeSGD,
    tensors: tuple[torch.Tensor, ...],
    config: dict,
    run_dir: Path,
    progress: dict,
) -> None:
    images, labels, test_images, test_labels = tensors
    device, batch = images.device, config["batch"]
    offsets = torch.arange(batch, device=device)
    with (run_dir / "metrics.csv").open("a", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=COLUMNS)
        loss_sum = torch.zeros((), device=device)
        correct = torch.zeros((), device=device)
        samples = 0
        model.train()
        synchronize(device)
        interval_started = time.perf_counter()
        for step in range(progress["step"], config["steps"]):
            indices = (offsets + step * batch) % len(images)
            inputs, target = images[indices], labels[indices]
            if config["dataset"] == "cifar":
                flip = torch.rand(batch, 1, 1, 1, device=device) < 0.5
                inputs = torch.where(flip, inputs.flip(-1), inputs)
            optimizer.zero_grad(set_to_none=True)
            logits = model(inputs)
            loss = nn.functional.cross_entropy(logits, target)
            loss.backward()
            lr, momentum = learning_rate(config["experiment"], step)
            optimizer.step(lr, momentum)
            loss_sum += loss.detach() * batch
            correct += (logits.detach().argmax(1) == target).sum()
            samples += batch
            completed = step + 1
            if completed % config["eval_every"] and completed != config["steps"]:
                continue
            synchronize(device)
            progress["training_seconds"] += time.perf_counter() - interval_started
            train_loss, train_accuracy = (
                loss_sum.item() / samples,
                correct.item() / samples,
            )
            if not math.isfinite(train_loss):
                raise FloatingPointError(f"Non-finite loss at step {completed}")
            evaluation_started = time.perf_counter()
            test_loss, test_accuracy = evaluate(
                model,
                test_images,
                test_labels,
                125 if config["dataset"] == "cifar" else 512,
            )
            if not math.isfinite(test_loss):
                raise FloatingPointError(f"Non-finite test loss at step {completed}")
            synchronize(device)
            progress["evaluation_seconds"] += time.perf_counter() - evaluation_started
            row = dict(
                zip(
                    COLUMNS,
                    [
                        completed,
                        completed * batch / len(images),
                        lr,
                        momentum,
                        train_loss,
                        train_accuracy,
                        test_loss,
                        test_accuracy,
                        progress["training_seconds"],
                        progress["evaluation_seconds"],
                        datetime.now(timezone.utc).isoformat(),
                    ],
                )
            )
            writer.writerow(row)
            stream.flush()
            progress.update(step=completed, last_metric=row)
            save_checkpoint(
                run_dir / "checkpoint.pt", model, optimizer, progress, device
            )
            logging.info(
                "step %d/%d test=%.2f%% loss=%.4f train_time=%.1fs",
                completed,
                config["steps"],
                100 * test_accuracy,
                train_loss,
                progress["training_seconds"],
            )
            model.train()
            loss_sum.zero_()
            correct.zero_()
            samples = 0
            synchronize(device)
            interval_started = time.perf_counter()
    write_summary(run_dir, config, progress)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment", choices=EXPERIMENTS)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("mps", "cpu", "cuda"), default="mps")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--pilot-steps", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results")
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    train(parser.parse_args())
