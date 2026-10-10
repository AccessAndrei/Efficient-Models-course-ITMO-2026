"""CPU integration tests for restartable training; no real data or GPU required."""

import argparse
import csv
import json
from pathlib import Path

import pytest
import torch
from torch import nn

from hw2 import train


@pytest.fixture
def tiny_run(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> argparse.Namespace:
    torch.set_num_threads(1)
    images = torch.arange(96, dtype=torch.float32).reshape(8, 3, 2, 2) / 96
    labels = torch.arange(8) % 2

    def synthetic_data(
        name: str, device: torch.device, data_dir: Path
    ) -> tuple[torch.Tensor, ...]:
        return images, labels, images[:4], labels[:4]

    def small_model(decay: float, batch_norm_groups: int) -> nn.Module:
        # Dropout and the CIFAR flips exercise restoration of the CPU RNG.
        return nn.Sequential(
            nn.Flatten(), nn.Linear(12, 6), nn.ReLU(), nn.Dropout(0.2), nn.Linear(6, 2)
        )

    monkeypatch.setattr(train, "load_data", synthetic_data)
    monkeypatch.setattr(train, "ResNet56", small_model)
    monkeypatch.setitem(
        train.EXPERIMENTS, "cifar_cycle", {"steps": 6, "batch": 4, "eval_every": 2}
    )
    return argparse.Namespace(
        experiment="cifar_cycle",
        seed=42,
        device="cpu",
        resume=False,
        pilot_steps=0,
        output_dir=tmp_path,
        data_dir=tmp_path / "data",
    )


def run_dir(args: argparse.Namespace) -> Path:
    return args.output_dir / "cifar_cycle_seed42"


def metric_values(path: Path) -> list[dict[str, str]]:
    with (path / "metrics.csv").open(newline="") as stream:
        return [
            {
                key: value
                for key, value in row.items()
                if key not in {"training_seconds", "evaluation_seconds", "utc"}
            }
            for row in csv.DictReader(stream)
        ]


@pytest.mark.parametrize("interruption", ["before_first_update", "before_checkpoint"])
def test_resume_matches_uninterrupted_training(
    tiny_run: argparse.Namespace,
    monkeypatch: pytest.MonkeyPatch,
    interruption: str,
) -> None:
    reference_args = argparse.Namespace(**vars(tiny_run))
    reference_args.output_dir /= "reference"
    train.train(reference_args)
    reference = torch.load(run_dir(reference_args) / "checkpoint.pt", weights_only=True)
    original_steps, original_save = train.run_steps, train.save_checkpoint

    def interrupt_steps(*args: object) -> None:
        raise RuntimeError("simulated interruption")

    def interrupt_save(
        path: Path,
        model: nn.Module,
        optimizer: train.CaffeSGD,
        progress: dict,
        device: torch.device,
    ) -> None:
        if progress["step"] == 4:
            raise RuntimeError("simulated interruption")
        original_save(path, model, optimizer, progress, device)

    if interruption == "before_first_update":
        monkeypatch.setattr(train, "run_steps", interrupt_steps)
    else:
        monkeypatch.setattr(train, "save_checkpoint", interrupt_save)
    with pytest.raises(RuntimeError, match="simulated interruption"):
        train.train(tiny_run)
    directory = run_dir(tiny_run)
    original_config = (directory / "config.json").read_bytes()
    original_metrics = (directory / "metrics.csv").read_bytes()
    saved = torch.load(directory / "checkpoint.pt", weights_only=True)
    assert saved["step"] == (0 if interruption == "before_first_update" else 2)
    assert not (directory / "summary.json").exists()

    monkeypatch.setattr(train, "run_steps", original_steps)
    monkeypatch.setattr(train, "save_checkpoint", original_save)
    tiny_run.resume = True
    tiny_run.data_dir /= "relocated"
    train.train(tiny_run)
    resumed = torch.load(directory / "checkpoint.pt", weights_only=True)
    for name, tensor in reference["model"].items():
        torch.testing.assert_close(tensor, resumed["model"][name], rtol=0, atol=0)
    for index, state in reference["optimizer"]["state"].items():
        torch.testing.assert_close(
            state["momentum_buffer"],
            resumed["optimizer"]["state"][index]["momentum_buffer"],
            rtol=0,
            atol=0,
        )
    assert metric_values(directory) == metric_values(run_dir(reference_args))
    assert [row["step"] for row in metric_values(directory)] == ["2", "4", "6"]
    assert (directory / "config.json").read_bytes() == original_config
    assert json.loads((directory / "summary.json").read_text())["status"] == "complete"
    assert "RESUME from committed step" in (directory / "training.log").read_text()
    if interruption == "before_checkpoint":
        archives = list(directory.glob("metrics.before_resume_*.csv"))
        assert len(archives) == 1
        assert archives[0].read_bytes() == original_metrics


def test_completed_resume_skips_data_and_preserves_files(
    tiny_run: argparse.Namespace,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    train.train(tiny_run)
    directory = run_dir(tiny_run)
    files = {path.name: path.read_bytes() for path in directory.iterdir()}

    def unexpected_data_load(*args: object, **kwargs: object) -> None:
        pytest.fail("Completed runs must not load data")

    monkeypatch.setattr(train, "load_data", unexpected_data_load)
    tiny_run.resume = True
    train.train(tiny_run)
    assert files == {path.name: path.read_bytes() for path in directory.iterdir()}
    (directory / "summary.json").unlink()
    train.train(tiny_run)
    assert json.loads((directory / "summary.json").read_text())["step"] == 6


@pytest.mark.parametrize("change", ["source_sha256", "steps", "device"])
def test_resume_rejects_changed_protocol_before_loading_data_or_writing_logs(
    tiny_run: argparse.Namespace,
    monkeypatch: pytest.MonkeyPatch,
    change: str,
) -> None:
    train.train(tiny_run)
    directory = run_dir(tiny_run)
    if change == "source_sha256":
        config_path = directory / "config.json"
        config = json.loads(config_path.read_text())
        config["source_sha256"]["train.py"] = "changed"
        config_path.write_text(json.dumps(config))
    elif change == "steps":
        monkeypatch.setitem(train.EXPERIMENTS["cifar_cycle"], "steps", 8)
    else:
        tiny_run.device = "mps"
    files = {path.name: path.read_bytes() for path in directory.iterdir()}

    def unexpected_data_load(*args: object, **kwargs: object) -> None:
        pytest.fail("Invalid resumes must not load data")

    monkeypatch.setattr(train, "load_data", unexpected_data_load)
    tiny_run.resume = True
    with pytest.raises(ValueError, match="changed configuration"):
        train.train(tiny_run)
    assert files == {path.name: path.read_bytes() for path in directory.iterdir()}


def test_existing_run_requires_explicit_resume_before_loading_data(
    tiny_run: argparse.Namespace,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    train.train(tiny_run)

    def unexpected_data_load(*args: object, **kwargs: object) -> None:
        pytest.fail("Existing runs must not load data without --resume")

    monkeypatch.setattr(train, "load_data", unexpected_data_load)
    with pytest.raises(FileExistsError, match="Use --resume"):
        train.train(tiny_run)


def test_nonfinite_evaluation_does_not_commit_invalid_weights(
    tiny_run: argparse.Namespace,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(train, "evaluate", lambda *args: (float("nan"), 0.0))
    with pytest.raises(FloatingPointError, match="Non-finite test loss"):
        train.train(tiny_run)
    directory = run_dir(tiny_run)
    saved = torch.load(directory / "checkpoint.pt", weights_only=True)
    assert saved["step"] == 0
    assert metric_values(directory) == []
    assert not (directory / "summary.json").exists()
