"""Summarize completed runs and regenerate the report's figures from raw logs."""

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from optim import learning_rate

EXPERIMENTS = {
    "mnist_baseline": (10_000, (42, 43, 44)),
    "mnist_cycle": (1_407, (42, 43, 44)),
    "mnist_short": (1_407, (42, 43, 44)),
    "cifar_baseline": (80_000, (42,)),
    "cifar_cycle": (10_000, (42,)),
}
COLORS = {"baseline": "#555555", "cycle": "#2878C8", "short": "#F36B32"}
LABELS = {
    "mnist_baseline": "Baseline · 85,33 эпохи",
    "mnist_cycle": "1cycle · 12,01 эпохи",
    "mnist_short": "Baseline · 12,01 эпохи",
    "cifar_baseline": "Baseline · 80 000 шагов",
    "cifar_cycle": "CLR · 10 000 шагов",
}


def mean_sd(values: list[float]) -> dict:
    """Sample standard deviation describes seed variability, not uncertainty."""
    return {
        "mean": float(np.mean(values)),
        "sample_sd": float(np.std(values, ddof=1)) if len(values) > 1 else None,
        "n": len(values),
    }


def first_observed_hit(rows: list[dict], threshold: float) -> dict | None:
    for index, row in enumerate(rows):
        if row["test_accuracy"] >= threshold:
            previous = rows[index - 1] if index else None
            return {
                "threshold_percent": 100 * threshold,
                "step": row["step"],
                "training_seconds": row["training_seconds"],
                "previous_logged_step": previous["step"] if previous else 0,
                "previous_training_seconds": (
                    previous["training_seconds"] if previous else 0
                ),
            }
    return None


def read_run(directory: Path, expected_steps: int) -> dict:
    config = json.loads((directory / "config.json").read_text())
    summary = json.loads((directory / "summary.json").read_text())
    with (directory / "metrics.csv").open(newline="") as stream:
        rows = [
            {key: value if key == "utc" else float(value) for key, value in row.items()}
            for row in csv.DictReader(stream)
        ]
    if (
        not rows
        or summary["status"] != "complete"
        or rows[-1]["step"] != expected_steps
    ):
        raise ValueError(f"Incomplete run: {directory}")
    final = rows[-1]
    best = max(rows, key=lambda row: row["test_accuracy"])
    wall = datetime.fromisoformat(final["utc"]) - datetime.fromisoformat(
        config["started_utc"]
    )
    threshold = 0.9 if config["dataset"] == "cifar" else 0.99
    return {
        "name": directory.name,
        "config": config,
        "rows": rows,
        "statistics": {
            "experiment": config["experiment"],
            "seed": config["seed"],
            "steps": int(final["step"]),
            "epochs": final["epoch"],
            "final_accuracy_percent": 100 * final["test_accuracy"],
            "best_accuracy_percent": 100 * best["test_accuracy"],
            "best_step": int(best["step"]),
            "training_seconds": final["training_seconds"],
            "evaluation_seconds": final["evaluation_seconds"],
            "wall_logged_seconds": wall.total_seconds(),
            "first_observed_threshold": first_observed_hit(rows, threshold),
        },
    }


def aggregate(runs: list[dict]) -> dict:
    groups = {}
    for experiment in EXPERIMENTS:
        selected = [
            run["statistics"]
            for run in runs
            if run["config"]["experiment"] == experiment
        ]
        groups[experiment] = {
            key: mean_sd([run[key] for run in selected])
            for key in (
                "final_accuracy_percent",
                "best_accuracy_percent",
                "training_seconds",
                "evaluation_seconds",
                "wall_logged_seconds",
            )
        }
    return groups


def write_tables(results: Path, runs: list[dict], groups: dict) -> None:
    by_name = {run["name"]: run for run in runs}
    baseline = by_name["cifar_baseline_seed42"]
    cycle = by_name["cifar_cycle_seed42"]
    matched = next(row for row in baseline["rows"] if row["step"] == 10_000)
    speedups = {}
    for dataset in ("mnist", "cifar"):
        speedups[dataset] = {
            metric: groups[f"{dataset}_baseline"][metric]["mean"]
            / groups[f"{dataset}_cycle"][metric]["mean"]
            for metric in ("training_seconds", "wall_logged_seconds")
        }
    analysis = {
        "notes": {
            "accuracy": "Percent; final checkpoint is primary. Best is a separate diagnostic, not model selection.",
            "spread": "Sample standard deviation across three MNIST seeds (ddof=1); CIFAR n=1, no SD.",
            "timing": "Synchronized training/evaluation excludes data loading, logging and checkpoints. Wall logged: config start to final evaluation timestamp, before final checkpoint.",
            "speedups": "Baseline-to-cycle time ratios for the full configured training budgets, not time to the same accuracy.",
            "threshold": "First logged accuracy at or above threshold; no interpolation. Earlier unobserved crossings between evaluations are possible.",
            "curves": "All original logged points are included; connecting segments are visual guides, not additional observations.",
        },
        "runs": [run["statistics"] for run in runs],
        "groups": groups,
        "speedups": speedups,
        "cifar_equal_budget": {
            "steps": 10_000,
            "baseline_accuracy_percent": 100 * matched["test_accuracy"],
            "cycle_accuracy_percent": cycle["statistics"]["final_accuracy_percent"],
        },
        "paper_reference": {
            "mnist_baseline": {"accuracy_percent": 99.03, "reported_plus_minus": 0.04},
            "mnist_cycle": {"accuracy_percent": 99.25, "reported_plus_minus": 0.03},
            "cifar_baseline": {"accuracy_percent": 91.2, "statistic": "reported_peak"},
            "cifar_cycle": {"accuracy_percent": 92.4, "statistic": "reported_final"},
        },
        "curves": {run["name"]: run["rows"] for run in runs},
    }
    (results / "analysis.json").write_text(json.dumps(analysis, indent=2) + "\n")
    columns = [
        key for key in runs[0]["statistics"] if key != "first_observed_threshold"
    ]
    with (results / "summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(
            {key: run["statistics"][key] for key in columns} for run in runs
        )


def style_axis(axis: Axes, xlabel: str, ylabel: str, title: str = "") -> None:
    axis.set(xlabel=xlabel, ylabel=ylabel, title=title)
    axis.spines[["top", "right"]].set_visible(False)
    axis.grid(color="#E0DED8", linewidth=0.7)
    axis.set_axisbelow(True)


def save_figure(figure: Figure, output: Path, name: str) -> None:
    figure.savefig(output / f"{name}.png", dpi=190, facecolor="white")
    figure.savefig(output / f"{name}.pdf", facecolor="white")
    plt.close(figure)


def plot_schedules(output: Path) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(11, 6.5), layout="constrained")
    for column, dataset in enumerate(("mnist", "cifar")):
        for regime in ("baseline", "cycle"):
            experiment = f"{dataset}_{regime}"
            steps = np.arange(EXPERIMENTS[experiment][0])
            rates = np.array([learning_rate(experiment, int(step)) for step in steps])
            x = steps * 512 / 60_000 if dataset == "mnist" else steps / 1_000
            for row in range(2):
                axes[row, column].plot(
                    x, rates[:, row], color=COLORS[regime], label=LABELS[experiment]
                )
        xlabel = (
            "Эпохи (перед обновлением)"
            if dataset == "mnist"
            else "Тысячи шагов (перед обновлением)"
        )
        style_axis(
            axes[0, column],
            xlabel,
            "Learning rate",
            dataset.upper().replace("CIFAR", "CIFAR-10"),
        )
        style_axis(axes[1, column], xlabel, "Momentum")
        axes[0, column].legend(frameon=False, fontsize=9)
        axes[1, column].set_ylim(0.78, 0.975)
    axes[0, 0].set_xlim(0, 13)
    axes[1, 0].set_xlim(0, 13)
    axes[0, 0].set_title("MNIST · первые 13 эпох")
    axes[0, 1].set_xlim(0, 80)
    axes[1, 1].set_xlim(0, 80)
    save_figure(figure, output, "schedules")


def plot_mnist(runs: list[dict], output: Path) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.5), layout="constrained")
    for experiment in ("mnist_baseline", "mnist_cycle", "mnist_short"):
        selected = [run for run in runs if run["config"]["experiment"] == experiment]
        x = np.array([row["epoch"] for row in selected[0]["rows"]])
        y = 100 * np.array(
            [[row["test_accuracy"] for row in run["rows"]] for run in selected]
        )
        color = COLORS[experiment.split("_")[1]]
        for axis in axes:
            axis.plot(
                x,
                y.mean(axis=0),
                label=LABELS[experiment],
                color=color,
                linewidth=2,
                linestyle="--" if experiment == "mnist_short" else "-",
            )
            axis.fill_between(
                x,
                y.mean(axis=0) - y.std(axis=0, ddof=1),
                y.mean(axis=0) + y.std(axis=0, ddof=1),
                color=color,
                alpha=0.14,
                linewidth=0,
            )
    for axis in axes:
        style_axis(axis, "Эпохи", "Точность на тесте, %")
        axis.legend(frameon=False, fontsize=9, loc="lower right")
    axes[0].set(xlim=(0, 86), ylim=(94, 99.4), title="Полный бюджет")
    axes[1].set(
        xlim=(0, 12.5), ylim=(94, 99.4), title="Первые 12 эпох · среднее ± SD, n = 3"
    )
    save_figure(figure, output, "mnist_accuracy")


def plot_cifar(runs: list[dict], output: Path, time_axis: bool = False) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.5), layout="constrained")
    for run in runs:
        experiment = run["config"]["experiment"]
        if not experiment.startswith("cifar_"):
            continue
        key, divisor = ("training_seconds", 60) if time_axis else ("step", 1_000)
        x = [row[key] / divisor for row in run["rows"]]
        y = [100 * row["test_accuracy"] for row in run["rows"]]
        for axis in axes:
            axis.plot(
                x,
                y,
                label=LABELS[experiment],
                color=COLORS[experiment.split("_")[1]],
                linewidth=1.8,
            )
    xlabel = "Чистое время обучения, мин" if time_axis else "Тысячи шагов"
    for axis in axes:
        style_axis(axis, xlabel, "Точность на тесте, %")
        axis.legend(frameon=False, fontsize=9, loc="lower right")
    axes[0].set(
        xlim=(0, 103 if time_axis else 80),
        ylim=(0, 100),
        title="Весь запуск · по одному seed",
    )
    axes[1].set(
        xlim=(0, 13 if time_axis else 10),
        ylim=(0, 100),
        title="Первые 13 минут" if time_axis else "Первые 10 000 шагов",
    )
    save_figure(figure, output, "cifar_time" if time_axis else "cifar_accuracy")


def plot_training_time(groups: dict, output: Path) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")
    for axis, dataset in zip(axes, ("mnist", "cifar")):
        experiments = [f"{dataset}_baseline", f"{dataset}_cycle"]
        divisor = 60 if dataset == "cifar" else 1
        values = [
            groups[name]["training_seconds"]["mean"] / divisor for name in experiments
        ]
        labels = ["Baseline", "CLR" if dataset == "cifar" else "1cycle"]
        bars = axis.bar(
            labels, values, color=[COLORS["baseline"], COLORS["cycle"]], width=0.6
        )
        axis.bar_label(bars, labels=[f"{value:.2f}" for value in values], padding=5)
        speedup = values[0] / values[1]
        style_axis(
            axis,
            "",
            "Минуты" if dataset == "cifar" else "Секунды",
            f"{dataset.upper().replace('CIFAR', 'CIFAR-10')} · отношение времени {speedup:.2f}×",
        )
        axis.set_ylim(0, max(values) * 1.2)
    figure.supxlabel(
        "Полные бюджеты; MNIST — среднее по 3 seed, CIFAR-10 — 1 seed.\n"
        "Сравнение длительности запусков, а не времени до одинаковой точности.",
        fontsize=9,
    )
    save_figure(figure, output, "training_time")


def plot_comparison(groups: dict, output: Path) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.5), layout="constrained")
    for position, experiment in enumerate(
        ("mnist_baseline", "mnist_cycle", "mnist_short")
    ):
        stats = groups[experiment]["final_accuracy_percent"]
        axes[0].errorbar(
            position,
            stats["mean"],
            yerr=stats["sample_sd"],
            fmt="o",
            capsize=5,
            markersize=8,
            color=COLORS[experiment.split("_")[1]],
        )
        axes[0].annotate(
            f"{stats['mean']:.2f} ± {stats['sample_sd']:.2f}",
            (position, stats["mean"]),
            xytext=(0, -25),
            textcoords="offset points",
            ha="center",
            fontsize=10,
        )
    axes[0].set(
        xticks=[0, 1, 2],
        xticklabels=[
            "Baseline\n85,33 эпохи",
            "1cycle\n12,01 эпохи",
            "Baseline\n12,01 эпохи",
        ],
        xlim=(-0.55, 2.55),
        ylim=(98.45, 99.4),
    )
    style_axis(
        axes[0], "", "Точность на тесте, %", "MNIST · финал, среднее ± SD, n = 3"
    )
    names = ("cifar_baseline", "cifar_cycle")
    for position, name in enumerate(names):
        final = groups[name]["final_accuracy_percent"]["mean"]
        axes[1].scatter(position, final, color=COLORS[name.split("_")[1]], s=65)
        axes[1].annotate(
            f"{final:.2f}%",
            (position, final),
            xytext=(0, -22),
            textcoords="offset points",
            ha="center",
        )
    axes[1].set(
        xticks=[0, 1],
        xticklabels=["Baseline\n80 000 шагов", "CLR\n10 000 шагов"],
        xlim=(-0.6, 1.6),
        ylim=(90, 92.5),
    )
    style_axis(axes[1], "", "Точность на тесте, %", "CIFAR-10 · финал, n = 1")
    save_figure(figure, output, "final_accuracy")


def main(results: Path) -> None:
    runs = [
        read_run(results / f"{experiment}_seed{seed}", steps)
        for experiment, (steps, seeds) in EXPERIMENTS.items()
        for seed in seeds
    ]
    groups = aggregate(runs)
    write_tables(results, runs, groups)
    output = results / "figures"
    output.mkdir(exist_ok=True)
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.titlepad": 12,
            "text.color": "#111111",
            "axes.labelcolor": "#555555",
            "axes.edgecolor": "#B8B6AF",
            "xtick.color": "#555555",
            "ytick.color": "#555555",
            "savefig.bbox": "tight",
            "pdf.fonttype": 42,
        }
    )
    plot_schedules(output)
    plot_mnist(runs, output)
    plot_cifar(runs, output)
    plot_cifar(runs, output, time_axis=True)
    plot_training_time(groups, output)
    plot_comparison(groups, output)
    print(f"Summarized {len(runs)} completed runs: {results / 'analysis.json'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results", type=Path, default=Path(__file__).parent / "results"
    )
    main(parser.parse_args().results)
