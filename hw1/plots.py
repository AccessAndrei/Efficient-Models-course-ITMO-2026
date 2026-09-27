import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from equations import energy, latency, memory

RESULTS = Path(__file__).parent / "results"
FIGURES = RESULTS / "figures"
BATCH_CURVE = np.arange(1, 257)


def load() -> tuple[pd.DataFrame, tuple, tuple]:
    data = pd.read_csv(RESULTS / "measurements.csv")
    data = data[data["latency"] != "OOM"]
    data = data.astype({"latency": float, "memory": float, "energy": float})
    with open(RESULTS / "theta.json") as file:
        fitted = json.load(file)
    lat, ener = fitted["latency"], fitted["energy"]
    theta = (lat["t_launch"], lat["flops_per_second"], lat["bytes_per_second"])
    theta_energy = (ener["e_flop"], ener["e_byte"], ener["p_static"], *theta)
    return data, theta, theta_energy


def plot_vs_batch(data, column, predict, scale, ylabel, title, filename) -> None:
    sizes = sorted(data["S"].unique())
    colors = plt.cm.viridis(np.linspace(0, 1, len(sizes)))
    fig, ax = plt.subplots(figsize=(10, 7))
    for size, color in zip(sizes, colors):
        points = data[data["S"] == size]
        base = points[~points["is_validation"]]
        validation = points[points["is_validation"]]
        ax.plot(
            BATCH_CURVE,
            predict(size, BATCH_CURVE) * scale,
            color=color,
            label=f"S = {size}",
        )
        ax.scatter(base["B"], base[column] * scale, color=color, marker="o")
        ax.scatter(
            validation["B"], validation[column] * scale, color=color, marker="x", s=60
        )
    ax.plot([], [], "k-", label="prediction")
    ax.scatter([], [], color="k", marker="o", label="measured, base grid")
    ax.scatter([], [], color="k", marker="x", s=60, label="measured, validation")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("batch size B")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(FIGURES / filename, dpi=150)
    plt.close(fig)


def plot_parity(data, theta, theta_energy) -> None:
    s, b = data["S"].to_numpy(), data["B"].to_numpy()
    base = ~data["is_validation"].to_numpy()
    panels = [
        ("latency, ms", data["latency"].to_numpy() * 1e3, latency(s, b, theta) * 1e3),
        ("energy, J", data["energy"].to_numpy(), energy(s, b, theta_energy)),
        ("memory, MiB", data["memory"].to_numpy() / 2**20, memory(s, b) / 2**20),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.5))
    for ax, (unit, measured, predicted) in zip(axes, panels):
        ax.scatter(measured[base], predicted[base], label="base grid")
        ax.scatter(measured[~base], predicted[~base], marker="x", label="validation")
        low = min(measured.min(), predicted.min())
        high = max(measured.max(), predicted.max())
        ax.plot([low, high], [low, high], "k--", label="predicted = measured")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(f"measured {unit}")
        ax.set_ylabel(f"predicted {unit}")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend()
    fig.suptitle("Predicted vs measured, all 132 configurations")
    fig.tight_layout()
    fig.savefig(FIGURES / "parity.png", dpi=150)
    plt.close(fig)


def plot_error(data, column, predicted, title, filename) -> None:
    errors = data.assign(error=(predicted / data[column] - 1) * 100)
    table = errors.pivot(index="S", columns="B", values="error")
    limit = np.abs(table.to_numpy()).max()
    fig, ax = plt.subplots(figsize=(11, 7))
    image = ax.imshow(table, cmap="RdBu_r", vmin=-limit, vmax=limit, aspect="auto")
    for row in range(table.shape[0]):
        for col in range(table.shape[1]):
            ax.text(
                col,
                row,
                f"{table.iloc[row, col]:.0f}",
                ha="center",
                va="center",
                fontsize=8,
            )
    ax.set_xticks(range(table.shape[1]), table.columns)
    ax.set_yticks(range(table.shape[0]), table.index)
    ax.set_xlabel("batch size B")
    ax.set_ylabel("image size S")
    ax.set_title(title)
    fig.colorbar(image, label="(predicted / measured - 1), %")
    fig.tight_layout()
    fig.savefig(FIGURES / filename, dpi=150)
    plt.close(fig)


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    data, theta, theta_energy = load()
    s, b = data["S"].to_numpy(), data["B"].to_numpy()

    plot_vs_batch(
        data,
        "latency",
        lambda size, batch: latency(size, batch, theta),
        1e3,
        "latency, ms",
        "Latency of one forward pass",
        "latency.png",
    )
    plot_vs_batch(
        data,
        "energy",
        lambda size, batch: energy(size, batch, theta_energy),
        1,
        "energy, J",
        "Energy of one forward pass",
        "energy.png",
    )
    plot_vs_batch(
        data,
        "memory",
        memory,
        1 / 2**20,
        "peak memory, MiB",
        "Peak memory of one forward pass",
        "memory.png",
    )
    plot_parity(data, theta, theta_energy)
    plot_error(
        data, "latency", latency(s, b, theta), "Latency error, %", "latency_error.png"
    )
    plot_error(
        data,
        "energy",
        energy(s, b, theta_energy),
        "Energy error, %",
        "energy_error.png",
    )
    plot_error(data, "memory", memory(s, b), "Memory error, %", "memory_error.png")


if __name__ == "__main__":
    main()
