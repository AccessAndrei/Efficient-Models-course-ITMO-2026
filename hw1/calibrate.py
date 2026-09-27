import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

from equations import energy, latency

RESULTS = Path(__file__).parent / "results"
LATENCY_START = [1e-5, 8.1e12, 320e9]
ENERGY_START = [5e-12, 5e-11, 30.0]


def load_measurements() -> pd.DataFrame:
    data = pd.read_csv(RESULTS / "measurements.csv")
    data = data[data["latency"] != "OOM"]
    return data.astype({"latency": float, "memory": float, "energy": float})


def fit(predict, measured: np.ndarray, start: list[float]) -> np.ndarray:
    def residuals(log_params: np.ndarray) -> np.ndarray:
        return np.log(predict(np.exp(log_params))) - np.log(measured)

    result = least_squares(residuals, np.log(start))
    return np.exp(result.x)


def error_percent(predicted: np.ndarray, measured: np.ndarray) -> tuple[float, float]:
    errors = np.abs(predicted / measured - 1) * 100
    return errors.mean(), errors.max()


def main() -> None:
    data = load_measurements()
    base = data[~data["is_validation"]]
    validation = data[data["is_validation"]]

    s, b = base["S"].to_numpy(), base["B"].to_numpy()
    theta = fit(lambda t: latency(s, b, t), base["latency"].to_numpy(), LATENCY_START)
    theta_energy = fit(
        lambda e: energy(s, b, (*e, *theta)), base["energy"].to_numpy(), ENERGY_START
    )

    t_launch, flops_per_second, bytes_per_second = theta
    e_flop, e_byte, p_static = theta_energy
    print(f"t_launch = {t_launch * 1e6:.1f} us")
    print(f"P        = {flops_per_second / 1e12:.2f} TFLOPS")
    print(f"BW       = {bytes_per_second / 1e9:.1f} GB/s")
    print(f"e_flop   = {e_flop * 1e12:.2f} pJ")
    print(f"e_byte   = {e_byte * 1e12:.2f} pJ")
    print(f"P_static = {p_static:.1f} W")

    for name, part in [("base", base), ("validation", validation)]:
        s, b = part["S"].to_numpy(), part["B"].to_numpy()
        latency_error = error_percent(latency(s, b, theta), part["latency"].to_numpy())
        energy_error = error_percent(
            energy(s, b, (*theta_energy, *theta)), part["energy"].to_numpy()
        )
        print(
            f"{name}: latency error mean {latency_error[0]:.1f}%, max {latency_error[1]:.1f}%"
        )
        print(
            f"{name}: energy error mean {energy_error[0]:.1f}%, max {energy_error[1]:.1f}%"
        )

    result = {
        "latency": {
            "t_launch": t_launch,
            "flops_per_second": flops_per_second,
            "bytes_per_second": bytes_per_second,
        },
        "energy": {"e_flop": e_flop, "e_byte": e_byte, "p_static": p_static},
    }
    with open(RESULTS / "theta.json", "w") as file:
        json.dump(result, file, indent=2)


if __name__ == "__main__":
    main()
