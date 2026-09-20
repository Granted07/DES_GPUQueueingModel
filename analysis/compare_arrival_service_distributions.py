"""
Plot the distinct stochastic roles in the queueing model:
exponential inter-arrival times and Gamma GPU service times.
"""
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gamma as gamma_distribution


GPU_INPUT = Path("googlenet_gpu_benchmark.csv")
ARRIVAL_RATE = 0.05
SERVICE_COVS = (0.0, 0.5, 1.4)
SAMPLE_COUNT = 100000


def load_service_means():
    with GPU_INPUT.open(newline="") as source:
        return list(csv.DictReader(source))


def main():
    rows = load_service_means()
    rng = np.random.default_rng(20240921)
    interarrivals = rng.exponential(1.0 / ARRIVAL_RATE, SAMPLE_COUNT)
    figure, axis = plt.subplots(figsize=(8, 5))
    axis.hist(interarrivals, bins=80, density=True, alpha=0.65)
    x_values = np.linspace(0, np.percentile(interarrivals, 99), 500)
    axis.plot(
        x_values,
        ARRIVAL_RATE * np.exp(-ARRIVAL_RATE * x_values),
        linewidth=2,
        label=f"Exponential inter-arrival, lambda={ARRIVAL_RATE}",
    )
    axis.set_title("Poisson arrivals: exponential inter-arrival times")
    axis.set_xlabel("Time between arriving requests (ms)")
    axis.set_ylabel("Density")
    axis.legend()
    figure.tight_layout()
    figure.savefig("dist_exponential_interarrival.png", dpi=200)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(9, 5))
    for row in rows:
        mean = float(row["mean_latency_ms"])
        for cov in SERVICE_COVS:
            if cov == 0.0:
                samples = np.full(SAMPLE_COUNT, mean)
                axis.axvline(mean, linewidth=2, label=f"batch {row['batch_size']}, CoV=0")
            else:
                shape = 1.0 / (cov * cov)
                scale = mean * cov * cov
                samples = rng.gamma(shape, scale, SAMPLE_COUNT)
                axis.hist(
                    samples,
                    bins=80,
                    density=True,
                    alpha=0.12,
                    label=f"batch {row['batch_size']}, CoV={cov}",
                )
    axis.set_title("GPU service time: Gamma variability by measured batch mean")
    axis.set_xlabel("GPU inference/service time (ms)")
    axis.set_ylabel("Density")
    axis.legend(fontsize=7, ncol=2)
    figure.tight_layout()
    figure.savefig("dist_gamma_service_cov.png", dpi=200)
    plt.close(figure)

    representative = float(rows[0]["mean_latency_ms"])
    figure, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].hist(interarrivals, bins=80, density=True, alpha=0.7)
    axes[0].set_title("Exponential: request inter-arrival")
    for cov in (0.5, 1.4):
        shape = 1.0 / (cov * cov)
        scale = representative * cov * cov
        axes[1].hist(
            rng.gamma(shape, scale, SAMPLE_COUNT),
            bins=80,
            density=True,
            alpha=0.5,
            label=f"Gamma service, CoV={cov}",
        )
    axes[1].axvline(representative, label="Deterministic service, CoV=0")
    axes[1].set_title("Gamma/deterministic: GPU service")
    for axis in axes:
        axis.set_xlabel("Time (ms)")
        axis.set_ylabel("Density")
        axis.legend()
    figure.suptitle("Different random variables: arrivals versus GPU service")
    figure.tight_layout()
    figure.savefig("dist_exponential_vs_gamma.png", dpi=200)
    plt.close(figure)


if __name__ == "__main__":
    main()
