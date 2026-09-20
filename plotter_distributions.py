"""
Plot distribution diagnostics and profile-driven GPU CoV benchmark results.

The script is intentionally tolerant of missing CSV inputs so distribution
and benchmark plots can be generated independently.
"""

import math
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DISTRIBUTION_INPUT = "distribution_samples.csv"
BENCHMARK_INPUT = "gpu_benchmark_cov_comparison.csv"
BENCHMARK_BATCH_SIZE = None  # Set an integer to override the maximum observed size.


def load_distribution_samples():
    """Load raw distribution samples, or return None when unavailable."""
    try:
        samples = pd.read_csv(DISTRIBUTION_INPUT)
        required = {"distribution", "parameter_label", "sample_value"}
        missing = required.difference(samples.columns)
        if missing:
            print(
                f"{DISTRIBUTION_INPUT} is missing columns {sorted(missing)}; "
                "distribution plots will be skipped."
            )
            return None
        return samples
    except FileNotFoundError:
        print(
            f"{DISTRIBUTION_INPUT} was not found; skipping exponential, gamma, "
            "and gamma-vs-exponential distribution plots."
        )
    except (OSError, pd.errors.ParserError, ValueError) as error:
        print(
            f"Could not read {DISTRIBUTION_INPUT}: {error}; skipping all "
            "distribution plots."
        )
    return None


def load_benchmark_results():
    """Load GPU CoV benchmark results, or return None when unavailable."""
    try:
        results = pd.read_csv(BENCHMARK_INPUT)
        required = {
            "gpu_name",
            "batch_size",
            "cov",
            "mean_E_W",
            "mean_E_L",
            "mean_P_block",
        }
        missing = required.difference(results.columns)
        if missing:
            print(
                f"{BENCHMARK_INPUT} is missing columns {sorted(missing)}; "
                "GPU benchmark plots will be skipped."
            )
            return None
        return results
    except FileNotFoundError:
        print(
            f"{BENCHMARK_INPUT} was not found; skipping GPU CoV latency, "
            "blocking, and grouped-bar plots."
        )
    except (OSError, pd.errors.ParserError, ValueError) as error:
        print(
            f"Could not read {BENCHMARK_INPUT}: {error}; skipping all "
            f"GPU benchmark plots."
        )
    return None


def parse_exponential_lambda(parameter_label):
    """Extract lambda from labels such as lambda=0.05."""
    match = re.search(r"lambda\s*=\s*([-+0-9.eE]+)", parameter_label)
    if not match:
        raise ValueError(f"Could not parse exponential label: {parameter_label}")
    return float(match.group(1))


def parse_gamma_parameters(parameter_label):
    """Extract mean and CoV from labels such as mean=20.0_cov=1.4."""
    mean_match = re.search(r"mean\s*=\s*([-+0-9.eE]+)", parameter_label)
    cov_match = re.search(r"cov\s*=\s*([-+0-9.eE]+)", parameter_label)
    if not mean_match or not cov_match:
        raise ValueError(f"Could not parse gamma label: {parameter_label}")
    return float(mean_match.group(1)), float(cov_match.group(1))


def plot_distribution_samples(samples):
    """Create the three requested distribution comparison figures."""
    exponential = samples[samples["distribution"].str.lower() == "exponential"]
    gamma = samples[samples["distribution"].str.lower() == "gamma"]

    if not exponential.empty:
        figure, axis = plt.subplots()
        for label, group in exponential.groupby("parameter_label"):
            values = group["sample_value"].dropna().to_numpy()
            if values.size == 0:
                continue
            axis.hist(values, bins=60, density=True, alpha=0.5, label=label)
            try:
                rate = parse_exponential_lambda(label)
                x_values = np.linspace(0.0, np.percentile(values, 99), 400)
                axis.plot(
                    x_values,
                    rate * np.exp(-rate * x_values),
                    linewidth=2,
                    label=f"{label} theoretical",
                )
            except ValueError as error:
                print(f"Skipping theoretical exponential overlay: {error}")
        axis.set_title("Exponential distribution: sampled vs theoretical")
        axis.set_xlabel("Sample value")
        axis.set_ylabel("Density")
        axis.legend()
        figure.tight_layout()
        figure.savefig("dist_exponential_check.png", dpi=150)
        plt.close(figure)
    else:
        print("No exponential rows found; skipping dist_exponential_check.png.")

    if not gamma.empty:
        figure, axis = plt.subplots()
        scipy_gamma = None
        try:
            from scipy.stats import gamma as scipy_gamma
        except ImportError:
            print(
                "SciPy is not installed; run `pip install scipy` to enable "
                "the theoretical Gamma overlays. Raw Gamma histograms will "
                "still be plotted."
            )

        for label, group in gamma.groupby("parameter_label"):
            values = group["sample_value"].dropna().to_numpy()
            if values.size == 0:
                continue
            axis.hist(values, bins=60, density=True, alpha=0.5, label=label)
            if scipy_gamma is not None:
                try:
                    mean, cov = parse_gamma_parameters(label)
                    # Must match gamma_random_from_mean_cov exactly:
                    # shape=1/cov^2 and scale=mean*cov^2.
                    shape = 1.0 / (cov * cov)
                    scale = mean * cov * cov
                    x_values = np.linspace(0.0, np.percentile(values, 99), 400)
                    axis.plot(
                        x_values,
                        scipy_gamma.pdf(x_values, a=shape, scale=scale),
                        linewidth=2,
                        label=f"{label} theoretical",
                    )
                except ValueError as error:
                    print(f"Skipping theoretical Gamma overlay: {error}")
        axis.set_title("Gamma distribution: sampled vs theoretical, by CoV")
        axis.set_xlabel("Sample value")
        axis.set_ylabel("Density")
        axis.legend()
        figure.tight_layout()
        figure.savefig("dist_gamma_check.png", dpi=150)
        plt.close(figure)
    else:
        print("No gamma rows found; skipping dist_gamma_check.png.")

    gamma_labels = list(gamma["parameter_label"].dropna().unique())
    figure, axes = plt.subplots(1, 3, figsize=(15, 4), squeeze=False)
    axes = axes[0]
    for index, label in enumerate(gamma_labels[:3]):
        axis = axes[index]
        gamma_group = gamma[gamma["parameter_label"] == label]["sample_value"].dropna()
        mean, _ = parse_gamma_parameters(label)
        matching_exponential = None
        for exponential_label, group in exponential.groupby("parameter_label"):
            try:
                rate = parse_exponential_lambda(exponential_label)
            except ValueError:
                continue
            if math.isclose(1.0 / rate, mean, rel_tol=1e-9, abs_tol=1e-9):
                matching_exponential = group["sample_value"].dropna()
                break
        if matching_exponential is None or gamma_group.empty:
            axis.text(
                0.5,
                0.5,
                "No matching comparison group was found",
                ha="center",
                va="center",
                transform=axis.transAxes,
            )
            axis.set_xticks([])
            axis.set_yticks([])
        else:
            axis.hist(
                gamma_group,
                bins=60,
                density=True,
                alpha=0.5,
                label="Gamma",
            )
            axis.hist(
                matching_exponential,
                bins=60,
                density=True,
                alpha=0.5,
                label="Exponential",
            )
            axis.legend()
        axis.set_title(label)
        axis.set_xlabel("Sample value")
        axis.set_ylabel("Density")
    for index in range(len(gamma_labels[:3]), 3):
        axes[index].axis("off")
    figure.suptitle("Gamma vs exponential shape comparison at matched means")
    figure.tight_layout()
    figure.savefig("dist_gamma_vs_exponential.png", dpi=150)
    plt.close(figure)


def gpu_subplot_grid(gpu_names):
    """Return a figure and axes arranged using a near-square subplot grid."""
    count = len(gpu_names)
    columns = min(3, max(1, math.ceil(math.sqrt(count))))
    rows = math.ceil(count / columns)
    figure, axes = plt.subplots(
        rows,
        columns,
        figsize=(5 * columns, 4 * rows),
        squeeze=False,
        sharex=True,
    )
    return figure, axes, rows, columns


def plot_gpu_lines(results, value_column, title, output_name, y_label):
    """Plot one CoV line chart per GPU for the selected result column."""
    gpu_names = list(results["gpu_name"].dropna().unique())
    figure, axes, rows, columns = gpu_subplot_grid(gpu_names)
    for index, gpu_name in enumerate(gpu_names):
        axis = axes[index // columns][index % columns]
        gpu_results = results[results["gpu_name"] == gpu_name]
        for cov, cov_results in gpu_results.groupby("cov"):
            cov_results = cov_results.sort_values("batch_size")
            label = f"CoV={cov:g} (deterministic)" if cov == 0 else f"CoV={cov:g}"
            axis.plot(
                cov_results["batch_size"],
                cov_results[value_column],
                marker="o",
                label=label,
            )
        axis.set_title(gpu_name)
        axis.set_xlabel("Batch size")
        axis.set_ylabel(y_label)
        axis.legend()
        axis.grid(alpha=0.25)
    for index in range(len(gpu_names), rows * columns):
        axes[index // columns][index % columns].axis("off")
    figure.suptitle(title)
    figure.tight_layout()
    figure.savefig(output_name, dpi=150)
    plt.close(figure)


def plot_gpu_benchmark(results):
    """Create the latency, blocking, and fixed-batch grouped-bar plots."""
    plot_gpu_lines(
        results,
        "mean_E_W",
        "Effect of service-time variability (CoV) on latency across GPUs",
        "cov_latency_by_gpu.png",
        "Expected wait time E[W] (ms)",
    )
    plot_gpu_lines(
        results,
        "mean_P_block",
        "Effect of CoV on blocking probability across GPUs",
        "cov_blocking_by_gpu.png",
        "Blocking probability",
    )

    batch_size = (
        BENCHMARK_BATCH_SIZE
        if BENCHMARK_BATCH_SIZE is not None
        else results["batch_size"].max()
    )
    selected = results[results["batch_size"] == batch_size]
    gpu_names = list(results["gpu_name"].dropna().unique())
    cov_values = sorted(results["cov"].dropna().unique())
    x_values = np.arange(len(gpu_names))
    width = 0.8 / max(1, len(cov_values))
    figure, axis = plt.subplots(figsize=(10, 6))
    for index, cov in enumerate(cov_values):
        values = []
        for gpu_name in gpu_names:
            matches = selected[
                (selected["gpu_name"] == gpu_name) & (selected["cov"] == cov)
            ]["mean_E_W"]
            values.append(matches.iloc[0] if not matches.empty else np.nan)
        label = f"CoV={cov:g} (deterministic)" if cov == 0 else f"CoV={cov:g}"
        axis.bar(x_values + (index - (len(cov_values) - 1) / 2) * width,
                 values, width, label=label)
    axis.set_xticks(x_values)
    axis.set_xticklabels(gpu_names, rotation=20, ha="right")
    axis.set_xlabel("GPU")
    axis.set_ylabel("Expected wait time E[W] (ms)")
    axis.set_title(
        f"Latency comparison across GPUs at CoV variants "
        f"(batch size = {batch_size})"
    )
    axis.legend()
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig("cov_latency_grouped_by_gpu.png", dpi=150)
    plt.close(figure)


def main():
    """Load available inputs and generate all applicable plots."""
    samples = load_distribution_samples()
    if samples is not None:
        plot_distribution_samples(samples)

    benchmark_results = load_benchmark_results()
    if benchmark_results is not None:
        plot_gpu_benchmark(benchmark_results)


if __name__ == "__main__":
    main()
