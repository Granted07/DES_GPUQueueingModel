"""
Run the existing DES using measured GoogLeNet batch latency means.

The C executable remains the authoritative queueing implementation; this
driver creates one measured profile per batch size and invokes that executable.
"""
import csv
import json
import subprocess
import time
from pathlib import Path


PROFILE_INPUT = Path("googlenet_gpu_benchmark.csv")
DES_EXECUTABLE = Path("build/DES_GPUQueingModel")
OUTPUT_PATH = Path("googlenet_des_cov_results.csv")
METADATA_PATH = Path("googlenet_des_metadata.json")
COV_VALUES = (0.0, 0.5, 1.4)
REPETITIONS = 5
ARRIVAL_RATE = 0.05
QUEUE_CAPACITY = 1024


def main():
    if not PROFILE_INPUT.exists():
        raise FileNotFoundError(
            "Run scripts/benchmark_googlenet.py before the DES experiment."
        )
    if not DES_EXECUTABLE.exists():
        raise FileNotFoundError(
            "Build the project first; expected build/DES_GPUQueingModel."
        )

    with PROFILE_INPUT.open(newline="") as profile_file:
        profile_rows = list(csv.DictReader(profile_file))
    rows = []
    for profile in profile_rows:
        for cov in COV_VALUES:
            measurements = []
            for repetition in range(REPETITIONS):
                command = [
                    str(DES_EXECUTABLE),
                    "--profile-batch",
                    profile["batch_size"],
                    "--service-mean",
                    profile["mean_latency_ms"],
                    "--service-cov",
                    str(cov),
                    "--seed",
                    str(800000 + int(profile["batch_size"]) * 100 + int(cov * 10)
                        + repetition),
                ]
                output = subprocess.check_output(command, text=True)
                result = next(
                    line for line in output.splitlines() if line.startswith("RESULT,")
                )
                measurements.append(
                    [float(value) for value in result.split(",")[1:]]
                )
            rows.append(
                {
                    "gpu_name": profile["gpu_name"],
                    "batch_size": profile["batch_size"],
                    "cov": cov,
                    "mean_E_W": sum(row[0] for row in measurements) / REPETITIONS,
                    "mean_E_L": sum(row[1] for row in measurements) / REPETITIONS,
                    "mean_P_block": sum(row[2] for row in measurements) / REPETITIONS,
                }
            )
    with OUTPUT_PATH.open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    METADATA_PATH.write_text(
        json.dumps(
            {
                "source_measurements": str(PROFILE_INPUT),
                "arrival_rate": ARRIVAL_RATE,
                "queue_capacity": QUEUE_CAPACITY,
                "cov_values": COV_VALUES,
                "repetitions": REPETITIONS,
                "warmup_arrivals": 5000,
                "measurement_arrivals": 100000,
                "random_seed": 424242,
                "timestamp_utc": time.strftime(
                    "%Y-%m-%dT%H:%M:%SZ", time.gmtime()
                ),
            },
            indent=2,
        )
        + "\n"
    )
    print(f"Wrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
