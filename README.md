# DES GPU Queueing Model

Research-oriented discrete-event simulator for GPU batch inference queues with
finite capacity, `(a,b)` batching, synthetic semantic-cache hits, warm-up
removal, and Little's Law diagnostics.

## Project layout

```text
include/       Public simulator interfaces and data types
src/           Simulation core and sweep executable
tests/         Deterministic regression tests
docs/          Research notes and experiment documentation
results/       Recommended location for generated CSV output
```

The implementation uses fixed-size storage and does not allocate memory
dynamically during simulation.

## Build

```text
cmake -S . -B build
cmake --build build
ctest --test-dir build --output-on-failure
```

Run the sweep executable from the project directory:

```text
build/DES_GPUQueingModel
```

It writes `queue_cache_results.csv` with one row for every cache probability,
batch limit, and independent seed combination.

## Reproducibility

Each repetition has an explicit seed. The simulator uses the C standard
library RNG and preserves the no-cache random stream when `p_hit == 0.0` or
cache simulation is disabled. Record the executable version, configuration
constants, compiler, and generated CSV together for each experiment.

## Metrics

The output includes overall latency, miss-only latency, mean queue length,
blocking probability, effective cache-hit probability, effective arrival
rate, and the Little's Law latency cross-check.
