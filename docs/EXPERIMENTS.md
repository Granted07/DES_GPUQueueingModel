# Experiment Protocol

The default sweep evaluates `p_hit` in `{0.0, 0.1, ..., 0.5}` and `b` in
`{4, 8, 12, 16}` with five independent seeds per combination.

Each run generates 105,000 arrivals. The first 5,000 are warm-up arrivals and
are excluded from reported counters, latency sums, blocking, cache-hit
statistics, and queue-area integration. The remaining 100,000 arrivals form
the measurement window.

The simulator reports GPU-batch metrics separately from cache-hit latency.
`E_W_littles_law` is a diagnostic computed from effective admitted arrival
rate and measured mean queue length; it should be interpreted as a
cross-check, not as a replacement for direct sojourn measurements.

For publication-quality results, preserve the source revision, compiler
version, build flags, sweep constants, seed list, and raw CSV output.
