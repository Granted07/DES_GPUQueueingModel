#include "sweep.h"

#include "sim_core.h"

#include <stdio.h>

#define NUM_REPETITIONS 5
#define NUM_ARRIVALS 100000
#define WARMUP_ARRIVALS 5000
#define ARRIVAL_RATE 0.05
#define BATCH_A 4
#define FIRST_B 4
#define LAST_B 16
#define B_STEP 4
#define CACHE_LOOKUP_LATENCY 0.5

int run_sweep(const char *filename)
{
    const double hit_probabilities[] = {0.0, 0.1, 0.2, 0.3, 0.4, 0.5};
    FILE *output = fopen(filename, "w");
    int probability_index;
    int b;
    int repetition;

    if (output == NULL) {
        perror("Unable to open results CSV");
        return 1;
    }
    fprintf(output,
            "p_hit,b,seed,E_W_overall,E_W_miss_only,E_L,P_block,"
            "p_hit_effective,lambda_effective,E_W_littles_law\n");

    for (probability_index = 0; probability_index < 6; probability_index++) {
        for (b = FIRST_B; b <= LAST_B; b += B_STEP) {
            for (repetition = 0; repetition < NUM_REPETITIONS; repetition++) {
                SimulationConfig config;
                SimulationResults results;
                config.lambda = ARRIVAL_RATE;
                config.a = BATCH_A;
                config.b = b;
                config.service_slope = 1.0;
                config.service_intercept = 4.0;
                config.queue_capacity = MAX_QUEUE_CAPACITY;
                config.arrivals_to_generate = NUM_ARRIVALS + WARMUP_ARRIVALS;
                config.warmup_arrivals = WARMUP_ARRIVALS;
                config.seed = (unsigned int)(1000 + probability_index * 100 +
                                             b * 10 + repetition);
                config.p_hit = hit_probabilities[probability_index];
                config.cache_lookup_latency = CACHE_LOOKUP_LATENCY;
                config.enable_cache_sim = 1;
                results = simulate(&config);
                fprintf(output,
                        "%.1f,%d,%u,%.10f,%.10f,%.10f,%.10f,%.10f,%.10f,%.10f\n",
                        config.p_hit, config.b, config.seed,
                        results.expected_wait_overall,
                        results.expected_wait_miss_only,
                        results.expected_number_in_queue,
                        results.blocking_probability,
                        results.p_hit_effective,
                        results.lambda_effective,
                        results.expected_wait_littles_law);
            }
        }
    }
    fclose(output);
    return 0;
}
