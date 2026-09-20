#include "sim_core.h"

#include <assert.h>
#include <stdio.h>

static SimulationConfig test_config(void)
{
    SimulationConfig config = {
        0.05, 4, 8, 1.0, 4.0, MAX_QUEUE_CAPACITY, 200, 1234,
        0.0, 0.5, 1, 0
    };
    return config;
}

int main(void)
{
    SimulationConfig config = test_config();
    SimulationResults first = simulate(&config);
    SimulationResults second = simulate(&config);

    assert(first.total_arrived == 200);
    assert(first.total_cache_hits == 0);
    assert(first.expected_wait_overall == first.expected_wait_miss_only);
    assert(first.expected_wait_overall == second.expected_wait_overall);
    assert(first.expected_number_in_queue == second.expected_number_in_queue);
    assert(first.blocking_probability == second.blocking_probability);
    puts("sim_core regression tests passed");
    return 0;
}
