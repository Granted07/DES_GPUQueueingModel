#include "sim_core.h"

#include <math.h>
#include <stdlib.h>

static void ring_init(TimestampRing *ring, int capacity)
{
    ring->head = 0;
    ring->tail = 0;
    ring->size = 0;
    ring->capacity = capacity;
}

static int ring_is_empty(const TimestampRing *ring)
{
    return ring->size == 0;
}

static int ring_is_full(const TimestampRing *ring)
{
    return ring->size == ring->capacity;
}

static void ring_push(TimestampRing *ring, double timestamp)
{
    ring->values[ring->tail] = timestamp;
    ring->tail = (ring->tail + 1) % ring->capacity;
    ring->size++;
}

static double ring_pop(TimestampRing *ring)
{
    double timestamp = ring->values[ring->head];
    ring->head = (ring->head + 1) % ring->capacity;
    ring->size--;
    return timestamp;
}

static double uniform_random(void)
{
    return ((double)rand() + 1.0) / ((double)RAND_MAX + 2.0);
}

static double exponential_interarrival(double lambda)
{
    return -log(uniform_random()) / lambda;
}

static void update_queue_area(SimulationStatistics *statistics,
                              double event_time,
                              int queue_length)
{
    statistics->area_under_l_curve +=
        (event_time - statistics->last_event_time) * (double)queue_length;
    statistics->last_event_time = event_time;
}

static int batch_trigger_check(const SimulationConfig *config,
                               const TimestampRing *queue,
                               int arrivals_finished)
{
    int waiting = queue->size;
    if (waiting == 0) {
        return 0;
    }
    if (waiting < config->a) {
        return arrivals_finished ? waiting : 0;
    }
    return (waiting < config->b) ? waiting : config->b;
}

static void start_batch(const SimulationConfig *config,
                        TimestampRing *queue,
                        ActiveBatch *batch,
                        double start_time,
                        int batch_size)
{
    int index;
    batch->size = batch_size;
    batch->departure_time = start_time +
        config->service_slope * (double)batch_size +
        config->service_intercept;
    batch->in_service = 1;
    for (index = 0; index < batch_size; index++) {
        batch->arrival_times[index] = ring_pop(queue);
    }
}

static void complete_batch(ActiveBatch *batch,
                           SimulationStatistics *statistics)
{
    int index;
    for (index = 0; index < batch->size; index++) {
        statistics->sum_sojourn_time +=
            batch->departure_time - batch->arrival_times[index];
        statistics->total_served++;
    }
    batch->size = 0;
    batch->in_service = 0;
}

SimulationResults simulate(const SimulationConfig *config)
{
    TimestampRing queue;
    ActiveBatch batch = {{0.0}, 0, INFINITY, 0};
    SimulationStatistics statistics = {0, 0, 0, 0, 0.0, 0.0, 0.0, 0.0};
    double next_arrival_time;
    double current_time = 0.0;
    int generated_arrivals = 0;

    srand(config->seed);
    ring_init(&queue, config->queue_capacity);
    next_arrival_time = exponential_interarrival(config->lambda);

    while (generated_arrivals < config->arrivals_to_generate ||
           !ring_is_empty(&queue) || batch.in_service) {
        int arrivals_finished = generated_arrivals >= config->arrivals_to_generate;
        int batch_size;
        int departure_due = batch.in_service &&
                            (arrivals_finished ||
                             batch.departure_time <= next_arrival_time);

        if (!arrivals_finished && !departure_due) {
            current_time = next_arrival_time;
            generated_arrivals++;
            next_arrival_time = current_time +
                                exponential_interarrival(config->lambda);

            if (generated_arrivals > config->warmup_arrivals) {
                if (generated_arrivals == config->warmup_arrivals + 1) {
                    statistics.last_event_time = current_time;
                } else {
                    update_queue_area(&statistics, current_time, queue.size);
                }
                statistics.total_arrived++;
            }

            if (config->enable_cache_sim && config->p_hit > 0.0 &&
                uniform_random() < config->p_hit) {
                if (generated_arrivals > config->warmup_arrivals) {
                    statistics.total_cache_hits++;
                    statistics.sum_cache_hit_latency +=
                        config->cache_lookup_latency;
                }
                continue;
            }

            if (ring_is_full(&queue)) {
                if (generated_arrivals > config->warmup_arrivals) {
                    statistics.total_blocked++;
                }
            } else {
                ring_push(&queue, current_time);
                if (!batch.in_service) {
                    batch_size = batch_trigger_check(config, &queue, 0);
                    if (batch_size > 0) {
                        start_batch(config, &queue, &batch,
                                    current_time, batch_size);
                    }
                }
            }
        } else {
            current_time = batch.departure_time;
            if (generated_arrivals > config->warmup_arrivals) {
                update_queue_area(&statistics, current_time, queue.size);
                complete_batch(&batch, &statistics);
            } else {
                batch.size = 0;
                batch.in_service = 0;
            }
            batch.departure_time = INFINITY;
        }

        if (!batch.in_service) {
            arrivals_finished = generated_arrivals >= config->arrivals_to_generate;
            batch_size = batch_trigger_check(config, &queue, arrivals_finished);
            if (batch_size > 0) {
                start_batch(config, &queue, &batch, current_time, batch_size);
            }
        }
    }

    {
        SimulationResults results;
        unsigned long resolved_tasks = statistics.total_served +
                                       statistics.total_cache_hits;
        results.expected_wait_overall = resolved_tasks == 0
            ? 0.0
            : (statistics.sum_sojourn_time +
               statistics.sum_cache_hit_latency) / (double)resolved_tasks;
        results.expected_wait_miss_only = statistics.total_served == 0
            ? 0.0
            : statistics.sum_sojourn_time / (double)statistics.total_served;
        results.expected_number_in_queue =
            current_time == 0.0 ? 0.0 :
            statistics.area_under_l_curve / current_time;
        results.blocking_probability = statistics.total_arrived == 0
            ? 0.0
            : (double)statistics.total_blocked /
              (double)statistics.total_arrived;
        results.p_hit_effective = statistics.total_arrived == 0
            ? 0.0
            : (double)statistics.total_cache_hits /
              (double)statistics.total_arrived;
        results.lambda_effective = config->lambda *
                                   (1.0 - results.p_hit_effective);
        results.expected_wait_littles_law =
            (results.lambda_effective * (1.0 - results.blocking_probability)) == 0.0
            ? 0.0
            : results.expected_number_in_queue /
              (results.lambda_effective * (1.0 - results.blocking_probability));
        results.total_arrived = statistics.total_arrived;
        results.total_blocked = statistics.total_blocked;
        results.total_served = statistics.total_served;
        results.total_cache_hits = statistics.total_cache_hits;
        return results;
    }
}
