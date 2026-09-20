#ifndef SIM_TYPES_H
#define SIM_TYPES_H

#include <stddef.h>

#define MAX_QUEUE_CAPACITY 1024
#define MAX_BATCH_SIZE MAX_QUEUE_CAPACITY

typedef struct {
    double lambda;
    int a;
    int b;
    double service_slope;
    double service_intercept;
    int queue_capacity;
    int arrivals_to_generate;
    unsigned int seed;
    double p_hit;
    double cache_lookup_latency;
    int enable_cache_sim;
    int warmup_arrivals;
} SimulationConfig;

typedef struct {
    double expected_wait_overall;
    double expected_wait_miss_only;
    double expected_number_in_queue;
    double blocking_probability;
    double p_hit_effective;
    double lambda_effective;
    double expected_wait_littles_law;
    unsigned long total_arrived;
    unsigned long total_blocked;
    unsigned long total_served;
    unsigned long total_cache_hits;
} SimulationResults;

typedef struct {
    double values[MAX_QUEUE_CAPACITY];
    int head;
    int tail;
    int size;
    int capacity;
} TimestampRing;

typedef struct {
    unsigned long total_arrived;
    unsigned long total_blocked;
    unsigned long total_served;
    unsigned long total_cache_hits;
    double sum_sojourn_time;
    double sum_cache_hit_latency;
    double area_under_l_curve;
    double last_event_time;
} SimulationStatistics;

typedef struct {
    double arrival_times[MAX_BATCH_SIZE];
    int size;
    double departure_time;
    int in_service;
} ActiveBatch;

#endif
