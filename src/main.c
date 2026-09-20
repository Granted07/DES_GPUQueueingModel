#include "sweep.h"

#include <stdio.h>

int main(void)
{
    if (run_sweep("queue_cache_results.csv") != 0) {
        return 1;
    }
    printf("Wrote queue_cache_results.csv\n");
    return 0;
}
