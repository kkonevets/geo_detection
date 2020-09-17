#include <stdio.h>
#include <sys/time.h>
#include <stdlib.h>
#include <math.h>
#include <unistd.h>
#include "matrix.h"

Dense timeit(CSR m, uint* ixs, size_t size) {
    struct timeval tm1, tm2;
    gettimeofday(&tm1, NULL);

    Dense d = slice(m, ixs, size);

    gettimeofday(&tm2, NULL);
    unsigned long long t = 1000 * (tm2.tv_sec - tm1.tv_sec) + (tm2.tv_usec - tm1.tv_usec) / 1000;
    printf("%llu ms\n", t);
    return d;
}

void print_dense(Dense d, uint ixs[]) {
    printf("(%zu,%zu)\n", d.nrows, d.ncols);
    for (size_t i = 0; i < d.nrows * d.ncols; ++i) {
        if (i % d.ncols == 0) printf("\n%zu: ", ixs[i / d.ncols]);
        float val = d.data[i];
        if (val) printf("%f ", val);
    }
    printf("\n");
}

int main() {
    size_t num_threads = max(1, sysconf(_SC_NPROCESSORS_ONLN) / 2);
    CSR m = init_csr("../data/vk/neib_ftrs_data.bin",
                     "../data/vk/neib_ftrs_indices.bin",
                     "../data/vk/neib_ftrs_indptr.bin", num_threads);

    printf("(%zu, %zu)\n", m.nrows, m.ncols);

    // uint* ixs;
    // size_t ixs_size = read_file("../data/vk/try_ixs.bin", &ixs);
    uint ixs[] = {235, 45}; //235,45
    size_t ixs_size = sizeof(ixs) / sizeof(ixs[0]);

    Dense d = timeit(m, ixs, ixs_size);

    print_dense(d, ixs);

    free_csr(&m);
    free_dense(&d);

    return EXIT_SUCCESS;
}
