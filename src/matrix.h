/*
    Data format same as of scipy.sparse.csr_matrix.
    Is the standard CSR representation where the column indices for row i are stored
    in indices[indptr[i]:indptr[i+1]] and their corresponding values are stored
    in data[indptr[i]:indptr[i+1]]. If the shape parameter is not supplied,
    the matrix dimensions are inferred from the index arrays.
*/

#ifndef MATRIX_H
#define MATRIX_H
#undef max

#include <stdbool.h>
#include <pthread.h>
#include "C-Thread-Pool/thpool.h"

#define max(a,b) (((a) > (b)) ? (a) : (b))
#define min(a,b) (((a) < (b)) ? (a) : (b))

typedef struct CSR {
    float* data;
    uint* indices;
    uint* indptr;
    size_t data_size;
    size_t indices_size;
    size_t indptr_size;
    size_t nrows;
    size_t ncols;
    threadpool thpool;
} CSR;

typedef struct Dense {
    float* data;
    size_t nrows;
    size_t ncols;
} Dense;

typedef struct TaskArg {
    size_t start;
    size_t end;
    uint* ixs;
    bool error;
    Dense* d;
    CSR* m;
} TaskArg;


void free_csr(CSR* m);

size_t read_file(const char *path, void** buffer, char dtype) {
    FILE *fp;
    fp = fopen(path, "r");
    if (fp == NULL) {
        perror(path);
        return 0;
    }

    // obtain file size:
    fseek(fp , 0 , SEEK_END);
    size_t size = ftell(fp);
    rewind(fp);

    size_t dsize;
    if (dtype == 'I') {
        *buffer = (uint*)malloc(size * sizeof(char));
        dsize = sizeof(uint);
    }
    else if (dtype == 'f') {
        *buffer = (float*)malloc(size * sizeof(char));
        dsize = sizeof(float);
    }
    else {
        fprintf(stderr, "supported data types are uint and float\n");
        return 0;
    }
    if (*buffer == NULL) {
        fputs (path, stderr); return 0;
    }

    size_t result = fread(*buffer, 1, size, fp);
    if (result != size) {
        fputs(path, stderr); return 0;
    }
    fclose(fp);

    return size / dsize;
}

bool check_csr(CSR m) {
    // TODO
    if (m.indptr[0] != 0 ) {
        fprintf(stderr, "index pointer should start with 0\n");
        return false;
    }
    if (m.data_size != m.indices_size) {
        fprintf(stderr, "indices and data should have the same size\n");
        return false;
    }
    if (m.indptr[m.indptr_size - 1] > m.indices_size) {
        fprintf(stderr, "Last value of index pointer should be less than "
                "the size of index and data arrays\n");
        return false;
    }
    for (size_t i = 1; i < m.indptr_size; i++) {
        if ( m.indptr[i] < m.indptr[i - 1] ) {
            fprintf(stderr, "index pointer values must form a "
                    "non-decreasing sequence\n");
            return false;
        }
    }
    return true;
}

CSR init_csr(const char *data_path, const char *indices_path,
             const char *indptr_path, size_t num_threads) {
    CSR m = {NULL, NULL, NULL, 0, 0, 0, 0, 0};

    m.data_size = read_file(data_path, (void**)&m.data, 'f');
    if (!m.data_size) {free_csr(&m); return m;}
    m.indices_size = read_file(indices_path, (void**)&m.indices, 'I');
    if (!m.indices_size) {free_csr(&m); return m;}
    m.indptr_size = read_file(indptr_path, (void**)&m.indptr, 'I');
    if (!m.indptr_size) {free_csr(&m); return m;}

    m.nrows = m.indptr_size - 1;
    m.ncols = 0;
    for (size_t j = 0; j < m.indices_size; ++j) {
        m.ncols = (size_t)max(m.ncols, m.indices[j]);
    }
    m.ncols += 1;

    if (!check_csr(m)) {
        free_csr(&m);
        return m;
    }

    if (!(m.thpool = thpool_init(num_threads))) {
        free_csr(&m);
        perror("");
        return m;
    }

    return m;
}

void free_csr(CSR* m) {
    if (m->data) {
        free(m->data);
        m->data = NULL;
    }
    if (m->indices) {
        free(m->indices);
        m->indices = NULL;
    }
    if (m->indptr) {
        free(m->indptr);
        m->indptr = NULL;
    }
    m->data_size = 0;
    m->indices_size = 0;
    m->indptr_size = 0;
    m->nrows = 0;
    m->ncols = 0;
    thpool_destroy(m->thpool);
}

void free_dense(Dense* m) {
    if (m->data) {
        free(m->data);
        m->data = NULL;
    }
    m->nrows = 0;
    m->ncols = 0;
}

void task(TaskArg* arg) {
    // printf("start %zu, end %zu, thread_id%u\n", arg->start, arg->end, (int)pthread_self());
    CSR* m = arg->m;
    Dense* d = arg->d;
    for (size_t i = arg->start; i < arg->end; ++i) {
        size_t ix = arg->ixs[i];
        if (ix > m->nrows) {
            fprintf(stderr, "Index %zu is out of range (0, %zu)\n", ix, m->nrows);
            arg->error = true;
            return;
        }
        for (size_t j = m->indptr[ix]; j < m->indptr[ix + 1]; ++j) {
            d->data[i * m->ncols + m->indices[j]] = m->data[j];
        }
    }
}

Dense slice(CSR m, uint* ixs, size_t size) {
    Dense d = {NULL, size, m.ncols};

    if (m.nrows == 0 && m.ncols == 0) {
        free_dense(&d);
        return d;
    }

    d.data = (float*)calloc(size * m.ncols, sizeof(float));
    if (d.data == NULL) {
        fputs ("Memory error", stderr);
        free_dense(&d);
        return d;
    }

    size_t num_threads = thpool_num_threads_alive(m.thpool);
    size_t minimum_chunk_size = 1;
    size_t chunk_size = max(minimum_chunk_size, size / num_threads);
    TaskArg* args = (TaskArg*)calloc(num_threads, sizeof(TaskArg));

    for (size_t start = 0, end = 0, i = 0; end != size; ++i) {
        if (i == num_threads - 1) { // last takes the rest
            end = size;
        }
        else {
            end += min(size - end, chunk_size);
        }

        TaskArg* arg = args + i;
        arg->start = start;
        arg->end = end;
        arg->ixs = ixs;
        arg->error = false;
        arg->d = &d;
        arg->m = &m;

        if (thpool_add_work(m.thpool, (void*)task, (void*)arg) == -1) {
            perror("");
            free_dense(&d);
            break;
        }

        start = end;
    }

    thpool_wait(m.thpool);

    for (TaskArg* arg = args; arg != args + num_threads; ++arg) {
        if (arg->error) {
            free_dense(&d);
            break;
        }
    }
    free(args);

    return d;
}

#endif
