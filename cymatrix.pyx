cimport cython
cimport numpy as np
from cpython cimport Py_buffer
from cython cimport view
from libc.stdlib cimport free

import numpy as np

ctypedef np.uint32_t DTYPE_t

cdef extern from "src/matrix.h":
    ctypedef struct CSR:
        size_t nrows
        size_t ncols

    ctypedef struct Dense:
        float* data
        size_t nrows
        size_t ncols

    CSR init_csr(const char *data_path, const char *indices_path, 
            const char *indptr_path, size_t num_threads)
    void free_csr(CSR* m)
    Dense slice(CSR m, unsigned int* ixs, size_t size)


@cython.boundscheck(False)
@cython.wraparound(False)
cdef class CSRMatrix:
    cdef CSR m
    cpdef Py_ssize_t shape[2]

    def __cinit__(self, data_path, indices_path, indptr_path, num_threads):
        self.m = init_csr(data_path.encode('utf-8'), 
                        indices_path.encode('utf-8'), 
                        indptr_path.encode('utf-8'), 
                        num_threads)
        self.shape[0] = self.m.nrows
        self.shape[1] = self.m.ncols

    property shape:
        def __get__(self):
            return self.shape

    def __getitem__(self, unsigned int[:] ixs):
        cdef Dense d = slice(self.m, &ixs[0], len(ixs))
        if d.data == NULL:
            return None
        cdef view.array arr = view.array(
            shape=(d.nrows, d.ncols), 
            itemsize=sizeof(float), format="f",
            mode="c", allocate_buffer=False)
        arr.data = <char *> d.data
        arr.callback_free_data = free
        return np.ndarray(arr.shape,
                   buffer=memoryview(arr),
                   order='C',
                   dtype=np.float32)

    def __dealloc__(self):
        free_csr(&self.m)


