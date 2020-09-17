from libcpp.unordered_map cimport unordered_map
from libcpp.utility cimport pair
from libc.stdint cimport uint64_t, uint32_t
from cython.operator cimport dereference as deref

cdef class MapU64U32:
    cdef:
        unordered_map[uint64_t, uint32_t] umap

        uint32_t get(self, uint64_t key) except *:
            cdef unordered_map[uint64_t, uint32_t].iterator it = self.umap.find(key)
            if it == self.umap.end():
                raise KeyError(key)
            else:
                return deref(it).second

    cpdef void reserve(self, size_t n):
        self.umap.reserve(n)

    def __setitem__(self, key: uint64_t, value: uint32_t):
        self.umap[key] = value

    def __getitem__(self, key: uint64_t) -> uint32_t:
        return self.get(key)

    def __dealloc__(self):
        # umap distructor is called automatically
        pass
