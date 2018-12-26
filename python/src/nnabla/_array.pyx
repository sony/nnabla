from _array cimport *

cdef class Array:
    """Holding a shared pointer of C Array class.
    """
    @staticmethod
    cdef object create(ConstArrayPtr arr):
        a = Array()
        a.arr = <shared_ptr[const CArray] > arr
        return a

    def __init__(self):
        pass
