from _array cimport *

cdef class Array:
    """Holding a shared pointer of C Array class.
    """
    @staticmethod
    cdef object create(ArrayPtr arr):
        a = Array()
        a.arr = arr
        return a

    def __init__(self):
        pass
