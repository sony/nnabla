from .._nd_array cimport NdArray
from dlpack cimport from_dlpack as c_from_dlpack, \
                    to_dlpack as c_to_dlpack, \
                    call_deleter
from cpython.pycapsule cimport *
from cpython cimport Py_INCREF
import nnabla as nn
import numpy as np

'''
NNabla expects the followings to a borrowed DLPack.
- The name of PyCapsule is "dltensor".
- If some framwork borrows the tensor from NNabla,
  it renames the PyCapsule other than "dltensor".
- DLManagedTensor::deleter deletes DLManagedTensor itself
  because NNabla will not do this to avoid double free.
'''

cdef const char *c_str_dltensor = "dltensor"
cdef const char *c_str_used_dltensor = "used_dltensor"


def from_dlpack(dlp, arr=None):
    if not PyCapsule_IsValid(dlp, c_str_dltensor):
        raise ValueError("Invalid PyCapsule.")

    cdef void *cdlp = PyCapsule_GetPointer(dlp, c_str_dltensor)
    if arr is None:
        arr = NdArray.create(c_from_dlpack(<CDLManagedTensor*> cdlp))
    else:
        c_from_dlpack(<CDLManagedTensor*> cdlp, (<NdArray?> arr).arrp)

    # NNabla gets the ownership of DLManagedTensor.
    # The rename from "dltensor" to "used_dltensor" informs
    # that this DLPack is already borrowed by NNabla.
    PyCapsule_SetName(dlp, c_str_used_dltensor)
    # The destructor is unset because NNabla will delete the object.
    PyCapsule_SetDestructor(dlp, NULL)

    return arr


# DLManagedTensor in PyCapsule which is not borrwed from any framework
# is deleted by this destructor of Pycapsule.
cdef void delete_unused_dltensor(object dlp):
    if PyCapsule_IsValid(dlp, c_str_dltensor):
        call_deleter(<CDLManagedTensor *>PyCapsule_GetPointer(dlp, c_str_dltensor));


# The head array of SyncedArray is set in DLPack.
# If dtype and ctx are given, the array is casted before the setting in DLPack.
def to_dlpack(a, dtype=None, ctx=None):
    cdef CDLManagedTensor *cdlp
    cdef CContext cctx
    cdef dtypes cdtype
    cdef int type_num
    
    if ctx is not None and dtype is not None:
        cdlp = c_to_dlpack((<NdArray ?> a).arrp)
    else:
        if ctx is None:
            ctx = nn.get_current_context()
        cctx = <CContext ?> ctx

        if dtype is None:
            cdtype = (<NdArray ?> a).arrp.array().get().dtype()
        else:
            type_num = np.dtype(dtype).num
            cdtype = <dtypes>type_num

        cdlp = c_to_dlpack((<NdArray ?> a).arrp, cdtype, cctx)

    return PyCapsule_New(<void*>cdlp, c_str_dltensor, &delete_unused_dltensor)
