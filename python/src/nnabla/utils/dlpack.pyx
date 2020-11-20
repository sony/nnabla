# Copyright (c) 2017-2020 Sony Corporation. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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
    '''
    Decode a DLPack to :obj:`~nnabla.NdArray`.

    Example:

    .. code-block:: python

        # Create a tensor of an external tool, and encode as an DLPack.
        import torch
        from torch.utils.dlpack import to_dlpack
        t = torch.ones((5, 5), dtype=torch.float32,
                       device=torch.device('cuda'))
        dlp = to_dlpack(t)

        # Borrow the DLPack tensor as nnabla.NdArray
        from nnabla.utils.dlpack import from_dlpack
        arr = from_dlpack(dlp)

    If you want to move an ownership of DLPack to an exiting ``NdArray``;

    .. code-block:: python

        from nnabla import NdArray
        arr = NdArray()
        from_dlpack(dlp, arr=arr)

    Args:

        dlp(PyCapsule):
            A PyCapsule object of a ``DLManagedTensor`` (as ``"dltensor"``) which
            internal memory is borrowed by a tensor of an external package.
            The ownership of the ``DLManagedTensor``
            is moved to an ``NdArray`` object, and the PyCapsule object is marked
            as ``"used_dltensor"`` to inform that the ownership has been moved.
        arr(~nnabla.NdArray):
            If specified, a given DLPack is decoded to it. Otherwise, it
            creates a new ``NdArray`` object and decodes the DLPack to it.

    Returns:
        ~nnabla.NdArray: an ``NdArray`` object borrowing the DLPack tensor.

    '''
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


def to_dlpack(a, dtype=None, ctx=None):
    '''
    Returns a DLPack which owns an internal array object borrowed by a
    specified :obj:`~nnabla.NdArray`.

    Example:

    .. code-block:: python

        # Create a nnabla.NdArray in CUDA.
        import nnabla as nn
        from nnabla.ext_utils import get_extension_context
        ctx = get_extension_context('cudnn')
        nn.set_default_context(ctx)

        a = nn.NdArray.from_numpy_array(np.ones((5, 5), dtype=np.float32))
        a.cast(np.float32, ctx)

        # Expose as a DLPack.
        from nnabla.utils.dlpack import to_dlpack
        dlp = to_dlpack(a)

        # Use the DLPack in PyTorch.
        import torch
        from torch.utils.dlpack import from_dlpack
        t = from_dlpack(dlp)

        # Changing the values in Torch will also be affected in nnabla
        # because they share memory.
        t.add_(1)
        print(a.data)  # All values become 2.

    Args:
        a (~nnabla.NdArray):
            An ``NdArray`` object. An internal array which is recently modified
            or created will be encoded into a DLPack.
        dtype (numpy.dtype):
            If specified, in-place cast operation may be performed before
            encoding it to a DLPack.
        ctx (~nnabla.Context):
            If specified, in-place device transfer operation may be performed
            before encoding it into a DLPack.

    Returns:
        PyCapsule:
            A PyCapsule object of a ``DLManagedTensor`` (as ``"dltensor"``) which
            internal memory is borrowed by the specified ``NdArray``.

    '''
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
