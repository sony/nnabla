# Copyright (c) 2020 Sony Corporation. All Rights Reserved.
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

import pytest
import nnabla as nn
import nnabla.utils.dlpack
import numpy as np
from nnabla.ext_utils import list_extensions, \
                             get_extension_context
from nnabla.testing import assert_allclose

# Try to import PyTorch
try:
    import torch
    import torch.utils.dlpack
except ImportError:
    pytest.skip('PyTorch is not installed.',
                allow_module_level=True)


ext_names = list_extensions()
types = [[np.float32, torch.float32]]


# PyTorch to NNabla
@pytest.mark.parametrize("ext_name", ext_names)
@pytest.mark.parametrize("numpy_type, torch_type", types)
def test_from_dlpack_new(ext_name, numpy_type, torch_type):
    ctx = get_extension_context(ext_name)
    device_name = ctx.backend[0].split(':')[0]
    if device_name == 'cudnn':
        device_name = 'cuda'  # for PyTorch
    nn.set_default_context(ctx)

    # Init PyTorch Tensor
    t = torch.ones((5, 5), dtype=torch_type,
                   device=torch.device(device_name))

    # PyTorch to DLPack
    dlp = torch.utils.dlpack.to_dlpack(t)

    # DLPack to NNabla
    a = nn.utils.dlpack.from_dlpack(dlp)
    assert a.dtype == numpy_type

    # Check if the memory locations are still same,
    # which means DlpackArray is not copied to other arrays
    # in the same ArrayGroup.
    a += 1
    assert np.all(a.data == t.to('cpu').detach().numpy().copy())


@pytest.mark.parametrize("ext_name", ext_names)
@pytest.mark.parametrize("numpy_type, torch_type", types)
def test_from_dlpack_given(ext_name, numpy_type, torch_type):
    ctx = get_extension_context(ext_name)
    device_name = ctx.backend[0].split(':')[0]
    if device_name == 'cudnn':
        device_name = 'cuda'  # for PyTorch
    nn.set_default_context(ctx)

    # Init PyTorch Tensor
    t = torch.ones((5, 5), dtype=torch_type,
                   device=torch.device(device_name))

    # PyTorch to DLPack
    dlp = torch.utils.dlpack.to_dlpack(t)

    # DLPack to NNabla
    a = nn.NdArray()
    nn.utils.dlpack.from_dlpack(dlp, a)
    assert a.dtype == numpy_type

    # Check if the memory locations are still same,
    # which means DlpackArray is not copied to other arrays
    # in the same ArrayGroup.
    a += 1
    assert np.all(a.data == t.to('cpu').detach().numpy().copy())


# NNabla to Pytorch
@pytest.mark.parametrize("ext_name", ext_names)
@pytest.mark.parametrize("numpy_type, torch_type", types)
@pytest.mark.parametrize("set_dtype, set_ctx", [(False, False), (True, False),
                                                (False, True), (True, True)])
def test_to_dlpack(ext_name, numpy_type, torch_type, set_dtype, set_ctx):
    ctx = get_extension_context(ext_name)
    nn.set_default_context(ctx)

    # Init nnabla.NdArray
    a = nn.NdArray.from_numpy_array(np.ones((5, 5), dtype=numpy_type))
    a.cast(numpy_type, ctx)

    # NNabla to DLPack
    passed_dtype = numpy_type if set_dtype else None
    passed_ctx = ctx if set_ctx else None
    dlp = nn.utils.dlpack.to_dlpack(a, passed_dtype, passed_ctx)

    # DLPack to PyTorch
    t = torch.utils.dlpack.from_dlpack(dlp)
    assert t.dtype == torch_type

    # Check if the memory locations are still same.
    t.add_(1)  # in-place add
    assert np.all(a.data == t.to('cpu').detach().numpy().copy())


# NNabla to Pytorch
@pytest.mark.parametrize("ext_name", ext_names)
@pytest.mark.parametrize("numpy_type, torch_type", types)
def test_to_dlpack_no_borrower(ext_name, numpy_type, torch_type):
    ctx = get_extension_context(ext_name)
    nn.set_default_context(ctx)

    # Init nnabla.NdArray
    a = nn.NdArray.from_numpy_array(np.ones((5, 5), dtype=numpy_type))

    # NNabla to DLPack
    dlp = nn.utils.dlpack.to_dlpack(a)

    # No one borrows dlp. It will be destruct by PyCapsule destructor.
