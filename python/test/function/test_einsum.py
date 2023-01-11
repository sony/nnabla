# Copyright 2018,2019,2020,2021 Sony Corporation.
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
import numpy as np

import nnabla.functions as F
from nbla_test_utils import list_context

ctxs = list_context('Einsum')


def ref_einsum(*x, equation):
    return np.einsum(equation, *x)


@pytest.mark.parametrize("ctx, func_name", ctxs)
@pytest.mark.parametrize("x_shapes, equation", [
    # Diagonal
    ([(4, 4)], "ii->i"),
    ([(2, 3, 4, 4)], "abii->abi"),
    ([(2, 3, 4, 4)], "...ii->...i"),
    ([(4, 2, 3, 4)], "i...i->...i"),
    ([(4, 4, 2, 3)], "ii...->i..."),
    # Trace
    ([(4, 4)], "ii->"),
    ([(4, 4)], "ii"),
    ([(2, 3, 4, 4)], "abii->ab"),
    ([(2, 3, 4, 4)], "abii"),
    ([(2, 3, 4, 4)], "...ii->..."),
    ([(4, 2, 3, 4)], "i...i->..."),
    ([(4, 4, 2, 3)], "ii...->..."),
    # Inner product
    ([(5,), (5,)], "i,i"),
    # Transpose
    ([(2, 3, 4)], "bij->bji"),
    ([(2, 3, 4)], "bji"),
    ([(2, 3, 4, 5)], "...ij->...ji"),
    ([(2, 3, 4, 5)], "i...j->ji..."),
    ([(2, 3, 4, 5)], "ij...->j...i"),
    ([(2, 3, 4, 5)], "ij..."),
    # Identity
    ([(5, 5)], "ij"),
    ([(5, 5)], "ij->"),
    ([(5, 5)], "ij->ij"),
    # Sum
    ([(2, 3, 4, 5)], "ijkl->i"),
    ([(2, 3, 4, 5)], "ij...->..."),
    ([(2, 3, 4, 5)], "i...j->..."),
    ([(2, 3, 4, 5)], "...ij->..."),
    # Matmul
    ([(2, 3, 4, 5), (2, 3, 5, 6)], "abij,abjk->abik"),
    ([(2, 3, 4, 5), (2, 3, 5, 6)], "abij,abjk"),
    ([(2, 3, 4, 5), (2, 3, 5, 6)], "...ij,...jk->...ik"),
    ([(1, 3, 4, 5), (2, 1, 5, 6)], "...ij,...jk->...ik"),
    ([(4, 2, 3, 5), (2, 3, 5, 6)], "i...j,...jk->ik..."),
    ([(4, 2, 3, 5), (2, 3, 5, 6)], "i...j,...jk"),
    # All pass
    ([(2, 3, 4, 2, 5), (2, 4, 3, 2)], "bjibw,zkjb->ikb"),
    ([(1, 1, 1, 1, 1), (1, 1, 1, 1)], "bjibw,zkjb"),
    ([(2, 2, 3, 2, 2, 2), (2, 2, 2, 2, 3), (2, 2, 2, 2, 3)],
     "wijw...,k...xj,...zkl->li..."),
    ([(2, 2, 3, 2, 2, 2), (2, 2, 2, 2, 3), (2, 2, 2, 2, 3)], "wijw...,k...xj,...zkl"),
    [[(1, 2, 3, 4), (2, 1, 4, 2), (2, 2, 1, 3),
      (3, 2, 3, 1)], "zabc,aycd,daxe,bdfw->bef"]
])
@pytest.mark.parametrize("seed", [313])
def test_einsum_forward_backward(seed, x_shapes, equation, ctx, func_name):
    from nbla_test_utils import function_tester
    rng = np.random.RandomState(seed)
    inputs = [rng.randn(*shape).astype(np.float32) for shape in x_shapes]
    backward = [True] * len(inputs)
    function_tester(rng, F.einsum, ref_einsum, inputs, ctx=ctx, func_name=func_name,
                    atol_f=0, atol_b=2e-2, func_kwargs=dict(equation=equation), backward=backward)
