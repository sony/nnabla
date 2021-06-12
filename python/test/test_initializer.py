# Copyright 2018,2019,2020,2021 Sony Corporation.
# Copyright 2021 Sony Group Corporation.
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

import nnabla as nn
import nnabla.initializer as I


def orthogonal_test(x):
    rows, cols = x.shape[0], int(np.prod(x.shape[1:]))
    flattened = x.view().reshape((rows, cols))
    if rows > cols:
        target = np.matmul(flattened.T, flattened)
        return np.allclose(target, np.eye(cols), atol=1e-6)
    else:
        target = np.matmul(flattened, flattened.T)
        return np.allclose(target, np.eye(rows), atol=1e-6)


@pytest.mark.parametrize('shape, dim', [((2, 3), 1),
                                        ((2, 3, 4, 5), 0),
                                        ((2, 3, 4, 5), 3),
                                        ])
@pytest.mark.parametrize('eps', [1e-12])
def test_weight_normalization_initializaer(shape, dim, eps):
    rng = np.random.RandomState(313)

    # Same w returns same results
    w = nn.Variable.from_numpy_array(rng.randn(*shape))
    initializer = I.WeightNormalizationScaleInitializer(w, dim, eps)
    ret0 = initializer(w.shape)
    ret1 = initializer(w.shape)
    np.testing.assert_allclose(ret0, ret1)

    # Different w return different results
    w = nn.Variable.from_numpy_array(rng.randn(*shape))
    initializer = I.WeightNormalizationScaleInitializer(w, dim, eps)
    ret1 = initializer(w.shape)
    assert np.any(ret0 != ret1)


@pytest.mark.parametrize('rng', [None, np.random.RandomState(313)])
@pytest.mark.parametrize('shape', [
    (10,),
    (10, 8,),
    (2, 3, 4, 2, 3, 4),
])
@pytest.mark.parametrize('initializer, opts, condition', [
    (I.NormalInitializer, dict(sigma=1.0), lambda x: True),
    (I.UniformInitializer, dict(lim=(-1, 10)),
     lambda x: np.all(x >= -1) and np.all(x < 10)),
    (I.ConstantInitializer, dict(value=-2), lambda x: np.all(x == -2)),
    (I.OrthogonalInitializer, dict(gain=1.0), orthogonal_test)
])
def test_initializer_execution(shape, initializer, opts, condition, rng):
    try:
        ini = initializer(rng=rng, **opts)
    except:
        ini = initializer(**opts)
    x = ini(shape)
    assert condition(x)

    # Check difference of initialization with the same shape using different initializer
    ini0 = initializer(**opts)
    ini1 = initializer(**opts)
    x = ini0(shape)
    y = ini1(shape)
    if not isinstance(ini0, I.ConstantInitializer):
        assert np.any(x != y)
