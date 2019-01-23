# Copyright (c) 2017 Sony Corporation. All Rights Reserved.
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


from __future__ import absolute_import
import pytest
from nnabla.utils.nnp_graph import NnpNetwork
import nnabla as nn
import numpy as np

from nnabla.models.utils import get_model_url_base_from_env


@pytest.mark.skipif(
    get_model_url_base_from_env() is None,
    reason='models are tested only when NNBLA_MODELS_URL_BASE is specified as an envvar')
@pytest.mark.parametrize('num_layers', [18, 34, 50, 101, 152])
@pytest.mark.parametrize('image_size', [224, 448])
@pytest.mark.parametrize('batch_size', [1, 5])
@pytest.mark.parametrize('seed', [1223])
def test_nnabla_models_resnet(num_layers, image_size, batch_size, seed):
    from nnabla.models.imagenet import ResNet
    nn.clear_parameters()
    rng = np.random.RandomState(seed)
    model = ResNet(num_layers)
    x = nn.Variable.from_numpy_array(rng.randint(
        0, 256, size=(batch_size, 3, image_size, image_size)))
    for use_up_to in ('classifier', 'pool', 'block4', 'block4+relu'):
        check_global_pooling = True
        force_global_pooling = False
        returns_net = False

        def _execute():
            y = model(x, use_up_to=use_up_to, force_global_pooling=force_global_pooling,
                      check_global_pooling=check_global_pooling)
            y.forward()

        if image_size == 448 and use_up_to in ('classifier', 'pool'):
            with pytest.raises(ValueError):
                _execute()
            if use_up_to == 'pool':
                check_global_pooling = False
                _execute()
            force_global_pooling = True
        _execute()
        net = model(x, use_up_to=use_up_to, force_global_pooling=force_global_pooling,
                    check_global_pooling=check_global_pooling, returns_net=True)
        assert isinstance(net, NnpNetwork)
        assert len(net.inputs.values()) == 1
        assert len(net.outputs.values()) == 1
