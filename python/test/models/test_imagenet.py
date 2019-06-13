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


def _check_trainable_parameters(y):

    def _check_trainable_inputs(f):
        for i in f.inputs:
            if i.need_grad:
                return True
        return False

    return y.visit_check(_check_trainable_inputs)


@pytest.mark.skipif(
    get_model_url_base_from_env() is None,
    reason='models are tested only when NNBLA_MODELS_URL_BASE is specified as an envvar')
@pytest.mark.parametrize('num_layers', [18, 34, 50, 101, 152])
@pytest.mark.parametrize('image_size', [224, 448])
@pytest.mark.parametrize('batch_size', [1, 5])
@pytest.mark.parametrize('training', [False, True])
@pytest.mark.parametrize('seed', [1223])
def test_nnabla_models_resnet(num_layers, image_size, batch_size, training, seed):
    from nnabla.models.imagenet import ResNet
    nn.clear_parameters()
    rng = np.random.RandomState(seed)
    model = ResNet(num_layers)
    x = nn.Variable.from_numpy_array(rng.randint(
        0, 256, size=(batch_size, 3, image_size, image_size)))
    for use_up_to in ('classifier', 'pool', 'lastconv', 'lastconv+relu'):
        check_global_pooling = True
        force_global_pooling = False
        returns_net = False

        def _execute():
            y = model(x, training=training, use_up_to=use_up_to, force_global_pooling=force_global_pooling,
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
        net = model(x, training=training, use_up_to=use_up_to, force_global_pooling=force_global_pooling,
                    check_global_pooling=check_global_pooling, returns_net=True)
        assert isinstance(net, NnpNetwork)
        assert len(net.inputs.values()) == 1
        assert len(net.outputs.values()) == 1
        y = list(net.outputs.values())[0]
        if training:
            assert _check_trainable_parameters(y)
        else:
            assert not _check_trainable_parameters(y)


@pytest.mark.skipif(
    get_model_url_base_from_env() is None,
    reason='models are tested only when NNBLA_MODELS_URL_BASE is specified as an envvar')
@pytest.mark.parametrize('model_class, up_to_list', [
    ('MobileNet', ['classifier', 'pool', 'lastconv', 'lastconv+relu']),
    ('MobileNetV2', ['classifier', 'pool', 'lastconv', 'lastconv+relu']),
    ('NIN', ['classifier', 'pool', 'lastconv', 'lastconv+relu']),
    ('SENet', ['classifier', 'pool', 'lastconv', 'lastconv+relu']),
    ('DenseNet', ['classifier', 'pool', 'lastconv', 'lastconv+relu']),
    ('InceptionV3', ['classifier', 'pool', 'prepool']),
    ('Xception', ['classifier', 'pool', 'lastconv', 'lastconv+relu'])
    ])
@pytest.mark.parametrize('image_size_factor', [1, 2])
@pytest.mark.parametrize('batch_size', [1, 5])
@pytest.mark.parametrize('training', [False, True])
@pytest.mark.parametrize('seed', [1223])
def test_nnabla_models_imagenet_etc(model_class, up_to_list, image_size_factor, batch_size, training, seed):
    nn.clear_parameters()
    rng = np.random.RandomState(seed)

    # Load model
    from nnabla.models import imagenet
    Model = getattr(imagenet, model_class)
    model = Model()

    # Determine input shape and create input variable
    input_shape = list(model.input_shape)
    input_shape[1] *= image_size_factor
    input_shape[2] *= image_size_factor
    input_shape = tuple(input_shape)
    x = nn.Variable.from_numpy_array(rng.randint(
        0, 256, size=(batch_size,) + input_shape))

    # Test cases for all intermediate outputs
    for use_up_to in up_to_list:
        check_global_pooling = True
        force_global_pooling = False
        returns_net = False

        def _execute():
            y = model(x, training=training, use_up_to=use_up_to, force_global_pooling=force_global_pooling,
                      check_global_pooling=check_global_pooling)
            y.forward()

        # Need special care for SENet because it contains global average
        # pooling in various points in a network.
        if image_size_factor != 1 and (model_class == 'SENet' or use_up_to in ('classifier', 'pool')):
            with pytest.raises(ValueError):
                _execute()
            if use_up_to == 'pool' and model_class != 'SENet':
                check_global_pooling = False
                _execute()
            force_global_pooling = True
        _execute()
        net = model(x, training=training, use_up_to=use_up_to, force_global_pooling=force_global_pooling,
                    check_global_pooling=check_global_pooling, returns_net=True)
        assert isinstance(net, NnpNetwork)
        assert len(net.inputs.values()) == 1
        assert len(net.outputs.values()) == 1
        y = list(net.outputs.values())[0]
        if training:
            assert _check_trainable_parameters(y)
        else:
            assert not _check_trainable_parameters(y)


@pytest.mark.skipif(
    get_model_url_base_from_env() is None,
    reason='models are tested only when NNBLA_MODELS_URL_BASE is specified as an envvar')
@pytest.mark.parametrize('num_layers', [11, 13, 16])
@pytest.mark.parametrize('batch_size', [1, 5])
@pytest.mark.parametrize('training', [False, True])
@pytest.mark.parametrize('seed', [1223])
def test_nnabla_models_vgg(num_layers, batch_size, training, seed):
    from nnabla.models.imagenet import VGG
    nn.clear_parameters()
    rng = np.random.RandomState(seed)
    model = VGG(num_layers)

    # Determine input shape and create input variable
    input_shape = list(model.input_shape)
    input_shape = tuple(input_shape)
    x = nn.Variable.from_numpy_array(rng.randint(
        0, 256, size=(batch_size,) + input_shape))

    # Test cases for all intermediate outputs
    for use_up_to in ('classifier', 'pool', 'lastconv', 'lastconv+relu'):
        returns_net = False

        def _execute():
            y = model(x, training=training, use_up_to=use_up_to)
            y.forward()

        _execute()
        net = model(x, training=training,
                    use_up_to=use_up_to, returns_net=True)
        assert isinstance(net, NnpNetwork)
        assert len(net.inputs.values()) == 1
        assert len(net.outputs.values()) == 1
        y = list(net.outputs.values())[0]
        if training:
            assert _check_trainable_parameters(y)
        else:
            assert not _check_trainable_parameters(y)


@pytest.mark.skipif(
    get_model_url_base_from_env() is None,
    reason='models are tested only when NNBLA_MODELS_URL_BASE is specified as an envvar')
@pytest.mark.parametrize('version', ['v1.0', 'v1.1'])
@pytest.mark.parametrize('image_size_factor', [1, 2])
@pytest.mark.parametrize('batch_size', [1, 5])
@pytest.mark.parametrize('training', [False, True])
@pytest.mark.parametrize('seed', [1223])
def test_nnabla_models_squeezenet(version, image_size_factor, batch_size, training, seed):
    nn.clear_parameters()
    rng = np.random.RandomState(seed)

    # Load model
    from nnabla.models.imagenet import SqueezeNet
    model = SqueezeNet(version)

    # Determine input shape and create input variable
    input_shape = list(model.input_shape)
    input_shape[1] *= image_size_factor
    input_shape[2] *= image_size_factor
    input_shape = tuple(input_shape)
    x = nn.Variable.from_numpy_array(rng.randint(
        0, 256, size=(batch_size,) + input_shape))

    # Test cases for all intermediate outputs
    for use_up_to in ('classifier', 'pool', 'lastconv', 'lastconv+relu'):
        check_global_pooling = True
        force_global_pooling = False
        returns_net = False

        def _execute():
            y = model(x, training=training, use_up_to=use_up_to, force_global_pooling=force_global_pooling,
                      check_global_pooling=check_global_pooling)
            y.forward()

        if image_size_factor != 1 and use_up_to in ('classifier', 'pool'):
            with pytest.raises(ValueError):
                _execute()
            if use_up_to == 'pool':
                check_global_pooling = False
                _execute()
            force_global_pooling = True
        _execute()
        net = model(x, training=training, use_up_to=use_up_to, force_global_pooling=force_global_pooling,
                    check_global_pooling=check_global_pooling, returns_net=True)
        assert isinstance(net, NnpNetwork)
        assert len(net.inputs.values()) == 1
        assert len(net.outputs.values()) == 1
        y = list(net.outputs.values())[0]
        if training:
            assert _check_trainable_parameters(y)
        else:
            assert not _check_trainable_parameters(y)
