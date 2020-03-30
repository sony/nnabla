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

from .base import ResNetBase
from . import registry


def resnet(x, num_classes=1000, num_layers=50, test=True, channel_last=False):
    '''
    Defines ResNet.

    See `ResNetBase` for more details.

    '''
    net = ResNetBase(num_classes, num_layers, test=test,
                     channel_last=channel_last)
    return net(x)


def resnet18(x, num_classes=1000, test=True, channel_last=False):
    '''
    Defines ResNet18.

    See `ResNetBase` for more details.

    '''
    return resnet(x, num_classes, 18, test=test, channel_last=channel_last)


def resnet34(x, num_classes=1000, test=True, channel_last=False):
    '''
    Defines ResNet34.

    See `ResNetBase` for more details.

    '''
    return resnet(x, num_classes, 34, test=test, channel_last=channel_last)


def resnet50(x, num_classes=1000, test=True, channel_last=False):
    '''
    Defines ResNet50.

    See `ResNetBase` for more details.

    '''
    return resnet(x, num_classes, 50, test=test, channel_last=channel_last)


def resnet101(x, num_classes=1000, test=True, channel_last=False):
    '''
    Defines ResNet101.

    See `ResNetBase` for more details.

    '''
    return resnet(x, num_classes, 101, test=test, channel_last=channel_last)


def resnet152(x, num_classes=1000, test=True, channel_last=False):
    '''
    Defines ResNet152.

    See `ResNetBase` for more details.

    '''
    return resnet(x, num_classes, 152, test=test, channel_last=channel_last)


# Register arch functions
for ln in (18, 34, 50, 101, 152):
    fn_name = 'resnet{}'.format(ln)
    registry.register_arch_fn(fn_name, eval(fn_name))
