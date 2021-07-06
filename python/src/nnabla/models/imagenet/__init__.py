# Copyright 2019,2020,2021 Sony Corporation.
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
from .resnet import ResNet, ResNet18, ResNet34, ResNet50, ResNet101, ResNet152
from .mobilenet import MobileNet, MobileNetV2
from .senet import SENet
from .squeezenet import SqueezeNet, SqueezeNetV10, SqueezeNetV11
from .vgg import VGG, VGG11, VGG13, VGG16
from .nin import NIN
from .densenet import DenseNet
from .inception import InceptionV3
from .xception import Xception
from .googlenet import GoogLeNet
from .resnext import ResNeXt, ResNeXt50, ResNeXt101
from .shufflenet import ShuffleNet, ShuffleNet10, ShuffleNet05, ShuffleNet20
from .alexnet import AlexNet
