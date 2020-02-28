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

import nnabla as nn
import numpy as np
import nnabla.functions as F
import nnabla.parametric_functions as PF
from args import get_config


def vgg_prediction(image, test=False, ncls=1000, nmaps=64, act=F.relu, config="VGG19", with_bias=True, with_bn=False, finetune=True):

    def convblock(x, nmaps, layer_idx, with_bias, with_bn=False):
        h = x
        scopenames = ["conv{}".format(_) for _ in layer_idx]
        for scopename in scopenames:
            with nn.parameter_scope(scopename):
                h = PF.convolution(h, nmaps, kernel=(3, 3), pad=(
                    1, 1), with_bias=with_bias, fix_parameters=finetune)
                if with_bn:
                    h = PF.batch_normalization(
                        h, batch_stat=not test, fix_parameters=finetune)
                h = F.relu(h, inplace=True)
        if not scopename == 'conv15':
            h = F.max_pooling(h, kernel=(2, 2), stride=(2, 2))
        return h

    assert config in ["VGG19"]
    if config == "VGG19":
        layer_indices = [(1, 2), (3, 4), (5, 6, 7, 8),
                         (9, 10, 11, 12), (13, 14, 15)]
    else:
        raise NotImplementedError

    # Preprocess
    # if not test:
        #image = F.image_augmentation(image, contrast=1.0,angle=0.25,flip_lr=True)
        #image.need_grad = False
    # image = PF.mean_subtraction(image)

    h = convblock(image, 64, layer_indices[0], with_bias, with_bn)
    h = convblock(h, 128, layer_indices[1], with_bias, with_bn)
    h = convblock(h, 256, layer_indices[2], with_bias, with_bn)
    h = convblock(h, 512, layer_indices[3], with_bias, with_bn)
    h = convblock(h, 512, layer_indices[4], with_bias, with_bn)

    with nn.parameter_scope('conv16'):
        h = (PF.convolution(h, 512, kernel=(3, 3), pad=(1, 1),
                            with_bias=with_bias, fix_parameters=finetune))
        if not finetune:
            h = F.relu(h, inplace=True)
            h = F.max_pooling(h, kernel=(2, 2), stride=(2, 2))
    if not finetune:
        with nn.parameter_scope("classifier/0"):
            nmaps = 4096
            h = PF.affine(h, nmaps, with_bias=with_bias,
                          fix_parameters=finetune)
            h = F.relu(h, inplace=True)

        with nn.parameter_scope("classifier/3"):
            nmaps = 4096
            h = PF.affine(h, nmaps, with_bias=with_bias,
                          fix_parameters=finetune)
            h = F.relu(h, inplace=True)

        with nn.parameter_scope("classifier/6"):
            h = PF.affine(h, ncls, with_bias=with_bias,
                          fix_parameters=finetune)

    return h


class PretrainedVgg19(object):
    def __init__(self):
        conf = get_config()
        self.h5_file = conf.train.vgg_pre_trained_weights
        with nn.parameter_scope("vgg19"):
            print('loading vgg19 parameters')
            nn.load_parameters(self.h5_file)
            # drop all the affine layers for finetuning.
            drop_layers = ['classifier/0/affine',
                           'classifier/3/affine', 'classifier/6/affine']
            for layers in drop_layers:
                nn.parameter.pop_parameter((layers + '/W'))
                nn.parameter.pop_parameter((layers + '/b'))
            self.mean = nn.Variable.from_numpy_array(np.asarray(
                [0.485, 0.456, 0.406]).reshape(1, 3, 1, 1))
            self.std = nn.Variable.from_numpy_array(np.asarray(
                [0.229, 0.224, 0.225]).reshape(1, 3, 1, 1))

    def __call__(self, x):
        with nn.parameter_scope("vgg19"):
            self.x = F.div2(F.sub2(x, self.mean), self.std)
            return vgg_prediction(self.x, finetune=True)
