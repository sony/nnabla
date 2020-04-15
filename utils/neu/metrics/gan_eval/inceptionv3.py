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
import nnabla.functions as F
import nnabla.parametric_functions as PF
import numpy as np


def construct_inceptionv3(x, use_up_to="pool"):

    def stem_block(input_variable, outmaps, kernel=(3, 3), pad=(0, 0), stride=(1, 1), eps=1e-3):

        with nn.parameter_scope(f"Convolution"):
            h = PF.convolution(input_variable, outmaps=outmaps,
                               kernel=kernel, pad=pad, stride=stride, with_bias=False)
        with nn.parameter_scope(f"BatchNormalization"):
            h = PF.batch_normalization(h, batch_stat=False, eps=eps)
        h = F.relu(h)

        return h

    def module_A(input_variable, is_first=False, eps=1e-3):

        with nn.parameter_scope(f"Conv"):
            with nn.parameter_scope("Convolution"):
                h0 = PF.convolution(input_variable, outmaps=64, kernel=(
                    1, 1), pad=(0, 0), stride=(1, 1), with_bias=False)
            with nn.parameter_scope("BatchNormalization"):
                h0 = PF.batch_normalization(h0, batch_stat=False, eps=eps)
            h0 = F.relu(h0)

        #################################################################

        with nn.parameter_scope(f"Conv_2"):
            with nn.parameter_scope("Convolution"):
                h1 = PF.convolution(input_variable, outmaps=48, kernel=(
                    1, 1), pad=(0, 0), stride=(1, 1), with_bias=False)
            with nn.parameter_scope("BatchNormalization"):
                h1 = PF.batch_normalization(h1, batch_stat=False, eps=eps)
            h1 = F.relu(h1)

        with nn.parameter_scope(f"Conv_3"):
            with nn.parameter_scope("Convolution"):
                h1 = PF.convolution(h1, outmaps=64, kernel=(
                    5, 5), pad=(2, 2), stride=(1, 1), with_bias=False)
            with nn.parameter_scope("BatchNormalization"):
                h1 = PF.batch_normalization(h1, batch_stat=False, eps=eps)
            h1 = F.relu(h1)

        #################################################################

        with nn.parameter_scope(f"Conv_4"):
            with nn.parameter_scope("Convolution"):
                h2 = PF.convolution(input_variable, outmaps=64, kernel=(
                    1, 1), pad=(0, 0), stride=(1, 1), with_bias=False)
            with nn.parameter_scope("BatchNormalization"):
                h2 = PF.batch_normalization(h2, batch_stat=False, eps=eps)
            h2 = F.relu(h2)

        with nn.parameter_scope(f"Conv_5"):
            with nn.parameter_scope("Convolution"):
                h2 = PF.convolution(h2, outmaps=96, kernel=(
                    3, 3), pad=(1, 1), stride=(1, 1), with_bias=False)
            with nn.parameter_scope("BatchNormalization"):
                h2 = PF.batch_normalization(h2, batch_stat=False, eps=eps)
            h2 = F.relu(h2)

        with nn.parameter_scope(f"Conv_6"):
            with nn.parameter_scope("Convolution"):
                h2 = PF.convolution(h2, outmaps=96, kernel=(
                    3, 3), pad=(1, 1), stride=(1, 1), with_bias=False)
            with nn.parameter_scope("BatchNormalization"):
                h2 = PF.batch_normalization(h2, batch_stat=False, eps=eps)
            h2 = F.relu(h2)

        #################################################################

        with nn.parameter_scope(f"Conv_7"):
            h3 = F.average_pooling(input_variable, kernel=(
                3, 3), pad=(1, 1), stride=(1, 1), including_pad=False)
            with nn.parameter_scope("Convolution"):
                if is_first:
                    outmaps = 32
                else:
                    outmaps = 64
                h3 = PF.convolution(h3, outmaps=outmaps, kernel=(
                    1, 1), pad=(0, 0), stride=(1, 1), with_bias=False)
            with nn.parameter_scope("BatchNormalization"):
                h3 = PF.batch_normalization(h3, batch_stat=False, eps=eps)
            h3 = F.relu(h3)

        h = F.concatenate(*[h0, h1, h2, h3], axis=1)

        return h

    def grid_size_reduction_A(input_variable, eps=1e-3):

        with nn.parameter_scope(f"Conv"):
            with nn.parameter_scope("Convolution"):
                h1 = PF.convolution(input_variable, outmaps=384, kernel=(
                    3, 3), pad=(0, 0), stride=(2, 2), with_bias=False)
            with nn.parameter_scope("BatchNormalization"):
                h1 = PF.batch_normalization(h1, batch_stat=False, eps=eps)
            h1 = F.relu(h1)

        #################################################################

        with nn.parameter_scope(f"Conv_4"):
            with nn.parameter_scope("Convolution"):
                h2 = PF.convolution(input_variable, outmaps=64, kernel=(
                    1, 1), pad=(0, 0), stride=(1, 1), with_bias=False)
            with nn.parameter_scope("BatchNormalization"):
                h2 = PF.batch_normalization(h2, batch_stat=False, eps=eps)
            h2 = F.relu(h2)

        with nn.parameter_scope(f"Conv_5"):
            with nn.parameter_scope("Convolution"):
                h2 = PF.convolution(h2, outmaps=96, kernel=(
                    3, 3), pad=(1, 1), stride=(1, 1), with_bias=False)
            with nn.parameter_scope("BatchNormalization"):
                h2 = PF.batch_normalization(h2, batch_stat=False, eps=eps)
            h2 = F.relu(h2)

        with nn.parameter_scope(f"Conv_6"):
            with nn.parameter_scope("Convolution"):
                h2 = PF.convolution(h2, outmaps=96, kernel=(
                    3, 3), pad=(0, 0), stride=(2, 2), with_bias=False)
            with nn.parameter_scope("BatchNormalization"):
                h2 = PF.batch_normalization(h2, batch_stat=False, eps=eps)
            h2 = F.relu(h2)

        #################################################################

        h3 = F.max_pooling(input_variable, kernel=(3, 3),
                           pad=(0, 0), stride=(2, 2))

        h = F.concatenate(*[h1, h2, h3], axis=1)

        return h

    def module_B(input_variable, internal_outmaps=128, eps=1e-3):

        with nn.parameter_scope(f"Conv"):
            with nn.parameter_scope("Convolution"):
                h0 = PF.convolution(input_variable, outmaps=192, kernel=(
                    1, 1), pad=(0, 0), stride=(1, 1), with_bias=False)
            with nn.parameter_scope("BatchNormalization"):
                h0 = PF.batch_normalization(h0, batch_stat=False, eps=eps)
            h0 = F.relu(h0)

        #################################################################

        with nn.parameter_scope(f"Conv_2"):
            with nn.parameter_scope("Convolution"):
                h1 = PF.convolution(input_variable, outmaps=internal_outmaps, kernel=(
                    1, 1), pad=(0, 0), stride=(1, 1), with_bias=False)
            with nn.parameter_scope("BatchNormalization"):
                h1 = PF.batch_normalization(h1, batch_stat=False, eps=eps)
            h1 = F.relu(h1)

        with nn.parameter_scope(f"Conv_8"):
            with nn.parameter_scope("Convolution"):
                h1 = PF.convolution(h1, outmaps=internal_outmaps, kernel=(
                    1, 7), pad=(0, 3), stride=(1, 1), with_bias=False)
            with nn.parameter_scope("BatchNormalization"):
                h1 = PF.batch_normalization(h1, batch_stat=False, eps=eps)
            h1 = F.relu(h1)

        with nn.parameter_scope(f"Conv_3"):
            with nn.parameter_scope("Convolution"):
                h1 = PF.convolution(h1, outmaps=192, kernel=(
                    7, 1), pad=(3, 0), stride=(1, 1), with_bias=False)
            with nn.parameter_scope("BatchNormalization"):
                h1 = PF.batch_normalization(h1, batch_stat=False, eps=eps)
            h1 = F.relu(h1)

        #################################################################

        with nn.parameter_scope(f"Conv_4"):
            with nn.parameter_scope("Convolution"):
                h2 = PF.convolution(input_variable, outmaps=internal_outmaps, kernel=(
                    1, 1), pad=(0, 0), stride=(1, 1), with_bias=False)
            with nn.parameter_scope("BatchNormalization"):
                h2 = PF.batch_normalization(h2, batch_stat=False, eps=eps)
            h2 = F.relu(h2)

        with nn.parameter_scope(f"Conv_9"):
            with nn.parameter_scope("Convolution"):
                h2 = PF.convolution(h2, outmaps=internal_outmaps, kernel=(
                    7, 1), pad=(3, 0), stride=(1, 1), with_bias=False)
            with nn.parameter_scope("BatchNormalization"):
                h2 = PF.batch_normalization(h2, batch_stat=False, eps=eps)
            h2 = F.relu(h2)

        with nn.parameter_scope(f"Conv_10"):
            with nn.parameter_scope("Convolution"):
                h2 = PF.convolution(h2, outmaps=internal_outmaps, kernel=(
                    1, 7), pad=(0, 3), stride=(1, 1), with_bias=False)
            with nn.parameter_scope("BatchNormalization"):
                h2 = PF.batch_normalization(h2, batch_stat=False, eps=eps)
            h2 = F.relu(h2)

        with nn.parameter_scope(f"Conv_5"):
            with nn.parameter_scope("Convolution"):
                h2 = PF.convolution(h2, outmaps=internal_outmaps, kernel=(
                    7, 1), pad=(3, 0), stride=(1, 1), with_bias=False)
            with nn.parameter_scope("BatchNormalization"):
                h2 = PF.batch_normalization(h2, batch_stat=False, eps=eps)
            h2 = F.relu(h2)

        with nn.parameter_scope(f"Conv_6"):
            with nn.parameter_scope("Convolution"):
                h2 = PF.convolution(h2, outmaps=192, kernel=(
                    1, 7), pad=(0, 3), stride=(1, 1), with_bias=False)
            with nn.parameter_scope("BatchNormalization"):
                h2 = PF.batch_normalization(h2, batch_stat=False, eps=eps)
            h2 = F.relu(h2)

        #################################################################

        with nn.parameter_scope(f"Conv_7"):
            h3 = F.average_pooling(input_variable, kernel=(
                3, 3), pad=(1, 1), stride=(1, 1), including_pad=False)
            with nn.parameter_scope("Convolution"):
                h3 = PF.convolution(h3, outmaps=192, kernel=(
                    1, 1), pad=(0, 0), stride=(1, 1), with_bias=False)
            with nn.parameter_scope("BatchNormalization"):
                h3 = PF.batch_normalization(h3, batch_stat=False, eps=eps)
            h3 = F.relu(h3)

        h = F.concatenate(*[h0, h1, h2, h3], axis=1)

        return h

    def grid_size_reduction_B(input_variable, eps=1e-3):

        with nn.parameter_scope(f"Conv_2"):
            with nn.parameter_scope("Convolution"):
                h1 = PF.convolution(input_variable, outmaps=192, kernel=(
                    1, 1), pad=(0, 0), stride=(1, 1), with_bias=False)
            with nn.parameter_scope("BatchNormalization"):
                h1 = PF.batch_normalization(h1, batch_stat=False, eps=eps)
            h1 = F.relu(h1)

        with nn.parameter_scope(f"Conv"):
            with nn.parameter_scope("Convolution"):
                h1 = PF.convolution(h1, outmaps=320, kernel=(
                    3, 3), pad=(0, 0), stride=(2, 2), with_bias=False)
            with nn.parameter_scope("BatchNormalization"):
                h1 = PF.batch_normalization(h1, batch_stat=False, eps=eps)
            h1 = F.relu(h1)

        ###########################################################

        with nn.parameter_scope(f"Conv_4"):
            with nn.parameter_scope("Convolution"):
                h2 = PF.convolution(input_variable, outmaps=192, kernel=(
                    1, 1), pad=(0, 0), stride=(1, 1), with_bias=False)
            with nn.parameter_scope("BatchNormalization"):
                h2 = PF.batch_normalization(h2, batch_stat=False, eps=eps)
            h2 = F.relu(h2)

        with nn.parameter_scope(f"Conv_5"):
            with nn.parameter_scope("Convolution"):
                h2 = PF.convolution(h2, outmaps=192, kernel=(
                    1, 7), pad=(0, 3), stride=(1, 1), with_bias=False)
            with nn.parameter_scope("BatchNormalization"):
                h2 = PF.batch_normalization(h2, batch_stat=False, eps=eps)
            h2 = F.relu(h2)

        with nn.parameter_scope(f"Conv_3"):
            with nn.parameter_scope("Convolution"):
                h2 = PF.convolution(h2, outmaps=192, kernel=(
                    7, 1), pad=(3, 0), stride=(1, 1), with_bias=False)
            with nn.parameter_scope("BatchNormalization"):
                h2 = PF.batch_normalization(h2, batch_stat=False, eps=eps)
            h2 = F.relu(h2)

        with nn.parameter_scope(f"Conv_6"):
            with nn.parameter_scope("Convolution"):
                h2 = PF.convolution(h2, outmaps=192, kernel=(
                    3, 3), pad=(0, 0), stride=(2, 2), with_bias=False)
            with nn.parameter_scope("BatchNormalization"):
                h2 = PF.batch_normalization(h2, batch_stat=False, eps=eps)
            h2 = F.relu(h2)

        #################################################################

        h3 = F.max_pooling(input_variable, kernel=(3, 3),
                           pad=(0, 0), stride=(2, 2))

        h = F.concatenate(*[h1, h2, h3], axis=1)

        return h

    def module_C(input_variable, use_max_pool=False, eps=1e-3):

        with nn.parameter_scope(f"Conv"):
            with nn.parameter_scope("Convolution"):
                h0 = PF.convolution(input_variable, outmaps=320, kernel=(
                    1, 1), pad=(0, 0), stride=(1, 1), with_bias=False)
            with nn.parameter_scope("BatchNormalization"):
                h0 = PF.batch_normalization(h0, batch_stat=False, eps=eps)
            h0 = F.relu(h0)

        #################################################################

        with nn.parameter_scope(f"Conv_2"):
            with nn.parameter_scope("Convolution"):
                h1 = PF.convolution(input_variable, outmaps=384, kernel=(
                    1, 1), pad=(0, 0), stride=(1, 1), with_bias=False)
            with nn.parameter_scope("BatchNormalization"):
                h1 = PF.batch_normalization(h1, batch_stat=False, eps=eps)
            h1 = F.relu(h1)

        with nn.parameter_scope(f"Conv_3"):
            with nn.parameter_scope("Convolution"):
                h11 = PF.convolution(h1, outmaps=384, kernel=(
                    1, 3), pad=(0, 1), stride=(1, 1), with_bias=False)
            with nn.parameter_scope("BatchNormalization"):
                h11 = PF.batch_normalization(h11, batch_stat=False, eps=eps)
            h11 = F.relu(h11)

        with nn.parameter_scope(f"Conv_8"):
            with nn.parameter_scope("Convolution"):
                h12 = PF.convolution(h1, outmaps=384, kernel=(
                    3, 1), pad=(1, 0), stride=(1, 1), with_bias=False)
            with nn.parameter_scope("BatchNormalization"):
                h12 = PF.batch_normalization(h12, batch_stat=False, eps=eps)
            h12 = F.relu(h12)

        #################################################################

        with nn.parameter_scope(f"Conv_4"):
            with nn.parameter_scope("Convolution"):
                h2 = PF.convolution(input_variable, outmaps=448, kernel=(
                    1, 1), pad=(0, 0), stride=(1, 1), with_bias=False)
            with nn.parameter_scope("BatchNormalization"):
                h2 = PF.batch_normalization(h2, batch_stat=False, eps=eps)
            h2 = F.relu(h2)

        with nn.parameter_scope(f"Conv_5"):
            with nn.parameter_scope("Convolution"):
                h2 = PF.convolution(h2, outmaps=384, kernel=(
                    3, 3), pad=(1, 1), stride=(1, 1), with_bias=False)
            with nn.parameter_scope("BatchNormalization"):
                h2 = PF.batch_normalization(h2, batch_stat=False, eps=eps)
            h2 = F.relu(h2)

        with nn.parameter_scope(f"Conv_6"):
            with nn.parameter_scope("Convolution"):
                h21 = PF.convolution(h2, outmaps=384, kernel=(
                    1, 3), pad=(0, 1), stride=(1, 1), with_bias=False)
            with nn.parameter_scope("BatchNormalization"):
                h21 = PF.batch_normalization(h21, batch_stat=False, eps=eps)
            h21 = F.relu(h21)

        with nn.parameter_scope(f"Conv_9"):
            with nn.parameter_scope("Convolution"):
                h22 = PF.convolution(h2, outmaps=384, kernel=(
                    3, 1), pad=(1, 0), stride=(1, 1), with_bias=False)
            with nn.parameter_scope("BatchNormalization"):
                h22 = PF.batch_normalization(h22, batch_stat=False, eps=eps)
            h22 = F.relu(h22)

        #################################################################

        with nn.parameter_scope(f"Conv_7"):
            if use_max_pool:
                h3 = F.max_pooling(input_variable, kernel=(
                    3, 3), stride=(1, 1), pad=(1, 1))
            else:
                h3 = F.average_pooling(input_variable, kernel=(
                    3, 3), pad=(1, 1), stride=(1, 1), including_pad=False)
            with nn.parameter_scope("Convolution"):
                h3 = PF.convolution(h3, outmaps=192, kernel=(
                    1, 1), pad=(0, 0), stride=(1, 1), with_bias=False)
            with nn.parameter_scope("BatchNormalization"):
                h3 = PF.batch_normalization(h3, batch_stat=False, eps=eps)
            h3 = F.relu(h3)

        h = F.concatenate(*[h0, h11, h12, h21, h22, h3], axis=1)

        return h

    with nn.parameter_scope("Conv"):
        conv1 = stem_block(x, outmaps=32, kernel=(3, 3), stride=(2, 2))

    with nn.parameter_scope("Conv_2"):
        conv2 = stem_block(conv1, outmaps=32, kernel=(3, 3), stride=(1, 1))

    with nn.parameter_scope("Conv_3"):
        conv3 = stem_block(conv2, outmaps=64, kernel=(3, 3),
                           pad=(1, 1), stride=(1, 1))
    pool1 = F.max_pooling(conv3, kernel=(3, 3), stride=(2, 2))

    with nn.parameter_scope("Conv_4"):
        conv4 = stem_block(pool1, outmaps=80, kernel=(1, 1), stride=(1, 1))

    with nn.parameter_scope("Conv_5"):
        conv5 = stem_block(conv4, outmaps=192, kernel=(3, 3), stride=(1, 1))
    pool2 = F.max_pooling(conv5, kernel=(3, 3), stride=(2, 2))

    with nn.parameter_scope("Inception"):
        mixed = module_A(pool2, is_first=True)

    with nn.parameter_scope("Inception_2"):
        mixed_1 = module_A(mixed)

    with nn.parameter_scope("Inception_3"):
        mixed_2 = module_A(mixed_1)

    with nn.parameter_scope("Inception_4"):
        mixed_3 = grid_size_reduction_A(mixed_2)

    with nn.parameter_scope("Inception_5"):
        mixed_4 = module_B(mixed_3)

    with nn.parameter_scope("Inception_6"):
        mixed_5 = module_B(mixed_4, internal_outmaps=160)

    with nn.parameter_scope("Inception_7"):
        mixed_6 = module_B(mixed_5, internal_outmaps=160)

    with nn.parameter_scope("Inception_8"):
        mixed_7 = module_B(mixed_6, internal_outmaps=192)

    with nn.parameter_scope("Inception_9"):
        mixed_8 = grid_size_reduction_B(mixed_7)

    with nn.parameter_scope("Inception_10"):
        mixed_9 = module_C(mixed_8)

    with nn.parameter_scope("Inception_11"):
        mixed_10 = module_C(mixed_9, use_max_pool=True)

    if use_up_to == "prepool":
        return mixed_10

    pooled = F.average_pooling(mixed_10, mixed_10.shape[2:])

    if use_up_to == "pool":
        pooled = F.reshape(pooled, pooled.shape[:2])
        return pooled

    with nn.parameter_scope("Affine"):
        # note that this contains bias NOT USED for Inception Score.
        classifier = PF.affine(pooled, 1008)

    return classifier


def main():
    x = nn.Variable((1, 3, 299, 299))
    x.d = np.random.random(x.shape)
    pooled = construct_inceptionv3(x)
    print(pooled.shape)


if __name__ == '__main__':
    main()
