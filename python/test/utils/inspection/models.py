# Copyright (c) 2019 Sony Corporation. All Rights Reserved.
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


import nnabla.functions as F
import nnabla.parametric_functions as PF


def simple_cnn(x, t, n_class):
    c1 = PF.convolution(x, 16, (5, 5), name='conv1')
    c1 = F.relu(F.max_pooling(c1, (2, 2)))
    c2 = PF.convolution(c1, 8, (5, 5), name='conv2')
    c2 = F.relu(F.max_pooling(c2, (2, 2)))
    c3 = F.relu(PF.affine(c2, 10, name='fc3'))
    c4 = PF.affine(c3, n_class, name='fc4')
    l = F.mean(F.softmax_cross_entropy(c4, t))

    return l
