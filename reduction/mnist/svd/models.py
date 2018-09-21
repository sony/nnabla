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

from collections import OrderedDict
import os
import time

import nnabla as nn
import nnabla.functions as F
import nnabla.logger as logger
import nnabla.parametric_functions as PF
import nnabla.solvers as S
import nnabla.utils.save as save
import numpy as np
from six.moves import range


def categorical_error(pred, label):
    """
    Compute categorical error given score vectors and labels as
    numpy.ndarray.
    """
    pred_label = pred.argmax(1)
    return (pred_label != label.flat).mean()


def mnist_lenet_prediction(image, scope="reference", test=False):
    """
    Construct LeNet for MNIST.
    """
    with nn.parameter_scope(scope):
        image /= 255.0
        c1 = PF.convolution(image, 16, (5, 5), name='conv1')
        c1 = F.relu(F.max_pooling(c1, (2, 2)), inplace=True)
        c2 = PF.convolution(c1, 16, (5, 5), name='conv2')
        c2 = F.relu(F.max_pooling(c2, (2, 2)), inplace=True)
        c3 = F.relu(PF.affine(c2, 50, name='fc3'), inplace=True)
        c4 = PF.affine(c3, 10, name='fc4')
    return c4


def mnist_lenet_prediction_slim(image, scope="slim", rrate=0.75, test=False):
    """
    Construct LeNet for MNIST.
    """
    with nn.parameter_scope(scope):
        image /= 255.0
        c1 = PF.convolution(image, 16, (5, 5), name='conv1')
        c1 = F.relu(F.max_pooling(c1, (2, 2)), inplace=True)
        c2 = PF.convolution(c1, 16, (5, 5), name='conv2')
        c2 = F.relu(F.max_pooling(c2, (2, 2)), inplace=True)

        # SVD applied
        inmaps = np.prod(c2.shape[1:])  # c * h * w
        outmaps0 = 50  # original outmaps
        outmaps1 = reduce_maps(inmaps, outmaps0, rrate)
        d0 = F.relu(PF.affine(c2, outmaps1, name='fc-d0'), inplace=True)
        d1 = F.relu(PF.affine(d0, outmaps0, name='fc-d1'), inplace=True)

        c4 = PF.affine(d1, 10, name='fc4')
    return c4


def reduce_maps(inmaps, outmaps, rrate):
    maps = int(rrate * inmaps * outmaps / (inmaps + outmaps))
    logger.info("###################")
    logger.info(
        "Num.Parameters is reduced {}x{} -> {}x{} + {}x{} by {}".format(
            inmaps, outmaps, inmaps, maps, maps, outmaps, rrate))
    logger.info("###################")
    time.sleep(1)
    return maps


def decompose_network_and_set_params(model_load_path,
                                     reference, slim, rrate=0.75):
    # Parameters loaded globally, but call here for consistency
    nn.load_parameters(model_load_path)

    # Decompose
    with nn.parameter_scope(reference):
        trained_params = nn.get_parameters()
    # original parameter
    W = trained_params["fc3/affine/W"].d
    # original maps
    inmaps = W.shape[0]
    outmaps0 = W.shape[1]
    # new maps, R < N*M / (N+M) * rrate
    outmaps1 = reduce_maps(inmaps, outmaps0, rrate)
    # singular value decomposition
    U, s, V = np.linalg.svd(W, full_matrices=False)
    S = np.diag(s)
    SV = S.dot(V)
    U_approx = U[:, :outmaps1]
    SV_approx = SV[:outmaps1, :outmaps0]

    # Set trained parameters and decomposed parameters
    # set trained parameters
    with nn.parameter_scope(slim):
        slim_params = nn.get_parameters()
    for n, v in trained_params.items():
        if not n in slim_params.keys():
            continue
        v_slim = slim_params[n]
        v_slim.d = v.d
    # set decomposed parameters and original bias
    # a new bias is introduced due to decomposition
    slim_params["fc-d0/affine/W"].d = U_approx
    slim_params["fc-d1/affine/W"].d = SV_approx
    b = trained_params["fc3/affine/b"]
    slim_params["fc-d1/affine/b"].d = b.d

    # Clear the parameters of the reference net
    with nn.parameter_scope(reference):
        nn.clear_parameters()
