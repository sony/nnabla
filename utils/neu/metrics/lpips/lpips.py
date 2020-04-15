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


import os
import nnabla as nn
import nnabla.functions as F
import nnabla.parametric_functions as PF


def unit_normalize(in_feat, eps=1e-10):
    """
        eps seems too small, but actually is the default value.
    """
    norm_factor = F.pow_scalar(F.sum(in_feat**2, axis=1, keepdims=True), 0.5)
    return in_feat / (norm_factor + eps)


def get_alex_feat(input_var):
    """
        Exactly the same architecture used for LPIPS.
        This is a little bit modified version (can't use nnabla models!). 
    """
    assert input_var.shape[1] == 3
    act1 = F.relu(PF.convolution(input_var, outmaps=64, kernel=(
        11, 11), pad=(2, 2), stride=(4, 4), name="conv0"), True)
    act2 = F.relu(PF.convolution(F.max_pooling(act1, kernel=(3, 3), stride=(
        2, 2)), outmaps=192, kernel=(5, 5), pad=(2, 2), name="conv3"), True)
    act3 = F.relu(PF.convolution(F.max_pooling(act2, kernel=(3, 3), stride=(
        2, 2)), outmaps=384, kernel=(3, 3), pad=(1, 1), name="conv6"), True)
    act4 = F.relu(PF.convolution(act3, outmaps=256, kernel=(
        3, 3), pad=(1, 1), name="conv8"), True)
    act5 = F.relu(PF.convolution(act4, outmaps=256, kernel=(
        3, 3), pad=(1, 1), name="conv10"), True)
    return [act1, act2, act3, act4, act5]


def get_vgg_feat(input_var):
    """
        Exactly the same architecture used for LPIPS.
    """
    assert input_var.shape[1] == 3
    act1 = F.relu(PF.convolution(input_var, outmaps=64,
                                 kernel=(3, 3), pad=(1, 1), name="conv0"), True)
    act1 = F.relu(PF.convolution(act1, outmaps=64, kernel=(
        3, 3), pad=(1, 1), name="conv2"), True)

    act2 = F.max_pooling(act1, kernel=(2, 2), stride=(2, 2))
    act2 = F.relu(PF.convolution(act2, outmaps=128, kernel=(
        3, 3), pad=(1, 1), name="conv5"), True)
    act2 = F.relu(PF.convolution(act2, outmaps=128, kernel=(
        3, 3), pad=(1, 1), name="conv7"), True)

    act3 = F.max_pooling(act2, kernel=(2, 2), stride=(2, 2))
    act3 = F.relu(PF.convolution(act3, outmaps=256, kernel=(
        3, 3), pad=(1, 1), name="conv10"), True)
    act3 = F.relu(PF.convolution(act3, outmaps=256, kernel=(
        3, 3), pad=(1, 1), name="conv12"), True)
    act3 = F.relu(PF.convolution(act3, outmaps=256, kernel=(
        3, 3), pad=(1, 1), name="conv14"), True)

    act4 = F.max_pooling(act3, kernel=(2, 2), stride=(2, 2))
    act4 = F.relu(PF.convolution(act4, outmaps=512, kernel=(
        3, 3), pad=(1, 1), name="conv17"), True)
    act4 = F.relu(PF.convolution(act4, outmaps=512, kernel=(
        3, 3), pad=(1, 1), name="conv19"), True)
    act4 = F.relu(PF.convolution(act4, outmaps=512, kernel=(
        3, 3), pad=(1, 1), name="conv21"), True)

    act5 = F.max_pooling(act4, kernel=(2, 2), stride=(2, 2))
    act5 = F.relu(PF.convolution(act5, outmaps=512, kernel=(
        3, 3), pad=(1, 1), name="conv24"), True)
    act5 = F.relu(PF.convolution(act5, outmaps=512, kernel=(
        3, 3), pad=(1, 1), name="conv26"), True)
    act5 = F.relu(PF.convolution(act5, outmaps=512, kernel=(
        3, 3), pad=(1, 1), name="conv28"), True)
    act5 = F.max_pooling(act5, kernel=(2, 2), stride=(2, 2))

    return [act1, act2, act3, act4, act5]


def compute_each_feat_dist(img0, img1, feat_extractor):
    """
        img0, img1(Variable): shape of (N, 3, H, W). Value ranges should be in [-1., +1.].
        feat_extrator(function): backbone network for getting features.
    """

    img0_feats = feat_extractor(img0)  # lists of Variables.
    img1_feats = feat_extractor(img1)  # each Variable is the activation.

    dists = list()

    for i, (feat0, feat1) in enumerate(zip(img0_feats, img1_feats)):
        feat0 = unit_normalize(feat0)  # normalize.
        feat1 = unit_normalize(feat1)
        # retrieve LPIPS weight.
        lpips_w = nn.parameter.get_parameter_or_create(
                f'lin{i}_model_1_weight', shape=(1, feat0.shape[1], 1, 1))
        # in the paper, it is described as multiplication,
        # but implemented as 1x1 convolution.
        dist = F.convolution(F.pow_scalar((feat0 - feat1), 2), lpips_w)
        dists.append(dist)  # store distrance at each layer.

    return dists


def load_parameters(params_path):
    if not os.path.isfile(params_path):
        from nnabla.utils.download import download
        url = os.path.join("https://nnabla.org/pretrained-models/nnabla-examples/eval_metrics/lpips",
                           params_path.split("/")[-1])
        download(url, params_path, False)
    nn.load_parameters(params_path)


class LPIPS(object):
    """
        LPIPS implementation. this uses the version "0.1".
    """

    def __init__(self, model="alex", params_dir="./converted_weights", spatial=False, apply_scale=True):
        """
            Args:
               model(str): network used for feature extractor. "alex"(AlexNet) or "vgg"(VGG16).
                      AlexNet is reported to work best.
               params_dir(str): directory containing the weights.
                           Note that the weights must be the same as the one used for the paper.
               spatial(bool): if True, returns the distance map instead of single value. False by default.
               apply_scale(bool): if True, the input values will be scaled. True by default.
                            using version 0.1 requires this scaling.
        """
        super(LPIPS, self).__init__()

        self.model = model
        params_path = os.path.join(params_dir, f"{model}_with_LPIPS.h5")
        if self.model == "alex":
            print("Use AlexNet's features")
            load_parameters(params_path)
            self.feat_extractor = get_alex_feat

        elif self.model == "vgg":
            print("Use VGG's features")
            load_parameters(params_path)
            self.feat_extractor = get_vgg_feat

        else:
            # currently only vgg and alexnet are supported.
            return NotImplementedError

        self.spatial = spatial
        self.apply_scale = apply_scale
        self._shift = F.reshape(nn.Variable.from_numpy_array(
            [-.030, -.088, -.188]), (1, 3, 1, 1))
        self._scale = F.reshape(nn.Variable.from_numpy_array(
            [.458, .448, .450]), (1, 3, 1, 1))

    def __call__(self, img0, img1, normalize=False, mean_batch=False):
        """
            Args:
               img0, img1(Variable): Variable containing images. N batch images can be used. 
               normalize(bool): if True, assumes inputs are in [0., 1.] and scales the inputs between [-1., +1.].
                                if False, assumes inputs are in [-1., +1.]
        """
        assert img0.shape == img1.shape, "img0 and img1 have different shape."
        assert isinstance(img0, nn.Variable), "img0 is not Variable."
        assert isinstance(img1, nn.Variable), "img1 is not Variable."

        if normalize:
            # scales the input between [-1., +1.]
            img0 = 2*img0 - 1
            img1 = 2*img1 - 1

        if self.apply_scale:
            img0 = (img0 - self._shift) / self._scale
            img1 = (img1 - self._shift) / self._scale

        dists = compute_each_feat_dist(
            img0, img1, feat_extractor=self.feat_extractor)

        if self.spatial:
            # note that this upsampling method is different from the original LPIPS.
            # in the original implementation, it is torch.nn.upsample(mode="bilinear")
            dists = [F.interpolate(
                        dist*(1.*img0.shape[2]/dist.shape[2]), output_size=img0.shape[2:]) for dist in dists]
        else:
            dists = [F.mean(dist, axis=[2, 3], keepdims=True)
                     for dist in dists]
        # returns N scores ((N, 1, 1, 1))
        lpips_val = F.sum(F.stack(*dists), axis=0)

        if mean_batch:
            lpips_val = F.mean(lpips_val, axis=0)

        return lpips_val
