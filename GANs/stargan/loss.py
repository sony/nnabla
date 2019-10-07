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


def recon_loss(x, y):
    return F.mean(F.absolute_error(x, y))


def classification_loss(x, label):
    return F.sum(F.sigmoid_cross_entropy(x, label)) / x.shape[0]


def gan_loss(feat):
    # Utilize WGAN loss.
    return F.mean(feat)
