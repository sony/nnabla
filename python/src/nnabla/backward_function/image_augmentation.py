# Copyright (c) 2017 Sony Corporation. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import nnabla.functions as F


def image_augmentation_backward(inputs, shape=None, pad=(0, 0), min_scale=1.0, max_scale=1.0, angle=0.0, aspect_ratio=1.0, distortion=0.0, flip_lr=False, flip_ud=False, brightness=0.0, brightness_each=False, contrast=1.0, contrast_center=0.0, contrast_each=False, noise=0.0, seed=-1):
    """
    Args:
      inputs (list of nn.Variable): Incomming grads/inputs to/of the forward function.
      kwargs (dict of arguments): Dictionary of the corresponding function arguments.

    Return:
      list of Variable: Return the gradients wrt inputs of the corresponding function.
    """
    dy = inputs[0]
    x0 = inputs[1]
    raise NotImplementedError(
        "image_augmentation_backward is not implemented.")
