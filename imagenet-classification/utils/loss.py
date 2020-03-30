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


import nnabla.functions as F


def apply_label_smoothing(xent, pred, label_smoothing, logp=None):
    if label_smoothing <= 0:
        return xent
    if logp is None:
        logp = F.log_softmax(pred)
    return (1 - label_smoothing) * xent - label_smoothing * F.mean(logp, axis=1, keepdims=True)


def softmax_cross_entropy_with_label_smoothing(pred, label, label_smoothing=0.1):
    '''
    Defines softmax activation followed by Cross entropy loss and label smoothing.


    Label smoothing loss is added by the following weight:
    `(1 - label_smoothing) * xent_loss + label_smoothing * label_smoothing_loss`

    Args:
        pred (Variable): Logits with a shape of `(batch_size, num_classes)`.
        label (Variable):
            A class index for each example if a shape of `(batch_size, 1`) is
            given, and a one-hot or probability over classes if
            `(batch_size, num_classes)`.
        label_smoothing (float):
            Coefficient of label smoothing loss. If 0, it omits label
            smoothing.


    '''
    logp = None
    if label.shape[1] > 1:
        # If mixup is enabled, we suppose the label shape is (batch_size, num_class)
        logp = F.log_softmax(pred)
        l = F.sum(-label * logp, axis=1, keepdims=True)
    else:
        l = F.softmax_cross_entropy(pred, label)
    return apply_label_smoothing(l, pred, label_smoothing, logp)
