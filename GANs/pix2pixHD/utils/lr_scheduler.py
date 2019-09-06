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


class LearningRateScheduler(object):
    """Learning rate decay scheduler.

    """

    def __init__(self, base_lr, training_epoch=200, decay_at=100):
        self.base_learning_rate = base_lr
        self.decay_at = decay_at
        self.traing_epoch = training_epoch

    def __call__(self, epoch):
        if epoch < self.decay_at:
            return self.base_learning_rate

        return self.base_learning_rate * (self.traing_epoch - epoch) / (self.traing_epoch - self.decay_at)
