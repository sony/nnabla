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


import numpy as np


class LinearDecayScheduler(object):
    """
    Linear decaying learning rate decay scheduler.

          | start_lr  iter <= start
    lr =  | end_lr    iter >= end
          | start_lr + (end_lr - start_lr) * (iter - start) / (end - start)  otherwise

    `start_lr` stands for the learning rate before decaying, and decaying starts at `start_iter` iteration.
    `end_lr` stands for the final learning rate in the end of decaying, and it is reached at `end_iter` iteration.

    """

    def __init__(self, start_lr, end_lr, start_iter, end_iter):
        self.start_lr = start_lr
        self.end_lr = end_lr
        self.start = start_iter
        self.end = end_iter

    def __call__(self, iter):
        if iter <= self.start:
            return self.start_lr

        if iter >= self.end:
            return self.end_lr

        return self.start_lr + (self.end_lr - self.start_lr) * (iter - self.start) / (self.end - self.start)
