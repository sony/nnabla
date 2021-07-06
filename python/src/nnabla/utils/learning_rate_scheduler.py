# Copyright 2018,2019,2020,2021 Sony Corporation.
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

'''
Learning Rate Scheduler is module for scheduling learning rate during training process
'''

import math


class BaseLearningRateScheduler(object):
    '''BaseLearningRateScheduler
    Base class of the learning rate scheduler.
    Decide learning rate based on current iteration.

    '''

    def get_learning_rate(self, iter):
        '''
        Get learning rate based on current iteration.

        Args:
            iter (int): Current iteration (starting with 0).

        Returns:
            float: Learning rate
        '''
        raise NotImplementedError()


class PolynomialScheduler(BaseLearningRateScheduler):
    '''PolynomialScheduler
    Polynominal decay

    '''

    def __init__(self, init_lr, max_iter, power):
        '''
        Args:
            init_lr (float): Initial learning rate.
            max_iter (int): Max iteration.
            power (float): Polymomial power.
        '''
        self.init_lr = init_lr
        self.max_iter = max_iter
        self.power = power

    def get_learning_rate(self, iter):
        '''
        Get learning rate with polymomial decay based on current iteration.

        Args:
            iter (int): current iteration (starting with 0).

        Returns:
            float: Learning rate
        '''
        return self.init_lr * ((1.0 - iter * 1.0 / self.max_iter) ** self.power)


class CosineScheduler(BaseLearningRateScheduler):
    '''CosineScheduler
    Cosine decay

    '''

    def __init__(self, init_lr, max_iter):
        '''
        Args:
            init_lr (float): Initial learning rate.
            max_iter (int): Max iteration.
        '''
        self.init_lr = init_lr
        self.max_iter = max_iter

    def get_learning_rate(self, iter):
        '''
        Get learning rate with cosine decay based on current iteration.

        Args:
            iter (int): Current iteration (starting with 0).

        Returns:
            float: Learning rate
        '''
        return self.init_lr * ((math.cos(iter * 1.0 / (self.max_iter) * math.pi) + 1.0) * 0.5)


class ExponentialScheduler(BaseLearningRateScheduler):
    '''ExponentialScheduler
    Exponential decay

    '''

    def __init__(self, init_lr, gamma, iter_interval=1):
        '''
        Args:
            init_lr (float): Initial learning rate.
            gamma (float): Multiplier.
            iter_interval (int): Period of iteration at which gamma is multiplied.
                The default is 1.
        '''
        self.init_lr = init_lr
        self.gamma = gamma
        self.iter_interval = iter_interval

    def get_learning_rate(self, iter):
        '''
        Get learning rate with exponential decay based on current iteration.

        Args:
            iter (int): Current iteration (starting with 0).

        Returns:
            float: Learning rate
        '''

        return self.init_lr * (self.gamma ** (iter // self.iter_interval))


class StepScheduler(BaseLearningRateScheduler):
    '''StepScheduler
    Step decay

    '''

    def __init__(self, init_lr, gamma, iter_steps):
        '''
        Args:
            init_lr (float): Initial learning rate.
            gamma (float): Multiplier.
            iter_steps (list of int): Iteration at which gamma is multiplied.

        Example:
            StepScheduler(0.1, 0.1, [150000, 300000, 400000])
        '''
        self.init_lr = init_lr
        self.gamma = gamma
        self.iter_steps = iter_steps

    def get_learning_rate(self, iter):
        '''
        Get learning rate with exponential decay based on current iteration.

        Args:
            iter (int): Current iteration (starting with 0).

        Returns:
            float: Learning rate
        '''
        lr = self.init_lr
        for iter_step in self.iter_steps:
            if iter >= iter_step:
                lr *= self.gamma
        return lr


class LinearWarmupScheduler(BaseLearningRateScheduler):
    '''Linear Warm-up Scheduler
    Linear warm-up

    '''

    def __init__(self, scheduler, warmup_iter):
        '''
        Args:
            scheduler (BaseLearningRateScheduler): learning_rate_scheduler
            warmup_iter (int): Iteration for warm-up.

        Example:
            LinearWarmupScheduler(scheduler, 25000)
        '''
        self.scheduler = scheduler
        self.warmup_iter = warmup_iter

    def get_learning_rate(self, iter):
        '''
        Get learning rate with exponential decay based on current iteration.

        Args:
            iter (int): Current iteration (starting with 0).

        Returns:
            float: Learning rate
        '''
        lr = self.scheduler.get_learning_rate(iter)
        if iter < self.warmup_iter:
            lr *= (iter + 1) * 1.0 / self.warmup_iter
        return lr
