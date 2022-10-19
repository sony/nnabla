# Copyright 2021 Sony Corporation.
# Copyright 2021 Sony Group Corporation.
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


def no_grad(x):
    return x.apply(need_grad=False)


def positive_axis(axis, D):
    axis = axis if axis > -1 else D + axis
    return axis


def create_slice(shape, axis, front=True):
    start = []
    stop = []
    step = []
    D = len(shape)
    axis = positive_axis(axis, D)
    for i, s in enumerate(shape):
        if i == axis:
            c = s // 2
            if front:
                start.append(0)
                stop.append(c)
            else:
                start.append(c)
                stop.append(s)
        else:
            start.append(0)
            stop.append(s)
        step.append(1)
    return start, stop, step


def force_list(x):
    if isinstance(x, tuple):
        return list(x)
    elif not isinstance(x, list):
        return [x]
    return x


def force_tuple(x):
    if isinstance(x, list):
        return tuple(x)
    elif not isinstance(x, tuple):
        return (x, )
    return x


def sum_for_arithmetics(dx, x):
    # F.{add2, sub2, mul2, div2} includes Broadcast internally (C++-level),
    # so we have to do F.sum explicitly
    axes = [a for a in range(len(x.shape)) if x.shape[a] == 1]
    dx = F.sum(dx, axes, keepdims=True)
    return dx


def sum_for_arithmetics_with_shape(dx, x_shape):
    # F.{add2, sub2, mul2, div2} includes Broadcast internally (C++-level),
    # so we have to do F.sum explicitly
    axes = [a for a in range(len(x_shape)) if x_shape[a] == 1]
    dx = F.sum(dx, axes, keepdims=True)
    return dx
