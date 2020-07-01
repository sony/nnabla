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

cimport callback
from _variable cimport function_hook_type, create_function_hook_with_object
from solver cimport update_hook_type, create_update_hook_with_object


def set_function_pre_hook(string key, object function_pre_hook):
    """
    Set function_pre_hook globally with key as an callback identifier.
    All callbacks registered through this API will be called just before performing forward / backward of all functions.

    Args:
        key (string): A callback identifier. This can be used when deleting callback.
        function_pre_hook (obj): Callable object.
    """
    if function_pre_hook is None:
        return

    cdef function_hook_type function_hook_c

    function_hook_c = create_function_hook_with_object(function_pre_hook)

    with nogil:
        callback.c_set_function_pre_hook(key, function_hook_c)


def set_function_post_hook(string key, object function_post_hook):
    """
    Set function_pre_hook globally with key as an callback identifier.
    All callbacks registered through this API will be called just after performing forward / backward of all functions.

    Args:
        key (string): A callback identifier. This can be used when deleting callback.
        function_post_hook (obj): Callable object.
    """
    if function_post_hook is None:
        return

    cdef function_hook_type function_hook_c

    function_hook_c = create_function_hook_with_object(function_post_hook)

    with nogil:
        callback.c_set_function_post_hook(key, function_hook_c)


def unset_function_pre_hook(string key):
    """
    Unset function_pre_hook which was previously set as global callback through set_function_pre_hook.

    Args:
        key (string): A callback identifier. Delete a callback whose key equals to this.
    """
    with nogil:
        callback.c_unset_function_pre_hook(key)


def unset_function_post_hook(string key):
    """
    Unset function_post_hook which was previously set as global callback through set_function_post_hook.

    Args:
        key (string): A callback identifier. Delete a callback whose key equals to this.
    """
    with nogil:
        callback.c_unset_function_post_hook(key)


def set_solver_pre_hook(string key, object solver_pre_hook):
    """
    Set solver_pre_hook globally with key as an callback identifier.
    All callbacks registered through this API will be called just before performing solver functions (e.g. update, weight_decay, clip_grad_by_norm, ...).
    In general, these solver functions are performed on every parameters sequentially,
     and registered callbacks are called at every parameters as well.

    Args:
        key (string): A callback identifier. This can be used when deleting callback.
        solver_pre_hook (obj): Callable object.
    """
    if solver_pre_hook is None:
        return

    cdef update_hook_type solver_hook_c

    solver_hook_c = create_update_hook_with_object(solver_pre_hook)

    with nogil:
        callback.c_set_solver_pre_hook(key, solver_hook_c)


def set_solver_post_hook(string key, object solver_post_hook):
    """
    Set solver_post_hook globally with key as an callback identifier.
    All callbacks registered through this API will be called just before performing solver functions (e.g. update, weight_decay, clip_grad_by_norm, ...).
    In general, these solver functions are performed on every parameters sequentially,
     and registered callbacks are called at every parameters as well.

    Args:
        key (string): A callback identifier. This can be used when deleting callback.
        solver_post_hook (obj): Callable object.
    """
    if solver_post_hook is None:
        return

    cdef update_hook_type solver_hook_c

    solver_hook_c = create_update_hook_with_object(solver_post_hook)

    with nogil:
        callback.c_set_solver_post_hook(key, solver_hook_c)


def unset_solver_pre_hook(string key):
    """
    Unset solver_pre_hook which was previously set as global callback through set_solver_pre_hook.

    Args:
        key (string): A callback identifier. Delete a callback whose key equals to this.
    """
    with nogil:
        callback.c_unset_solver_pre_hook(key)


def unset_solver_post_hook(string key):
    """
    Unset solver_post_hook which was previously set as global callback through set_solver_post_hook.

    Args:
        key (string): A callback identifier. Delete a callback whose key equals to this.
    """
    with nogil:
        callback.c_unset_solver_post_hook(key)
