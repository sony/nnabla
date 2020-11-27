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

cimport callback
from _variable cimport function_hook_type, create_function_hook_with_object
from solver cimport update_hook_type, create_update_hook_with_object


def set_function_pre_hook(string key, object function_pre_hook):
    """
    Set function_pre_hook globally with a key as an callback identifier.
    All callbacks registered through this API will be called just before performing forward / backward of all functions.

    Args:
        key (string): A name of callback which identifies registered callback. This can be used when deleting callback.
        function_pre_hook (obj): Callable object to be registered which takes a function as a argument.

    Example:

        .. code-block:: python

            from nnabla import set_function_pre_hook

            def callback(f):
                print(f)

            # callback() is executed right before every function during fwd/bwd
            set_function_pre_hook("print_function_name", callback)

            loss = graph(...)

            # Names of all functions will be printed out in stdout during fwd/bwd
            loss.forward()
            loss.backward()
    """
    if function_pre_hook is None:
        return

    cdef function_hook_type function_hook_c

    function_hook_c = create_function_hook_with_object(function_pre_hook)

    with nogil:
        callback.c_set_function_pre_hook(key, function_hook_c)


def set_function_post_hook(string key, object function_post_hook):
    """
    Set function_post_hook globally with a key as an callback identifier.
    All callbacks registered through this API will be called just after performing forward / backward of all functions.

    Args:
        key (string): A name of callback which identifies registered callback. This can be used when deleting callback.
        function_pre_hook (obj): Callable object to be registered which takes a function as a argument.

    Example:

        .. code-block:: python

            from nnabla import set_function_post_hook

            def callback(f):
                print(f)

            # callback() is executed right after every function during fwd/bwd
            set_function_post_hook("print_function_name", callback)

            loss = graph(...)

            # Names of all functions will be printed out in stdout during fwd/bwd
            loss.forward()
            loss.backward()
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
        key (string): A name of callback which identifies registered callback. The callback whose name equals to `key` is deleted.
    
    Example:

        .. code-block:: python

            from nnabla import set_function_pre_hook, unset_function_pre_hook

            def callback(f):
                print(f)

            # callback() is executed right after every function during fwd/bwd
            set_function_pre_hook("print_function_name", callback)

            loss = graph(...)

            # Names of all functions will be printed out in stdout during fwd/bwd
            loss.forward()
            loss.backward()

            # Unset callback()
            unset_function_pre_hook("print_function_name")

            # Nothing will be shown.
            loss.forward()
            loss.backward()
    
    """
    with nogil:
        callback.c_unset_function_pre_hook(key)


def unset_function_post_hook(string key):
    """
    Unset function_post_hook which was previously set as global callback through set_function_post_hook.

    Args:
        key (string): A name of callback which identifies registered callback. The callback whose name equals to `key` is deleted.
    
    Example:

        .. code-block:: python

            from nnabla import set_function_post_hook, unset_function_post_hook

            def callback(f):
                print(f)

            # callback() is executed right after every function during fwd/bwd
            set_function_post_hook("print_function_name", callback)

            loss = graph(...)

            # Names of all functions will be printed out in stdout during fwd/bwd
            loss.forward()
            loss.backward()

            # Unset callback()
            unset_function_post_hook("print_function_name")

            # Nothing will be shown.
            loss.forward()
            loss.backward()
    
    """
    with nogil:
        callback.c_unset_function_post_hook(key)


def set_solver_pre_hook(string key, object solver_pre_hook):
    """
    Set solver_pre_hook globally with key as a callback identifier.
    All callbacks registered through this API will be called right before performing solver functions,
     e.g. update, weight_decay, clip_grad_by_norm, and so on.
    The registerd callbacks are performed sequentially in order of registration before processing each parameter.

    Args:
        key (string): A callback identifier. This can be used when deleting callback.
        solver_pre_hook (obj): Callable object which takes no arguments.
    """
    if solver_pre_hook is None:
        return

    cdef update_hook_type solver_hook_c

    solver_hook_c = create_update_hook_with_object(solver_pre_hook)

    with nogil:
        callback.c_set_solver_pre_hook(key, solver_hook_c)


def set_solver_post_hook(string key, object solver_post_hook):
    """
    Set solver_post_hook globally with key as a callback identifier.
    All callbacks registered through this API will be called right before performing solver functions,
     e.g. update, weight_decay, clip_grad_by_norm, and so on.
    The registerd callbacks are performed sequentially in order of registration before processing each parameter.

    Args:
        key (string): A callback identifier. This can be used when deleting callback.
        solver_pre_hook (obj): Callable object which takes no arguments.
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
