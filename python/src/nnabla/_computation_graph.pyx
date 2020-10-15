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

from libcpp cimport bool as cpp_bool
from libcpp.vector cimport vector
from _variable cimport Variable as _Variable, create_function_hook_with_object
from _computation_graph cimport forward_all as cforward_all


def forward_all(variables,
                cpp_bool clear_buffer=False,
                cpp_bool clear_no_need_grad=False,
                function_pre_hook=None, function_post_hook=None):
    '''Performs a forward propagation up to variables specified as the 1st
    argument.
    See also :obj:`~nnnabla.Variable.forward`.


    Args:
            clear_buffer (bool): Clear the no longer referenced variables
                during forward propagation to save memory.
                This is usually set as True in an inference
                or a validation phase. Default is False.
                Note that starting variable and destination variable of the input graph will not be cleared, regardless of their `persistent` flag.
                All intermediate variables will be cleared unless set explicitly as `persistent=True`.
                For example, 

                .. code-block:: python

                   forward_all([h_i, y], clear_buffer=True)

                will clear all intermediate variables between `h_i` and `y` unless set explicitly as `persistent=True`, but `h_i` and `y` will not be cleared regardless of their `persistent` flag.
            clear_no_need_grad (bool): Clear the unreferenced variables with
                need_grad=False during forward propagation.
                True is usually used when calling this during training.
                This is ignored when clear_buffer=True.
            function_pre_hook(callable):
                This callable object is called immediately before each function is executed.
                It must take :obj:`~nnabla.function.Function` as an input.
                The default is None.
            function_post_hook(callable):
                This callable object is called immediately after each function is executed.
                It must take :obj:`~nnabla.function.Function` as an input.
                The default is None.

    Example:

        .. code-block:: python

            import numpy as np
            import nnabla as nn
            import nnabla.parametric_functions as PF

            # Create a graph which has two outputs
            x = nn.Variable.from_numpy_array(np.array([[1, 2], [3, 4]]))
            y = PF.affine(x, 4, name="y")
            z = PF.affine(x, 8, name="z")

            # Execute a forward propagation recursively up to y and z
            nn.forward_all([y, z], clear_buffer)

    '''
    cdef vector[CgVariablePtr] cg_variables
    cdef int i
    cdef int size
    cdef function_hook_type function_pre_hook_c
    cdef function_hook_type function_post_hook_c
    if function_pre_hook is not None:
        function_pre_hook_c = create_function_hook_with_object(function_pre_hook)
    if function_post_hook is not None:
        function_post_hook_c = create_function_hook_with_object(function_post_hook)
    size = len(variables)
    cg_variables.resize(size)
    for i in range(size):
        cg_variables[i] = (<_Variable?> variables[i]).var
    with nogil:
        cforward_all(cg_variables, clear_buffer, clear_no_need_grad, function_pre_hook_c, function_post_hook_c)
