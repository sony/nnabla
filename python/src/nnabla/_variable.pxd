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

from libcpp.vector cimport vector
from libcpp.string cimport string
from libcpp.unordered_set cimport unordered_set
from libcpp.functional cimport function as std_function
from libcpp cimport bool as cpp_bool
from libc.stdint cimport int64_t
from libcpp.memory cimport shared_ptr
from _common import *
from _array cimport *
from _context cimport *
from _nd_array cimport *


cdef extern from "nbla/variable.hpp" namespace "nbla":
    cdef cppclass CVariable "nbla::Variable":
        CVariable(Shape_t) except +
        CVariable(NdArrayPtr) except +
        Shape_t shape()
        Size_t size(Size_t) except +
        Size_t ndim()
        void reshape(Shape_t, cpp_bool) except +
        shared_ptr[CVariable] view() except +
        shared_ptr[CVariable] view(const Shape_t & ) except +
        NdArrayPtr data() except +
        NdArrayPtr grad() except +
        void set_data(NdArrayPtr) except +
        void set_grad(NdArrayPtr) except +
    ctypedef shared_ptr[CVariable] VariablePtr

cdef extern from "nbla/computation_graph/variable.hpp" namespace "nbla":
    cdef cppclass CgFunction
    ctypedef shared_ptr[CgFunction] CgFunctionPtr
    cdef cppclass CCommunicatorBackwardCallback "nbla::CommunicatorBackwardCallback":
        CCommunicatorBackwardCallback() except+
    ctypedef shared_ptr[CCommunicatorBackwardCallback] CommunicatorBackwardCallbackPtr
    ctypedef std_function[void(const CgFunctionPtr &)] function_hook_type

    cdef cppclass FunctionHookWithObject:
        ctypedef std_function[void(void *)] setup_callback_type
        ctypedef std_function[void(void *)] cleanup_callback_type
        ctypedef std_function[void(void *, const CgFunctionPtr &)] callback_type
        FunctionHookWithObject()
        FunctionHookWithObject(void *obj, callback_type cb,
                               setup_callback_type setup_cb,
                               cleanup_callback_type clean_cb)

    cdef cppclass CgVariable:
        CgVariable() except+
        CgVariable(cpp_bool need_grad) except+
        CgVariable(Shape_t shape) except+
        CgVariable(Shape_t shape, cpp_bool need_grad) except+
        CgVariable(VariablePtr)
        CgVariable(VariablePtr, cpp_bool need_grad)
        cpp_bool need_grad() const
        cpp_bool need_grad_is_set() const
        void set_need_grad(cpp_bool b)
        void unset_need_grad()
        cpp_bool need_grad_state() const
        cpp_bool need_grad_state_is_set() const
        void set_need_grad_state(cpp_bool b)
        void unset_need_grad_state()
        void set_parent(CgFunctionPtr func) except+
        CgFunctionPtr parent()
        VariablePtr variable()
        int rank() const
        void set_rank(int rank) except+
        void forward(cpp_bool clear_buffer, cpp_bool clear_no_need_grad, unordered_set[CgFunctionPtr] *fclosed, function_hook_type function_pre_hook, function_hook_type function_post_hook) nogil except+
        void backward(NdArrayPtr grad, cpp_bool clear_buffer, vector[CommunicatorBackwardCallbackPtr] communicator_callbacks, function_hook_type function_pre_hook, function_hook_type function_post_hook, cpp_bool clear_initial_grad) nogil except+
        void set_persistent(cpp_bool b)
        cpp_bool persistent()
        string name() except +
        void set_name(string name) except +
        vector[CgFunctionPtr] function_references() except+
        void remove_function_reference(CgFunction * func) except+
    ctypedef shared_ptr[CgVariable] CgVariablePtr

cdef extern from "nbla/computation_graph/function.hpp" namespace "nbla":
    cdef cppclass CFunction
    ctypedef shared_ptr[CFunction] FunctionPtr
    cdef cppclass CgFunction:
        CgFunction(FunctionPtr func) except+
        FunctionPtr function() const
        cpp_bool need_grad() const
        int rank() const
        void set_outputs(const vector[CgVariablePtr] & outputs) except+
        const vector[CgVariablePtr] inputs()
        vector[CVariable *] function_inputs() except+
        vector[VariablePtr] function_outputs_shared() except+
        string info() const
        void set_info(const string & info)

cdef class Context:
    cdef vector[string] backend_
    cdef public str array_class
    cdef public str device_id


cdef class CommunicatorBackwardCallback:
    cdef CommunicatorBackwardCallbackPtr var

    @staticmethod
    cdef create_from_ccallback(CommunicatorBackwardCallbackPtr varsp)

cdef class Variable:
    cdef CgVariablePtr var
    cdef CgVariable * varp
    cdef public object info
    """
    Information of the variable.
    """

    @staticmethod
    cdef create_from_cvariable(shared_ptr[CVariable] varsp)

    @staticmethod
    cdef create_from_cg_variable(CgVariablePtr cgv)

cdef FunctionHookWithObject create_function_hook_with_object(object callback)
