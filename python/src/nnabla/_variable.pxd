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
from libcpp cimport bool as cpp_bool
from libc.stdint cimport int64_t
from libcpp.memory cimport shared_ptr
from _common import *
from _array cimport *
from _context cimport *
from _nd_array cimport *


cdef extern from "nbla/variable.hpp" namespace "nbla":
    cdef cppclass CVariable "nbla::Variable":
        CVariable(Shape_t, cpp_bool) except +
        CVariable(NdArrayPtr, cpp_bool) except +
        cpp_bool need_grad()
        void set_need_grad(cpp_bool)
        Shape_t shape()
        Size_t size(Size_t) except +
        Size_t ndim()
        void reshape(Shape_t, cpp_bool) except +
        shared_ptr[CVariable] view() except +
        shared_ptr[CVariable] view(const Shape_t &) except +
        NdArrayPtr data() except +
        NdArrayPtr grad() except +
        void set_data(NdArrayPtr) except +
        void set_grad(NdArrayPtr) except +
    ctypedef shared_ptr[CVariable] VariablePtr

cdef extern from "nbla/computation_graph/variable.hpp" namespace "nbla":
    cdef cppclass CgFunction
    ctypedef shared_ptr[CgFunction] CgFunctionPtr
    cdef cppclass CgVariable:
        CgVariable(cpp_bool need_grad) except+
        CgVariable(Shape_t shape, cpp_bool need_grad) except+
        CgVariable(VariablePtr)
        void set_parent(CgFunctionPtr func) except+
        CgFunctionPtr parent()
        VariablePtr variable()
        int rank() const
        void set_rank(int rank) except+
        void forward(cpp_bool clear_buffer, cpp_bool clear_no_need_grad) nogil except+
        void backward(NdArrayPtr grad, cpp_bool clear_buffer) nogil except+
        void set_persistent(cpp_bool b)
        cpp_bool persistent()
    ctypedef shared_ptr[CgVariable] CgVariablePtr

cdef extern from "nbla/computation_graph/function.hpp" namespace "nbla":
    cdef cppclass CFunction
    ctypedef shared_ptr[CFunction] FunctionPtr
    cdef cppclass CgFunction:
        CgFunction(FunctionPtr func) except+
        FunctionPtr function() const
        cpp_bool need_grad() const
        cpp_bool update_need_grad() except+
        int rank() const
        void set_outputs(const vector[CgVariablePtr] & outputs) except+
        const vector[CgVariablePtr] inputs()
        vector[CVariable * ] function_inputs() except+
        vector[VariablePtr] function_outputs_shared() except+
        string info() const
        void set_info(const string & info)
    ctypedef shared_ptr[CgFunction] CgFunctionPtr


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
