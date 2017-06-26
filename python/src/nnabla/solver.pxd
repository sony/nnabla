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
from libcpp.pair cimport pair
from libcpp.string cimport string
from libcpp.memory cimport shared_ptr
from libcpp cimport bool as cpp_bool
cimport _variable
from _variable cimport CVariable, CContext, dtypes


cdef extern from "nbla/solver.hpp" namespace "nbla":
    cdef cppclass CSolver "nbla::Solver":
        void zero_grad() except +
        void set_parameters(const vector[pair[string, shared_ptr[CVariable]]] & params,
                            cpp_bool reset, cpp_bool retain_state) except +
        void remove_parameters(const vector[string] & keys) except +
        void clear_parameters() except +
        void update() except +
        void weight_decay(float decay_rate) except +
        string name() except +
        float learning_rate() except +
        void set_learning_rate(float learning_rate) except +

cdef extern from "nbla/solver/adadelta.hpp" namespace "nbla":
    shared_ptr[CSolver] create_AdadeltaSolver(
        const CContext &, float lr, float decay, float eps) except +

cdef extern from "nbla/solver/adagrad.hpp" namespace "nbla":
    shared_ptr[CSolver] create_AdagradSolver(
        const CContext &, float lr, float eps) except +

cdef extern from "nbla/solver/adam.hpp" namespace "nbla":
    shared_ptr[CSolver] create_AdamSolver(
        const CContext &, float alpha, float beta1, float beta2,
        float eps) except +

cdef extern from "nbla/solver/adamax.hpp" namespace "nbla":
    shared_ptr[CSolver] create_AdamaxSolver(
        const CContext &, float alpha, float beta1, float beta2,
        float eps) except +

cdef extern from "nbla/solver/momentum.hpp" namespace "nbla":
    shared_ptr[CSolver] create_MomentumSolver(const CContext &, float lr, float momentum) except +

cdef extern from "nbla/solver/nesterov.hpp" namespace "nbla":
    shared_ptr[CSolver] create_NesterovSolver(const CContext &, float lr, float momentum) except +

cdef extern from "nbla/solver/rmsprop.hpp" namespace "nbla":
    shared_ptr[CSolver] create_RMSpropSolver(
        const CContext &, float lr, float decay, float eps) except +

cdef extern from "nbla/solver/sgd.hpp" namespace "nbla":
    shared_ptr[CSolver] create_SgdSolver(const CContext &, float lr) except +

cdef class Solver:

    cdef shared_ptr[CSolver] solver
    cdef CSolver * solverp
    cdef public object info

    @staticmethod
    cdef create(shared_ptr[CSolver] solver, info)
