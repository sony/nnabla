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

from six import iteritems

from libcpp.vector cimport vector
from libcpp.pair cimport pair
from libcpp.string cimport string
from libcpp.unordered_map cimport unordered_map
from libcpp.memory cimport shared_ptr
from libcpp cimport bool as cpp_bool
from libc.stdint cimport int64_t
from nnabla import _nd_array

cimport communicator
from communicator cimport CCommunicator

cimport _variable
from _variable cimport Variable as _Variable, CVariable
from _variable import Context

cimport _nd_array
from _nd_array cimport NdArray


# Numpy
import numpy as np
cimport numpy as np
np.import_array()

cdef class Communicator:
    """Communicator interface class.

    Communicator exchanges data (e.g., gradient) using MPI-like 
    collectives. This class is used for the distributed training.

    """

    @staticmethod
    cdef create(shared_ptr[CCommunicator] communicator):
        c = Communicator()
        c.communicator = communicator
        c.communicatorp = communicator.get()
        return c

    @property
    def name(self):
        """
        Get communicator name.
        """
        return self.communicatorp.name()

    @property
    def size(self):
        """
        Get size of communicator.
        """
        return self.communicatorp.size()

    @property
    def rank(self):
        """
        Get rank of communicator.
        """
        return self.communicatorp.rank()

    @property
    def local_rank(self):
        """
        Get local rank of communicator.
        """
        return self.communicatorp.local_rank()

    def add_context_and_parameters(self, ctx_param_dict):
        """Add context and parameters.

        Args: 
            ctx_param_dict (:obj:`tuple` of :obj:`Context`, :obj:`dict`): 
                Key of the dictionary is :obj:`string` and value of the
                dictionry is :obj:`Varible`.  
        """
        if type(ctx_param_dict) != tuple:
            raise Exception("ctx_param_dict must be tuple of two elements")
        if len(ctx_param_dict) != 2:
            raise Exception("ctx_param_dict must be tuple of two elements")

        cdef vector[pair[string, shared_ptr[CVariable]]] cparams
        cdef _Variable x
        cdef string key
        for key, x in iteritems(ctx_param_dict[1]):
            cparams.push_back(pair[string, shared_ptr[CVariable]](key, (< _Variable > x).varp.variable()))

        self.communicatorp.add_context_and_parameters(
            pair[CContext, vector[pair[string, shared_ptr[CVariable]]]](ctx_param_dict[0], cparams))

    def init(self, ):
        """Initialize a communicator.

        Initall or initrank, depending multi-threads or multi-processes.
        This function *MUST* be called after all parameters communicated
        are added by `add_context_and_parameters`.        

        """
        self.communicatorp.init()

    def allreduce(self, cpp_bool division=False, cpp_bool inplace=False):
        """Deprecated. See all_reduce, instead. 

        Allreduce over parameters added.
        Currently, `allreduce` is applied to gradient regions.

        Args:
            division (bool): Flag to divide the reduce data by the 
                number of `contexts` added, or the number of devices. 
            inplace (bool): Flag to use a packed array. Default is false.
                When true, it is memory-efficient but slow. When false, 
                it is not memory efficient but fast. In both case, one can 
                get the result in the same memory region.
        """
        with nogil:
            self.communicatorp.allreduce(division, inplace)

    def all_reduce(self, data, cpp_bool division=False, cpp_bool inplace=False):
        """All reduce over parameters added.

        Args:
            data (:obj:`NdArray` or list of :obj:`NdArray`)
            division (bool): Flag to divide the reduce data by the 
                number of `contexts` added, or the number of devices. 
            inplace (bool): Flag to use a packed array. Default is false.
                When true, it is memory-efficient but slow. When false, 
                it is not memory efficient but fast. In both case, one can 
                get the result in the same memory region.
        """
        cdef vector[shared_ptr[CNdArray]] cndarray_list
        if type(data) == list:
            for x in data:
                cndarray_list.push_back(( < NdArray > x).arr)
            with nogil:
                self.communicatorp.all_reduce(cndarray_list, division, inplace)
        else:
            cndarray_list.push_back(( < NdArray > data).arr)
            with nogil:
                self.communicatorp.all_reduce(cndarray_list, division, inplace)


def DataParalellCommunicator(CContext ctx):
    """Data Parallel Communicator for Distributed Training.

    This class does collectives in a single-process in a machine.

    Args:
        context (:obj:`Context`): context used in this communicator.

    Example: 

    In case of the multi-thread data parallel distributed training,

    .. code-block:: python

        # Networks and Solvers building comes above
        import nnabla.communicators as C
        comm = C.DataParalellCommunicator(ctx)

        # Add contexts and parameters to the communicator 
        for i in range(n_devices):
            device_scope_name = "device{}".format(i)
            with nn.parameter_scope(device_scope_name):
                ctx = ctxs[i]
                params = nn.get_parameters()
                comm.add_context_and_parameters((ctx, params))
        comm.init()

        # Training loop
        for itr in range(num_itr):

            # Forward, zerograd, backward
            for i in range(n_devices):
                losses[i].forward()
                solvers[i].zero_grad()
                losses[i].backward()

            # Allreduce
            comm.allreduce()

            # Update
            for i in range(n_devices):
                solvers[i].update()            

    """
    import platform
    import ctypes
    if platform.system() != 'Linux':
        raise Exception(
            "DataParalellCommunicator is not supported other than linux.")

    return Communicator.create(create_DataParallelCommunicatorCommunicator(ctx))


def MultiProcessDataParalellCommunicator(CContext ctx):
    """
    Multi Process Data Parallel Communicator for Distributed Training.

    Args:
        context (:obj:`Context`): context used in this communicator.

    Example: 

    In case of the multi-process data parallel distributed training,

    .. code-block:: python

        # Communicator and Context
        extension_module = "cuda.cudnn"
        ctx = extension_context(extension_module)
        comm = C.MultiProcessDataParalellCommunicator(ctx)
        comm.init()
        n_devices = comm.size
        mpi_rank = comm.rank
        device_id = comm.local_rank
        ctx = extension_context(extension_module, device_id=device_id)
        nn.set_default_context(ctx)

        # Network and Solver created here

        ...


        # Training loop
        for itr in range(num_itr):
            # Forward, zerograd, backward
            losse.forward()
            solver.zero_grad()
            loss.backward()

            # Allreduce
            comm.allreduce([v.grad for v in nn.get_parameters().values()])

            # Update
            solver.update()

    """

    # There is the known bug in python used with MPI
    # described in https://xrunhprof.wordpress.com/2014/11/04/an-openmpi-python-and-dlopen-issue/
    import platform
    import ctypes
    if platform.system() == 'Linux':
        ctypes.CDLL("libmpi.so", mode=ctypes.RTLD_GLOBAL)
    else:
        raise Exception(
            "MultiProcessDataParalellCommunicator is not supported other than linux.")
    return Communicator.create(create_MultiProcessDataParallelCommunicatorCommunicator(ctx))
