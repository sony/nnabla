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
from _variable cimport CommunicatorBackwardCallback
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
                dictionary is :obj:`Variable`.  
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

    def clear_context_parameters(self):
        '''Clear all registered contexts and parameters.
        '''
        self.communicatorp.clear_context_parameters()

    def init(self, ):
        """Initialize a communicator.

        Initall or initrank, depending multi-threads or multi-processes.
        This function *MUST* be called after all parameters communicated
        are added by `add_context_and_parameters`.        

        """
        self.communicatorp.init()

    def barrier(self):
        """Blocks until all processes in the communicator have reached this routine. 
        """
        self.communicatorp.barrier()

    def abort(self):
        """Terminates MPI execution environment
        """
        self.communicatorp.abort()

    def new_group(self, name_ranks):
        """
        Args:
            name_ranks (tuple): Tuple of name (`str`) and ranks (`list`).

        Returns:
            group name (str)

        Example: 

        .. code-block:: python

            # Communicator and Context
            extension_module = "cudnn"
            ctx = get_extension_context(extension_module)
            comm = C.MultiProcessCommunicator(ctx)
            comm.init()

            # New group
            group = comm.new_group("node0", [0, 1, 2, 3])

        """
        return self.communicatorp.new_group(pair[string, vector[int]](name_ranks[0], name_ranks[1]))

    def list_groups(self, ):
        """
        Returns:
            groups (dict): Groups (`str`) of name to ranks (`list`).

        """
        return self.communicatorp.list_groups()

    def find_group(self, group):
        """
        Return the list of ranks in the group. If the group does not exist, 
        the empty list is returned.

        Args: 
            group (str): Name of the group.
        Returns:
            ranks (list): List of ranks (`int`).
        """
        return self.communicatorp.find_group(group)

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

    def all_reduce(self, data, cpp_bool division=False, cpp_bool inplace=False, string group="world"):
        """All reduce over data in different device.

        Args:
            data (:obj:`NdArray` or list of :obj:`NdArray`)
            division (bool): Flag to divide the reduce data by the 
                number of `contexts` added, or the number of devices. 
            inplace (bool): Flag to use a packed array. Default is false.
                When true, it is memory-efficient but slow. When false, 
                it is not memory efficient but fast. In both case, one can 
                get the result in the same memory region.
            group (string): Name of a group. This groups is used when the collective is called.

        Example: 

        .. code-block:: python

            # Run like `mpirun -n 2 python <code_snippet.py>`
            # note: the order of the output to stdout are stochastic because of multiprocesses.
             
            # Communicator and Context
            import numpy as np
            import nnabla as nn
            import nnabla.communicators as C
            from nnabla.ext_utils import get_extension_context
             
            extension_module = "cudnn"
            ctx = get_extension_context(extension_module)
            comm = C.MultiProcessCommunicator(ctx)
            comm.init()
             
            # Data
            x_list = [nn.Variable([2, 2]), nn.Variable([2, 2])]
            print("Before the collective ({}-th)".format(comm.rank))
            for x in x_list:
                x.d = np.random.rand(*x.shape)
                print(x.d)
                
            # AllReduce
            comm.all_reduce([x.data for x in x_list], inplace=True)
             
            # Check
            print("After the collective ({}-th)".format(comm.rank))
            for x in x_list:
                print(x.d)

        """
        cdef vector[shared_ptr[CNdArray]] cndarray_list
        if type(data) == list:
            for x in data:
                cndarray_list.push_back(( < NdArray > x).arr)
            with nogil:
                self.communicatorp.all_reduce(
                    cndarray_list, division, inplace, group)
        else:
            cndarray_list.push_back(( < NdArray > data).arr)
            with nogil:
                self.communicatorp.all_reduce(
                    cndarray_list, division, inplace, group)

    def reduce(self, data, int dst, cpp_bool division=False, cpp_bool inplace=False, string group="world"):
        """Reduce over data in different device.

        Args:
            data (:obj:`NdArray` or list of :obj:`NdArray`)
            dst (int): Destination rank where the result is saved.
            division (bool): Flag to divide the reduce data by the 
                number of `contexts` added, or the number of devices. 
            inplace (bool): Flag to use a packed array. Default is false.
                When true, it is memory-efficient but slow. When false, 
                it is not memory efficient but fast. In both case, one can 
                get the result in the same memory region.
            group (string): Name of a group. This groups is used when the collective is called.

        Example: 

        .. code-block:: python

            # Run like `mpirun -n 2 python <code_snippet.py>`
            # note: the order of the output to stdout are stochastic because of multiprocesses.
             
            # Communicator and Context
            import numpy as np
            import nnabla as nn
            import nnabla.communicators as C
            from nnabla.ext_utils import get_extension_context
             
            extension_module = "cudnn"
            ctx = get_extension_context(extension_module)
            comm = C.MultiProcessCommunicator(ctx)
            comm.init()
             
            # Data
            x_list = [nn.Variable([2, 2]), nn.Variable([2, 2])]
            print("Before the collective ({}-th)".format(comm.rank))
            for x in x_list:
                x.d = np.random.rand(*x.shape)
                print(x.d)
                
            # Reduce
            comm.reduce([x.data for x in x_list], dst=0, inplace=True)
             
            # Check
            print("After the collective ({}-th)".format(comm.rank))
            for x in x_list:
                print(x.d)

        """
        cdef vector[shared_ptr[CNdArray]] cndarray_list
        if type(data) == list:
            for x in data:
                cndarray_list.push_back(( < NdArray > x).arr)
            with nogil:
                self.communicatorp.reduce(
                    cndarray_list, dst, division, inplace, group)
        else:
            cndarray_list.push_back(( < NdArray > data).arr)
            with nogil:
                self.communicatorp.reduce(
                    cndarray_list, dst, division, inplace, group)

    def bcast(self, data, int src, cpp_bool inplace=False, string group="world"):
        """Broadcast data to different devices.

        Args:
            data (:obj:`NdArray` or list of :obj:`NdArray`)
            src (int): Source rank where the data is broadcasted.
            inplace (bool): Flag to use a packed array. Default is false.
                When true, it is memory-efficient but slow. When false, 
                it is not memory efficient but fast. In both case, one can 
                get the result in the same memory region.
            group (string): Name of a group. This groups is used when the collective is called.

        Example: 

        .. code-block:: python

            # Run like `mpirun -n 2 python <code_snippet.py>`
            # note: the order of the output to stdout are stochastic because of multiprocesses.
             
            # Communicator and Context
            import numpy as np
            import nnabla as nn
            import nnabla.communicators as C
            from nnabla.ext_utils import get_extension_context
             
            extension_module = "cudnn"
            ctx = get_extension_context(extension_module)
            comm = C.MultiProcessCommunicator(ctx)
            comm.init()
             
            # Data
            x_list = [nn.Variable([2, 2]), nn.Variable([2, 2])]
            print("Before the collective ({}-th)".format(comm.rank))
            for x in x_list:
                x.d = np.random.rand(*x.shape)
                print(x.d)
                
            # Bcast
            comm.bcast([x.data for x in x_list], src=0, inplace=True)
             
            # Check
            print("After the collective ({}-th)".format(comm.rank))
            for x in x_list:
                print(x.d)

        """
        cdef vector[shared_ptr[CNdArray]] cndarray_list
        if type(data) == list:
            for x in data:
                cndarray_list.push_back(( < NdArray > x).arr)
            with nogil:
                self.communicatorp.bcast(cndarray_list, src, inplace, group)
        else:
            cndarray_list.push_back(( < NdArray > data).arr)
            with nogil:
                self.communicatorp.bcast(cndarray_list, src, inplace, group)

    def all_gather(self, ndarray, ndarray_list, string group="world"):
        """All gather over data in different device.

        Args:
            ndarray (:obj:`NdArray`): Data to be gathered. 
            ndarray_list (:obj:`NdArray`):  Data to be saved.
            group (string): Name of a group. This groups is used when the collective is called.

        Example: 

        .. code-block:: python

            # Run like `mpirun -n 2 python <code_snippet.py>`
            # note: the order of the output to stdout are stochastic because of multiprocesses.
             
            # Communicator and Context
            import numpy as np
            import nnabla as nn
            import nnabla.communicators as C
            from nnabla.ext_utils import get_extension_context
             
            extension_module = "cudnn"
            ctx = get_extension_context(extension_module)
            comm = C.MultiProcessCommunicator(ctx)
            comm.init()
             
            # Data
            x = nn.Variable([2, 2])
            x.d = np.random.rand(*x.shape)
            y_list = [nn.Variable([2, 2]), nn.Variable([2, 2])]
            print("Before the collective ({}-th)".format(comm.rank))
            print(x.d)
             
            # AllGather
            comm.all_gather(x.data, [y.data for y in y_list])
             
            # Check
            print("After the collective ({}-th)".format(comm.rank))
            for y in y_list:
                print(y.d)

        """
        cdef shared_ptr[CNdArray] cndarray = ( < NdArray > ndarray).arr
        cdef vector[shared_ptr[CNdArray]] cndarray_list
        for x in ndarray_list:
            cndarray_list.push_back(( < NdArray > x).arr)
        with nogil:
            self.communicatorp.all_gather(cndarray, cndarray_list, group)

    def reduce_scatter(self, ndarray_list, ndarray, cpp_bool division=False, string group="world"):
        """Reduce scatter over data in different device.

        Args:
            ndarray_list (:obj:`NdArray`):  List of data to be reduced over different devices.
            ndarray (:obj:`NdArray`): Data to be saved.
            group (string): Name of a group. This groups is used when the collective is called.

        Example: 

        .. code-block:: python

            # Run like `mpirun -n 2 python <code_snippet.py>`
            # note: the order of the output to stdout are stochastic because of multiprocesses.
             
            # Communicator and Context
            import numpy as np
            import nnabla as nn
            import nnabla.communicators as C
            from nnabla.ext_utils import get_extension_context
             
            extension_module = "cudnn"
            ctx = get_extension_context(extension_module)
            comm = C.MultiProcessCommunicator(ctx)
            comm.init()
             
            # Data
            x_list = [nn.Variable([2, 2]), nn.Variable([2, 2])]
            y = nn.Variable([2, 2])
            print("Before the collective ({}-th)".format(comm.rank))
            for x in x_list:
                x.d = np.random.rand(*x.shape)
                print(x.d)
                
            # ReduceScatter
            comm.reduce_scatter([x.data for x in x_list], y.data)
             
            # Check
            print("After the collective ({}-th)".format(comm.rank))
            print(y.d)

        """
        cdef shared_ptr[CNdArray] cndarray = ( < NdArray > ndarray).arr
        cdef vector[shared_ptr[CNdArray]] cndarray_list
        for x in ndarray_list:
            cndarray_list.push_back(( < NdArray > x).arr)
        with nogil:
            self.communicatorp.reduce_scatter(
                cndarray_list, cndarray, division, group)

    def all_reduce_callback(self, data, size_t pack_size, cpp_bool division=False, string group="world"):
        """All reduce over data in different device.

        Note:
            This function does not support shared parameters (such as RNNs) currently.

        Args:
            data (:obj:`NdArray` or list of :obj:`NdArray`)
            pack_size (int): The number of values contained in the packed data.
            division (bool): Flag to divide the reduce data by the
                number of `contexts` added, or the number of devices.
            group (string): Name of a group. This groups is used when the collective is called.

        Example:

        In case of the multi-process data parallel distributed training,

        .. code-block:: python

            # Run like `mpirun -n 2 python <code_snippet.py>`
             
            # Communicator and Context
            import numpy as np
            import nnabla as nn
            import nnabla.communicators as C
            from nnabla.ext_utils import get_extension_context

            extension_module = "cudnn"
            ctx = get_extension_context(extension_module)
            comm = C.MultiProcessCommunicator(ctx)
            comm.init()

            n_class = 2
            b, c, h, w = 4, 1, 32, 32

            # Data
            x = nn.Variable([b, c, h, w])
            y = nn.Variable([b, 1])

            # Network setting
            h = PF.convolution(x, 1, (3, 3), (1, 1), (1, 1))
            pred = PF.affine(h, 2)
            loss = F.mean(F.softmax_cross_entropy(pred, y))

            loss.forward()
            # AllReduce during backward
            loss.backward(communicator_callbacks = comm.all_reduce_callback([v.grad for v in nn.get_parameters().values()], 1024 * 1024 * 2))

        """
        cdef vector[shared_ptr[CNdArray]] cndarray_list
        if type(data) == list:
            for x in data:
                cndarray_list.push_back((< NdArray > x).arr)
        else:
            cndarray_list.push_back((< NdArray > data).arr)
        return CommunicatorBackwardCallback.create_from_ccallback(
            self.communicatorp.all_reduce_callback(cndarray_list, pack_size, division, group))


def DataParallelCommunicator(CContext ctx):
    """Data Parallel Communicator for Distributed Training.

    This class does collectives in a single-process in a machine.

    Args:
        context (:obj:`Context`): context used in this communicator.

    Example: 

    In case of the multi-thread data parallel distributed training,

    .. code-block:: python

        # Networks and Solvers building comes above
        import nnabla.communicators as C
        comm = C.DataParallelCommunicator(ctx)

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
            "DataParallelCommunicator is not supported other than linux.")

    return Communicator.create(create_DataParallelCommunicatorCommunicator(ctx))


def MultiProcessDataParallelCommunicator(CContext ctx):
    """
    Multi Process Data Parallel Communicator for Distributed Training.

    Args:
        context (:obj:`Context`): context used in this communicator.

    Example: 

    In case of the multi-process data parallel distributed training,

    .. code-block:: python

        # Communicator and Context
        extension_module = "cudnn"
        ctx = get_extension_context(extension_module)
        comm = C.MultiProcessCommunicator(ctx)
        comm.init()
        n_devices = comm.size
        mpi_rank = comm.rank
        device_id = comm.local_rank
        ctx.device_id = str(device_id)
        nn.set_default_context(ctx)

        # Network and Solver created here

        ...


        # Training loop
        for itr in range(num_itr):
            # Forward, zerograd, backward
            loss.forward()
            solver.zero_grad()
            loss.backward()

            # Allreduce
            comm.all_reduce([v.grad for v in nn.get_parameters().values()])

            # Update
            solver.update()

    """

    import platform
    import ctypes
    if platform.system() == 'Linux':
        mpi_loaded = False
        for libmpi in ['libmpi.so', 'libmpi.so.12', 'libmpi.so.20', 'libmpi.so.40']:
            try:
                # There is the known bug in python used with MPI
                # described in https://xrunhprof.wordpress.com/2014/11/04/an-openmpi-python-and-dlopen-issue/
                ctypes.CDLL(libmpi, mode=ctypes.RTLD_GLOBAL)
                mpi_loaded = True
            except:
                pass
            if mpi_loaded:
                break
        if not mpi_loaded:
            msg = '\n' \
                '###########################################################################\n' \
                '# ctypes.CDLL("libmpi.so", mode=ctypes.RTLD_GLOBAL) failed. \n' \
                '# This is the workaround for "An OpenMPI, python and dlopen issue", \n' \
                '# but it may not necessary for Open MPI versions other than Open MPI 1.x.\n' \
                '###########################################################################\n'
            import nnabla as nn
            nn.logger.warn(msg)
    else:
        raise Exception(
            "MultiProcessDataParallelCommunicator is not supported other than linux.")
    return Communicator.create(create_MultiProcessDataParallelCommunicatorCommunicator(ctx))


# Aliases 
## backward compatibility
MultiProcessDataParalellCommunicator = MultiProcessDataParallelCommunicator
# More suitable name since `MultiProcessDataParallelCommunicator` is not limited to the data parallel distributed training
MultiProcessCommunicator = MultiProcessDataParallelCommunicator
Comm = MultiProcessDataParallelCommunicator
Dist = MultiProcessDataParallelCommunicator
