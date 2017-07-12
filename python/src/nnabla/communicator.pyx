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
from libcpp.unordered_map cimport unordered_map
from libcpp.memory cimport shared_ptr
from libcpp cimport bool as cpp_bool
from libc.stdint cimport int64_t

cimport communicator
from communicator cimport CCommunicator
 
cimport _variable
from _variable cimport Variable as _Variable, CVariable
from _variable import Context

# Numpy
import numpy as np
cimport numpy as np
np.import_array()

cdef class Communicator:
    """Communicator interface class.
    
    Communicator exchanges data (e.g., gradient) using MPI-like 
    collectives. This class is used for the distributed training.
    
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
            
            # Inplace-allreduce
            comm.iallreduce()
            
            # Update
            for i in range(n_devices):
                solvers[i].update()
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
        for key, x in ctx_param_dict[1].iteritems():
            cparams.push_back(pair[string, shared_ptr[CVariable]](key, ( < _Variable > x).varp.variable()))
         
        self.communicatorp.add_context_and_parameters(
            pair[CContext, vector[pair[string, shared_ptr[CVariable]]]](ctx_param_dict[0], cparams))
        
    def init(self, ):
        """Initialize a communicator.
        
        Initall or initrank, depending multi-threads or multi-processes.
        This function *MUST* be called after all parameters communicated
        are added by `add_context_and_parameters`.        
         
        """
        self.communicatorp.init()
        
    def iallreduce(self, division=True):
        """Inplace allreduce over parameters added.
        This method is \b sync before and after iallreduce w.r.t. a host thread.
        Currently, `iallreduce` is applied to gradient regions.
        
        Args:
            division (bool): Flag to divide the reduce data by the 
                number of `contexts` added, in other words, the number
                of devices. 
        
        """
        self.communicatorp.iallreduce(division)


def DataParalellCommunicator(CContext ctx):
        """
        Data Parallle Communicator for Distributed Training.
        
        Args:
            context (:obj:`Context`): context used in this communicator.
        """
        return  Communicator.create(create_DataParallelCommunicatorCommunicator(ctx))        

def mpDataParalellCommunicator(CContext ctx):
        """
        Multi Process Data Parallle Communicator for Distributed Training.
        
        Args:
            context (:obj:`Context`): context used in this communicator.
        """
        # There is the known bug in python used with MPI
        # described https://xrunhprof.wordpress.com/2014/11/04/an-openmpi-python-and-dlopen-issue/
        import platform
        import ctypes
        if platform.system() == 'Linux':
            ctypes.CDLL("libmpi.so.12", mode=ctypes.RTLD_GLOBAL)
        return  Communicator.create(create_MultiProcessDataParallelCommunicatorCommunicator(ctx))        
