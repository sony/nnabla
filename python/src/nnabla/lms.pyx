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

from libcpp.memory cimport make_shared, shared_ptr
from . cimport lms
from nnabla.function cimport Function
import callback

cdef class SwapInOutScheduler:
    """Interface class for out-of-core execution / training.

    This API enables training neural networks whose size is larger than alloted GPU memory.
    See https://arxiv.org/abs/2010.14109 for more detail of shcheduling strategy.

    Note:
        `cuda_init.prefer_cuda_virtual_array()` used in following example can be used under cuda >= 10.2 and cudnn >= 8.
        We utilize `virtual memory management <https://developer.nvidia.com/blog/introducing-low-level-gpu-virtual-memory-management/#:~:text=CUDA%2010.2%20introduces%20a%20new,GPU%20memory%20usage%20in%20applications.&text=There%20are%20plenty%20of%20applications,your%20initial%20allocation%20should%20be.>`_ supported from cuda 10.2.
        Additionally, when we tested virtual memory management with cuda >= 10.2 and cudnn < 8, we found the computation results of some cuddn functions are inaccurate.
        So, when your environment has cuda < 10.2 or cudnn < 8, the virtual memory allocator in nnabla will not be built and you can't use it.
        If you would like to use `SwapInOutScheduler` to the fullest extent, please install cuda >= 10.2 and cudnn >= 8 and reinstall the corresponding nnabla-ext-cuda package.
    Example: 

    .. code-block:: python

        from nnabla.lms import SwapInOutScheduler

        # Change memory allocator which is preferable for SwapInOutScheduler.
        from nnabla_ext.cuda.init as cuda_init
        cuda_init.prefer_cpu_pinned_array()  # To accelerate memory transfer, using pinned memory for cpu memory will be preferable.
        
        # Only for cuda >= 10.2 and cudnn >= 8. This setting is the best for SwapInOutScheduler.
        cuda_init.prefer_cuda_virtual_array()  # To reduce memory fragmentation due to cpu-gpu memory transfers, using virtual allocator for gpu memory will be preferable.

        # create context for both host and device
        from nnabla.ext_utils import get_extension_context
        host_ctx = get_extension_context("cpu", device_id="", type_config="float") # device_id is dummy
        device_ctx = get_extension_context("cudnn", device_id="0", type_config="float")
        
        scheduler = SwapInOutScheduler(host_ctx, device_ctx, size=max_gpu_memory_size)

        # Make sure to call `nn.set_default_context` after calling prefer_xxx_array() to activate a change of memory preference.
        nn.set_default_context(device_ctx)

        x = nn.Variable(...)
        loss = build_network(x)

        solver = S.Sgd(nn.get_parameters())

        for i in range(iteration):
            # scheduling memory transfers for all tensors appearing under the context of scheduler.
            with scheduler:
                x.d = next_data()

                loss.forward(clear_no_need_grad=True)
                
                solver.zero_grad()
                loss.backward(clear_buffer=True)

                solver.update()
    

    When you get Out-of-Memory (OOM) error under the SwapInOutScheduler, possibly there are 2 options to avoid this OOM.

    1. Set small budget of GPU memory for scheduling.
    2. Set small size for a physical memory chunk allocated by virtual memory allocator.

    These are examplified as follows:

    Example: 

    .. code-block:: python

        # 1. Set small budget of GPU memory for scheduling
        # You can reduce the below ratio until you can execute your network.
        memsize_for_scheduler = max_gpu_memory_size * 0.8   
        scheduler = SwapInOutScheduler(..., size=memsize_for_scheduler)

        # 2. Set small size for a physical memory chunk allocated by virtual memory allocator
        # In default, the chunk size is set as 20MB (20 << 20).
        from nnabla_ext.cuda.init import set_cuda_virtual_memory_chunk_size
        set_cuda_virtual_memory_chunk_size(2 << 20)  # Set 2MB, for example.
    """

    def __cinit__(self, h_ctx, d_ctx, size, prefetch_size=None,
                  cpp_bool save_host_mem=False,
                  cpp_bool save_host_mem_no_abort=False):
        cdef CContext ch_ctx = h_ctx
        cdef CContext cd_ctx = d_ctx
        cdef size_t csize = size

        # prefetch size is 1.5 x size according to ResNet50 experiment.
        if prefetch_size is None:
            prefetch_size = size * 1.5

        cdef size_t cprefetch_size = prefetch_size

        self.scheduler = make_shared[CSwapInOutScheduler] \
            (ch_ctx, cd_ctx, csize, cprefetch_size,
             save_host_mem, save_host_mem_no_abort)

    def __dealloc__(self):
        pass

    def start_scheduling(self):
        """
        An interface to specify the starting point for scheduling.
        A range between `start_scheduling()` and `end_scheduling()` is a target for a single scheduling.
        
        Note that, using with statement of SwapInOutScheduler, `start_scheduling()` will be automatically called when entering with statement.
        In general, avoid to use `start_scheduling()` and `end_scheduling()` and use with statement insted (`with scheduler:`, see an example above).
        """
        
        self.scheduler.get().start_scheduling()

    def end_scheduling(self):
        """
        An interface to specify the end point for scheduling.
        A range between `start_scheduling()` and `end_scheduling()` is a target for a single scheduling.
        
        Note that, using with statement of SwapInOutScheduler, `end_scheduling()` will be automatically called when exiting with statement.
        In general, avoid to use `start_scheduling()` and `end_scheduling()` and use with statement insted (`with scheduler:`, see an example above).
        """
        
        self.scheduler.get().end_scheduling()

    def function_pre_hook(self, func):
        """
        A callback executed as `function_pre_hook` in forward and backward.
        
        For all forward and backward wrapped by with statement of SwapInOutScheduler, this callback is automatically set.
        In general, avoid to set this manually and use with statement of SwapInOutScheduler.
        """
        
        cdef CgFunctionPtr cg_func = (<Function>func).fun
        self.scheduler.get().pre_function_callback(cg_func)

    def function_post_hook(self, func):
        """
        A callback executed as `function_post_hook` in forward and backward.
        
        For all forward and backward wrapped by with statement of SwapInOutScheduler, this callback is automatically set.
        In general, avoid to set this manually and use with statement of SwapInOutScheduler.
        """
        
        cdef CgFunctionPtr cg_func = (<Function>func).fun
        self.scheduler.get().post_function_callback(cg_func)

    def update_pre_hook(self):
        """
        A callback executed as `pre_hook` in all solver functions, e.g. solver.update, solver.weight_decay, solver.clip_grad_by_norm, and so on.
        
        For all solver functions wrapped by with statement of SwapInOutScheduler, this callback is automatically set.
        In general, avoid to set this manually and use with statement of SwapInOutScheduler.
        """

        self.scheduler.get().pre_update_callback()

    def update_post_hook(self):
        """
        A callback executed as `post_hook` in all solver functions, e.g. solver.update, solver.weight_decay, solver.clip_grad_by_norm, and so on.
        
        For all solver functions wrapped by with statement of SwapInOutScheduler, this callback is automatically set.
        In general, avoid to set this manually and use with statement of SwapInOutScheduler.
        """
        self.scheduler.get().post_update_callback()

    def __enter__(self):
        self.start_scheduling()

        callback.set_function_pre_hook("lms_function_pre_hook", self.function_pre_hook)
        callback.set_function_post_hook("lms_function_post_hook", self.function_post_hook)
        callback.set_solver_pre_hook("lms_solver_pre_hook", self.update_pre_hook)
        callback.set_solver_post_hook("lms_solver_post_hook", self.update_post_hook)

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        callback.unset_function_pre_hook("lms_function_pre_hook")
        callback.unset_function_post_hook("lms_function_post_hook")
        callback.unset_solver_pre_hook("lms_solver_pre_hook")
        callback.unset_solver_post_hook("lms_solver_post_hook")

        self.end_scheduling()

