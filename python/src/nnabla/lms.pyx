from libcpp.memory cimport make_shared, shared_ptr
from . cimport lms
from nnabla.function cimport Function


cdef class SwapInOutScheduler:
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
        
        self.scheduler = make_shared[CSwapInOutScheduler]\
                         (ch_ctx, cd_ctx, csize, cprefetch_size,
                          save_host_mem, save_host_mem_no_abort)

    def __dealloc__(self):
        pass
    
    def start_scheduling(self):
        self.scheduler.get().start_scheduling()

    def end_scheduling(self):
        self.scheduler.get().end_scheduling()

    def function_pre_hook(self, func):
        cdef CgFunctionPtr cg_func = (<Function>func).fun
        self.scheduler.get().pre_function_callback(cg_func)

    def function_post_hook(self, func):
        cdef CgFunctionPtr cg_func = (<Function>func).fun
        self.scheduler.get().post_function_callback(cg_func)

    def update_pre_hook(self):
        self.scheduler.get().pre_update_callback()

    def update_post_hook(self):
        self.scheduler.get().post_update_callback()
        