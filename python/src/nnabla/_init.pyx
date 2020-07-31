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

from .logger import logger
cimport _init
from _init cimport(
    register_gc, SingletonManager,
    _cpu_array_classes, _cpu_set_array_classes)

available_contexts = []


def add_available_context(ctx):
    if ctx not in available_contexts:
        available_contexts.append(ctx)

# Explicitly initialize NNabla.


logger.info('Initializing CPU extension...')
add_available_context('cpu')
_init.init_cpu()


def clear_memory_cache():
    """clear_memory_cache()

    Clear CPU memory cache.
    """
    clear_cpu_memory_cache()

def print_memory_cache_map():
    """Print CPU memory cache map."""
    print_cpu_memory_cache_map()


cdef public void call_gc() with gil:
    """
    Callback function to be registered GC singleton in C++.
    """
    import gc
    gc.collect()


def finalize():
    """
    Clear all resources before exiting Python. Register to atexit.
    """
    import gc
    import nnabla as nn
    nn.clear_parameters()
    gc.collect()
    SingletonManager.clear()


register_gc(call_gc)
import atexit
atexit.register(finalize)

###############################################################################
# Array preference API
# TODO: Move  to C++.
###############################################################################

__cache_preferred = None
_original_array_classes = _cpu_array_classes()
_callbacks_prefer_cached_array = []
_callbacks_reset_array_preference = []


def _cached_array_preferred():
    global __cache_preferred
    return __cache_preferred


def _add_callback_prefer_cached_array(func):
    global _callbacks_prefer_cached_array
    _callbacks_prefer_cached_array.append(func)


def _add_callback_reset_array_preference(func):
    global _callbacks_reset_array_preference
    _callbacks_reset_array_preference.append(func)


def prefer_cached_array(prefer):
    global __cache_preferred
    global _callbacks_prefer_cached_array
    if __cache_preferred == prefer:
        return
    __cache_preferred = prefer

    a = _cpu_array_classes()
    a = sorted(enumerate(a), key=lambda x: (prefer ^ ('Cached' in x[1]), x[0]))
    _cpu_set_array_classes(map(lambda x: x[1], a))

    # Call all prefer_cached_array function registered
    for func in _callbacks_prefer_cached_array:
        func(prefer)


def reset_array_preference():
    """reset_array_preference()

    Reset array class preference.
    """
    global __cache_preferred
    global _original_array_classes
    global _callbacks_reset_array_preference
    __cache_preferred = None
    _cpu_set_array_classes(_original_array_classes)
    for func in _callbacks_reset_array_preference:
        func()


def array_classes():
    """Get CPU array classes"""
    return _cpu_array_classes()

###############################################################################


def device_synchronize(str device):
    """Dummy.

    Args:
        device (int): Device ID.

    """
    cpu_device_synchronize(device)


def get_device_count():
    """Always returns 1.

    Returns:
        int: Number of devices available.

    """
    return cpu_get_device_count()


def get_devices():
    """Dummy.

    Returns:
        list of str: List of available devices.

    """
    return cpu_get_devices()
