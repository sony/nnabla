# Copyright 2021 Sony Corporation.
# Copyright 2021 Sony Group Corporation.
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

import os
import sys
import ctypes
import threading

# NVML return codes
NVML_SUCCESS = 0
NVML_ERROR_UNINITIALIZED = 1
NVML_ERROR_INVALID_ARGUMENT = 2
NVML_ERROR_NOT_SUPPORTED = 3
NVML_ERROR_NO_PERMISSION = 4
NVML_ERROR_ALREADY_INITIALIZED = 5
NVML_ERROR_NOT_FOUND = 6
NVML_ERROR_INSUFFICIENT_SIZE = 7
NVML_ERROR_INSUFFICIENT_POWER = 8
NVML_ERROR_DRIVER_NOT_LOADED = 9
NVML_ERROR_TIMEOUT = 10
NVML_ERROR_IRQ_ISSUE = 11
NVML_ERROR_LIBRARY_NOT_FOUND = 12
NVML_ERROR_FUNCTION_NOT_FOUND = 13
NVML_ERROR_CORRUPTED_INFOROM = 14
NVML_ERROR_GPU_IS_LOST = 15
NVML_ERROR_RESET_REQUIRED = 16
NVML_ERROR_OPERATING_SYSTEM = 17
NVML_ERROR_LIB_RM_VERSION_MISMATCH = 18
NVML_ERROR_IN_USE = 19
NVML_ERROR_MEMORY = 20
NVML_ERROR_NO_DATA = 21
NVML_ERROR_VGPU_ECC_NOT_SUPPORTED = 22
NVML_ERROR_INSUFFICIENT_RESOURCES = 23
NVML_ERROR_UNKNOWN = 999

# CONTSTANT: buffer size
NVML_DEVICE_NAME_BUFFER_SIZE = 64

nvml_lib = None
lib_load_lock = threading.Lock()
_nvml_function_cache = dict()

# Device structures
c_nvmlDevice_t = ctypes.POINTER(
    type('struct_c_nvmlDevice_t', (ctypes.Structure, ), {}))


class c_nvmlUtilization_t(ctypes.Structure):
    _fields_ = [
        ('gpu', ctypes.c_uint),
        ('memory', ctypes.c_uint),
    ]


class c_nvmlMemory_t(ctypes.Structure):
    _fields_ = [
        ('total', ctypes.c_ulonglong),
        ('free', ctypes.c_ulonglong),
        ('used', ctypes.c_ulonglong),
    ]


class NVMLError(Exception):
    # Error codes list in use
    _errcode_to_string = {
        NVML_ERROR_UNINITIALIZED: "Uninitialized",
        NVML_ERROR_INVALID_ARGUMENT: "Invalid Argument",
        NVML_ERROR_NOT_SUPPORTED: "Not Supported",
        NVML_ERROR_NO_PERMISSION: "Insufficient Permissions",
        NVML_ERROR_ALREADY_INITIALIZED: "Already Initialized",
        NVML_ERROR_NOT_FOUND: "Not Found",
        NVML_ERROR_INSUFFICIENT_SIZE: "Insufficient Size",
        NVML_ERROR_INSUFFICIENT_POWER: "Insufficient External Power",
        NVML_ERROR_DRIVER_NOT_LOADED: "Driver Not Loaded",
        NVML_ERROR_TIMEOUT: "Timeout",
        NVML_ERROR_IRQ_ISSUE: "Interrupt Request Issue",
        NVML_ERROR_LIBRARY_NOT_FOUND: "NVML Shared Library Not Found",
        NVML_ERROR_FUNCTION_NOT_FOUND: "Function Not Found",
        NVML_ERROR_CORRUPTED_INFOROM: "Corrupted infoROM",
        NVML_ERROR_GPU_IS_LOST: "GPU is lost",
        NVML_ERROR_RESET_REQUIRED: "GPU requires restart",
        NVML_ERROR_OPERATING_SYSTEM: "The operating system has blocked the request.",
        NVML_ERROR_LIB_RM_VERSION_MISMATCH: "RM has detected an NVML/RM version mismatch.",
        NVML_ERROR_MEMORY: "Insufficient Memory",
        NVML_ERROR_UNKNOWN: "Unknown Error",
    }

    def __init__(self, value):
        self.value = value

    def __str__(self):
        try:
            if self.value not in NVMLError._errcode_to_string:
                NVMLError._errcode_to_string[self.value] = str(
                    nvmlErrorString(self.value))
            return NVMLError._errcode_to_string[self.value]
        except NVMLError:
            return f"NVML Error code {self.value}"


def _get_nvml_function(name):
    """Get NVML function from the NVML library
    """
    global nvml_lib

    if name in _nvml_function_cache:
        return _nvml_function_cache[name]

    lib_load_lock.acquire()
    try:
        if nvml_lib is None:
            raise NVMLError(NVML_ERROR_UNINITIALIZED)
        _nvml_function_cache[name] = getattr(nvml_lib, name)
        return _nvml_function_cache[name]
    except AttributeError:
        raise NVMLError(NVML_ERROR_FUNCTION_NOT_FOUND)
    finally:
        lib_load_lock.release()


def _check_return(ret):
    if ret != NVML_SUCCESS:
        raise NVMLError(ret)
    return ret


def _load_nvml_library():
    """Load the NVML Shared Library 
    """
    global nvml_lib

    lib_load_lock.acquire()
    try:
        # check if the library is loaded
        if nvml_lib is not None:
            return

        if sys.platform[:3] == "win":
            try:
                # The file nvml.dll maybe in the `C:/Windows/System32` or
                # the`C:/Program Files/NVIDIA Corporation/NVSMI`.
                nvml_lib = ctypes.CDLL(os.path.join(
                    os.getenv("WINDIR", "C:/Windows"), "System32/nvml.dll"))
            except OSError:
                nvml_lib = ctypes.CDLL(
                    os.path.join(os.getenv("ProgramFiles", "C:/Program Files"), "NVIDIA Corporation/NVSMI/nvml.dll"))
        else:
            nvml_lib = ctypes.CDLL("libnvidia-ml.so.1")

        if nvml_lib is None:
            _check_return(NVML_ERROR_LIBRARY_NOT_FOUND)

    except OSError:
        _check_return(NVML_ERROR_LIBRARY_NOT_FOUND)
    finally:
        lib_load_lock.release()


def nvmlErrorString(result):
    """Convert NVML error codes into readable strings.
    """
    fn = _get_nvml_function("nvmlErrorString")
    fn.restype = ctypes.c_char_p
    return fn(result)


def nvmlInit(flags=0):
    _load_nvml_library()

    fn = _get_nvml_function("nvmlInitWithFlags")
    ret = fn(flags)
    _check_return(ret)
    return True


def nvmlShutdown():
    fn = _get_nvml_function("nvmlShutdown")
    ret = fn()
    _check_return(ret)
    return None


def nvmlDeviceGetHandleByIndex(index):
    c_index = ctypes.c_uint(index)
    device = c_nvmlDevice_t()
    fn = _get_nvml_function("nvmlDeviceGetHandleByIndex_v2")
    ret = fn(c_index, ctypes.byref(device))
    _check_return(ret)
    return device


def nvmlDeviceGetUtilizationRates(handle):
    c_util = c_nvmlUtilization_t()
    fn = _get_nvml_function("nvmlDeviceGetUtilizationRates")
    ret = fn(handle, ctypes.byref(c_util))
    _check_return(ret)
    return c_util


def nvmlDeviceGetName(handle):
    c_name = ctypes.create_string_buffer(NVML_DEVICE_NAME_BUFFER_SIZE)
    fn = _get_nvml_function("nvmlDeviceGetName")
    ret = fn(handle, c_name, ctypes.c_uint(NVML_DEVICE_NAME_BUFFER_SIZE))
    _check_return(ret)
    return c_name.value


def nvmlDeviceGetCount():
    c_count = ctypes.c_uint()
    fn = _get_nvml_function("nvmlDeviceGetCount_v2")
    ret = fn(ctypes.byref(c_count))
    _check_return(ret)
    return c_count.value


def nvmlDeviceGetMemoryInfo(handle):
    c_memory = c_nvmlMemory_t()
    fn = _get_nvml_function("nvmlDeviceGetMemoryInfo")
    ret = fn(handle, ctypes.byref(c_memory))
    _check_return(ret)
    return c_memory
