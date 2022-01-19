# Copyright 2020,2021 Sony Corporation.
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

from __future__ import absolute_import

import csv
import os
import time
from collections import OrderedDict
from contextlib import contextmanager
from importlib import import_module

import numpy as np
from nnabla.ext_utils import import_extension_module

from .base import FunctionHookCallbackBase


class TimeProfiler(FunctionHookCallbackBase):
    """
    An utility API to create function_hook callbacks to profile the execution time of each function.
    Passing ``ext_name`` and ``device_id``, you can define which device time you want to profile.
    If `ext_name` = "cuda" or "cudnn", then cudaEvent will be used to measure the execution time.
    For more information about cudaEvent, see the CUDA `document <https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__EVENT.html>`_.
    If `ext_name`="cpu" , then wall-clock-time on host will be used.

    Example:

    .. code-block:: python

        ext_name = "cpu"
        device_id = "0"

        from nnabla.ext_utils import get_extension_context
        ctx = get_extension_context(ext_name, device_id=device_id)
        nn.set_default_context(ctx)

        y = model(...)

        from nnabla.utils.inspection import TimeProfiler
        tp = TimeProfiler(ext_name=ext_name, device_id=device_id)

        for i in range(max_iter):
            # All results of executions under "forward" scope are registered as "forward" execution.
            with tp.scope("forward"):
                y.forward(function_pre_hook=tp.pre_hook, function_post_hook=tp.post_hook)

            # All results of executions under "backward" scope are registered as "backward" execution.
            with tp.scope("backward") as tp:
                y.backward(function_pre_hook=tp.pre_hook, function_post_hook=tp.post_hook)

            # All results are evaluated by passing scopes to .calc_elapsed_time().
            # Be sure to call calc_elapsed_time at each iteration, otherwise nothing is measured.
            tp.calc_elapsed_time(["forward", "backward", "summary"])

        # To output results on stdout, call instance as a function.
        tp()

        # To write out as csv file, call .to_csv().
        tp.to_csv(output_file_name)
    """

    def __init__(self, ext_name, device_id):
        """
        Args:
             ext_name (str): backend extension name (e.g. cpu, cuda, or cudnn)
             device_id (str): device id
        """
        if ext_name == "cpu":
            self.profiler = import_module(
                "nnabla.utils.inspection").CpuTimerCallback

        elif ext_name in ["cuda", "cudnn"]:
            self.profiler = import_extension_module(
                "cuda.utils.inspection").CudaEventTimerCallback

        else:
            # Unsupported extension.
            raise NotImplementedError(
                "Profiler for the extension '{}' is not implemented.".format(ext_name))

        self.ext_name = ext_name
        self.device_id = device_id
        self._scope_name = ""
        self.profilers = {}
        self.create_new_profiler("summary")

    def create_new_profiler(self, scope_name):
        self.profilers[scope_name] = self.profiler(
            ext_name=self.ext_name, device_id=self.device_id)

    @contextmanager
    def scope(self, scope_name):
        """
        Change a scope to aggregate results.
        This function is used as context (`with` statement),
         and all results under the context are labeled by ``scope_name``.
        In adttion to the execution time of each function,
        the elapsed times between entering and exiting the each context are also recorded
         and they are aggregated as "summary" scope.

        Args:
            scope_name (str): Scope name.
        """
        prev_scope = self._scope_name

        try:
            # create new profiler for the scope if not exist
            if scope_name not in self.profilers:
                self.create_new_profiler(scope_name)

            self.call_pre_hook("summary", scope_name)

            self._scope_name = scope_name

            yield self

        finally:
            self._scope_name = prev_scope
            self.call_post_hook("summary", scope_name)

    @property
    def pre_hook(self):
        """
        Get a callback for function_pre_hook.
        This function can be used like the example below:

        .. code-block:: python

            tp = TimeProfiler(..)
            with tp.scope("forward"):
                v.forward(function_pre_hook=tp.pre_hook())

            with tp.scope("backward"):
                v.backward(function_pre_hook=tp.pre_hook())

        """
        profiler = self.profilers[self._scope_name]

        def callback(key):
            profiler.pre_hook(key)

        return callback

    @property
    def post_hook(self):
        """
        Get a callback for function_post_hook.
        This function can be used like the example below:

        .. code-block:: python

            tp = TimeProfiler(..)
            with tp.scope("forward"):
                v.forward(function_post_hook=tp.post_hook())

            with tp.scope("backward"):
                v.backward(function_post_hook=tp.post_hook())

        """
        profiler = self.profilers[self._scope_name]

        def callback(key):
            profiler.post_hook(key)

        return callback

    def call_pre_hook(self, scope_name, key):
        if scope_name not in self.profilers:
            self.profilers[scope_name] = self.profiler(
                ext_name=self.ext_name, device_id=self.device_id)

        self.profilers[scope_name].pre_hook(key)

    def call_post_hook(self, scope_name, key):
        if scope_name not in self.profilers:
            raise ValueError(
                "profiler instance for '{0}' scope is not found. ".format(scope_name))

        self.profilers[scope_name].post_hook(key)

    def calc_elapsed_time(self, names=None):
        """
        Evaluate all elapsed times.
        Note that elapsed time is not recorded until calc_elapsed_time is called.

        Args:
            names (str or list of str): Scope name(s) to evaluate elapsed time.
        """
        if isinstance(names, str):
            names = [names]

        if names is not None:
            # get specified profilers` elapsed time
            for name in list(names):
                self.profilers[name].calc_elapsed_time()
        else:
            # get all profilers` elapsed time
            for profiler in self.profilers.values():
                profiler.calc_elapsed_time()

    def average(self, start_index=0):
        ret = {}
        for name, profiler in self.profilers.items():
            ret[name] = profiler.average(start_index)

        return ret

    def get_all_results(self):
        ret = {}
        for name, profiler in self.profilers.items():
            if len(profiler.results) == 0:
                continue

            ret[name] = profiler.results

        return ret

    def to_csv(self, out_dir="./", ignore_init=True):
        """
        Writes out to csv file. Output directory can be specified by `out_dir`.
        As default, the elapsed times of first iteration will be omitted.
        If you evaluate the first iteration as well, pass True to `ignore_init`.

        Args:
             out_dir (str): Output directory.
             ignore_init (bool): Ignore the result of the first iteration or not.
        """
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

        res_all = self.average(start_index=1 if ignore_init else 0)

        summary_out = [dict(res, scope=scope)
                       for scope, res in res_all["summary"].items()]

        with open(os.path.join(out_dir, "profile_summary.csv"), "w") as f:
            writer = csv.DictWriter(f, ["scope", "time", "scale", "num"])
            writer.writeheader()
            writer.writerows(summary_out)

        # below is basically for nnabla functions
        # key is identically distinguished by its id. (not str expression)
        keystr_to_count = {}  # count number based on id
        key_to_unique = {}  # id of key to unique str expression
        out_dict = {}

        for scope, scope_res in res_all.items():
            if scope == "summary":
                continue

            for key, key_res in scope_res.items():
                insert = {"{}_{}".format(
                    k, scope): v for k, v in key_res.items()}

                # same key has already existed. -> update
                if key in key_to_unique:
                    out_dict[key_to_unique[key]].update(insert)
                    continue

                # new key -> create
                count = keystr_to_count.get(str(key), 0)
                keystr_to_count[str(key)] = count + 1
                key_to_unique[key] = "{}_{}".format(str(key), count)

                out_dict[key_to_unique[key]] = insert

        # preparation for writing to csv
        detail_out = []
        header = set()
        unique_to_key = {v: k for k, v in key_to_unique.items()}
        for scope, res in out_dict.items():
            header.update([*res.keys()])
            detail_out.append(
                dict(res, scope=scope, original_key=unique_to_key[scope]))

        with open(os.path.join(out_dir, "profile_detail.csv"), "w") as f:
            writer = csv.DictWriter(
                f, ["scope", "original_key"] + list(header))
            writer.writeheader()
            writer.writerows(detail_out)

    def __call__(self, *args, **kwargs):
        """
        Outputs results on stdout.
        The results of the first iteration will be omitted.
        """
        ret = self.average(start_index=1)

        for name, results in ret.items():
            if len(results) == 0:
                continue

            print("### {} ###".format(name))
            for key, result in results.items():
                print("{}: {} [{}]".format(
                    key, result["time"], result["scale"]))
            print()


class CpuTimerCallback(FunctionHookCallbackBase):
    def __init__(self, ext_name, device_id):
        super(CpuTimerCallback, self).__init__()

        self.results = OrderedDict()
        self.device_id = str(device_id)
        self.ext_module = import_extension_module(ext_name)
        self.key_to_times = OrderedDict()

    @property
    def pre_hook(self):
        def callback(key):
            if key not in self.key_to_times:
                self.key_to_times[key] = [0, 0]

            self.key_to_times[key][0] = self.timer()

        return callback

    @property
    def post_hook(self):
        def callback(key):
            if key not in self.key_to_times:
                raise ValueError()

            self.ext_module.synchronize(device_id=self.device_id)
            self.key_to_times[key][1] = self.timer()

        return callback

    @staticmethod
    def timer():
        return time.time()

    def calc_elapsed_time(self):
        for key, (start_time, end_time) in self.key_to_times.items():
            elapsed = end_time - start_time

            if key not in self.results:
                self.results[key] = []

            self.results[key].append(elapsed)

    def average(self, start_index=0):
        ret = {}
        for key, times in self.results.items():
            ret[key] = {
                "time": np.mean(times[start_index:]) * (10**3),
                "num": len(times) - start_index,
                "scale": "ms"
            }

        return ret
