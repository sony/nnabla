# Copyright 2018,2019,2020,2021 Sony Corporation.
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

from __future__ import division, print_function

import csv
import os
import sys
import time
from collections import namedtuple
from functools import partial

import nnabla as nn
from nnabla.ext_utils import import_extension_module

ProfileStat = namedtuple("ProfileStat", ["parameter_scope", "inputs_shape",
                                         "args_info", "function_name", "mean_time", "n_run"])


def convert_time_scale(sec, format="m"):
    converter = {"m": 1e3, "u": 1e6, "n": 1e9}

    if format not in converter.keys():
        raise NotImplementedError("format must be {}".format(converter.keys()))

    return sec * converter[format]


def _zero_variables(variables):
    for v in variables:
        if v.parent is None:
            continue
        v.data.zero()
        v.grad.zero()


class GraphProfilerCsvWriter:
    """GraphProfilerCsvWriter
    csv writer for GraphProfiler class.

    Example:

    .. code-block:: python

        from nnabla.utils.profiler import GraphProfiler, GraphProfilerCsvWriter

        # Network building comes above

        B = GraphProfiler(variable, solver=solver, device_id=0, ext_name=device, n_run=1000)
        B.run()

        with open("./profile.csv", "w") as f:
            writer = GraphProfilerCsvWriter(B, file=f)
            writer.write()

    Args:
        gb (:py:class:`GraphProfiler <nnabla.utils.profile.GraphProfiler>`):
            Instance of GraphProfiler class which is main executor of profiling.
        file (Python file object):
            Output file object.
            Profile results will be written to the file which is specified by this argument.
    """

    def __init__(self, gb, file=sys.stdout):
        self.file = file
        self.gb = gb

        self.fields = ["parameter_scope", "function_name", "inputs_shape",
                       "args_info", "forward", "backward", "forward_n_run", "backward_n_run"]

        self.write_header()

    def write_header(self):
        writer = csv.writer(self.file)

        writer.writerow(["num. of run", self.gb.n_run])
        writer.writerow(["device id", self.gb.device_id])
        writer.writerow(["ext name", self.gb.ext_name])
        writer.writerow(["time scale", self.gb.time_scale])
        writer.writerow(["nnabla version", nn.__version__])
        writer.writerow([])

        writer.writerow(self.fields)

    def check_same(self, f, b):
        for field in self.fields[:4]:
            if f[field] != b[field]:
                return False

        return True

    def write(self):
        """
        Write result to the file.
        The output file is specified by ``file``.
        """
        writer = csv.writer(self.file)

        for f, b in zip(self.gb.result["forward"], self.gb.result["backward"]):
            f = f._asdict()
            b = b._asdict()
            if not self.check_same(f, b):
                raise AssertionError()

            args_info = ", ".join(["{}: {}".format(k, v)
                                   for k, v in f["args_info"]])

            out = [f["parameter_scope"], f["function_name"], f["inputs_shape"], args_info,
                   f["mean_time"], b["mean_time"], f["n_run"], b["n_run"]]

            writer.writerow(out)

        writer.writerow([])
        writer.writerow(["forward all", self.gb.result["forward_all"]])
        writer.writerow(
            ["forward_all_n_run", self.gb.result["n_run_forward_all"]])

        writer.writerow([])
        writer.writerow(["backward all", self.gb.result["backward_all"]])
        writer.writerow(
            ["backward_all_n_run", self.gb.result["n_run_backward_all"]])

        if set(self.gb.result.keys()) >= {"training", "n_run_training"}:
            writer.writerow([])
            writer.writerow(
                ["training(forward + backward + update)", self.gb.result["training"]])
            writer.writerow(
                ["training_n_run", self.gb.result["n_run_training"]])


class GraphProfiler:
    """GraphProfiler
    Class for measuring calculation time of each functions which compose nnabla computation graph.

    You can check some performances of your nnabla network.
    This can measure the calculation times of :

    * function-wise forward
    * function-wise backward
    * whole graph forward
    * whole graph backward
    * training (forward + backward + update) (if ``solver`` is not None)

    Example:

    .. code-block:: python

        import nnabla as nn
        import nnabla.functions as F
        import nnabla.solvers as S
        from nnabla.utils.profiler import GraphProfiler

        # Set up nnabla context
        device = "cpu"  # you can also use GPU ("cudnn")
        ctx = get_extension_context(device)
        nn.set_default_context(ctx)

        # Network building
        x = nn.Variable(shape=...)
        t = nn.Variable(shape=...)
        y = CNN(x) # you can build not only CNN but any networks
        loss = F.mean(F.softmax_cross_entropy(y, t)) # any loss functions or variables can be used

        # solver setting
        solver = S.Sgd()
        solver.set_parameters(nn.get_parameters())

        # SOME CODE (data loading or so on)

        B = GraphProfiler(loss, solver=solver, device_id=0, ext_name=device, n_run=1000)
        B.run()

    Args:
        graph (:class:`nnabla.Variable`):
            Instance of `nnabla.Variable` class.
            GraphProfiler find all functions which compose network graph from root `nnabla.Variable` to this `nnabla.Variable`.
        device_id (str):
            gpu device id.
        ext_name (str):
            Extension name. e.g. 'cpu', 'cuda', 'cudnn' etc.
        solver (:class:`nnabla.solvers.Solver`):
            Instance of `nnabla.solvers.Solver` for optimizing the parameters of the computation graph.
            if None, the training process is ignored.
            Default value is None.
        n_run (int):
            This argument specifies how many times the each functions` execution time are measured.
            Default value is 100.
        max_measure_execution_time (float):
            Maximum time of executing measurement for each functions.
            This argument has higher priority than ``n_run``.
            When the measurement time for each functions get bigger than this argument,
            this class stops measuring and goes to next function, unless the total times of measurement are less than n_run.
            Default value is 1 [sec].
        time_scale (str):
            Time scale to display. ['m', 'u', 'n'] (which stands for 'mili', 'micro' and 'nano')
        backward_accum(bool):
            Accumulation flag passed to the each backward function. The flag will fulfill the all accumulation flags
            with the same value of backward_accum. This flag is only valid for the time measurement of each function.
            For whole graph comutation, the NNabla graph engine set the appropriate accumulation flags to functions.
            Pay attention to inplace flag for your graph because accumulation and inplace flags cannot be set
            at the same time. If even one inplace flag is true in your graph, this backward_accum must be false.
            Default value is False.
    """

    def __init__(self, graph, device_id, ext_name, solver=None, n_run=100, max_measure_execution_time=1,
                 time_scale="m", backward_accum=False):
        self.graph = graph
        # if solver is None, training time (forward + backward + update) is not calculated
        self.solver = solver
        self.n_run = n_run
        self.device_id = str(device_id)
        self.ext_name = ext_name
        self.ext_module = import_extension_module(self.ext_name)
        self.max_measure_execution_time = max_measure_execution_time
        self.time_scale = time_scale
        self.result = dict()
        self.name2val = {v: k for k, v in nn.get_parameters().items()}
        self.backward_accum = backward_accum

        if self.n_run < 1:
            raise AssertionError("n_run must be bigger than 1")

    def _measure_execution_time(self, execution, *execution_args):
        result = 0.
        measured_count = 0

        # warm-up
        execution(*execution_args)
        self.ext_module.synchronize(device_id=self.device_id)

        start_0 = time.time()
        for i in range(self.n_run):
            start = time.time()
            execution(*execution_args)
            self.ext_module.synchronize(device_id=self.device_id)
            stop = time.time()
            result += stop - start

            measured_count += 1

            # if elapsed time is greater than self.max_measure_execution_time, break loop.
            if stop - start_0 > self.max_measure_execution_time:
                break

        mean_time = result / measured_count
        mean_time = convert_time_scale(mean_time, format=self.time_scale)
        mean_time = "{:.8}".format(mean_time, self.time_scale)

        return mean_time, measured_count

    def _time_profiling(self, f, target_process):
        # Zero-ing to avoid invalid memory access in some layers
        # such as softmax cross entropy.
        _zero_variables(f.inputs)
        _zero_variables(f.outputs)

        if target_process is "forward":
            mean_time, measured_count = self._measure_execution_time(
                f.forward, f.inputs, f.outputs)
        elif target_process is "backward":
            accum = [self.backward_accum] * len(f.inputs)
            mean_time, measured_count = self._measure_execution_time(
                f.backward, f.inputs, f.outputs, accum)
        else:
            raise NotImplementedError(
                "target process must be [forward, backward]")

        # Releasing array memory to avoid the increasing memory usage
        # (`NdArray.zero()` releases any device memory internally.)
        _zero_variables(f.inputs)
        _zero_variables(f.outputs)

        parameter_scope = None
        if len(f.inputs) > 1:
            if f.inputs[1] in self.name2val.keys():
                parameter_scope = os.path.dirname(self.name2val[f.inputs[1]])

        inputs_shape = [x.shape for x in f.inputs]

        function_name = f.name

        args_info = f.info.args.items()

        self.result[target_process].append(ProfileStat(parameter_scope=parameter_scope,
                                                       inputs_shape=inputs_shape,
                                                       args_info=args_info,
                                                       function_name=function_name,
                                                       mean_time=mean_time,
                                                       n_run=measured_count))

    def time_profiling_forward(self):
        self.result["forward"] = list()
        func = partial(self._time_profiling, target_process="forward")
        self.graph.visit(func)

    def time_profiling_backward(self):
        self.result["backward"] = list()
        func = partial(self._time_profiling, target_process="backward")
        self.graph.visit(func)

    def training_function(self):
        self.graph.forward(clear_no_need_grad=True)
        self.solver.zero_grad()
        self.graph.backward(clear_buffer=True)
        self.solver.update()

    def time_profiling_whole_graph(self):
        self.result["forward_all"], self.result["n_run_forward_all"] = self._measure_execution_time(
            self.graph.forward)

        self.result["backward_all"], self.result["n_run_backward_all"] = self._measure_execution_time(
            self.graph.backward)

        if self.solver is not None:
            self.result["training"], self.result["n_run_training"] = self._measure_execution_time(
                self.training_function)

    def run(self):
        """
        Execute profiling.

        This executes the 5 types of measurement:

        * function-wise forward
        * function-wise backward
        * whole graph forward
        * whole graph backward
        * training (forward + backward + update) (if ``solver`` is not None.)
        """
        self.time_profiling_forward()
        self.time_profiling_backward()
        self.time_profiling_whole_graph()

    def get_result(self):
        return self.result

    def print_result(self):
        print("time scale: {}".format(self.time_scale))
        print("-----------------forward--------------------")
        for x in self.result["forward"]:
            print(x)
        print()

        print("-----------------backward--------------------")
        for x in self.result["backward"]:
            print(x)
        print()

        print("all forward: {}, all backward: {}".format(
            self.result["forward_all"], self.result["backward_all"]))
