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

from __future__ import print_function
import nnabla as nn
import nnabla.initializer as I

import sys
import importlib
import time
from collections import namedtuple
import csv


Inspec = namedtuple("Inspec", ['shape', 'init', 'need_grad'])
Inspec.__new__.__defaults__ = (I.NormalInitializer(), True)
BenchmarkStat = namedtuple("Benchmark", ['mean_time', 'run_count'])


class Timer:

    """Timer.

    See :func:`Timer.lap()`.

    """

    def __init__(self):
        self.start = time.time()
        self.lap_time = self.start

    def lap(self):
        """Calculate lap time.

        Returns:
            float: Lap time. The duration from the previous call of ``lap()``
                 or initialization at first call.
            float: Total time. The duration from initialization.

        """
        now = time.time()
        lap_time = now - self.lap_time
        total_time = now - self.start
        self.lap_time = now
        return lap_time, total_time


class FunctionBenchmarkWriter:

    """Benchmark writer class.

    It writes benchmark statistics taken by :class:`FunctionBenchmark`
    in .ini file format. As it's not so readable,
    use :class:`FunctionBenchmarkCsvWriter` instead.

    Args:
        file (Python file object): The benchmark will be written to the file.

    """

    def __init__(self, file=sys.stdout):
        self.file = file
        self.write_header()

    def write_header(self):
        """Writing header function that is called at initialization.
        """
        pass

    def _write_a_stat(self, k, v):
        print('{} = ({:3.2f}, {})'.format(
            k, v.mean_time * 1000, v.run_count), file=self.file)

    def write(self, fb):
        """Write a single function benchmark.

        Args:
            fb (FunctionBenchmark): FunctionBenchmark class instance.
                Before passing to this, you should call ``fb.benchmark()``.

        """
        print('[{}.{}]'.format(fb.module, fb.func.__name__), file=self.file)
        print('class = {}'.format(fb.func_ins.name), file=self.file)
        print('inspecs = {}'.format(repr(fb.inspecs)), file=self.file)
        print('func_args = {}'.format(repr(fb.func_args)), file=self.file)
        print('func_kwargs = {}'.format(repr(fb.func_kwargs)), file=self.file)
        print('ext = ({}, {})'.format(
            repr(fb.ext), repr(fb.ext_kwargs)), file=self.file)
        if self.setup_stat is not None:
            self._write_a_stat('setup', self.setup_stat)
        if self.foward_stat is not None:
            self._write_a_stat('forward', self.forward_stat)
        if self.backward_stat is not None:
            self._write_a_stat('backward', self.backward_stat)


class FunctionBenchmarkCsvWriter(FunctionBenchmarkWriter):

    """CSV format benchmark writer.

    It writes benchmark statistics taken by :class:`FunctionBenchmark`
    in CSV file format.

    Args:
        file (Python file object): The benchmark will be written to the file.

    """

    def __init__(self, file=sys.stdout):
        self.file = file
        self.write_header()

    def write_header(self):
        fields = [
            'module', 'function', 'class', 'inspecs',
            'func_args', 'func_kwargs', 'ext', 'ext_kwargs',
            'setup [ms/run]', 'setup [run]',
            'forward [ms/run]', 'forward [run]',
            'backward [ms/run]', 'backward [run]',
        ]
        writer = csv.writer(self.file)
        writer.writerow(fields)

    def _stat_as_a_list(self, stat):
        return ['{:.2f}'.format(stat.mean_time * 1000), str(stat.run_count)]

    def write(self, fb):
        values = [
            fb.module, fb.func.__name__, fb.func_ins.name,
            repr(fb.inspecs),
            repr(fb.func_args), repr(fb.func_kwargs),
            fb.ext, repr(fb.ext_kwargs),
        ]
        values.extend(self._stat_as_a_list(fb.setup_stat))
        values.extend(self._stat_as_a_list(fb.forward_stat))
        if fb.backward_stat is not None:
            values.extend(self._stat_as_a_list(fb.backward_stat))
        writer = csv.writer(self.file)
        writer.writerow(values)


def create_inputs(inspecs):
    """Create input :obj:`nnabla.Variable` from :obj:`Inspec`.

    Args:
        inspecs (:obj:`list` of :obj:`Inspec`): A list of ``Inspec``.

    Returns:
        :obj:`list` of :obj:`nnabla.Variable`: Input variables.

    """
    ret = []
    for i in inspecs:
        v = nn.Variable(i.shape, need_grad=i.need_grad)
        v.d = i.init(v.shape)
        ret.append(v)
    return ret


class FunctionBenchmark:

    r"""Benchmarking a function of a parametric function.

    This will calculate time to execute setup, forward and backward
    methods of Function class.

    Args:
        func (function): It can be a function in either
            :module:`nnabla.functions` or
            :module:`nnabla.parametric_functions`.
        inspecs (:obj:`list` of :obj:`Inspec`): A list of ``Inspec``.
            They specify shape, initializer, and need_grad attributes.
        func_args (list): A list of function arguments passed to func.
        func_kwargs (dict): Keyword arguments passed to func.
        ext (str): Extension module, e.g. 'cuda', 'cudnn'.
        ext_kwargs (dict): Keyword arguments passed to extension APIs,
            e.g. ``context(*kw)``, ``synchronize(**kw).
        min_run (int): Minimum number of calling function

    Note:
        You should not pass any compositional function
        (a function constructed by multiple Functions) to ``func`` argument.
        Benchmark will take place only in the last function instance in the
        chain of a compositional function.

    """

    def __init__(self, func, inspecs, func_args, func_kwargs,
                 ext, ext_kwargs, min_run=1, min_time=1.0):
        nn.clear_parameters()
        self.inputs = None
        self.outputs = None
        self.func_ins = None
        self.setup_stat = None
        self.forward_stat = None
        self.backward_stat = None
        self.func = func
        self.module = func.__module__
        self.inspecs = inspecs
        self.inputs_f = create_inputs(inspecs)
        self.func_args = func_args
        self.func_kwargs = func_kwargs
        self.ext = ext
        self.ext_kwargs = ext_kwargs
        self.mod_ext = importlib.import_module(
            '.' + ext, 'nnabla_ext')
        self.ctx = self.mod_ext.context(**ext_kwargs)
        self.min_run = min_run
        self.min_time = min_time

    def _calc_benchmark_stat(self, f):
        timer = Timer()
        i = 0
        while True:
            f()
            i += 1
            if i >= self.min_run:
                _, elapsed = timer.lap()
                if elapsed > self.min_time:
                    break
        return BenchmarkStat(elapsed / i, i)

    def clear(self):
        """Clear computation graph internally kept.
        """
        self.inputs = None
        self.outputs = None
        self.func_ins = None

    def _setup(self, delete=True):
        """Create a function instance and execute setup.

        Args:
            delete (bool): Delete buffered variables.

        """
        if delete:
            self.clear()
        with nn.context_scope(self.ctx):
            outputs = self.func(
                *(self.inputs_f + self.func_args), **self.func_kwargs)
            if not hasattr(outputs, '__iter__'):
                self.outputs = [outputs]
            else:
                self.outputs = outputs
        self.func_ins = self.outputs[0].parent
        self.inputs = self.func_ins.inputs

    def _forward(self):
        """Execute forward.

        This must be called after ``setup()`` called.
        """
        self.func_ins.forward(self.inputs, self.outputs)

    def _backward(self):
        """Execute backward.

        This should be called after ``setup()`` and ``forward()`` called once.
        """
        self.func_ins.backward(self.inputs, self.outputs)

    def benchmark_setup(self):
        """Benchmark setup execution.
        """
        def f():
            self._setup()
            self.mod_ext.synchronize(**self.ext_kwargs)
        f()  # Ignore first
        self.setup_stat = self._calc_benchmark_stat(f)

    def benchmark_forward(self):
        """Benchmark forward execution.
        """
        self._setup()

        def f():
            self._forward()
            self.mod_ext.synchronize(**self.ext_kwargs)
        f()  # Ignore first
        self.forward_stat = self._calc_benchmark_stat(f)

    def _benchmark_backward(self):
        self._setup()
        self._forward()
        for o in self.outputs:
            o.grad.fill(1)

        def f():
            self._backward()
            self.mod_ext.synchronize(**self.ext_kwargs)
        f()  # Ignore first
        self.backward_stat = self._calc_benchmark_stat(f)

    def benchmark_backward(self):
        """Benchmark backward execution.

        Note:
            If backward execution throws any exception,
            this benchmark system considers the error is because the function
            doesn't support backward operation, then set the benchmark
            ``None``.

        """
        try:
            self._benchmark_backward()
        except RuntimeError as e:
            # Seems like not implemented.
            print(e)
            self.mod_ext.synchronize(**self.ext_kwargs)
            self.backward_stat = None

    def benchmark(self):
        """Do all benchmarks of setup, forward and backward.
        """
        self.benchmark_setup()
        self.benchmark_forward()
        self.benchmark_backward()

    def write(self, writer=FunctionBenchmarkCsvWriter()):
        """Write the function benchmark results using a writer class.

        The benchmark result will be written according to the format
        defined in a given writer class.

        Args:
            writer (FunctionBenchmarkWriter): Writer class.
                It is recommended to use :func:`FunctionBenchmarkCsvWriter`.

        """
        writer.write(self)
