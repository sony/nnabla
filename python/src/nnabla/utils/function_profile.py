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


__all__ = ['profile']

import atexit
import cProfile
import pstats


def _null_condition(*args, **kw):
    return True


class FunctionProfile(object):
    '''Function profiler object.

    This is usually not directly used by users. It's created via
    :func:`profile`, and attached to a decorated
    function object as an attribute ``profiler``. See ``profile`` function for
    details.

    '''

    # A global flag to check if anyone is profiling
    profiling = False

    def __init__(self, fn, condition=None, profile_class=cProfile.Profile, print_freq=0, sort_keys=None, print_restrictions=None):
        self.fn = fn
        if condition is None:
            condition = _null_condition
        self.condition = condition
        self.profile_class = profile_class
        self.print_freq = print_freq
        self.reset_stats()
        # https://docs.python.org/2/library/profile.html#pstats.Stats.sort_stats
        if sort_keys is None:
            sort_keys = ('cumulative', 'time', 'calls')
        self.sort_keys = sort_keys
        if print_restrictions is None:
            print_restrictions = (40,)
        self.print_restrictions = print_restrictions
        atexit.register(self._atexit)

    def __call__(self, *args, **kw):
        if not self.condition(*args, **kw):
            return self.fn(*args, **kw)
        self.ncalls += 1

        if FunctionProfile.profiling:
            # Don't profile if Any other function is being profiled
            return self.fn(*args, **kw)

        profiler = self.profile_class()

        try:
            FunctionProfile.profiling = True
            return profiler.runcall(self.fn, *args, **kw)
        finally:
            FunctionProfile.profiling = False
            if self.stats is None:
                self.stats = pstats.Stats(profiler)
            else:
                self.stats.add(profiler)
            if self.print_freq and (self.ncalls % self.print_freq == 0):
                self.print_stats()

    def reset_stats(self):
        '''Manually reset the profiling statistics collected so far.
        '''
        self.stats = None
        self.ncalls = 0

    def print_stats(self, reset=True):
        '''Manually print profiling result.

        Args:
            reset (bool): If False is specified, the profiling statistics so
                far is maintained. If ``True`` (default),
                :obj:`~reset_stats`
                is called to reset the profiling statistics.

        '''
        if not self.ncalls:
            return

        stats = self.stats
        code = self.fn.__code__
        print('--- Function Profiling ---')
        print('File "{}", line {}, function {}'.format(
            code.co_filename,
            code.co_firstlineno,
            self.fn.__name__))
        stats.sort_stats(*self.sort_keys)
        stats.print_stats(*self.print_restrictions)
        print('--------------------------')
        if reset:
            self.reset_stats()

    def _atexit(self):
        try:
            self.print_stats()
        except ValueError:
            # To avoid error in pytest `capture.py`.
            print('Ignore atexit error during pytest.')
            print('--------------------------')


def profile(fn=None, condition=None, profile_class=cProfile.Profile, print_freq=0, sort_keys=None, print_restrictions=None):
    '''Decorating a function that is profiled with a Python profiler
    such as :class:`cProfile.Profile`.

    **Note**: ``function`` doesn't refer to :obj:`~nnabla.function.Function`.
    A Python function.

    Args:
        fn (function):
            A function that is profiled. If None is specified (default), it
            returns a new decorator function. It is used when you want to
            specify optional arguments of this decorator function.
        condition (function):
            A function object which takes the same inputs with the decorated
            function, and returns a boolean value. The decorated function is
            profiled only when the ``condition`` function returns ``True``.
            By default, it returns always `True`, hence profiling is performed
            everytime the decorated function is called.
        profile_class (class):
            A profiler class such as :obj:`cProfile.Profile` and
            :obj:`Profile.Profile`. The default value is
            :obj:`cProfile.Profile`.
        print_freq (int):
            The profiling result is printed at function calls with an interval
            specified by ``print_freq``. If 0 is specified (default), the
            profiling result is only printed at the end of the Python process
            unless ``decorated_func.profiler.print_stats()`` is called
            manually.
        sort_keys (iterable):
            A list or tuple of string, which is passed to
            :meth:`pstats.Stats.sort_stats` as arguments. The default is
            ``('cumulative', 'time', 'calls')``.
        print_restriction (iterable):
            A list or tuple which is passed to
            :meth:`pstats.Stats.print_stats` as arguments. The default
            value is ``(40,)``, which results in only 40 functions inside the
            decorated function are printed in the profiling result.

    Returns: function

        A decorated function. If ``fn`` is ``None``, a new decorator function
        with optional arguments specified.

    Example:

        By decorating a function as following, the profling result is printed
        at the end of the Python process.

        .. code-block:: python

            from nnabla.utils import function_profile

            @function_profile.profile
            def foo(a, b, c=None, d=None):
                ...

        If you want to manually print the profiling result so far, use
        :meth:`FunctionProfile.print_stats`
        of the :obj:`FunctionProfile` object
        attached to the decorated function as ``profiler`` attribute.

        .. code-block:: python

            foo.profiler.print_stats()

        If you want to profile the function only when a specific argument is
        passed to, use the ``condition`` argument as following.

        .. code-block:: python

            def profile_only_if_c_is_not_none(a, b, c=None, d=None):
                return c is not None

            @function_profile.profile(condition=profile_only_if_c_is_not_none)
            def foo(a, b, c=None, d=None):
                ...


    '''
    if fn is None:
        def new_decorator(fn):
            return profile(fn, condition=condition, profile_class=profile_class, print_freq=print_freq, sort_keys=sort_keys, print_restrictions=print_restrictions)
        return new_decorator
    profiler = FunctionProfile(fn, condition=condition, profile_class=profile_class,
                               print_freq=print_freq, sort_keys=sort_keys, print_restrictions=print_restrictions)

    def new_fn(*args, **kw):
        return profiler(*args, **kw)

    new_fn.__name__ = fn.__name__
    new_fn.__doc__ = fn.__doc__
    new_fn.__dict__ = fn.__dict__
    new_fn.__module__ = fn.__module__
    new_fn.profiler = profiler
    return new_fn
