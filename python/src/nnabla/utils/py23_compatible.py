# Copyright 2018,2019,2020,2021 Sony Corporation.
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

__all__ = ['getargspec']

import inspect
import sys
from collections import namedtuple

# avoid deprecation warning for inspect.getargspec in python3
if sys.version_info.major == 3:
    ArgSpec = namedtuple(
        "ArgSpec", ("args", "varargs", "keywords", "defaults"))

    def getargspec_py3(func):
        """
        This function wraps insepct.getfullargspec and return namedtuple whose properties are same as the result of conventional inspect.getargspec.

        Currently, it is assumed that all functions have no keyword-only arguments.
        If the function which has keyword-only arguments is passed, this function dose not work correctly.

        """
        spec = inspect.getfullargspec(func)

        return ArgSpec(args=spec.args, varargs=spec.varargs, keywords=spec.varkw, defaults=spec.defaults)
    getargspec = getargspec_py3
else:
    getargspec = inspect.getargspec
