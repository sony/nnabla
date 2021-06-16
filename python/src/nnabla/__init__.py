# Copyright 2017,2018,2019,2020,2021 Sony Corporation.
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

from .logger import logger
from . import _init  # Must be imported first
from ._init import (
    prefer_cached_array,
    reset_array_preference,
    array_classes,
    add_available_context,
    available_contexts
)
from ._version import (
    __version__,
    __author__,
    __email__,
    __build_number__
)
from .variable import Variable, Context
from ._nd_array import NdArray
from .parameter import (
    get_current_parameter_scope,
    parameter_scope, get_parameters, clear_parameters,
    load_parameters, save_parameters,
    no_grad)
from .context import (
    context_scope, set_default_context, get_current_context)
from .auto_forward import auto_forward, set_auto_forward, get_auto_forward
from .recompute import recompute, recompute_fn, set_global_recompute
from ._computation_graph import forward_all
from .grad import grad
from .callback import (
    set_function_pre_hook,
    set_function_post_hook,
    unset_function_pre_hook,
    unset_function_post_hook,
    set_solver_pre_hook,
    set_solver_post_hook,
    unset_solver_pre_hook,
    unset_solver_post_hook
)

from .core.module import Module
from .core import graph_def
from .core.graph_def import ProtoVariable
from .core.sequential import Sequential

from .random import seed

# Prefer cached array by default for performance.
prefer_cached_array(True)
