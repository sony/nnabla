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


from contextlib import contextmanager

# State of auto forward computation.
__auto_forward_state = False


@contextmanager
def auto_forward(auto=True):
    """
    Context for dynamic graph execution mode.

    Args:
        auto (bool): Whether forward computation is executed during a
            computation graph construction.

    Returns: bool

    """
    global __auto_forward_state
    prev = __auto_forward_state
    __auto_forward_state = auto
    try:
        yield
    finally:
        __auto_forward_state = prev


def get_auto_forward():
    """Get the state of automatic forward execution.

    When it is true, forward execution is invoked during a computation graph
    definition.

    Note:
        This is called by users usually.  
    """
    global __auto_forward_state
    return __auto_forward_state


def set_auto_forward(auto):
    """Set the default mode for automatic forward propagation.

    When it is set to `True` , forward propagation is invoked immediately when the computation graph
    is updated.

    Args:
        auto (bool): Whether forward computation is executed when the
            computation graph is updated.

    Returns: bool
    """
    global __auto_forward_state
    __auto_forward_state = auto
