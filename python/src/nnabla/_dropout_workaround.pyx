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

from _variable cimport Variable
from _dropout_workaround cimport get_dropout_mask as c_get_dropout_mask


def _get_dropout_mask(dropout_input):
    """ Danger. Do not call this as a user interface.

    Get the unlinked Variable holding a mask, meaning the externally 
    accessible member variable of Dropout.

    Args:
        dropout_input(:class:`nnabla.Variable`): The input variable of
                                                 nnabla.functions.dropout
    """
    cdef VariablePtr v = (<Variable?>dropout_input).get_varp().variable()
    cdef VariablePtr mask = c_get_dropout_mask(v)
    return Variable.create_from_cvariable(mask)

