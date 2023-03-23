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

"""
This module implements a series of optimizer based on proto_graph object.
"""
from nnabla.logger import logger


def remove_function(pf, renamed):
    proto_network = pf.owner()
    input_name = None
    need_grad = False
    for pv_name in pf.inputs:
        pv = proto_network.variables[pv_name] \
            if pv_name in proto_network.variables \
            else proto_network.parameters[pv_name]
        parent = pv.parent
        required = filter(
            lambda k: proto_network.functions[k] != pf, pv.required)
        input_name = pv_name
        need_grad = pv.need_grad
        break

    for pv_name in pf.outputs:
        pv = proto_network.variables[pv_name]
        pv.parent = parent
        if len(pv.required) > 0:
            for r in pv.required:
                r_func = proto_network.functions[r]
                index = r_func.inputs.index(pv.name)
                r_func.inputs[index] = input_name

            pv.required += required
            if pv_name in proto_network.outputs:
                index = proto_network.outputs.index(pv_name)
                proto_network.outputs[index] = input_name
            renamed[pv.name] = input_name
            pv.name = input_name
            pv.need_grad = need_grad
            proto_network.variables[pv.name] = pv
            logger.info(
                f"[upper bound] proto_variable:{pv_name} is deleted, output name: {input_name} is kept.")
            del proto_network.variables[pv_name]
        else:
            if parent is not None:
                r_func = proto_network.functions[parent]
                index = r_func.outputs.index(input_name)
                r_func.outputs[index] = pv.name
            # Re-check input variable
            for f_name in required:
                u_func = proto_network.functions[f_name]
                index = u_func.inputs.index(input_name)
                u_func.inputs[index] = pv.name
            pv.need_grad = need_grad
            pv.required += required
            renamed[input_name] = pv.name
            del proto_network.variables[input_name]
            logger.info(
                f"[lower bound] proto_variable:{input_name} is deleted, output name: {pv.name} is kept.")
        break

    del proto_network.functions[pf.name]
    logger.info(f"proto_function:{pf.name} is deleted.")


class IdentityRemover:
    def __init__(self, renamed=None, is_required=None):
        self.renamed = renamed
        self.is_required = is_required

    def __call__(self, pf):
        if pf.type == 'Identity':
            if self.is_required:
                if not self.is_required(pf):
                    return
            remove_function(pf, self.renamed)
