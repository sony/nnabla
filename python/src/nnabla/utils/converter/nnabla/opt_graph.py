# Copyright 2022 Sony Group Corporation.
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
import nnabla as nn


class Token:
    tokens = (
        'Reshape',
        'BatchMatmul',
        'Add2',
        'CommonFunc'
    )

    def __init__(self, op):
        if op.type in Token.tokens:
            self.type = op.type
            self.value = op
        else:
            self.type = 'CommonFunc'
            self.value = op
        self.lineno = 0
        self.lexpos = 0

    def __repr__(self):
        if self.value:
            return "{}:{}".format(self.type, self.value.name)
        else:
            return "{}:{}".format(self.type, "None")


class Branch:
    pass


class Resolver:
    def resolve(self, old_graph, new_graph):
        pass


class AffineResolver(Resolver):
    def __init__(self, branch):
        self.branch = branch

    def resolve(self, ob):
        ob.clone_func(self.branch.batchmatmul.x)
        ob.clone_variable(self.branch.batchmatmul.w.inputs[0])
        if self.branch.add2 is not None:
            ob.clone_func(self.branch.add2.bias)
        func_name = ob.get_unique_name("MatmulAddToAffine")
        affine = nn.graph_def.ProtoFunction(None,
                                            'Affine',
                                            {'base_axis': 1},
                                            func_name,
                                            ob.new_graph)
        x = self.branch.batchmatmul.x.outputs[0]
        w = self.branch.batchmatmul.w.inputs[0]
        affine.inputs = [x, w]
        if self.branch.add2:
            b = self.branch.add2.bias.inputs[0]
            affine.inputs.append(b)
        affine.outputs = self.branch.outputs
        for n in affine.outputs:
            ob.clone_variable(n)
        ob.new_graph.functions[func_name] = affine


class OptimizerBuilder:
    """
    This class handles the parser event from opt_parser.
    When a target statement is matched, an event is triggered,
    we collected necessary information from this event and construct
    the branches as what we want to understand.
    """

    def __init__(self, nnp_graph):
        self.nnp_graph = nnp_graph
        self.token_list = []
        self.new_graph = nn.graph_def.ProtoNetwork(
            self.nnp_graph.owner(), self.nnp_graph.name, self.nnp_graph.batch_size)
        self.names = {}

    def get_unique_name(self, name):
        if name in self.names:
            ret_name = f"{name}_{self.names[name]}"
            self.names[name] += 1
        else:
            self.names[name] = 1
            ret_name = name
        return ret_name

    def prepare(self):
        for pf in self.nnp_graph.forward_sequence():
            self.token_list.append(Token(pf))

    def output(self):
        return self.new_graph

    def token(self):
        if self.token_list:
            tok = self.token_list.pop(0)
            return tok
        return None

    def rf_affine(self, op_list):
        """
        This function constructs branches from op_list.
        The knowledge of how to construct branches is known according
        to what we want to optimize.
        """
        def set_op(branch, attr, op):
            if getattr(branch, attr) is None:
                setattr(branch, attr, op)
            else:
                raise ValueError("Incorrect branch!")

        op_list = [op_list[0], *op_list[1], op_list[2]]
        op_list = [op_list[0],
                   op_list[1],
                   self.rf_mul_add_resolve([
                       self.rf_mul([op_list[2], op_list[3]]),
                       self.rf_add([op_list[4], op_list[5]])
                   ])
                   ]

        branch = op_list[2]
        for op in op_list[:2]:
            if branch.batchmatmul.batchmatmul.inputs[0] in op.outputs:
                set_op(branch.batchmatmul, "x", op)
            elif branch.batchmatmul.batchmatmul.inputs[1] in op.outputs:
                set_op(branch.batchmatmul, "w", op)
            if branch.add2 is not None:
                if branch.add2.add2.inputs[0] in op.outputs:
                    set_op(branch.add2, "x", op)
                elif branch.add2.add2.inputs[1] in op.outputs:
                    set_op(branch.add2, "bias", op)
        return AffineResolver(branch)

    def rf_mul(self, op_list):
        branch = Branch()
        branch.x = None
        branch.w = None
        branch.batchmatmul = op_list[1]
        reshape = op_list[0]
        if reshape.outputs[0] == branch.batchmatmul.inputs[0]:
            branch.x = reshape
        elif reshape.outputs[0] == branch.batchmatmul.inputs[1]:
            branch.w = reshape
        branch.outputs = branch.batchmatmul.outputs
        return branch

    def rf_add(self, op_list):
        branch = Branch()
        branch.x = None
        branch.bias = None
        branch.add2 = op_list[1]
        reshape = op_list[0]
        if reshape.outputs[0] == branch.add2.inputs[0]:
            branch.x = reshape
        elif reshape.outputs[0] == branch.add2.inputs[1]:
            branch.bias = reshape
        branch.outputs = branch.add2.outputs
        return branch

    def rf_mul_add_resolve(self, op_list):
        branch = Branch()
        branch.batchmatmul = op_list[0]
        branch.outputs = branch.batchmatmul.outputs
        branch.add2 = None
        if len(op_list) == 2:
            branch.add2 = op_list[1]
            branch.outputs = branch.add2.outputs
        return branch

    def rf_mul_add(self, op_list):
        return op_list

    def clone_variable(self, vname):
        self.get_unique_name(vname)
        if vname in self.nnp_graph.variables:
            self.new_graph.variables[vname] = self.nnp_graph.variables[vname].clone(
            )
        elif vname in self.nnp_graph.parameters:
            self.new_graph.parameters[vname] = self.nnp_graph.parameters[vname].clone(
            )

    def clone_func(self, op):
        for i in op.inputs:
            self.clone_variable(i)
        for o in op.outputs:
            self.clone_variable(o)
        self.get_unique_name(op.name)
        self.new_graph.functions[op.name] = op.clone(self.new_graph)

    def accept_layer(self, op):
        if isinstance(op, nn.graph_def.ProtoFunction):
            self.clone_func(op)
        elif isinstance(op, Resolver):
            op.resolve(self)
        elif isinstance(op, list):
            op_list = op
            for op in op_list:
                self.clone_func(op)
