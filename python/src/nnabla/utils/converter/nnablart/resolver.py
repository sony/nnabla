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

from collections import OrderedDict

import nnabla as nn
import nnabla.functions as F
import numpy as np
from nnabla.utils import nnabla_pb2
from nnabla.utils.save_function import _create_function_nntxt


class ProtoGenerator:
    def __init__(self, model_name, names):
        if names is not None:
            self.names = {v.data: k for k, v in names.items()}
        else:
            self.names = {}
        self.model_name = model_name
        self.params = {v.data: k for k, v in nn.get_parameters().items()}
        self.names.update(self.params)
        self.functions = []
        self.variables = {}
        self.fork_name_number = 0

    def join_name(self, name):
        v_name = '_'.join([self.model_name, name, str(self.fork_name_number)])
        self.fork_name_number += 1
        return v_name

    def _create_variable(self, data, name):
        proto_variable = nnabla_pb2.Variable()
        shape = list(data.shape)
        proto_variable.name = name
        proto_variable.type = 'Buffer'
        proto_variable.shape.dim.extend(shape)

        return proto_variable

    def __call__(self, func):
        if str(func) == "Sink":
            return
        inputs = []
        outputs = []
        for v in func.inputs:
            if v.data in self.names:
                v_name = self.names[v.data]
            else:
                v_name = self.join_name(func.name + "_in")
                self.names[v.data] = v_name
            inputs.append(v_name)
            if v_name not in self.variables:
                var = self._create_variable(v.data, v_name)
                self.variables[v_name] = var

        for v in func.outputs:
            if v.data in self.names:
                v_name = self.names[v.data]
            else:
                v_name = self.join_name(func.name + "_out")
                self.names[v.data] = v_name
            outputs.append(v_name)
            if v_name not in self.variables:
                var = self._create_variable(v.data, v_name)
                self.variables[v_name] = var

        function = {
            'type': func.info.type_name,
            'inputs': inputs,
            'outputs': outputs,
            'args': func.info.args,
        }
        proto_func = nnabla_pb2.Function()
        _create_function_nntxt(proto_func, func.name, function)
        self.functions.append(proto_func)


class Resolver:
    def __init__(self, nnp):
        self._nnp = nnp
        self._functions = []
        self._variables = OrderedDict()
        self._parameters = OrderedDict()
        self._referred = set()
        self._unique_name = {}
        self._resolve_table = {
            "GRU": self.GRU
        }

    def _get_unique_name(self, f_name):
        if f_name in self._unique_name:
            self._unique_name[f_name] += 1
            return "{}_{}".format(f_name, self._unique_name[f_name])
        self._unique_name[f_name] = 0
        return f_name

    def _get_variable_name(self, var_name):
        return var_name

    def _get_parameter(self, param_name):
        if param_name not in self._parameters:
            raise ValueError("parameter {} is not found.")
        pa = self._parameters[param_name]
        pa_np = np.array(pa.data[:]).reshape(pa.shape.dim[:])
        return pa_np

    def GRU(self, network, func):
        def register_parameters(h, w, b, with_bias):
            hidden_size = h.shape[1]
            w0, w1, w2 = (np.squeeze(wd, 0)
                          for wd in np.split(w, w.shape[0], axis=0))
            w0_nn = nn.Variable.from_numpy_array(np.transpose(w0, (1, 0)))
            w1_nn = nn.Variable.from_numpy_array(np.transpose(w1, (1, 0)))
            w2_0 = w2[:, :w2.shape[1] - hidden_size]
            w2_1 = w2[:, w2.shape[1] - hidden_size:]
            w2_0_nn = nn.Variable.from_numpy_array(np.transpose(w2_0, (1, 0)))
            w2_1_nn = nn.Variable.from_numpy_array(np.transpose(w2_1, (1, 0)))

            w_dict = {
                self._get_unique_name("@{}/gru/w0_nn".format(func_model_name)): w0_nn,
                self._get_unique_name("@{}/gru/w1_nn".format(func_model_name)): w1_nn,
                self._get_unique_name("@{}/gru/w2_0_nn".format(func_model_name)): w2_0_nn,
                self._get_unique_name("@{}/gru/w2_1_nn".format(func_model_name)): w2_1_nn
            }
            params.update(w_dict)
            names.update(w_dict)

            b0 = b1 = b2 = b3 = None
            if with_bias:
                b_dict = {self._get_unique_name("@{}/gru/b{}_nn".format(func_model_name, i)):
                          nn.Variable.from_numpy_array(np.squeeze(b_item, 0))
                          for i, b_item in enumerate(np.split(b, b.shape[0], axis=0))}
                b0, b1, b2, b3 = b_dict.values()
                names.update(b_dict)
                params.update(b_dict)

            parameters_dict = {
                'w0_nn': w0_nn,
                'w1_nn': w1_nn,
                'w2_0_nn': w2_0_nn,
                'w2_1_nn': w2_1_nn,
                'b0': b0,
                'b1': b1,
                'b2': b2,
                'b3': b3,
            }
            return parameters_dict

        def gru(x, h, parameters_dict):
            xh = F.concatenate(*(x, h), axis=1)
            w0_nn = parameters_dict.get('w0_nn', None)
            w1_nn = parameters_dict.get('w1_nn', None)
            w2_0_nn = parameters_dict.get('w2_0_nn', None)
            w2_1_nn = parameters_dict.get('w2_1_nn', None)
            b0 = parameters_dict.get('b0', None)
            b1 = parameters_dict.get('b1', None)
            b2 = parameters_dict.get('b2', None)
            b3 = parameters_dict.get('b3', None)

            r_t = F.sigmoid(F.affine(xh, w0_nn, b0))
            z_t = F.sigmoid(F.affine(xh, w1_nn, b1))

            n_t = F.tanh(F.affine(x, w2_0_nn, b2) +
                         r_t * F.affine(h, w2_1_nn, b3))
            h_t = (1 - z_t) * n_t + z_t * h

            return h_t

        def create_fixed_length_gru(xs0, h0, w0, w, b, num_layers, num_directions, with_bias):
            # xs : [T, B, I]
            # h0 : [L, D, B, H]
            # c0 : [L, D, B, H]
            # w0 : [D, 3, H, I+H]
            # w : [L-1, D, 3, H, D * H + H]
            # b : [L, D, 3, H]

            batch_size = xs0.shape[1]
            hidden_size = h0.shape[3]

            if xs0.shape[0] == 1:
                xs = [xs0[0]]
            else:
                xs = F.split(xs0, axis=0)
            hn = []
            for i in range(num_layers):
                wi = w0
                if i > 0:
                    wi = w[i - 1]
                # wi : [D, 3, H, ?]
                # Forward direction
                hif = h0[i, 0]  # [B, H]
                wif = wi[0]
                bif = None
                if with_bias:
                    bif = b[i, 0]
                p_dict = register_parameters(hif, wif, bif, with_bias)
                hs = []
                for j, x in enumerate(xs):
                    # x : [B, I]
                    hif = gru(x, hif, p_dict)
                    hs.append(hif)
                hn.append(hif)

                if num_directions == 1:
                    xs = hs
                    continue

                # Backward direction
                hib = h0[i, 1]  # [B, H]
                wib = wi[1]
                bib = None
                if with_bias:
                    bib = b[i, 1]
                p_dict = register_parameters(hib, wib, bib, with_bias)
                for k, x, in enumerate(reversed(xs)):
                    j = len(xs) - 1 - k
                    # x : [B, I]
                    hib = gru(x, hib, p_dict)
                    hs[j] = F.concatenate(hs[j], hib, axis=1)
                hn.append(hib)
                xs = hs

            ys = xs  # list of [B, HD]
            ys = F.stack(*ys, axis=0)  # [T, B, HD]
            hn = F.reshape(F.stack(*hn, axis=0), (num_layers, num_directions,
                                                  batch_size, hidden_size))  # LD list of [B, H] --> [L, D, B, H]
            return ys, hn

        num_layers = func.gru_param.num_layers
        drop_out = func.gru_param.dropout  # no use
        bidirectional = func.gru_param.bidirectional
        training = func.gru_param.training  # no use
        num_directions = 2 if bidirectional else 1

        xs_nn = nn.Variable(self._variables[func.input[0]].shape.dim[:])
        h0_nn = nn.Variable(self._variables[func.input[1]].shape.dim[:])
        w0_np = self._get_parameter(func.input[2])
        w_np = None
        b_np = None
        with_bias = False
        if num_layers > 1:
            w_np = self._get_parameter(func.input[3])
            if len(func.input) == 5:
                b_np = self._get_parameter(func.input[4])
                with_bias = True
        else:
            if len(func.input) == 4:
                b_np = self._get_parameter(func.input[3])
                with_bias = True

        nn.graph_def.reset_default_graph()
        names = {
            func.input[0]: xs_nn,
            func.input[1]: h0_nn
        }
        params = {}

        func_model_name = self._get_unique_name("gru")

        ys, hn = create_fixed_length_gru(
            xs_nn, h0_nn, w0_np, w_np, b_np, num_layers, num_directions, with_bias)  # returns Variables
        names.update({
            func.output[0]: ys,
            func.output[1]: hn
        })

        output = F.sink(ys, hn)
        pg = ProtoGenerator(func_model_name, names)
        output.visit(pg)

        for _, proto_v in pg.variables.items():
            self._variables[proto_v.name] = proto_v

        for pv_name, pv in params.items():
            if pv_name in self._variables:
                self._variables[pv_name].type = "Parameter"
            parameter = self._nnp.protobuf.parameter.add()
            parameter.variable_name = pv_name
            parameter.shape.dim.extend(pv.shape)
            parameter.data.extend(np.array(pv.d).flatten().tolist())
            parameter.need_grad = pv.need_grad
            self._parameters[pv_name] = parameter

        for proto_f in pg.functions:
            self._default_resolver(network, proto_f)

    def _default_resolver(self, network, func):
        self._functions.append(func)
        for i in func.input:
            self._referred.add(i)
        for o in func.output:
            self._referred.add(o)

    def execute(self):
        for p in self._nnp.protobuf.parameter:
            self._parameters[p.variable_name] = p
        networks = []
        for network in self._nnp.protobuf.network:
            rebuild = False
            for func in network.function:
                if func.type in self._resolve_table:
                    rebuild = True
                    break

            self._variables = OrderedDict()
            self._functions = []
            for v in network.variable:
                self._variables[v.name] = v

            if not rebuild:
                networks.append(network)
                for func in network.function:
                    for i in func.input:
                        if self._variables[i].type == 'Parameter':
                            self._referred.add(i)
                continue

            for func in network.function:
                resolver = self._resolve_table.get(
                    func.type, self._default_resolver)
                resolver(network, func)

            network.ClearField('function')
            network.function.extend(self._functions)
            network.ClearField('variable')
            network.variable.extend(
                filter(lambda x: x.name in self._referred, self._variables.values()))
            networks.append(network)

        self._nnp.protobuf.ClearField('network')
        self._nnp.protobuf.network.extend(networks)

        self._nnp.protobuf.ClearField('parameter')
        self._nnp.protobuf.parameter.extend(
            filter(lambda x: x.variable_name in self._referred, self._parameters.values()))
        exe = self._nnp.protobuf.executor[0]
        exe.ClearField('parameter_variable')
        for name in self._parameters.keys():
            if name in self._referred:
                p = exe.parameter_variable.add()
                p.variable_name = name

        return self._nnp
