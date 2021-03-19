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

import json
import os
from collections import defaultdict

import numpy as np


class SimpleGraph(object):
    """Simple Graph with GraphViz.

    Example: 

    .. code-block:: python

      import nnabla as nn
      import nnabla.functions as F
      import nnabla.parametric_functions as PF

      import nnabla.experimental.viewers as V

      # Model definition
      def network(image, test=False):
          h = image
          h /= 255.0
          h = PF.convolution(h, 16, kernel=(3, 3), pad=(1, 1), name="conv")
          h = PF.batch_normalization(h, name="bn", batch_stat=not test)
          h = F.relu(h)
          pred = PF.affine(h, 10, name='fc')
          return pred

      # Model
      image = nn.Variable([4, 3, 32, 32])
      pred = network(image, test=False)

      # Graph Viewer
      graph = V.SimpleGraph(verbose=False)
      graph.view(pred)
      graph.save(pred, "sample_grpah")


    If the parameters are module-scoped, for example, the ``pred`` comes from a
    module output, parameters should be obtained beforehand then passed to view():

    Example:

    .. code-block:: python

      import nnabla as nn
      import nnabla.functions as F
      from nnabla.core.modules import ConvBn

      import nnabla.experimental.viewers as V

      class TSTNetNormal(nn.Module):
          def __init__(self):
              self.conv_bn_1 = ConvBn(1)
              self.conv_bn_2 = ConvBn(1)

          def call(self, x1, x2):
              y1 = self.conv_bn_1(x1)
              y2 = self.conv_bn_2(x2)
              y = F.concatenate(y1, y2, axis=1)
              return y

      tnd = TSTNetNormal()

      v1 = nn.Variable((4, 3, 32, 32))
      v2 = nn.Variable((4, 3, 32, 32))

      ya = tnd(v1, v2)

      graph = V.SimpleGraph(verbose=False)
      graph.view(ya, params=tnd.get_parameters(grad_only=False))

    """

    def __init__(self, format="png", verbose=False, fname_color_map=None, vname_color_map=None):
        """
        Args: 
          format (`str`): Image format used to save.
          verbose (`bool`): When set as True. The redundant information is also added. For example, the shape of a variable and arguments of a function. Default is False.
          fname_color_map (`dict`): Mapping of a function name to a color name. Color name should be one supported in the graphviz. For example, `fname_color_map = {"Convolution": "red", "Affine": "blue"}`. Default is None and a color is automatically set according to a type of function.
          vname_color_map (`dict`): Mapping of a variable name (:obj:`Variable`.name) to a color name. Color name should be one supported in the graphviz. For example, `fname_color_map = {"input": "blue", "pred": "red"}`. Default is None and pink is used for all variables.
        """
        self._format = format
        self._verbose = verbose
        self._fname_color_map = fname_color_map
        self._vname_color_map = vname_color_map

        class Functor(object):
            def __init__(self, graph, verbose=False,
                         fname_color_map=fname_color_map,
                         vname_color_map=vname_color_map,
                         fun2scope=None, var2name=None):
                self._var_idx = 0
                self._fname_to_idx = defaultdict(int)
                self._vhash_to_idx = defaultdict(int)
                self._graph = graph
                self._verbose = verbose
                self._fname_color_map = fname_color_map
                self._vname_color_map = vname_color_map
                self._fun2scope = fun2scope
                self._var2name = var2name

            def _map_fname_to_color(self, fname):
                if self._fname_color_map is not None:
                    if fname in self._fname_color_map:
                        return self._fname_color_map[fname]
                    return "lightgray"

                if "Conv" in fname:
                    return "blue"
                if "Affine" in fname:
                    return "red"
                if "Norm" in fname:
                    return "orange"
                if "ReLU" in fname:
                    return "green"
                if "Pool" in fname:
                    return "lightblue2"
                if np.sum([x in fname for x in ["Add", "Sub", "Mul", "Div"]]):
                    return "purple"

                return "gray"

            def _map_vname_to_color(self, v, vname):
                if v.data in self._var2name:
                    return 'pink'
                if self._vname_color_map is not None:
                    if vname in self._vname_color_map:
                        return self._vname_color_map[vname]
                return "lightgray"

            def _map_fname_to_idx(self, fname):
                self._fname_to_idx[fname] += 1
                return self._fname_to_idx[fname]

            def _map_vhash_to_idx(self, vhash):
                self._vhash_to_idx[vhash] += 1
                return self._vhash_to_idx[vhash]

            def _var_label(self, var):
                vname = var.name if var.name != "" else "h"
                if self._var2name is not None and var.data in self._var2name:
                    return self._var2name[var.data]
                if not self._verbose:
                    return vname
                return "{}\n({})".format(vname,
                                         "({}, need_grad={})".format(var.shape, var.need_grad))

            def _fun_label(self, fun):
                if self._fun2scope is not None and fun in self._fun2scope:
                    return fun.name + '\n' + self._fun2scope[fun]
                if not self._verbose:
                    return fun.name
                return "{}\n({})".format(fun.name, json.dumps(fun.info.args))

            def _var_shape(self, ):
                if not self._verbose:
                    return "circle"
                return ""

            def _fun_shape(self, ):
                if not self._verbose:
                    return "square"
                return "box"

            def __call__(self, f):
                fname = "{}-{}".format(f.name,
                                       self._map_fname_to_idx(f.name))
                # v => f
                for i in f.inputs:
                    hash_i = str(hash(i))
                    tail_name = "{}-{}".format(hash_i,
                                               self._vhash_to_idx[hash_i])
                    self._graph.edge(tail_name, fname)
                    fillcolor = self._map_vname_to_color(i, i.name)
                    self._graph.node(tail_name, label=self._var_label(i),
                                     shape=self._var_shape(),
                                     color='black' if i.need_grad else fillcolor,
                                     fillcolor=fillcolor)
                # f => v
                for o in f.outputs:
                    hash_o = str(hash(o))
                    head_name = "{}-{}".format(hash_o,
                                               self._map_vhash_to_idx(hash_o))
                    self._graph.edge(fname, head_name)
                    fillcolor = self._map_vname_to_color(o, o.name)
                    self._graph.node(head_name, label=self._var_label(o),
                                     shape=self._var_shape(),
                                     color='black' if o.need_grad else fillcolor,
                                     fillcolor=fillcolor)

                # f
                self._graph.node(fname, label=self._fun_label(f),
                                 shape=self._fun_shape(),
                                 color=self._map_fname_to_color(fname),
                                 fontcolor="white")

        self.functor = Functor

    def save(self, vleaf, fpath, cleanup=False, format=None):
        """Save the graph to a given file path.

        Args:
          vleaf (`nnabla.Variable`): End variable. All variables and functions which can be traversed from this variable are shown in the reuslt.
          fpath (`str`): The file path used to save. 
          cleanup (`bool`): Clean up the source file after rendering. Default is False.
          format (str):
              Force overwrite ``format`` (``'pdf', 'png', ...)``) configuration.

        """
        graph = self.create_graphviz_digraph(vleaf, format=format)
        graph.render(fpath, cleanup=cleanup)

    def view(self, vleaf, fpath=None, cleanup=True, format=None, params=None):
        """View the graph.

        Args:
          vleaf (`nnabla.Variable`): End variable. All variables and functions which can be traversed from this variable are shown in the reuslt.
          fpath (`str`): The file path used to save. 
          cleanup (`bool`): Clean up the source file after rendering. Default is True.
          format (str):
              Force overwrite ``format`` (``'pdf', 'png', ...)``) configuration.
          params (dict):
              Parameter dictionary, which can be obtained by get_parameters() function. Default is None.
              If params is None, global parameters are obtained.

        """
        graph = self.create_graphviz_digraph(vleaf, params, format=format)
        graph.view(fpath, cleanup=cleanup)

    def create_graphviz_digraph(self, vleaf, params=None, format=None):
        '''
        Create a :obj:`graphviz.Digraph` object given the leaf variable of a
        computation graph.

        One of nice things of getting ``Digraph`` directly is that the drawn
        graph can be displayed inline in a Jupyter notebook as described in
        `Graphviz documentation <https://graphviz.readthedocs.io/en/stable/manual.html#jupyter-notebooks>`_.

        Args:
            vleaf (`nnabla.Variable`):
                End variable. All variables and functions which can be
                traversed from this variable are shown in the reuslt.
            params (dict):
                The parameters dictionary, it can be obtained by nn.get_parameters().
            format (str):
                Force overwrite ``format`` (``'pdf', 'png', ...)``) configuration.

        Returns: graphviz.Digraph

        '''
        from nnabla import get_parameters
        import copy
        try:
            from graphviz import Digraph
        except:
            raise ImportError("Install graphviz. `pip install graphviz.`")
        if format is None:
            format = self._format
        graph = Digraph(format=format)
        graph.attr("node", style="filled")

        if params is None:
            params = get_parameters(grad_only=False)
        var2name = {v.data: k for k, v in params.items()}
        fun2scope = {}
        var2postname = copy.copy(var2name)

        def fscope(f):
            names = [var2name[v.data] for v in f.inputs if v.data in var2name]
            if names:
                f_names = [os.path.dirname(names[0]), *names[1:]]
                c = os.path.commonprefix(f_names)
                fun2scope[f] = c
                for n in names:
                    var2postname[params[n].data] = n[len(c)+1:]
        vleaf.visit(fscope)
        func = self.functor(graph, self._verbose,
                            fun2scope=fun2scope, var2name=var2postname)
        vleaf.visit(func)
        return graph
