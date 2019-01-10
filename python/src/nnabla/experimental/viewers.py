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

from collections import defaultdict
import json
import os
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

    """

    def __init__(self, format="png", verbose=False, fname_color_map=None, vname_color_map=None):
        """
        Args: 
          format (`str`): Image format used to save.
          verbose (`bool`): When set as True. The redundant information is also added. For example, the shape of a variable and arguments of a function. Default is False.
          fname_color_map (`dict`): Mapping of a function name to a color name. Color name should be one supported in the graphviz. For example, `fname_color_map = {"Convolution": "red", "Affine": "blue"}`. Default is None and a color is automatically set according to a type of function.
          vname_color_map (`dict`): Mapping of a variable name (:obj:`Variable`.name) to a color name. Color name should be one supported in the graphviz. For example, `fname_color_map = {"input": "blue", "pred": "red"}`. Default is None and pink is used for all variables.
        """
        try:
            from graphviz import Digraph
        except:
            raise ImportError("Install graphviz. `pip install graphviz.`")

        self._graph = Digraph(format=format)
        self._graph.attr("node", style="filled")
        self._verbose = verbose
        self._fname_color_map = fname_color_map
        self._vname_color_map = vname_color_map

        class Functor(object):
            def __init__(self, graph, verbose=False,
                         fname_color_map=fname_color_map,
                         vname_color_map=vname_color_map):
                self._var_idx = 0
                self._fname_to_idx = defaultdict(int)
                self._vhash_to_idx = defaultdict(int)
                self._graph = graph
                self._verbose = verbose
                self._fname_color_map = fname_color_map
                self._vname_color_map = vname_color_map

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

                return "lightgray"

            def _map_vname_to_color(self, vname):
                if self._vname_color_map is not None:
                    if vname in self._vname_color_map:
                        return self._vname_color_map[vname]
                return "pink"

            def _map_fname_to_idx(self, fname):
                self._fname_to_idx[fname] += 1
                return self._fname_to_idx[fname]

            def _map_vhash_to_idx(self, vhash):
                self._vhash_to_idx[vhash] += 1
                return self._vhash_to_idx[vhash]

            def _var_label(self, var):
                vname = var.name if var.name != "" else "v"
                if not self._verbose:
                    return vname
                return "{}\n({})".format(vname,
                                         "({}, need_grad={})".format(var.shape, var.need_grad))

            def _fun_label(self, fun):
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
                    self._graph.node(tail_name, label=self._var_label(i),
                                     shape=self._var_shape(),
                                     color=self._map_vname_to_color(i.name))
                # f => v
                for o in f.outputs:
                    hash_o = str(hash(o))
                    head_name = "{}-{}".format(hash_o,
                                               self._map_vhash_to_idx(hash_o))
                    self._graph.edge(fname, head_name)
                    self._graph.node(head_name, label=self._var_label(o),
                                     shape=self._var_shape(),
                                     color=self._map_vname_to_color(o.name))

                # f
                self._graph.node(fname, label=self._fun_label(f),
                                 shape=self._fun_shape(),
                                 color=self._map_fname_to_color(fname),
                                 fontcolor="white")

        self.functor = Functor

    def save(self, vleaf, fpath, cleanup=False):
        """Save the graph to a given file path.

        Args:
          vleaf (`nnabla.Variable`): End variable. All variables and functions which can be traversed from this variable are shown in the reuslt.
          fpath (`str`): The file path used to save. 
          cleanup (`bool`): Clean up the source file after rendering. Default is False.

        """
        func = self.functor(self._graph, self._verbose)
        vleaf.visit(func)
        self._graph.render(fpath, cleanup=cleanup)

    def view(self, vleaf, fpath=None, cleanup=True):
        """View the graph.

        Args:
          vleaf (`nnabla.Variable`): End variable. All variables and functions which can be traversed from this variable are shown in the reuslt.
          fpath (`str`): The file path used to save. 
          cleanup (`bool`): Clean up the source file after rendering. Default is True.

        """

        func = self.functor(self._graph, self._verbose)
        vleaf.visit(func)
        self._graph.view(fpath, cleanup=cleanup)
