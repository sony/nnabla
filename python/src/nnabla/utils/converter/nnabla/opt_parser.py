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
from ply import yacc

from .opt_graph import Token


class OptimizerParser:
    def __init__(self, graph):
        self.graph = graph
        self.tokens = Token.tokens
        self.parser = yacc.yacc(module=self,
                                start='rf_net',
                                debug=False)
        self.layer_num = 0

    def parse(self):
        return self.parser.parse(input=None,
                                 lexer=self.graph,
                                 debug=False,
                                 tracking=False
                                 )

    def p_rf_net(self, p):
        """ rf_net   :    rf_layers
        """
        pass

    def p_rf_layers(self, p):
        """ rf_layers   :    rf_layers rf_layer_stmt
                        |    rf_layer_stmt
        """
        pass

    def p_rf_layer_stmt(self, p):
        """ rf_layer_stmt   :  rf_affine
                         |     rf_mul_add
                         |     BatchMatmul
                         |     Reshape
                         |     Add2
                         |     CommonFunc

        """
        self.graph.accept_layer(p[1])

    def p_rf_mul_add(self, p):
        """ rf_mul_add   :     Reshape Reshape BatchMatmul Reshape
         """
        p[0] = self.graph.rf_mul_add(p[1:])

    def p_rf_affine(self, p):
        """ rf_affine    :    Reshape rf_mul_add Add2
        """
        p[0] = self.graph.rf_affine(p[1:])

    def p_error(self, p):
        raise ValueError("Failed to optimize the model.")
