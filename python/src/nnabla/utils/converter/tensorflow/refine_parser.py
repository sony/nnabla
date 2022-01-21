# Copyright 2019,2020,2021 Sony Corporation.
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
from ply import yacc

from .refine_graph import Token


class RefineParser:
    def __init__(self, graph):
        self.graph = graph
        self.tokens = Token.tokens
        self.parser = yacc.yacc(module=self,
                                start='rf_net',
                                debug=False)

    def parse(self):
        return self.parser.parse(input=None,
                                 lexer=self.graph,
                                 debug=False,
                                 tracking=False
                                 )

    def p_rf_net(self, p):
        """ rf_net   :    rf_layers
        """
        self.graph.set_layers(p[1])

    def p_rf_layers(self, p):
        """ rf_layers   :    rf_layers rf_layer_stmt
                        |    rf_layer_stmt
        """
        if len(p) == 3:
            p[1].append(p[2])
            p[0] = p[1]
        else:
            p[0] = [p[1]]

    def p_rf_layer_stmt(self, p):
        """ rf_layer_stmt   :  rf_conv_2d
                         |     rf_pool
                         |     rf_conv_bn
                         |     rf_p_relu
                         |     rf_conv_transpose
                         |     rf_bn
                         |     rf_affine
                         |     rf_binary_sigmoid
                         |     CommonFunc
                         |     Conv2D
                         |     Split
                         |     SplitV
                         |     MaxPool
                         |     AvgPool
                         |     Add
                         |     Pad
                         |     Mul
                         |     Identity
                         |     Transpose
                         |     Relu
                         |     Abs
                         |     Sub
                         |     ConcatV2
                         |     Slice
                         |     Conv2DBackpropInput
                         |     PadV2
                         |     MatMul
                         |     Reshape
                         |     Greater
                         |     Select
                         |     Mean
                         |     StopGradient
                         |     SquaredDifference
                         |     Rsqrt
                         |     AddV2
                         |     RealDiv
                         |     Sum
         """
        p[0] = p[1]

    def p_rf_split_stmt(self, p):
        """ rf_split_stmt     :    Split
                             |    SplitV
        """
        p[0] = p[1]

    def p_rf_conv2d_loop_stmt(self, p):
        """ rf_conv2d_loop_stmt     :   rf_conv2d_loop_stmt Conv2D
                                    |   Conv2D
        """
        p[0] = p[1]

    def p_rf_conv_transpose(self, p):
        """ rf_conv_transpose  :    Transpose rf_split_stmt Conv2DBackpropInput Slice Identity Add Transpose
                               |    Transpose rf_split_stmt Conv2DBackpropInput Slice Identity Transpose
                               |    Transpose rf_split_stmt Conv2DBackpropInput Pad Slice Identity Add Transpose
        """
        p[0] = self.graph.conv_transpose(p[1:])

    def p_rf_p_relu(self, p):
        """ rf_p_relu  :    Relu Abs Sub Mul Mul Add
                       |    Relu Abs Sub Mul Mul AddV2
        """
        p[0] = self.graph.p_relu(p[1:])

    def p_rf_conv_bn(self, p):
        """ rf_conv_bn  :    rf_conv_2d Mul Add
                        |    rf_conv_2d Mul AddV2
        """
        p[0] = self.graph.conv_bn(p[1:])

    def p_rf_conv_2d(self, p):
        """ rf_conv_2d  :    Pad Transpose rf_split_stmt Conv2D Identity Add Transpose
                        |   Pad Transpose rf_split_stmt Conv2D Identity Transpose
                        |   Pad Transpose rf_split_stmt rf_conv2d_loop_stmt ConcatV2 Add Transpose
                        |   Pad Transpose rf_split_stmt rf_conv2d_loop_stmt ConcatV2 Transpose
                        |   Transpose rf_split_stmt rf_conv2d_loop_stmt Identity Add Transpose
                        |   Transpose rf_split_stmt rf_conv2d_loop_stmt Identity Transpose
                        |   Transpose rf_split_stmt rf_conv2d_loop_stmt ConcatV2 Transpose
        """
        p[0] = self.graph.conv2d(p[1:])

    def p_rf_pool_stmt(self, p):
        """ rf_pool_stmt     :    MaxPool
                             |    AvgPool
        """
        p[0] = p[1]

    def p_rf_pool(self, p):
        """ rf_pool     :    Transpose rf_pool_stmt Transpose
                        |    PadV2 Transpose rf_pool_stmt Transpose
        """
        p[0] = self.graph.pool(p[1:])

    def p_rf_bn(self, p):
        """ rf_bn     :    Mul Add
                      |    Mean StopGradient SquaredDifference Mean Add Rsqrt Mul Mul Mul Sub Add
                      |    Mean Sub Mul Sum RealDiv Reshape Reshape AddV2 Rsqrt Mul Mul Mul Sub AddV2
                      |    Mean Sub Mul Sum RealDiv Reshape AddV2 Rsqrt Mul Mul Reshape Mul Sub AddV2
                      |    Mul AddV2
        """
        p[0] = self.graph.bn(p[1:])

    def p_rf_affine(self, p):
        """ rf_affine    :    Reshape Reshape MatMul Mul Add Reshape
                         |    Reshape Reshape MatMul Mul Add
                         |    Reshape Reshape MatMul Mul Add Reshape Reshape Mul Add
                         |    Reshape Reshape MatMul Mul AddV2
        """
        p[0] = self.graph.affine(p[1:])

    def p_rf_binary_sigmoid(self, p):
        """ rf_binary_sigmoid     :    Greater Select
        """
        p[0] = self.graph.binary_sigmoid(p[1:])

    def p_error(self, p):
        raise ValueError("Failed to optimize the model.")
