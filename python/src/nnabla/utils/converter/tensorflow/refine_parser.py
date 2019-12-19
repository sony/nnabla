import tensorflow as tf
from ply import yacc
import re
from .refine_graph import Token, RefineGraph


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
        """
        p[0] = self.graph.conv_transpose(p[1:])

    def p_rf_p_relu(self, p):
        """ rf_p_relu  :    Relu Abs Sub Mul Mul Add
        """
        p[0] = self.graph.p_relu(p[1:])

    def p_rf_conv_bn(self, p):
        """ rf_conv_bn  :    rf_conv_2d Mul Add
        """
        p[0] = self.graph.conv_bn(p[1:])

    def p_rf_conv_2d(self, p):
        """ rf_conv_2d  :    Pad Transpose rf_split_stmt Conv2D Identity Add Transpose
                        |   Pad Transpose rf_split_stmt Conv2D Identity Transpose
                        |   Pad Transpose rf_split_stmt rf_conv2d_loop_stmt ConcatV2 Add Transpose
                        |   Pad Transpose rf_split_stmt rf_conv2d_loop_stmt ConcatV2 Transpose
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
        """
        p[0] = self.graph.bn(p[1:])

    def p_rf_affine(self, p):
        """ rf_affine    :    Reshape Reshape MatMul Mul Add Reshape
                         |    Reshape Reshape MatMul Mul Add Reshape Reshape Mul Add
        """
        p[0] = self.graph.affine(p[1:])

    def p_rf_binary_sigmoid(self, p):
        """ rf_binary_sigmoid     :    Greater Select
        """
        p[0] = self.graph.binary_sigmoid(p[1:])

    def p_error(self, p):
        if p:
            print("error: {}".format(p.type))
        else:
            print('Error at the end of input.')
