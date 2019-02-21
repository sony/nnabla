Operator support status
=======================

Support status importing from ONNX
----------------------------------

This is a status list of [ONNX operators](https://github.com/onnx/onnx/blob/master/docs/Operators.md)
that indicates if each operator can be converted to NNP.

- OK The ONNX operator can map to a NNabla operator.
- Not test means the ONNX operator has not been verified. It might be one of the following cases:
    - The ONNX operator hasn't been checked if it can be converted to NNabla.
    - The ONNX operator has been checked if it can be converted to NNabla, but the implementation has not started.
    - The solution is not perfect/finished, for example, the operator can map to a combination of NNabla operators.
    - Hard to find a solution with existing NNabla operators.

Total 44/108

As the following table, Opset column means the maximal opset version are supported to convert to NNP.
In user's model, if there is any function opset version exceed the maximal opset(as the following table), the importer
might fail to convert NNP model due to this function.

======================================== =============== =============== =================================================
Operator                                 Opset           Status          Description
======================================== =============== =============== =================================================
Abs                                      1,6             OK
Add                                      1,6,7           OK              broadcast will be converted to a BroadcastTo
And                                      1,7             OK              broadcast will be converted to a BroadcastTo
ArgMax                                                   Unimplemented   Operator does not exist in NNabla
ArgMin                                                   Unimplemented   Operator does not exist in NNabla
AveragePool                              1,7             OK              autopad not supported. pads must have same
                                                                         value for begin and end.
BatchNormalization                       1,6,9           OK              is_test=false not supported (only inference)
Cast                                                     Unimplemented   Operator does not exist in NNabla(No type
                                                                         information is exposed in NNP)
Ceil                                                     Unimplemented   Should map to Ceil
Clip                                     6               OK              Converted to Identity, MaximumScalar,
                                                                         MinimumScalar, or both depending on the attribute
Concat                                   1,4             OK
Constant                                 1,9             OK              Converted to an input parameter
Conv                                     1               OK              auto_pad not supported. pads must have same value
                                                                         for begin and end.
ConvTranspose                                            Unimplemented   Should map to Deconvolution?
DepthToSpace                                             Unimplemented   Operator does not exist in NNabla
Div                                      1,6,7           OK              broadcast will be converted to a BroadcastTo
Dropout                                  1,6,7           OK              mask output will be removed since NNabla does
                                                                         not produce mask output.
Elu                                      1,6             OK
Equal                                    1,7             OK              broadcast will be converted to a BroadcastTo.
                                                                         Input data type will all be converted to int64
                                                                         since NNP does not have type information
Exp                                      1,6             OK
Flatten                                                  Unimplemented   Operator does not exist in NNabla
Floor                                                    Unimplemented   Should map to Floor
GRU                                                      Unimplemented   Operator does not exist in NNabla
Gather                                                   Unimplemented   Operator does not exist in NNabla
Gemm                                     1,6,7,9         OK              alpha and beta is not supported.
                                                                         Input A and B must be two dimensional,
                                                                         and input C must be one dimensional.
                                                                         transA, transB will be converted to
                                                                         a separate transpose operator
GlobalAveragePool                        1               OK
GlobalLpPool                                             Unimplemented   Operator does not exist in NNabla
GlobalMaxPool                                            Unimplemented   Operator does not exist in NNabla
Greater                                  1,7,9           OK              broadcast will be converted to a BroadcastTo
HardSigmoid                                              Unimplemented   Should be able to map to
                                                                         MulScalar+AddScalar+MinimumScalar+ReLU
Hardmax                                                  Unimplemented   Operator does not exist in NNabla
Identity                                 1               OK
InstanceNormalization                                    Unimplemented   Operator does not exist in NNabla
LRN                                      1               OK              Converted to
                                                                         PowScalar+Tranpose+SumPooling+Transpose+MulScalar+AddScalar+PowScalar.
                                                                         Currently only odd size is allowed.
LSTM                                                     Unimplemented
LeakyRelu                                1,6             OK
Less                                     1,7,9           OK              broadcast will be converted to a BroadcastTo
Log                                      1,6             OK
LogSoftmax                               1               Not test        Converted to Exp+Sum+Log+Sub2.
                                                                         Only works on input shape like N*C*1*1
LpNormalization                                          Unimplemented   Operator does not exist in NNabla
LpPool                                                   Unimplemented   Operator does not exist in NNabla
MatMul                                   1,9             OK
Max                                      1,6,8           OK              Only input of two tensors is currently supported
MaxPool                                  1,8             OK              auto_pad is not supported.
                                                                         pads must have same value for begin and end.
MaxRoiPool                                               Unimplemented   Operator does not exist in NNabla
Mean                                     1,6,8           Not test        Operator does not exist in NNabla
Min                                      1,6,8           OK              Only input of two tensors is currently supported
Mul                                      1,6,7           OK              broadcast will be converted to a BroadcastTo
Neg                                      1,6             Not test        Converted to MulScalar
Not                                      1               OK
Or                                       1,7             OK              broadcast will be converted to a BroadcastTo
PRelu                                    1,6             OK
Pad                                      1,2             Not test        For NNP to ONNX conversion, input buffer's
                                                                         dimension is assumed to be 4D if the shape cannot be determined.
Pow                                      1,7             OK              broadcast will be converted to a BroadcastTo
RNN                                                      Unimplemented   Operator does not exist in NNabla
RandomNormal                                             Unimplemented   Should be able to map to Randn
RandomNormalLike                                         Unimplemented   Operator does not exist in NNabla
RandomUniform                                            Unimplemented   Should be able to map to Rand
RandomUniformLike                                        Unimplemented   Operator does not exist in NNabla
Reciprocal                               1,6             Not test        Converted to RDivScalar
ReduceL1                                                 Unimplemented   Operator does not exist in NNabla
ReduceL2                                                 Unimplemented   Operator does not exist in NNabla
ReduceLogSum                                             Unimplemented   Operator does not exist in NNabla
ReduceLogSumExp                                          Unimplemented   Operator does not exist in NNabla
ReduceMax                                1               Not test
ReduceMean                               1               OK
ReduceMin                                1               Not test
ReduceProd                               1               Not test
ReduceSum                                1               OK
ReduceSumSquare                                          Unimplemented   Operator does not exist in NNabla
Relu                                     1,6             OK
Reshape                                  1,5             Not test        Not completedly supported.
Selu                                     1,6             OK
Sigmoid                                  1,6             OK
Size                                                     Unimplemented   Operator does not exist in NNabla
Slice                                                    Unimplemented   Operator does not exist in NNabla
Softmax                                  1               OK              Only works on input shape like N*C*1*1
Softplus                                 1               OK              Converted to Exp + AddScalar + Log
Softsign                                 1               OK              Converted to Abs + AddScalar + Div2
SpaceToDepth                                             Unimplemented   Operator does not exist in NNabla
Split                                                    Unimplemented   Operator does not exist in NNabla
Sqrt                                                     Unimplemented   Operator does not exist in NNabla
Squeeze                                                  Unimplemented   Operator does not exist in NNabla
Sub                                      1,6,7           OK              broadcast will be converted to a BroadcastTo
Sum                                      1,6,8           OK              Supporting two inputs only
Tanh                                     1,6             OK
Tile                                                     Unimplemented   Operator does not exist in NNabla
TopK                                                     Unimplemented   Operator does not exist in NNabla
Transpose                                1               OK
Unsqueeze                                                Unimplemented   Operator does not exist in NNabla
Xor                                      1,7             OK              broadcast will be converted to a BroadcastTo
experimental ATen                                        Unimplemented
experimental Affine                                      Unimplemented
experimental ConstantFill                                Unimplemented
experimental Crop                                        Unimplemented
experimental FC                                          Unimplemented
experimental GRUUnit                                     Unimplemented
experimental GivenTensorFill                             Unimplemented
experimental If                                          Unimplemented
experimental ImageScaler                                 Unimplemented
experimental Loop                                        Unimplemented
experimental LoopIndexTensor                             Unimplemented
experimental MeanVarianceNormalization                   Unimplemented
experimental ParametricSoftplus                          Unimplemented
experimental Scale                                       Unimplemented
experimental ScaledTanh                                  Unimplemented
experimental ThresholdedRelu                             Unimplemented
experimental Upsample                                    Unimplemented
======================================== =============== =============== =================================================

Support status exporting to ONNX
----------------------------------

The column of opset means which opset version can be converted to. For example, if Affine() has opset 6,9,
that means Affine() can be converted to both opset version 6 and opset version 9. Users may define which 
opset version to export by nnabla_cli command line parameters.

Total 45/136

Neural Network Layer
++++++++++++++++++++

Count 4/11

======================================== =============== =============== =================================================
Operator                                 Opset           Status          Description
======================================== =============== =============== =================================================
Affine                                   6,9             Not test        Implemented by Reshape,Flatten,Gemm
Convolution                              6,9             OK              Implemented by Conv
DepthwiseConvolution                     6,9             Not test        Implemented by Conv
Deconvolution                            6,9             Not test        Implemented by ConvTranspose,Add
DepthwiseDeconvolution                                   Not test        Not implemented
MaxPooling                               6,9             OK              Implemented by MaxPool
AveragePooling                           6,9             OK              Implemented by AveragePool
GlobalAveragePooling                     6,9             OK              Implemented by GlobalAveragePool
SumPooling                               6,9             Not test        Implemented by Mul
Unpooling                                6,9             Not test        Implemented by Upsample
Embed                                                    Not test        Not implemented
======================================== =============== =============== =================================================

Neural Network Activation Functions
+++++++++++++++++++++++++++++++++++

Count 8/11

======================================== =============== =============== =================================================
Operator                                 Opset           Status          Description
======================================== =============== =============== =================================================
Sigmoid                                  6,9             OK              Implemented by Sigmoid
Swish                                                    Not test        Not implemented
Tanh                                     6,9             OK              Implemented by Tanh
ReLU                                     6,9             OK              Implemented by Relu
LeakyReLU                                6,9             OK              Implemented by LeakyRelu
Softmax                                  6,9             OK              Implemented by Softmax
ELU                                      6,9             OK              Implemented by ELU
SELU                                     6,9             OK              Implemented by SELU
CReLU                                                    Not test        Not implemented
CELU                                                     Not test        Not implemented
PReLU                                    6,9             OK              Implemented by PRelu
======================================== =============== =============== =================================================

Normalization
+++++++++++++

Count 1/4

======================================== =============== =============== =================================================
Operator                                 Opset           Status          Description
======================================== =============== =============== =================================================
BatchNormalization                       6,9             OK              Implemented by InstanceNormalization,BatchNormalization
MeanSubtraction                                          Not test        Not implemented
ClipGradByValue                                          Not test        Not implemented
ClipGradByNorm                                           Not test        Not implemented
======================================== =============== =============== =================================================

Reduction
+++++++++

Count 5/7

======================================== =============== =============== =================================================
Operator                                 Opset           Status          Description
======================================== =============== =============== =================================================
Sum                                      6,9             OK              Implemented by ReduceSum
Mean                                     6,9             OK              Implemented by ReduceMean
Max                                      6,9             OK              Implemented by ReduceMax
Min                                      6,9             OK              Implemented by ReduceMin
Prod                                     6,9             OK              Implemented by ReduceProd
ReduceSum                                                Not test        Not implemented
ReduceMean                                               Not test        Not implemented
======================================== =============== =============== =================================================

Arithmetic
++++++++++

Count 8/12

======================================== =============== =============== =================================================
Operator                                 Opset           Status          Description
======================================== =============== =============== =================================================
Add2                                     6,9             OK              Implemented by Add
BcAdd2                                                   Not test        Not implemented
Sub2                                     6,9             OK              Implemented by Sub
Mul2                                     6,9             OK              Implemented by Mul
Div2                                     6,9             OK              Implemented by Div
Pow2                                     6,9             OK              Implemented by Pow
AddScalar                                6,9             Not test        Implemented by Add
MulScalar                                6,9             OK              Implemented by Mul
PowScalar                                6,9             Partial OK      Implemented by Pow, opset_6 status is OK, opset_9 status is NG
RSubScalar                               6,9             Not test        Implemented by Sub
RDivScalar                               6,9             OK              Implemented by Div
RPowScalar                               6,9             Not test        Implemented by Pow
======================================== =============== =============== =================================================

Logical
+++++++

Count 11/24

======================================== =============== =============== =================================================
Operator                                 Opset           Status          Description
======================================== =============== =============== =================================================
Sign                                                     Not test        Not implemented
Minimum2                                 6,9             OK              Implemented by Min
Maximum2                                 6,9             OK              Implemented by Max
MinimumScalar                            6,9             OK              Implemented by Clip
MaximumScalar                            6,9             OK              Implemented by Clip
LogicalAnd                               6,9             OK              Implemented by And
LogicalOr                                6,9             OK              Implemented by Or
LogicalXor                               6,9             OK              Implemented by Xor
Equal                                    6,9             OK              Implemented by Equal
NotEqual                                                 Not test        Not implemented
GreaterEqual                                             Not test        Not implemented
Greater                                  6,9             OK              Implemented by Greater
LessEqual                                                Not test        Not implemented
Less                                     6,9             OK              Implemented by Less
LogicalAndScalar                                         Not test        Not implemented
LogicalOrScalar                                          Not test        Not implemented
LogicalXorScalar                                         Not test        Not implemented
EqualScalar                                              Not test        Not implemented
NotEqualScalar                                           Not test        Not implemented
GreaterEqualScalar                                       Not test        Not implemented
GreaterScalar                                            Not test        Not implemented
LessEqualScalar                                          Not test        Not implemented
LessScalar                                               Not test        Not implemented
LogicalNot                               6,9             OK              Implemented by Not
======================================== =============== =============== =================================================

Math
++++

Count 5/18

======================================== =============== =============== =================================================
Operator                                 Opset           Status          Description
======================================== =============== =============== =================================================
Constant                                                 Not test        Not implemented
Abs                                      6,9             OK              Implemented by Abs
Exp                                      6,9             OK              Implemented by Exp
Log                                      6,9             OK              Implemented by Log
Identity                                 6,9             OK              Implemented by Identity
BatchMatmul                              6,9             OK              Implemented by Matmul
Round                                                    Not test        Not implemented
Sin                                                      Not test        Not implemented
Cos                                                      Not test        Not implemented
Tan                                                      Not test        Not implemented
Sinh                                                     Not test        Not implemented
Cosh                                                     Not test        Not implemented
ASin                                                     Not test        Not implemented
ACos                                                     Not test        Not implemented
ATan                                                     Not test        Not implemented
ASinh                                                    Not test        Not implemented
ACosh                                                    Not test        Not implemented
ATanh                                                    Not test        Not implemented
======================================== =============== =============== =================================================

Array Manipulation
++++++++++++++++++

Count 3/13

======================================== =============== =============== =================================================
Operator                                 Opset           Status          Description
======================================== =============== =============== =================================================
Concatenate                              6,9             OK              Implemented by Concat
Split                                    6,9             Not test        Implemented by Split,Squeeze
Stack                                    6,9             Not test        Implemented by Unsqueeze,Concat
Slice                                    6,9             Not test        Implemented by Slice
Pad                                      6,9             OK              Implemented by Pad
Transpose                                6,9             OK              Implemented by Transpose
Broadcast                                                Not test        Not implemented
OneHot                                   6,9             Not test        Implemented by Flatten,Gather,Reshape
Flip                                     6,9             Not test        Implemented by Gather,Transpose,Identity
Shift                                                    Not test        Not implemented
Reshape                                  6,9             Not test        Implemented by Reshape
MatrixDiag                                               Not test        Not implemented
MatrixDiagPart                                           Not test        Not implemented
======================================== =============== =============== =================================================

Stochasticity
+++++++++++++

Count 0/10

======================================== =============== =============== =================================================
Operator                                 Opset           Status          Description
======================================== =============== =============== =================================================
Dropout                                  6,9             NG              Implemented by Dropout
TopKData                                                 Not test        Not implemented
TopKGrad                                                 Not test        Not implemented
Rand                                                     Not test        Not implemented
Randint                                                  Not test        Not implemented
Randn                                                    Not test        Not implemented
RandomCrop                                               Not test        Not implemented
RandomFlip                                               Not test        Not implemented
RandomShift                                              Not test        Not implemented
ImageAugmentation                                        Not test        Not implemented
======================================== =============== =============== =================================================

Loss Functions
++++++++++++++

Count 0/9

======================================== =============== =============== =================================================
Operator                                 Opset           Status          Description
======================================== =============== =============== =================================================
SigmoidCrossEntropy                                      Not test        Not implemented
BinaryCrossEntropy                                       Not test        Not implemented
SoftmaxCrossEntropy                                      Not test        Not implemented
CategoricalCrossEntropy                                  Not test        Not implemented
SquaredError                                             Not test        Not implemented
AbsoluteError                                            Not test        Not implemented
HuberLoss                                                Not test        Not implemented
EpsilonInsensitiveLoss                                   Not test        Not implemented
KLMultinomial                                            Not test        Not implemented
======================================== =============== =============== =================================================

Quantization Neural Network Layers
++++++++++++++++++++++++++++++++++

Count 0/10

======================================== =============== =============== =================================================
Operator                                 Opset           Status          Description
======================================== =============== =============== =================================================
BinarySigmoid                            6,9             Not test        Implemented by HardSigmoid
BinaryTanh                                               Not test        Not implemented
BinaryConnectAffine                                      Not test        Not implemented
BinaryConnectConvolution                 6,9             Not test        Implemented by Conv,Reshape
BinaryWeightAffine                                       Not test        Not implemented
BinaryWeightConvolution                                  Not test        Not implemented
INQAffine                                                Not test        Not implemented
INQConvolution                                           Not test        Not implemented
FixedPointQuantize                                       Not test        Not implemented
Pow2Quantize                                             Not test        Not implemented
======================================== =============== =============== =================================================

Validation
++++++++++

Count 0/3

======================================== =============== =============== =================================================
Operator                                 Opset           Status          Description
======================================== =============== =============== =================================================
TopNError                                                Not test        Not implemented
BinaryError                                              Not test        Not implemented
ConfusionMatrix                                          Not test        Not implemented
======================================== =============== =============== =================================================

Unsupported,SpecialUse
++++++++++++++++++++++

Count 0/4

======================================== =============== =============== =================================================
Operator                                 Opset           Status          Description
======================================== =============== =============== =================================================
VATNoise                                                 Not test        Not implemented
Unlink                                                   Not test        Not implemented
Sink                                                     Not test        Not implemented
NmsDetection2d                                           Not test        Not implemented
======================================== =============== =============== =================================================
