Operator support status
=======================

This is a status list of [ONNX operators](https://github.com/onnx/onnx/blob/master/docs/Operators.md)
that indicates if each operator can be converted to NNP.

- Not started The ONNX operator hasn't been checked if it can be converted to NNabla.
- Not implemented The ONNX operator has been checked if it can be converted to NNabla, but the implementation has not started.
- OK The ONNX operator can map to a NNabla operator.
- Not finished The solution is not perfect/finished, for example, the operator can map to a combination of NNabla operators.
- Not in NNabla Hard to find a solution with existing NNabla operators.

======================================== =============== =================================================
Operator                                 Status          Description	
======================================== =============== =================================================
Abs                                      OK
Add                                      Not finished    broadcast will be converted to a BroadcastTo
And                                      Not finished    broadcast will be converted to a BroadcastTo
ArgMax                                   Not in NNabla   Operator does not exist in NNabla
ArgMin                                   Not in NNabla   Operator does not exist in NNabla
AveragePool                              Not finished    autopad not supported. pads must have same
                                                         value for begin and end.
BatchNormalization                       Not finished    is_test=false not supported (only inference)
Cast                                     Not in NNabla   Operator does not exist in NNabla(No type
                                                         information is exposed in NNP)
Ceil                                     Not implemented Should map to Ceil
Clip                                     Not finished    Converted to Identity, MaximumScalar,
                                                         MinimumScalar, or both depending on the attribute
Concat                                   OK
Constant                                 Not finished    Converted to an input parameter
Conv                                     Not finished    auto_pad not supported. pads must have same value
                                                         for begin and end.
ConvTranspose                            Not implemented Should map to Deconvolution?
DepthToSpace                             Not in NNabla   Operator does not exist in NNabla
Div                                      Not finished    broadcast will be converted to a BroadcastTo
Dropout                                  Not finished    mask output will be removed since NNabla does
                                                         not produce mask output.
Elu                                      OK
Equal                                    Not finished    broadcast will be converted to a BroadcastTo.
                                                         Input data type will all be converted to int64
                                                         since NNP does not have type information
Exp                                      OK
Flatten                                  Not in NNabla   Operator does not exist in NNabla
Floor                                    Not implemented Should map to Floor
GRU                                      Not in NNabla   Operator does not exist in NNabla
Gather                                   Not in NNabla   Operator does not exist in NNabla
Gemm                                     Not finished    alpha and beta is not supported.
                                                         Input A and B must be two dimensional,
                                                         and input C must be one dimensional.
                                                         transA, transB will be converted to
                                                         a separate transpose operator
GlobalAveragePool                        OK
GlobalLpPool                             Not in NNabla   Operator does not exist in NNabla
GlobalMaxPool                            Not in NNabla   Operator does not exist in NNabla
Greater                                  Not finished    broadcast will be converted to a BroadcastTo
HardSigmoid                              Not implemented Should be able to map to
                                                         MulScalar+AddScalar+MinimumScalar+ReLU
Hardmax                                  Not in NNabla   Operator does not exist in NNabla
Identity                                 OK
InstanceNormalization                    Not in NNabla   Operator does not exist in NNabla
LRN                                      Not finished    Converted to
                                                         PowScalar+Tranpose+SumPooling+Transpose+MulScalar+AddScalar+PowScalar.
                                                         Currently only odd size is allowed.
LSTM                                     Not in NNabla   Operator does not exist in NNabla
LeakyRelu                                OK
Less                                     Not finished    broadcast will be converted to a BroadcastTo
Log                                      OK
LogSoftmax                               Not finished    Converted to Exp+Sum+Log+Sub2.
                                                         Only works on input shape like N*C*1*1
LpNormalization                          Not in NNabla   Operator does not exist in NNabla
LpPool                                   Not in NNabla   Operator does not exist in NNabla
MatMul                                   OK
Max                                      Not finished    Only input of two tensors is currently supported
MaxPool                                  Not finished    auto_pad is not supported.
                                                         pads must have same value for begin and end.
MaxRoiPool                               Not in NNabla   Operator does not exist in NNabla
Mean                                     Not in NNabla   Operator does not exist in NNabla
Min                                      Not finished    Only input of two tensors is currently supported
Mul                                      Not finished    broadcast will be converted to a BroadcastTo
Neg                                      Not finished    Converted to MulScalar
Not                                      OK
Or                                       Not finished    broadcast will be converted to a BroadcastTo
PRelu                                    OK
Pad                                      Not finished    For NNP to ONNX conversion, input buffer's
                                                         dimension is assumed to be 4D if the shape cannot be determined.
Pow                                      Not finished    broadcast will be converted to a BroadcastTo
RNN                                      Not in NNabla   Operator does not exist in NNabla
RandomNormal                             Not implemented Should be able to map to Randn
RandomNormalLike                         Not in NNabla   Operator does not exist in NNabla
RandomUniform                            Not implemented Should be able to map to Rand
RandomUniformLike                        Not in NNabla   Operator does not exist in NNabla
Reciprocal                               Not finished    Converted to RDivScalar
ReduceL1                                 Not in NNabla   Operator does not exist in NNabla
ReduceL2                                 Not in NNabla   Operator does not exist in NNabla
ReduceLogSum                             Not in NNabla   Operator does not exist in NNabla
ReduceLogSumExp                          Not in NNabla   Operator does not exist in NNabla
ReduceMax                                OK
ReduceMean                               OK
ReduceMin                                OK
ReduceProd                               OK
ReduceSum                                OK
ReduceSumSquare                          Not in NNabla   Operator does not exit in NNabla
Relu                                     OK
Reshape                                  Not finished    implementing
Selu                                     OK
Sigmoid                                  OK
Size                                     Not in NNabla   Operator does not exist in NNabla
Slice                                    Not in NNabla   Operator does not exist in NNabla
Softmax                                  Not finished    Only works on input shape like N*C*1*1
Softplus                                 Not finished    Converted to Exp + AddScalar + Log
Softsign                                 Not finished    Converted to Abs + AddScalar + Div2
SpaceToDepth                             Not in NNabla   Operator does not exist in NNabla
Split                                    Not in NNabla   Operator does not exist in NNabla
Sqrt                                     Not in NNabla   Operator does not exist in NNabla
Squeeze                                  Not in NNabla   Operator does not exist in NNabla
Sub                                      Not finished    broadcast will be converted to a BroadcastTo
Sum                                      Not finished    Supporting two inputs only
Tanh                                     OK
Tile                                     Not in NNabla   Operator does not exist in NNabla
TopK                                     Not in NNabla   Operator does not exist in NNabla
Transpose                                OK
Unsqueeze                                Not in NNabla   Operator does not exist in NNabla
Xor                                      Not finished    broadcast will be converted to a BroadcastTo
experimental ATen                        Not started
experimental Affine                      Not started
experimental ConstantFill                Not started
experimental Crop                        Not started
experimental FC                          Not started
experimental GRUUnit                     Not started
experimental GivenTensorFill             Not started
experimental If                          Not started
experimental ImageScaler                 Not started
experimental Loop                        Not started
experimental LoopIndexTensor             Not started
experimental MeanVarianceNormalization   Not started
experimental ParametricSoftplus          Not started
experimental Scale                       Not started
experimental ScaledTanh                  Not started
experimental ThresholdedRelu             Not started
experimental Upsample                    Not started
======================================== =============== =================================================
