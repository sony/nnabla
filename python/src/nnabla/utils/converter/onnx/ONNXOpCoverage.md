# Tracking operator coverage for ONNX to NNP

- :black_heart: The ONNX operator hasn't been checked if it can be converted to NNabla.
- :heart: The ONNX operator can map to a NNabla operator.
- :yellow_heart: The solution is not perfect/finished, for example, the operator can map to a combination of NNabla operators.
- :broken_heart: Hard to find a solution with existing NNabla operators.

| Operator | Status | Description |
|---|:---:|:---:|
|[Abs](https://github.com/onnx/onnx/blob/master/docs/Operators.md#Abs)|:black_heart:||
|Add||
|And||
|ArgMax||
|ArgMin||
|AveragePool||
|BatchNormalization||
|Cast||
|Ceil||
|Clip||
|Concat||
|Constant||
|Conv||
|ConvTranspose||
|DepthToSpace||
|Div||
|Dropout||
|Elu||
|Equal||
|Exp||
|Flatten||
|Floor||
|GRU||
|Gather||
|Gemm||
|GlobalAveragePool||
|GlobalLpPool||
|GlobalMaxPool||
|Greater||
|HardSigmoid||
|Hardmax||
|InstanceNormalization||
|LRN||
|LSTM||
|LeakyRelu||
|Less||
|Log||
|LogSoftmax||
|LpNormalization||
|LpPool||
|MatMul||
|Max||
|MaxPool||
|MaxRoiPool||
|Mean||
|Min||
|Mul||
|Neg||
|Not||
|Or||
|PRelu||
|Pad||
|Pow||
|RNN||
|RandomNormal||
|RandomNormalLike||
|RandomUniform||
|RandomUniformLike||
|Reciprocal||
|ReduceL1||
|ReduceL2||
|ReduceLogSum||
|ReduceLogSumExp||
|ReduceMax||
|ReduceMean||
|ReduceMin||
|ReduceProd||
|ReduceSum||
|ReduceSumSquare||
|Relu||
|Reshape||
|Selu||
|Sigmoid||
|Slice||
|Softmax||
|Softplus||
|Softsign||
|SpaceToDepth||
|Split||
|Sqrt||
|Squeeze||
|Sub||
|Sum||
|Tanh||
|Tile||
|Transpose||
|Xor||
|experimental ATen||
|experimental Affine||
|experimental ConstantFill||
|experimental Crop||
|experimental FC||
|experimental GRUUnit||
|experimental GivenTensorFill||
|experimental Identity||
|experimental ImageScaler||
|experimental MeanVarianceNormalization||
|experimental ParametricSoftplus||
|experimental Scale||
|experimental ScaledTanh||
|experimental ThresholdedRelu||
|experimental Upsample||

