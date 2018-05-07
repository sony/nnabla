# Tracking operator coverage for ONNX to NNP

This is a status list of [ONNX operators](https://github.com/onnx/onnx/blob/master/docs/Operators.md)
that indicates if each operator can be converted to NNP.

- :black_heart: The ONNX operator hasn't been checked if it can be converted to NNabla.
- :green_heart: The ONNX operator can map to a NNabla operator.
- :yellow_heart: The solution is not perfect/finished, for example, the operator can map to a combination of NNabla operators.
- :broken_heart: Hard to find a solution with existing NNabla operators.

- All operators have been tested with float tensors ONLY.

| Operator | Status | Description |
|---|:---:|:---:|
|Abs|:green_heart:||
|Add|:yellow_heart:|broadcast will be converted to a BroadcastTo|
|And|:yellow_heart:|broadcast will be converted to a BroadcastTo|
|ArgMax|:black_heart:||
|ArgMin|:black_heart:||
|AveragePool|:yellow_heart:|autopad not supported|
|BatchNormalization|:yellow_heart:|is_test=false not supported (only inference)|
|Cast|:black_heart:||
|Ceil|:black_heart:||
|Clip|:black_heart:||
|Concat|:green_heart:||
|Constant|:yellow_heart:|Converted to an input parameter|
|Conv|:yellow_heart:|auto_pad not supported|
|ConvTranspose|:black_heart:||
|DepthToSpace|:black_heart:||
|Div|:yellow_heart:|broadcast will be converted to a BroadcastTo|
|Dropout|:yellow_heart:|mask output will be removed since NNabla does not produce mask output.|
|Elu|:green_heart:||
|Equal|:yellow_heart:|broadcast will be converted to a BroadcastTo. Input data type will all be converted to int64 since NNP does not have type information|
|Exp|:green_heart:||
|Flatten|:black_heart:||
|Floor|:black_heart:||
|GRU|:black_heart:||
|Gather|:black_heart:||
|Gemm|:yellow_heart:|alpha and beta is not supported. Input A and B must be two dimensional, and input C must be one dimensional. transA, transB will be converted to a separate transpose operator|
|GlobalAveragePool|:green_heart:||
|GlobalLpPool|:black_heart:||
|GlobalMaxPool|:black_heart:||
|Greater|:yellow_heart:|broadcast will be converted to a BroadcastTo|
|HardSigmoid|:black_heart:||
|Hardmax|:black_heart:||
|InstanceNormalization|:black_heart:||
|LRN|:broken_heart:|Operator does not exist in NNabla|
|LSTM|:black_heart:||
|LeakyRelu|:green_heart:||
|Less|:yellow_heart:|broadcast will be converted to a BroadcastTo|
|Log|:green_heart:||
|LogSoftmax|:black_heart:||
|LpNormalization|:black_heart:||
|LpPool|:black_heart:||
|MatMul|:green_heart:||
|Max|:yellow_heart:|Only input of two tensors is currently supported|
|MaxPool|:yellow_heart:|auto_pad is not supported|
|MaxRoiPool|:black_heart:||
|Mean|:black_heart:||
|Min|:yellow_heart:|Only input of two tensors is currently supported|
|Mul|:yellow_heart:|broadcast will be converted to a BroadcastTo|
|Neg|:black_heart:||
|Not|:green_heart:||
|Or|:yellow_heart:|broadcast will be converted to a BroadcastTo|
|PRelu|:black_heart:||
|Pad|:black_heart:||
|Pow|:yellow_heart:|broadcast will be converted to a BroadcastTo|
|RNN|:black_heart:||
|RandomNormal|:black_heart:||
|RandomNormalLike|:black_heart:||
|RandomUniform|:black_heart:||
|RandomUniformLike|:black_heart:||
|Reciprocal|:black_heart:||
|ReduceL1|:black_heart:||
|ReduceL2|:black_heart:||
|ReduceLogSum|:black_heart:||
|ReduceLogSumExp|:black_heart:||
|ReduceMax|:black_heart:||
|ReduceMean|:green_heart:||
|ReduceMin|:black_heart:||
|ReduceProd|:black_heart:||
|ReduceSum|:green_heart:||
|ReduceSumSquare|:black_heart:||
|Relu|:green_heart:||
|Reshape|:yellow_heart:|implementing|
|Selu|:green_heart:||
|Sigmoid|:green_heart:||
|Slice|:black_heart:||
|Softmax|:yellow_heart:|Supporting 2D input only|
|Softplus|:black_heart:||
|Softsign|:black_heart:||
|SpaceToDepth|:black_heart:||
|Split|:black_heart:||
|Sqrt|:black_heart:||
|Squeeze|:black_heart:||
|Sub|:yellow_heart:|broadcast will be converted to a BroadcastTo|
|Sum|:yellow_heart:|Supporting two inputs only|
|Tanh|:green_heart:||
|Tile|:black_heart:||
|Transpose|:green_heart:||
|Xor|:yellow_heart:|broadcast will be converted to a BroadcastTo|
|experimental ATen|:black_heart:||
|experimental Affine|:black_heart:||
|experimental ConstantFill|:black_heart:||
|experimental Crop|:black_heart:||
|experimental FC|:black_heart:||
|experimental GRUUnit|:black_heart:||
|experimental GivenTensorFill|:black_heart:||
|experimental Identity|:black_heart:||
|experimental ImageScaler|:black_heart:||
|experimental MeanVarianceNormalization|:black_heart:||
|experimental ParametricSoftplus|:black_heart:||
|experimental Scale|:black_heart:||
|experimental ScaledTanh|:black_heart:||
|experimental ThresholdedRelu|:black_heart:||
|experimental Upsample|:black_heart:||

