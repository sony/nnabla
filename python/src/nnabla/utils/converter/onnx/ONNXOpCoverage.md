# Tracking operator coverage for ONNX to NNP

This is a status list of [ONNX operators](https://github.com/onnx/onnx/blob/master/docs/Operators.md)
that indicates if each operator can be converted to NNP.

- :black_heart: The ONNX operator hasn't been checked if it can be converted to NNabla.
- :purple_heart: The ONNX operator has been checked if it can be converted to NNabla, but the implementation has not started.
- :green_heart: The ONNX operator can map to a NNabla operator.
- :yellow_heart: The solution is not perfect/finished, for example, the operator can map to a combination of NNabla operators.
- :broken_heart: Hard to find a solution with existing NNabla operators.

| Operator | Status | Description |
|---|:---:|:---:|
|Abs|:green_heart:||
|Add|:yellow_heart:|broadcast will be converted to a BroadcastTo|
|And|:yellow_heart:|broadcast will be converted to a BroadcastTo|
|ArgMax|:broken_heart:|Operator does not exist in NNabla|
|ArgMin|:broken_heart:|Operator does not exist in NNabla|
|AveragePool|:yellow_heart:|autopad not supported. pads must have same value for begin and end.|
|BatchNormalization|:yellow_heart:|is_test=false not supported (only inference)|
|Cast|:broken_heart:|Operator does not exist in NNabla(No type information is exposed in NNP)|
|Ceil|:broken_heart:|Operator does not exist in NNabla|
|Clip|:purple_heart:|Should be able to map to Min + Max|
|Concat|:green_heart:||
|Constant|:yellow_heart:|Converted to an input parameter|
|Conv|:yellow_heart:|auto_pad not supported. pads must have same value for begin and end.|
|ConvTranspose|:purple_heart:|Should map to Deconvolution?|
|DepthToSpace|:broken_heart:|Operator does not exist in NNabla|
|Div|:yellow_heart:|broadcast will be converted to a BroadcastTo|
|Dropout|:yellow_heart:|mask output will be removed since NNabla does not produce mask output.|
|Elu|:green_heart:||
|Equal|:yellow_heart:|broadcast will be converted to a BroadcastTo. Input data type will all be converted to int64 since NNP does not have type information|
|Exp|:green_heart:||
|Flatten|:broken_heart:|Operator does not exist in NNabla|
|Floor|:broken_heart:|Operator does not exist in NNabla|
|GRU|:broken_heart:|Operator does not exist in NNabla|
|Gather|:broken_heart:|Operator does not exist in NNabla|
|Gemm|:yellow_heart:|alpha and beta is not supported. Input A and B must be two dimensional, and input C must be one dimensional. transA, transB will be converted to a separate transpose operator|
|GlobalAveragePool|:green_heart:||
|GlobalLpPool|:broken_heart:|Operator does not exist in NNabla|
|GlobalMaxPool|:broken_heart:|Operator does not exist in NNabla|
|Greater|:yellow_heart:|broadcast will be converted to a BroadcastTo|
|HardSigmoid|:purple_heart:|Should be able to map to MulScalar+AddScalar+MinimumScalar+ReLU|
|Hardmax|:broken_heart:|Operator does not exist in NNabla|
|Identity|:green_heart:||
|InstanceNormalization|:broken_heart:|Operator does not exist in NNabla|
|LRN|:broken_heart:|Operator does not exist in NNabla|
|LSTM|:broken_heart:|Operator does not exist in NNabla|
|LeakyRelu|:green_heart:||
|Less|:yellow_heart:|broadcast will be converted to a BroadcastTo|
|Log|:green_heart:||
|LogSoftmax|:yellow_heart:|Converted to Exp+Sum+Log+Sub2. Only works on input shape like N&ast;C&ast;1&ast;1|
|LpNormalization|:broken_heart:|Operator does not exist in NNabla|
|LpPool|:broken_heart:|Operator does not exist in NNabla|
|MatMul|:green_heart:||
|Max|:yellow_heart:|Only input of two tensors is currently supported|
|MaxPool|:yellow_heart:|auto_pad is not supported. pads must have same value for begin and end.|
|MaxRoiPool|:broken_heart:|Operator does not exist in NNabla|
|Mean|:broken_heart:|Operator does not exist in NNabla|
|Min|:yellow_heart:|Only input of two tensors is currently supported|
|Mul|:yellow_heart:|broadcast will be converted to a BroadcastTo|
|Neg|:yellow_heart:|Converted to MulScalar|
|Not|:green_heart:||
|Or|:yellow_heart:|broadcast will be converted to a BroadcastTo|
|PRelu|:green_heart:||
|Pad|:broken_heart:|Operator does not exist in NNabla|
|Pow|:yellow_heart:|broadcast will be converted to a BroadcastTo|
|RNN|:broken_heart:|Operator does not exist in NNabla|
|RandomNormal|:purple_heart:|Should be able to map to Randn|
|RandomNormalLike|:broken_heart:|Operator does not exist in NNabla|
|RandomUniform|:purple_heart:|Should be able to map to Rand|
|RandomUniformLike|:broken_heart:|Operator does not exist in NNabla|
|Reciprocal|:yellow_heart:|Converted to RDivScalar|
|ReduceL1|:broken_heart:|Operator does not exist in NNabla|
|ReduceL2|:broken_heart:|Operator does not exist in NNabla|
|ReduceLogSum|:broken_heart:|Operator does not exist in NNabla|
|ReduceLogSumExp|:broken_heart:|Operator does not exist in NNabla|
|ReduceMax|:green_heart:||
|ReduceMean|:green_heart:||
|ReduceMin|:green_heart:||
|ReduceProd|:purple_heart:|Should be able to map to Prod. No reference implementation in CNTK or Caffe2|
|ReduceSum|:green_heart:||
|ReduceSumSquare|:broken_heart:|Operator does not exit in NNabla|
|Relu|:green_heart:||
|Reshape|:yellow_heart:|implementing|
|Selu|:green_heart:||
|Sigmoid|:green_heart:||
|Size|:broken_heart:|Operator does not exist in NNabla|
|Slice|:broken_heart:|Operator does not exist in NNabla|
|Softmax|:yellow_heart:|Only works on input shape like N&ast;C&ast;1&ast;1||
|Softplus|:purple_heart:|Should be able to map to Exp + AddScalar + Log|
|Softsign|:purple_heart:|Should be able to map to Abs + AddScalar + Div2|
|SpaceToDepth|:broken_heart:|Operator does not exist in NNabla|
|Split|:broken_heart:|Operator does not exist in NNabla|
|Sqrt|:broken_heart:|Operator does not exist in NNabla|
|Squeeze|:broken_heart:|Operator does not exist in NNabla|
|Sub|:yellow_heart:|broadcast will be converted to a BroadcastTo|
|Sum|:yellow_heart:|Supporting two inputs only|
|Tanh|:green_heart:||
|Tile|:broken_heart:|Operator does not exist in NNabla|
|TopK|:broken_heart:|Operator does not exist in NNabla|
|Transpose|:green_heart:||
|Unsqueeze|:broken_heart:|Operator does not exist in NNabla|
|Xor|:yellow_heart:|broadcast will be converted to a BroadcastTo|
|experimental ATen|:black_heart:||
|experimental Affine|:black_heart:||
|experimental ConstantFill|:black_heart:||
|experimental Crop|:black_heart:||
|experimental FC|:black_heart:||
|experimental GRUUnit|:black_heart:||
|experimental GivenTensorFill|:black_heart:||
|experimental If|:black_heart:||
|experimental ImageScaler|:black_heart:||
|experimental Loop|:black_heart:||
|experimental LoopIndexTensor|:black_heart:||
|experimental MeanVarianceNormalization|:black_heart:||
|experimental ParametricSoftplus|:black_heart:||
|experimental Scale|:black_heart:||
|experimental ScaledTanh|:black_heart:||
|experimental ThresholdedRelu|:black_heart:||
|experimental Upsample|:black_heart:||

