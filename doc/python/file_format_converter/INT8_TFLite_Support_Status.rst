=============================
INT8 Quantized TFLite Support Status
=============================
Only a subset of tflite ops supports INT8 data type.
If your model includes an unsupported op, the quantized converter will raise an error.

You can check all INT8 ops below:

- ADD
- AVERAGE_POOL_2D
- CONCATENATION
- CONV_2D
- DEPTHWISE_CONV_2D
- FULLY_CONNECTED
- L2_NORMALIZATION
- LOGISTIC
- MAX_POOL_2D
- MUL
- RESHAPE
- RESIZE_BILINEAR
- SOFTMAX
- SPACE_TO_DEPTH
- TANH
- PAD
- GATHER
- BATCH_TO_SPACE_ND
- SPACE_TO_BATCH_ND
- TRANSPOSE
- MEAN
- SUB
- SUM
- SQUEEZE
- LOG_SOFTMAX
- MAXIMUM
- ARG_MAX
- MINIMUM
- LESS
- PADV2
- GREATER
- GREATER_EQUAL
- LESS_EQUAL
- SLICE
- EQUAL
- NOT_EQUAL
- SHAPE
- QUANTIZE
- RELU
- LEAKY_RELU

Note that `CONCATENATION` is in the supported op list, but we recommend that you avoid using it.
`CONCATENATION` will bring additional quantization error and may lead to significant degradation of accuracy.