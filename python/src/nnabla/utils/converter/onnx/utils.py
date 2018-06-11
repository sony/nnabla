NNABLA_DOMAIN = "org.nnabla"
NNABLA_OPSET_VERSION = 1
ONNX_IR_VERSION = 3
ONNX_OPSET_VERSION = 6
PRODUCER_NAME = "nnabla-onnx"
PRODUCER_VERSION = "0.1"

SOFTMAX_WARNING = """Softmax on NNabla will calculate on the specified axis ONLY. If the incoming tensor is two dimensional (such as N*C*1*1),
NNabla's Softmax and ONNX's Softmax should match. If the incoming tensor has more than two dimensions, the Softmax results may differ."""
