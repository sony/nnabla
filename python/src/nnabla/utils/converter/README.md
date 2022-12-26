# File Format Converter 

nnabla-converter enables you to convert NN models. Currently we are supporting following models:

* nnp (model file for nnabla and Neural Network Console)
* nnb (model file for nnabla c runtime)
* source code for nnabla c runtime
* ONNX
* Tensorflow (saved model, checkpoint, and frozen graph)
* TFLite

## Install

Before using nnabla-converter, please use command `pip install nnabla_converter` to install nnabla_converter.

## How to use

```
$ nnabla_cli convert [input model] [output model]
```
The model file is associated with the extension.
Convert from nnp to onnx: `nnabla_cli convert input.nnp output.onnx`
Convert from nnp to tflite: `nnabla_cli convert input.nnp output.tflite`
and so on.

For detail instruction and limitation see [Documentation](https://nnabla.readthedocs.io/en/latest/python/file_format_converter/file_format_converter.html)
