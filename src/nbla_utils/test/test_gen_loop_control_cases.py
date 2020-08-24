N00000001 = r"""
global_config {
  default_context {
    array_class: "CpuCachedArray"
    backends: "cpu:float"
  }
}
training_config {
  max_epoch: 100
  iter_per_epoch: 23
  save_best: true
  monitor_interval: 10
}
network {
  name: "Main"
  batch_size: 64
  repeat_info {
    id: "RepeatStart_2"
    times: 2
  }
  repeat_info {
    id: "RepeatStart"
    times: 2
  }
  variable {
    name: "Input"
    type: "Buffer"
    shape: { dim:-1 dim: 1 dim: 28 dim: 28 }
  }
  variable {
    name: "Convolution/conv/W"
    type: "Parameter"
    shape: { dim: 4 dim: 1 dim: 5 dim: 5 }
    initializer {
      type: "NormalConvolutionGlorot"
      multiplier: 1
    }
  }
  variable {
    name: "Convolution/conv/b"
    type: "Parameter"
    shape: { dim: 4 }
    initializer {
      type: "Constant"
      multiplier: 0
    }
  }
  variable {
    name: "Convolution_2{RepeatStart_2}{RepeatStart}/conv/W"
    type: "Parameter"
    repeat_id: "RepeatStart_2"
    repeat_id: "RepeatStart"
    shape: { dim: 4 dim: 4 dim: 3 dim: 3 }
    initializer {
      type: "NormalConvolutionGlorot"
      multiplier: 1
    }
  }
  variable {
    name: "Convolution_2{RepeatStart_2}{RepeatStart}/conv/b"
    type: "Parameter"
    repeat_id: "RepeatStart_2"
    repeat_id: "RepeatStart"
    shape: { dim: 4 }
    initializer {
      type: "Constant"
      multiplier: 0
    }
  }
  variable {
    name: "Affine/affine/W"
    type: "Parameter"
    shape: { dim: 144 dim: 10 }
    initializer {
      type: "NormalAffineGlorot"
      multiplier: 1
    }
  }
  variable {
    name: "Affine/affine/b"
    type: "Parameter"
    shape: { dim: 10 }
    initializer {
      type: "Constant"
      multiplier: 0
    }
  }
  variable {
    name: "Affine_2/affine/W"
    type: "Parameter"
    shape: { dim: 10 dim: 1 }
    initializer {
      type: "NormalAffineGlorot"
      multiplier: 1
    }
  }
  variable {
    name: "Affine_2/affine/b"
    type: "Parameter"
    shape: { dim: 1 }
    initializer {
      type: "Constant"
      multiplier: 0
    }
  }
  variable {
    name: "BinaryCrossEntropy_T"
    type: "Buffer"
    shape: { dim:-1 dim: 1 }
  }
  variable {
    name: "Convolution"
    type: "Buffer"
    shape: { dim:-1 dim: 4 dim: 24 dim: 24 }
  }
  variable {
    name: "MaxPooling"
    type: "Buffer"
    shape: { dim:-1 dim: 4 dim: 12 dim: 12 }
  }
  variable {
    name: "Tanh"
    type: "Buffer"
    shape: { dim:-1 dim: 4 dim: 12 dim: 12 }
  }
  variable {
    name: "RepeatStart_2"
    type: "Buffer"
    repeat_id: "RepeatStart_2"
    shape: { dim:-1 dim: 4 dim: 12 dim: 12 }
  }
  variable {
    name: "RepeatStart"
    type: "Buffer"
    repeat_id: "RepeatStart_2"
    repeat_id: "RepeatStart"
    shape: { dim:-1 dim: 4 dim: 12 dim: 12 }
  }
  variable {
    name: "Convolution_2"
    type: "Buffer"
    repeat_id: "RepeatStart_2"
    repeat_id: "RepeatStart"
    shape: { dim:-1 dim: 4 dim: 12 dim: 12 }
  }
  variable {
    name: "Tanh_2"
    type: "Buffer"
    repeat_id: "RepeatStart_2"
    repeat_id: "RepeatStart"
    shape: { dim:-1 dim: 4 dim: 12 dim: 12 }
  }
  variable {
    name: "RepeatEnd"
    type: "Buffer"
    repeat_id: "RepeatStart_2"
    shape: { dim:-1 dim: 4 dim: 12 dim: 12 }
  }
  variable {
    name: "RepeatEnd_2"
    type: "Buffer"
    shape: { dim:-1 dim: 4 dim: 12 dim: 12 }
  }
  variable {
    name: "MaxPooling_2"
    type: "Buffer"
    shape: { dim:-1 dim: 4 dim: 6 dim: 6 }
  }
  variable {
    name: "Affine"
    type: "Buffer"
    shape: { dim:-1 dim: 10 }
  }
  variable {
    name: "Tanh_3"
    type: "Buffer"
    shape: { dim:-1 dim: 10 }
  }
  variable {
    name: "Affine_2"
    type: "Buffer"
    shape: { dim:-1 dim: 1 }
  }
  variable {
    name: "Sigmoid"
    type: "Buffer"
    shape: { dim:-1 dim: 1 }
  }
  variable {
    name: "BinaryCrossEntropy"
    type: "Buffer"
    shape: { dim:-1 dim: 1 }
  }
  function {
    name: "Convolution"
    type: "Convolution"
    input: "Input"
    input: "Convolution/conv/W"
    input: "Convolution/conv/b"
    output: "Convolution"
    convolution_param {
      pad: { dim: 0 dim: 0 }
      stride: { dim: 1 dim: 1 }
      dilation: { dim: 1 dim: 1 }
      group: 1
      base_axis: 1
    }
  }
  function {
    name: "MaxPooling"
    type: "MaxPooling"
    input: "Convolution"
    output: "MaxPooling"
    max_pooling_param {
      kernel: { dim: 2 dim: 2 }
      stride: { dim: 2 dim: 2 }
      ignore_border: true
      pad: { dim: 0 dim: 0 }
    }
  }
  function {
    name: "Tanh"
    type: "Tanh"
    input: "MaxPooling"
    output: "Tanh"
  }
  function {
    name: "RepeatStart_2"
    type: "RepeatStart"
    repeat_id: "RepeatStart_2"
    input: "Tanh"
    input: "RepeatEnd"
    output: "RepeatStart_2"
    repeat_param {
      repeat_id: "RepeatStart_2"
    }
  }
  function {
    name: "RepeatStart"
    type: "RepeatStart"
    repeat_id: "RepeatStart_2"
    repeat_id: "RepeatStart"
    input: "RepeatStart_2"
    input: "Tanh_2"
    output: "RepeatStart"
    repeat_param {
      repeat_id: "RepeatStart"
    }
  }
  function {
    name: "Convolution_2"
    type: "Convolution"
    repeat_id: "RepeatStart_2"
    repeat_id: "RepeatStart"
    input: "RepeatStart"
    input: "Convolution_2{RepeatStart_2}{RepeatStart}/conv/W"
    input: "Convolution_2{RepeatStart_2}{RepeatStart}/conv/b"
    output: "Convolution_2"
    convolution_param {
      pad: { dim: 1 dim: 1 }
      stride: { dim: 1 dim: 1 }
      dilation: { dim: 1 dim: 1 }
      group: 1
      base_axis: 1
    }
  }
  function {
    name: "Tanh_2"
    type: "Tanh"
    repeat_id: "RepeatStart_2"
    repeat_id: "RepeatStart"
    input: "Convolution_2"
    output: "Tanh_2"
  }
  function {
    name: "RepeatEnd"
    type: "RepeatEnd"
    repeat_id: "RepeatStart_2"
    input: "Tanh_2"
    output: "RepeatEnd"
    repeat_param {
      repeat_id: "RepeatStart"
      times: 2
    }
  }
  function {
    name: "RepeatEnd_2"
    type: "RepeatEnd"
    input: "RepeatEnd"
    output: "RepeatEnd_2"
    repeat_param {
      repeat_id: "RepeatStart_2"
      times: 2
    }
  }
  function {
    name: "MaxPooling_2"
    type: "MaxPooling"
    input: "RepeatEnd_2"
    output: "MaxPooling_2"
    max_pooling_param {
      kernel: { dim: 2 dim: 2 }
      stride: { dim: 2 dim: 2 }
      ignore_border: true
      pad: { dim: 0 dim: 0 }
    }
  }
  function {
    name: "Affine"
    type: "Affine"
    input: "MaxPooling_2"
    input: "Affine/affine/W"
    input: "Affine/affine/b"
    output: "Affine"
    affine_param {
      base_axis: 1
    }
  }
  function {
    name: "Tanh_3"
    type: "Tanh"
    input: "Affine"
    output: "Tanh_3"
  }
  function {
    name: "Affine_2"
    type: "Affine"
    input: "Tanh_3"
    input: "Affine_2/affine/W"
    input: "Affine_2/affine/b"
    output: "Affine_2"
    affine_param {
      base_axis: 1
    }
  }
  function {
    name: "Sigmoid"
    type: "Sigmoid"
    input: "Affine_2"
    output: "Sigmoid"
  }
  function {
    name: "BinaryCrossEntropy"
    type: "BinaryCrossEntropy"
    input: "Sigmoid"
    input: "BinaryCrossEntropy_T"
    output: "BinaryCrossEntropy"
  }
}
network {
  name: "MainValidation"
  batch_size: 64
  repeat_info {
    id: "RepeatStart_2"
    times: 2
  }
  repeat_info {
    id: "RepeatStart"
    times: 2
  }
  variable {
    name: "Input"
    type: "Buffer"
    shape: { dim:-1 dim: 1 dim: 28 dim: 28 }
  }
  variable {
    name: "Convolution/conv/W"
    type: "Parameter"
    shape: { dim: 4 dim: 1 dim: 5 dim: 5 }
    initializer {
      type: "NormalConvolutionGlorot"
      multiplier: 1
    }
  }
  variable {
    name: "Convolution/conv/b"
    type: "Parameter"
    shape: { dim: 4 }
    initializer {
      type: "Constant"
      multiplier: 0
    }
  }
  variable {
    name: "Convolution_2{RepeatStart_2}{RepeatStart}/conv/W"
    type: "Parameter"
    repeat_id: "RepeatStart_2"
    repeat_id: "RepeatStart"
    shape: { dim: 4 dim: 4 dim: 3 dim: 3 }
    initializer {
      type: "NormalConvolutionGlorot"
      multiplier: 1
    }
  }
  variable {
    name: "Convolution_2{RepeatStart_2}{RepeatStart}/conv/b"
    type: "Parameter"
    repeat_id: "RepeatStart_2"
    repeat_id: "RepeatStart"
    shape: { dim: 4 }
    initializer {
      type: "Constant"
      multiplier: 0
    }
  }
  variable {
    name: "Affine/affine/W"
    type: "Parameter"
    shape: { dim: 144 dim: 10 }
    initializer {
      type: "NormalAffineGlorot"
      multiplier: 1
    }
  }
  variable {
    name: "Affine/affine/b"
    type: "Parameter"
    shape: { dim: 10 }
    initializer {
      type: "Constant"
      multiplier: 0
    }
  }
  variable {
    name: "Affine_2/affine/W"
    type: "Parameter"
    shape: { dim: 10 dim: 1 }
    initializer {
      type: "NormalAffineGlorot"
      multiplier: 1
    }
  }
  variable {
    name: "Affine_2/affine/b"
    type: "Parameter"
    shape: { dim: 1 }
    initializer {
      type: "Constant"
      multiplier: 0
    }
  }
  variable {
    name: "BinaryCrossEntropy_T"
    type: "Buffer"
    shape: { dim:-1 dim: 1 }
  }
  variable {
    name: "Convolution"
    type: "Buffer"
    shape: { dim:-1 dim: 4 dim: 24 dim: 24 }
  }
  variable {
    name: "MaxPooling"
    type: "Buffer"
    shape: { dim:-1 dim: 4 dim: 12 dim: 12 }
  }
  variable {
    name: "Tanh"
    type: "Buffer"
    shape: { dim:-1 dim: 4 dim: 12 dim: 12 }
  }
  variable {
    name: "RepeatStart_2"
    type: "Buffer"
    repeat_id: "RepeatStart_2"
    shape: { dim:-1 dim: 4 dim: 12 dim: 12 }
  }
  variable {
    name: "RepeatStart"
    type: "Buffer"
    repeat_id: "RepeatStart_2"
    repeat_id: "RepeatStart"
    shape: { dim:-1 dim: 4 dim: 12 dim: 12 }
  }
  variable {
    name: "Convolution_2"
    type: "Buffer"
    repeat_id: "RepeatStart_2"
    repeat_id: "RepeatStart"
    shape: { dim:-1 dim: 4 dim: 12 dim: 12 }
  }
  variable {
    name: "Tanh_2"
    type: "Buffer"
    repeat_id: "RepeatStart_2"
    repeat_id: "RepeatStart"
    shape: { dim:-1 dim: 4 dim: 12 dim: 12 }
  }
  variable {
    name: "RepeatEnd"
    type: "Buffer"
    repeat_id: "RepeatStart_2"
    shape: { dim:-1 dim: 4 dim: 12 dim: 12 }
  }
  variable {
    name: "RepeatEnd_2"
    type: "Buffer"
    shape: { dim:-1 dim: 4 dim: 12 dim: 12 }
  }
  variable {
    name: "MaxPooling_2"
    type: "Buffer"
    shape: { dim:-1 dim: 4 dim: 6 dim: 6 }
  }
  variable {
    name: "Affine"
    type: "Buffer"
    shape: { dim:-1 dim: 10 }
  }
  variable {
    name: "Tanh_3"
    type: "Buffer"
    shape: { dim:-1 dim: 10 }
  }
  variable {
    name: "Affine_2"
    type: "Buffer"
    shape: { dim:-1 dim: 1 }
  }
  variable {
    name: "Sigmoid"
    type: "Buffer"
    shape: { dim:-1 dim: 1 }
  }
  variable {
    name: "BinaryCrossEntropy"
    type: "Buffer"
    shape: { dim:-1 dim: 1 }
  }
  function {
    name: "Convolution"
    type: "Convolution"
    input: "Input"
    input: "Convolution/conv/W"
    input: "Convolution/conv/b"
    output: "Convolution"
    convolution_param {
      pad: { dim: 0 dim: 0 }
      stride: { dim: 1 dim: 1 }
      dilation: { dim: 1 dim: 1 }
      group: 1
      base_axis: 1
    }
  }
  function {
    name: "MaxPooling"
    type: "MaxPooling"
    input: "Convolution"
    output: "MaxPooling"
    max_pooling_param {
      kernel: { dim: 2 dim: 2 }
      stride: { dim: 2 dim: 2 }
      ignore_border: true
      pad: { dim: 0 dim: 0 }
    }
  }
  function {
    name: "Tanh"
    type: "Tanh"
    input: "MaxPooling"
    output: "Tanh"
  }
  function {
    name: "RepeatStart_2"
    type: "RepeatStart"
    repeat_id: "RepeatStart_2"
    input: "Tanh"
    input: "RepeatEnd"
    output: "RepeatStart_2"
    repeat_param {
      repeat_id: "RepeatStart_2"
    }
  }
  function {
    name: "RepeatStart"
    type: "RepeatStart"
    repeat_id: "RepeatStart_2"
    repeat_id: "RepeatStart"
    input: "RepeatStart_2"
    input: "Tanh_2"
    output: "RepeatStart"
    repeat_param {
      repeat_id: "RepeatStart"
    }
  }
  function {
    name: "Convolution_2"
    type: "Convolution"
    repeat_id: "RepeatStart_2"
    repeat_id: "RepeatStart"
    input: "RepeatStart"
    input: "Convolution_2{RepeatStart_2}{RepeatStart}/conv/W"
    input: "Convolution_2{RepeatStart_2}{RepeatStart}/conv/b"
    output: "Convolution_2"
    convolution_param {
      pad: { dim: 1 dim: 1 }
      stride: { dim: 1 dim: 1 }
      dilation: { dim: 1 dim: 1 }
      group: 1
      base_axis: 1
    }
  }
  function {
    name: "Tanh_2"
    type: "Tanh"
    repeat_id: "RepeatStart_2"
    repeat_id: "RepeatStart"
    input: "Convolution_2"
    output: "Tanh_2"
  }
  function {
    name: "RepeatEnd"
    type: "RepeatEnd"
    repeat_id: "RepeatStart_2"
    input: "Tanh_2"
    output: "RepeatEnd"
    repeat_param {
      repeat_id: "RepeatStart"
      times: 2
    }
  }
  function {
    name: "RepeatEnd_2"
    type: "RepeatEnd"
    input: "RepeatEnd"
    output: "RepeatEnd_2"
    repeat_param {
      repeat_id: "RepeatStart_2"
      times: 2
    }
  }
  function {
    name: "MaxPooling_2"
    type: "MaxPooling"
    input: "RepeatEnd_2"
    output: "MaxPooling_2"
    max_pooling_param {
      kernel: { dim: 2 dim: 2 }
      stride: { dim: 2 dim: 2 }
      ignore_border: true
      pad: { dim: 0 dim: 0 }
    }
  }
  function {
    name: "Affine"
    type: "Affine"
    input: "MaxPooling_2"
    input: "Affine/affine/W"
    input: "Affine/affine/b"
    output: "Affine"
    affine_param {
      base_axis: 1
    }
  }
  function {
    name: "Tanh_3"
    type: "Tanh"
    input: "Affine"
    output: "Tanh_3"
  }
  function {
    name: "Affine_2"
    type: "Affine"
    input: "Tanh_3"
    input: "Affine_2/affine/W"
    input: "Affine_2/affine/b"
    output: "Affine_2"
    affine_param {
      base_axis: 1
    }
  }
  function {
    name: "Sigmoid"
    type: "Sigmoid"
    input: "Affine_2"
    output: "Sigmoid"
  }
  function {
    name: "BinaryCrossEntropy"
    type: "BinaryCrossEntropy"
    input: "Sigmoid"
    input: "BinaryCrossEntropy_T"
    output: "BinaryCrossEntropy"
  }
}
network {
  name: "MainRuntime"
  batch_size: 64
  repeat_info {
    id: "RepeatStart_2"
    times: 2
  }
  repeat_info {
    id: "RepeatStart"
    times: 2
  }
  variable {
    name: "Input"
    type: "Buffer"
    shape: { dim:-1 dim: 1 dim: 28 dim: 28 }
  }
  variable {
    name: "Convolution/conv/W"
    type: "Parameter"
    shape: { dim: 4 dim: 1 dim: 5 dim: 5 }
    initializer {
      type: "NormalConvolutionGlorot"
      multiplier: 1
    }
  }
  variable {
    name: "Convolution/conv/b"
    type: "Parameter"
    shape: { dim: 4 }
    initializer {
      type: "Constant"
      multiplier: 0
    }
  }
  variable {
    name: "Convolution_2{RepeatStart_2}{RepeatStart}/conv/W"
    type: "Parameter"
    repeat_id: "RepeatStart_2"
    repeat_id: "RepeatStart"
    shape: { dim: 4 dim: 4 dim: 3 dim: 3 }
    initializer {
      type: "NormalConvolutionGlorot"
      multiplier: 1
    }
  }
  variable {
    name: "Convolution_2{RepeatStart_2}{RepeatStart}/conv/b"
    type: "Parameter"
    repeat_id: "RepeatStart_2"
    repeat_id: "RepeatStart"
    shape: { dim: 4 }
    initializer {
      type: "Constant"
      multiplier: 0
    }
  }
  variable {
    name: "Affine/affine/W"
    type: "Parameter"
    shape: { dim: 144 dim: 10 }
    initializer {
      type: "NormalAffineGlorot"
      multiplier: 1
    }
  }
  variable {
    name: "Affine/affine/b"
    type: "Parameter"
    shape: { dim: 10 }
    initializer {
      type: "Constant"
      multiplier: 0
    }
  }
  variable {
    name: "Affine_2/affine/W"
    type: "Parameter"
    shape: { dim: 10 dim: 1 }
    initializer {
      type: "NormalAffineGlorot"
      multiplier: 1
    }
  }
  variable {
    name: "Affine_2/affine/b"
    type: "Parameter"
    shape: { dim: 1 }
    initializer {
      type: "Constant"
      multiplier: 0
    }
  }
  variable {
    name: "Convolution"
    type: "Buffer"
    shape: { dim:-1 dim: 4 dim: 24 dim: 24 }
  }
  variable {
    name: "MaxPooling"
    type: "Buffer"
    shape: { dim:-1 dim: 4 dim: 12 dim: 12 }
  }
  variable {
    name: "Tanh"
    type: "Buffer"
    shape: { dim:-1 dim: 4 dim: 12 dim: 12 }
  }
  variable {
    name: "RepeatStart_2"
    type: "Buffer"
    repeat_id: "RepeatStart_2"
    shape: { dim:-1 dim: 4 dim: 12 dim: 12 }
  }
  variable {
    name: "RepeatStart"
    type: "Buffer"
    repeat_id: "RepeatStart_2"
    repeat_id: "RepeatStart"
    shape: { dim:-1 dim: 4 dim: 12 dim: 12 }
  }
  variable {
    name: "Convolution_2"
    type: "Buffer"
    repeat_id: "RepeatStart_2"
    repeat_id: "RepeatStart"
    shape: { dim:-1 dim: 4 dim: 12 dim: 12 }
  }
  variable {
    name: "Tanh_2"
    type: "Buffer"
    repeat_id: "RepeatStart_2"
    repeat_id: "RepeatStart"
    shape: { dim:-1 dim: 4 dim: 12 dim: 12 }
  }
  variable {
    name: "RepeatEnd"
    type: "Buffer"
    repeat_id: "RepeatStart_2"
    shape: { dim:-1 dim: 4 dim: 12 dim: 12 }
  }
  variable {
    name: "RepeatEnd_2"
    type: "Buffer"
    shape: { dim:-1 dim: 4 dim: 12 dim: 12 }
  }
  variable {
    name: "MaxPooling_2"
    type: "Buffer"
    shape: { dim:-1 dim: 4 dim: 6 dim: 6 }
  }
  variable {
    name: "Affine"
    type: "Buffer"
    shape: { dim:-1 dim: 10 }
  }
  variable {
    name: "Tanh_3"
    type: "Buffer"
    shape: { dim:-1 dim: 10 }
  }
  variable {
    name: "Affine_2"
    type: "Buffer"
    shape: { dim:-1 dim: 1 }
  }
  variable {
    name: "Sigmoid"
    type: "Buffer"
    shape: { dim:-1 dim: 1 }
  }
  function {
    name: "Convolution"
    type: "Convolution"
    input: "Input"
    input: "Convolution/conv/W"
    input: "Convolution/conv/b"
    output: "Convolution"
    convolution_param {
      pad: { dim: 0 dim: 0 }
      stride: { dim: 1 dim: 1 }
      dilation: { dim: 1 dim: 1 }
      group: 1
      base_axis: 1
    }
  }
  function {
    name: "MaxPooling"
    type: "MaxPooling"
    input: "Convolution"
    output: "MaxPooling"
    max_pooling_param {
      kernel: { dim: 2 dim: 2 }
      stride: { dim: 2 dim: 2 }
      ignore_border: true
      pad: { dim: 0 dim: 0 }
    }
  }
  function {
    name: "Tanh"
    type: "Tanh"
    input: "MaxPooling"
    output: "Tanh"
  }
  function {
    name: "RepeatStart_2"
    type: "RepeatStart"
    repeat_id: "RepeatStart_2"
    input: "Tanh"
    input: "RepeatEnd"
    output: "RepeatStart_2"
    repeat_param {
      repeat_id: "RepeatStart_2"
    }
  }
  function {
    name: "RepeatStart"
    type: "RepeatStart"
    repeat_id: "RepeatStart_2"
    repeat_id: "RepeatStart"
    input: "RepeatStart_2"
    input: "Tanh_2"
    output: "RepeatStart"
    repeat_param {
      repeat_id: "RepeatStart"
    }
  }
  function {
    name: "Convolution_2"
    type: "Convolution"
    repeat_id: "RepeatStart_2"
    repeat_id: "RepeatStart"
    input: "RepeatStart"
    input: "Convolution_2{RepeatStart_2}{RepeatStart}/conv/W"
    input: "Convolution_2{RepeatStart_2}{RepeatStart}/conv/b"
    output: "Convolution_2"
    convolution_param {
      pad: { dim: 1 dim: 1 }
      stride: { dim: 1 dim: 1 }
      dilation: { dim: 1 dim: 1 }
      group: 1
      base_axis: 1
    }
  }
  function {
    name: "Tanh_2"
    type: "Tanh"
    repeat_id: "RepeatStart_2"
    repeat_id: "RepeatStart"
    input: "Convolution_2"
    output: "Tanh_2"
  }
  function {
    name: "RepeatEnd"
    type: "RepeatEnd"
    repeat_id: "RepeatStart_2"
    input: "Tanh_2"
    output: "RepeatEnd"
    repeat_param {
      repeat_id: "RepeatStart"
      times: 2
    }
  }
  function {
    name: "RepeatEnd_2"
    type: "RepeatEnd"
    input: "RepeatEnd"
    output: "RepeatEnd_2"
    repeat_param {
      repeat_id: "RepeatStart_2"
      times: 2
    }
  }
  function {
    name: "MaxPooling_2"
    type: "MaxPooling"
    input: "RepeatEnd_2"
    output: "MaxPooling_2"
    max_pooling_param {
      kernel: { dim: 2 dim: 2 }
      stride: { dim: 2 dim: 2 }
      ignore_border: true
      pad: { dim: 0 dim: 0 }
    }
  }
  function {
    name: "Affine"
    type: "Affine"
    input: "MaxPooling_2"
    input: "Affine/affine/W"
    input: "Affine/affine/b"
    output: "Affine"
    affine_param {
      base_axis: 1
    }
  }
  function {
    name: "Tanh_3"
    type: "Tanh"
    input: "Affine"
    output: "Tanh_3"
  }
  function {
    name: "Affine_2"
    type: "Affine"
    input: "Tanh_3"
    input: "Affine_2/affine/W"
    input: "Affine_2/affine/b"
    output: "Affine_2"
    affine_param {
      base_axis: 1
    }
  }
  function {
    name: "Sigmoid"
    type: "Sigmoid"
    input: "Affine_2"
    output: "Sigmoid"
  }
}
dataset {
  name: "Training"
  uri: "C:\\neural_network_console\\samples\\sample_dataset\\MNIST\\small_mnist_4or9_training.csv"
  cache_dir: "C:\\neural_network_console\\samples\\sample_dataset\\MNIST\\small_mnist_4or9_training.cache"
  overwrite_cache: False
  create_cache_explicitly: True
  shuffle: true
  no_image_normalization: False
  batch_size: 64
}
dataset {
  name: "Validation"
  uri: "C:\\neural_network_console\\samples\\sample_dataset\\MNIST\\small_mnist_4or9_test.csv"
  cache_dir: "C:\\neural_network_console\\samples\\sample_dataset\\MNIST\\small_mnist_4or9_test.cache"
  overwrite_cache: False
  create_cache_explicitly: True
  shuffle: false
  no_image_normalization: False
  batch_size: 64
}
optimizer {
  name: "Optimizer"
  update_interval: 1
  network_name: "Main"
  dataset_name: "Training"
  solver {
    type: "Adam"
    weight_decay: 0
    adam_param {
      alpha: 0.001
      beta1: 0.9
      beta2: 0.999
      eps: 1e-08
    }
    lr_decay: 1
    lr_decay_interval: 1
  }
  data_variable {
    variable_name: "Input"
    data_name: "x"
  }
  data_variable {
    variable_name: "BinaryCrossEntropy_T"
    data_name: "y"
  }
  loss_variable {
    variable_name: "BinaryCrossEntropy"
  }
  parameter_variable {
    variable_name: "Convolution/conv/W"
    learning_rate_multiplier: 1
  }
  parameter_variable {
    variable_name: "Convolution/conv/b"
    learning_rate_multiplier: 1
  }
  parameter_variable {
    variable_name: "Convolution_2{RepeatStart_2}{RepeatStart}/conv/W"
    learning_rate_multiplier: 1
  }
  parameter_variable {
    variable_name: "Convolution_2{RepeatStart_2}{RepeatStart}/conv/b"
    learning_rate_multiplier: 1
  }
  parameter_variable {
    variable_name: "Affine/affine/W"
    learning_rate_multiplier: 1
  }
  parameter_variable {
    variable_name: "Affine/affine/b"
    learning_rate_multiplier: 1
  }
  parameter_variable {
    variable_name: "Affine_2/affine/W"
    learning_rate_multiplier: 1
  }
  parameter_variable {
    variable_name: "Affine_2/affine/b"
    learning_rate_multiplier: 1
  }
}
monitor {
  name: "train_error"
  network_name: "MainValidation"
  dataset_name: "Training"
  data_variable {
    variable_name: "Input"
    data_name: "x"
  }
  data_variable {
    variable_name: "BinaryCrossEntropy_T"
    data_name: "y"
  }
  monitor_variable {
    type: "Error"
    variable_name: "BinaryCrossEntropy"
  }
}
monitor {
  name: "valid_error"
  network_name: "MainValidation"
  dataset_name: "Validation"
  data_variable {
    variable_name: "Input"
    data_name: "x"
  }
  data_variable {
    variable_name: "BinaryCrossEntropy_T"
    data_name: "y"
  }
  monitor_variable {
    type: "Error"
    variable_name: "BinaryCrossEntropy"
  }
}
executor {
  name: "Executor"
  network_name: "MainRuntime"
  num_evaluations: 1
  repeat_evaluation_type: "mean"
  need_back_propagation: false
  data_variable {
    variable_name: "Input"
    data_name: "x"
  }
  output_variable {
    variable_name: "Sigmoid"
    data_name: "y'"
  }
  parameter_variable {
    variable_name: "Convolution/conv/W"
  }
  parameter_variable {
    variable_name: "Convolution/conv/b"
  }
  parameter_variable {
    variable_name: "Convolution_2{RepeatStart_2}{RepeatStart}/conv/W"
  }
  parameter_variable {
    variable_name: "Convolution_2{RepeatStart_2}{RepeatStart}/conv/b"
  }
  parameter_variable {
    variable_name: "Affine/affine/W"
  }
  parameter_variable {
    variable_name: "Affine/affine/b"
  }
  parameter_variable {
    variable_name: "Affine_2/affine/W"
  }
  parameter_variable {
    variable_name: "Affine_2/affine/b"
  }
}
"""

N00000002 = r"""
global_config {
  default_context {
    backend: "cpu|cuda"
    array_class: "CudaArray"
    device_id: "0"
    compute_backend: "default|cudnn"
  }
}
training_config {
  max_epoch: 100
  iter_per_epoch: 23
  save_best: true
}
network {
  name: "Main"
  batch_size: 64
  repeat_info {
    id: "RecurrentInput"
    times: 28
  }
  variable {
    name: "Input"
    type: "Buffer"
    shape: { dim:-1 dim: 1 dim: 28 dim: 28 }
  }
  variable {
    name: "RecurrentInput"
    type: "Buffer"
    repeat_id: "RecurrentInput"
    shape: { dim:-1 dim: 1 dim: 28 }
  }
  variable {
    name: "Affine/affine/W"
    type: "Parameter"
    shape: { dim: 28 dim: 1 }
    initializer {
      type: "NormalAffineGlorot"
      multiplier: 1.0
    }
  }
  variable {
    name: "Affine/affine/b"
    type: "Parameter"
    shape: { dim: 1 }
    initializer {
      type: "Constant"
      multiplier: 0.0
    }
  }
  variable {
    name: "Delay_Initial"
    type: "Buffer"
    shape: { dim:-1 dim: 1 dim: 28 }
  }
  variable {
    name: "Affine_2/affine/W"
    type: "Parameter"
    shape: { dim: 56 dim: 1 dim: 28 }
    initializer {
      type: "NormalAffineGlorot"
      multiplier: 1.0
    }
  }
  variable {
    name: "Affine_2/affine/b"
    type: "Parameter"
    shape: { dim: 1 dim: 28 }
    initializer {
      type: "Constant"
      multiplier: 0.0
    }
  }
  variable {
    name: "BinaryCrossEntropy_T"
    type: "Buffer"
    shape: { dim:-1 dim: 1 }
  }
  variable {
    name: "Affine"
    type: "Buffer"
    shape: { dim:-1 dim: 1 }
  }
  variable {
    name: "Sigmoid"
    type: "Buffer"
    shape: { dim:-1 dim: 1 }
  }
  variable {
    name: "Delay"
    type: "Buffer"
    repeat_id: "RecurrentInput"
    shape: { dim:-1 dim: 1 dim: 28 }
  }
  variable {
    name: "Affine_2"
    type: "Buffer"
    repeat_id: "RecurrentInput"
    shape: { dim:-1 dim: 1 dim: 28 }
  }
  variable {
    name: "Concatenate"
    type: "Buffer"
    repeat_id: "RecurrentInput"
    shape: { dim:-1 dim: 2 dim: 28 }
  }
  variable {
    name: "Tanh"
    type: "Buffer"
    repeat_id: "RecurrentInput"
    shape: { dim:-1 dim: 1 dim: 28 }
  }
  variable {
    name: "BinaryCrossEntropy"
    type: "Buffer"
    shape: { dim:-1 dim: 1 }
  }
  variable {
    name: "RecurrentOutput"
    type: "Buffer"
    shape: { dim:-1 dim: 1 dim: 28 dim: 28 }
  }
  variable {
    name: "Slice"
    type: "Buffer"
    shape: { dim:-1 dim: 1 dim: 1 dim: 28 }
  }
  function {
    name: "RecurrentInput"
    type: "RecurrentInput"
    repeat_id: "RecurrentInput"
    input: "Input"
    output: "RecurrentInput"
    recurrent_param {
      axis: 2
      repeat_id: "RecurrentInput"
    }
  }
  function {
    name: "Affine"
    type: "Affine"
    input: "Slice"
    input: "Affine/affine/W"
    input: "Affine/affine/b"
    output: "Affine"
    affine_param {
      base_axis: 1
    }
  }
  function {
    name: "Sigmoid"
    type: "Sigmoid"
    input: "Affine"
    output: "Sigmoid"
  }
  function {
    name: "Delay"
    type: "Delay"
    repeat_id: "RecurrentInput"
    input: "Tanh"
    input: "Delay_Initial"
    output: "Delay"
    recurrent_param {
      repeat_id: "RecurrentInput"
    }
  }
  function {
    name: "Affine_2"
    type: "Affine"
    repeat_id: "RecurrentInput"
    input: "Concatenate"
    input: "Affine_2/affine/W"
    input: "Affine_2/affine/b"
    output: "Affine_2"
    affine_param {
      base_axis: 1
    }
  }
  function {
    name: "Concatenate"
    type: "Concatenate"
    repeat_id: "RecurrentInput"
    input: "RecurrentInput"
    input: "Delay"
    output: "Concatenate"
    concatenate_param {
      axis: 1
    }
  }
  function {
    name: "Tanh"
    type: "Tanh"
    repeat_id: "RecurrentInput"
    input: "Affine_2"
    output: "Tanh"
  }
  function {
    name: "BinaryCrossEntropy"
    type: "BinaryCrossEntropy"
    input: "Sigmoid"
    input: "BinaryCrossEntropy_T"
    output: "BinaryCrossEntropy"
  }
  function {
    name: "RecurrentOutput"
    type: "RecurrentOutput"
    input: "Tanh"
    output: "RecurrentOutput"
    recurrent_param {
      axis: 2
      repeat_id: "RecurrentInput"
      length: 28
    }
  }
  function {
    name: "Slice"
    type: "Slice"
    input: "RecurrentOutput"
    output: "Slice"
    slice_param {
      start: 0
      start: 27
      start: 0
      stop: 1
      stop: 28
      stop: 28
      step: 1
      step: 1
      step: 1
    }
  }
}
network {
  name: "MainValidation"
  batch_size: 64
  repeat_info {
    id: "RecurrentInput"
    times: 28
  }
  variable {
    name: "Input"
    type: "Buffer"
    shape: { dim:-1 dim: 1 dim: 28 dim: 28 }
  }
  variable {
    name: "RecurrentInput"
    type: "Buffer"
    repeat_id: "RecurrentInput"
    shape: { dim:-1 dim: 1 dim: 28 }
  }
  variable {
    name: "Affine/affine/W"
    type: "Parameter"
    shape: { dim: 28 dim: 1 }
    initializer {
      type: "NormalAffineGlorot"
      multiplier: 1.0
    }
  }
  variable {
    name: "Affine/affine/b"
    type: "Parameter"
    shape: { dim: 1 }
    initializer {
      type: "Constant"
      multiplier: 0.0
    }
  }
  variable {
    name: "Delay_Initial"
    type: "Buffer"
    shape: { dim:-1 dim: 1 dim: 28 }
  }
  variable {
    name: "Affine_2/affine/W"
    type: "Parameter"
    shape: { dim: 56 dim: 1 dim: 28 }
    initializer {
      type: "NormalAffineGlorot"
      multiplier: 1.0
    }
  }
  variable {
    name: "Affine_2/affine/b"
    type: "Parameter"
    shape: { dim: 1 dim: 28 }
    initializer {
      type: "Constant"
      multiplier: 0.0
    }
  }
  variable {
    name: "BinaryCrossEntropy_T"
    type: "Buffer"
    shape: { dim:-1 dim: 1 }
  }
  variable {
    name: "Affine"
    type: "Buffer"
    shape: { dim:-1 dim: 1 }
  }
  variable {
    name: "Sigmoid"
    type: "Buffer"
    shape: { dim:-1 dim: 1 }
  }
  variable {
    name: "Delay"
    type: "Buffer"
    repeat_id: "RecurrentInput"
    shape: { dim:-1 dim: 1 dim: 28 }
  }
  variable {
    name: "Affine_2"
    type: "Buffer"
    repeat_id: "RecurrentInput"
    shape: { dim:-1 dim: 1 dim: 28 }
  }
  variable {
    name: "Concatenate"
    type: "Buffer"
    repeat_id: "RecurrentInput"
    shape: { dim:-1 dim: 2 dim: 28 }
  }
  variable {
    name: "Tanh"
    type: "Buffer"
    repeat_id: "RecurrentInput"
    shape: { dim:-1 dim: 1 dim: 28 }
  }
  variable {
    name: "BinaryCrossEntropy"
    type: "Buffer"
    shape: { dim:-1 dim: 1 }
  }
  variable {
    name: "RecurrentOutput"
    type: "Buffer"
    shape: { dim:-1 dim: 1 dim: 28 dim: 28 }
  }
  variable {
    name: "Slice"
    type: "Buffer"
    shape: { dim:-1 dim: 1 dim: 1 dim: 28 }
  }
  function {
    name: "RecurrentInput"
    type: "RecurrentInput"
    repeat_id: "RecurrentInput"
    input: "Input"
    output: "RecurrentInput"
    recurrent_param {
      axis: 2
      repeat_id: "RecurrentInput"
    }
  }
  function {
    name: "Affine"
    type: "Affine"
    input: "Slice"
    input: "Affine/affine/W"
    input: "Affine/affine/b"
    output: "Affine"
    affine_param {
      base_axis: 1
    }
  }
  function {
    name: "Sigmoid"
    type: "Sigmoid"
    input: "Affine"
    output: "Sigmoid"
  }
  function {
    name: "Delay"
    type: "Delay"
    repeat_id: "RecurrentInput"
    input: "Tanh"
    input: "Delay_Initial"
    output: "Delay"
    recurrent_param {
      repeat_id: "RecurrentInput"
    }
  }
  function {
    name: "Affine_2"
    type: "Affine"
    repeat_id: "RecurrentInput"
    input: "Concatenate"
    input: "Affine_2/affine/W"
    input: "Affine_2/affine/b"
    output: "Affine_2"
    affine_param {
      base_axis: 1
    }
  }
  function {
    name: "Concatenate"
    type: "Concatenate"
    repeat_id: "RecurrentInput"
    input: "RecurrentInput"
    input: "Delay"
    output: "Concatenate"
    concatenate_param {
      axis: 1
    }
  }
  function {
    name: "Tanh"
    type: "Tanh"
    repeat_id: "RecurrentInput"
    input: "Affine_2"
    output: "Tanh"
  }
  function {
    name: "BinaryCrossEntropy"
    type: "BinaryCrossEntropy"
    input: "Sigmoid"
    input: "BinaryCrossEntropy_T"
    output: "BinaryCrossEntropy"
  }
  function {
    name: "RecurrentOutput"
    type: "RecurrentOutput"
    input: "Tanh"
    output: "RecurrentOutput"
    recurrent_param {
      axis: 2
      repeat_id: "RecurrentInput"
      length: 28
    }
  }
  function {
    name: "Slice"
    type: "Slice"
    input: "RecurrentOutput"
    output: "Slice"
    slice_param {
      start: 0
      start: 27
      start: 0
      stop: 1
      stop: 28
      stop: 28
      step: 1
      step: 1
      step: 1
    }
  }
}
network {
  name: "MainRuntime"
  batch_size: 64
  repeat_info {
    id: "RecurrentInput"
    times: 28
  }
  variable {
    name: "Input"
    type: "Buffer"
    shape: { dim:-1 dim: 1 dim: 28 dim: 28 }
  }
  variable {
    name: "RecurrentInput"
    type: "Buffer"
    repeat_id: "RecurrentInput"
    shape: { dim:-1 dim: 1 dim: 28 }
  }
  variable {
    name: "Affine/affine/W"
    type: "Parameter"
    shape: { dim: 28 dim: 1 }
    initializer {
      type: "NormalAffineGlorot"
      multiplier: 1.0
    }
  }
  variable {
    name: "Affine/affine/b"
    type: "Parameter"
    shape: { dim: 1 }
    initializer {
      type: "Constant"
      multiplier: 0.0
    }
  }
  variable {
    name: "Delay_Initial"
    type: "Buffer"
    shape: { dim:-1 dim: 1 dim: 28 }
  }
  variable {
    name: "Affine_2/affine/W"
    type: "Parameter"
    shape: { dim: 56 dim: 1 dim: 28 }
    initializer {
      type: "NormalAffineGlorot"
      multiplier: 1.0
    }
  }
  variable {
    name: "Affine_2/affine/b"
    type: "Parameter"
    shape: { dim: 1 dim: 28 }
    initializer {
      type: "Constant"
      multiplier: 0.0
    }
  }
  variable {
    name: "Affine"
    type: "Buffer"
    shape: { dim:-1 dim: 1 }
  }
  variable {
    name: "Sigmoid"
    type: "Buffer"
    shape: { dim:-1 dim: 1 }
  }
  variable {
    name: "Delay"
    type: "Buffer"
    repeat_id: "RecurrentInput"
    shape: { dim:-1 dim: 1 dim: 28 }
  }
  variable {
    name: "Affine_2"
    type: "Buffer"
    repeat_id: "RecurrentInput"
    shape: { dim:-1 dim: 1 dim: 28 }
  }
  variable {
    name: "Concatenate"
    type: "Buffer"
    repeat_id: "RecurrentInput"
    shape: { dim:-1 dim: 2 dim: 28 }
  }
  variable {
    name: "Tanh"
    type: "Buffer"
    repeat_id: "RecurrentInput"
    shape: { dim:-1 dim: 1 dim: 28 }
  }
  variable {
    name: "RecurrentOutput"
    type: "Buffer"
    shape: { dim:-1 dim: 1 dim: 28 dim: 28 }
  }
  variable {
    name: "Slice"
    type: "Buffer"
    shape: { dim:-1 dim: 1 dim: 1 dim: 28 }
  }
  function {
    name: "RecurrentInput"
    type: "RecurrentInput"
    repeat_id: "RecurrentInput"
    input: "Input"
    output: "RecurrentInput"
    recurrent_param {
      axis: 2
      repeat_id: "RecurrentInput"
    }
  }
  function {
    name: "Affine"
    type: "Affine"
    input: "Slice"
    input: "Affine/affine/W"
    input: "Affine/affine/b"
    output: "Affine"
    affine_param {
      base_axis: 1
    }
  }
  function {
    name: "Sigmoid"
    type: "Sigmoid"
    input: "Affine"
    output: "Sigmoid"
  }
  function {
    name: "Delay"
    type: "Delay"
    repeat_id: "RecurrentInput"
    input: "Tanh"
    input: "Delay_Initial"
    output: "Delay"
    recurrent_param {
      repeat_id: "RecurrentInput"
    }
  }
  function {
    name: "Affine_2"
    type: "Affine"
    repeat_id: "RecurrentInput"
    input: "Concatenate"
    input: "Affine_2/affine/W"
    input: "Affine_2/affine/b"
    output: "Affine_2"
    affine_param {
      base_axis: 1
    }
  }
  function {
    name: "Concatenate"
    type: "Concatenate"
    repeat_id: "RecurrentInput"
    input: "RecurrentInput"
    input: "Delay"
    output: "Concatenate"
    concatenate_param {
      axis: 1
    }
  }
  function {
    name: "Tanh"
    type: "Tanh"
    repeat_id: "RecurrentInput"
    input: "Affine_2"
    output: "Tanh"
  }
  function {
    name: "RecurrentOutput"
    type: "RecurrentOutput"
    input: "Tanh"
    output: "RecurrentOutput"
    recurrent_param {
      axis: 2
      repeat_id: "RecurrentInput"
      length: 28
    }
  }
  function {
    name: "Slice"
    type: "Slice"
    input: "RecurrentOutput"
    output: "Slice"
    slice_param {
      start: 0
      start: 27
      start: 0
      stop: 1
      stop: 28
      stop: 28
      step: 1
      step: 1
      step: 1
    }
  }
}
dataset {
  name: "Training"
  uri: "K:\\neural_network_console_170725\\samples\\sample_dataset\\mnist\\small_mnist_4or9_training.csv"
  cache_dir: "K:\\neural_network_console_170725\\samples\\sample_dataset\\mnist\\small_mnist_4or9_training.cache"
  overwrite_cache: False
  create_cache_explicitly: True
  shuffle: true
  no_image_normalization: False
  batch_size: 64
}
dataset {
  name: "Validation"
  uri: "K:\\neural_network_console_170725\\samples\\sample_dataset\\mnist\\small_mnist_4or9_test.csv"
  cache_dir: "K:\\neural_network_console_170725\\samples\\sample_dataset\\mnist\\small_mnist_4or9_test.cache"
  overwrite_cache: False
  create_cache_explicitly: True
  shuffle: false
  no_image_normalization: False
  batch_size: 64
}
optimizer {
  name: "Optimizer"
  update_interval: 1
  network_name: "Main"
  dataset_name: "Training"
  solver {
    type: "Adam"
    weight_decay: 0
    lr_decay: 1
    lr_decay_interval: 1
    adam_param {
      alpha: 0.001
      beta1: 0.9
      beta2: 0.999
      eps: 1e-08
    }
  }
  data_variable {
    variable_name: "Input"
    data_name: "x"
  }
  generator_variable {
    variable_name: "Delay_Initial"
    type: "Constant"
    multiplier: 0.0
  }
  data_variable {
    variable_name: "BinaryCrossEntropy_T"
    data_name: "y"
  }
  loss_variable {
    variable_name: "BinaryCrossEntropy"
  }
  parameter_variable {
    variable_name: "Affine/affine/W"
    learning_rate_multiplier: 1.0
  }
  parameter_variable {
    variable_name: "Affine/affine/b"
    learning_rate_multiplier: 1.0
  }
  parameter_variable {
    variable_name: "Affine_2/affine/W"
    learning_rate_multiplier: 1.0
  }
  parameter_variable {
    variable_name: "Affine_2/affine/b"
    learning_rate_multiplier: 1.0
  }
}
monitor {
  name: "train_error"
  network_name: "MainValidation"
  dataset_name: "Training"
  data_variable {
    variable_name: "Input"
    data_name: "x"
  }
  generator_variable {
    variable_name: "Delay_Initial"
    type: "Constant"
    multiplier: 0.0
  }
  data_variable {
    variable_name: "BinaryCrossEntropy_T"
    data_name: "y"
  }
  monitor_variable {
    type: "Error"
    variable_name: "BinaryCrossEntropy"
  }
}
monitor {
  name: "valid_error"
  network_name: "MainValidation"
  dataset_name: "Validation"
  data_variable {
    variable_name: "Input"
    data_name: "x"
  }
  generator_variable {
    variable_name: "Delay_Initial"
    type: "Constant"
    multiplier: 0.0
  }
  data_variable {
    variable_name: "BinaryCrossEntropy_T"
    data_name: "y"
  }
  monitor_variable {
    type: "Error"
    variable_name: "BinaryCrossEntropy"
  }
}
executor {
  name: "Executor"
  network_name: "MainRuntime"
  num_evaluations: 1
  repeat_evaluation_type: "mean"
  need_back_propagation: false
  data_variable {
    variable_name: "Input"
    data_name: "x"
  }
  generator_variable {
    variable_name: "Delay_Initial"
    type: "Constant"
    multiplier: 0.0
  }
  output_variable {
    variable_name: "Sigmoid"
    data_name: "y'"
  }
  parameter_variable {
    variable_name: "Affine/affine/W"
  }
  parameter_variable {
    variable_name: "Affine/affine/b"
  }
  parameter_variable {
    variable_name: "Affine_2/affine/W"
  }
  parameter_variable {
    variable_name: "Affine_2/affine/b"
  }
}
"""


N00000003 = r"""
global_config {
  default_context {
    array_class: "CpuCachedArray"
    backends: "cpu:float"
  }
}
training_config {
  max_epoch: 100
  iter_per_epoch: 23
  save_best: true
  monitor_interval: 10
}
network {
  name: "Main"
  batch_size: 64
  variable {
    name: "Input"
    type: "Buffer"
    shape: { dim:-1 dim: 1 dim: 28 dim: 28 }
  }
  variable {
    name: "Affine/affine/W"
    type: "Parameter"
    shape: { dim: 784 dim: 1 }
    initializer {
      type: "NormalAffineGlorot"
      multiplier: 1
    }
  }
  variable {
    name: "Affine/affine/b"
    type: "Parameter"
    shape: { dim: 1 }
    initializer {
      type: "Constant"
      multiplier: 0
    }
  }
  variable {
    name: "BinaryCrossEntropy_T"
    type: "Buffer"
    shape: { dim:-1 dim: 1 }
  }
  variable {
    name: "Affine"
    type: "Buffer"
    shape: { dim:-1 dim: 1 }
  }
  variable {
    name: "Sigmoid"
    type: "Buffer"
    shape: { dim:-1 dim: 1 }
  }
  variable {
    name: "BinaryCrossEntropy"
    type: "Buffer"
    shape: { dim:-1 dim: 1 }
  }
  function {
    name: "Affine"
    type: "Affine"
    input: "Input"
    input: "Affine/affine/W"
    input: "Affine/affine/b"
    output: "Affine"
    affine_param {
      base_axis: 1
    }
  }
  function {
    name: "Sigmoid"
    type: "Sigmoid"
    input: "Affine"
    output: "Sigmoid"
  }
  function {
    name: "BinaryCrossEntropy"
    type: "BinaryCrossEntropy"
    input: "Sigmoid"
    input: "BinaryCrossEntropy_T"
    output: "BinaryCrossEntropy"
  }
}
network {
  name: "MainValidation"
  batch_size: 64
  variable {
    name: "Input"
    type: "Buffer"
    shape: { dim:-1 dim: 1 dim: 28 dim: 28 }
  }
  variable {
    name: "Affine/affine/W"
    type: "Parameter"
    shape: { dim: 784 dim: 1 }
    initializer {
      type: "NormalAffineGlorot"
      multiplier: 1
    }
  }
  variable {
    name: "Affine/affine/b"
    type: "Parameter"
    shape: { dim: 1 }
    initializer {
      type: "Constant"
      multiplier: 0
    }
  }
  variable {
    name: "BinaryCrossEntropy_T"
    type: "Buffer"
    shape: { dim:-1 dim: 1 }
  }
  variable {
    name: "Affine"
    type: "Buffer"
    shape: { dim:-1 dim: 1 }
  }
  variable {
    name: "Sigmoid"
    type: "Buffer"
    shape: { dim:-1 dim: 1 }
  }
  variable {
    name: "BinaryCrossEntropy"
    type: "Buffer"
    shape: { dim:-1 dim: 1 }
  }
  function {
    name: "Affine"
    type: "Affine"
    input: "Input"
    input: "Affine/affine/W"
    input: "Affine/affine/b"
    output: "Affine"
    affine_param {
      base_axis: 1
    }
  }
  function {
    name: "Sigmoid"
    type: "Sigmoid"
    input: "Affine"
    output: "Sigmoid"
  }
  function {
    name: "BinaryCrossEntropy"
    type: "BinaryCrossEntropy"
    input: "Sigmoid"
    input: "BinaryCrossEntropy_T"
    output: "BinaryCrossEntropy"
  }
}
network {
  name: "MainRuntime"
  batch_size: 64
  variable {
    name: "Input"
    type: "Buffer"
    shape: { dim:-1 dim: 1 dim: 28 dim: 28 }
  }
  variable {
    name: "Affine/affine/W"
    type: "Parameter"
    shape: { dim: 784 dim: 1 }
    initializer {
      type: "NormalAffineGlorot"
      multiplier: 1
    }
  }
  variable {
    name: "Affine/affine/b"
    type: "Parameter"
    shape: { dim: 1 }
    initializer {
      type: "Constant"
      multiplier: 0
    }
  }
  variable {
    name: "Affine"
    type: "Buffer"
    shape: { dim:-1 dim: 1 }
  }
  variable {
    name: "Sigmoid"
    type: "Buffer"
    shape: { dim:-1 dim: 1 }
  }
  function {
    name: "Affine"
    type: "Affine"
    input: "Input"
    input: "Affine/affine/W"
    input: "Affine/affine/b"
    output: "Affine"
    affine_param {
      base_axis: 1
    }
  }
  function {
    name: "Sigmoid"
    type: "Sigmoid"
    input: "Affine"
    output: "Sigmoid"
  }
}
dataset {
  name: "Training"
  uri: "/home/woody/disk/aid/sdeep-nnabla-builder/sample_dataset/MNIST/"
  cache_dir: "/home/woody/disk/aid/sdeep-nnabla-builder/sample_dataset/MNIST/train.cache"
  overwrite_cache: True
  create_cache_explicitly: True
  shuffle: true
  no_image_normalization: False
  batch_size: 64
}
dataset {
  name: "Validation"
  uri: "C:\\Utils\\neural_network_console_internal_v120\\samples\\sample_dataset\\mnist\\small_mnist_4or9_test.csv"
  cache_dir: "C:\\Utils\\neural_network_console_internal_v120\\samples\\sample_dataset\\mnist\\small_mnist_4or9_test.cache"
  overwrite_cache: True
  create_cache_explicitly: True
  shuffle: false
  no_image_normalization: False
  batch_size: 64
}
optimizer {
  name: "Optimizer"
  update_interval: 1
  network_name: "Main"
  dataset_name: "Training"
  solver {
    type: "Adam"
    weight_decay: 0
    lr_decay: 1
    lr_decay_interval: 1
    adam_param {
      alpha: 0.001
      beta1: 0.9
      beta2: 0.999
      eps: 1e-08
    }
  }
  data_variable {
    variable_name: "Input"
    data_name: "x"
  }
  data_variable {
    variable_name: "BinaryCrossEntropy_T"
    data_name: "y"
  }
  loss_variable {
    variable_name: "BinaryCrossEntropy"
  }
  parameter_variable {
    variable_name: "Affine/affine/W"
    learning_rate_multiplier: 1
  }
  parameter_variable {
    variable_name: "Affine/affine/b"
    learning_rate_multiplier: 1
  }
}
monitor {
  name: "train_error"
  network_name: "MainValidation"
  dataset_name: "Training"
  data_variable {
    variable_name: "Input"
    data_name: "x"
  }
  data_variable {
    variable_name: "BinaryCrossEntropy_T"
    data_name: "y"
  }
  monitor_variable {
    type: "Error"
    variable_name: "BinaryCrossEntropy"
  }
}
monitor {
  name: "valid_error"
  network_name: "MainValidation"
  dataset_name: "Validation"
  data_variable {
    variable_name: "Input"
    data_name: "x"
  }
  data_variable {
    variable_name: "BinaryCrossEntropy_T"
    data_name: "y"
  }
  monitor_variable {
    type: "Error"
    variable_name: "BinaryCrossEntropy"
  }
}
executor {
  name: "Executor"
  network_name: "MainRuntime"
  num_evaluations: 1
  repeat_evaluation_type: "mean"
  need_back_propagation: false
  data_variable {
    variable_name: "Input"
    data_name: "x"
  }
  output_variable {
    variable_name: "Sigmoid"
    data_name: "y'"
  }
  parameter_variable {
    variable_name: "Affine/affine/W"
  }
  parameter_variable {
    variable_name: "Affine/affine/b"
  }
}
"""

P00000001 = [
    ('Convolution/conv/W', (4, 1, 5, 5)),
    ('Convolution/conv/b', (4,)),
    ('Convolution_2[0][0]/conv/W', (4, 4, 3, 3)),
    ('Convolution_2[0][0]/conv/b', (4,)),
    ('Convolution_2[0][1]/conv/W', (4, 4, 3, 3)),
    ('Convolution_2[0][1]/conv/b', (4,)),
    ('Convolution_2[1][0]/conv/W', (4, 4, 3, 3)),
    ('Convolution_2[1][0]/conv/b', (4,)),
    ('Convolution_2[1][1]/conv/W', (4, 4, 3, 3)),
    ('Convolution_2[1][1]/conv/b', (4,)),
    ('Affine/affine/W', (144, 10)),
    ('Affine/affine/b', (10,)),
    ('Affine_2/affine/W', (10, 1)),
    ('Affine_2/affine/b', (1,))
]

P00000002 = [
    ('Affine/affine/W', (28, 1)),
    ('Affine/affine/b', (1,)),
    ('Affine_2/affine/W', (56, 1, 28)),
    ('Affine_2/affine/b', (1, 28))
]

P00000003 = [
    ('Affine/affine/W', (784, 1)),
    ('Affine/affine/b', (1,))
]


import os
import pytest
import h5py
import numpy as np


cases = [
    ("test_loop_controls/nested_loop_test", N00000001, P00000001),
    ("test_loop_controls/recurrent_delay_test", N00000002, P00000002),
    ("test_loop_controls/no_repeat_test", N00000003, P00000003),
]


@pytest.mark.parametrize("cases", cases)
def test_gen_loop_control_cases(cases):
    d, s, params = cases
    if not os.path.exists(d):
        os.makedirs(d)
    with open(os.path.join(d, "net.nntxt"), "w") as f:
        f.write(s)

    with h5py.File(os.path.join(d, "parameters.h5"), 'w') as f:
        for i, (param_name, shape) in enumerate(params):
            f[param_name] = np.random.random(shape)
            f[param_name].attrs['need_grad'] = 1
            f[param_name].attrs['index'] = i
