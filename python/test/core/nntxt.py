# The nntxt is generated from /home/woody/aid/nnabla-sample-data/sample_project/tutorial/recurrent_neural_networks/elman_net.files/20180720_190147/net.nntxt.
N00000000 = r'''global_config {
  default_context {
    array_class: "CudaCachedArray"
    device_id: "0"
    backends: "cudnn:float"
    backends: "cuda:float"
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
    id: "RecurrentInput"
    times: 28
  }
  variable {
    name: "Input"
    type: "Buffer"
    shape: { dim:-1 dim: 1 dim: 28 dim: 28 }
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
      multiplier: 1
    }
  }
  variable {
    name: "Affine_2/affine/b"
    type: "Parameter"
    shape: { dim: 1 dim: 28 }
    initializer {
      type: "Constant"
      multiplier: 0
    }
  }
  variable {
    name: "Affine/affine/W"
    type: "Parameter"
    shape: { dim: 28 dim: 1 }
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
    name: "RecurrentInput"
    type: "Buffer"
    repeat_id: "RecurrentInput"
    shape: { dim:-1 dim: 1 dim: 28 }
  }
  variable {
    name: "Delay"
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
    name: "Affine_2"
    type: "Buffer"
    repeat_id: "RecurrentInput"
    shape: { dim:-1 dim: 1 dim: 28 }
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
    id: "RecurrentInput"
    times: 28
  }
  variable {
    name: "Input"
    type: "Buffer"
    shape: { dim:-1 dim: 1 dim: 28 dim: 28 }
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
      multiplier: 1
    }
  }
  variable {
    name: "Affine_2/affine/b"
    type: "Parameter"
    shape: { dim: 1 dim: 28 }
    initializer {
      type: "Constant"
      multiplier: 0
    }
  }
  variable {
    name: "Affine/affine/W"
    type: "Parameter"
    shape: { dim: 28 dim: 1 }
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
    name: "RecurrentInput"
    type: "Buffer"
    repeat_id: "RecurrentInput"
    shape: { dim:-1 dim: 1 dim: 28 }
  }
  variable {
    name: "Delay"
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
    name: "Affine_2"
    type: "Buffer"
    repeat_id: "RecurrentInput"
    shape: { dim:-1 dim: 1 dim: 28 }
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
    id: "RecurrentInput"
    times: 28
  }
  variable {
    name: "Input"
    type: "Buffer"
    shape: { dim:-1 dim: 1 dim: 28 dim: 28 }
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
      multiplier: 1
    }
  }
  variable {
    name: "Affine_2/affine/b"
    type: "Parameter"
    shape: { dim: 1 dim: 28 }
    initializer {
      type: "Constant"
      multiplier: 0
    }
  }
  variable {
    name: "Affine/affine/W"
    type: "Parameter"
    shape: { dim: 28 dim: 1 }
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
    name: "RecurrentInput"
    type: "Buffer"
    repeat_id: "RecurrentInput"
    shape: { dim:-1 dim: 1 dim: 28 }
  }
  variable {
    name: "Delay"
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
    name: "Affine_2"
    type: "Buffer"
    repeat_id: "RecurrentInput"
    shape: { dim:-1 dim: 1 dim: 28 }
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
}
dataset {
  name: "Training"
  uri: "C:\\Utils\\neural_network_console_internal_v120\\samples\\sample_dataset\\mnist\\small_mnist_4or9_training.csv"
  cache_dir: "C:\\Utils\\neural_network_console_internal_v120\\samples\\sample_dataset\\mnist\\small_mnist_4or9_training.cache"
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
  generator_variable {
    variable_name: "Delay_Initial"
    type: "Constant"
    multiplier: 0
  }
  data_variable {
    variable_name: "BinaryCrossEntropy_T"
    data_name: "y"
  }
  loss_variable {
    variable_name: "BinaryCrossEntropy"
  }
  parameter_variable {
    variable_name: "Affine_2/affine/W"
    learning_rate_multiplier: 1
  }
  parameter_variable {
    variable_name: "Affine_2/affine/b"
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
    multiplier: 0
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
    multiplier: 0
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
    multiplier: 0
  }
  output_variable {
    variable_name: "Sigmoid"
    data_name: "y'"
  }
  parameter_variable {
    variable_name: "Affine_2/affine/W"
  }
  parameter_variable {
    variable_name: "Affine_2/affine/b"
  }
  parameter_variable {
    variable_name: "Affine/affine/W"
  }
  parameter_variable {
    variable_name: "Affine/affine/b"
  }
}
'''
# The nntxt is generated from /home/woody/aid/nnabla-sample-data/sample_project/tutorial/recurrent_neural_networks/elman_net.files/20170804_142035/net.nntxt.
N00000001 = r'''global_config {
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
'''
# The nntxt is generated from /home/woody/aid/nnabla-sample-data/sample_project/tutorial/recurrent_neural_networks/elman_net_with_attention.files/20180720_190245/net.nntxt.
N00000002 = r'''global_config {
  default_context {
    array_class: "CudaCachedArray"
    device_id: "0"
    backends: "cudnn:float"
    backends: "cuda:float"
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
  name: "Main_"
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
      multiplier: 1
    }
  }
  variable {
    name: "Affine_2/affine/b"
    type: "Parameter"
    shape: { dim: 1 dim: 28 }
    initializer {
      type: "Constant"
      multiplier: 0
    }
  }
  variable {
    name: "Convolution/conv/W"
    type: "Parameter"
    shape: { dim: 1 dim: 1 dim: 1 dim: 28 }
    initializer {
      type: "NormalConvolutionGlorot"
      multiplier: 1
    }
  }
  variable {
    name: "Convolution/conv/b"
    type: "Parameter"
    shape: { dim: 1 }
    initializer {
      type: "Constant"
      multiplier: 0
    }
  }
  variable {
    name: "Affine/affine/W"
    type: "Parameter"
    shape: { dim: 28 dim: 1 }
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
    name: "RecurrentInput"
    type: "Buffer"
    repeat_id: "RecurrentInput"
    shape: { dim:-1 dim: 1 dim: 28 }
  }
  variable {
    name: "Delay"
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
    name: "Affine_2"
    type: "Buffer"
    repeat_id: "RecurrentInput"
    shape: { dim:-1 dim: 1 dim: 28 }
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
    name: "Convolution"
    type: "Buffer"
    shape: { dim:-1 dim: 1 dim: 28 dim: 1 }
  }
  variable {
    name: "Softmax"
    type: "Buffer"
    shape: { dim:-1 dim: 1 dim: 28 dim: 1 }
  }
  variable {
    name: "Unpooling"
    type: "Buffer"
    shape: { dim:-1 dim: 1 dim: 28 dim: 28 }
  }
  variable {
    name: "Mul2"
    type: "Buffer"
    shape: { dim:-1 dim: 1 dim: 28 dim: 28 }
  }
  variable {
    name: "SumPooling"
    type: "Buffer"
    shape: { dim:-1 dim: 1 dim: 1 dim: 28 }
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
    name: "Convolution"
    type: "Convolution"
    input: "RecurrentOutput"
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
    name: "Softmax"
    type: "Softmax"
    input: "Convolution"
    output: "Softmax"
    softmax_param {
      axis: 2
    }
  }
  function {
    name: "Unpooling"
    type: "Unpooling"
    input: "Softmax"
    output: "Unpooling"
    unpooling_param {
      kernel: { dim: 1 dim: 28 }
    }
  }
  function {
    name: "Mul2"
    type: "Mul2"
    input: "RecurrentOutput"
    input: "Unpooling"
    output: "Mul2"
  }
  function {
    name: "SumPooling"
    type: "SumPooling"
    input: "Mul2"
    output: "SumPooling"
    sum_pooling_param {
      kernel: { dim: 28 dim: 1 }
      stride: { dim: 1 dim: 1 }
      ignore_border: true
      pad: { dim: 0 dim: 0 }
    }
  }
  function {
    name: "Affine"
    type: "Affine"
    input: "SumPooling"
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
  name: "Runtime"
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
      multiplier: 1
    }
  }
  variable {
    name: "Affine_2/affine/b"
    type: "Parameter"
    shape: { dim: 1 dim: 28 }
    initializer {
      type: "Constant"
      multiplier: 0
    }
  }
  variable {
    name: "Convolution/conv/W"
    type: "Parameter"
    shape: { dim: 1 dim: 1 dim: 1 dim: 28 }
    initializer {
      type: "NormalConvolutionGlorot"
      multiplier: 1
    }
  }
  variable {
    name: "Convolution/conv/b"
    type: "Parameter"
    shape: { dim: 1 }
    initializer {
      type: "Constant"
      multiplier: 0
    }
  }
  variable {
    name: "Affine/affine/W"
    type: "Parameter"
    shape: { dim: 28 dim: 1 }
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
    name: "RecurrentInput"
    type: "Buffer"
    repeat_id: "RecurrentInput"
    shape: { dim:-1 dim: 1 dim: 28 }
  }
  variable {
    name: "Delay"
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
    name: "Affine_2"
    type: "Buffer"
    repeat_id: "RecurrentInput"
    shape: { dim:-1 dim: 1 dim: 28 }
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
    name: "Convolution"
    type: "Buffer"
    shape: { dim:-1 dim: 1 dim: 28 dim: 1 }
  }
  variable {
    name: "rnn_out"
    type: "Buffer"
    shape: { dim:-1 dim: 1 dim: 28 dim: 28 }
  }
  variable {
    name: "Softmax"
    type: "Buffer"
    shape: { dim:-1 dim: 1 dim: 28 dim: 1 }
  }
  variable {
    name: "Unpooling"
    type: "Buffer"
    shape: { dim:-1 dim: 1 dim: 28 dim: 28 }
  }
  variable {
    name: "Mul2"
    type: "Buffer"
    shape: { dim:-1 dim: 1 dim: 28 dim: 28 }
  }
  variable {
    name: "attention"
    type: "Buffer"
    shape: { dim:-1 dim: 1 dim: 28 dim: 28 }
  }
  variable {
    name: "SumPooling"
    type: "Buffer"
    shape: { dim:-1 dim: 1 dim: 1 dim: 28 }
  }
  variable {
    name: "Affine"
    type: "Buffer"
    shape: { dim:-1 dim: 1 }
  }
  variable {
    name: "y'"
    type: "Buffer"
    shape: { dim:-1 dim: 1 }
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
    name: "Convolution"
    type: "Convolution"
    input: "RecurrentOutput"
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
    name: "rnn_out"
    type: "Identity"
    input: "RecurrentOutput"
    output: "rnn_out"
  }
  function {
    name: "Softmax"
    type: "Softmax"
    input: "Convolution"
    output: "Softmax"
    softmax_param {
      axis: 2
    }
  }
  function {
    name: "Unpooling"
    type: "Unpooling"
    input: "Softmax"
    output: "Unpooling"
    unpooling_param {
      kernel: { dim: 1 dim: 28 }
    }
  }
  function {
    name: "Mul2"
    type: "Mul2"
    input: "RecurrentOutput"
    input: "Unpooling"
    output: "Mul2"
  }
  function {
    name: "attention"
    type: "Identity"
    input: "Unpooling"
    output: "attention"
  }
  function {
    name: "SumPooling"
    type: "SumPooling"
    input: "Mul2"
    output: "SumPooling"
    sum_pooling_param {
      kernel: { dim: 28 dim: 1 }
      stride: { dim: 1 dim: 1 }
      ignore_border: true
      pad: { dim: 0 dim: 0 }
    }
  }
  function {
    name: "Affine"
    type: "Affine"
    input: "SumPooling"
    input: "Affine/affine/W"
    input: "Affine/affine/b"
    output: "Affine"
    affine_param {
      base_axis: 1
    }
  }
  function {
    name: "y'"
    type: "Sigmoid"
    input: "Affine"
    output: "y'"
  }
}
dataset {
  name: "Training"
  uri: "C:\\Utils\\neural_network_console_internal_v120\\samples\\sample_dataset\\mnist\\small_mnist_4or9_training.csv"
  cache_dir: "C:\\Utils\\neural_network_console_internal_v120\\samples\\sample_dataset\\mnist\\small_mnist_4or9_training.cache"
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
  network_name: "Main_"
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
    multiplier: 0
  }
  data_variable {
    variable_name: "BinaryCrossEntropy_T"
    data_name: "y"
  }
  loss_variable {
    variable_name: "BinaryCrossEntropy"
  }
  parameter_variable {
    variable_name: "Affine_2/affine/W"
    learning_rate_multiplier: 1
  }
  parameter_variable {
    variable_name: "Affine_2/affine/b"
    learning_rate_multiplier: 1
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
  network_name: "Main_"
  dataset_name: "Training"
  data_variable {
    variable_name: "Input"
    data_name: "x"
  }
  generator_variable {
    variable_name: "Delay_Initial"
    type: "Constant"
    multiplier: 0
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
  network_name: "Main_"
  dataset_name: "Validation"
  data_variable {
    variable_name: "Input"
    data_name: "x"
  }
  generator_variable {
    variable_name: "Delay_Initial"
    type: "Constant"
    multiplier: 0
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
  network_name: "Runtime"
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
    multiplier: 0
  }
  output_variable {
    variable_name: "attention"
    data_name: "attention"
  }
  output_variable {
    variable_name: "rnn_out"
    data_name: "rnn_out"
  }
  output_variable {
    variable_name: "y'"
    data_name: "y'"
  }
  parameter_variable {
    variable_name: "Affine_2/affine/W"
  }
  parameter_variable {
    variable_name: "Affine_2/affine/b"
  }
  parameter_variable {
    variable_name: "Convolution/conv/W"
  }
  parameter_variable {
    variable_name: "Convolution/conv/b"
  }
  parameter_variable {
    variable_name: "Affine/affine/W"
  }
  parameter_variable {
    variable_name: "Affine/affine/b"
  }
}
'''
# The nntxt is generated from /home/woody/aid/nnabla-sample-data/sample_project/tutorial/recurrent_neural_networks/elman_net_with_attention.files/20170804_142242/net.nntxt.
N00000003 = r'''global_config {
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
  name: "Main_"
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
    name: "Convolution/conv/W"
    type: "Parameter"
    shape: { dim: 1 dim: 1 dim: 1 dim: 28 }
    initializer {
      type: "NormalConvolutionGlorot"
      multiplier: 1.0
    }
  }
  variable {
    name: "Convolution/conv/b"
    type: "Parameter"
    shape: { dim: 1 }
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
    name: "Convolution"
    type: "Buffer"
    shape: { dim:-1 dim: 1 dim: 28 dim: 1 }
  }
  variable {
    name: "Softmax"
    type: "Buffer"
    shape: { dim:-1 dim: 1 dim: 28 dim: 1 }
  }
  variable {
    name: "Unpooling"
    type: "Buffer"
    shape: { dim:-1 dim: 1 dim: 28 dim: 28 }
  }
  variable {
    name: "SumPooling"
    type: "Buffer"
    shape: { dim:-1 dim: 1 dim: 1 dim: 28 }
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
    name: "Mul2"
    type: "Buffer"
    shape: { dim:-1 dim: 1 dim: 28 dim: 28 }
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
    input: "SumPooling"
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
    name: "Convolution"
    type: "Convolution"
    input: "RecurrentOutput"
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
    name: "Softmax"
    type: "Softmax"
    input: "Convolution"
    output: "Softmax"
    softmax_param {
      axis: 2
    }
  }
  function {
    name: "Unpooling"
    type: "Unpooling"
    input: "Softmax"
    output: "Unpooling"
    unpooling_param {
      kernel: { dim: 1 dim: 28 }
    }
  }
  function {
    name: "SumPooling"
    type: "SumPooling"
    input: "Mul2"
    output: "SumPooling"
    sum_pooling_param {
      kernel: { dim: 28 dim: 1 }
      stride: { dim: 1 dim: 1 }
      ignore_border: true
      pad: { dim: 0 dim: 0 }
    }
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
    name: "Mul2"
    type: "Mul2"
    input: "RecurrentOutput"
    input: "Unpooling"
    output: "Mul2"
  }
}
network {
  name: "Runtime"
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
    name: "Convolution/conv/W"
    type: "Parameter"
    shape: { dim: 1 dim: 1 dim: 1 dim: 28 }
    initializer {
      type: "NormalConvolutionGlorot"
      multiplier: 1.0
    }
  }
  variable {
    name: "Convolution/conv/b"
    type: "Parameter"
    shape: { dim: 1 }
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
    name: "y'"
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
    name: "Convolution"
    type: "Buffer"
    shape: { dim:-1 dim: 1 dim: 28 dim: 1 }
  }
  variable {
    name: "Softmax"
    type: "Buffer"
    shape: { dim:-1 dim: 1 dim: 28 dim: 1 }
  }
  variable {
    name: "Unpooling"
    type: "Buffer"
    shape: { dim:-1 dim: 1 dim: 28 dim: 28 }
  }
  variable {
    name: "SumPooling"
    type: "Buffer"
    shape: { dim:-1 dim: 1 dim: 1 dim: 28 }
  }
  variable {
    name: "RecurrentOutput"
    type: "Buffer"
    shape: { dim:-1 dim: 1 dim: 28 dim: 28 }
  }
  variable {
    name: "Mul2"
    type: "Buffer"
    shape: { dim:-1 dim: 1 dim: 28 dim: 28 }
  }
  variable {
    name: "rnn_out"
    type: "Buffer"
    shape: { dim:-1 dim: 1 dim: 28 dim: 28 }
  }
  variable {
    name: "attention"
    type: "Buffer"
    shape: { dim:-1 dim: 1 dim: 28 dim: 28 }
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
    input: "SumPooling"
    input: "Affine/affine/W"
    input: "Affine/affine/b"
    output: "Affine"
    affine_param {
      base_axis: 1
    }
  }
  function {
    name: "y'"
    type: "Sigmoid"
    input: "Affine"
    output: "y'"
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
    name: "Convolution"
    type: "Convolution"
    input: "RecurrentOutput"
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
    name: "Softmax"
    type: "Softmax"
    input: "Convolution"
    output: "Softmax"
    softmax_param {
      axis: 2
    }
  }
  function {
    name: "Unpooling"
    type: "Unpooling"
    input: "Softmax"
    output: "Unpooling"
    unpooling_param {
      kernel: { dim: 1 dim: 28 }
    }
  }
  function {
    name: "SumPooling"
    type: "SumPooling"
    input: "Mul2"
    output: "SumPooling"
    sum_pooling_param {
      kernel: { dim: 28 dim: 1 }
      stride: { dim: 1 dim: 1 }
      ignore_border: true
      pad: { dim: 0 dim: 0 }
    }
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
    name: "Mul2"
    type: "Mul2"
    input: "RecurrentOutput"
    input: "Unpooling"
    output: "Mul2"
  }
  function {
    name: "rnn_out"
    type: "Identity"
    input: "RecurrentOutput"
    output: "rnn_out"
  }
  function {
    name: "attention"
    type: "Identity"
    input: "Unpooling"
    output: "attention"
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
  network_name: "Main_"
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
  parameter_variable {
    variable_name: "Convolution/conv/W"
    learning_rate_multiplier: 1.0
  }
  parameter_variable {
    variable_name: "Convolution/conv/b"
    learning_rate_multiplier: 1.0
  }
}
monitor {
  name: "train_error"
  network_name: "Main_"
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
  network_name: "Main_"
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
  network_name: "Runtime"
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
    variable_name: "y'"
    data_name: "y'"
  }
  output_variable {
    variable_name: "rnn_out"
    data_name: "rnn_out"
  }
  output_variable {
    variable_name: "attention"
    data_name: "attention"
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
  parameter_variable {
    variable_name: "Convolution/conv/W"
  }
  parameter_variable {
    variable_name: "Convolution/conv/b"
  }
}
'''
# The nntxt is generated from /home/woody/aid/nnabla-sample-data/sample_project/tutorial/recurrent_neural_networks/bidirectional_elman_net.files/20180720_185920/net.nntxt.
N00000004 = r'''global_config {
  default_context {
    array_class: "CudaCachedArray"
    device_id: "0"
    backends: "cudnn:float"
    backends: "cuda:float"
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
    id: "RecurrentInput"
    times: 28
  }
  repeat_info {
    id: "RecurrentInput_2"
    times: 28
  }
  variable {
    name: "Input"
    type: "Buffer"
    shape: { dim:-1 dim: 1 dim: 28 dim: 28 }
  }
  variable {
    name: "Delay_Initial"
    type: "Buffer"
    shape: { dim:-1 dim: 1 dim: 28 }
  }
  variable {
    name: "Delay_2_Initial"
    type: "Buffer"
    shape: { dim:-1 dim: 1 dim: 28 }
  }
  variable {
    name: "Affine_2/affine/W"
    type: "Parameter"
    shape: { dim: 56 dim: 1 dim: 28 }
    initializer {
      type: "NormalAffineGlorot"
      multiplier: 1
    }
  }
  variable {
    name: "Affine_2/affine/b"
    type: "Parameter"
    shape: { dim: 1 dim: 28 }
    initializer {
      type: "Constant"
      multiplier: 0
    }
  }
  variable {
    name: "Affine_3/affine/W"
    type: "Parameter"
    shape: { dim: 56 dim: 1 dim: 28 }
    initializer {
      type: "NormalAffineGlorot"
      multiplier: 1
    }
  }
  variable {
    name: "Affine_3/affine/b"
    type: "Parameter"
    shape: { dim: 1 dim: 28 }
    initializer {
      type: "Constant"
      multiplier: 0
    }
  }
  variable {
    name: "Affine/affine/W"
    type: "Parameter"
    shape: { dim: 56 dim: 1 }
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
    name: "Output_T"
    type: "Buffer"
    shape: { dim:-1 dim: 1 }
  }
  variable {
    name: "RecurrentInput"
    type: "Buffer"
    repeat_id: "RecurrentInput"
    shape: { dim:-1 dim: 1 dim: 28 }
  }
  variable {
    name: "Flip"
    type: "Buffer"
    shape: { dim:-1 dim: 1 dim: 28 dim: 28 }
  }
  variable {
    name: "RecurrentInput_2"
    type: "Buffer"
    repeat_id: "RecurrentInput_2"
    shape: { dim:-1 dim: 1 dim: 28 }
  }
  variable {
    name: "Delay"
    type: "Buffer"
    repeat_id: "RecurrentInput"
    shape: { dim:-1 dim: 1 dim: 28 }
  }
  variable {
    name: "Delay_2"
    type: "Buffer"
    repeat_id: "RecurrentInput_2"
    shape: { dim:-1 dim: 1 dim: 28 }
  }
  variable {
    name: "Concatenate"
    type: "Buffer"
    repeat_id: "RecurrentInput"
    shape: { dim:-1 dim: 2 dim: 28 }
  }
  variable {
    name: "Concatenate_2"
    type: "Buffer"
    repeat_id: "RecurrentInput_2"
    shape: { dim:-1 dim: 2 dim: 28 }
  }
  variable {
    name: "Affine_2"
    type: "Buffer"
    repeat_id: "RecurrentInput"
    shape: { dim:-1 dim: 1 dim: 28 }
  }
  variable {
    name: "Tanh"
    type: "Buffer"
    repeat_id: "RecurrentInput"
    shape: { dim:-1 dim: 1 dim: 28 }
  }
  variable {
    name: "Affine_3"
    type: "Buffer"
    repeat_id: "RecurrentInput_2"
    shape: { dim:-1 dim: 1 dim: 28 }
  }
  variable {
    name: "Tanh_2"
    type: "Buffer"
    repeat_id: "RecurrentInput_2"
    shape: { dim:-1 dim: 1 dim: 28 }
  }
  variable {
    name: "RecurrentOutput"
    type: "Buffer"
    shape: { dim:-1 dim: 1 dim: 28 dim: 28 }
  }
  variable {
    name: "RecurrentOutput_2"
    type: "Buffer"
    shape: { dim:-1 dim: 1 dim: 28 dim: 28 }
  }
  variable {
    name: "Slice"
    type: "Buffer"
    shape: { dim:-1 dim: 1 dim: 1 dim: 28 }
  }
  variable {
    name: "Slice_2"
    type: "Buffer"
    shape: { dim:-1 dim: 1 dim: 1 dim: 28 }
  }
  variable {
    name: "Concatenate_3"
    type: "Buffer"
    shape: { dim:-1 dim: 2 dim: 1 dim: 28 }
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
    name: "Output"
    type: "Buffer"
    shape: { dim:-1 dim: 1 }
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
    name: "Flip"
    type: "Flip"
    input: "Input"
    output: "Flip"
    flip_param {
      axes: 2
    }
  }
  function {
    name: "RecurrentInput_2"
    type: "RecurrentInput"
    repeat_id: "RecurrentInput_2"
    input: "Flip"
    output: "RecurrentInput_2"
    recurrent_param {
      axis: 2
      repeat_id: "RecurrentInput_2"
    }
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
    name: "Delay_2"
    type: "Delay"
    repeat_id: "RecurrentInput_2"
    input: "Tanh_2"
    input: "Delay_2_Initial"
    output: "Delay_2"
    recurrent_param {
      repeat_id: "RecurrentInput_2"
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
    name: "Concatenate_2"
    type: "Concatenate"
    repeat_id: "RecurrentInput_2"
    input: "RecurrentInput_2"
    input: "Delay_2"
    output: "Concatenate_2"
    concatenate_param {
      axis: 1
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
    name: "Tanh"
    type: "Tanh"
    repeat_id: "RecurrentInput"
    input: "Affine_2"
    output: "Tanh"
  }
  function {
    name: "Affine_3"
    type: "Affine"
    repeat_id: "RecurrentInput_2"
    input: "Concatenate_2"
    input: "Affine_3/affine/W"
    input: "Affine_3/affine/b"
    output: "Affine_3"
    affine_param {
      base_axis: 1
    }
  }
  function {
    name: "Tanh_2"
    type: "Tanh"
    repeat_id: "RecurrentInput_2"
    input: "Affine_3"
    output: "Tanh_2"
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
    name: "RecurrentOutput_2"
    type: "RecurrentOutput"
    input: "Tanh_2"
    output: "RecurrentOutput_2"
    recurrent_param {
      axis: 2
      repeat_id: "RecurrentInput_2"
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
  function {
    name: "Slice_2"
    type: "Slice"
    input: "RecurrentOutput_2"
    output: "Slice_2"
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
  function {
    name: "Concatenate_3"
    type: "Concatenate"
    input: "Slice"
    input: "Slice_2"
    output: "Concatenate_3"
    concatenate_param {
      axis: 1
    }
  }
  function {
    name: "Affine"
    type: "Affine"
    input: "Concatenate_3"
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
    name: "Output"
    type: "BinaryCrossEntropy"
    input: "Sigmoid"
    input: "Output_T"
    output: "Output"
  }
}
network {
  name: "MainValidation"
  batch_size: 64
  repeat_info {
    id: "RecurrentInput"
    times: 28
  }
  repeat_info {
    id: "RecurrentInput_2"
    times: 28
  }
  variable {
    name: "Input"
    type: "Buffer"
    shape: { dim:-1 dim: 1 dim: 28 dim: 28 }
  }
  variable {
    name: "Delay_Initial"
    type: "Buffer"
    shape: { dim:-1 dim: 1 dim: 28 }
  }
  variable {
    name: "Delay_2_Initial"
    type: "Buffer"
    shape: { dim:-1 dim: 1 dim: 28 }
  }
  variable {
    name: "Affine_2/affine/W"
    type: "Parameter"
    shape: { dim: 56 dim: 1 dim: 28 }
    initializer {
      type: "NormalAffineGlorot"
      multiplier: 1
    }
  }
  variable {
    name: "Affine_2/affine/b"
    type: "Parameter"
    shape: { dim: 1 dim: 28 }
    initializer {
      type: "Constant"
      multiplier: 0
    }
  }
  variable {
    name: "Affine_3/affine/W"
    type: "Parameter"
    shape: { dim: 56 dim: 1 dim: 28 }
    initializer {
      type: "NormalAffineGlorot"
      multiplier: 1
    }
  }
  variable {
    name: "Affine_3/affine/b"
    type: "Parameter"
    shape: { dim: 1 dim: 28 }
    initializer {
      type: "Constant"
      multiplier: 0
    }
  }
  variable {
    name: "Affine/affine/W"
    type: "Parameter"
    shape: { dim: 56 dim: 1 }
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
    name: "Output_T"
    type: "Buffer"
    shape: { dim:-1 dim: 1 }
  }
  variable {
    name: "RecurrentInput"
    type: "Buffer"
    repeat_id: "RecurrentInput"
    shape: { dim:-1 dim: 1 dim: 28 }
  }
  variable {
    name: "Flip"
    type: "Buffer"
    shape: { dim:-1 dim: 1 dim: 28 dim: 28 }
  }
  variable {
    name: "RecurrentInput_2"
    type: "Buffer"
    repeat_id: "RecurrentInput_2"
    shape: { dim:-1 dim: 1 dim: 28 }
  }
  variable {
    name: "Delay"
    type: "Buffer"
    repeat_id: "RecurrentInput"
    shape: { dim:-1 dim: 1 dim: 28 }
  }
  variable {
    name: "Delay_2"
    type: "Buffer"
    repeat_id: "RecurrentInput_2"
    shape: { dim:-1 dim: 1 dim: 28 }
  }
  variable {
    name: "Concatenate"
    type: "Buffer"
    repeat_id: "RecurrentInput"
    shape: { dim:-1 dim: 2 dim: 28 }
  }
  variable {
    name: "Concatenate_2"
    type: "Buffer"
    repeat_id: "RecurrentInput_2"
    shape: { dim:-1 dim: 2 dim: 28 }
  }
  variable {
    name: "Affine_2"
    type: "Buffer"
    repeat_id: "RecurrentInput"
    shape: { dim:-1 dim: 1 dim: 28 }
  }
  variable {
    name: "Tanh"
    type: "Buffer"
    repeat_id: "RecurrentInput"
    shape: { dim:-1 dim: 1 dim: 28 }
  }
  variable {
    name: "Affine_3"
    type: "Buffer"
    repeat_id: "RecurrentInput_2"
    shape: { dim:-1 dim: 1 dim: 28 }
  }
  variable {
    name: "Tanh_2"
    type: "Buffer"
    repeat_id: "RecurrentInput_2"
    shape: { dim:-1 dim: 1 dim: 28 }
  }
  variable {
    name: "RecurrentOutput"
    type: "Buffer"
    shape: { dim:-1 dim: 1 dim: 28 dim: 28 }
  }
  variable {
    name: "RecurrentOutput_2"
    type: "Buffer"
    shape: { dim:-1 dim: 1 dim: 28 dim: 28 }
  }
  variable {
    name: "Slice"
    type: "Buffer"
    shape: { dim:-1 dim: 1 dim: 1 dim: 28 }
  }
  variable {
    name: "Slice_2"
    type: "Buffer"
    shape: { dim:-1 dim: 1 dim: 1 dim: 28 }
  }
  variable {
    name: "Concatenate_3"
    type: "Buffer"
    shape: { dim:-1 dim: 2 dim: 1 dim: 28 }
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
    name: "Output"
    type: "Buffer"
    shape: { dim:-1 dim: 1 }
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
    name: "Flip"
    type: "Flip"
    input: "Input"
    output: "Flip"
    flip_param {
      axes: 2
    }
  }
  function {
    name: "RecurrentInput_2"
    type: "RecurrentInput"
    repeat_id: "RecurrentInput_2"
    input: "Flip"
    output: "RecurrentInput_2"
    recurrent_param {
      axis: 2
      repeat_id: "RecurrentInput_2"
    }
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
    name: "Delay_2"
    type: "Delay"
    repeat_id: "RecurrentInput_2"
    input: "Tanh_2"
    input: "Delay_2_Initial"
    output: "Delay_2"
    recurrent_param {
      repeat_id: "RecurrentInput_2"
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
    name: "Concatenate_2"
    type: "Concatenate"
    repeat_id: "RecurrentInput_2"
    input: "RecurrentInput_2"
    input: "Delay_2"
    output: "Concatenate_2"
    concatenate_param {
      axis: 1
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
    name: "Tanh"
    type: "Tanh"
    repeat_id: "RecurrentInput"
    input: "Affine_2"
    output: "Tanh"
  }
  function {
    name: "Affine_3"
    type: "Affine"
    repeat_id: "RecurrentInput_2"
    input: "Concatenate_2"
    input: "Affine_3/affine/W"
    input: "Affine_3/affine/b"
    output: "Affine_3"
    affine_param {
      base_axis: 1
    }
  }
  function {
    name: "Tanh_2"
    type: "Tanh"
    repeat_id: "RecurrentInput_2"
    input: "Affine_3"
    output: "Tanh_2"
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
    name: "RecurrentOutput_2"
    type: "RecurrentOutput"
    input: "Tanh_2"
    output: "RecurrentOutput_2"
    recurrent_param {
      axis: 2
      repeat_id: "RecurrentInput_2"
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
  function {
    name: "Slice_2"
    type: "Slice"
    input: "RecurrentOutput_2"
    output: "Slice_2"
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
  function {
    name: "Concatenate_3"
    type: "Concatenate"
    input: "Slice"
    input: "Slice_2"
    output: "Concatenate_3"
    concatenate_param {
      axis: 1
    }
  }
  function {
    name: "Affine"
    type: "Affine"
    input: "Concatenate_3"
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
    name: "Output"
    type: "BinaryCrossEntropy"
    input: "Sigmoid"
    input: "Output_T"
    output: "Output"
  }
}
network {
  name: "MainRuntime"
  batch_size: 64
  repeat_info {
    id: "RecurrentInput"
    times: 28
  }
  repeat_info {
    id: "RecurrentInput_2"
    times: 28
  }
  variable {
    name: "Input"
    type: "Buffer"
    shape: { dim:-1 dim: 1 dim: 28 dim: 28 }
  }
  variable {
    name: "Delay_Initial"
    type: "Buffer"
    shape: { dim:-1 dim: 1 dim: 28 }
  }
  variable {
    name: "Delay_2_Initial"
    type: "Buffer"
    shape: { dim:-1 dim: 1 dim: 28 }
  }
  variable {
    name: "Affine_2/affine/W"
    type: "Parameter"
    shape: { dim: 56 dim: 1 dim: 28 }
    initializer {
      type: "NormalAffineGlorot"
      multiplier: 1
    }
  }
  variable {
    name: "Affine_2/affine/b"
    type: "Parameter"
    shape: { dim: 1 dim: 28 }
    initializer {
      type: "Constant"
      multiplier: 0
    }
  }
  variable {
    name: "Affine_3/affine/W"
    type: "Parameter"
    shape: { dim: 56 dim: 1 dim: 28 }
    initializer {
      type: "NormalAffineGlorot"
      multiplier: 1
    }
  }
  variable {
    name: "Affine_3/affine/b"
    type: "Parameter"
    shape: { dim: 1 dim: 28 }
    initializer {
      type: "Constant"
      multiplier: 0
    }
  }
  variable {
    name: "Affine/affine/W"
    type: "Parameter"
    shape: { dim: 56 dim: 1 }
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
    name: "RecurrentInput"
    type: "Buffer"
    repeat_id: "RecurrentInput"
    shape: { dim:-1 dim: 1 dim: 28 }
  }
  variable {
    name: "Flip"
    type: "Buffer"
    shape: { dim:-1 dim: 1 dim: 28 dim: 28 }
  }
  variable {
    name: "RecurrentInput_2"
    type: "Buffer"
    repeat_id: "RecurrentInput_2"
    shape: { dim:-1 dim: 1 dim: 28 }
  }
  variable {
    name: "Delay"
    type: "Buffer"
    repeat_id: "RecurrentInput"
    shape: { dim:-1 dim: 1 dim: 28 }
  }
  variable {
    name: "Delay_2"
    type: "Buffer"
    repeat_id: "RecurrentInput_2"
    shape: { dim:-1 dim: 1 dim: 28 }
  }
  variable {
    name: "Concatenate"
    type: "Buffer"
    repeat_id: "RecurrentInput"
    shape: { dim:-1 dim: 2 dim: 28 }
  }
  variable {
    name: "Concatenate_2"
    type: "Buffer"
    repeat_id: "RecurrentInput_2"
    shape: { dim:-1 dim: 2 dim: 28 }
  }
  variable {
    name: "Affine_2"
    type: "Buffer"
    repeat_id: "RecurrentInput"
    shape: { dim:-1 dim: 1 dim: 28 }
  }
  variable {
    name: "Tanh"
    type: "Buffer"
    repeat_id: "RecurrentInput"
    shape: { dim:-1 dim: 1 dim: 28 }
  }
  variable {
    name: "Affine_3"
    type: "Buffer"
    repeat_id: "RecurrentInput_2"
    shape: { dim:-1 dim: 1 dim: 28 }
  }
  variable {
    name: "Tanh_2"
    type: "Buffer"
    repeat_id: "RecurrentInput_2"
    shape: { dim:-1 dim: 1 dim: 28 }
  }
  variable {
    name: "RecurrentOutput"
    type: "Buffer"
    shape: { dim:-1 dim: 1 dim: 28 dim: 28 }
  }
  variable {
    name: "RecurrentOutput_2"
    type: "Buffer"
    shape: { dim:-1 dim: 1 dim: 28 dim: 28 }
  }
  variable {
    name: "Slice"
    type: "Buffer"
    shape: { dim:-1 dim: 1 dim: 1 dim: 28 }
  }
  variable {
    name: "Slice_2"
    type: "Buffer"
    shape: { dim:-1 dim: 1 dim: 1 dim: 28 }
  }
  variable {
    name: "Concatenate_3"
    type: "Buffer"
    shape: { dim:-1 dim: 2 dim: 1 dim: 28 }
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
    name: "Flip"
    type: "Flip"
    input: "Input"
    output: "Flip"
    flip_param {
      axes: 2
    }
  }
  function {
    name: "RecurrentInput_2"
    type: "RecurrentInput"
    repeat_id: "RecurrentInput_2"
    input: "Flip"
    output: "RecurrentInput_2"
    recurrent_param {
      axis: 2
      repeat_id: "RecurrentInput_2"
    }
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
    name: "Delay_2"
    type: "Delay"
    repeat_id: "RecurrentInput_2"
    input: "Tanh_2"
    input: "Delay_2_Initial"
    output: "Delay_2"
    recurrent_param {
      repeat_id: "RecurrentInput_2"
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
    name: "Concatenate_2"
    type: "Concatenate"
    repeat_id: "RecurrentInput_2"
    input: "RecurrentInput_2"
    input: "Delay_2"
    output: "Concatenate_2"
    concatenate_param {
      axis: 1
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
    name: "Tanh"
    type: "Tanh"
    repeat_id: "RecurrentInput"
    input: "Affine_2"
    output: "Tanh"
  }
  function {
    name: "Affine_3"
    type: "Affine"
    repeat_id: "RecurrentInput_2"
    input: "Concatenate_2"
    input: "Affine_3/affine/W"
    input: "Affine_3/affine/b"
    output: "Affine_3"
    affine_param {
      base_axis: 1
    }
  }
  function {
    name: "Tanh_2"
    type: "Tanh"
    repeat_id: "RecurrentInput_2"
    input: "Affine_3"
    output: "Tanh_2"
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
    name: "RecurrentOutput_2"
    type: "RecurrentOutput"
    input: "Tanh_2"
    output: "RecurrentOutput_2"
    recurrent_param {
      axis: 2
      repeat_id: "RecurrentInput_2"
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
  function {
    name: "Slice_2"
    type: "Slice"
    input: "RecurrentOutput_2"
    output: "Slice_2"
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
  function {
    name: "Concatenate_3"
    type: "Concatenate"
    input: "Slice"
    input: "Slice_2"
    output: "Concatenate_3"
    concatenate_param {
      axis: 1
    }
  }
  function {
    name: "Affine"
    type: "Affine"
    input: "Concatenate_3"
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
  uri: "C:\\Utils\\neural_network_console_internal_v120\\samples\\sample_dataset\\mnist\\small_mnist_4or9_training.csv"
  cache_dir: "C:\\Utils\\neural_network_console_internal_v120\\samples\\sample_dataset\\mnist\\small_mnist_4or9_training.cache"
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
  generator_variable {
    variable_name: "Delay_Initial"
    type: "Constant"
    multiplier: 0
  }
  generator_variable {
    variable_name: "Delay_2_Initial"
    type: "Constant"
    multiplier: 0
  }
  data_variable {
    variable_name: "Output_T"
    data_name: "y"
  }
  loss_variable {
    variable_name: "Output"
  }
  parameter_variable {
    variable_name: "Affine_2/affine/W"
    learning_rate_multiplier: 1
  }
  parameter_variable {
    variable_name: "Affine_2/affine/b"
    learning_rate_multiplier: 1
  }
  parameter_variable {
    variable_name: "Affine_3/affine/W"
    learning_rate_multiplier: 1
  }
  parameter_variable {
    variable_name: "Affine_3/affine/b"
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
    multiplier: 0
  }
  generator_variable {
    variable_name: "Delay_2_Initial"
    type: "Constant"
    multiplier: 0
  }
  data_variable {
    variable_name: "Output_T"
    data_name: "y"
  }
  monitor_variable {
    type: "Error"
    variable_name: "Output"
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
    multiplier: 0
  }
  generator_variable {
    variable_name: "Delay_2_Initial"
    type: "Constant"
    multiplier: 0
  }
  data_variable {
    variable_name: "Output_T"
    data_name: "y"
  }
  monitor_variable {
    type: "Error"
    variable_name: "Output"
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
    multiplier: 0
  }
  generator_variable {
    variable_name: "Delay_2_Initial"
    type: "Constant"
    multiplier: 0
  }
  output_variable {
    variable_name: "Sigmoid"
    data_name: "y'"
  }
  parameter_variable {
    variable_name: "Affine_2/affine/W"
  }
  parameter_variable {
    variable_name: "Affine_2/affine/b"
  }
  parameter_variable {
    variable_name: "Affine_3/affine/W"
  }
  parameter_variable {
    variable_name: "Affine_3/affine/b"
  }
  parameter_variable {
    variable_name: "Affine/affine/W"
  }
  parameter_variable {
    variable_name: "Affine/affine/b"
  }
}
'''
# The nntxt is generated from /home/woody/aid/nnabla-sample-data/sample_project/tutorial/recurrent_neural_networks/bidirectional_elman_net.files/20170804_142121/net.nntxt.
N00000005 = r'''global_config {
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
  repeat_info {
    id: "RecurrentInput_2"
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
    name: "Flip"
    type: "Buffer"
    shape: { dim:-1 dim: 1 dim: 28 dim: 28 }
  }
  variable {
    name: "RecurrentInput_2"
    type: "Buffer"
    repeat_id: "RecurrentInput_2"
    shape: { dim:-1 dim: 1 dim: 28 }
  }
  variable {
    name: "Affine/affine/W"
    type: "Parameter"
    shape: { dim: 56 dim: 1 }
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
    name: "Output_T"
    type: "Buffer"
    shape: { dim:-1 dim: 1 }
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
    name: "Delay_2_Initial"
    type: "Buffer"
    shape: { dim:-1 dim: 1 dim: 28 }
  }
  variable {
    name: "Affine_3/affine/W"
    type: "Parameter"
    shape: { dim: 56 dim: 1 dim: 28 }
    initializer {
      type: "NormalAffineGlorot"
      multiplier: 1.0
    }
  }
  variable {
    name: "Affine_3/affine/b"
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
    name: "Output"
    type: "Buffer"
    shape: { dim:-1 dim: 1 }
  }
  variable {
    name: "RecurrentOutput"
    type: "Buffer"
    shape: { dim:-1 dim: 1 dim: 28 dim: 28 }
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
    name: "RecurrentOutput_2"
    type: "Buffer"
    shape: { dim:-1 dim: 1 dim: 28 dim: 28 }
  }
  variable {
    name: "Delay_2"
    type: "Buffer"
    repeat_id: "RecurrentInput_2"
    shape: { dim:-1 dim: 1 dim: 28 }
  }
  variable {
    name: "Affine_3"
    type: "Buffer"
    repeat_id: "RecurrentInput_2"
    shape: { dim:-1 dim: 1 dim: 28 }
  }
  variable {
    name: "Concatenate_2"
    type: "Buffer"
    repeat_id: "RecurrentInput_2"
    shape: { dim:-1 dim: 2 dim: 28 }
  }
  variable {
    name: "Tanh_2"
    type: "Buffer"
    repeat_id: "RecurrentInput_2"
    shape: { dim:-1 dim: 1 dim: 28 }
  }
  variable {
    name: "Concatenate_3"
    type: "Buffer"
    shape: { dim:-1 dim: 2 dim: 1 dim: 28 }
  }
  variable {
    name: "Slice"
    type: "Buffer"
    shape: { dim:-1 dim: 1 dim: 1 dim: 28 }
  }
  variable {
    name: "Slice_2"
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
    name: "Flip"
    type: "Flip"
    input: "Input"
    output: "Flip"
    flip_param {
      axes: 2
    }
  }
  function {
    name: "RecurrentInput_2"
    type: "RecurrentInput"
    repeat_id: "RecurrentInput_2"
    input: "Flip"
    output: "RecurrentInput_2"
    recurrent_param {
      axis: 2
      repeat_id: "RecurrentInput_2"
    }
  }
  function {
    name: "Affine"
    type: "Affine"
    input: "Concatenate_3"
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
    name: "Output"
    type: "BinaryCrossEntropy"
    input: "Sigmoid"
    input: "Output_T"
    output: "Output"
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
    name: "RecurrentOutput_2"
    type: "RecurrentOutput"
    input: "Tanh_2"
    output: "RecurrentOutput_2"
    recurrent_param {
      axis: 2
      repeat_id: "RecurrentInput_2"
      length: 28
    }
  }
  function {
    name: "Delay_2"
    type: "Delay"
    repeat_id: "RecurrentInput_2"
    input: "Tanh_2"
    input: "Delay_2_Initial"
    output: "Delay_2"
    recurrent_param {
      repeat_id: "RecurrentInput_2"
    }
  }
  function {
    name: "Affine_3"
    type: "Affine"
    repeat_id: "RecurrentInput_2"
    input: "Concatenate_2"
    input: "Affine_3/affine/W"
    input: "Affine_3/affine/b"
    output: "Affine_3"
    affine_param {
      base_axis: 1
    }
  }
  function {
    name: "Concatenate_2"
    type: "Concatenate"
    repeat_id: "RecurrentInput_2"
    input: "RecurrentInput_2"
    input: "Delay_2"
    output: "Concatenate_2"
    concatenate_param {
      axis: 1
    }
  }
  function {
    name: "Tanh_2"
    type: "Tanh"
    repeat_id: "RecurrentInput_2"
    input: "Affine_3"
    output: "Tanh_2"
  }
  function {
    name: "Concatenate_3"
    type: "Concatenate"
    input: "Slice"
    input: "Slice_2"
    output: "Concatenate_3"
    concatenate_param {
      axis: 1
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
  function {
    name: "Slice_2"
    type: "Slice"
    input: "RecurrentOutput_2"
    output: "Slice_2"
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
  repeat_info {
    id: "RecurrentInput_2"
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
    name: "Flip"
    type: "Buffer"
    shape: { dim:-1 dim: 1 dim: 28 dim: 28 }
  }
  variable {
    name: "RecurrentInput_2"
    type: "Buffer"
    repeat_id: "RecurrentInput_2"
    shape: { dim:-1 dim: 1 dim: 28 }
  }
  variable {
    name: "Affine/affine/W"
    type: "Parameter"
    shape: { dim: 56 dim: 1 }
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
    name: "Output_T"
    type: "Buffer"
    shape: { dim:-1 dim: 1 }
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
    name: "Delay_2_Initial"
    type: "Buffer"
    shape: { dim:-1 dim: 1 dim: 28 }
  }
  variable {
    name: "Affine_3/affine/W"
    type: "Parameter"
    shape: { dim: 56 dim: 1 dim: 28 }
    initializer {
      type: "NormalAffineGlorot"
      multiplier: 1.0
    }
  }
  variable {
    name: "Affine_3/affine/b"
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
    name: "Output"
    type: "Buffer"
    shape: { dim:-1 dim: 1 }
  }
  variable {
    name: "RecurrentOutput"
    type: "Buffer"
    shape: { dim:-1 dim: 1 dim: 28 dim: 28 }
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
    name: "RecurrentOutput_2"
    type: "Buffer"
    shape: { dim:-1 dim: 1 dim: 28 dim: 28 }
  }
  variable {
    name: "Delay_2"
    type: "Buffer"
    repeat_id: "RecurrentInput_2"
    shape: { dim:-1 dim: 1 dim: 28 }
  }
  variable {
    name: "Affine_3"
    type: "Buffer"
    repeat_id: "RecurrentInput_2"
    shape: { dim:-1 dim: 1 dim: 28 }
  }
  variable {
    name: "Concatenate_2"
    type: "Buffer"
    repeat_id: "RecurrentInput_2"
    shape: { dim:-1 dim: 2 dim: 28 }
  }
  variable {
    name: "Tanh_2"
    type: "Buffer"
    repeat_id: "RecurrentInput_2"
    shape: { dim:-1 dim: 1 dim: 28 }
  }
  variable {
    name: "Concatenate_3"
    type: "Buffer"
    shape: { dim:-1 dim: 2 dim: 1 dim: 28 }
  }
  variable {
    name: "Slice"
    type: "Buffer"
    shape: { dim:-1 dim: 1 dim: 1 dim: 28 }
  }
  variable {
    name: "Slice_2"
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
    name: "Flip"
    type: "Flip"
    input: "Input"
    output: "Flip"
    flip_param {
      axes: 2
    }
  }
  function {
    name: "RecurrentInput_2"
    type: "RecurrentInput"
    repeat_id: "RecurrentInput_2"
    input: "Flip"
    output: "RecurrentInput_2"
    recurrent_param {
      axis: 2
      repeat_id: "RecurrentInput_2"
    }
  }
  function {
    name: "Affine"
    type: "Affine"
    input: "Concatenate_3"
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
    name: "Output"
    type: "BinaryCrossEntropy"
    input: "Sigmoid"
    input: "Output_T"
    output: "Output"
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
    name: "RecurrentOutput_2"
    type: "RecurrentOutput"
    input: "Tanh_2"
    output: "RecurrentOutput_2"
    recurrent_param {
      axis: 2
      repeat_id: "RecurrentInput_2"
      length: 28
    }
  }
  function {
    name: "Delay_2"
    type: "Delay"
    repeat_id: "RecurrentInput_2"
    input: "Tanh_2"
    input: "Delay_2_Initial"
    output: "Delay_2"
    recurrent_param {
      repeat_id: "RecurrentInput_2"
    }
  }
  function {
    name: "Affine_3"
    type: "Affine"
    repeat_id: "RecurrentInput_2"
    input: "Concatenate_2"
    input: "Affine_3/affine/W"
    input: "Affine_3/affine/b"
    output: "Affine_3"
    affine_param {
      base_axis: 1
    }
  }
  function {
    name: "Concatenate_2"
    type: "Concatenate"
    repeat_id: "RecurrentInput_2"
    input: "RecurrentInput_2"
    input: "Delay_2"
    output: "Concatenate_2"
    concatenate_param {
      axis: 1
    }
  }
  function {
    name: "Tanh_2"
    type: "Tanh"
    repeat_id: "RecurrentInput_2"
    input: "Affine_3"
    output: "Tanh_2"
  }
  function {
    name: "Concatenate_3"
    type: "Concatenate"
    input: "Slice"
    input: "Slice_2"
    output: "Concatenate_3"
    concatenate_param {
      axis: 1
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
  function {
    name: "Slice_2"
    type: "Slice"
    input: "RecurrentOutput_2"
    output: "Slice_2"
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
  repeat_info {
    id: "RecurrentInput_2"
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
    name: "Flip"
    type: "Buffer"
    shape: { dim:-1 dim: 1 dim: 28 dim: 28 }
  }
  variable {
    name: "RecurrentInput_2"
    type: "Buffer"
    repeat_id: "RecurrentInput_2"
    shape: { dim:-1 dim: 1 dim: 28 }
  }
  variable {
    name: "Affine/affine/W"
    type: "Parameter"
    shape: { dim: 56 dim: 1 }
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
    name: "Delay_2_Initial"
    type: "Buffer"
    shape: { dim:-1 dim: 1 dim: 28 }
  }
  variable {
    name: "Affine_3/affine/W"
    type: "Parameter"
    shape: { dim: 56 dim: 1 dim: 28 }
    initializer {
      type: "NormalAffineGlorot"
      multiplier: 1.0
    }
  }
  variable {
    name: "Affine_3/affine/b"
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
    name: "RecurrentOutput"
    type: "Buffer"
    shape: { dim:-1 dim: 1 dim: 28 dim: 28 }
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
    name: "RecurrentOutput_2"
    type: "Buffer"
    shape: { dim:-1 dim: 1 dim: 28 dim: 28 }
  }
  variable {
    name: "Delay_2"
    type: "Buffer"
    repeat_id: "RecurrentInput_2"
    shape: { dim:-1 dim: 1 dim: 28 }
  }
  variable {
    name: "Affine_3"
    type: "Buffer"
    repeat_id: "RecurrentInput_2"
    shape: { dim:-1 dim: 1 dim: 28 }
  }
  variable {
    name: "Concatenate_2"
    type: "Buffer"
    repeat_id: "RecurrentInput_2"
    shape: { dim:-1 dim: 2 dim: 28 }
  }
  variable {
    name: "Tanh_2"
    type: "Buffer"
    repeat_id: "RecurrentInput_2"
    shape: { dim:-1 dim: 1 dim: 28 }
  }
  variable {
    name: "Concatenate_3"
    type: "Buffer"
    shape: { dim:-1 dim: 2 dim: 1 dim: 28 }
  }
  variable {
    name: "Slice"
    type: "Buffer"
    shape: { dim:-1 dim: 1 dim: 1 dim: 28 }
  }
  variable {
    name: "Slice_2"
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
    name: "Flip"
    type: "Flip"
    input: "Input"
    output: "Flip"
    flip_param {
      axes: 2
    }
  }
  function {
    name: "RecurrentInput_2"
    type: "RecurrentInput"
    repeat_id: "RecurrentInput_2"
    input: "Flip"
    output: "RecurrentInput_2"
    recurrent_param {
      axis: 2
      repeat_id: "RecurrentInput_2"
    }
  }
  function {
    name: "Affine"
    type: "Affine"
    input: "Concatenate_3"
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
    name: "RecurrentOutput_2"
    type: "RecurrentOutput"
    input: "Tanh_2"
    output: "RecurrentOutput_2"
    recurrent_param {
      axis: 2
      repeat_id: "RecurrentInput_2"
      length: 28
    }
  }
  function {
    name: "Delay_2"
    type: "Delay"
    repeat_id: "RecurrentInput_2"
    input: "Tanh_2"
    input: "Delay_2_Initial"
    output: "Delay_2"
    recurrent_param {
      repeat_id: "RecurrentInput_2"
    }
  }
  function {
    name: "Affine_3"
    type: "Affine"
    repeat_id: "RecurrentInput_2"
    input: "Concatenate_2"
    input: "Affine_3/affine/W"
    input: "Affine_3/affine/b"
    output: "Affine_3"
    affine_param {
      base_axis: 1
    }
  }
  function {
    name: "Concatenate_2"
    type: "Concatenate"
    repeat_id: "RecurrentInput_2"
    input: "RecurrentInput_2"
    input: "Delay_2"
    output: "Concatenate_2"
    concatenate_param {
      axis: 1
    }
  }
  function {
    name: "Tanh_2"
    type: "Tanh"
    repeat_id: "RecurrentInput_2"
    input: "Affine_3"
    output: "Tanh_2"
  }
  function {
    name: "Concatenate_3"
    type: "Concatenate"
    input: "Slice"
    input: "Slice_2"
    output: "Concatenate_3"
    concatenate_param {
      axis: 1
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
  function {
    name: "Slice_2"
    type: "Slice"
    input: "RecurrentOutput_2"
    output: "Slice_2"
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
  data_variable {
    variable_name: "Output_T"
    data_name: "y"
  }
  generator_variable {
    variable_name: "Delay_Initial"
    type: "Constant"
    multiplier: 0.0
  }
  generator_variable {
    variable_name: "Delay_2_Initial"
    type: "Constant"
    multiplier: 0.0
  }
  loss_variable {
    variable_name: "Output"
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
  parameter_variable {
    variable_name: "Affine_3/affine/W"
    learning_rate_multiplier: 1.0
  }
  parameter_variable {
    variable_name: "Affine_3/affine/b"
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
  data_variable {
    variable_name: "Output_T"
    data_name: "y"
  }
  generator_variable {
    variable_name: "Delay_Initial"
    type: "Constant"
    multiplier: 0.0
  }
  generator_variable {
    variable_name: "Delay_2_Initial"
    type: "Constant"
    multiplier: 0.0
  }
  monitor_variable {
    type: "Error"
    variable_name: "Output"
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
    variable_name: "Output_T"
    data_name: "y"
  }
  generator_variable {
    variable_name: "Delay_Initial"
    type: "Constant"
    multiplier: 0.0
  }
  generator_variable {
    variable_name: "Delay_2_Initial"
    type: "Constant"
    multiplier: 0.0
  }
  monitor_variable {
    type: "Error"
    variable_name: "Output"
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
  generator_variable {
    variable_name: "Delay_2_Initial"
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
  parameter_variable {
    variable_name: "Affine_3/affine/W"
  }
  parameter_variable {
    variable_name: "Affine_3/affine/b"
  }
}
'''
# The nntxt is generated from /home/woody/aid/nnabla-sample-data/sample_project/tutorial/recurrent_neural_networks/gated_recurrent_unit(GRU).files/20170804_142332/net.nntxt.
N00000006 = r'''global_config {
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
  name: "Main_"
  batch_size: 64
  repeat_info {
    id: "RecurrentInput"
    times: 28
  }
  variable {
    name: "X"
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
    name: "ResetGate/affine/W"
    type: "Parameter"
    shape: { dim: 56 dim: 1 dim: 28 }
    initializer {
      type: "NormalAffineGlorot"
      multiplier: 1.0
    }
  }
  variable {
    name: "ResetGate/affine/b"
    type: "Parameter"
    shape: { dim: 1 dim: 28 }
    initializer {
      type: "Constant"
      multiplier: 0.0
    }
  }
  variable {
    name: "Affine/affine/W"
    type: "Parameter"
    shape: { dim: 56 dim: 1 dim: 28 }
    initializer {
      type: "NormalAffineGlorot"
      multiplier: 1.0
    }
  }
  variable {
    name: "Affine/affine/b"
    type: "Parameter"
    shape: { dim: 1 dim: 28 }
    initializer {
      type: "Constant"
      multiplier: 0.0
    }
  }
  variable {
    name: "UpdateGate/affine/W"
    type: "Parameter"
    shape: { dim: 56 dim: 1 dim: 28 }
    initializer {
      type: "NormalAffineGlorot"
      multiplier: 1.0
    }
  }
  variable {
    name: "UpdateGate/affine/b"
    type: "Parameter"
    shape: { dim: 1 dim: 28 }
    initializer {
      type: "Constant"
      multiplier: 0.0
    }
  }
  variable {
    name: "Output_T"
    type: "Buffer"
    shape: { dim:-1 dim: 1 }
  }
  variable {
    name: "Delay_Initial"
    type: "Buffer"
    shape: { dim:-1 dim: 1 dim: 28 }
  }
  variable {
    name: "Affine_2/affine/W"
    type: "Parameter"
    shape: { dim: 28 dim: 1 }
    initializer {
      type: "NormalAffineGlorot"
      multiplier: 1.0
    }
  }
  variable {
    name: "Affine_2/affine/b"
    type: "Parameter"
    shape: { dim: 1 }
    initializer {
      type: "Constant"
      multiplier: 0.0
    }
  }
  variable {
    name: "Concatenate"
    type: "Buffer"
    repeat_id: "RecurrentInput"
    shape: { dim:-1 dim: 2 dim: 28 }
  }
  variable {
    name: "ResetGate"
    type: "Buffer"
    repeat_id: "RecurrentInput"
    shape: { dim:-1 dim: 1 dim: 28 }
  }
  variable {
    name: "Sigmoid"
    type: "Buffer"
    repeat_id: "RecurrentInput"
    shape: { dim:-1 dim: 1 dim: 28 }
  }
  variable {
    name: "Product"
    type: "Buffer"
    repeat_id: "RecurrentInput"
    shape: { dim:-1 dim: 1 dim: 28 }
  }
  variable {
    name: "Concatenate_2"
    type: "Buffer"
    repeat_id: "RecurrentInput"
    shape: { dim:-1 dim: 2 dim: 28 }
  }
  variable {
    name: "Affine"
    type: "Buffer"
    repeat_id: "RecurrentInput"
    shape: { dim:-1 dim: 1 dim: 28 }
  }
  variable {
    name: "Tanh"
    type: "Buffer"
    repeat_id: "RecurrentInput"
    shape: { dim:-1 dim: 1 dim: 28 }
  }
  variable {
    name: "UpdateGate"
    type: "Buffer"
    repeat_id: "RecurrentInput"
    shape: { dim:-1 dim: 1 dim: 28 }
  }
  variable {
    name: "Sigmoid_2"
    type: "Buffer"
    repeat_id: "RecurrentInput"
    shape: { dim:-1 dim: 1 dim: 28 }
  }
  variable {
    name: "Product_2"
    type: "Buffer"
    repeat_id: "RecurrentInput"
    shape: { dim:-1 dim: 1 dim: 28 }
  }
  variable {
    name: "Sum"
    type: "Buffer"
    repeat_id: "RecurrentInput"
    shape: { dim:-1 dim: 1 dim: 28 }
  }
  variable {
    name: "Product_3"
    type: "Buffer"
    repeat_id: "RecurrentInput"
    shape: { dim:-1 dim: 1 dim: 28 }
  }
  variable {
    name: "Not"
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
    name: "Output"
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
    shape: { dim:-1 dim: 1 }
  }
  variable {
    name: "Sigmoid_3"
    type: "Buffer"
    shape: { dim:-1 dim: 1 }
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
    input: "X"
    output: "RecurrentInput"
    recurrent_param {
      axis: 2
      repeat_id: "RecurrentInput"
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
    name: "ResetGate"
    type: "Affine"
    repeat_id: "RecurrentInput"
    input: "Concatenate"
    input: "ResetGate/affine/W"
    input: "ResetGate/affine/b"
    output: "ResetGate"
    affine_param {
      base_axis: 1
    }
  }
  function {
    name: "Sigmoid"
    type: "Sigmoid"
    repeat_id: "RecurrentInput"
    input: "ResetGate"
    output: "Sigmoid"
  }
  function {
    name: "Product"
    type: "Mul2"
    repeat_id: "RecurrentInput"
    input: "Sigmoid"
    input: "Delay"
    output: "Product"
  }
  function {
    name: "Concatenate_2"
    type: "Concatenate"
    repeat_id: "RecurrentInput"
    input: "Product"
    input: "RecurrentInput"
    output: "Concatenate_2"
    concatenate_param {
      axis: 1
    }
  }
  function {
    name: "Affine"
    type: "Affine"
    repeat_id: "RecurrentInput"
    input: "Concatenate_2"
    input: "Affine/affine/W"
    input: "Affine/affine/b"
    output: "Affine"
    affine_param {
      base_axis: 1
    }
  }
  function {
    name: "Tanh"
    type: "Tanh"
    repeat_id: "RecurrentInput"
    input: "Affine"
    output: "Tanh"
  }
  function {
    name: "UpdateGate"
    type: "Affine"
    repeat_id: "RecurrentInput"
    input: "Concatenate"
    input: "UpdateGate/affine/W"
    input: "UpdateGate/affine/b"
    output: "UpdateGate"
    affine_param {
      base_axis: 1
    }
  }
  function {
    name: "Sigmoid_2"
    type: "Sigmoid"
    repeat_id: "RecurrentInput"
    input: "UpdateGate"
    output: "Sigmoid_2"
  }
  function {
    name: "Product_2"
    type: "Mul2"
    repeat_id: "RecurrentInput"
    input: "Not"
    input: "Delay"
    output: "Product_2"
  }
  function {
    name: "Sum"
    type: "Add2"
    repeat_id: "RecurrentInput"
    input: "Product_2"
    input: "Product_3"
    output: "Sum"
    add2_param {
      inplace: True
    }
  }
  function {
    name: "Product_3"
    type: "Mul2"
    repeat_id: "RecurrentInput"
    input: "Tanh"
    input: "Sigmoid_2"
    output: "Product_3"
  }
  function {
    name: "Not"
    type: "RSubScalar"
    repeat_id: "RecurrentInput"
    input: "Sigmoid_2"
    output: "Not"
    r_sub_scalar_param {
      val: 1.0
    }
  }
  function {
    name: "RecurrentOutput"
    type: "RecurrentOutput"
    input: "Sum"
    output: "RecurrentOutput"
    recurrent_param {
      axis: 2
      repeat_id: "RecurrentInput"
      length: 28
    }
  }
  function {
    name: "Output"
    type: "BinaryCrossEntropy"
    input: "Sigmoid_3"
    input: "Output_T"
    output: "Output"
  }
  function {
    name: "Delay"
    type: "Delay"
    repeat_id: "RecurrentInput"
    input: "Sum"
    input: "Delay_Initial"
    output: "Delay"
    recurrent_param {
      repeat_id: "RecurrentInput"
    }
  }
  function {
    name: "Affine_2"
    type: "Affine"
    input: "Slice"
    input: "Affine_2/affine/W"
    input: "Affine_2/affine/b"
    output: "Affine_2"
    affine_param {
      base_axis: 1
    }
  }
  function {
    name: "Sigmoid_3"
    type: "Sigmoid"
    input: "Affine_2"
    output: "Sigmoid_3"
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
  name: "Runtime"
  batch_size: 64
  repeat_info {
    id: "RecurrentInput"
    times: 28
  }
  variable {
    name: "X"
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
    name: "ResetGate/affine/W"
    type: "Parameter"
    shape: { dim: 56 dim: 1 dim: 28 }
    initializer {
      type: "NormalAffineGlorot"
      multiplier: 1.0
    }
  }
  variable {
    name: "ResetGate/affine/b"
    type: "Parameter"
    shape: { dim: 1 dim: 28 }
    initializer {
      type: "Constant"
      multiplier: 0.0
    }
  }
  variable {
    name: "Affine/affine/W"
    type: "Parameter"
    shape: { dim: 56 dim: 1 dim: 28 }
    initializer {
      type: "NormalAffineGlorot"
      multiplier: 1.0
    }
  }
  variable {
    name: "Affine/affine/b"
    type: "Parameter"
    shape: { dim: 1 dim: 28 }
    initializer {
      type: "Constant"
      multiplier: 0.0
    }
  }
  variable {
    name: "UpdateGate/affine/W"
    type: "Parameter"
    shape: { dim: 56 dim: 1 dim: 28 }
    initializer {
      type: "NormalAffineGlorot"
      multiplier: 1.0
    }
  }
  variable {
    name: "UpdateGate/affine/b"
    type: "Parameter"
    shape: { dim: 1 dim: 28 }
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
    shape: { dim: 28 dim: 1 }
    initializer {
      type: "NormalAffineGlorot"
      multiplier: 1.0
    }
  }
  variable {
    name: "Affine_2/affine/b"
    type: "Parameter"
    shape: { dim: 1 }
    initializer {
      type: "Constant"
      multiplier: 0.0
    }
  }
  variable {
    name: "Concatenate"
    type: "Buffer"
    repeat_id: "RecurrentInput"
    shape: { dim:-1 dim: 2 dim: 28 }
  }
  variable {
    name: "ResetGate"
    type: "Buffer"
    repeat_id: "RecurrentInput"
    shape: { dim:-1 dim: 1 dim: 28 }
  }
  variable {
    name: "Sigmoid"
    type: "Buffer"
    repeat_id: "RecurrentInput"
    shape: { dim:-1 dim: 1 dim: 28 }
  }
  variable {
    name: "Product"
    type: "Buffer"
    repeat_id: "RecurrentInput"
    shape: { dim:-1 dim: 1 dim: 28 }
  }
  variable {
    name: "Concatenate_2"
    type: "Buffer"
    repeat_id: "RecurrentInput"
    shape: { dim:-1 dim: 2 dim: 28 }
  }
  variable {
    name: "Affine"
    type: "Buffer"
    repeat_id: "RecurrentInput"
    shape: { dim:-1 dim: 1 dim: 28 }
  }
  variable {
    name: "Tanh"
    type: "Buffer"
    repeat_id: "RecurrentInput"
    shape: { dim:-1 dim: 1 dim: 28 }
  }
  variable {
    name: "UpdateGate"
    type: "Buffer"
    repeat_id: "RecurrentInput"
    shape: { dim:-1 dim: 1 dim: 28 }
  }
  variable {
    name: "Sigmoid_2"
    type: "Buffer"
    repeat_id: "RecurrentInput"
    shape: { dim:-1 dim: 1 dim: 28 }
  }
  variable {
    name: "Product_2"
    type: "Buffer"
    repeat_id: "RecurrentInput"
    shape: { dim:-1 dim: 1 dim: 28 }
  }
  variable {
    name: "Sum"
    type: "Buffer"
    repeat_id: "RecurrentInput"
    shape: { dim:-1 dim: 1 dim: 28 }
  }
  variable {
    name: "Product_3"
    type: "Buffer"
    repeat_id: "RecurrentInput"
    shape: { dim:-1 dim: 1 dim: 28 }
  }
  variable {
    name: "Not"
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
    name: "Delay"
    type: "Buffer"
    repeat_id: "RecurrentInput"
    shape: { dim:-1 dim: 1 dim: 28 }
  }
  variable {
    name: "Affine_2"
    type: "Buffer"
    shape: { dim:-1 dim: 1 }
  }
  variable {
    name: "y'"
    type: "Buffer"
    shape: { dim:-1 dim: 1 }
  }
  variable {
    name: "Slice"
    type: "Buffer"
    shape: { dim:-1 dim: 1 dim: 1 dim: 28 }
  }
  variable {
    name: "rnn_out"
    type: "Buffer"
    shape: { dim:-1 dim: 1 dim: 28 dim: 28 }
  }
  function {
    name: "RecurrentInput"
    type: "RecurrentInput"
    repeat_id: "RecurrentInput"
    input: "X"
    output: "RecurrentInput"
    recurrent_param {
      axis: 2
      repeat_id: "RecurrentInput"
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
    name: "ResetGate"
    type: "Affine"
    repeat_id: "RecurrentInput"
    input: "Concatenate"
    input: "ResetGate/affine/W"
    input: "ResetGate/affine/b"
    output: "ResetGate"
    affine_param {
      base_axis: 1
    }
  }
  function {
    name: "Sigmoid"
    type: "Sigmoid"
    repeat_id: "RecurrentInput"
    input: "ResetGate"
    output: "Sigmoid"
  }
  function {
    name: "Product"
    type: "Mul2"
    repeat_id: "RecurrentInput"
    input: "Sigmoid"
    input: "Delay"
    output: "Product"
  }
  function {
    name: "Concatenate_2"
    type: "Concatenate"
    repeat_id: "RecurrentInput"
    input: "Product"
    input: "RecurrentInput"
    output: "Concatenate_2"
    concatenate_param {
      axis: 1
    }
  }
  function {
    name: "Affine"
    type: "Affine"
    repeat_id: "RecurrentInput"
    input: "Concatenate_2"
    input: "Affine/affine/W"
    input: "Affine/affine/b"
    output: "Affine"
    affine_param {
      base_axis: 1
    }
  }
  function {
    name: "Tanh"
    type: "Tanh"
    repeat_id: "RecurrentInput"
    input: "Affine"
    output: "Tanh"
  }
  function {
    name: "UpdateGate"
    type: "Affine"
    repeat_id: "RecurrentInput"
    input: "Concatenate"
    input: "UpdateGate/affine/W"
    input: "UpdateGate/affine/b"
    output: "UpdateGate"
    affine_param {
      base_axis: 1
    }
  }
  function {
    name: "Sigmoid_2"
    type: "Sigmoid"
    repeat_id: "RecurrentInput"
    input: "UpdateGate"
    output: "Sigmoid_2"
  }
  function {
    name: "Product_2"
    type: "Mul2"
    repeat_id: "RecurrentInput"
    input: "Not"
    input: "Delay"
    output: "Product_2"
  }
  function {
    name: "Sum"
    type: "Add2"
    repeat_id: "RecurrentInput"
    input: "Product_2"
    input: "Product_3"
    output: "Sum"
    add2_param {
      inplace: True
    }
  }
  function {
    name: "Product_3"
    type: "Mul2"
    repeat_id: "RecurrentInput"
    input: "Tanh"
    input: "Sigmoid_2"
    output: "Product_3"
  }
  function {
    name: "Not"
    type: "RSubScalar"
    repeat_id: "RecurrentInput"
    input: "Sigmoid_2"
    output: "Not"
    r_sub_scalar_param {
      val: 1.0
    }
  }
  function {
    name: "RecurrentOutput"
    type: "RecurrentOutput"
    input: "Sum"
    output: "RecurrentOutput"
    recurrent_param {
      axis: 2
      repeat_id: "RecurrentInput"
      length: 28
    }
  }
  function {
    name: "Delay"
    type: "Delay"
    repeat_id: "RecurrentInput"
    input: "Sum"
    input: "Delay_Initial"
    output: "Delay"
    recurrent_param {
      repeat_id: "RecurrentInput"
    }
  }
  function {
    name: "Affine_2"
    type: "Affine"
    input: "Slice"
    input: "Affine_2/affine/W"
    input: "Affine_2/affine/b"
    output: "Affine_2"
    affine_param {
      base_axis: 1
    }
  }
  function {
    name: "y'"
    type: "Sigmoid"
    input: "Affine_2"
    output: "y'"
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
  function {
    name: "rnn_out"
    type: "Identity"
    input: "RecurrentOutput"
    output: "rnn_out"
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
  network_name: "Main_"
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
    variable_name: "X"
    data_name: "x"
  }
  data_variable {
    variable_name: "Output_T"
    data_name: "y"
  }
  generator_variable {
    variable_name: "Delay_Initial"
    type: "Constant"
    multiplier: 0.0
  }
  loss_variable {
    variable_name: "Output"
  }
  parameter_variable {
    variable_name: "ResetGate/affine/W"
    learning_rate_multiplier: 1.0
  }
  parameter_variable {
    variable_name: "ResetGate/affine/b"
    learning_rate_multiplier: 1.0
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
    variable_name: "UpdateGate/affine/W"
    learning_rate_multiplier: 1.0
  }
  parameter_variable {
    variable_name: "UpdateGate/affine/b"
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
  network_name: "Main_"
  dataset_name: "Training"
  data_variable {
    variable_name: "X"
    data_name: "x"
  }
  data_variable {
    variable_name: "Output_T"
    data_name: "y"
  }
  generator_variable {
    variable_name: "Delay_Initial"
    type: "Constant"
    multiplier: 0.0
  }
  monitor_variable {
    type: "Error"
    variable_name: "Output"
  }
}
monitor {
  name: "valid_error"
  network_name: "Main_"
  dataset_name: "Validation"
  data_variable {
    variable_name: "X"
    data_name: "x"
  }
  data_variable {
    variable_name: "Output_T"
    data_name: "y"
  }
  generator_variable {
    variable_name: "Delay_Initial"
    type: "Constant"
    multiplier: 0.0
  }
  monitor_variable {
    type: "Error"
    variable_name: "Output"
  }
}
executor {
  name: "Executor"
  network_name: "Runtime"
  num_evaluations: 1
  repeat_evaluation_type: "mean"
  need_back_propagation: false
  data_variable {
    variable_name: "X"
    data_name: "x"
  }
  generator_variable {
    variable_name: "Delay_Initial"
    type: "Constant"
    multiplier: 0.0
  }
  output_variable {
    variable_name: "y'"
    data_name: "y'"
  }
  output_variable {
    variable_name: "rnn_out"
    data_name: "rnn_out"
  }
  parameter_variable {
    variable_name: "ResetGate/affine/W"
  }
  parameter_variable {
    variable_name: "ResetGate/affine/b"
  }
  parameter_variable {
    variable_name: "Affine/affine/W"
  }
  parameter_variable {
    variable_name: "Affine/affine/b"
  }
  parameter_variable {
    variable_name: "UpdateGate/affine/W"
  }
  parameter_variable {
    variable_name: "UpdateGate/affine/b"
  }
  parameter_variable {
    variable_name: "Affine_2/affine/W"
  }
  parameter_variable {
    variable_name: "Affine_2/affine/b"
  }
}
'''
# The nntxt is generated from /home/woody/aid/nnabla-sample-data/sample_project/tutorial/recurrent_neural_networks/gated_recurrent_unit(GRU).files/20180720_191314/net.nntxt.
N00000007 = r'''global_config {
  default_context {
    array_class: "CudaCachedArray"
    device_id: "0"
    backends: "cudnn:float"
    backends: "cuda:float"
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
  name: "Main_"
  batch_size: 64
  repeat_info {
    id: "RecurrentInput"
    times: 28
  }
  variable {
    name: "X"
    type: "Buffer"
    shape: { dim:-1 dim: 1 dim: 28 dim: 28 }
  }
  variable {
    name: "Delay_Initial"
    type: "Buffer"
    shape: { dim:-1 dim: 1 dim: 28 }
  }
  variable {
    name: "ResetGate/affine/W"
    type: "Parameter"
    shape: { dim: 56 dim: 1 dim: 28 }
    initializer {
      type: "NormalAffineGlorot"
      multiplier: 1
    }
  }
  variable {
    name: "ResetGate/affine/b"
    type: "Parameter"
    shape: { dim: 1 dim: 28 }
    initializer {
      type: "Constant"
      multiplier: 0
    }
  }
  variable {
    name: "UpdateGate/affine/W"
    type: "Parameter"
    shape: { dim: 56 dim: 1 dim: 28 }
    initializer {
      type: "NormalAffineGlorot"
      multiplier: 1
    }
  }
  variable {
    name: "UpdateGate/affine/b"
    type: "Parameter"
    shape: { dim: 1 dim: 28 }
    initializer {
      type: "Constant"
      multiplier: 0
    }
  }
  variable {
    name: "Affine/affine/W"
    type: "Parameter"
    shape: { dim: 56 dim: 1 dim: 28 }
    initializer {
      type: "NormalAffineGlorot"
      multiplier: 1
    }
  }
  variable {
    name: "Affine/affine/b"
    type: "Parameter"
    shape: { dim: 1 dim: 28 }
    initializer {
      type: "Constant"
      multiplier: 0
    }
  }
  variable {
    name: "Affine_2/affine/W"
    type: "Parameter"
    shape: { dim: 28 dim: 1 }
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
    name: "Output_T"
    type: "Buffer"
    shape: { dim:-1 dim: 1 }
  }
  variable {
    name: "RecurrentInput"
    type: "Buffer"
    repeat_id: "RecurrentInput"
    shape: { dim:-1 dim: 1 dim: 28 }
  }
  variable {
    name: "Delay"
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
    name: "ResetGate"
    type: "Buffer"
    repeat_id: "RecurrentInput"
    shape: { dim:-1 dim: 1 dim: 28 }
  }
  variable {
    name: "UpdateGate"
    type: "Buffer"
    repeat_id: "RecurrentInput"
    shape: { dim:-1 dim: 1 dim: 28 }
  }
  variable {
    name: "Sigmoid"
    type: "Buffer"
    repeat_id: "RecurrentInput"
    shape: { dim:-1 dim: 1 dim: 28 }
  }
  variable {
    name: "Sigmoid_2"
    type: "Buffer"
    repeat_id: "RecurrentInput"
    shape: { dim:-1 dim: 1 dim: 28 }
  }
  variable {
    name: "Product"
    type: "Buffer"
    repeat_id: "RecurrentInput"
    shape: { dim:-1 dim: 1 dim: 28 }
  }
  variable {
    name: "Not"
    type: "Buffer"
    repeat_id: "RecurrentInput"
    shape: { dim:-1 dim: 1 dim: 28 }
  }
  variable {
    name: "Concatenate_2"
    type: "Buffer"
    repeat_id: "RecurrentInput"
    shape: { dim:-1 dim: 2 dim: 28 }
  }
  variable {
    name: "Product_2"
    type: "Buffer"
    repeat_id: "RecurrentInput"
    shape: { dim:-1 dim: 1 dim: 28 }
  }
  variable {
    name: "Affine"
    type: "Buffer"
    repeat_id: "RecurrentInput"
    shape: { dim:-1 dim: 1 dim: 28 }
  }
  variable {
    name: "Tanh"
    type: "Buffer"
    repeat_id: "RecurrentInput"
    shape: { dim:-1 dim: 1 dim: 28 }
  }
  variable {
    name: "Product_3"
    type: "Buffer"
    repeat_id: "RecurrentInput"
    shape: { dim:-1 dim: 1 dim: 28 }
  }
  variable {
    name: "Sum"
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
  variable {
    name: "Affine_2"
    type: "Buffer"
    shape: { dim:-1 dim: 1 }
  }
  variable {
    name: "Sigmoid_3"
    type: "Buffer"
    shape: { dim:-1 dim: 1 }
  }
  variable {
    name: "Output"
    type: "Buffer"
    shape: { dim:-1 dim: 1 }
  }
  function {
    name: "RecurrentInput"
    type: "RecurrentInput"
    repeat_id: "RecurrentInput"
    input: "X"
    output: "RecurrentInput"
    recurrent_param {
      axis: 2
      repeat_id: "RecurrentInput"
    }
  }
  function {
    name: "Delay"
    type: "Delay"
    repeat_id: "RecurrentInput"
    input: "Sum"
    input: "Delay_Initial"
    output: "Delay"
    recurrent_param {
      repeat_id: "RecurrentInput"
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
    name: "ResetGate"
    type: "Affine"
    repeat_id: "RecurrentInput"
    input: "Concatenate"
    input: "ResetGate/affine/W"
    input: "ResetGate/affine/b"
    output: "ResetGate"
    affine_param {
      base_axis: 1
    }
  }
  function {
    name: "UpdateGate"
    type: "Affine"
    repeat_id: "RecurrentInput"
    input: "Concatenate"
    input: "UpdateGate/affine/W"
    input: "UpdateGate/affine/b"
    output: "UpdateGate"
    affine_param {
      base_axis: 1
    }
  }
  function {
    name: "Sigmoid"
    type: "Sigmoid"
    repeat_id: "RecurrentInput"
    input: "ResetGate"
    output: "Sigmoid"
  }
  function {
    name: "Sigmoid_2"
    type: "Sigmoid"
    repeat_id: "RecurrentInput"
    input: "UpdateGate"
    output: "Sigmoid_2"
  }
  function {
    name: "Product"
    type: "Mul2"
    repeat_id: "RecurrentInput"
    input: "Sigmoid"
    input: "Delay"
    output: "Product"
  }
  function {
    name: "Not"
    type: "RSubScalar"
    repeat_id: "RecurrentInput"
    input: "Sigmoid_2"
    output: "Not"
    r_sub_scalar_param {
      val: 1
    }
  }
  function {
    name: "Concatenate_2"
    type: "Concatenate"
    repeat_id: "RecurrentInput"
    input: "Product"
    input: "RecurrentInput"
    output: "Concatenate_2"
    concatenate_param {
      axis: 1
    }
  }
  function {
    name: "Product_2"
    type: "Mul2"
    repeat_id: "RecurrentInput"
    input: "Not"
    input: "Delay"
    output: "Product_2"
  }
  function {
    name: "Affine"
    type: "Affine"
    repeat_id: "RecurrentInput"
    input: "Concatenate_2"
    input: "Affine/affine/W"
    input: "Affine/affine/b"
    output: "Affine"
    affine_param {
      base_axis: 1
    }
  }
  function {
    name: "Tanh"
    type: "Tanh"
    repeat_id: "RecurrentInput"
    input: "Affine"
    output: "Tanh"
  }
  function {
    name: "Product_3"
    type: "Mul2"
    repeat_id: "RecurrentInput"
    input: "Tanh"
    input: "Sigmoid_2"
    output: "Product_3"
  }
  function {
    name: "Sum"
    type: "Add2"
    repeat_id: "RecurrentInput"
    input: "Product_2"
    input: "Product_3"
    output: "Sum"
    add2_param {
      inplace: True
    }
  }
  function {
    name: "RecurrentOutput"
    type: "RecurrentOutput"
    input: "Sum"
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
  function {
    name: "Affine_2"
    type: "Affine"
    input: "Slice"
    input: "Affine_2/affine/W"
    input: "Affine_2/affine/b"
    output: "Affine_2"
    affine_param {
      base_axis: 1
    }
  }
  function {
    name: "Sigmoid_3"
    type: "Sigmoid"
    input: "Affine_2"
    output: "Sigmoid_3"
  }
  function {
    name: "Output"
    type: "BinaryCrossEntropy"
    input: "Sigmoid_3"
    input: "Output_T"
    output: "Output"
  }
}
network {
  name: "Runtime"
  batch_size: 64
  repeat_info {
    id: "RecurrentInput"
    times: 28
  }
  variable {
    name: "X"
    type: "Buffer"
    shape: { dim:-1 dim: 1 dim: 28 dim: 28 }
  }
  variable {
    name: "Delay_Initial"
    type: "Buffer"
    shape: { dim:-1 dim: 1 dim: 28 }
  }
  variable {
    name: "ResetGate/affine/W"
    type: "Parameter"
    shape: { dim: 56 dim: 1 dim: 28 }
    initializer {
      type: "NormalAffineGlorot"
      multiplier: 1
    }
  }
  variable {
    name: "ResetGate/affine/b"
    type: "Parameter"
    shape: { dim: 1 dim: 28 }
    initializer {
      type: "Constant"
      multiplier: 0
    }
  }
  variable {
    name: "UpdateGate/affine/W"
    type: "Parameter"
    shape: { dim: 56 dim: 1 dim: 28 }
    initializer {
      type: "NormalAffineGlorot"
      multiplier: 1
    }
  }
  variable {
    name: "UpdateGate/affine/b"
    type: "Parameter"
    shape: { dim: 1 dim: 28 }
    initializer {
      type: "Constant"
      multiplier: 0
    }
  }
  variable {
    name: "Affine/affine/W"
    type: "Parameter"
    shape: { dim: 56 dim: 1 dim: 28 }
    initializer {
      type: "NormalAffineGlorot"
      multiplier: 1
    }
  }
  variable {
    name: "Affine/affine/b"
    type: "Parameter"
    shape: { dim: 1 dim: 28 }
    initializer {
      type: "Constant"
      multiplier: 0
    }
  }
  variable {
    name: "Affine_2/affine/W"
    type: "Parameter"
    shape: { dim: 28 dim: 1 }
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
    name: "RecurrentInput"
    type: "Buffer"
    repeat_id: "RecurrentInput"
    shape: { dim:-1 dim: 1 dim: 28 }
  }
  variable {
    name: "Delay"
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
    name: "ResetGate"
    type: "Buffer"
    repeat_id: "RecurrentInput"
    shape: { dim:-1 dim: 1 dim: 28 }
  }
  variable {
    name: "UpdateGate"
    type: "Buffer"
    repeat_id: "RecurrentInput"
    shape: { dim:-1 dim: 1 dim: 28 }
  }
  variable {
    name: "Sigmoid"
    type: "Buffer"
    repeat_id: "RecurrentInput"
    shape: { dim:-1 dim: 1 dim: 28 }
  }
  variable {
    name: "Sigmoid_2"
    type: "Buffer"
    repeat_id: "RecurrentInput"
    shape: { dim:-1 dim: 1 dim: 28 }
  }
  variable {
    name: "Product"
    type: "Buffer"
    repeat_id: "RecurrentInput"
    shape: { dim:-1 dim: 1 dim: 28 }
  }
  variable {
    name: "Not"
    type: "Buffer"
    repeat_id: "RecurrentInput"
    shape: { dim:-1 dim: 1 dim: 28 }
  }
  variable {
    name: "Concatenate_2"
    type: "Buffer"
    repeat_id: "RecurrentInput"
    shape: { dim:-1 dim: 2 dim: 28 }
  }
  variable {
    name: "Product_2"
    type: "Buffer"
    repeat_id: "RecurrentInput"
    shape: { dim:-1 dim: 1 dim: 28 }
  }
  variable {
    name: "Affine"
    type: "Buffer"
    repeat_id: "RecurrentInput"
    shape: { dim:-1 dim: 1 dim: 28 }
  }
  variable {
    name: "Tanh"
    type: "Buffer"
    repeat_id: "RecurrentInput"
    shape: { dim:-1 dim: 1 dim: 28 }
  }
  variable {
    name: "Product_3"
    type: "Buffer"
    repeat_id: "RecurrentInput"
    shape: { dim:-1 dim: 1 dim: 28 }
  }
  variable {
    name: "Sum"
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
  variable {
    name: "rnn_out"
    type: "Buffer"
    shape: { dim:-1 dim: 1 dim: 28 dim: 28 }
  }
  variable {
    name: "Affine_2"
    type: "Buffer"
    shape: { dim:-1 dim: 1 }
  }
  variable {
    name: "y'"
    type: "Buffer"
    shape: { dim:-1 dim: 1 }
  }
  function {
    name: "RecurrentInput"
    type: "RecurrentInput"
    repeat_id: "RecurrentInput"
    input: "X"
    output: "RecurrentInput"
    recurrent_param {
      axis: 2
      repeat_id: "RecurrentInput"
    }
  }
  function {
    name: "Delay"
    type: "Delay"
    repeat_id: "RecurrentInput"
    input: "Sum"
    input: "Delay_Initial"
    output: "Delay"
    recurrent_param {
      repeat_id: "RecurrentInput"
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
    name: "ResetGate"
    type: "Affine"
    repeat_id: "RecurrentInput"
    input: "Concatenate"
    input: "ResetGate/affine/W"
    input: "ResetGate/affine/b"
    output: "ResetGate"
    affine_param {
      base_axis: 1
    }
  }
  function {
    name: "UpdateGate"
    type: "Affine"
    repeat_id: "RecurrentInput"
    input: "Concatenate"
    input: "UpdateGate/affine/W"
    input: "UpdateGate/affine/b"
    output: "UpdateGate"
    affine_param {
      base_axis: 1
    }
  }
  function {
    name: "Sigmoid"
    type: "Sigmoid"
    repeat_id: "RecurrentInput"
    input: "ResetGate"
    output: "Sigmoid"
  }
  function {
    name: "Sigmoid_2"
    type: "Sigmoid"
    repeat_id: "RecurrentInput"
    input: "UpdateGate"
    output: "Sigmoid_2"
  }
  function {
    name: "Product"
    type: "Mul2"
    repeat_id: "RecurrentInput"
    input: "Sigmoid"
    input: "Delay"
    output: "Product"
  }
  function {
    name: "Not"
    type: "RSubScalar"
    repeat_id: "RecurrentInput"
    input: "Sigmoid_2"
    output: "Not"
    r_sub_scalar_param {
      val: 1
    }
  }
  function {
    name: "Concatenate_2"
    type: "Concatenate"
    repeat_id: "RecurrentInput"
    input: "Product"
    input: "RecurrentInput"
    output: "Concatenate_2"
    concatenate_param {
      axis: 1
    }
  }
  function {
    name: "Product_2"
    type: "Mul2"
    repeat_id: "RecurrentInput"
    input: "Not"
    input: "Delay"
    output: "Product_2"
  }
  function {
    name: "Affine"
    type: "Affine"
    repeat_id: "RecurrentInput"
    input: "Concatenate_2"
    input: "Affine/affine/W"
    input: "Affine/affine/b"
    output: "Affine"
    affine_param {
      base_axis: 1
    }
  }
  function {
    name: "Tanh"
    type: "Tanh"
    repeat_id: "RecurrentInput"
    input: "Affine"
    output: "Tanh"
  }
  function {
    name: "Product_3"
    type: "Mul2"
    repeat_id: "RecurrentInput"
    input: "Tanh"
    input: "Sigmoid_2"
    output: "Product_3"
  }
  function {
    name: "Sum"
    type: "Add2"
    repeat_id: "RecurrentInput"
    input: "Product_2"
    input: "Product_3"
    output: "Sum"
    add2_param {
      inplace: True
    }
  }
  function {
    name: "RecurrentOutput"
    type: "RecurrentOutput"
    input: "Sum"
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
  function {
    name: "rnn_out"
    type: "Identity"
    input: "RecurrentOutput"
    output: "rnn_out"
  }
  function {
    name: "Affine_2"
    type: "Affine"
    input: "Slice"
    input: "Affine_2/affine/W"
    input: "Affine_2/affine/b"
    output: "Affine_2"
    affine_param {
      base_axis: 1
    }
  }
  function {
    name: "y'"
    type: "Sigmoid"
    input: "Affine_2"
    output: "y'"
  }
}
dataset {
  name: "Training"
  uri: "C:\\Utils\\neural_network_console_internal_v120\\samples\\sample_dataset\\mnist\\small_mnist_4or9_training.csv"
  cache_dir: "C:\\Utils\\neural_network_console_internal_v120\\samples\\sample_dataset\\mnist\\small_mnist_4or9_training.cache"
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
  network_name: "Main_"
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
    variable_name: "X"
    data_name: "x"
  }
  generator_variable {
    variable_name: "Delay_Initial"
    type: "Constant"
    multiplier: 0
  }
  data_variable {
    variable_name: "Output_T"
    data_name: "y"
  }
  loss_variable {
    variable_name: "Output"
  }
  parameter_variable {
    variable_name: "ResetGate/affine/W"
    learning_rate_multiplier: 1
  }
  parameter_variable {
    variable_name: "ResetGate/affine/b"
    learning_rate_multiplier: 1
  }
  parameter_variable {
    variable_name: "UpdateGate/affine/W"
    learning_rate_multiplier: 1
  }
  parameter_variable {
    variable_name: "UpdateGate/affine/b"
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
  network_name: "Main_"
  dataset_name: "Training"
  data_variable {
    variable_name: "X"
    data_name: "x"
  }
  generator_variable {
    variable_name: "Delay_Initial"
    type: "Constant"
    multiplier: 0
  }
  data_variable {
    variable_name: "Output_T"
    data_name: "y"
  }
  monitor_variable {
    type: "Error"
    variable_name: "Output"
  }
}
monitor {
  name: "valid_error"
  network_name: "Main_"
  dataset_name: "Validation"
  data_variable {
    variable_name: "X"
    data_name: "x"
  }
  generator_variable {
    variable_name: "Delay_Initial"
    type: "Constant"
    multiplier: 0
  }
  data_variable {
    variable_name: "Output_T"
    data_name: "y"
  }
  monitor_variable {
    type: "Error"
    variable_name: "Output"
  }
}
executor {
  name: "Executor"
  network_name: "Runtime"
  num_evaluations: 1
  repeat_evaluation_type: "mean"
  need_back_propagation: false
  data_variable {
    variable_name: "X"
    data_name: "x"
  }
  generator_variable {
    variable_name: "Delay_Initial"
    type: "Constant"
    multiplier: 0
  }
  output_variable {
    variable_name: "rnn_out"
    data_name: "rnn_out"
  }
  output_variable {
    variable_name: "y'"
    data_name: "y'"
  }
  parameter_variable {
    variable_name: "ResetGate/affine/W"
  }
  parameter_variable {
    variable_name: "ResetGate/affine/b"
  }
  parameter_variable {
    variable_name: "UpdateGate/affine/W"
  }
  parameter_variable {
    variable_name: "UpdateGate/affine/b"
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
'''
# The nntxt is generated from /home/woody/aid/nnabla-sample-data/sample_project/tutorial/recurrent_neural_networks/stacked_GRU.files/20170804_143636/net.nntxt.
N00000008 = r'''global_config {
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
  name: "Main_"
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
    name: "Affine_4/affine/W"
    type: "Parameter"
    shape: { dim: 56 dim: 1 dim: 28 }
    initializer {
      type: "NormalAffineGlorot"
      multiplier: 1.0
    }
  }
  variable {
    name: "Affine_4/affine/b"
    type: "Parameter"
    shape: { dim: 1 dim: 28 }
    initializer {
      type: "Constant"
      multiplier: 0.0
    }
  }
  variable {
    name: "Affine/affine/W"
    type: "Parameter"
    shape: { dim: 56 dim: 1 dim: 28 }
    initializer {
      type: "NormalAffineGlorot"
      multiplier: 1.0
    }
  }
  variable {
    name: "Affine/affine/b"
    type: "Parameter"
    shape: { dim: 1 dim: 28 }
    initializer {
      type: "Constant"
      multiplier: 0.0
    }
  }
  variable {
    name: "Affine_5/affine/W"
    type: "Parameter"
    shape: { dim: 56 dim: 1 dim: 28 }
    initializer {
      type: "NormalAffineGlorot"
      multiplier: 1.0
    }
  }
  variable {
    name: "Affine_5/affine/b"
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
    name: "Delay_Initial"
    type: "Buffer"
    shape: { dim:-1 dim: 1 dim: 28 }
  }
  variable {
    name: "Affine_2/affine/W"
    type: "Parameter"
    shape: { dim: 28 dim: 1 }
    initializer {
      type: "NormalAffineGlorot"
      multiplier: 1.0
    }
  }
  variable {
    name: "Affine_2/affine/b"
    type: "Parameter"
    shape: { dim: 1 }
    initializer {
      type: "Constant"
      multiplier: 0.0
    }
  }
  variable {
    name: "Affine_6/affine/W"
    type: "Parameter"
    shape: { dim: 56 dim: 1 dim: 28 }
    initializer {
      type: "NormalAffineGlorot"
      multiplier: 1.0
    }
  }
  variable {
    name: "Affine_6/affine/b"
    type: "Parameter"
    shape: { dim: 1 dim: 28 }
    initializer {
      type: "Constant"
      multiplier: 0.0
    }
  }
  variable {
    name: "Affine_3/affine/W"
    type: "Parameter"
    shape: { dim: 56 dim: 1 dim: 28 }
    initializer {
      type: "NormalAffineGlorot"
      multiplier: 1.0
    }
  }
  variable {
    name: "Affine_3/affine/b"
    type: "Parameter"
    shape: { dim: 1 dim: 28 }
    initializer {
      type: "Constant"
      multiplier: 0.0
    }
  }
  variable {
    name: "Affine_7/affine/W"
    type: "Parameter"
    shape: { dim: 56 dim: 1 dim: 28 }
    initializer {
      type: "NormalAffineGlorot"
      multiplier: 1.0
    }
  }
  variable {
    name: "Affine_7/affine/b"
    type: "Parameter"
    shape: { dim: 1 dim: 28 }
    initializer {
      type: "Constant"
      multiplier: 0.0
    }
  }
  variable {
    name: "Delay_2_Initial"
    type: "Buffer"
    shape: { dim:-1 dim: 1 dim: 28 }
  }
  variable {
    name: "Concatenate"
    type: "Buffer"
    repeat_id: "RecurrentInput"
    shape: { dim:-1 dim: 2 dim: 28 }
  }
  variable {
    name: "Affine_4"
    type: "Buffer"
    repeat_id: "RecurrentInput"
    shape: { dim:-1 dim: 1 dim: 28 }
  }
  variable {
    name: "Sigmoid"
    type: "Buffer"
    repeat_id: "RecurrentInput"
    shape: { dim:-1 dim: 1 dim: 28 }
  }
  variable {
    name: "Mul2"
    type: "Buffer"
    repeat_id: "RecurrentInput"
    shape: { dim:-1 dim: 1 dim: 28 }
  }
  variable {
    name: "Concatenate_2"
    type: "Buffer"
    repeat_id: "RecurrentInput"
    shape: { dim:-1 dim: 2 dim: 28 }
  }
  variable {
    name: "Affine"
    type: "Buffer"
    repeat_id: "RecurrentInput"
    shape: { dim:-1 dim: 1 dim: 28 }
  }
  variable {
    name: "Tanh"
    type: "Buffer"
    repeat_id: "RecurrentInput"
    shape: { dim:-1 dim: 1 dim: 28 }
  }
  variable {
    name: "Affine_5"
    type: "Buffer"
    repeat_id: "RecurrentInput"
    shape: { dim:-1 dim: 1 dim: 28 }
  }
  variable {
    name: "Sigmoid_2"
    type: "Buffer"
    repeat_id: "RecurrentInput"
    shape: { dim:-1 dim: 1 dim: 28 }
  }
  variable {
    name: "Mul2_2"
    type: "Buffer"
    repeat_id: "RecurrentInput"
    shape: { dim:-1 dim: 1 dim: 28 }
  }
  variable {
    name: "Add2"
    type: "Buffer"
    repeat_id: "RecurrentInput"
    shape: { dim:-1 dim: 1 dim: 28 }
  }
  variable {
    name: "Mul2_3"
    type: "Buffer"
    repeat_id: "RecurrentInput"
    shape: { dim:-1 dim: 1 dim: 28 }
  }
  variable {
    name: "RSubScalar"
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
    name: "BinaryCrossEntropy"
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
    shape: { dim:-1 dim: 1 }
  }
  variable {
    name: "Sigmoid_3"
    type: "Buffer"
    shape: { dim:-1 dim: 1 }
  }
  variable {
    name: "Slice"
    type: "Buffer"
    shape: { dim:-1 dim: 1 dim: 1 dim: 28 }
  }
  variable {
    name: "Concatenate_3"
    type: "Buffer"
    repeat_id: "RecurrentInput"
    shape: { dim:-1 dim: 2 dim: 28 }
  }
  variable {
    name: "Affine_6"
    type: "Buffer"
    repeat_id: "RecurrentInput"
    shape: { dim:-1 dim: 1 dim: 28 }
  }
  variable {
    name: "Sigmoid_4"
    type: "Buffer"
    repeat_id: "RecurrentInput"
    shape: { dim:-1 dim: 1 dim: 28 }
  }
  variable {
    name: "Mul2_4"
    type: "Buffer"
    repeat_id: "RecurrentInput"
    shape: { dim:-1 dim: 1 dim: 28 }
  }
  variable {
    name: "Concatenate_4"
    type: "Buffer"
    repeat_id: "RecurrentInput"
    shape: { dim:-1 dim: 2 dim: 28 }
  }
  variable {
    name: "Affine_3"
    type: "Buffer"
    repeat_id: "RecurrentInput"
    shape: { dim:-1 dim: 1 dim: 28 }
  }
  variable {
    name: "Tanh_2"
    type: "Buffer"
    repeat_id: "RecurrentInput"
    shape: { dim:-1 dim: 1 dim: 28 }
  }
  variable {
    name: "Affine_7"
    type: "Buffer"
    repeat_id: "RecurrentInput"
    shape: { dim:-1 dim: 1 dim: 28 }
  }
  variable {
    name: "Sigmoid_5"
    type: "Buffer"
    repeat_id: "RecurrentInput"
    shape: { dim:-1 dim: 1 dim: 28 }
  }
  variable {
    name: "Mul2_5"
    type: "Buffer"
    repeat_id: "RecurrentInput"
    shape: { dim:-1 dim: 1 dim: 28 }
  }
  variable {
    name: "Add2_2"
    type: "Buffer"
    repeat_id: "RecurrentInput"
    shape: { dim:-1 dim: 1 dim: 28 }
  }
  variable {
    name: "Mul2_6"
    type: "Buffer"
    repeat_id: "RecurrentInput"
    shape: { dim:-1 dim: 1 dim: 28 }
  }
  variable {
    name: "RSubScalar_2"
    type: "Buffer"
    repeat_id: "RecurrentInput"
    shape: { dim:-1 dim: 1 dim: 28 }
  }
  variable {
    name: "Delay_2"
    type: "Buffer"
    repeat_id: "RecurrentInput"
    shape: { dim:-1 dim: 1 dim: 28 }
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
    name: "Affine_4"
    type: "Affine"
    repeat_id: "RecurrentInput"
    input: "Concatenate"
    input: "Affine_4/affine/W"
    input: "Affine_4/affine/b"
    output: "Affine_4"
    affine_param {
      base_axis: 1
    }
  }
  function {
    name: "Sigmoid"
    type: "Sigmoid"
    repeat_id: "RecurrentInput"
    input: "Affine_4"
    output: "Sigmoid"
  }
  function {
    name: "Mul2"
    type: "Mul2"
    repeat_id: "RecurrentInput"
    input: "Sigmoid"
    input: "Delay"
    output: "Mul2"
  }
  function {
    name: "Concatenate_2"
    type: "Concatenate"
    repeat_id: "RecurrentInput"
    input: "Mul2"
    input: "RecurrentInput"
    output: "Concatenate_2"
    concatenate_param {
      axis: 1
    }
  }
  function {
    name: "Affine"
    type: "Affine"
    repeat_id: "RecurrentInput"
    input: "Concatenate_2"
    input: "Affine/affine/W"
    input: "Affine/affine/b"
    output: "Affine"
    affine_param {
      base_axis: 1
    }
  }
  function {
    name: "Tanh"
    type: "Tanh"
    repeat_id: "RecurrentInput"
    input: "Affine"
    output: "Tanh"
  }
  function {
    name: "Affine_5"
    type: "Affine"
    repeat_id: "RecurrentInput"
    input: "Concatenate"
    input: "Affine_5/affine/W"
    input: "Affine_5/affine/b"
    output: "Affine_5"
    affine_param {
      base_axis: 1
    }
  }
  function {
    name: "Sigmoid_2"
    type: "Sigmoid"
    repeat_id: "RecurrentInput"
    input: "Affine_5"
    output: "Sigmoid_2"
  }
  function {
    name: "Mul2_2"
    type: "Mul2"
    repeat_id: "RecurrentInput"
    input: "RSubScalar"
    input: "Delay"
    output: "Mul2_2"
  }
  function {
    name: "Add2"
    type: "Add2"
    repeat_id: "RecurrentInput"
    input: "Mul2_2"
    input: "Mul2_3"
    output: "Add2"
    add2_param {
      inplace: True
    }
  }
  function {
    name: "Mul2_3"
    type: "Mul2"
    repeat_id: "RecurrentInput"
    input: "Tanh"
    input: "Sigmoid_2"
    output: "Mul2_3"
  }
  function {
    name: "RSubScalar"
    type: "RSubScalar"
    repeat_id: "RecurrentInput"
    input: "Sigmoid_2"
    output: "RSubScalar"
    r_sub_scalar_param {
      val: 1.0
    }
  }
  function {
    name: "RecurrentOutput"
    type: "RecurrentOutput"
    input: "Add2_2"
    output: "RecurrentOutput"
    recurrent_param {
      axis: 2
      repeat_id: "RecurrentInput"
      length: 28
    }
  }
  function {
    name: "BinaryCrossEntropy"
    type: "BinaryCrossEntropy"
    input: "Sigmoid_3"
    input: "BinaryCrossEntropy_T"
    output: "BinaryCrossEntropy"
  }
  function {
    name: "Delay"
    type: "Delay"
    repeat_id: "RecurrentInput"
    input: "Add2"
    input: "Delay_Initial"
    output: "Delay"
    recurrent_param {
      repeat_id: "RecurrentInput"
    }
  }
  function {
    name: "Affine_2"
    type: "Affine"
    input: "Slice"
    input: "Affine_2/affine/W"
    input: "Affine_2/affine/b"
    output: "Affine_2"
    affine_param {
      base_axis: 1
    }
  }
  function {
    name: "Sigmoid_3"
    type: "Sigmoid"
    input: "Affine_2"
    output: "Sigmoid_3"
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
  function {
    name: "Concatenate_3"
    type: "Concatenate"
    repeat_id: "RecurrentInput"
    input: "Delay_2"
    input: "Add2"
    output: "Concatenate_3"
    concatenate_param {
      axis: 1
    }
  }
  function {
    name: "Affine_6"
    type: "Affine"
    repeat_id: "RecurrentInput"
    input: "Concatenate_3"
    input: "Affine_6/affine/W"
    input: "Affine_6/affine/b"
    output: "Affine_6"
    affine_param {
      base_axis: 1
    }
  }
  function {
    name: "Sigmoid_4"
    type: "Sigmoid"
    repeat_id: "RecurrentInput"
    input: "Affine_6"
    output: "Sigmoid_4"
  }
  function {
    name: "Mul2_4"
    type: "Mul2"
    repeat_id: "RecurrentInput"
    input: "Sigmoid_4"
    input: "Delay_2"
    output: "Mul2_4"
  }
  function {
    name: "Concatenate_4"
    type: "Concatenate"
    repeat_id: "RecurrentInput"
    input: "Mul2_4"
    input: "Add2"
    output: "Concatenate_4"
    concatenate_param {
      axis: 1
    }
  }
  function {
    name: "Affine_3"
    type: "Affine"
    repeat_id: "RecurrentInput"
    input: "Concatenate_4"
    input: "Affine_3/affine/W"
    input: "Affine_3/affine/b"
    output: "Affine_3"
    affine_param {
      base_axis: 1
    }
  }
  function {
    name: "Tanh_2"
    type: "Tanh"
    repeat_id: "RecurrentInput"
    input: "Affine_3"
    output: "Tanh_2"
  }
  function {
    name: "Affine_7"
    type: "Affine"
    repeat_id: "RecurrentInput"
    input: "Concatenate_3"
    input: "Affine_7/affine/W"
    input: "Affine_7/affine/b"
    output: "Affine_7"
    affine_param {
      base_axis: 1
    }
  }
  function {
    name: "Sigmoid_5"
    type: "Sigmoid"
    repeat_id: "RecurrentInput"
    input: "Affine_7"
    output: "Sigmoid_5"
  }
  function {
    name: "Mul2_5"
    type: "Mul2"
    repeat_id: "RecurrentInput"
    input: "RSubScalar_2"
    input: "Delay_2"
    output: "Mul2_5"
  }
  function {
    name: "Add2_2"
    type: "Add2"
    repeat_id: "RecurrentInput"
    input: "Mul2_5"
    input: "Mul2_6"
    output: "Add2_2"
    add2_param {
      inplace: True
    }
  }
  function {
    name: "Mul2_6"
    type: "Mul2"
    repeat_id: "RecurrentInput"
    input: "Tanh_2"
    input: "Sigmoid_5"
    output: "Mul2_6"
  }
  function {
    name: "RSubScalar_2"
    type: "RSubScalar"
    repeat_id: "RecurrentInput"
    input: "Sigmoid_5"
    output: "RSubScalar_2"
    r_sub_scalar_param {
      val: 1.0
    }
  }
  function {
    name: "Delay_2"
    type: "Delay"
    repeat_id: "RecurrentInput"
    input: "Add2_2"
    input: "Delay_2_Initial"
    output: "Delay_2"
    recurrent_param {
      repeat_id: "RecurrentInput"
    }
  }
}
network {
  name: "Runtime"
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
    name: "Affine_4/affine/W"
    type: "Parameter"
    shape: { dim: 56 dim: 1 dim: 28 }
    initializer {
      type: "NormalAffineGlorot"
      multiplier: 1.0
    }
  }
  variable {
    name: "Affine_4/affine/b"
    type: "Parameter"
    shape: { dim: 1 dim: 28 }
    initializer {
      type: "Constant"
      multiplier: 0.0
    }
  }
  variable {
    name: "Affine/affine/W"
    type: "Parameter"
    shape: { dim: 56 dim: 1 dim: 28 }
    initializer {
      type: "NormalAffineGlorot"
      multiplier: 1.0
    }
  }
  variable {
    name: "Affine/affine/b"
    type: "Parameter"
    shape: { dim: 1 dim: 28 }
    initializer {
      type: "Constant"
      multiplier: 0.0
    }
  }
  variable {
    name: "Affine_5/affine/W"
    type: "Parameter"
    shape: { dim: 56 dim: 1 dim: 28 }
    initializer {
      type: "NormalAffineGlorot"
      multiplier: 1.0
    }
  }
  variable {
    name: "Affine_5/affine/b"
    type: "Parameter"
    shape: { dim: 1 dim: 28 }
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
    shape: { dim: 28 dim: 1 }
    initializer {
      type: "NormalAffineGlorot"
      multiplier: 1.0
    }
  }
  variable {
    name: "Affine_2/affine/b"
    type: "Parameter"
    shape: { dim: 1 }
    initializer {
      type: "Constant"
      multiplier: 0.0
    }
  }
  variable {
    name: "Affine_6/affine/W"
    type: "Parameter"
    shape: { dim: 56 dim: 1 dim: 28 }
    initializer {
      type: "NormalAffineGlorot"
      multiplier: 1.0
    }
  }
  variable {
    name: "Affine_6/affine/b"
    type: "Parameter"
    shape: { dim: 1 dim: 28 }
    initializer {
      type: "Constant"
      multiplier: 0.0
    }
  }
  variable {
    name: "Affine_3/affine/W"
    type: "Parameter"
    shape: { dim: 56 dim: 1 dim: 28 }
    initializer {
      type: "NormalAffineGlorot"
      multiplier: 1.0
    }
  }
  variable {
    name: "Affine_3/affine/b"
    type: "Parameter"
    shape: { dim: 1 dim: 28 }
    initializer {
      type: "Constant"
      multiplier: 0.0
    }
  }
  variable {
    name: "Affine_7/affine/W"
    type: "Parameter"
    shape: { dim: 56 dim: 1 dim: 28 }
    initializer {
      type: "NormalAffineGlorot"
      multiplier: 1.0
    }
  }
  variable {
    name: "Affine_7/affine/b"
    type: "Parameter"
    shape: { dim: 1 dim: 28 }
    initializer {
      type: "Constant"
      multiplier: 0.0
    }
  }
  variable {
    name: "Delay_2_Initial"
    type: "Buffer"
    shape: { dim:-1 dim: 1 dim: 28 }
  }
  variable {
    name: "Concatenate"
    type: "Buffer"
    repeat_id: "RecurrentInput"
    shape: { dim:-1 dim: 2 dim: 28 }
  }
  variable {
    name: "Affine_4"
    type: "Buffer"
    repeat_id: "RecurrentInput"
    shape: { dim:-1 dim: 1 dim: 28 }
  }
  variable {
    name: "Sigmoid"
    type: "Buffer"
    repeat_id: "RecurrentInput"
    shape: { dim:-1 dim: 1 dim: 28 }
  }
  variable {
    name: "Mul2"
    type: "Buffer"
    repeat_id: "RecurrentInput"
    shape: { dim:-1 dim: 1 dim: 28 }
  }
  variable {
    name: "Concatenate_2"
    type: "Buffer"
    repeat_id: "RecurrentInput"
    shape: { dim:-1 dim: 2 dim: 28 }
  }
  variable {
    name: "Affine"
    type: "Buffer"
    repeat_id: "RecurrentInput"
    shape: { dim:-1 dim: 1 dim: 28 }
  }
  variable {
    name: "Tanh"
    type: "Buffer"
    repeat_id: "RecurrentInput"
    shape: { dim:-1 dim: 1 dim: 28 }
  }
  variable {
    name: "Affine_5"
    type: "Buffer"
    repeat_id: "RecurrentInput"
    shape: { dim:-1 dim: 1 dim: 28 }
  }
  variable {
    name: "Sigmoid_2"
    type: "Buffer"
    repeat_id: "RecurrentInput"
    shape: { dim:-1 dim: 1 dim: 28 }
  }
  variable {
    name: "Mul2_2"
    type: "Buffer"
    repeat_id: "RecurrentInput"
    shape: { dim:-1 dim: 1 dim: 28 }
  }
  variable {
    name: "Add2"
    type: "Buffer"
    repeat_id: "RecurrentInput"
    shape: { dim:-1 dim: 1 dim: 28 }
  }
  variable {
    name: "Mul2_3"
    type: "Buffer"
    repeat_id: "RecurrentInput"
    shape: { dim:-1 dim: 1 dim: 28 }
  }
  variable {
    name: "RSubScalar"
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
    name: "Delay"
    type: "Buffer"
    repeat_id: "RecurrentInput"
    shape: { dim:-1 dim: 1 dim: 28 }
  }
  variable {
    name: "Affine_2"
    type: "Buffer"
    shape: { dim:-1 dim: 1 }
  }
  variable {
    name: "y'"
    type: "Buffer"
    shape: { dim:-1 dim: 1 }
  }
  variable {
    name: "Slice"
    type: "Buffer"
    shape: { dim:-1 dim: 1 dim: 1 dim: 28 }
  }
  variable {
    name: "Concatenate_3"
    type: "Buffer"
    repeat_id: "RecurrentInput"
    shape: { dim:-1 dim: 2 dim: 28 }
  }
  variable {
    name: "Affine_6"
    type: "Buffer"
    repeat_id: "RecurrentInput"
    shape: { dim:-1 dim: 1 dim: 28 }
  }
  variable {
    name: "Sigmoid_4"
    type: "Buffer"
    repeat_id: "RecurrentInput"
    shape: { dim:-1 dim: 1 dim: 28 }
  }
  variable {
    name: "Mul2_4"
    type: "Buffer"
    repeat_id: "RecurrentInput"
    shape: { dim:-1 dim: 1 dim: 28 }
  }
  variable {
    name: "Concatenate_4"
    type: "Buffer"
    repeat_id: "RecurrentInput"
    shape: { dim:-1 dim: 2 dim: 28 }
  }
  variable {
    name: "Affine_3"
    type: "Buffer"
    repeat_id: "RecurrentInput"
    shape: { dim:-1 dim: 1 dim: 28 }
  }
  variable {
    name: "Tanh_2"
    type: "Buffer"
    repeat_id: "RecurrentInput"
    shape: { dim:-1 dim: 1 dim: 28 }
  }
  variable {
    name: "Affine_7"
    type: "Buffer"
    repeat_id: "RecurrentInput"
    shape: { dim:-1 dim: 1 dim: 28 }
  }
  variable {
    name: "Sigmoid_5"
    type: "Buffer"
    repeat_id: "RecurrentInput"
    shape: { dim:-1 dim: 1 dim: 28 }
  }
  variable {
    name: "Mul2_5"
    type: "Buffer"
    repeat_id: "RecurrentInput"
    shape: { dim:-1 dim: 1 dim: 28 }
  }
  variable {
    name: "Add2_2"
    type: "Buffer"
    repeat_id: "RecurrentInput"
    shape: { dim:-1 dim: 1 dim: 28 }
  }
  variable {
    name: "Mul2_6"
    type: "Buffer"
    repeat_id: "RecurrentInput"
    shape: { dim:-1 dim: 1 dim: 28 }
  }
  variable {
    name: "RSubScalar_2"
    type: "Buffer"
    repeat_id: "RecurrentInput"
    shape: { dim:-1 dim: 1 dim: 28 }
  }
  variable {
    name: "Delay_2"
    type: "Buffer"
    repeat_id: "RecurrentInput"
    shape: { dim:-1 dim: 1 dim: 28 }
  }
  variable {
    name: "rnn_out"
    type: "Buffer"
    shape: { dim:-1 dim: 1 dim: 28 dim: 28 }
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
    name: "Affine_4"
    type: "Affine"
    repeat_id: "RecurrentInput"
    input: "Concatenate"
    input: "Affine_4/affine/W"
    input: "Affine_4/affine/b"
    output: "Affine_4"
    affine_param {
      base_axis: 1
    }
  }
  function {
    name: "Sigmoid"
    type: "Sigmoid"
    repeat_id: "RecurrentInput"
    input: "Affine_4"
    output: "Sigmoid"
  }
  function {
    name: "Mul2"
    type: "Mul2"
    repeat_id: "RecurrentInput"
    input: "Sigmoid"
    input: "Delay"
    output: "Mul2"
  }
  function {
    name: "Concatenate_2"
    type: "Concatenate"
    repeat_id: "RecurrentInput"
    input: "Mul2"
    input: "RecurrentInput"
    output: "Concatenate_2"
    concatenate_param {
      axis: 1
    }
  }
  function {
    name: "Affine"
    type: "Affine"
    repeat_id: "RecurrentInput"
    input: "Concatenate_2"
    input: "Affine/affine/W"
    input: "Affine/affine/b"
    output: "Affine"
    affine_param {
      base_axis: 1
    }
  }
  function {
    name: "Tanh"
    type: "Tanh"
    repeat_id: "RecurrentInput"
    input: "Affine"
    output: "Tanh"
  }
  function {
    name: "Affine_5"
    type: "Affine"
    repeat_id: "RecurrentInput"
    input: "Concatenate"
    input: "Affine_5/affine/W"
    input: "Affine_5/affine/b"
    output: "Affine_5"
    affine_param {
      base_axis: 1
    }
  }
  function {
    name: "Sigmoid_2"
    type: "Sigmoid"
    repeat_id: "RecurrentInput"
    input: "Affine_5"
    output: "Sigmoid_2"
  }
  function {
    name: "Mul2_2"
    type: "Mul2"
    repeat_id: "RecurrentInput"
    input: "RSubScalar"
    input: "Delay"
    output: "Mul2_2"
  }
  function {
    name: "Add2"
    type: "Add2"
    repeat_id: "RecurrentInput"
    input: "Mul2_2"
    input: "Mul2_3"
    output: "Add2"
    add2_param {
      inplace: True
    }
  }
  function {
    name: "Mul2_3"
    type: "Mul2"
    repeat_id: "RecurrentInput"
    input: "Tanh"
    input: "Sigmoid_2"
    output: "Mul2_3"
  }
  function {
    name: "RSubScalar"
    type: "RSubScalar"
    repeat_id: "RecurrentInput"
    input: "Sigmoid_2"
    output: "RSubScalar"
    r_sub_scalar_param {
      val: 1.0
    }
  }
  function {
    name: "RecurrentOutput"
    type: "RecurrentOutput"
    input: "Add2_2"
    output: "RecurrentOutput"
    recurrent_param {
      axis: 2
      repeat_id: "RecurrentInput"
      length: 28
    }
  }
  function {
    name: "Delay"
    type: "Delay"
    repeat_id: "RecurrentInput"
    input: "Add2"
    input: "Delay_Initial"
    output: "Delay"
    recurrent_param {
      repeat_id: "RecurrentInput"
    }
  }
  function {
    name: "Affine_2"
    type: "Affine"
    input: "Slice"
    input: "Affine_2/affine/W"
    input: "Affine_2/affine/b"
    output: "Affine_2"
    affine_param {
      base_axis: 1
    }
  }
  function {
    name: "y'"
    type: "Sigmoid"
    input: "Affine_2"
    output: "y'"
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
  function {
    name: "Concatenate_3"
    type: "Concatenate"
    repeat_id: "RecurrentInput"
    input: "Delay_2"
    input: "Add2"
    output: "Concatenate_3"
    concatenate_param {
      axis: 1
    }
  }
  function {
    name: "Affine_6"
    type: "Affine"
    repeat_id: "RecurrentInput"
    input: "Concatenate_3"
    input: "Affine_6/affine/W"
    input: "Affine_6/affine/b"
    output: "Affine_6"
    affine_param {
      base_axis: 1
    }
  }
  function {
    name: "Sigmoid_4"
    type: "Sigmoid"
    repeat_id: "RecurrentInput"
    input: "Affine_6"
    output: "Sigmoid_4"
  }
  function {
    name: "Mul2_4"
    type: "Mul2"
    repeat_id: "RecurrentInput"
    input: "Sigmoid_4"
    input: "Delay_2"
    output: "Mul2_4"
  }
  function {
    name: "Concatenate_4"
    type: "Concatenate"
    repeat_id: "RecurrentInput"
    input: "Mul2_4"
    input: "Add2"
    output: "Concatenate_4"
    concatenate_param {
      axis: 1
    }
  }
  function {
    name: "Affine_3"
    type: "Affine"
    repeat_id: "RecurrentInput"
    input: "Concatenate_4"
    input: "Affine_3/affine/W"
    input: "Affine_3/affine/b"
    output: "Affine_3"
    affine_param {
      base_axis: 1
    }
  }
  function {
    name: "Tanh_2"
    type: "Tanh"
    repeat_id: "RecurrentInput"
    input: "Affine_3"
    output: "Tanh_2"
  }
  function {
    name: "Affine_7"
    type: "Affine"
    repeat_id: "RecurrentInput"
    input: "Concatenate_3"
    input: "Affine_7/affine/W"
    input: "Affine_7/affine/b"
    output: "Affine_7"
    affine_param {
      base_axis: 1
    }
  }
  function {
    name: "Sigmoid_5"
    type: "Sigmoid"
    repeat_id: "RecurrentInput"
    input: "Affine_7"
    output: "Sigmoid_5"
  }
  function {
    name: "Mul2_5"
    type: "Mul2"
    repeat_id: "RecurrentInput"
    input: "RSubScalar_2"
    input: "Delay_2"
    output: "Mul2_5"
  }
  function {
    name: "Add2_2"
    type: "Add2"
    repeat_id: "RecurrentInput"
    input: "Mul2_5"
    input: "Mul2_6"
    output: "Add2_2"
    add2_param {
      inplace: True
    }
  }
  function {
    name: "Mul2_6"
    type: "Mul2"
    repeat_id: "RecurrentInput"
    input: "Tanh_2"
    input: "Sigmoid_5"
    output: "Mul2_6"
  }
  function {
    name: "RSubScalar_2"
    type: "RSubScalar"
    repeat_id: "RecurrentInput"
    input: "Sigmoid_5"
    output: "RSubScalar_2"
    r_sub_scalar_param {
      val: 1.0
    }
  }
  function {
    name: "Delay_2"
    type: "Delay"
    repeat_id: "RecurrentInput"
    input: "Add2_2"
    input: "Delay_2_Initial"
    output: "Delay_2"
    recurrent_param {
      repeat_id: "RecurrentInput"
    }
  }
  function {
    name: "rnn_out"
    type: "Identity"
    input: "RecurrentOutput"
    output: "rnn_out"
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
  network_name: "Main_"
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
  generator_variable {
    variable_name: "Delay_Initial"
    type: "Constant"
    multiplier: 0.0
  }
  generator_variable {
    variable_name: "Delay_2_Initial"
    type: "Constant"
    multiplier: 0.0
  }
  loss_variable {
    variable_name: "BinaryCrossEntropy"
  }
  parameter_variable {
    variable_name: "Affine_4/affine/W"
    learning_rate_multiplier: 1.0
  }
  parameter_variable {
    variable_name: "Affine_4/affine/b"
    learning_rate_multiplier: 1.0
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
    variable_name: "Affine_5/affine/W"
    learning_rate_multiplier: 1.0
  }
  parameter_variable {
    variable_name: "Affine_5/affine/b"
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
  parameter_variable {
    variable_name: "Affine_6/affine/W"
    learning_rate_multiplier: 1.0
  }
  parameter_variable {
    variable_name: "Affine_6/affine/b"
    learning_rate_multiplier: 1.0
  }
  parameter_variable {
    variable_name: "Affine_3/affine/W"
    learning_rate_multiplier: 1.0
  }
  parameter_variable {
    variable_name: "Affine_3/affine/b"
    learning_rate_multiplier: 1.0
  }
  parameter_variable {
    variable_name: "Affine_7/affine/W"
    learning_rate_multiplier: 1.0
  }
  parameter_variable {
    variable_name: "Affine_7/affine/b"
    learning_rate_multiplier: 1.0
  }
}
monitor {
  name: "train_error"
  network_name: "Main_"
  dataset_name: "Training"
  data_variable {
    variable_name: "Input"
    data_name: "x"
  }
  data_variable {
    variable_name: "BinaryCrossEntropy_T"
    data_name: "y"
  }
  generator_variable {
    variable_name: "Delay_Initial"
    type: "Constant"
    multiplier: 0.0
  }
  generator_variable {
    variable_name: "Delay_2_Initial"
    type: "Constant"
    multiplier: 0.0
  }
  monitor_variable {
    type: "Error"
    variable_name: "BinaryCrossEntropy"
  }
}
monitor {
  name: "valid_error"
  network_name: "Main_"
  dataset_name: "Validation"
  data_variable {
    variable_name: "Input"
    data_name: "x"
  }
  data_variable {
    variable_name: "BinaryCrossEntropy_T"
    data_name: "y"
  }
  generator_variable {
    variable_name: "Delay_Initial"
    type: "Constant"
    multiplier: 0.0
  }
  generator_variable {
    variable_name: "Delay_2_Initial"
    type: "Constant"
    multiplier: 0.0
  }
  monitor_variable {
    type: "Error"
    variable_name: "BinaryCrossEntropy"
  }
}
executor {
  name: "Executor"
  network_name: "Runtime"
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
  generator_variable {
    variable_name: "Delay_2_Initial"
    type: "Constant"
    multiplier: 0.0
  }
  output_variable {
    variable_name: "y'"
    data_name: "y'"
  }
  output_variable {
    variable_name: "rnn_out"
    data_name: "rnn_out"
  }
  parameter_variable {
    variable_name: "Affine_4/affine/W"
  }
  parameter_variable {
    variable_name: "Affine_4/affine/b"
  }
  parameter_variable {
    variable_name: "Affine/affine/W"
  }
  parameter_variable {
    variable_name: "Affine/affine/b"
  }
  parameter_variable {
    variable_name: "Affine_5/affine/W"
  }
  parameter_variable {
    variable_name: "Affine_5/affine/b"
  }
  parameter_variable {
    variable_name: "Affine_2/affine/W"
  }
  parameter_variable {
    variable_name: "Affine_2/affine/b"
  }
  parameter_variable {
    variable_name: "Affine_6/affine/W"
  }
  parameter_variable {
    variable_name: "Affine_6/affine/b"
  }
  parameter_variable {
    variable_name: "Affine_3/affine/W"
  }
  parameter_variable {
    variable_name: "Affine_3/affine/b"
  }
  parameter_variable {
    variable_name: "Affine_7/affine/W"
  }
  parameter_variable {
    variable_name: "Affine_7/affine/b"
  }
}
'''
# The nntxt is generated from /home/woody/aid/nnabla-sample-data/sample_project/tutorial/recurrent_neural_networks/stacked_GRU.files/20180720_193132/net.nntxt.
N00000009 = r'''global_config {
  default_context {
    array_class: "CudaCachedArray"
    device_id: "0"
    backends: "cudnn:float"
    backends: "cuda:float"
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
  name: "Main_"
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
    name: "Delay_Initial"
    type: "Buffer"
    shape: { dim:-1 dim: 1 dim: 28 }
  }
  variable {
    name: "Delay_2_Initial"
    type: "Buffer"
    shape: { dim:-1 dim: 1 dim: 28 }
  }
  variable {
    name: "Affine_4/affine/W"
    type: "Parameter"
    shape: { dim: 56 dim: 1 dim: 28 }
    initializer {
      type: "NormalAffineGlorot"
      multiplier: 1
    }
  }
  variable {
    name: "Affine_4/affine/b"
    type: "Parameter"
    shape: { dim: 1 dim: 28 }
    initializer {
      type: "Constant"
      multiplier: 0
    }
  }
  variable {
    name: "Affine_5/affine/W"
    type: "Parameter"
    shape: { dim: 56 dim: 1 dim: 28 }
    initializer {
      type: "NormalAffineGlorot"
      multiplier: 1
    }
  }
  variable {
    name: "Affine_5/affine/b"
    type: "Parameter"
    shape: { dim: 1 dim: 28 }
    initializer {
      type: "Constant"
      multiplier: 0
    }
  }
  variable {
    name: "Affine/affine/W"
    type: "Parameter"
    shape: { dim: 56 dim: 1 dim: 28 }
    initializer {
      type: "NormalAffineGlorot"
      multiplier: 1
    }
  }
  variable {
    name: "Affine/affine/b"
    type: "Parameter"
    shape: { dim: 1 dim: 28 }
    initializer {
      type: "Constant"
      multiplier: 0
    }
  }
  variable {
    name: "Affine_6/affine/W"
    type: "Parameter"
    shape: { dim: 56 dim: 1 dim: 28 }
    initializer {
      type: "NormalAffineGlorot"
      multiplier: 1
    }
  }
  variable {
    name: "Affine_6/affine/b"
    type: "Parameter"
    shape: { dim: 1 dim: 28 }
    initializer {
      type: "Constant"
      multiplier: 0
    }
  }
  variable {
    name: "Affine_7/affine/W"
    type: "Parameter"
    shape: { dim: 56 dim: 1 dim: 28 }
    initializer {
      type: "NormalAffineGlorot"
      multiplier: 1
    }
  }
  variable {
    name: "Affine_7/affine/b"
    type: "Parameter"
    shape: { dim: 1 dim: 28 }
    initializer {
      type: "Constant"
      multiplier: 0
    }
  }
  variable {
    name: "Affine_3/affine/W"
    type: "Parameter"
    shape: { dim: 56 dim: 1 dim: 28 }
    initializer {
      type: "NormalAffineGlorot"
      multiplier: 1
    }
  }
  variable {
    name: "Affine_3/affine/b"
    type: "Parameter"
    shape: { dim: 1 dim: 28 }
    initializer {
      type: "Constant"
      multiplier: 0
    }
  }
  variable {
    name: "Affine_2/affine/W"
    type: "Parameter"
    shape: { dim: 28 dim: 1 }
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
    name: "RecurrentInput"
    type: "Buffer"
    repeat_id: "RecurrentInput"
    shape: { dim:-1 dim: 1 dim: 28 }
  }
  variable {
    name: "Delay"
    type: "Buffer"
    repeat_id: "RecurrentInput"
    shape: { dim:-1 dim: 1 dim: 28 }
  }
  variable {
    name: "Delay_2"
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
    name: "Affine_4"
    type: "Buffer"
    repeat_id: "RecurrentInput"
    shape: { dim:-1 dim: 1 dim: 28 }
  }
  variable {
    name: "Affine_5"
    type: "Buffer"
    repeat_id: "RecurrentInput"
    shape: { dim:-1 dim: 1 dim: 28 }
  }
  variable {
    name: "Sigmoid"
    type: "Buffer"
    repeat_id: "RecurrentInput"
    shape: { dim:-1 dim: 1 dim: 28 }
  }
  variable {
    name: "Sigmoid_2"
    type: "Buffer"
    repeat_id: "RecurrentInput"
    shape: { dim:-1 dim: 1 dim: 28 }
  }
  variable {
    name: "Mul2"
    type: "Buffer"
    repeat_id: "RecurrentInput"
    shape: { dim:-1 dim: 1 dim: 28 }
  }
  variable {
    name: "RSubScalar"
    type: "Buffer"
    repeat_id: "RecurrentInput"
    shape: { dim:-1 dim: 1 dim: 28 }
  }
  variable {
    name: "Concatenate_2"
    type: "Buffer"
    repeat_id: "RecurrentInput"
    shape: { dim:-1 dim: 2 dim: 28 }
  }
  variable {
    name: "Mul2_2"
    type: "Buffer"
    repeat_id: "RecurrentInput"
    shape: { dim:-1 dim: 1 dim: 28 }
  }
  variable {
    name: "Affine"
    type: "Buffer"
    repeat_id: "RecurrentInput"
    shape: { dim:-1 dim: 1 dim: 28 }
  }
  variable {
    name: "Tanh"
    type: "Buffer"
    repeat_id: "RecurrentInput"
    shape: { dim:-1 dim: 1 dim: 28 }
  }
  variable {
    name: "Mul2_3"
    type: "Buffer"
    repeat_id: "RecurrentInput"
    shape: { dim:-1 dim: 1 dim: 28 }
  }
  variable {
    name: "Add2"
    type: "Buffer"
    repeat_id: "RecurrentInput"
    shape: { dim:-1 dim: 1 dim: 28 }
  }
  variable {
    name: "Concatenate_3"
    type: "Buffer"
    repeat_id: "RecurrentInput"
    shape: { dim:-1 dim: 2 dim: 28 }
  }
  variable {
    name: "Affine_6"
    type: "Buffer"
    repeat_id: "RecurrentInput"
    shape: { dim:-1 dim: 1 dim: 28 }
  }
  variable {
    name: "Affine_7"
    type: "Buffer"
    repeat_id: "RecurrentInput"
    shape: { dim:-1 dim: 1 dim: 28 }
  }
  variable {
    name: "Sigmoid_4"
    type: "Buffer"
    repeat_id: "RecurrentInput"
    shape: { dim:-1 dim: 1 dim: 28 }
  }
  variable {
    name: "Sigmoid_5"
    type: "Buffer"
    repeat_id: "RecurrentInput"
    shape: { dim:-1 dim: 1 dim: 28 }
  }
  variable {
    name: "Mul2_4"
    type: "Buffer"
    repeat_id: "RecurrentInput"
    shape: { dim:-1 dim: 1 dim: 28 }
  }
  variable {
    name: "RSubScalar_2"
    type: "Buffer"
    repeat_id: "RecurrentInput"
    shape: { dim:-1 dim: 1 dim: 28 }
  }
  variable {
    name: "Concatenate_4"
    type: "Buffer"
    repeat_id: "RecurrentInput"
    shape: { dim:-1 dim: 2 dim: 28 }
  }
  variable {
    name: "Mul2_5"
    type: "Buffer"
    repeat_id: "RecurrentInput"
    shape: { dim:-1 dim: 1 dim: 28 }
  }
  variable {
    name: "Affine_3"
    type: "Buffer"
    repeat_id: "RecurrentInput"
    shape: { dim:-1 dim: 1 dim: 28 }
  }
  variable {
    name: "Tanh_2"
    type: "Buffer"
    repeat_id: "RecurrentInput"
    shape: { dim:-1 dim: 1 dim: 28 }
  }
  variable {
    name: "Mul2_6"
    type: "Buffer"
    repeat_id: "RecurrentInput"
    shape: { dim:-1 dim: 1 dim: 28 }
  }
  variable {
    name: "Add2_2"
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
  variable {
    name: "Affine_2"
    type: "Buffer"
    shape: { dim:-1 dim: 1 }
  }
  variable {
    name: "Sigmoid_3"
    type: "Buffer"
    shape: { dim:-1 dim: 1 }
  }
  variable {
    name: "BinaryCrossEntropy"
    type: "Buffer"
    shape: { dim:-1 dim: 1 }
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
    name: "Delay"
    type: "Delay"
    repeat_id: "RecurrentInput"
    input: "Add2"
    input: "Delay_Initial"
    output: "Delay"
    recurrent_param {
      repeat_id: "RecurrentInput"
    }
  }
  function {
    name: "Delay_2"
    type: "Delay"
    repeat_id: "RecurrentInput"
    input: "Add2_2"
    input: "Delay_2_Initial"
    output: "Delay_2"
    recurrent_param {
      repeat_id: "RecurrentInput"
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
    name: "Affine_4"
    type: "Affine"
    repeat_id: "RecurrentInput"
    input: "Concatenate"
    input: "Affine_4/affine/W"
    input: "Affine_4/affine/b"
    output: "Affine_4"
    affine_param {
      base_axis: 1
    }
  }
  function {
    name: "Affine_5"
    type: "Affine"
    repeat_id: "RecurrentInput"
    input: "Concatenate"
    input: "Affine_5/affine/W"
    input: "Affine_5/affine/b"
    output: "Affine_5"
    affine_param {
      base_axis: 1
    }
  }
  function {
    name: "Sigmoid"
    type: "Sigmoid"
    repeat_id: "RecurrentInput"
    input: "Affine_4"
    output: "Sigmoid"
  }
  function {
    name: "Sigmoid_2"
    type: "Sigmoid"
    repeat_id: "RecurrentInput"
    input: "Affine_5"
    output: "Sigmoid_2"
  }
  function {
    name: "Mul2"
    type: "Mul2"
    repeat_id: "RecurrentInput"
    input: "Sigmoid"
    input: "Delay"
    output: "Mul2"
  }
  function {
    name: "RSubScalar"
    type: "RSubScalar"
    repeat_id: "RecurrentInput"
    input: "Sigmoid_2"
    output: "RSubScalar"
    r_sub_scalar_param {
      val: 1
    }
  }
  function {
    name: "Concatenate_2"
    type: "Concatenate"
    repeat_id: "RecurrentInput"
    input: "Mul2"
    input: "RecurrentInput"
    output: "Concatenate_2"
    concatenate_param {
      axis: 1
    }
  }
  function {
    name: "Mul2_2"
    type: "Mul2"
    repeat_id: "RecurrentInput"
    input: "RSubScalar"
    input: "Delay"
    output: "Mul2_2"
  }
  function {
    name: "Affine"
    type: "Affine"
    repeat_id: "RecurrentInput"
    input: "Concatenate_2"
    input: "Affine/affine/W"
    input: "Affine/affine/b"
    output: "Affine"
    affine_param {
      base_axis: 1
    }
  }
  function {
    name: "Tanh"
    type: "Tanh"
    repeat_id: "RecurrentInput"
    input: "Affine"
    output: "Tanh"
  }
  function {
    name: "Mul2_3"
    type: "Mul2"
    repeat_id: "RecurrentInput"
    input: "Tanh"
    input: "Sigmoid_2"
    output: "Mul2_3"
  }
  function {
    name: "Add2"
    type: "Add2"
    repeat_id: "RecurrentInput"
    input: "Mul2_2"
    input: "Mul2_3"
    output: "Add2"
    add2_param {
      inplace: True
    }
  }
  function {
    name: "Concatenate_3"
    type: "Concatenate"
    repeat_id: "RecurrentInput"
    input: "Delay_2"
    input: "Add2"
    output: "Concatenate_3"
    concatenate_param {
      axis: 1
    }
  }
  function {
    name: "Affine_6"
    type: "Affine"
    repeat_id: "RecurrentInput"
    input: "Concatenate_3"
    input: "Affine_6/affine/W"
    input: "Affine_6/affine/b"
    output: "Affine_6"
    affine_param {
      base_axis: 1
    }
  }
  function {
    name: "Affine_7"
    type: "Affine"
    repeat_id: "RecurrentInput"
    input: "Concatenate_3"
    input: "Affine_7/affine/W"
    input: "Affine_7/affine/b"
    output: "Affine_7"
    affine_param {
      base_axis: 1
    }
  }
  function {
    name: "Sigmoid_4"
    type: "Sigmoid"
    repeat_id: "RecurrentInput"
    input: "Affine_6"
    output: "Sigmoid_4"
  }
  function {
    name: "Sigmoid_5"
    type: "Sigmoid"
    repeat_id: "RecurrentInput"
    input: "Affine_7"
    output: "Sigmoid_5"
  }
  function {
    name: "Mul2_4"
    type: "Mul2"
    repeat_id: "RecurrentInput"
    input: "Sigmoid_4"
    input: "Delay_2"
    output: "Mul2_4"
  }
  function {
    name: "RSubScalar_2"
    type: "RSubScalar"
    repeat_id: "RecurrentInput"
    input: "Sigmoid_5"
    output: "RSubScalar_2"
    r_sub_scalar_param {
      val: 1
    }
  }
  function {
    name: "Concatenate_4"
    type: "Concatenate"
    repeat_id: "RecurrentInput"
    input: "Mul2_4"
    input: "Add2"
    output: "Concatenate_4"
    concatenate_param {
      axis: 1
    }
  }
  function {
    name: "Mul2_5"
    type: "Mul2"
    repeat_id: "RecurrentInput"
    input: "RSubScalar_2"
    input: "Delay_2"
    output: "Mul2_5"
  }
  function {
    name: "Affine_3"
    type: "Affine"
    repeat_id: "RecurrentInput"
    input: "Concatenate_4"
    input: "Affine_3/affine/W"
    input: "Affine_3/affine/b"
    output: "Affine_3"
    affine_param {
      base_axis: 1
    }
  }
  function {
    name: "Tanh_2"
    type: "Tanh"
    repeat_id: "RecurrentInput"
    input: "Affine_3"
    output: "Tanh_2"
  }
  function {
    name: "Mul2_6"
    type: "Mul2"
    repeat_id: "RecurrentInput"
    input: "Tanh_2"
    input: "Sigmoid_5"
    output: "Mul2_6"
  }
  function {
    name: "Add2_2"
    type: "Add2"
    repeat_id: "RecurrentInput"
    input: "Mul2_5"
    input: "Mul2_6"
    output: "Add2_2"
    add2_param {
      inplace: True
    }
  }
  function {
    name: "RecurrentOutput"
    type: "RecurrentOutput"
    input: "Add2_2"
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
  function {
    name: "Affine_2"
    type: "Affine"
    input: "Slice"
    input: "Affine_2/affine/W"
    input: "Affine_2/affine/b"
    output: "Affine_2"
    affine_param {
      base_axis: 1
    }
  }
  function {
    name: "Sigmoid_3"
    type: "Sigmoid"
    input: "Affine_2"
    output: "Sigmoid_3"
  }
  function {
    name: "BinaryCrossEntropy"
    type: "BinaryCrossEntropy"
    input: "Sigmoid_3"
    input: "BinaryCrossEntropy_T"
    output: "BinaryCrossEntropy"
  }
}
network {
  name: "Runtime"
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
    name: "Delay_Initial"
    type: "Buffer"
    shape: { dim:-1 dim: 1 dim: 28 }
  }
  variable {
    name: "Delay_2_Initial"
    type: "Buffer"
    shape: { dim:-1 dim: 1 dim: 28 }
  }
  variable {
    name: "Affine_4/affine/W"
    type: "Parameter"
    shape: { dim: 56 dim: 1 dim: 28 }
    initializer {
      type: "NormalAffineGlorot"
      multiplier: 1
    }
  }
  variable {
    name: "Affine_4/affine/b"
    type: "Parameter"
    shape: { dim: 1 dim: 28 }
    initializer {
      type: "Constant"
      multiplier: 0
    }
  }
  variable {
    name: "Affine_5/affine/W"
    type: "Parameter"
    shape: { dim: 56 dim: 1 dim: 28 }
    initializer {
      type: "NormalAffineGlorot"
      multiplier: 1
    }
  }
  variable {
    name: "Affine_5/affine/b"
    type: "Parameter"
    shape: { dim: 1 dim: 28 }
    initializer {
      type: "Constant"
      multiplier: 0
    }
  }
  variable {
    name: "Affine/affine/W"
    type: "Parameter"
    shape: { dim: 56 dim: 1 dim: 28 }
    initializer {
      type: "NormalAffineGlorot"
      multiplier: 1
    }
  }
  variable {
    name: "Affine/affine/b"
    type: "Parameter"
    shape: { dim: 1 dim: 28 }
    initializer {
      type: "Constant"
      multiplier: 0
    }
  }
  variable {
    name: "Affine_6/affine/W"
    type: "Parameter"
    shape: { dim: 56 dim: 1 dim: 28 }
    initializer {
      type: "NormalAffineGlorot"
      multiplier: 1
    }
  }
  variable {
    name: "Affine_6/affine/b"
    type: "Parameter"
    shape: { dim: 1 dim: 28 }
    initializer {
      type: "Constant"
      multiplier: 0
    }
  }
  variable {
    name: "Affine_7/affine/W"
    type: "Parameter"
    shape: { dim: 56 dim: 1 dim: 28 }
    initializer {
      type: "NormalAffineGlorot"
      multiplier: 1
    }
  }
  variable {
    name: "Affine_7/affine/b"
    type: "Parameter"
    shape: { dim: 1 dim: 28 }
    initializer {
      type: "Constant"
      multiplier: 0
    }
  }
  variable {
    name: "Affine_3/affine/W"
    type: "Parameter"
    shape: { dim: 56 dim: 1 dim: 28 }
    initializer {
      type: "NormalAffineGlorot"
      multiplier: 1
    }
  }
  variable {
    name: "Affine_3/affine/b"
    type: "Parameter"
    shape: { dim: 1 dim: 28 }
    initializer {
      type: "Constant"
      multiplier: 0
    }
  }
  variable {
    name: "Affine_2/affine/W"
    type: "Parameter"
    shape: { dim: 28 dim: 1 }
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
    name: "RecurrentInput"
    type: "Buffer"
    repeat_id: "RecurrentInput"
    shape: { dim:-1 dim: 1 dim: 28 }
  }
  variable {
    name: "Delay"
    type: "Buffer"
    repeat_id: "RecurrentInput"
    shape: { dim:-1 dim: 1 dim: 28 }
  }
  variable {
    name: "Delay_2"
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
    name: "Affine_4"
    type: "Buffer"
    repeat_id: "RecurrentInput"
    shape: { dim:-1 dim: 1 dim: 28 }
  }
  variable {
    name: "Affine_5"
    type: "Buffer"
    repeat_id: "RecurrentInput"
    shape: { dim:-1 dim: 1 dim: 28 }
  }
  variable {
    name: "Sigmoid"
    type: "Buffer"
    repeat_id: "RecurrentInput"
    shape: { dim:-1 dim: 1 dim: 28 }
  }
  variable {
    name: "Sigmoid_2"
    type: "Buffer"
    repeat_id: "RecurrentInput"
    shape: { dim:-1 dim: 1 dim: 28 }
  }
  variable {
    name: "Mul2"
    type: "Buffer"
    repeat_id: "RecurrentInput"
    shape: { dim:-1 dim: 1 dim: 28 }
  }
  variable {
    name: "RSubScalar"
    type: "Buffer"
    repeat_id: "RecurrentInput"
    shape: { dim:-1 dim: 1 dim: 28 }
  }
  variable {
    name: "Concatenate_2"
    type: "Buffer"
    repeat_id: "RecurrentInput"
    shape: { dim:-1 dim: 2 dim: 28 }
  }
  variable {
    name: "Mul2_2"
    type: "Buffer"
    repeat_id: "RecurrentInput"
    shape: { dim:-1 dim: 1 dim: 28 }
  }
  variable {
    name: "Affine"
    type: "Buffer"
    repeat_id: "RecurrentInput"
    shape: { dim:-1 dim: 1 dim: 28 }
  }
  variable {
    name: "Tanh"
    type: "Buffer"
    repeat_id: "RecurrentInput"
    shape: { dim:-1 dim: 1 dim: 28 }
  }
  variable {
    name: "Mul2_3"
    type: "Buffer"
    repeat_id: "RecurrentInput"
    shape: { dim:-1 dim: 1 dim: 28 }
  }
  variable {
    name: "Add2"
    type: "Buffer"
    repeat_id: "RecurrentInput"
    shape: { dim:-1 dim: 1 dim: 28 }
  }
  variable {
    name: "Concatenate_3"
    type: "Buffer"
    repeat_id: "RecurrentInput"
    shape: { dim:-1 dim: 2 dim: 28 }
  }
  variable {
    name: "Affine_6"
    type: "Buffer"
    repeat_id: "RecurrentInput"
    shape: { dim:-1 dim: 1 dim: 28 }
  }
  variable {
    name: "Affine_7"
    type: "Buffer"
    repeat_id: "RecurrentInput"
    shape: { dim:-1 dim: 1 dim: 28 }
  }
  variable {
    name: "Sigmoid_4"
    type: "Buffer"
    repeat_id: "RecurrentInput"
    shape: { dim:-1 dim: 1 dim: 28 }
  }
  variable {
    name: "Sigmoid_5"
    type: "Buffer"
    repeat_id: "RecurrentInput"
    shape: { dim:-1 dim: 1 dim: 28 }
  }
  variable {
    name: "Mul2_4"
    type: "Buffer"
    repeat_id: "RecurrentInput"
    shape: { dim:-1 dim: 1 dim: 28 }
  }
  variable {
    name: "RSubScalar_2"
    type: "Buffer"
    repeat_id: "RecurrentInput"
    shape: { dim:-1 dim: 1 dim: 28 }
  }
  variable {
    name: "Concatenate_4"
    type: "Buffer"
    repeat_id: "RecurrentInput"
    shape: { dim:-1 dim: 2 dim: 28 }
  }
  variable {
    name: "Mul2_5"
    type: "Buffer"
    repeat_id: "RecurrentInput"
    shape: { dim:-1 dim: 1 dim: 28 }
  }
  variable {
    name: "Affine_3"
    type: "Buffer"
    repeat_id: "RecurrentInput"
    shape: { dim:-1 dim: 1 dim: 28 }
  }
  variable {
    name: "Tanh_2"
    type: "Buffer"
    repeat_id: "RecurrentInput"
    shape: { dim:-1 dim: 1 dim: 28 }
  }
  variable {
    name: "Mul2_6"
    type: "Buffer"
    repeat_id: "RecurrentInput"
    shape: { dim:-1 dim: 1 dim: 28 }
  }
  variable {
    name: "Add2_2"
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
  variable {
    name: "rnn_out"
    type: "Buffer"
    shape: { dim:-1 dim: 1 dim: 28 dim: 28 }
  }
  variable {
    name: "Affine_2"
    type: "Buffer"
    shape: { dim:-1 dim: 1 }
  }
  variable {
    name: "y'"
    type: "Buffer"
    shape: { dim:-1 dim: 1 }
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
    name: "Delay"
    type: "Delay"
    repeat_id: "RecurrentInput"
    input: "Add2"
    input: "Delay_Initial"
    output: "Delay"
    recurrent_param {
      repeat_id: "RecurrentInput"
    }
  }
  function {
    name: "Delay_2"
    type: "Delay"
    repeat_id: "RecurrentInput"
    input: "Add2_2"
    input: "Delay_2_Initial"
    output: "Delay_2"
    recurrent_param {
      repeat_id: "RecurrentInput"
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
    name: "Affine_4"
    type: "Affine"
    repeat_id: "RecurrentInput"
    input: "Concatenate"
    input: "Affine_4/affine/W"
    input: "Affine_4/affine/b"
    output: "Affine_4"
    affine_param {
      base_axis: 1
    }
  }
  function {
    name: "Affine_5"
    type: "Affine"
    repeat_id: "RecurrentInput"
    input: "Concatenate"
    input: "Affine_5/affine/W"
    input: "Affine_5/affine/b"
    output: "Affine_5"
    affine_param {
      base_axis: 1
    }
  }
  function {
    name: "Sigmoid"
    type: "Sigmoid"
    repeat_id: "RecurrentInput"
    input: "Affine_4"
    output: "Sigmoid"
  }
  function {
    name: "Sigmoid_2"
    type: "Sigmoid"
    repeat_id: "RecurrentInput"
    input: "Affine_5"
    output: "Sigmoid_2"
  }
  function {
    name: "Mul2"
    type: "Mul2"
    repeat_id: "RecurrentInput"
    input: "Sigmoid"
    input: "Delay"
    output: "Mul2"
  }
  function {
    name: "RSubScalar"
    type: "RSubScalar"
    repeat_id: "RecurrentInput"
    input: "Sigmoid_2"
    output: "RSubScalar"
    r_sub_scalar_param {
      val: 1
    }
  }
  function {
    name: "Concatenate_2"
    type: "Concatenate"
    repeat_id: "RecurrentInput"
    input: "Mul2"
    input: "RecurrentInput"
    output: "Concatenate_2"
    concatenate_param {
      axis: 1
    }
  }
  function {
    name: "Mul2_2"
    type: "Mul2"
    repeat_id: "RecurrentInput"
    input: "RSubScalar"
    input: "Delay"
    output: "Mul2_2"
  }
  function {
    name: "Affine"
    type: "Affine"
    repeat_id: "RecurrentInput"
    input: "Concatenate_2"
    input: "Affine/affine/W"
    input: "Affine/affine/b"
    output: "Affine"
    affine_param {
      base_axis: 1
    }
  }
  function {
    name: "Tanh"
    type: "Tanh"
    repeat_id: "RecurrentInput"
    input: "Affine"
    output: "Tanh"
  }
  function {
    name: "Mul2_3"
    type: "Mul2"
    repeat_id: "RecurrentInput"
    input: "Tanh"
    input: "Sigmoid_2"
    output: "Mul2_3"
  }
  function {
    name: "Add2"
    type: "Add2"
    repeat_id: "RecurrentInput"
    input: "Mul2_2"
    input: "Mul2_3"
    output: "Add2"
    add2_param {
      inplace: True
    }
  }
  function {
    name: "Concatenate_3"
    type: "Concatenate"
    repeat_id: "RecurrentInput"
    input: "Delay_2"
    input: "Add2"
    output: "Concatenate_3"
    concatenate_param {
      axis: 1
    }
  }
  function {
    name: "Affine_6"
    type: "Affine"
    repeat_id: "RecurrentInput"
    input: "Concatenate_3"
    input: "Affine_6/affine/W"
    input: "Affine_6/affine/b"
    output: "Affine_6"
    affine_param {
      base_axis: 1
    }
  }
  function {
    name: "Affine_7"
    type: "Affine"
    repeat_id: "RecurrentInput"
    input: "Concatenate_3"
    input: "Affine_7/affine/W"
    input: "Affine_7/affine/b"
    output: "Affine_7"
    affine_param {
      base_axis: 1
    }
  }
  function {
    name: "Sigmoid_4"
    type: "Sigmoid"
    repeat_id: "RecurrentInput"
    input: "Affine_6"
    output: "Sigmoid_4"
  }
  function {
    name: "Sigmoid_5"
    type: "Sigmoid"
    repeat_id: "RecurrentInput"
    input: "Affine_7"
    output: "Sigmoid_5"
  }
  function {
    name: "Mul2_4"
    type: "Mul2"
    repeat_id: "RecurrentInput"
    input: "Sigmoid_4"
    input: "Delay_2"
    output: "Mul2_4"
  }
  function {
    name: "RSubScalar_2"
    type: "RSubScalar"
    repeat_id: "RecurrentInput"
    input: "Sigmoid_5"
    output: "RSubScalar_2"
    r_sub_scalar_param {
      val: 1
    }
  }
  function {
    name: "Concatenate_4"
    type: "Concatenate"
    repeat_id: "RecurrentInput"
    input: "Mul2_4"
    input: "Add2"
    output: "Concatenate_4"
    concatenate_param {
      axis: 1
    }
  }
  function {
    name: "Mul2_5"
    type: "Mul2"
    repeat_id: "RecurrentInput"
    input: "RSubScalar_2"
    input: "Delay_2"
    output: "Mul2_5"
  }
  function {
    name: "Affine_3"
    type: "Affine"
    repeat_id: "RecurrentInput"
    input: "Concatenate_4"
    input: "Affine_3/affine/W"
    input: "Affine_3/affine/b"
    output: "Affine_3"
    affine_param {
      base_axis: 1
    }
  }
  function {
    name: "Tanh_2"
    type: "Tanh"
    repeat_id: "RecurrentInput"
    input: "Affine_3"
    output: "Tanh_2"
  }
  function {
    name: "Mul2_6"
    type: "Mul2"
    repeat_id: "RecurrentInput"
    input: "Tanh_2"
    input: "Sigmoid_5"
    output: "Mul2_6"
  }
  function {
    name: "Add2_2"
    type: "Add2"
    repeat_id: "RecurrentInput"
    input: "Mul2_5"
    input: "Mul2_6"
    output: "Add2_2"
    add2_param {
      inplace: True
    }
  }
  function {
    name: "RecurrentOutput"
    type: "RecurrentOutput"
    input: "Add2_2"
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
  function {
    name: "rnn_out"
    type: "Identity"
    input: "RecurrentOutput"
    output: "rnn_out"
  }
  function {
    name: "Affine_2"
    type: "Affine"
    input: "Slice"
    input: "Affine_2/affine/W"
    input: "Affine_2/affine/b"
    output: "Affine_2"
    affine_param {
      base_axis: 1
    }
  }
  function {
    name: "y'"
    type: "Sigmoid"
    input: "Affine_2"
    output: "y'"
  }
}
dataset {
  name: "Training"
  uri: "C:\\Utils\\neural_network_console_internal_v120\\samples\\sample_dataset\\mnist\\small_mnist_4or9_training.csv"
  cache_dir: "C:\\Utils\\neural_network_console_internal_v120\\samples\\sample_dataset\\mnist\\small_mnist_4or9_training.cache"
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
  network_name: "Main_"
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
    multiplier: 0
  }
  generator_variable {
    variable_name: "Delay_2_Initial"
    type: "Constant"
    multiplier: 0
  }
  data_variable {
    variable_name: "BinaryCrossEntropy_T"
    data_name: "y"
  }
  loss_variable {
    variable_name: "BinaryCrossEntropy"
  }
  parameter_variable {
    variable_name: "Affine_4/affine/W"
    learning_rate_multiplier: 1
  }
  parameter_variable {
    variable_name: "Affine_4/affine/b"
    learning_rate_multiplier: 1
  }
  parameter_variable {
    variable_name: "Affine_5/affine/W"
    learning_rate_multiplier: 1
  }
  parameter_variable {
    variable_name: "Affine_5/affine/b"
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
    variable_name: "Affine_6/affine/W"
    learning_rate_multiplier: 1
  }
  parameter_variable {
    variable_name: "Affine_6/affine/b"
    learning_rate_multiplier: 1
  }
  parameter_variable {
    variable_name: "Affine_7/affine/W"
    learning_rate_multiplier: 1
  }
  parameter_variable {
    variable_name: "Affine_7/affine/b"
    learning_rate_multiplier: 1
  }
  parameter_variable {
    variable_name: "Affine_3/affine/W"
    learning_rate_multiplier: 1
  }
  parameter_variable {
    variable_name: "Affine_3/affine/b"
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
  network_name: "Main_"
  dataset_name: "Training"
  data_variable {
    variable_name: "Input"
    data_name: "x"
  }
  generator_variable {
    variable_name: "Delay_Initial"
    type: "Constant"
    multiplier: 0
  }
  generator_variable {
    variable_name: "Delay_2_Initial"
    type: "Constant"
    multiplier: 0
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
  network_name: "Main_"
  dataset_name: "Validation"
  data_variable {
    variable_name: "Input"
    data_name: "x"
  }
  generator_variable {
    variable_name: "Delay_Initial"
    type: "Constant"
    multiplier: 0
  }
  generator_variable {
    variable_name: "Delay_2_Initial"
    type: "Constant"
    multiplier: 0
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
  network_name: "Runtime"
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
    multiplier: 0
  }
  generator_variable {
    variable_name: "Delay_2_Initial"
    type: "Constant"
    multiplier: 0
  }
  output_variable {
    variable_name: "rnn_out"
    data_name: "rnn_out"
  }
  output_variable {
    variable_name: "y'"
    data_name: "y'"
  }
  parameter_variable {
    variable_name: "Affine_4/affine/W"
  }
  parameter_variable {
    variable_name: "Affine_4/affine/b"
  }
  parameter_variable {
    variable_name: "Affine_5/affine/W"
  }
  parameter_variable {
    variable_name: "Affine_5/affine/b"
  }
  parameter_variable {
    variable_name: "Affine/affine/W"
  }
  parameter_variable {
    variable_name: "Affine/affine/b"
  }
  parameter_variable {
    variable_name: "Affine_6/affine/W"
  }
  parameter_variable {
    variable_name: "Affine_6/affine/b"
  }
  parameter_variable {
    variable_name: "Affine_7/affine/W"
  }
  parameter_variable {
    variable_name: "Affine_7/affine/b"
  }
  parameter_variable {
    variable_name: "Affine_3/affine/W"
  }
  parameter_variable {
    variable_name: "Affine_3/affine/b"
  }
  parameter_variable {
    variable_name: "Affine_2/affine/W"
  }
  parameter_variable {
    variable_name: "Affine_2/affine/b"
  }
}
'''
# The nntxt is generated from /home/woody/aid/nnabla-sample-data/sample_project/tutorial/recurrent_neural_networks/LSTM_auto_encoder.files/20170804_142847/net.nntxt.
N00000010 = r'''global_config {
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
  repeat_info {
    id: "RecurrentInput_2"
    times: 28
  }
  variable {
    name: "Input"
    type: "Buffer"
    shape: { dim:-1 dim: 1 dim: 28 dim: 28 }
  }
  variable {
    name: "Flip"
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
    name: "Affine_2/affine/W"
    type: "Parameter"
    shape: { dim: 284 dim: 1 dim: 256 }
    initializer {
      type: "NormalAffineGlorot"
      multiplier: 1.0
    }
  }
  variable {
    name: "Affine_2/affine/b"
    type: "Parameter"
    shape: { dim: 1 dim: 256 }
    initializer {
      type: "Constant"
      multiplier: 0.0
    }
  }
  variable {
    name: "Affine_4/affine/W"
    type: "Parameter"
    shape: { dim: 284 dim: 1 dim: 256 }
    initializer {
      type: "NormalAffineGlorot"
      multiplier: 1.0
    }
  }
  variable {
    name: "Affine_4/affine/b"
    type: "Parameter"
    shape: { dim: 1 dim: 256 }
    initializer {
      type: "Constant"
      multiplier: 0.0
    }
  }
  variable {
    name: "Affine_5/affine/W"
    type: "Parameter"
    shape: { dim: 284 dim: 1 dim: 256 }
    initializer {
      type: "NormalAffineGlorot"
      multiplier: 1.0
    }
  }
  variable {
    name: "Affine_5/affine/b"
    type: "Parameter"
    shape: { dim: 1 dim: 256 }
    initializer {
      type: "Constant"
      multiplier: 0.0
    }
  }
  variable {
    name: "Affine_6/affine/W"
    type: "Parameter"
    shape: { dim: 284 dim: 1 dim: 256 }
    initializer {
      type: "NormalAffineGlorot"
      multiplier: 1.0
    }
  }
  variable {
    name: "Affine_6/affine/b"
    type: "Parameter"
    shape: { dim: 1 dim: 256 }
    initializer {
      type: "Constant"
      multiplier: 0.0
    }
  }
  variable {
    name: "Delay_Initial"
    type: "Buffer"
    shape: { dim:-1 dim: 1 dim: 256 }
  }
  variable {
    name: "Delay_2_Initial"
    type: "Buffer"
    shape: { dim:-1 dim: 1 dim: 256 }
  }
  variable {
    name: "Affine_7/affine/W"
    type: "Parameter"
    shape: { dim: 284 dim: 1 dim: 28 }
    initializer {
      type: "NormalAffineGlorot"
      multiplier: 1.0
    }
  }
  variable {
    name: "Affine_7/affine/b"
    type: "Parameter"
    shape: { dim: 1 dim: 28 }
    initializer {
      type: "Constant"
      multiplier: 0.0
    }
  }
  variable {
    name: "Affine_8/affine/W"
    type: "Parameter"
    shape: { dim: 284 dim: 1 dim: 28 }
    initializer {
      type: "NormalAffineGlorot"
      multiplier: 1.0
    }
  }
  variable {
    name: "Affine_8/affine/b"
    type: "Parameter"
    shape: { dim: 1 dim: 28 }
    initializer {
      type: "Constant"
      multiplier: 0.0
    }
  }
  variable {
    name: "Affine_9/affine/W"
    type: "Parameter"
    shape: { dim: 284 dim: 1 dim: 28 }
    initializer {
      type: "NormalAffineGlorot"
      multiplier: 1.0
    }
  }
  variable {
    name: "Affine_9/affine/b"
    type: "Parameter"
    shape: { dim: 1 dim: 28 }
    initializer {
      type: "Constant"
      multiplier: 0.0
    }
  }
  variable {
    name: "Affine_10/affine/W"
    type: "Parameter"
    shape: { dim: 284 dim: 1 dim: 28 }
    initializer {
      type: "NormalAffineGlorot"
      multiplier: 1.0
    }
  }
  variable {
    name: "Affine_10/affine/b"
    type: "Parameter"
    shape: { dim: 1 dim: 28 }
    initializer {
      type: "Constant"
      multiplier: 0.0
    }
  }
  variable {
    name: "Delay_3_Initial"
    type: "Buffer"
    shape: { dim:-1 dim: 1 dim: 28 }
  }
  variable {
    name: "Delay_4_Initial"
    type: "Buffer"
    shape: { dim:-1 dim: 1 dim: 28 }
  }
  variable {
    name: "Convolution/conv/W"
    type: "Parameter"
    shape: { dim: 1 dim: 1 dim: 1 dim: 1 }
    initializer {
      type: "NormalConvolutionGlorot"
      multiplier: 1.0
    }
  }
  variable {
    name: "Convolution/conv/b"
    type: "Parameter"
    shape: { dim: 1 }
    initializer {
      type: "Constant"
      multiplier: 0.0
    }
  }
  variable {
    name: "SquaredError_T"
    type: "Buffer"
    shape: { dim:-1 dim: 1 dim: 28 dim: 28 }
  }
  variable {
    name: "Affine_2"
    type: "Buffer"
    repeat_id: "RecurrentInput"
    shape: { dim:-1 dim: 1 dim: 256 }
  }
  variable {
    name: "Tanh"
    type: "Buffer"
    repeat_id: "RecurrentInput"
    shape: { dim:-1 dim: 1 dim: 256 }
  }
  variable {
    name: "Affine_4"
    type: "Buffer"
    repeat_id: "RecurrentInput"
    shape: { dim:-1 dim: 1 dim: 256 }
  }
  variable {
    name: "Sigmoid"
    type: "Buffer"
    repeat_id: "RecurrentInput"
    shape: { dim:-1 dim: 1 dim: 256 }
  }
  variable {
    name: "Concatenate"
    type: "Buffer"
    repeat_id: "RecurrentInput"
    shape: { dim:-1 dim: 1 dim: 284 }
  }
  variable {
    name: "Affine_5"
    type: "Buffer"
    repeat_id: "RecurrentInput"
    shape: { dim:-1 dim: 1 dim: 256 }
  }
  variable {
    name: "Sigmoid_2"
    type: "Buffer"
    repeat_id: "RecurrentInput"
    shape: { dim:-1 dim: 1 dim: 256 }
  }
  variable {
    name: "Tanh_2"
    type: "Buffer"
    repeat_id: "RecurrentInput"
    shape: { dim:-1 dim: 1 dim: 256 }
  }
  variable {
    name: "Affine_6"
    type: "Buffer"
    repeat_id: "RecurrentInput"
    shape: { dim:-1 dim: 1 dim: 256 }
  }
  variable {
    name: "Sigmoid_3"
    type: "Buffer"
    repeat_id: "RecurrentInput"
    shape: { dim:-1 dim: 1 dim: 256 }
  }
  variable {
    name: "Delay"
    type: "Buffer"
    repeat_id: "RecurrentInput"
    shape: { dim:-1 dim: 1 dim: 256 }
  }
  variable {
    name: "Delay_2"
    type: "Buffer"
    repeat_id: "RecurrentInput"
    shape: { dim:-1 dim: 1 dim: 256 }
  }
  variable {
    name: "Sigmoid_4"
    type: "Buffer"
    shape: { dim:-1 dim: 1 dim: 28 dim: 28 }
  }
  variable {
    name: "Affine_7"
    type: "Buffer"
    repeat_id: "RecurrentInput_2"
    shape: { dim:-1 dim: 1 dim: 28 }
  }
  variable {
    name: "Tanh_3"
    type: "Buffer"
    repeat_id: "RecurrentInput_2"
    shape: { dim:-1 dim: 1 dim: 28 }
  }
  variable {
    name: "Affine_8"
    type: "Buffer"
    repeat_id: "RecurrentInput_2"
    shape: { dim:-1 dim: 1 dim: 28 }
  }
  variable {
    name: "Sigmoid_5"
    type: "Buffer"
    repeat_id: "RecurrentInput_2"
    shape: { dim:-1 dim: 1 dim: 28 }
  }
  variable {
    name: "Concatenate_2"
    type: "Buffer"
    repeat_id: "RecurrentInput_2"
    shape: { dim:-1 dim: 1 dim: 284 }
  }
  variable {
    name: "Affine_9"
    type: "Buffer"
    repeat_id: "RecurrentInput_2"
    shape: { dim:-1 dim: 1 dim: 28 }
  }
  variable {
    name: "Sigmoid_6"
    type: "Buffer"
    repeat_id: "RecurrentInput_2"
    shape: { dim:-1 dim: 1 dim: 28 }
  }
  variable {
    name: "Tanh_4"
    type: "Buffer"
    repeat_id: "RecurrentInput_2"
    shape: { dim:-1 dim: 1 dim: 28 }
  }
  variable {
    name: "Affine_10"
    type: "Buffer"
    repeat_id: "RecurrentInput_2"
    shape: { dim:-1 dim: 1 dim: 28 }
  }
  variable {
    name: "Sigmoid_7"
    type: "Buffer"
    repeat_id: "RecurrentInput_2"
    shape: { dim:-1 dim: 1 dim: 28 }
  }
  variable {
    name: "Delay_3"
    type: "Buffer"
    repeat_id: "RecurrentInput_2"
    shape: { dim:-1 dim: 1 dim: 28 }
  }
  variable {
    name: "Delay_4"
    type: "Buffer"
    repeat_id: "RecurrentInput_2"
    shape: { dim:-1 dim: 1 dim: 28 }
  }
  variable {
    name: "RecurrentInput_2"
    type: "Buffer"
    repeat_id: "RecurrentInput_2"
    shape: { dim:-1 dim: 1 dim: 256 }
  }
  variable {
    name: "Convolution"
    type: "Buffer"
    shape: { dim:-1 dim: 1 dim: 28 dim: 28 }
  }
  variable {
    name: "Mul2"
    type: "Buffer"
    repeat_id: "RecurrentInput"
    shape: { dim:-1 dim: 1 dim: 256 }
  }
  variable {
    name: "Add2"
    type: "Buffer"
    repeat_id: "RecurrentInput"
    shape: { dim:-1 dim: 1 dim: 256 }
  }
  variable {
    name: "Mul2_2"
    type: "Buffer"
    repeat_id: "RecurrentInput"
    shape: { dim:-1 dim: 1 dim: 256 }
  }
  variable {
    name: "RecurrentOutput"
    type: "Buffer"
    shape: { dim:-1 dim: 1 dim: 28 dim: 256 }
  }
  variable {
    name: "Mul2_3"
    type: "Buffer"
    repeat_id: "RecurrentInput"
    shape: { dim:-1 dim: 1 dim: 256 }
  }
  variable {
    name: "Mul2_4"
    type: "Buffer"
    repeat_id: "RecurrentInput_2"
    shape: { dim:-1 dim: 1 dim: 28 }
  }
  variable {
    name: "Add2_2"
    type: "Buffer"
    repeat_id: "RecurrentInput_2"
    shape: { dim:-1 dim: 1 dim: 28 }
  }
  variable {
    name: "Mul2_5"
    type: "Buffer"
    repeat_id: "RecurrentInput_2"
    shape: { dim:-1 dim: 1 dim: 28 }
  }
  variable {
    name: "Mul2_6"
    type: "Buffer"
    repeat_id: "RecurrentInput_2"
    shape: { dim:-1 dim: 1 dim: 28 }
  }
  variable {
    name: "RecurrentOutput_2"
    type: "Buffer"
    shape: { dim:-1 dim: 1 dim: 28 dim: 28 }
  }
  variable {
    name: "SquaredError"
    type: "Buffer"
    shape: { dim:-1 dim: 1 dim: 28 dim: 28 }
  }
  variable {
    name: "Slice"
    type: "Buffer"
    shape: { dim:-1 dim: 1 dim: 1 dim: 256 }
  }
  variable {
    name: "Unpooling"
    type: "Buffer"
    shape: { dim:-1 dim: 1 dim: 28 dim: 256 }
  }
  function {
    name: "Flip"
    type: "Flip"
    input: "Input"
    output: "Flip"
    flip_param {
      axes: 2
    }
  }
  function {
    name: "RecurrentInput"
    type: "RecurrentInput"
    repeat_id: "RecurrentInput"
    input: "Flip"
    output: "RecurrentInput"
    recurrent_param {
      axis: 2
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
    name: "Tanh"
    type: "Tanh"
    repeat_id: "RecurrentInput"
    input: "Affine_2"
    output: "Tanh"
  }
  function {
    name: "Affine_4"
    type: "Affine"
    repeat_id: "RecurrentInput"
    input: "Concatenate"
    input: "Affine_4/affine/W"
    input: "Affine_4/affine/b"
    output: "Affine_4"
    affine_param {
      base_axis: 1
    }
  }
  function {
    name: "Sigmoid"
    type: "Sigmoid"
    repeat_id: "RecurrentInput"
    input: "Affine_4"
    output: "Sigmoid"
  }
  function {
    name: "Concatenate"
    type: "Concatenate"
    repeat_id: "RecurrentInput"
    input: "Delay_2"
    input: "RecurrentInput"
    output: "Concatenate"
    concatenate_param {
      axis: 2
    }
  }
  function {
    name: "Affine_5"
    type: "Affine"
    repeat_id: "RecurrentInput"
    input: "Concatenate"
    input: "Affine_5/affine/W"
    input: "Affine_5/affine/b"
    output: "Affine_5"
    affine_param {
      base_axis: 1
    }
  }
  function {
    name: "Sigmoid_2"
    type: "Sigmoid"
    repeat_id: "RecurrentInput"
    input: "Affine_5"
    output: "Sigmoid_2"
  }
  function {
    name: "Tanh_2"
    type: "Tanh"
    repeat_id: "RecurrentInput"
    input: "Add2"
    output: "Tanh_2"
  }
  function {
    name: "Affine_6"
    type: "Affine"
    repeat_id: "RecurrentInput"
    input: "Concatenate"
    input: "Affine_6/affine/W"
    input: "Affine_6/affine/b"
    output: "Affine_6"
    affine_param {
      base_axis: 1
    }
  }
  function {
    name: "Sigmoid_3"
    type: "Sigmoid"
    repeat_id: "RecurrentInput"
    input: "Affine_6"
    output: "Sigmoid_3"
  }
  function {
    name: "Delay"
    type: "Delay"
    repeat_id: "RecurrentInput"
    input: "Add2"
    input: "Delay_Initial"
    output: "Delay"
    recurrent_param {
      repeat_id: "RecurrentInput"
    }
  }
  function {
    name: "Delay_2"
    type: "Delay"
    repeat_id: "RecurrentInput"
    input: "Mul2_2"
    input: "Delay_2_Initial"
    output: "Delay_2"
    recurrent_param {
      repeat_id: "RecurrentInput"
    }
  }
  function {
    name: "Sigmoid_4"
    type: "Sigmoid"
    input: "Convolution"
    output: "Sigmoid_4"
  }
  function {
    name: "Affine_7"
    type: "Affine"
    repeat_id: "RecurrentInput_2"
    input: "Concatenate_2"
    input: "Affine_7/affine/W"
    input: "Affine_7/affine/b"
    output: "Affine_7"
    affine_param {
      base_axis: 1
    }
  }
  function {
    name: "Tanh_3"
    type: "Tanh"
    repeat_id: "RecurrentInput_2"
    input: "Affine_7"
    output: "Tanh_3"
  }
  function {
    name: "Affine_8"
    type: "Affine"
    repeat_id: "RecurrentInput_2"
    input: "Concatenate_2"
    input: "Affine_8/affine/W"
    input: "Affine_8/affine/b"
    output: "Affine_8"
    affine_param {
      base_axis: 1
    }
  }
  function {
    name: "Sigmoid_5"
    type: "Sigmoid"
    repeat_id: "RecurrentInput_2"
    input: "Affine_8"
    output: "Sigmoid_5"
  }
  function {
    name: "Concatenate_2"
    type: "Concatenate"
    repeat_id: "RecurrentInput_2"
    input: "Delay_4"
    input: "RecurrentInput_2"
    output: "Concatenate_2"
    concatenate_param {
      axis: 2
    }
  }
  function {
    name: "Affine_9"
    type: "Affine"
    repeat_id: "RecurrentInput_2"
    input: "Concatenate_2"
    input: "Affine_9/affine/W"
    input: "Affine_9/affine/b"
    output: "Affine_9"
    affine_param {
      base_axis: 1
    }
  }
  function {
    name: "Sigmoid_6"
    type: "Sigmoid"
    repeat_id: "RecurrentInput_2"
    input: "Affine_9"
    output: "Sigmoid_6"
  }
  function {
    name: "Tanh_4"
    type: "Tanh"
    repeat_id: "RecurrentInput_2"
    input: "Add2_2"
    output: "Tanh_4"
  }
  function {
    name: "Affine_10"
    type: "Affine"
    repeat_id: "RecurrentInput_2"
    input: "Concatenate_2"
    input: "Affine_10/affine/W"
    input: "Affine_10/affine/b"
    output: "Affine_10"
    affine_param {
      base_axis: 1
    }
  }
  function {
    name: "Sigmoid_7"
    type: "Sigmoid"
    repeat_id: "RecurrentInput_2"
    input: "Affine_10"
    output: "Sigmoid_7"
  }
  function {
    name: "Delay_3"
    type: "Delay"
    repeat_id: "RecurrentInput_2"
    input: "Add2_2"
    input: "Delay_3_Initial"
    output: "Delay_3"
    recurrent_param {
      repeat_id: "RecurrentInput_2"
    }
  }
  function {
    name: "Delay_4"
    type: "Delay"
    repeat_id: "RecurrentInput_2"
    input: "Mul2_6"
    input: "Delay_4_Initial"
    output: "Delay_4"
    recurrent_param {
      repeat_id: "RecurrentInput_2"
    }
  }
  function {
    name: "RecurrentInput_2"
    type: "RecurrentInput"
    repeat_id: "RecurrentInput_2"
    input: "Unpooling"
    output: "RecurrentInput_2"
    recurrent_param {
      axis: 2
      repeat_id: "RecurrentInput_2"
    }
  }
  function {
    name: "Convolution"
    type: "Convolution"
    input: "RecurrentOutput_2"
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
    name: "Mul2"
    type: "Mul2"
    repeat_id: "RecurrentInput"
    input: "Delay"
    input: "Sigmoid_2"
    output: "Mul2"
  }
  function {
    name: "Add2"
    type: "Add2"
    repeat_id: "RecurrentInput"
    input: "Mul2"
    input: "Mul2_3"
    output: "Add2"
    add2_param {
      inplace: True
    }
  }
  function {
    name: "Mul2_2"
    type: "Mul2"
    repeat_id: "RecurrentInput"
    input: "Tanh_2"
    input: "Sigmoid_3"
    output: "Mul2_2"
  }
  function {
    name: "RecurrentOutput"
    type: "RecurrentOutput"
    input: "Mul2_2"
    output: "RecurrentOutput"
    recurrent_param {
      axis: 2
      repeat_id: "RecurrentInput"
      length: 28
    }
  }
  function {
    name: "Mul2_3"
    type: "Mul2"
    repeat_id: "RecurrentInput"
    input: "Tanh"
    input: "Sigmoid"
    output: "Mul2_3"
  }
  function {
    name: "Mul2_4"
    type: "Mul2"
    repeat_id: "RecurrentInput_2"
    input: "Sigmoid_6"
    input: "Delay_3"
    output: "Mul2_4"
  }
  function {
    name: "Add2_2"
    type: "Add2"
    repeat_id: "RecurrentInput_2"
    input: "Mul2_4"
    input: "Mul2_5"
    output: "Add2_2"
    add2_param {
      inplace: True
    }
  }
  function {
    name: "Mul2_5"
    type: "Mul2"
    repeat_id: "RecurrentInput_2"
    input: "Tanh_3"
    input: "Sigmoid_5"
    output: "Mul2_5"
  }
  function {
    name: "Mul2_6"
    type: "Mul2"
    repeat_id: "RecurrentInput_2"
    input: "Tanh_4"
    input: "Sigmoid_7"
    output: "Mul2_6"
  }
  function {
    name: "RecurrentOutput_2"
    type: "RecurrentOutput"
    input: "Mul2_6"
    output: "RecurrentOutput_2"
    recurrent_param {
      axis: 2
      repeat_id: "RecurrentInput_2"
      length: 28
    }
  }
  function {
    name: "SquaredError"
    type: "SquaredError"
    input: "Sigmoid_4"
    input: "SquaredError_T"
    output: "SquaredError"
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
      stop: 256
      step: 1
      step: 1
      step: 1
    }
  }
  function {
    name: "Unpooling"
    type: "Unpooling"
    input: "Slice"
    output: "Unpooling"
    unpooling_param {
      kernel: { dim: 28 dim: 1 }
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
  repeat_info {
    id: "RecurrentInput_2"
    times: 28
  }
  variable {
    name: "Input"
    type: "Buffer"
    shape: { dim:-1 dim: 1 dim: 28 dim: 28 }
  }
  variable {
    name: "Flip"
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
    name: "Affine_2/affine/W"
    type: "Parameter"
    shape: { dim: 284 dim: 1 dim: 256 }
    initializer {
      type: "NormalAffineGlorot"
      multiplier: 1.0
    }
  }
  variable {
    name: "Affine_2/affine/b"
    type: "Parameter"
    shape: { dim: 1 dim: 256 }
    initializer {
      type: "Constant"
      multiplier: 0.0
    }
  }
  variable {
    name: "Affine_4/affine/W"
    type: "Parameter"
    shape: { dim: 284 dim: 1 dim: 256 }
    initializer {
      type: "NormalAffineGlorot"
      multiplier: 1.0
    }
  }
  variable {
    name: "Affine_4/affine/b"
    type: "Parameter"
    shape: { dim: 1 dim: 256 }
    initializer {
      type: "Constant"
      multiplier: 0.0
    }
  }
  variable {
    name: "Affine_5/affine/W"
    type: "Parameter"
    shape: { dim: 284 dim: 1 dim: 256 }
    initializer {
      type: "NormalAffineGlorot"
      multiplier: 1.0
    }
  }
  variable {
    name: "Affine_5/affine/b"
    type: "Parameter"
    shape: { dim: 1 dim: 256 }
    initializer {
      type: "Constant"
      multiplier: 0.0
    }
  }
  variable {
    name: "Affine_6/affine/W"
    type: "Parameter"
    shape: { dim: 284 dim: 1 dim: 256 }
    initializer {
      type: "NormalAffineGlorot"
      multiplier: 1.0
    }
  }
  variable {
    name: "Affine_6/affine/b"
    type: "Parameter"
    shape: { dim: 1 dim: 256 }
    initializer {
      type: "Constant"
      multiplier: 0.0
    }
  }
  variable {
    name: "Delay_Initial"
    type: "Buffer"
    shape: { dim:-1 dim: 1 dim: 256 }
  }
  variable {
    name: "Delay_2_Initial"
    type: "Buffer"
    shape: { dim:-1 dim: 1 dim: 256 }
  }
  variable {
    name: "Affine_7/affine/W"
    type: "Parameter"
    shape: { dim: 284 dim: 1 dim: 28 }
    initializer {
      type: "NormalAffineGlorot"
      multiplier: 1.0
    }
  }
  variable {
    name: "Affine_7/affine/b"
    type: "Parameter"
    shape: { dim: 1 dim: 28 }
    initializer {
      type: "Constant"
      multiplier: 0.0
    }
  }
  variable {
    name: "Affine_8/affine/W"
    type: "Parameter"
    shape: { dim: 284 dim: 1 dim: 28 }
    initializer {
      type: "NormalAffineGlorot"
      multiplier: 1.0
    }
  }
  variable {
    name: "Affine_8/affine/b"
    type: "Parameter"
    shape: { dim: 1 dim: 28 }
    initializer {
      type: "Constant"
      multiplier: 0.0
    }
  }
  variable {
    name: "Affine_9/affine/W"
    type: "Parameter"
    shape: { dim: 284 dim: 1 dim: 28 }
    initializer {
      type: "NormalAffineGlorot"
      multiplier: 1.0
    }
  }
  variable {
    name: "Affine_9/affine/b"
    type: "Parameter"
    shape: { dim: 1 dim: 28 }
    initializer {
      type: "Constant"
      multiplier: 0.0
    }
  }
  variable {
    name: "Affine_10/affine/W"
    type: "Parameter"
    shape: { dim: 284 dim: 1 dim: 28 }
    initializer {
      type: "NormalAffineGlorot"
      multiplier: 1.0
    }
  }
  variable {
    name: "Affine_10/affine/b"
    type: "Parameter"
    shape: { dim: 1 dim: 28 }
    initializer {
      type: "Constant"
      multiplier: 0.0
    }
  }
  variable {
    name: "Delay_3_Initial"
    type: "Buffer"
    shape: { dim:-1 dim: 1 dim: 28 }
  }
  variable {
    name: "Delay_4_Initial"
    type: "Buffer"
    shape: { dim:-1 dim: 1 dim: 28 }
  }
  variable {
    name: "Convolution/conv/W"
    type: "Parameter"
    shape: { dim: 1 dim: 1 dim: 1 dim: 1 }
    initializer {
      type: "NormalConvolutionGlorot"
      multiplier: 1.0
    }
  }
  variable {
    name: "Convolution/conv/b"
    type: "Parameter"
    shape: { dim: 1 }
    initializer {
      type: "Constant"
      multiplier: 0.0
    }
  }
  variable {
    name: "SquaredError_T"
    type: "Buffer"
    shape: { dim:-1 dim: 1 dim: 28 dim: 28 }
  }
  variable {
    name: "Affine_2"
    type: "Buffer"
    repeat_id: "RecurrentInput"
    shape: { dim:-1 dim: 1 dim: 256 }
  }
  variable {
    name: "Tanh"
    type: "Buffer"
    repeat_id: "RecurrentInput"
    shape: { dim:-1 dim: 1 dim: 256 }
  }
  variable {
    name: "Affine_4"
    type: "Buffer"
    repeat_id: "RecurrentInput"
    shape: { dim:-1 dim: 1 dim: 256 }
  }
  variable {
    name: "Sigmoid"
    type: "Buffer"
    repeat_id: "RecurrentInput"
    shape: { dim:-1 dim: 1 dim: 256 }
  }
  variable {
    name: "Concatenate"
    type: "Buffer"
    repeat_id: "RecurrentInput"
    shape: { dim:-1 dim: 1 dim: 284 }
  }
  variable {
    name: "Affine_5"
    type: "Buffer"
    repeat_id: "RecurrentInput"
    shape: { dim:-1 dim: 1 dim: 256 }
  }
  variable {
    name: "Sigmoid_2"
    type: "Buffer"
    repeat_id: "RecurrentInput"
    shape: { dim:-1 dim: 1 dim: 256 }
  }
  variable {
    name: "Tanh_2"
    type: "Buffer"
    repeat_id: "RecurrentInput"
    shape: { dim:-1 dim: 1 dim: 256 }
  }
  variable {
    name: "Affine_6"
    type: "Buffer"
    repeat_id: "RecurrentInput"
    shape: { dim:-1 dim: 1 dim: 256 }
  }
  variable {
    name: "Sigmoid_3"
    type: "Buffer"
    repeat_id: "RecurrentInput"
    shape: { dim:-1 dim: 1 dim: 256 }
  }
  variable {
    name: "Delay"
    type: "Buffer"
    repeat_id: "RecurrentInput"
    shape: { dim:-1 dim: 1 dim: 256 }
  }
  variable {
    name: "Delay_2"
    type: "Buffer"
    repeat_id: "RecurrentInput"
    shape: { dim:-1 dim: 1 dim: 256 }
  }
  variable {
    name: "Sigmoid_4"
    type: "Buffer"
    shape: { dim:-1 dim: 1 dim: 28 dim: 28 }
  }
  variable {
    name: "Affine_7"
    type: "Buffer"
    repeat_id: "RecurrentInput_2"
    shape: { dim:-1 dim: 1 dim: 28 }
  }
  variable {
    name: "Tanh_3"
    type: "Buffer"
    repeat_id: "RecurrentInput_2"
    shape: { dim:-1 dim: 1 dim: 28 }
  }
  variable {
    name: "Affine_8"
    type: "Buffer"
    repeat_id: "RecurrentInput_2"
    shape: { dim:-1 dim: 1 dim: 28 }
  }
  variable {
    name: "Sigmoid_5"
    type: "Buffer"
    repeat_id: "RecurrentInput_2"
    shape: { dim:-1 dim: 1 dim: 28 }
  }
  variable {
    name: "Concatenate_2"
    type: "Buffer"
    repeat_id: "RecurrentInput_2"
    shape: { dim:-1 dim: 1 dim: 284 }
  }
  variable {
    name: "Affine_9"
    type: "Buffer"
    repeat_id: "RecurrentInput_2"
    shape: { dim:-1 dim: 1 dim: 28 }
  }
  variable {
    name: "Sigmoid_6"
    type: "Buffer"
    repeat_id: "RecurrentInput_2"
    shape: { dim:-1 dim: 1 dim: 28 }
  }
  variable {
    name: "Tanh_4"
    type: "Buffer"
    repeat_id: "RecurrentInput_2"
    shape: { dim:-1 dim: 1 dim: 28 }
  }
  variable {
    name: "Affine_10"
    type: "Buffer"
    repeat_id: "RecurrentInput_2"
    shape: { dim:-1 dim: 1 dim: 28 }
  }
  variable {
    name: "Sigmoid_7"
    type: "Buffer"
    repeat_id: "RecurrentInput_2"
    shape: { dim:-1 dim: 1 dim: 28 }
  }
  variable {
    name: "Delay_3"
    type: "Buffer"
    repeat_id: "RecurrentInput_2"
    shape: { dim:-1 dim: 1 dim: 28 }
  }
  variable {
    name: "Delay_4"
    type: "Buffer"
    repeat_id: "RecurrentInput_2"
    shape: { dim:-1 dim: 1 dim: 28 }
  }
  variable {
    name: "RecurrentInput_2"
    type: "Buffer"
    repeat_id: "RecurrentInput_2"
    shape: { dim:-1 dim: 1 dim: 256 }
  }
  variable {
    name: "Convolution"
    type: "Buffer"
    shape: { dim:-1 dim: 1 dim: 28 dim: 28 }
  }
  variable {
    name: "Mul2"
    type: "Buffer"
    repeat_id: "RecurrentInput"
    shape: { dim:-1 dim: 1 dim: 256 }
  }
  variable {
    name: "Add2"
    type: "Buffer"
    repeat_id: "RecurrentInput"
    shape: { dim:-1 dim: 1 dim: 256 }
  }
  variable {
    name: "Mul2_2"
    type: "Buffer"
    repeat_id: "RecurrentInput"
    shape: { dim:-1 dim: 1 dim: 256 }
  }
  variable {
    name: "RecurrentOutput"
    type: "Buffer"
    shape: { dim:-1 dim: 1 dim: 28 dim: 256 }
  }
  variable {
    name: "Mul2_3"
    type: "Buffer"
    repeat_id: "RecurrentInput"
    shape: { dim:-1 dim: 1 dim: 256 }
  }
  variable {
    name: "Mul2_4"
    type: "Buffer"
    repeat_id: "RecurrentInput_2"
    shape: { dim:-1 dim: 1 dim: 28 }
  }
  variable {
    name: "Add2_2"
    type: "Buffer"
    repeat_id: "RecurrentInput_2"
    shape: { dim:-1 dim: 1 dim: 28 }
  }
  variable {
    name: "Mul2_5"
    type: "Buffer"
    repeat_id: "RecurrentInput_2"
    shape: { dim:-1 dim: 1 dim: 28 }
  }
  variable {
    name: "Mul2_6"
    type: "Buffer"
    repeat_id: "RecurrentInput_2"
    shape: { dim:-1 dim: 1 dim: 28 }
  }
  variable {
    name: "RecurrentOutput_2"
    type: "Buffer"
    shape: { dim:-1 dim: 1 dim: 28 dim: 28 }
  }
  variable {
    name: "SquaredError"
    type: "Buffer"
    shape: { dim:-1 dim: 1 dim: 28 dim: 28 }
  }
  variable {
    name: "Slice"
    type: "Buffer"
    shape: { dim:-1 dim: 1 dim: 1 dim: 256 }
  }
  variable {
    name: "Unpooling"
    type: "Buffer"
    shape: { dim:-1 dim: 1 dim: 28 dim: 256 }
  }
  function {
    name: "Flip"
    type: "Flip"
    input: "Input"
    output: "Flip"
    flip_param {
      axes: 2
    }
  }
  function {
    name: "RecurrentInput"
    type: "RecurrentInput"
    repeat_id: "RecurrentInput"
    input: "Flip"
    output: "RecurrentInput"
    recurrent_param {
      axis: 2
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
    name: "Tanh"
    type: "Tanh"
    repeat_id: "RecurrentInput"
    input: "Affine_2"
    output: "Tanh"
  }
  function {
    name: "Affine_4"
    type: "Affine"
    repeat_id: "RecurrentInput"
    input: "Concatenate"
    input: "Affine_4/affine/W"
    input: "Affine_4/affine/b"
    output: "Affine_4"
    affine_param {
      base_axis: 1
    }
  }
  function {
    name: "Sigmoid"
    type: "Sigmoid"
    repeat_id: "RecurrentInput"
    input: "Affine_4"
    output: "Sigmoid"
  }
  function {
    name: "Concatenate"
    type: "Concatenate"
    repeat_id: "RecurrentInput"
    input: "Delay_2"
    input: "RecurrentInput"
    output: "Concatenate"
    concatenate_param {
      axis: 2
    }
  }
  function {
    name: "Affine_5"
    type: "Affine"
    repeat_id: "RecurrentInput"
    input: "Concatenate"
    input: "Affine_5/affine/W"
    input: "Affine_5/affine/b"
    output: "Affine_5"
    affine_param {
      base_axis: 1
    }
  }
  function {
    name: "Sigmoid_2"
    type: "Sigmoid"
    repeat_id: "RecurrentInput"
    input: "Affine_5"
    output: "Sigmoid_2"
  }
  function {
    name: "Tanh_2"
    type: "Tanh"
    repeat_id: "RecurrentInput"
    input: "Add2"
    output: "Tanh_2"
  }
  function {
    name: "Affine_6"
    type: "Affine"
    repeat_id: "RecurrentInput"
    input: "Concatenate"
    input: "Affine_6/affine/W"
    input: "Affine_6/affine/b"
    output: "Affine_6"
    affine_param {
      base_axis: 1
    }
  }
  function {
    name: "Sigmoid_3"
    type: "Sigmoid"
    repeat_id: "RecurrentInput"
    input: "Affine_6"
    output: "Sigmoid_3"
  }
  function {
    name: "Delay"
    type: "Delay"
    repeat_id: "RecurrentInput"
    input: "Add2"
    input: "Delay_Initial"
    output: "Delay"
    recurrent_param {
      repeat_id: "RecurrentInput"
    }
  }
  function {
    name: "Delay_2"
    type: "Delay"
    repeat_id: "RecurrentInput"
    input: "Mul2_2"
    input: "Delay_2_Initial"
    output: "Delay_2"
    recurrent_param {
      repeat_id: "RecurrentInput"
    }
  }
  function {
    name: "Sigmoid_4"
    type: "Sigmoid"
    input: "Convolution"
    output: "Sigmoid_4"
  }
  function {
    name: "Affine_7"
    type: "Affine"
    repeat_id: "RecurrentInput_2"
    input: "Concatenate_2"
    input: "Affine_7/affine/W"
    input: "Affine_7/affine/b"
    output: "Affine_7"
    affine_param {
      base_axis: 1
    }
  }
  function {
    name: "Tanh_3"
    type: "Tanh"
    repeat_id: "RecurrentInput_2"
    input: "Affine_7"
    output: "Tanh_3"
  }
  function {
    name: "Affine_8"
    type: "Affine"
    repeat_id: "RecurrentInput_2"
    input: "Concatenate_2"
    input: "Affine_8/affine/W"
    input: "Affine_8/affine/b"
    output: "Affine_8"
    affine_param {
      base_axis: 1
    }
  }
  function {
    name: "Sigmoid_5"
    type: "Sigmoid"
    repeat_id: "RecurrentInput_2"
    input: "Affine_8"
    output: "Sigmoid_5"
  }
  function {
    name: "Concatenate_2"
    type: "Concatenate"
    repeat_id: "RecurrentInput_2"
    input: "Delay_4"
    input: "RecurrentInput_2"
    output: "Concatenate_2"
    concatenate_param {
      axis: 2
    }
  }
  function {
    name: "Affine_9"
    type: "Affine"
    repeat_id: "RecurrentInput_2"
    input: "Concatenate_2"
    input: "Affine_9/affine/W"
    input: "Affine_9/affine/b"
    output: "Affine_9"
    affine_param {
      base_axis: 1
    }
  }
  function {
    name: "Sigmoid_6"
    type: "Sigmoid"
    repeat_id: "RecurrentInput_2"
    input: "Affine_9"
    output: "Sigmoid_6"
  }
  function {
    name: "Tanh_4"
    type: "Tanh"
    repeat_id: "RecurrentInput_2"
    input: "Add2_2"
    output: "Tanh_4"
  }
  function {
    name: "Affine_10"
    type: "Affine"
    repeat_id: "RecurrentInput_2"
    input: "Concatenate_2"
    input: "Affine_10/affine/W"
    input: "Affine_10/affine/b"
    output: "Affine_10"
    affine_param {
      base_axis: 1
    }
  }
  function {
    name: "Sigmoid_7"
    type: "Sigmoid"
    repeat_id: "RecurrentInput_2"
    input: "Affine_10"
    output: "Sigmoid_7"
  }
  function {
    name: "Delay_3"
    type: "Delay"
    repeat_id: "RecurrentInput_2"
    input: "Add2_2"
    input: "Delay_3_Initial"
    output: "Delay_3"
    recurrent_param {
      repeat_id: "RecurrentInput_2"
    }
  }
  function {
    name: "Delay_4"
    type: "Delay"
    repeat_id: "RecurrentInput_2"
    input: "Mul2_6"
    input: "Delay_4_Initial"
    output: "Delay_4"
    recurrent_param {
      repeat_id: "RecurrentInput_2"
    }
  }
  function {
    name: "RecurrentInput_2"
    type: "RecurrentInput"
    repeat_id: "RecurrentInput_2"
    input: "Unpooling"
    output: "RecurrentInput_2"
    recurrent_param {
      axis: 2
      repeat_id: "RecurrentInput_2"
    }
  }
  function {
    name: "Convolution"
    type: "Convolution"
    input: "RecurrentOutput_2"
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
    name: "Mul2"
    type: "Mul2"
    repeat_id: "RecurrentInput"
    input: "Delay"
    input: "Sigmoid_2"
    output: "Mul2"
  }
  function {
    name: "Add2"
    type: "Add2"
    repeat_id: "RecurrentInput"
    input: "Mul2"
    input: "Mul2_3"
    output: "Add2"
    add2_param {
      inplace: True
    }
  }
  function {
    name: "Mul2_2"
    type: "Mul2"
    repeat_id: "RecurrentInput"
    input: "Tanh_2"
    input: "Sigmoid_3"
    output: "Mul2_2"
  }
  function {
    name: "RecurrentOutput"
    type: "RecurrentOutput"
    input: "Mul2_2"
    output: "RecurrentOutput"
    recurrent_param {
      axis: 2
      repeat_id: "RecurrentInput"
      length: 28
    }
  }
  function {
    name: "Mul2_3"
    type: "Mul2"
    repeat_id: "RecurrentInput"
    input: "Tanh"
    input: "Sigmoid"
    output: "Mul2_3"
  }
  function {
    name: "Mul2_4"
    type: "Mul2"
    repeat_id: "RecurrentInput_2"
    input: "Sigmoid_6"
    input: "Delay_3"
    output: "Mul2_4"
  }
  function {
    name: "Add2_2"
    type: "Add2"
    repeat_id: "RecurrentInput_2"
    input: "Mul2_4"
    input: "Mul2_5"
    output: "Add2_2"
    add2_param {
      inplace: True
    }
  }
  function {
    name: "Mul2_5"
    type: "Mul2"
    repeat_id: "RecurrentInput_2"
    input: "Tanh_3"
    input: "Sigmoid_5"
    output: "Mul2_5"
  }
  function {
    name: "Mul2_6"
    type: "Mul2"
    repeat_id: "RecurrentInput_2"
    input: "Tanh_4"
    input: "Sigmoid_7"
    output: "Mul2_6"
  }
  function {
    name: "RecurrentOutput_2"
    type: "RecurrentOutput"
    input: "Mul2_6"
    output: "RecurrentOutput_2"
    recurrent_param {
      axis: 2
      repeat_id: "RecurrentInput_2"
      length: 28
    }
  }
  function {
    name: "SquaredError"
    type: "SquaredError"
    input: "Sigmoid_4"
    input: "SquaredError_T"
    output: "SquaredError"
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
      stop: 256
      step: 1
      step: 1
      step: 1
    }
  }
  function {
    name: "Unpooling"
    type: "Unpooling"
    input: "Slice"
    output: "Unpooling"
    unpooling_param {
      kernel: { dim: 28 dim: 1 }
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
  repeat_info {
    id: "RecurrentInput_2"
    times: 28
  }
  variable {
    name: "Input"
    type: "Buffer"
    shape: { dim:-1 dim: 1 dim: 28 dim: 28 }
  }
  variable {
    name: "Flip"
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
    name: "Affine_2/affine/W"
    type: "Parameter"
    shape: { dim: 284 dim: 1 dim: 256 }
    initializer {
      type: "NormalAffineGlorot"
      multiplier: 1.0
    }
  }
  variable {
    name: "Affine_2/affine/b"
    type: "Parameter"
    shape: { dim: 1 dim: 256 }
    initializer {
      type: "Constant"
      multiplier: 0.0
    }
  }
  variable {
    name: "Affine_4/affine/W"
    type: "Parameter"
    shape: { dim: 284 dim: 1 dim: 256 }
    initializer {
      type: "NormalAffineGlorot"
      multiplier: 1.0
    }
  }
  variable {
    name: "Affine_4/affine/b"
    type: "Parameter"
    shape: { dim: 1 dim: 256 }
    initializer {
      type: "Constant"
      multiplier: 0.0
    }
  }
  variable {
    name: "Affine_5/affine/W"
    type: "Parameter"
    shape: { dim: 284 dim: 1 dim: 256 }
    initializer {
      type: "NormalAffineGlorot"
      multiplier: 1.0
    }
  }
  variable {
    name: "Affine_5/affine/b"
    type: "Parameter"
    shape: { dim: 1 dim: 256 }
    initializer {
      type: "Constant"
      multiplier: 0.0
    }
  }
  variable {
    name: "Affine_6/affine/W"
    type: "Parameter"
    shape: { dim: 284 dim: 1 dim: 256 }
    initializer {
      type: "NormalAffineGlorot"
      multiplier: 1.0
    }
  }
  variable {
    name: "Affine_6/affine/b"
    type: "Parameter"
    shape: { dim: 1 dim: 256 }
    initializer {
      type: "Constant"
      multiplier: 0.0
    }
  }
  variable {
    name: "Delay_Initial"
    type: "Buffer"
    shape: { dim:-1 dim: 1 dim: 256 }
  }
  variable {
    name: "Delay_2_Initial"
    type: "Buffer"
    shape: { dim:-1 dim: 1 dim: 256 }
  }
  variable {
    name: "Affine_7/affine/W"
    type: "Parameter"
    shape: { dim: 284 dim: 1 dim: 28 }
    initializer {
      type: "NormalAffineGlorot"
      multiplier: 1.0
    }
  }
  variable {
    name: "Affine_7/affine/b"
    type: "Parameter"
    shape: { dim: 1 dim: 28 }
    initializer {
      type: "Constant"
      multiplier: 0.0
    }
  }
  variable {
    name: "Affine_8/affine/W"
    type: "Parameter"
    shape: { dim: 284 dim: 1 dim: 28 }
    initializer {
      type: "NormalAffineGlorot"
      multiplier: 1.0
    }
  }
  variable {
    name: "Affine_8/affine/b"
    type: "Parameter"
    shape: { dim: 1 dim: 28 }
    initializer {
      type: "Constant"
      multiplier: 0.0
    }
  }
  variable {
    name: "Affine_9/affine/W"
    type: "Parameter"
    shape: { dim: 284 dim: 1 dim: 28 }
    initializer {
      type: "NormalAffineGlorot"
      multiplier: 1.0
    }
  }
  variable {
    name: "Affine_9/affine/b"
    type: "Parameter"
    shape: { dim: 1 dim: 28 }
    initializer {
      type: "Constant"
      multiplier: 0.0
    }
  }
  variable {
    name: "Affine_10/affine/W"
    type: "Parameter"
    shape: { dim: 284 dim: 1 dim: 28 }
    initializer {
      type: "NormalAffineGlorot"
      multiplier: 1.0
    }
  }
  variable {
    name: "Affine_10/affine/b"
    type: "Parameter"
    shape: { dim: 1 dim: 28 }
    initializer {
      type: "Constant"
      multiplier: 0.0
    }
  }
  variable {
    name: "Delay_3_Initial"
    type: "Buffer"
    shape: { dim:-1 dim: 1 dim: 28 }
  }
  variable {
    name: "Delay_4_Initial"
    type: "Buffer"
    shape: { dim:-1 dim: 1 dim: 28 }
  }
  variable {
    name: "Convolution/conv/W"
    type: "Parameter"
    shape: { dim: 1 dim: 1 dim: 1 dim: 1 }
    initializer {
      type: "NormalConvolutionGlorot"
      multiplier: 1.0
    }
  }
  variable {
    name: "Convolution/conv/b"
    type: "Parameter"
    shape: { dim: 1 }
    initializer {
      type: "Constant"
      multiplier: 0.0
    }
  }
  variable {
    name: "Affine_2"
    type: "Buffer"
    repeat_id: "RecurrentInput"
    shape: { dim:-1 dim: 1 dim: 256 }
  }
  variable {
    name: "Tanh"
    type: "Buffer"
    repeat_id: "RecurrentInput"
    shape: { dim:-1 dim: 1 dim: 256 }
  }
  variable {
    name: "Affine_4"
    type: "Buffer"
    repeat_id: "RecurrentInput"
    shape: { dim:-1 dim: 1 dim: 256 }
  }
  variable {
    name: "Sigmoid"
    type: "Buffer"
    repeat_id: "RecurrentInput"
    shape: { dim:-1 dim: 1 dim: 256 }
  }
  variable {
    name: "Concatenate"
    type: "Buffer"
    repeat_id: "RecurrentInput"
    shape: { dim:-1 dim: 1 dim: 284 }
  }
  variable {
    name: "Affine_5"
    type: "Buffer"
    repeat_id: "RecurrentInput"
    shape: { dim:-1 dim: 1 dim: 256 }
  }
  variable {
    name: "Sigmoid_2"
    type: "Buffer"
    repeat_id: "RecurrentInput"
    shape: { dim:-1 dim: 1 dim: 256 }
  }
  variable {
    name: "Tanh_2"
    type: "Buffer"
    repeat_id: "RecurrentInput"
    shape: { dim:-1 dim: 1 dim: 256 }
  }
  variable {
    name: "Affine_6"
    type: "Buffer"
    repeat_id: "RecurrentInput"
    shape: { dim:-1 dim: 1 dim: 256 }
  }
  variable {
    name: "Sigmoid_3"
    type: "Buffer"
    repeat_id: "RecurrentInput"
    shape: { dim:-1 dim: 1 dim: 256 }
  }
  variable {
    name: "Delay"
    type: "Buffer"
    repeat_id: "RecurrentInput"
    shape: { dim:-1 dim: 1 dim: 256 }
  }
  variable {
    name: "Delay_2"
    type: "Buffer"
    repeat_id: "RecurrentInput"
    shape: { dim:-1 dim: 1 dim: 256 }
  }
  variable {
    name: "Sigmoid_4"
    type: "Buffer"
    shape: { dim:-1 dim: 1 dim: 28 dim: 28 }
  }
  variable {
    name: "Affine_7"
    type: "Buffer"
    repeat_id: "RecurrentInput_2"
    shape: { dim:-1 dim: 1 dim: 28 }
  }
  variable {
    name: "Tanh_3"
    type: "Buffer"
    repeat_id: "RecurrentInput_2"
    shape: { dim:-1 dim: 1 dim: 28 }
  }
  variable {
    name: "Affine_8"
    type: "Buffer"
    repeat_id: "RecurrentInput_2"
    shape: { dim:-1 dim: 1 dim: 28 }
  }
  variable {
    name: "Sigmoid_5"
    type: "Buffer"
    repeat_id: "RecurrentInput_2"
    shape: { dim:-1 dim: 1 dim: 28 }
  }
  variable {
    name: "Concatenate_2"
    type: "Buffer"
    repeat_id: "RecurrentInput_2"
    shape: { dim:-1 dim: 1 dim: 284 }
  }
  variable {
    name: "Affine_9"
    type: "Buffer"
    repeat_id: "RecurrentInput_2"
    shape: { dim:-1 dim: 1 dim: 28 }
  }
  variable {
    name: "Sigmoid_6"
    type: "Buffer"
    repeat_id: "RecurrentInput_2"
    shape: { dim:-1 dim: 1 dim: 28 }
  }
  variable {
    name: "Tanh_4"
    type: "Buffer"
    repeat_id: "RecurrentInput_2"
    shape: { dim:-1 dim: 1 dim: 28 }
  }
  variable {
    name: "Affine_10"
    type: "Buffer"
    repeat_id: "RecurrentInput_2"
    shape: { dim:-1 dim: 1 dim: 28 }
  }
  variable {
    name: "Sigmoid_7"
    type: "Buffer"
    repeat_id: "RecurrentInput_2"
    shape: { dim:-1 dim: 1 dim: 28 }
  }
  variable {
    name: "Delay_3"
    type: "Buffer"
    repeat_id: "RecurrentInput_2"
    shape: { dim:-1 dim: 1 dim: 28 }
  }
  variable {
    name: "Delay_4"
    type: "Buffer"
    repeat_id: "RecurrentInput_2"
    shape: { dim:-1 dim: 1 dim: 28 }
  }
  variable {
    name: "RecurrentInput_2"
    type: "Buffer"
    repeat_id: "RecurrentInput_2"
    shape: { dim:-1 dim: 1 dim: 256 }
  }
  variable {
    name: "Convolution"
    type: "Buffer"
    shape: { dim:-1 dim: 1 dim: 28 dim: 28 }
  }
  variable {
    name: "Mul2"
    type: "Buffer"
    repeat_id: "RecurrentInput"
    shape: { dim:-1 dim: 1 dim: 256 }
  }
  variable {
    name: "Add2"
    type: "Buffer"
    repeat_id: "RecurrentInput"
    shape: { dim:-1 dim: 1 dim: 256 }
  }
  variable {
    name: "Mul2_2"
    type: "Buffer"
    repeat_id: "RecurrentInput"
    shape: { dim:-1 dim: 1 dim: 256 }
  }
  variable {
    name: "RecurrentOutput"
    type: "Buffer"
    shape: { dim:-1 dim: 1 dim: 28 dim: 256 }
  }
  variable {
    name: "Mul2_3"
    type: "Buffer"
    repeat_id: "RecurrentInput"
    shape: { dim:-1 dim: 1 dim: 256 }
  }
  variable {
    name: "Mul2_4"
    type: "Buffer"
    repeat_id: "RecurrentInput_2"
    shape: { dim:-1 dim: 1 dim: 28 }
  }
  variable {
    name: "Add2_2"
    type: "Buffer"
    repeat_id: "RecurrentInput_2"
    shape: { dim:-1 dim: 1 dim: 28 }
  }
  variable {
    name: "Mul2_5"
    type: "Buffer"
    repeat_id: "RecurrentInput_2"
    shape: { dim:-1 dim: 1 dim: 28 }
  }
  variable {
    name: "Mul2_6"
    type: "Buffer"
    repeat_id: "RecurrentInput_2"
    shape: { dim:-1 dim: 1 dim: 28 }
  }
  variable {
    name: "RecurrentOutput_2"
    type: "Buffer"
    shape: { dim:-1 dim: 1 dim: 28 dim: 28 }
  }
  variable {
    name: "Slice"
    type: "Buffer"
    shape: { dim:-1 dim: 1 dim: 1 dim: 256 }
  }
  variable {
    name: "Unpooling"
    type: "Buffer"
    shape: { dim:-1 dim: 1 dim: 28 dim: 256 }
  }
  function {
    name: "Flip"
    type: "Flip"
    input: "Input"
    output: "Flip"
    flip_param {
      axes: 2
    }
  }
  function {
    name: "RecurrentInput"
    type: "RecurrentInput"
    repeat_id: "RecurrentInput"
    input: "Flip"
    output: "RecurrentInput"
    recurrent_param {
      axis: 2
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
    name: "Tanh"
    type: "Tanh"
    repeat_id: "RecurrentInput"
    input: "Affine_2"
    output: "Tanh"
  }
  function {
    name: "Affine_4"
    type: "Affine"
    repeat_id: "RecurrentInput"
    input: "Concatenate"
    input: "Affine_4/affine/W"
    input: "Affine_4/affine/b"
    output: "Affine_4"
    affine_param {
      base_axis: 1
    }
  }
  function {
    name: "Sigmoid"
    type: "Sigmoid"
    repeat_id: "RecurrentInput"
    input: "Affine_4"
    output: "Sigmoid"
  }
  function {
    name: "Concatenate"
    type: "Concatenate"
    repeat_id: "RecurrentInput"
    input: "Delay_2"
    input: "RecurrentInput"
    output: "Concatenate"
    concatenate_param {
      axis: 2
    }
  }
  function {
    name: "Affine_5"
    type: "Affine"
    repeat_id: "RecurrentInput"
    input: "Concatenate"
    input: "Affine_5/affine/W"
    input: "Affine_5/affine/b"
    output: "Affine_5"
    affine_param {
      base_axis: 1
    }
  }
  function {
    name: "Sigmoid_2"
    type: "Sigmoid"
    repeat_id: "RecurrentInput"
    input: "Affine_5"
    output: "Sigmoid_2"
  }
  function {
    name: "Tanh_2"
    type: "Tanh"
    repeat_id: "RecurrentInput"
    input: "Add2"
    output: "Tanh_2"
  }
  function {
    name: "Affine_6"
    type: "Affine"
    repeat_id: "RecurrentInput"
    input: "Concatenate"
    input: "Affine_6/affine/W"
    input: "Affine_6/affine/b"
    output: "Affine_6"
    affine_param {
      base_axis: 1
    }
  }
  function {
    name: "Sigmoid_3"
    type: "Sigmoid"
    repeat_id: "RecurrentInput"
    input: "Affine_6"
    output: "Sigmoid_3"
  }
  function {
    name: "Delay"
    type: "Delay"
    repeat_id: "RecurrentInput"
    input: "Add2"
    input: "Delay_Initial"
    output: "Delay"
    recurrent_param {
      repeat_id: "RecurrentInput"
    }
  }
  function {
    name: "Delay_2"
    type: "Delay"
    repeat_id: "RecurrentInput"
    input: "Mul2_2"
    input: "Delay_2_Initial"
    output: "Delay_2"
    recurrent_param {
      repeat_id: "RecurrentInput"
    }
  }
  function {
    name: "Sigmoid_4"
    type: "Sigmoid"
    input: "Convolution"
    output: "Sigmoid_4"
  }
  function {
    name: "Affine_7"
    type: "Affine"
    repeat_id: "RecurrentInput_2"
    input: "Concatenate_2"
    input: "Affine_7/affine/W"
    input: "Affine_7/affine/b"
    output: "Affine_7"
    affine_param {
      base_axis: 1
    }
  }
  function {
    name: "Tanh_3"
    type: "Tanh"
    repeat_id: "RecurrentInput_2"
    input: "Affine_7"
    output: "Tanh_3"
  }
  function {
    name: "Affine_8"
    type: "Affine"
    repeat_id: "RecurrentInput_2"
    input: "Concatenate_2"
    input: "Affine_8/affine/W"
    input: "Affine_8/affine/b"
    output: "Affine_8"
    affine_param {
      base_axis: 1
    }
  }
  function {
    name: "Sigmoid_5"
    type: "Sigmoid"
    repeat_id: "RecurrentInput_2"
    input: "Affine_8"
    output: "Sigmoid_5"
  }
  function {
    name: "Concatenate_2"
    type: "Concatenate"
    repeat_id: "RecurrentInput_2"
    input: "Delay_4"
    input: "RecurrentInput_2"
    output: "Concatenate_2"
    concatenate_param {
      axis: 2
    }
  }
  function {
    name: "Affine_9"
    type: "Affine"
    repeat_id: "RecurrentInput_2"
    input: "Concatenate_2"
    input: "Affine_9/affine/W"
    input: "Affine_9/affine/b"
    output: "Affine_9"
    affine_param {
      base_axis: 1
    }
  }
  function {
    name: "Sigmoid_6"
    type: "Sigmoid"
    repeat_id: "RecurrentInput_2"
    input: "Affine_9"
    output: "Sigmoid_6"
  }
  function {
    name: "Tanh_4"
    type: "Tanh"
    repeat_id: "RecurrentInput_2"
    input: "Add2_2"
    output: "Tanh_4"
  }
  function {
    name: "Affine_10"
    type: "Affine"
    repeat_id: "RecurrentInput_2"
    input: "Concatenate_2"
    input: "Affine_10/affine/W"
    input: "Affine_10/affine/b"
    output: "Affine_10"
    affine_param {
      base_axis: 1
    }
  }
  function {
    name: "Sigmoid_7"
    type: "Sigmoid"
    repeat_id: "RecurrentInput_2"
    input: "Affine_10"
    output: "Sigmoid_7"
  }
  function {
    name: "Delay_3"
    type: "Delay"
    repeat_id: "RecurrentInput_2"
    input: "Add2_2"
    input: "Delay_3_Initial"
    output: "Delay_3"
    recurrent_param {
      repeat_id: "RecurrentInput_2"
    }
  }
  function {
    name: "Delay_4"
    type: "Delay"
    repeat_id: "RecurrentInput_2"
    input: "Mul2_6"
    input: "Delay_4_Initial"
    output: "Delay_4"
    recurrent_param {
      repeat_id: "RecurrentInput_2"
    }
  }
  function {
    name: "RecurrentInput_2"
    type: "RecurrentInput"
    repeat_id: "RecurrentInput_2"
    input: "Unpooling"
    output: "RecurrentInput_2"
    recurrent_param {
      axis: 2
      repeat_id: "RecurrentInput_2"
    }
  }
  function {
    name: "Convolution"
    type: "Convolution"
    input: "RecurrentOutput_2"
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
    name: "Mul2"
    type: "Mul2"
    repeat_id: "RecurrentInput"
    input: "Delay"
    input: "Sigmoid_2"
    output: "Mul2"
  }
  function {
    name: "Add2"
    type: "Add2"
    repeat_id: "RecurrentInput"
    input: "Mul2"
    input: "Mul2_3"
    output: "Add2"
    add2_param {
      inplace: True
    }
  }
  function {
    name: "Mul2_2"
    type: "Mul2"
    repeat_id: "RecurrentInput"
    input: "Tanh_2"
    input: "Sigmoid_3"
    output: "Mul2_2"
  }
  function {
    name: "RecurrentOutput"
    type: "RecurrentOutput"
    input: "Mul2_2"
    output: "RecurrentOutput"
    recurrent_param {
      axis: 2
      repeat_id: "RecurrentInput"
      length: 28
    }
  }
  function {
    name: "Mul2_3"
    type: "Mul2"
    repeat_id: "RecurrentInput"
    input: "Tanh"
    input: "Sigmoid"
    output: "Mul2_3"
  }
  function {
    name: "Mul2_4"
    type: "Mul2"
    repeat_id: "RecurrentInput_2"
    input: "Sigmoid_6"
    input: "Delay_3"
    output: "Mul2_4"
  }
  function {
    name: "Add2_2"
    type: "Add2"
    repeat_id: "RecurrentInput_2"
    input: "Mul2_4"
    input: "Mul2_5"
    output: "Add2_2"
    add2_param {
      inplace: True
    }
  }
  function {
    name: "Mul2_5"
    type: "Mul2"
    repeat_id: "RecurrentInput_2"
    input: "Tanh_3"
    input: "Sigmoid_5"
    output: "Mul2_5"
  }
  function {
    name: "Mul2_6"
    type: "Mul2"
    repeat_id: "RecurrentInput_2"
    input: "Tanh_4"
    input: "Sigmoid_7"
    output: "Mul2_6"
  }
  function {
    name: "RecurrentOutput_2"
    type: "RecurrentOutput"
    input: "Mul2_6"
    output: "RecurrentOutput_2"
    recurrent_param {
      axis: 2
      repeat_id: "RecurrentInput_2"
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
      stop: 256
      step: 1
      step: 1
      step: 1
    }
  }
  function {
    name: "Unpooling"
    type: "Unpooling"
    input: "Slice"
    output: "Unpooling"
    unpooling_param {
      kernel: { dim: 28 dim: 1 }
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
  generator_variable {
    variable_name: "Delay_2_Initial"
    type: "Constant"
    multiplier: 0.0
  }
  generator_variable {
    variable_name: "Delay_3_Initial"
    type: "Constant"
    multiplier: 0.0
  }
  generator_variable {
    variable_name: "Delay_4_Initial"
    type: "Constant"
    multiplier: 0.0
  }
  data_variable {
    variable_name: "SquaredError_T"
    data_name: "x"
  }
  loss_variable {
    variable_name: "SquaredError"
  }
  parameter_variable {
    variable_name: "Affine_2/affine/W"
    learning_rate_multiplier: 1.0
  }
  parameter_variable {
    variable_name: "Affine_2/affine/b"
    learning_rate_multiplier: 1.0
  }
  parameter_variable {
    variable_name: "Affine_4/affine/W"
    learning_rate_multiplier: 1.0
  }
  parameter_variable {
    variable_name: "Affine_4/affine/b"
    learning_rate_multiplier: 1.0
  }
  parameter_variable {
    variable_name: "Affine_5/affine/W"
    learning_rate_multiplier: 1.0
  }
  parameter_variable {
    variable_name: "Affine_5/affine/b"
    learning_rate_multiplier: 1.0
  }
  parameter_variable {
    variable_name: "Affine_6/affine/W"
    learning_rate_multiplier: 1.0
  }
  parameter_variable {
    variable_name: "Affine_6/affine/b"
    learning_rate_multiplier: 1.0
  }
  parameter_variable {
    variable_name: "Affine_7/affine/W"
    learning_rate_multiplier: 1.0
  }
  parameter_variable {
    variable_name: "Affine_7/affine/b"
    learning_rate_multiplier: 1.0
  }
  parameter_variable {
    variable_name: "Affine_8/affine/W"
    learning_rate_multiplier: 1.0
  }
  parameter_variable {
    variable_name: "Affine_8/affine/b"
    learning_rate_multiplier: 1.0
  }
  parameter_variable {
    variable_name: "Affine_9/affine/W"
    learning_rate_multiplier: 1.0
  }
  parameter_variable {
    variable_name: "Affine_9/affine/b"
    learning_rate_multiplier: 1.0
  }
  parameter_variable {
    variable_name: "Affine_10/affine/W"
    learning_rate_multiplier: 1.0
  }
  parameter_variable {
    variable_name: "Affine_10/affine/b"
    learning_rate_multiplier: 1.0
  }
  parameter_variable {
    variable_name: "Convolution/conv/W"
    learning_rate_multiplier: 1.0
  }
  parameter_variable {
    variable_name: "Convolution/conv/b"
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
  generator_variable {
    variable_name: "Delay_2_Initial"
    type: "Constant"
    multiplier: 0.0
  }
  generator_variable {
    variable_name: "Delay_3_Initial"
    type: "Constant"
    multiplier: 0.0
  }
  generator_variable {
    variable_name: "Delay_4_Initial"
    type: "Constant"
    multiplier: 0.0
  }
  data_variable {
    variable_name: "SquaredError_T"
    data_name: "x"
  }
  monitor_variable {
    type: "Error"
    variable_name: "SquaredError"
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
  generator_variable {
    variable_name: "Delay_2_Initial"
    type: "Constant"
    multiplier: 0.0
  }
  generator_variable {
    variable_name: "Delay_3_Initial"
    type: "Constant"
    multiplier: 0.0
  }
  generator_variable {
    variable_name: "Delay_4_Initial"
    type: "Constant"
    multiplier: 0.0
  }
  data_variable {
    variable_name: "SquaredError_T"
    data_name: "x"
  }
  monitor_variable {
    type: "Error"
    variable_name: "SquaredError"
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
  generator_variable {
    variable_name: "Delay_2_Initial"
    type: "Constant"
    multiplier: 0.0
  }
  generator_variable {
    variable_name: "Delay_3_Initial"
    type: "Constant"
    multiplier: 0.0
  }
  generator_variable {
    variable_name: "Delay_4_Initial"
    type: "Constant"
    multiplier: 0.0
  }
  output_variable {
    variable_name: "Sigmoid_4"
    data_name: "x'"
  }
  parameter_variable {
    variable_name: "Affine_2/affine/W"
  }
  parameter_variable {
    variable_name: "Affine_2/affine/b"
  }
  parameter_variable {
    variable_name: "Affine_4/affine/W"
  }
  parameter_variable {
    variable_name: "Affine_4/affine/b"
  }
  parameter_variable {
    variable_name: "Affine_5/affine/W"
  }
  parameter_variable {
    variable_name: "Affine_5/affine/b"
  }
  parameter_variable {
    variable_name: "Affine_6/affine/W"
  }
  parameter_variable {
    variable_name: "Affine_6/affine/b"
  }
  parameter_variable {
    variable_name: "Affine_7/affine/W"
  }
  parameter_variable {
    variable_name: "Affine_7/affine/b"
  }
  parameter_variable {
    variable_name: "Affine_8/affine/W"
  }
  parameter_variable {
    variable_name: "Affine_8/affine/b"
  }
  parameter_variable {
    variable_name: "Affine_9/affine/W"
  }
  parameter_variable {
    variable_name: "Affine_9/affine/b"
  }
  parameter_variable {
    variable_name: "Affine_10/affine/W"
  }
  parameter_variable {
    variable_name: "Affine_10/affine/b"
  }
  parameter_variable {
    variable_name: "Convolution/conv/W"
  }
  parameter_variable {
    variable_name: "Convolution/conv/b"
  }
}
'''
# The nntxt is generated from /home/woody/aid/nnabla-sample-data/sample_project/tutorial/recurrent_neural_networks/LSTM_auto_encoder.files/20180720_192730/net.nntxt.
N00000011 = r'''global_config {
  default_context {
    array_class: "CudaCachedArray"
    device_id: "0"
    backends: "cudnn:float"
    backends: "cuda:float"
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
    id: "RecurrentInput"
    times: 28
  }
  repeat_info {
    id: "RecurrentInput_2"
    times: 28
  }
  variable {
    name: "Input"
    type: "Buffer"
    shape: { dim:-1 dim: 1 dim: 28 dim: 28 }
  }
  variable {
    name: "Delay_Initial"
    type: "Buffer"
    shape: { dim:-1 dim: 1 dim: 256 }
  }
  variable {
    name: "Delay_2_Initial"
    type: "Buffer"
    shape: { dim:-1 dim: 1 dim: 256 }
  }
  variable {
    name: "Affine_2/affine/W"
    type: "Parameter"
    shape: { dim: 284 dim: 1 dim: 256 }
    initializer {
      type: "NormalAffineGlorot"
      multiplier: 1
    }
  }
  variable {
    name: "Affine_2/affine/b"
    type: "Parameter"
    shape: { dim: 1 dim: 256 }
    initializer {
      type: "Constant"
      multiplier: 0
    }
  }
  variable {
    name: "Affine_4/affine/W"
    type: "Parameter"
    shape: { dim: 284 dim: 1 dim: 256 }
    initializer {
      type: "NormalAffineGlorot"
      multiplier: 1
    }
  }
  variable {
    name: "Affine_4/affine/b"
    type: "Parameter"
    shape: { dim: 1 dim: 256 }
    initializer {
      type: "Constant"
      multiplier: 0
    }
  }
  variable {
    name: "Affine_5/affine/W"
    type: "Parameter"
    shape: { dim: 284 dim: 1 dim: 256 }
    initializer {
      type: "NormalAffineGlorot"
      multiplier: 1
    }
  }
  variable {
    name: "Affine_5/affine/b"
    type: "Parameter"
    shape: { dim: 1 dim: 256 }
    initializer {
      type: "Constant"
      multiplier: 0
    }
  }
  variable {
    name: "Affine_6/affine/W"
    type: "Parameter"
    shape: { dim: 284 dim: 1 dim: 256 }
    initializer {
      type: "NormalAffineGlorot"
      multiplier: 1
    }
  }
  variable {
    name: "Affine_6/affine/b"
    type: "Parameter"
    shape: { dim: 1 dim: 256 }
    initializer {
      type: "Constant"
      multiplier: 0
    }
  }
  variable {
    name: "Delay_3_Initial"
    type: "Buffer"
    shape: { dim:-1 dim: 1 dim: 28 }
  }
  variable {
    name: "Delay_4_Initial"
    type: "Buffer"
    shape: { dim:-1 dim: 1 dim: 28 }
  }
  variable {
    name: "Affine_7/affine/W"
    type: "Parameter"
    shape: { dim: 284 dim: 1 dim: 28 }
    initializer {
      type: "NormalAffineGlorot"
      multiplier: 1
    }
  }
  variable {
    name: "Affine_7/affine/b"
    type: "Parameter"
    shape: { dim: 1 dim: 28 }
    initializer {
      type: "Constant"
      multiplier: 0
    }
  }
  variable {
    name: "Affine_8/affine/W"
    type: "Parameter"
    shape: { dim: 284 dim: 1 dim: 28 }
    initializer {
      type: "NormalAffineGlorot"
      multiplier: 1
    }
  }
  variable {
    name: "Affine_8/affine/b"
    type: "Parameter"
    shape: { dim: 1 dim: 28 }
    initializer {
      type: "Constant"
      multiplier: 0
    }
  }
  variable {
    name: "Affine_9/affine/W"
    type: "Parameter"
    shape: { dim: 284 dim: 1 dim: 28 }
    initializer {
      type: "NormalAffineGlorot"
      multiplier: 1
    }
  }
  variable {
    name: "Affine_9/affine/b"
    type: "Parameter"
    shape: { dim: 1 dim: 28 }
    initializer {
      type: "Constant"
      multiplier: 0
    }
  }
  variable {
    name: "Affine_10/affine/W"
    type: "Parameter"
    shape: { dim: 284 dim: 1 dim: 28 }
    initializer {
      type: "NormalAffineGlorot"
      multiplier: 1
    }
  }
  variable {
    name: "Affine_10/affine/b"
    type: "Parameter"
    shape: { dim: 1 dim: 28 }
    initializer {
      type: "Constant"
      multiplier: 0
    }
  }
  variable {
    name: "Convolution/conv/W"
    type: "Parameter"
    shape: { dim: 1 dim: 1 dim: 1 dim: 1 }
    initializer {
      type: "NormalConvolutionGlorot"
      multiplier: 1
    }
  }
  variable {
    name: "Convolution/conv/b"
    type: "Parameter"
    shape: { dim: 1 }
    initializer {
      type: "Constant"
      multiplier: 0
    }
  }
  variable {
    name: "SquaredError_T"
    type: "Buffer"
    shape: { dim:-1 dim: 1 dim: 28 dim: 28 }
  }
  variable {
    name: "Flip"
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
    name: "Delay"
    type: "Buffer"
    repeat_id: "RecurrentInput"
    shape: { dim:-1 dim: 1 dim: 256 }
  }
  variable {
    name: "Delay_2"
    type: "Buffer"
    repeat_id: "RecurrentInput"
    shape: { dim:-1 dim: 1 dim: 256 }
  }
  variable {
    name: "Concatenate"
    type: "Buffer"
    repeat_id: "RecurrentInput"
    shape: { dim:-1 dim: 1 dim: 284 }
  }
  variable {
    name: "Affine_2"
    type: "Buffer"
    repeat_id: "RecurrentInput"
    shape: { dim:-1 dim: 1 dim: 256 }
  }
  variable {
    name: "Affine_4"
    type: "Buffer"
    repeat_id: "RecurrentInput"
    shape: { dim:-1 dim: 1 dim: 256 }
  }
  variable {
    name: "Affine_5"
    type: "Buffer"
    repeat_id: "RecurrentInput"
    shape: { dim:-1 dim: 1 dim: 256 }
  }
  variable {
    name: "Affine_6"
    type: "Buffer"
    repeat_id: "RecurrentInput"
    shape: { dim:-1 dim: 1 dim: 256 }
  }
  variable {
    name: "Tanh"
    type: "Buffer"
    repeat_id: "RecurrentInput"
    shape: { dim:-1 dim: 1 dim: 256 }
  }
  variable {
    name: "Sigmoid"
    type: "Buffer"
    repeat_id: "RecurrentInput"
    shape: { dim:-1 dim: 1 dim: 256 }
  }
  variable {
    name: "Sigmoid_2"
    type: "Buffer"
    repeat_id: "RecurrentInput"
    shape: { dim:-1 dim: 1 dim: 256 }
  }
  variable {
    name: "Sigmoid_3"
    type: "Buffer"
    repeat_id: "RecurrentInput"
    shape: { dim:-1 dim: 1 dim: 256 }
  }
  variable {
    name: "Mul2"
    type: "Buffer"
    repeat_id: "RecurrentInput"
    shape: { dim:-1 dim: 1 dim: 256 }
  }
  variable {
    name: "Mul2_3"
    type: "Buffer"
    repeat_id: "RecurrentInput"
    shape: { dim:-1 dim: 1 dim: 256 }
  }
  variable {
    name: "Add2"
    type: "Buffer"
    repeat_id: "RecurrentInput"
    shape: { dim:-1 dim: 1 dim: 256 }
  }
  variable {
    name: "Tanh_2"
    type: "Buffer"
    repeat_id: "RecurrentInput"
    shape: { dim:-1 dim: 1 dim: 256 }
  }
  variable {
    name: "Mul2_2"
    type: "Buffer"
    repeat_id: "RecurrentInput"
    shape: { dim:-1 dim: 1 dim: 256 }
  }
  variable {
    name: "RecurrentOutput"
    type: "Buffer"
    shape: { dim:-1 dim: 1 dim: 28 dim: 256 }
  }
  variable {
    name: "Slice"
    type: "Buffer"
    shape: { dim:-1 dim: 1 dim: 1 dim: 256 }
  }
  variable {
    name: "Unpooling"
    type: "Buffer"
    shape: { dim:-1 dim: 1 dim: 28 dim: 256 }
  }
  variable {
    name: "RecurrentInput_2"
    type: "Buffer"
    repeat_id: "RecurrentInput_2"
    shape: { dim:-1 dim: 1 dim: 256 }
  }
  variable {
    name: "Delay_3"
    type: "Buffer"
    repeat_id: "RecurrentInput_2"
    shape: { dim:-1 dim: 1 dim: 28 }
  }
  variable {
    name: "Delay_4"
    type: "Buffer"
    repeat_id: "RecurrentInput_2"
    shape: { dim:-1 dim: 1 dim: 28 }
  }
  variable {
    name: "Concatenate_2"
    type: "Buffer"
    repeat_id: "RecurrentInput_2"
    shape: { dim:-1 dim: 1 dim: 284 }
  }
  variable {
    name: "Affine_7"
    type: "Buffer"
    repeat_id: "RecurrentInput_2"
    shape: { dim:-1 dim: 1 dim: 28 }
  }
  variable {
    name: "Affine_8"
    type: "Buffer"
    repeat_id: "RecurrentInput_2"
    shape: { dim:-1 dim: 1 dim: 28 }
  }
  variable {
    name: "Affine_9"
    type: "Buffer"
    repeat_id: "RecurrentInput_2"
    shape: { dim:-1 dim: 1 dim: 28 }
  }
  variable {
    name: "Affine_10"
    type: "Buffer"
    repeat_id: "RecurrentInput_2"
    shape: { dim:-1 dim: 1 dim: 28 }
  }
  variable {
    name: "Tanh_3"
    type: "Buffer"
    repeat_id: "RecurrentInput_2"
    shape: { dim:-1 dim: 1 dim: 28 }
  }
  variable {
    name: "Sigmoid_5"
    type: "Buffer"
    repeat_id: "RecurrentInput_2"
    shape: { dim:-1 dim: 1 dim: 28 }
  }
  variable {
    name: "Sigmoid_6"
    type: "Buffer"
    repeat_id: "RecurrentInput_2"
    shape: { dim:-1 dim: 1 dim: 28 }
  }
  variable {
    name: "Sigmoid_7"
    type: "Buffer"
    repeat_id: "RecurrentInput_2"
    shape: { dim:-1 dim: 1 dim: 28 }
  }
  variable {
    name: "Mul2_4"
    type: "Buffer"
    repeat_id: "RecurrentInput_2"
    shape: { dim:-1 dim: 1 dim: 28 }
  }
  variable {
    name: "Mul2_5"
    type: "Buffer"
    repeat_id: "RecurrentInput_2"
    shape: { dim:-1 dim: 1 dim: 28 }
  }
  variable {
    name: "Add2_2"
    type: "Buffer"
    repeat_id: "RecurrentInput_2"
    shape: { dim:-1 dim: 1 dim: 28 }
  }
  variable {
    name: "Tanh_4"
    type: "Buffer"
    repeat_id: "RecurrentInput_2"
    shape: { dim:-1 dim: 1 dim: 28 }
  }
  variable {
    name: "Mul2_6"
    type: "Buffer"
    repeat_id: "RecurrentInput_2"
    shape: { dim:-1 dim: 1 dim: 28 }
  }
  variable {
    name: "RecurrentOutput_2"
    type: "Buffer"
    shape: { dim:-1 dim: 1 dim: 28 dim: 28 }
  }
  variable {
    name: "Convolution"
    type: "Buffer"
    shape: { dim:-1 dim: 1 dim: 28 dim: 28 }
  }
  variable {
    name: "Sigmoid_4"
    type: "Buffer"
    shape: { dim:-1 dim: 1 dim: 28 dim: 28 }
  }
  variable {
    name: "SquaredError"
    type: "Buffer"
    shape: { dim:-1 dim: 1 dim: 28 dim: 28 }
  }
  function {
    name: "Flip"
    type: "Flip"
    input: "Input"
    output: "Flip"
    flip_param {
      axes: 2
    }
  }
  function {
    name: "RecurrentInput"
    type: "RecurrentInput"
    repeat_id: "RecurrentInput"
    input: "Flip"
    output: "RecurrentInput"
    recurrent_param {
      axis: 2
      repeat_id: "RecurrentInput"
    }
  }
  function {
    name: "Delay"
    type: "Delay"
    repeat_id: "RecurrentInput"
    input: "Add2"
    input: "Delay_Initial"
    output: "Delay"
    recurrent_param {
      repeat_id: "RecurrentInput"
    }
  }
  function {
    name: "Delay_2"
    type: "Delay"
    repeat_id: "RecurrentInput"
    input: "Mul2_2"
    input: "Delay_2_Initial"
    output: "Delay_2"
    recurrent_param {
      repeat_id: "RecurrentInput"
    }
  }
  function {
    name: "Concatenate"
    type: "Concatenate"
    repeat_id: "RecurrentInput"
    input: "Delay_2"
    input: "RecurrentInput"
    output: "Concatenate"
    concatenate_param {
      axis: 2
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
    name: "Affine_4"
    type: "Affine"
    repeat_id: "RecurrentInput"
    input: "Concatenate"
    input: "Affine_4/affine/W"
    input: "Affine_4/affine/b"
    output: "Affine_4"
    affine_param {
      base_axis: 1
    }
  }
  function {
    name: "Affine_5"
    type: "Affine"
    repeat_id: "RecurrentInput"
    input: "Concatenate"
    input: "Affine_5/affine/W"
    input: "Affine_5/affine/b"
    output: "Affine_5"
    affine_param {
      base_axis: 1
    }
  }
  function {
    name: "Affine_6"
    type: "Affine"
    repeat_id: "RecurrentInput"
    input: "Concatenate"
    input: "Affine_6/affine/W"
    input: "Affine_6/affine/b"
    output: "Affine_6"
    affine_param {
      base_axis: 1
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
    name: "Sigmoid"
    type: "Sigmoid"
    repeat_id: "RecurrentInput"
    input: "Affine_4"
    output: "Sigmoid"
  }
  function {
    name: "Sigmoid_2"
    type: "Sigmoid"
    repeat_id: "RecurrentInput"
    input: "Affine_5"
    output: "Sigmoid_2"
  }
  function {
    name: "Sigmoid_3"
    type: "Sigmoid"
    repeat_id: "RecurrentInput"
    input: "Affine_6"
    output: "Sigmoid_3"
  }
  function {
    name: "Mul2"
    type: "Mul2"
    repeat_id: "RecurrentInput"
    input: "Delay"
    input: "Sigmoid_2"
    output: "Mul2"
  }
  function {
    name: "Mul2_3"
    type: "Mul2"
    repeat_id: "RecurrentInput"
    input: "Tanh"
    input: "Sigmoid"
    output: "Mul2_3"
  }
  function {
    name: "Add2"
    type: "Add2"
    repeat_id: "RecurrentInput"
    input: "Mul2"
    input: "Mul2_3"
    output: "Add2"
    add2_param {
      inplace: True
    }
  }
  function {
    name: "Tanh_2"
    type: "Tanh"
    repeat_id: "RecurrentInput"
    input: "Add2"
    output: "Tanh_2"
  }
  function {
    name: "Mul2_2"
    type: "Mul2"
    repeat_id: "RecurrentInput"
    input: "Tanh_2"
    input: "Sigmoid_3"
    output: "Mul2_2"
  }
  function {
    name: "RecurrentOutput"
    type: "RecurrentOutput"
    input: "Mul2_2"
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
      stop: 256
      step: 1
      step: 1
      step: 1
    }
  }
  function {
    name: "Unpooling"
    type: "Unpooling"
    input: "Slice"
    output: "Unpooling"
    unpooling_param {
      kernel: { dim: 28 dim: 1 }
    }
  }
  function {
    name: "RecurrentInput_2"
    type: "RecurrentInput"
    repeat_id: "RecurrentInput_2"
    input: "Unpooling"
    output: "RecurrentInput_2"
    recurrent_param {
      axis: 2
      repeat_id: "RecurrentInput_2"
    }
  }
  function {
    name: "Delay_3"
    type: "Delay"
    repeat_id: "RecurrentInput_2"
    input: "Add2_2"
    input: "Delay_3_Initial"
    output: "Delay_3"
    recurrent_param {
      repeat_id: "RecurrentInput_2"
    }
  }
  function {
    name: "Delay_4"
    type: "Delay"
    repeat_id: "RecurrentInput_2"
    input: "Mul2_6"
    input: "Delay_4_Initial"
    output: "Delay_4"
    recurrent_param {
      repeat_id: "RecurrentInput_2"
    }
  }
  function {
    name: "Concatenate_2"
    type: "Concatenate"
    repeat_id: "RecurrentInput_2"
    input: "Delay_4"
    input: "RecurrentInput_2"
    output: "Concatenate_2"
    concatenate_param {
      axis: 2
    }
  }
  function {
    name: "Affine_7"
    type: "Affine"
    repeat_id: "RecurrentInput_2"
    input: "Concatenate_2"
    input: "Affine_7/affine/W"
    input: "Affine_7/affine/b"
    output: "Affine_7"
    affine_param {
      base_axis: 1
    }
  }
  function {
    name: "Affine_8"
    type: "Affine"
    repeat_id: "RecurrentInput_2"
    input: "Concatenate_2"
    input: "Affine_8/affine/W"
    input: "Affine_8/affine/b"
    output: "Affine_8"
    affine_param {
      base_axis: 1
    }
  }
  function {
    name: "Affine_9"
    type: "Affine"
    repeat_id: "RecurrentInput_2"
    input: "Concatenate_2"
    input: "Affine_9/affine/W"
    input: "Affine_9/affine/b"
    output: "Affine_9"
    affine_param {
      base_axis: 1
    }
  }
  function {
    name: "Affine_10"
    type: "Affine"
    repeat_id: "RecurrentInput_2"
    input: "Concatenate_2"
    input: "Affine_10/affine/W"
    input: "Affine_10/affine/b"
    output: "Affine_10"
    affine_param {
      base_axis: 1
    }
  }
  function {
    name: "Tanh_3"
    type: "Tanh"
    repeat_id: "RecurrentInput_2"
    input: "Affine_7"
    output: "Tanh_3"
  }
  function {
    name: "Sigmoid_5"
    type: "Sigmoid"
    repeat_id: "RecurrentInput_2"
    input: "Affine_8"
    output: "Sigmoid_5"
  }
  function {
    name: "Sigmoid_6"
    type: "Sigmoid"
    repeat_id: "RecurrentInput_2"
    input: "Affine_9"
    output: "Sigmoid_6"
  }
  function {
    name: "Sigmoid_7"
    type: "Sigmoid"
    repeat_id: "RecurrentInput_2"
    input: "Affine_10"
    output: "Sigmoid_7"
  }
  function {
    name: "Mul2_4"
    type: "Mul2"
    repeat_id: "RecurrentInput_2"
    input: "Sigmoid_6"
    input: "Delay_3"
    output: "Mul2_4"
  }
  function {
    name: "Mul2_5"
    type: "Mul2"
    repeat_id: "RecurrentInput_2"
    input: "Tanh_3"
    input: "Sigmoid_5"
    output: "Mul2_5"
  }
  function {
    name: "Add2_2"
    type: "Add2"
    repeat_id: "RecurrentInput_2"
    input: "Mul2_4"
    input: "Mul2_5"
    output: "Add2_2"
    add2_param {
      inplace: True
    }
  }
  function {
    name: "Tanh_4"
    type: "Tanh"
    repeat_id: "RecurrentInput_2"
    input: "Add2_2"
    output: "Tanh_4"
  }
  function {
    name: "Mul2_6"
    type: "Mul2"
    repeat_id: "RecurrentInput_2"
    input: "Tanh_4"
    input: "Sigmoid_7"
    output: "Mul2_6"
  }
  function {
    name: "RecurrentOutput_2"
    type: "RecurrentOutput"
    input: "Mul2_6"
    output: "RecurrentOutput_2"
    recurrent_param {
      axis: 2
      repeat_id: "RecurrentInput_2"
      length: 28
    }
  }
  function {
    name: "Convolution"
    type: "Convolution"
    input: "RecurrentOutput_2"
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
    name: "Sigmoid_4"
    type: "Sigmoid"
    input: "Convolution"
    output: "Sigmoid_4"
  }
  function {
    name: "SquaredError"
    type: "SquaredError"
    input: "Sigmoid_4"
    input: "SquaredError_T"
    output: "SquaredError"
  }
}
network {
  name: "MainValidation"
  batch_size: 64
  repeat_info {
    id: "RecurrentInput"
    times: 28
  }
  repeat_info {
    id: "RecurrentInput_2"
    times: 28
  }
  variable {
    name: "Input"
    type: "Buffer"
    shape: { dim:-1 dim: 1 dim: 28 dim: 28 }
  }
  variable {
    name: "Delay_Initial"
    type: "Buffer"
    shape: { dim:-1 dim: 1 dim: 256 }
  }
  variable {
    name: "Delay_2_Initial"
    type: "Buffer"
    shape: { dim:-1 dim: 1 dim: 256 }
  }
  variable {
    name: "Affine_2/affine/W"
    type: "Parameter"
    shape: { dim: 284 dim: 1 dim: 256 }
    initializer {
      type: "NormalAffineGlorot"
      multiplier: 1
    }
  }
  variable {
    name: "Affine_2/affine/b"
    type: "Parameter"
    shape: { dim: 1 dim: 256 }
    initializer {
      type: "Constant"
      multiplier: 0
    }
  }
  variable {
    name: "Affine_4/affine/W"
    type: "Parameter"
    shape: { dim: 284 dim: 1 dim: 256 }
    initializer {
      type: "NormalAffineGlorot"
      multiplier: 1
    }
  }
  variable {
    name: "Affine_4/affine/b"
    type: "Parameter"
    shape: { dim: 1 dim: 256 }
    initializer {
      type: "Constant"
      multiplier: 0
    }
  }
  variable {
    name: "Affine_5/affine/W"
    type: "Parameter"
    shape: { dim: 284 dim: 1 dim: 256 }
    initializer {
      type: "NormalAffineGlorot"
      multiplier: 1
    }
  }
  variable {
    name: "Affine_5/affine/b"
    type: "Parameter"
    shape: { dim: 1 dim: 256 }
    initializer {
      type: "Constant"
      multiplier: 0
    }
  }
  variable {
    name: "Affine_6/affine/W"
    type: "Parameter"
    shape: { dim: 284 dim: 1 dim: 256 }
    initializer {
      type: "NormalAffineGlorot"
      multiplier: 1
    }
  }
  variable {
    name: "Affine_6/affine/b"
    type: "Parameter"
    shape: { dim: 1 dim: 256 }
    initializer {
      type: "Constant"
      multiplier: 0
    }
  }
  variable {
    name: "Delay_3_Initial"
    type: "Buffer"
    shape: { dim:-1 dim: 1 dim: 28 }
  }
  variable {
    name: "Delay_4_Initial"
    type: "Buffer"
    shape: { dim:-1 dim: 1 dim: 28 }
  }
  variable {
    name: "Affine_7/affine/W"
    type: "Parameter"
    shape: { dim: 284 dim: 1 dim: 28 }
    initializer {
      type: "NormalAffineGlorot"
      multiplier: 1
    }
  }
  variable {
    name: "Affine_7/affine/b"
    type: "Parameter"
    shape: { dim: 1 dim: 28 }
    initializer {
      type: "Constant"
      multiplier: 0
    }
  }
  variable {
    name: "Affine_8/affine/W"
    type: "Parameter"
    shape: { dim: 284 dim: 1 dim: 28 }
    initializer {
      type: "NormalAffineGlorot"
      multiplier: 1
    }
  }
  variable {
    name: "Affine_8/affine/b"
    type: "Parameter"
    shape: { dim: 1 dim: 28 }
    initializer {
      type: "Constant"
      multiplier: 0
    }
  }
  variable {
    name: "Affine_9/affine/W"
    type: "Parameter"
    shape: { dim: 284 dim: 1 dim: 28 }
    initializer {
      type: "NormalAffineGlorot"
      multiplier: 1
    }
  }
  variable {
    name: "Affine_9/affine/b"
    type: "Parameter"
    shape: { dim: 1 dim: 28 }
    initializer {
      type: "Constant"
      multiplier: 0
    }
  }
  variable {
    name: "Affine_10/affine/W"
    type: "Parameter"
    shape: { dim: 284 dim: 1 dim: 28 }
    initializer {
      type: "NormalAffineGlorot"
      multiplier: 1
    }
  }
  variable {
    name: "Affine_10/affine/b"
    type: "Parameter"
    shape: { dim: 1 dim: 28 }
    initializer {
      type: "Constant"
      multiplier: 0
    }
  }
  variable {
    name: "Convolution/conv/W"
    type: "Parameter"
    shape: { dim: 1 dim: 1 dim: 1 dim: 1 }
    initializer {
      type: "NormalConvolutionGlorot"
      multiplier: 1
    }
  }
  variable {
    name: "Convolution/conv/b"
    type: "Parameter"
    shape: { dim: 1 }
    initializer {
      type: "Constant"
      multiplier: 0
    }
  }
  variable {
    name: "SquaredError_T"
    type: "Buffer"
    shape: { dim:-1 dim: 1 dim: 28 dim: 28 }
  }
  variable {
    name: "Flip"
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
    name: "Delay"
    type: "Buffer"
    repeat_id: "RecurrentInput"
    shape: { dim:-1 dim: 1 dim: 256 }
  }
  variable {
    name: "Delay_2"
    type: "Buffer"
    repeat_id: "RecurrentInput"
    shape: { dim:-1 dim: 1 dim: 256 }
  }
  variable {
    name: "Concatenate"
    type: "Buffer"
    repeat_id: "RecurrentInput"
    shape: { dim:-1 dim: 1 dim: 284 }
  }
  variable {
    name: "Affine_2"
    type: "Buffer"
    repeat_id: "RecurrentInput"
    shape: { dim:-1 dim: 1 dim: 256 }
  }
  variable {
    name: "Affine_4"
    type: "Buffer"
    repeat_id: "RecurrentInput"
    shape: { dim:-1 dim: 1 dim: 256 }
  }
  variable {
    name: "Affine_5"
    type: "Buffer"
    repeat_id: "RecurrentInput"
    shape: { dim:-1 dim: 1 dim: 256 }
  }
  variable {
    name: "Affine_6"
    type: "Buffer"
    repeat_id: "RecurrentInput"
    shape: { dim:-1 dim: 1 dim: 256 }
  }
  variable {
    name: "Tanh"
    type: "Buffer"
    repeat_id: "RecurrentInput"
    shape: { dim:-1 dim: 1 dim: 256 }
  }
  variable {
    name: "Sigmoid"
    type: "Buffer"
    repeat_id: "RecurrentInput"
    shape: { dim:-1 dim: 1 dim: 256 }
  }
  variable {
    name: "Sigmoid_2"
    type: "Buffer"
    repeat_id: "RecurrentInput"
    shape: { dim:-1 dim: 1 dim: 256 }
  }
  variable {
    name: "Sigmoid_3"
    type: "Buffer"
    repeat_id: "RecurrentInput"
    shape: { dim:-1 dim: 1 dim: 256 }
  }
  variable {
    name: "Mul2"
    type: "Buffer"
    repeat_id: "RecurrentInput"
    shape: { dim:-1 dim: 1 dim: 256 }
  }
  variable {
    name: "Mul2_3"
    type: "Buffer"
    repeat_id: "RecurrentInput"
    shape: { dim:-1 dim: 1 dim: 256 }
  }
  variable {
    name: "Add2"
    type: "Buffer"
    repeat_id: "RecurrentInput"
    shape: { dim:-1 dim: 1 dim: 256 }
  }
  variable {
    name: "Tanh_2"
    type: "Buffer"
    repeat_id: "RecurrentInput"
    shape: { dim:-1 dim: 1 dim: 256 }
  }
  variable {
    name: "Mul2_2"
    type: "Buffer"
    repeat_id: "RecurrentInput"
    shape: { dim:-1 dim: 1 dim: 256 }
  }
  variable {
    name: "RecurrentOutput"
    type: "Buffer"
    shape: { dim:-1 dim: 1 dim: 28 dim: 256 }
  }
  variable {
    name: "Slice"
    type: "Buffer"
    shape: { dim:-1 dim: 1 dim: 1 dim: 256 }
  }
  variable {
    name: "Unpooling"
    type: "Buffer"
    shape: { dim:-1 dim: 1 dim: 28 dim: 256 }
  }
  variable {
    name: "RecurrentInput_2"
    type: "Buffer"
    repeat_id: "RecurrentInput_2"
    shape: { dim:-1 dim: 1 dim: 256 }
  }
  variable {
    name: "Delay_3"
    type: "Buffer"
    repeat_id: "RecurrentInput_2"
    shape: { dim:-1 dim: 1 dim: 28 }
  }
  variable {
    name: "Delay_4"
    type: "Buffer"
    repeat_id: "RecurrentInput_2"
    shape: { dim:-1 dim: 1 dim: 28 }
  }
  variable {
    name: "Concatenate_2"
    type: "Buffer"
    repeat_id: "RecurrentInput_2"
    shape: { dim:-1 dim: 1 dim: 284 }
  }
  variable {
    name: "Affine_7"
    type: "Buffer"
    repeat_id: "RecurrentInput_2"
    shape: { dim:-1 dim: 1 dim: 28 }
  }
  variable {
    name: "Affine_8"
    type: "Buffer"
    repeat_id: "RecurrentInput_2"
    shape: { dim:-1 dim: 1 dim: 28 }
  }
  variable {
    name: "Affine_9"
    type: "Buffer"
    repeat_id: "RecurrentInput_2"
    shape: { dim:-1 dim: 1 dim: 28 }
  }
  variable {
    name: "Affine_10"
    type: "Buffer"
    repeat_id: "RecurrentInput_2"
    shape: { dim:-1 dim: 1 dim: 28 }
  }
  variable {
    name: "Tanh_3"
    type: "Buffer"
    repeat_id: "RecurrentInput_2"
    shape: { dim:-1 dim: 1 dim: 28 }
  }
  variable {
    name: "Sigmoid_5"
    type: "Buffer"
    repeat_id: "RecurrentInput_2"
    shape: { dim:-1 dim: 1 dim: 28 }
  }
  variable {
    name: "Sigmoid_6"
    type: "Buffer"
    repeat_id: "RecurrentInput_2"
    shape: { dim:-1 dim: 1 dim: 28 }
  }
  variable {
    name: "Sigmoid_7"
    type: "Buffer"
    repeat_id: "RecurrentInput_2"
    shape: { dim:-1 dim: 1 dim: 28 }
  }
  variable {
    name: "Mul2_4"
    type: "Buffer"
    repeat_id: "RecurrentInput_2"
    shape: { dim:-1 dim: 1 dim: 28 }
  }
  variable {
    name: "Mul2_5"
    type: "Buffer"
    repeat_id: "RecurrentInput_2"
    shape: { dim:-1 dim: 1 dim: 28 }
  }
  variable {
    name: "Add2_2"
    type: "Buffer"
    repeat_id: "RecurrentInput_2"
    shape: { dim:-1 dim: 1 dim: 28 }
  }
  variable {
    name: "Tanh_4"
    type: "Buffer"
    repeat_id: "RecurrentInput_2"
    shape: { dim:-1 dim: 1 dim: 28 }
  }
  variable {
    name: "Mul2_6"
    type: "Buffer"
    repeat_id: "RecurrentInput_2"
    shape: { dim:-1 dim: 1 dim: 28 }
  }
  variable {
    name: "RecurrentOutput_2"
    type: "Buffer"
    shape: { dim:-1 dim: 1 dim: 28 dim: 28 }
  }
  variable {
    name: "Convolution"
    type: "Buffer"
    shape: { dim:-1 dim: 1 dim: 28 dim: 28 }
  }
  variable {
    name: "Sigmoid_4"
    type: "Buffer"
    shape: { dim:-1 dim: 1 dim: 28 dim: 28 }
  }
  variable {
    name: "SquaredError"
    type: "Buffer"
    shape: { dim:-1 dim: 1 dim: 28 dim: 28 }
  }
  function {
    name: "Flip"
    type: "Flip"
    input: "Input"
    output: "Flip"
    flip_param {
      axes: 2
    }
  }
  function {
    name: "RecurrentInput"
    type: "RecurrentInput"
    repeat_id: "RecurrentInput"
    input: "Flip"
    output: "RecurrentInput"
    recurrent_param {
      axis: 2
      repeat_id: "RecurrentInput"
    }
  }
  function {
    name: "Delay"
    type: "Delay"
    repeat_id: "RecurrentInput"
    input: "Add2"
    input: "Delay_Initial"
    output: "Delay"
    recurrent_param {
      repeat_id: "RecurrentInput"
    }
  }
  function {
    name: "Delay_2"
    type: "Delay"
    repeat_id: "RecurrentInput"
    input: "Mul2_2"
    input: "Delay_2_Initial"
    output: "Delay_2"
    recurrent_param {
      repeat_id: "RecurrentInput"
    }
  }
  function {
    name: "Concatenate"
    type: "Concatenate"
    repeat_id: "RecurrentInput"
    input: "Delay_2"
    input: "RecurrentInput"
    output: "Concatenate"
    concatenate_param {
      axis: 2
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
    name: "Affine_4"
    type: "Affine"
    repeat_id: "RecurrentInput"
    input: "Concatenate"
    input: "Affine_4/affine/W"
    input: "Affine_4/affine/b"
    output: "Affine_4"
    affine_param {
      base_axis: 1
    }
  }
  function {
    name: "Affine_5"
    type: "Affine"
    repeat_id: "RecurrentInput"
    input: "Concatenate"
    input: "Affine_5/affine/W"
    input: "Affine_5/affine/b"
    output: "Affine_5"
    affine_param {
      base_axis: 1
    }
  }
  function {
    name: "Affine_6"
    type: "Affine"
    repeat_id: "RecurrentInput"
    input: "Concatenate"
    input: "Affine_6/affine/W"
    input: "Affine_6/affine/b"
    output: "Affine_6"
    affine_param {
      base_axis: 1
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
    name: "Sigmoid"
    type: "Sigmoid"
    repeat_id: "RecurrentInput"
    input: "Affine_4"
    output: "Sigmoid"
  }
  function {
    name: "Sigmoid_2"
    type: "Sigmoid"
    repeat_id: "RecurrentInput"
    input: "Affine_5"
    output: "Sigmoid_2"
  }
  function {
    name: "Sigmoid_3"
    type: "Sigmoid"
    repeat_id: "RecurrentInput"
    input: "Affine_6"
    output: "Sigmoid_3"
  }
  function {
    name: "Mul2"
    type: "Mul2"
    repeat_id: "RecurrentInput"
    input: "Delay"
    input: "Sigmoid_2"
    output: "Mul2"
  }
  function {
    name: "Mul2_3"
    type: "Mul2"
    repeat_id: "RecurrentInput"
    input: "Tanh"
    input: "Sigmoid"
    output: "Mul2_3"
  }
  function {
    name: "Add2"
    type: "Add2"
    repeat_id: "RecurrentInput"
    input: "Mul2"
    input: "Mul2_3"
    output: "Add2"
    add2_param {
      inplace: True
    }
  }
  function {
    name: "Tanh_2"
    type: "Tanh"
    repeat_id: "RecurrentInput"
    input: "Add2"
    output: "Tanh_2"
  }
  function {
    name: "Mul2_2"
    type: "Mul2"
    repeat_id: "RecurrentInput"
    input: "Tanh_2"
    input: "Sigmoid_3"
    output: "Mul2_2"
  }
  function {
    name: "RecurrentOutput"
    type: "RecurrentOutput"
    input: "Mul2_2"
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
      stop: 256
      step: 1
      step: 1
      step: 1
    }
  }
  function {
    name: "Unpooling"
    type: "Unpooling"
    input: "Slice"
    output: "Unpooling"
    unpooling_param {
      kernel: { dim: 28 dim: 1 }
    }
  }
  function {
    name: "RecurrentInput_2"
    type: "RecurrentInput"
    repeat_id: "RecurrentInput_2"
    input: "Unpooling"
    output: "RecurrentInput_2"
    recurrent_param {
      axis: 2
      repeat_id: "RecurrentInput_2"
    }
  }
  function {
    name: "Delay_3"
    type: "Delay"
    repeat_id: "RecurrentInput_2"
    input: "Add2_2"
    input: "Delay_3_Initial"
    output: "Delay_3"
    recurrent_param {
      repeat_id: "RecurrentInput_2"
    }
  }
  function {
    name: "Delay_4"
    type: "Delay"
    repeat_id: "RecurrentInput_2"
    input: "Mul2_6"
    input: "Delay_4_Initial"
    output: "Delay_4"
    recurrent_param {
      repeat_id: "RecurrentInput_2"
    }
  }
  function {
    name: "Concatenate_2"
    type: "Concatenate"
    repeat_id: "RecurrentInput_2"
    input: "Delay_4"
    input: "RecurrentInput_2"
    output: "Concatenate_2"
    concatenate_param {
      axis: 2
    }
  }
  function {
    name: "Affine_7"
    type: "Affine"
    repeat_id: "RecurrentInput_2"
    input: "Concatenate_2"
    input: "Affine_7/affine/W"
    input: "Affine_7/affine/b"
    output: "Affine_7"
    affine_param {
      base_axis: 1
    }
  }
  function {
    name: "Affine_8"
    type: "Affine"
    repeat_id: "RecurrentInput_2"
    input: "Concatenate_2"
    input: "Affine_8/affine/W"
    input: "Affine_8/affine/b"
    output: "Affine_8"
    affine_param {
      base_axis: 1
    }
  }
  function {
    name: "Affine_9"
    type: "Affine"
    repeat_id: "RecurrentInput_2"
    input: "Concatenate_2"
    input: "Affine_9/affine/W"
    input: "Affine_9/affine/b"
    output: "Affine_9"
    affine_param {
      base_axis: 1
    }
  }
  function {
    name: "Affine_10"
    type: "Affine"
    repeat_id: "RecurrentInput_2"
    input: "Concatenate_2"
    input: "Affine_10/affine/W"
    input: "Affine_10/affine/b"
    output: "Affine_10"
    affine_param {
      base_axis: 1
    }
  }
  function {
    name: "Tanh_3"
    type: "Tanh"
    repeat_id: "RecurrentInput_2"
    input: "Affine_7"
    output: "Tanh_3"
  }
  function {
    name: "Sigmoid_5"
    type: "Sigmoid"
    repeat_id: "RecurrentInput_2"
    input: "Affine_8"
    output: "Sigmoid_5"
  }
  function {
    name: "Sigmoid_6"
    type: "Sigmoid"
    repeat_id: "RecurrentInput_2"
    input: "Affine_9"
    output: "Sigmoid_6"
  }
  function {
    name: "Sigmoid_7"
    type: "Sigmoid"
    repeat_id: "RecurrentInput_2"
    input: "Affine_10"
    output: "Sigmoid_7"
  }
  function {
    name: "Mul2_4"
    type: "Mul2"
    repeat_id: "RecurrentInput_2"
    input: "Sigmoid_6"
    input: "Delay_3"
    output: "Mul2_4"
  }
  function {
    name: "Mul2_5"
    type: "Mul2"
    repeat_id: "RecurrentInput_2"
    input: "Tanh_3"
    input: "Sigmoid_5"
    output: "Mul2_5"
  }
  function {
    name: "Add2_2"
    type: "Add2"
    repeat_id: "RecurrentInput_2"
    input: "Mul2_4"
    input: "Mul2_5"
    output: "Add2_2"
    add2_param {
      inplace: True
    }
  }
  function {
    name: "Tanh_4"
    type: "Tanh"
    repeat_id: "RecurrentInput_2"
    input: "Add2_2"
    output: "Tanh_4"
  }
  function {
    name: "Mul2_6"
    type: "Mul2"
    repeat_id: "RecurrentInput_2"
    input: "Tanh_4"
    input: "Sigmoid_7"
    output: "Mul2_6"
  }
  function {
    name: "RecurrentOutput_2"
    type: "RecurrentOutput"
    input: "Mul2_6"
    output: "RecurrentOutput_2"
    recurrent_param {
      axis: 2
      repeat_id: "RecurrentInput_2"
      length: 28
    }
  }
  function {
    name: "Convolution"
    type: "Convolution"
    input: "RecurrentOutput_2"
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
    name: "Sigmoid_4"
    type: "Sigmoid"
    input: "Convolution"
    output: "Sigmoid_4"
  }
  function {
    name: "SquaredError"
    type: "SquaredError"
    input: "Sigmoid_4"
    input: "SquaredError_T"
    output: "SquaredError"
  }
}
network {
  name: "MainRuntime"
  batch_size: 64
  repeat_info {
    id: "RecurrentInput"
    times: 28
  }
  repeat_info {
    id: "RecurrentInput_2"
    times: 28
  }
  variable {
    name: "Input"
    type: "Buffer"
    shape: { dim:-1 dim: 1 dim: 28 dim: 28 }
  }
  variable {
    name: "Delay_Initial"
    type: "Buffer"
    shape: { dim:-1 dim: 1 dim: 256 }
  }
  variable {
    name: "Delay_2_Initial"
    type: "Buffer"
    shape: { dim:-1 dim: 1 dim: 256 }
  }
  variable {
    name: "Affine_2/affine/W"
    type: "Parameter"
    shape: { dim: 284 dim: 1 dim: 256 }
    initializer {
      type: "NormalAffineGlorot"
      multiplier: 1
    }
  }
  variable {
    name: "Affine_2/affine/b"
    type: "Parameter"
    shape: { dim: 1 dim: 256 }
    initializer {
      type: "Constant"
      multiplier: 0
    }
  }
  variable {
    name: "Affine_4/affine/W"
    type: "Parameter"
    shape: { dim: 284 dim: 1 dim: 256 }
    initializer {
      type: "NormalAffineGlorot"
      multiplier: 1
    }
  }
  variable {
    name: "Affine_4/affine/b"
    type: "Parameter"
    shape: { dim: 1 dim: 256 }
    initializer {
      type: "Constant"
      multiplier: 0
    }
  }
  variable {
    name: "Affine_5/affine/W"
    type: "Parameter"
    shape: { dim: 284 dim: 1 dim: 256 }
    initializer {
      type: "NormalAffineGlorot"
      multiplier: 1
    }
  }
  variable {
    name: "Affine_5/affine/b"
    type: "Parameter"
    shape: { dim: 1 dim: 256 }
    initializer {
      type: "Constant"
      multiplier: 0
    }
  }
  variable {
    name: "Affine_6/affine/W"
    type: "Parameter"
    shape: { dim: 284 dim: 1 dim: 256 }
    initializer {
      type: "NormalAffineGlorot"
      multiplier: 1
    }
  }
  variable {
    name: "Affine_6/affine/b"
    type: "Parameter"
    shape: { dim: 1 dim: 256 }
    initializer {
      type: "Constant"
      multiplier: 0
    }
  }
  variable {
    name: "Delay_3_Initial"
    type: "Buffer"
    shape: { dim:-1 dim: 1 dim: 28 }
  }
  variable {
    name: "Delay_4_Initial"
    type: "Buffer"
    shape: { dim:-1 dim: 1 dim: 28 }
  }
  variable {
    name: "Affine_7/affine/W"
    type: "Parameter"
    shape: { dim: 284 dim: 1 dim: 28 }
    initializer {
      type: "NormalAffineGlorot"
      multiplier: 1
    }
  }
  variable {
    name: "Affine_7/affine/b"
    type: "Parameter"
    shape: { dim: 1 dim: 28 }
    initializer {
      type: "Constant"
      multiplier: 0
    }
  }
  variable {
    name: "Affine_8/affine/W"
    type: "Parameter"
    shape: { dim: 284 dim: 1 dim: 28 }
    initializer {
      type: "NormalAffineGlorot"
      multiplier: 1
    }
  }
  variable {
    name: "Affine_8/affine/b"
    type: "Parameter"
    shape: { dim: 1 dim: 28 }
    initializer {
      type: "Constant"
      multiplier: 0
    }
  }
  variable {
    name: "Affine_9/affine/W"
    type: "Parameter"
    shape: { dim: 284 dim: 1 dim: 28 }
    initializer {
      type: "NormalAffineGlorot"
      multiplier: 1
    }
  }
  variable {
    name: "Affine_9/affine/b"
    type: "Parameter"
    shape: { dim: 1 dim: 28 }
    initializer {
      type: "Constant"
      multiplier: 0
    }
  }
  variable {
    name: "Affine_10/affine/W"
    type: "Parameter"
    shape: { dim: 284 dim: 1 dim: 28 }
    initializer {
      type: "NormalAffineGlorot"
      multiplier: 1
    }
  }
  variable {
    name: "Affine_10/affine/b"
    type: "Parameter"
    shape: { dim: 1 dim: 28 }
    initializer {
      type: "Constant"
      multiplier: 0
    }
  }
  variable {
    name: "Convolution/conv/W"
    type: "Parameter"
    shape: { dim: 1 dim: 1 dim: 1 dim: 1 }
    initializer {
      type: "NormalConvolutionGlorot"
      multiplier: 1
    }
  }
  variable {
    name: "Convolution/conv/b"
    type: "Parameter"
    shape: { dim: 1 }
    initializer {
      type: "Constant"
      multiplier: 0
    }
  }
  variable {
    name: "Flip"
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
    name: "Delay"
    type: "Buffer"
    repeat_id: "RecurrentInput"
    shape: { dim:-1 dim: 1 dim: 256 }
  }
  variable {
    name: "Delay_2"
    type: "Buffer"
    repeat_id: "RecurrentInput"
    shape: { dim:-1 dim: 1 dim: 256 }
  }
  variable {
    name: "Concatenate"
    type: "Buffer"
    repeat_id: "RecurrentInput"
    shape: { dim:-1 dim: 1 dim: 284 }
  }
  variable {
    name: "Affine_2"
    type: "Buffer"
    repeat_id: "RecurrentInput"
    shape: { dim:-1 dim: 1 dim: 256 }
  }
  variable {
    name: "Affine_4"
    type: "Buffer"
    repeat_id: "RecurrentInput"
    shape: { dim:-1 dim: 1 dim: 256 }
  }
  variable {
    name: "Affine_5"
    type: "Buffer"
    repeat_id: "RecurrentInput"
    shape: { dim:-1 dim: 1 dim: 256 }
  }
  variable {
    name: "Affine_6"
    type: "Buffer"
    repeat_id: "RecurrentInput"
    shape: { dim:-1 dim: 1 dim: 256 }
  }
  variable {
    name: "Tanh"
    type: "Buffer"
    repeat_id: "RecurrentInput"
    shape: { dim:-1 dim: 1 dim: 256 }
  }
  variable {
    name: "Sigmoid"
    type: "Buffer"
    repeat_id: "RecurrentInput"
    shape: { dim:-1 dim: 1 dim: 256 }
  }
  variable {
    name: "Sigmoid_2"
    type: "Buffer"
    repeat_id: "RecurrentInput"
    shape: { dim:-1 dim: 1 dim: 256 }
  }
  variable {
    name: "Sigmoid_3"
    type: "Buffer"
    repeat_id: "RecurrentInput"
    shape: { dim:-1 dim: 1 dim: 256 }
  }
  variable {
    name: "Mul2"
    type: "Buffer"
    repeat_id: "RecurrentInput"
    shape: { dim:-1 dim: 1 dim: 256 }
  }
  variable {
    name: "Mul2_3"
    type: "Buffer"
    repeat_id: "RecurrentInput"
    shape: { dim:-1 dim: 1 dim: 256 }
  }
  variable {
    name: "Add2"
    type: "Buffer"
    repeat_id: "RecurrentInput"
    shape: { dim:-1 dim: 1 dim: 256 }
  }
  variable {
    name: "Tanh_2"
    type: "Buffer"
    repeat_id: "RecurrentInput"
    shape: { dim:-1 dim: 1 dim: 256 }
  }
  variable {
    name: "Mul2_2"
    type: "Buffer"
    repeat_id: "RecurrentInput"
    shape: { dim:-1 dim: 1 dim: 256 }
  }
  variable {
    name: "RecurrentOutput"
    type: "Buffer"
    shape: { dim:-1 dim: 1 dim: 28 dim: 256 }
  }
  variable {
    name: "Slice"
    type: "Buffer"
    shape: { dim:-1 dim: 1 dim: 1 dim: 256 }
  }
  variable {
    name: "Unpooling"
    type: "Buffer"
    shape: { dim:-1 dim: 1 dim: 28 dim: 256 }
  }
  variable {
    name: "RecurrentInput_2"
    type: "Buffer"
    repeat_id: "RecurrentInput_2"
    shape: { dim:-1 dim: 1 dim: 256 }
  }
  variable {
    name: "Delay_3"
    type: "Buffer"
    repeat_id: "RecurrentInput_2"
    shape: { dim:-1 dim: 1 dim: 28 }
  }
  variable {
    name: "Delay_4"
    type: "Buffer"
    repeat_id: "RecurrentInput_2"
    shape: { dim:-1 dim: 1 dim: 28 }
  }
  variable {
    name: "Concatenate_2"
    type: "Buffer"
    repeat_id: "RecurrentInput_2"
    shape: { dim:-1 dim: 1 dim: 284 }
  }
  variable {
    name: "Affine_7"
    type: "Buffer"
    repeat_id: "RecurrentInput_2"
    shape: { dim:-1 dim: 1 dim: 28 }
  }
  variable {
    name: "Affine_8"
    type: "Buffer"
    repeat_id: "RecurrentInput_2"
    shape: { dim:-1 dim: 1 dim: 28 }
  }
  variable {
    name: "Affine_9"
    type: "Buffer"
    repeat_id: "RecurrentInput_2"
    shape: { dim:-1 dim: 1 dim: 28 }
  }
  variable {
    name: "Affine_10"
    type: "Buffer"
    repeat_id: "RecurrentInput_2"
    shape: { dim:-1 dim: 1 dim: 28 }
  }
  variable {
    name: "Tanh_3"
    type: "Buffer"
    repeat_id: "RecurrentInput_2"
    shape: { dim:-1 dim: 1 dim: 28 }
  }
  variable {
    name: "Sigmoid_5"
    type: "Buffer"
    repeat_id: "RecurrentInput_2"
    shape: { dim:-1 dim: 1 dim: 28 }
  }
  variable {
    name: "Sigmoid_6"
    type: "Buffer"
    repeat_id: "RecurrentInput_2"
    shape: { dim:-1 dim: 1 dim: 28 }
  }
  variable {
    name: "Sigmoid_7"
    type: "Buffer"
    repeat_id: "RecurrentInput_2"
    shape: { dim:-1 dim: 1 dim: 28 }
  }
  variable {
    name: "Mul2_4"
    type: "Buffer"
    repeat_id: "RecurrentInput_2"
    shape: { dim:-1 dim: 1 dim: 28 }
  }
  variable {
    name: "Mul2_5"
    type: "Buffer"
    repeat_id: "RecurrentInput_2"
    shape: { dim:-1 dim: 1 dim: 28 }
  }
  variable {
    name: "Add2_2"
    type: "Buffer"
    repeat_id: "RecurrentInput_2"
    shape: { dim:-1 dim: 1 dim: 28 }
  }
  variable {
    name: "Tanh_4"
    type: "Buffer"
    repeat_id: "RecurrentInput_2"
    shape: { dim:-1 dim: 1 dim: 28 }
  }
  variable {
    name: "Mul2_6"
    type: "Buffer"
    repeat_id: "RecurrentInput_2"
    shape: { dim:-1 dim: 1 dim: 28 }
  }
  variable {
    name: "RecurrentOutput_2"
    type: "Buffer"
    shape: { dim:-1 dim: 1 dim: 28 dim: 28 }
  }
  variable {
    name: "Convolution"
    type: "Buffer"
    shape: { dim:-1 dim: 1 dim: 28 dim: 28 }
  }
  variable {
    name: "Sigmoid_4"
    type: "Buffer"
    shape: { dim:-1 dim: 1 dim: 28 dim: 28 }
  }
  function {
    name: "Flip"
    type: "Flip"
    input: "Input"
    output: "Flip"
    flip_param {
      axes: 2
    }
  }
  function {
    name: "RecurrentInput"
    type: "RecurrentInput"
    repeat_id: "RecurrentInput"
    input: "Flip"
    output: "RecurrentInput"
    recurrent_param {
      axis: 2
      repeat_id: "RecurrentInput"
    }
  }
  function {
    name: "Delay"
    type: "Delay"
    repeat_id: "RecurrentInput"
    input: "Add2"
    input: "Delay_Initial"
    output: "Delay"
    recurrent_param {
      repeat_id: "RecurrentInput"
    }
  }
  function {
    name: "Delay_2"
    type: "Delay"
    repeat_id: "RecurrentInput"
    input: "Mul2_2"
    input: "Delay_2_Initial"
    output: "Delay_2"
    recurrent_param {
      repeat_id: "RecurrentInput"
    }
  }
  function {
    name: "Concatenate"
    type: "Concatenate"
    repeat_id: "RecurrentInput"
    input: "Delay_2"
    input: "RecurrentInput"
    output: "Concatenate"
    concatenate_param {
      axis: 2
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
    name: "Affine_4"
    type: "Affine"
    repeat_id: "RecurrentInput"
    input: "Concatenate"
    input: "Affine_4/affine/W"
    input: "Affine_4/affine/b"
    output: "Affine_4"
    affine_param {
      base_axis: 1
    }
  }
  function {
    name: "Affine_5"
    type: "Affine"
    repeat_id: "RecurrentInput"
    input: "Concatenate"
    input: "Affine_5/affine/W"
    input: "Affine_5/affine/b"
    output: "Affine_5"
    affine_param {
      base_axis: 1
    }
  }
  function {
    name: "Affine_6"
    type: "Affine"
    repeat_id: "RecurrentInput"
    input: "Concatenate"
    input: "Affine_6/affine/W"
    input: "Affine_6/affine/b"
    output: "Affine_6"
    affine_param {
      base_axis: 1
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
    name: "Sigmoid"
    type: "Sigmoid"
    repeat_id: "RecurrentInput"
    input: "Affine_4"
    output: "Sigmoid"
  }
  function {
    name: "Sigmoid_2"
    type: "Sigmoid"
    repeat_id: "RecurrentInput"
    input: "Affine_5"
    output: "Sigmoid_2"
  }
  function {
    name: "Sigmoid_3"
    type: "Sigmoid"
    repeat_id: "RecurrentInput"
    input: "Affine_6"
    output: "Sigmoid_3"
  }
  function {
    name: "Mul2"
    type: "Mul2"
    repeat_id: "RecurrentInput"
    input: "Delay"
    input: "Sigmoid_2"
    output: "Mul2"
  }
  function {
    name: "Mul2_3"
    type: "Mul2"
    repeat_id: "RecurrentInput"
    input: "Tanh"
    input: "Sigmoid"
    output: "Mul2_3"
  }
  function {
    name: "Add2"
    type: "Add2"
    repeat_id: "RecurrentInput"
    input: "Mul2"
    input: "Mul2_3"
    output: "Add2"
    add2_param {
      inplace: True
    }
  }
  function {
    name: "Tanh_2"
    type: "Tanh"
    repeat_id: "RecurrentInput"
    input: "Add2"
    output: "Tanh_2"
  }
  function {
    name: "Mul2_2"
    type: "Mul2"
    repeat_id: "RecurrentInput"
    input: "Tanh_2"
    input: "Sigmoid_3"
    output: "Mul2_2"
  }
  function {
    name: "RecurrentOutput"
    type: "RecurrentOutput"
    input: "Mul2_2"
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
      stop: 256
      step: 1
      step: 1
      step: 1
    }
  }
  function {
    name: "Unpooling"
    type: "Unpooling"
    input: "Slice"
    output: "Unpooling"
    unpooling_param {
      kernel: { dim: 28 dim: 1 }
    }
  }
  function {
    name: "RecurrentInput_2"
    type: "RecurrentInput"
    repeat_id: "RecurrentInput_2"
    input: "Unpooling"
    output: "RecurrentInput_2"
    recurrent_param {
      axis: 2
      repeat_id: "RecurrentInput_2"
    }
  }
  function {
    name: "Delay_3"
    type: "Delay"
    repeat_id: "RecurrentInput_2"
    input: "Add2_2"
    input: "Delay_3_Initial"
    output: "Delay_3"
    recurrent_param {
      repeat_id: "RecurrentInput_2"
    }
  }
  function {
    name: "Delay_4"
    type: "Delay"
    repeat_id: "RecurrentInput_2"
    input: "Mul2_6"
    input: "Delay_4_Initial"
    output: "Delay_4"
    recurrent_param {
      repeat_id: "RecurrentInput_2"
    }
  }
  function {
    name: "Concatenate_2"
    type: "Concatenate"
    repeat_id: "RecurrentInput_2"
    input: "Delay_4"
    input: "RecurrentInput_2"
    output: "Concatenate_2"
    concatenate_param {
      axis: 2
    }
  }
  function {
    name: "Affine_7"
    type: "Affine"
    repeat_id: "RecurrentInput_2"
    input: "Concatenate_2"
    input: "Affine_7/affine/W"
    input: "Affine_7/affine/b"
    output: "Affine_7"
    affine_param {
      base_axis: 1
    }
  }
  function {
    name: "Affine_8"
    type: "Affine"
    repeat_id: "RecurrentInput_2"
    input: "Concatenate_2"
    input: "Affine_8/affine/W"
    input: "Affine_8/affine/b"
    output: "Affine_8"
    affine_param {
      base_axis: 1
    }
  }
  function {
    name: "Affine_9"
    type: "Affine"
    repeat_id: "RecurrentInput_2"
    input: "Concatenate_2"
    input: "Affine_9/affine/W"
    input: "Affine_9/affine/b"
    output: "Affine_9"
    affine_param {
      base_axis: 1
    }
  }
  function {
    name: "Affine_10"
    type: "Affine"
    repeat_id: "RecurrentInput_2"
    input: "Concatenate_2"
    input: "Affine_10/affine/W"
    input: "Affine_10/affine/b"
    output: "Affine_10"
    affine_param {
      base_axis: 1
    }
  }
  function {
    name: "Tanh_3"
    type: "Tanh"
    repeat_id: "RecurrentInput_2"
    input: "Affine_7"
    output: "Tanh_3"
  }
  function {
    name: "Sigmoid_5"
    type: "Sigmoid"
    repeat_id: "RecurrentInput_2"
    input: "Affine_8"
    output: "Sigmoid_5"
  }
  function {
    name: "Sigmoid_6"
    type: "Sigmoid"
    repeat_id: "RecurrentInput_2"
    input: "Affine_9"
    output: "Sigmoid_6"
  }
  function {
    name: "Sigmoid_7"
    type: "Sigmoid"
    repeat_id: "RecurrentInput_2"
    input: "Affine_10"
    output: "Sigmoid_7"
  }
  function {
    name: "Mul2_4"
    type: "Mul2"
    repeat_id: "RecurrentInput_2"
    input: "Sigmoid_6"
    input: "Delay_3"
    output: "Mul2_4"
  }
  function {
    name: "Mul2_5"
    type: "Mul2"
    repeat_id: "RecurrentInput_2"
    input: "Tanh_3"
    input: "Sigmoid_5"
    output: "Mul2_5"
  }
  function {
    name: "Add2_2"
    type: "Add2"
    repeat_id: "RecurrentInput_2"
    input: "Mul2_4"
    input: "Mul2_5"
    output: "Add2_2"
    add2_param {
      inplace: True
    }
  }
  function {
    name: "Tanh_4"
    type: "Tanh"
    repeat_id: "RecurrentInput_2"
    input: "Add2_2"
    output: "Tanh_4"
  }
  function {
    name: "Mul2_6"
    type: "Mul2"
    repeat_id: "RecurrentInput_2"
    input: "Tanh_4"
    input: "Sigmoid_7"
    output: "Mul2_6"
  }
  function {
    name: "RecurrentOutput_2"
    type: "RecurrentOutput"
    input: "Mul2_6"
    output: "RecurrentOutput_2"
    recurrent_param {
      axis: 2
      repeat_id: "RecurrentInput_2"
      length: 28
    }
  }
  function {
    name: "Convolution"
    type: "Convolution"
    input: "RecurrentOutput_2"
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
    name: "Sigmoid_4"
    type: "Sigmoid"
    input: "Convolution"
    output: "Sigmoid_4"
  }
}
dataset {
  name: "Training"
  uri: "C:\\Utils\\neural_network_console_internal_v120\\samples\\sample_dataset\\mnist\\small_mnist_4or9_training.csv"
  cache_dir: "C:\\Utils\\neural_network_console_internal_v120\\samples\\sample_dataset\\mnist\\small_mnist_4or9_training.cache"
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
  generator_variable {
    variable_name: "Delay_Initial"
    type: "Constant"
    multiplier: 0
  }
  generator_variable {
    variable_name: "Delay_2_Initial"
    type: "Constant"
    multiplier: 0
  }
  generator_variable {
    variable_name: "Delay_3_Initial"
    type: "Constant"
    multiplier: 0
  }
  generator_variable {
    variable_name: "Delay_4_Initial"
    type: "Constant"
    multiplier: 0
  }
  data_variable {
    variable_name: "SquaredError_T"
    data_name: "x"
  }
  loss_variable {
    variable_name: "SquaredError"
  }
  parameter_variable {
    variable_name: "Affine_2/affine/W"
    learning_rate_multiplier: 1
  }
  parameter_variable {
    variable_name: "Affine_2/affine/b"
    learning_rate_multiplier: 1
  }
  parameter_variable {
    variable_name: "Affine_4/affine/W"
    learning_rate_multiplier: 1
  }
  parameter_variable {
    variable_name: "Affine_4/affine/b"
    learning_rate_multiplier: 1
  }
  parameter_variable {
    variable_name: "Affine_5/affine/W"
    learning_rate_multiplier: 1
  }
  parameter_variable {
    variable_name: "Affine_5/affine/b"
    learning_rate_multiplier: 1
  }
  parameter_variable {
    variable_name: "Affine_6/affine/W"
    learning_rate_multiplier: 1
  }
  parameter_variable {
    variable_name: "Affine_6/affine/b"
    learning_rate_multiplier: 1
  }
  parameter_variable {
    variable_name: "Affine_7/affine/W"
    learning_rate_multiplier: 1
  }
  parameter_variable {
    variable_name: "Affine_7/affine/b"
    learning_rate_multiplier: 1
  }
  parameter_variable {
    variable_name: "Affine_8/affine/W"
    learning_rate_multiplier: 1
  }
  parameter_variable {
    variable_name: "Affine_8/affine/b"
    learning_rate_multiplier: 1
  }
  parameter_variable {
    variable_name: "Affine_9/affine/W"
    learning_rate_multiplier: 1
  }
  parameter_variable {
    variable_name: "Affine_9/affine/b"
    learning_rate_multiplier: 1
  }
  parameter_variable {
    variable_name: "Affine_10/affine/W"
    learning_rate_multiplier: 1
  }
  parameter_variable {
    variable_name: "Affine_10/affine/b"
    learning_rate_multiplier: 1
  }
  parameter_variable {
    variable_name: "Convolution/conv/W"
    learning_rate_multiplier: 1
  }
  parameter_variable {
    variable_name: "Convolution/conv/b"
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
  generator_variable {
    variable_name: "Delay_Initial"
    type: "Constant"
    multiplier: 0
  }
  generator_variable {
    variable_name: "Delay_2_Initial"
    type: "Constant"
    multiplier: 0
  }
  generator_variable {
    variable_name: "Delay_3_Initial"
    type: "Constant"
    multiplier: 0
  }
  generator_variable {
    variable_name: "Delay_4_Initial"
    type: "Constant"
    multiplier: 0
  }
  data_variable {
    variable_name: "SquaredError_T"
    data_name: "x"
  }
  monitor_variable {
    type: "Error"
    variable_name: "SquaredError"
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
    multiplier: 0
  }
  generator_variable {
    variable_name: "Delay_2_Initial"
    type: "Constant"
    multiplier: 0
  }
  generator_variable {
    variable_name: "Delay_3_Initial"
    type: "Constant"
    multiplier: 0
  }
  generator_variable {
    variable_name: "Delay_4_Initial"
    type: "Constant"
    multiplier: 0
  }
  data_variable {
    variable_name: "SquaredError_T"
    data_name: "x"
  }
  monitor_variable {
    type: "Error"
    variable_name: "SquaredError"
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
    multiplier: 0
  }
  generator_variable {
    variable_name: "Delay_2_Initial"
    type: "Constant"
    multiplier: 0
  }
  generator_variable {
    variable_name: "Delay_3_Initial"
    type: "Constant"
    multiplier: 0
  }
  generator_variable {
    variable_name: "Delay_4_Initial"
    type: "Constant"
    multiplier: 0
  }
  output_variable {
    variable_name: "Sigmoid_4"
    data_name: "x'"
  }
  parameter_variable {
    variable_name: "Affine_2/affine/W"
  }
  parameter_variable {
    variable_name: "Affine_2/affine/b"
  }
  parameter_variable {
    variable_name: "Affine_4/affine/W"
  }
  parameter_variable {
    variable_name: "Affine_4/affine/b"
  }
  parameter_variable {
    variable_name: "Affine_5/affine/W"
  }
  parameter_variable {
    variable_name: "Affine_5/affine/b"
  }
  parameter_variable {
    variable_name: "Affine_6/affine/W"
  }
  parameter_variable {
    variable_name: "Affine_6/affine/b"
  }
  parameter_variable {
    variable_name: "Affine_7/affine/W"
  }
  parameter_variable {
    variable_name: "Affine_7/affine/b"
  }
  parameter_variable {
    variable_name: "Affine_8/affine/W"
  }
  parameter_variable {
    variable_name: "Affine_8/affine/b"
  }
  parameter_variable {
    variable_name: "Affine_9/affine/W"
  }
  parameter_variable {
    variable_name: "Affine_9/affine/b"
  }
  parameter_variable {
    variable_name: "Affine_10/affine/W"
  }
  parameter_variable {
    variable_name: "Affine_10/affine/b"
  }
  parameter_variable {
    variable_name: "Convolution/conv/W"
  }
  parameter_variable {
    variable_name: "Convolution/conv/b"
  }
}
'''
# The nntxt is generated from /home/woody/aid/nnabla-sample-data/sample_project/tutorial/recurrent_neural_networks/long_short_term_memory(LSTM).files/20170804_142559/net.nntxt.
N00000012 = r'''global_config {
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
  name: "Main_"
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
    name: "Affine_3/affine/W"
    type: "Parameter"
    shape: { dim: 56 dim: 1 dim: 28 }
    initializer {
      type: "NormalAffineGlorot"
      multiplier: 1.0
    }
  }
  variable {
    name: "Affine_3/affine/b"
    type: "Parameter"
    shape: { dim: 1 dim: 28 }
    initializer {
      type: "Constant"
      multiplier: 0.0
    }
  }
  variable {
    name: "Affine_4/affine/W"
    type: "Parameter"
    shape: { dim: 56 dim: 1 dim: 28 }
    initializer {
      type: "NormalAffineGlorot"
      multiplier: 1.0
    }
  }
  variable {
    name: "Affine_4/affine/b"
    type: "Parameter"
    shape: { dim: 1 dim: 28 }
    initializer {
      type: "Constant"
      multiplier: 0.0
    }
  }
  variable {
    name: "Affine_5/affine/W"
    type: "Parameter"
    shape: { dim: 56 dim: 1 dim: 28 }
    initializer {
      type: "NormalAffineGlorot"
      multiplier: 1.0
    }
  }
  variable {
    name: "Affine_5/affine/b"
    type: "Parameter"
    shape: { dim: 1 dim: 28 }
    initializer {
      type: "Constant"
      multiplier: 0.0
    }
  }
  variable {
    name: "Affine_6/affine/W"
    type: "Parameter"
    shape: { dim: 56 dim: 1 dim: 28 }
    initializer {
      type: "NormalAffineGlorot"
      multiplier: 1.0
    }
  }
  variable {
    name: "Affine_6/affine/b"
    type: "Parameter"
    shape: { dim: 1 dim: 28 }
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
    name: "Delay_2_Initial"
    type: "Buffer"
    shape: { dim:-1 dim: 1 dim: 28 }
  }
  variable {
    name: "BinaryCrossEntropy_T"
    type: "Buffer"
    shape: { dim:-1 dim: 1 }
  }
  variable {
    name: "Affine_2/affine/W"
    type: "Parameter"
    shape: { dim: 28 dim: 1 }
    initializer {
      type: "NormalAffineGlorot"
      multiplier: 1.0
    }
  }
  variable {
    name: "Affine_2/affine/b"
    type: "Parameter"
    shape: { dim: 1 }
    initializer {
      type: "Constant"
      multiplier: 0.0
    }
  }
  variable {
    name: "Affine_3"
    type: "Buffer"
    repeat_id: "RecurrentInput"
    shape: { dim:-1 dim: 1 dim: 28 }
  }
  variable {
    name: "Tanh"
    type: "Buffer"
    repeat_id: "RecurrentInput"
    shape: { dim:-1 dim: 1 dim: 28 }
  }
  variable {
    name: "Affine_4"
    type: "Buffer"
    repeat_id: "RecurrentInput"
    shape: { dim:-1 dim: 1 dim: 28 }
  }
  variable {
    name: "Sigmoid"
    type: "Buffer"
    repeat_id: "RecurrentInput"
    shape: { dim:-1 dim: 1 dim: 28 }
  }
  variable {
    name: "Mul2"
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
    name: "Add2"
    type: "Buffer"
    repeat_id: "RecurrentInput"
    shape: { dim:-1 dim: 1 dim: 28 }
  }
  variable {
    name: "Mul2_2"
    type: "Buffer"
    repeat_id: "RecurrentInput"
    shape: { dim:-1 dim: 1 dim: 28 }
  }
  variable {
    name: "Affine_5"
    type: "Buffer"
    repeat_id: "RecurrentInput"
    shape: { dim:-1 dim: 1 dim: 28 }
  }
  variable {
    name: "Sigmoid_2"
    type: "Buffer"
    repeat_id: "RecurrentInput"
    shape: { dim:-1 dim: 1 dim: 28 }
  }
  variable {
    name: "Tanh_2"
    type: "Buffer"
    repeat_id: "RecurrentInput"
    shape: { dim:-1 dim: 1 dim: 28 }
  }
  variable {
    name: "Affine_6"
    type: "Buffer"
    repeat_id: "RecurrentInput"
    shape: { dim:-1 dim: 1 dim: 28 }
  }
  variable {
    name: "Sigmoid_4"
    type: "Buffer"
    repeat_id: "RecurrentInput"
    shape: { dim:-1 dim: 1 dim: 28 }
  }
  variable {
    name: "Mul2_3"
    type: "Buffer"
    repeat_id: "RecurrentInput"
    shape: { dim:-1 dim: 1 dim: 28 }
  }
  variable {
    name: "Delay"
    type: "Buffer"
    repeat_id: "RecurrentInput"
    shape: { dim:-1 dim: 1 dim: 28 }
  }
  variable {
    name: "Delay_2"
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
    name: "BinaryCrossEntropy"
    type: "Buffer"
    shape: { dim:-1 dim: 1 }
  }
  variable {
    name: "Affine_2"
    type: "Buffer"
    shape: { dim:-1 dim: 1 }
  }
  variable {
    name: "Sigmoid_3"
    type: "Buffer"
    shape: { dim:-1 dim: 1 }
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
    name: "Affine_3"
    type: "Affine"
    repeat_id: "RecurrentInput"
    input: "Concatenate"
    input: "Affine_3/affine/W"
    input: "Affine_3/affine/b"
    output: "Affine_3"
    affine_param {
      base_axis: 1
    }
  }
  function {
    name: "Tanh"
    type: "Tanh"
    repeat_id: "RecurrentInput"
    input: "Affine_3"
    output: "Tanh"
  }
  function {
    name: "Affine_4"
    type: "Affine"
    repeat_id: "RecurrentInput"
    input: "Concatenate"
    input: "Affine_4/affine/W"
    input: "Affine_4/affine/b"
    output: "Affine_4"
    affine_param {
      base_axis: 1
    }
  }
  function {
    name: "Sigmoid"
    type: "Sigmoid"
    repeat_id: "RecurrentInput"
    input: "Affine_4"
    output: "Sigmoid"
  }
  function {
    name: "Mul2"
    type: "Mul2"
    repeat_id: "RecurrentInput"
    input: "Sigmoid"
    input: "Tanh"
    output: "Mul2"
  }
  function {
    name: "Concatenate"
    type: "Concatenate"
    repeat_id: "RecurrentInput"
    input: "Delay_2"
    input: "RecurrentInput"
    output: "Concatenate"
    concatenate_param {
      axis: 1
    }
  }
  function {
    name: "Add2"
    type: "Add2"
    repeat_id: "RecurrentInput"
    input: "Mul2"
    input: "Mul2_2"
    output: "Add2"
    add2_param {
      inplace: True
    }
  }
  function {
    name: "Mul2_2"
    type: "Mul2"
    repeat_id: "RecurrentInput"
    input: "Sigmoid_2"
    input: "Delay"
    output: "Mul2_2"
  }
  function {
    name: "Affine_5"
    type: "Affine"
    repeat_id: "RecurrentInput"
    input: "Concatenate"
    input: "Affine_5/affine/W"
    input: "Affine_5/affine/b"
    output: "Affine_5"
    affine_param {
      base_axis: 1
    }
  }
  function {
    name: "Sigmoid_2"
    type: "Sigmoid"
    repeat_id: "RecurrentInput"
    input: "Affine_5"
    output: "Sigmoid_2"
  }
  function {
    name: "Tanh_2"
    type: "Tanh"
    repeat_id: "RecurrentInput"
    input: "Add2"
    output: "Tanh_2"
  }
  function {
    name: "Affine_6"
    type: "Affine"
    repeat_id: "RecurrentInput"
    input: "Concatenate"
    input: "Affine_6/affine/W"
    input: "Affine_6/affine/b"
    output: "Affine_6"
    affine_param {
      base_axis: 1
    }
  }
  function {
    name: "Sigmoid_4"
    type: "Sigmoid"
    repeat_id: "RecurrentInput"
    input: "Affine_6"
    output: "Sigmoid_4"
  }
  function {
    name: "Mul2_3"
    type: "Mul2"
    repeat_id: "RecurrentInput"
    input: "Sigmoid_4"
    input: "Tanh_2"
    output: "Mul2_3"
  }
  function {
    name: "Delay"
    type: "Delay"
    repeat_id: "RecurrentInput"
    input: "Add2"
    input: "Delay_Initial"
    output: "Delay"
    recurrent_param {
      repeat_id: "RecurrentInput"
    }
  }
  function {
    name: "Delay_2"
    type: "Delay"
    repeat_id: "RecurrentInput"
    input: "Mul2_3"
    input: "Delay_2_Initial"
    output: "Delay_2"
    recurrent_param {
      repeat_id: "RecurrentInput"
    }
  }
  function {
    name: "RecurrentOutput"
    type: "RecurrentOutput"
    input: "Mul2_3"
    output: "RecurrentOutput"
    recurrent_param {
      axis: 2
      repeat_id: "RecurrentInput"
      length: 28
    }
  }
  function {
    name: "BinaryCrossEntropy"
    type: "BinaryCrossEntropy"
    input: "Sigmoid_3"
    input: "BinaryCrossEntropy_T"
    output: "BinaryCrossEntropy"
  }
  function {
    name: "Affine_2"
    type: "Affine"
    input: "Slice"
    input: "Affine_2/affine/W"
    input: "Affine_2/affine/b"
    output: "Affine_2"
    affine_param {
      base_axis: 1
    }
  }
  function {
    name: "Sigmoid_3"
    type: "Sigmoid"
    input: "Affine_2"
    output: "Sigmoid_3"
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
  name: "Runtime"
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
    name: "Affine_3/affine/W"
    type: "Parameter"
    shape: { dim: 56 dim: 1 dim: 28 }
    initializer {
      type: "NormalAffineGlorot"
      multiplier: 1.0
    }
  }
  variable {
    name: "Affine_3/affine/b"
    type: "Parameter"
    shape: { dim: 1 dim: 28 }
    initializer {
      type: "Constant"
      multiplier: 0.0
    }
  }
  variable {
    name: "Affine_4/affine/W"
    type: "Parameter"
    shape: { dim: 56 dim: 1 dim: 28 }
    initializer {
      type: "NormalAffineGlorot"
      multiplier: 1.0
    }
  }
  variable {
    name: "Affine_4/affine/b"
    type: "Parameter"
    shape: { dim: 1 dim: 28 }
    initializer {
      type: "Constant"
      multiplier: 0.0
    }
  }
  variable {
    name: "Affine_5/affine/W"
    type: "Parameter"
    shape: { dim: 56 dim: 1 dim: 28 }
    initializer {
      type: "NormalAffineGlorot"
      multiplier: 1.0
    }
  }
  variable {
    name: "Affine_5/affine/b"
    type: "Parameter"
    shape: { dim: 1 dim: 28 }
    initializer {
      type: "Constant"
      multiplier: 0.0
    }
  }
  variable {
    name: "Affine/affine/W"
    type: "Parameter"
    shape: { dim: 28 dim: 1 dim: 28 }
    initializer {
      type: "NormalAffineGlorot"
      multiplier: 1.0
    }
  }
  variable {
    name: "Affine/affine/b"
    type: "Parameter"
    shape: { dim: 1 dim: 28 }
    initializer {
      type: "Constant"
      multiplier: 0.0
    }
  }
  variable {
    name: "Affine_6/affine/W"
    type: "Parameter"
    shape: { dim: 56 dim: 1 dim: 28 }
    initializer {
      type: "NormalAffineGlorot"
      multiplier: 1.0
    }
  }
  variable {
    name: "Affine_6/affine/b"
    type: "Parameter"
    shape: { dim: 1 dim: 28 }
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
    name: "Delay_2_Initial"
    type: "Buffer"
    shape: { dim:-1 dim: 1 dim: 28 }
  }
  variable {
    name: "Affine_2/affine/W"
    type: "Parameter"
    shape: { dim: 28 dim: 1 }
    initializer {
      type: "NormalAffineGlorot"
      multiplier: 1.0
    }
  }
  variable {
    name: "Affine_2/affine/b"
    type: "Parameter"
    shape: { dim: 1 }
    initializer {
      type: "Constant"
      multiplier: 0.0
    }
  }
  variable {
    name: "Affine_3"
    type: "Buffer"
    repeat_id: "RecurrentInput"
    shape: { dim:-1 dim: 1 dim: 28 }
  }
  variable {
    name: "Tanh"
    type: "Buffer"
    repeat_id: "RecurrentInput"
    shape: { dim:-1 dim: 1 dim: 28 }
  }
  variable {
    name: "Affine_4"
    type: "Buffer"
    repeat_id: "RecurrentInput"
    shape: { dim:-1 dim: 1 dim: 28 }
  }
  variable {
    name: "Sigmoid"
    type: "Buffer"
    repeat_id: "RecurrentInput"
    shape: { dim:-1 dim: 1 dim: 28 }
  }
  variable {
    name: "Mul2"
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
    name: "Add2"
    type: "Buffer"
    repeat_id: "RecurrentInput"
    shape: { dim:-1 dim: 1 dim: 28 }
  }
  variable {
    name: "Mul2_2"
    type: "Buffer"
    repeat_id: "RecurrentInput"
    shape: { dim:-1 dim: 1 dim: 28 }
  }
  variable {
    name: "Affine_5"
    type: "Buffer"
    repeat_id: "RecurrentInput"
    shape: { dim:-1 dim: 1 dim: 28 }
  }
  variable {
    name: "Sigmoid_2"
    type: "Buffer"
    repeat_id: "RecurrentInput"
    shape: { dim:-1 dim: 1 dim: 28 }
  }
  variable {
    name: "Affine"
    type: "Buffer"
    repeat_id: "RecurrentInput"
    shape: { dim:-1 dim: 1 dim: 28 }
  }
  variable {
    name: "Tanh_2"
    type: "Buffer"
    repeat_id: "RecurrentInput"
    shape: { dim:-1 dim: 1 dim: 28 }
  }
  variable {
    name: "Affine_6"
    type: "Buffer"
    repeat_id: "RecurrentInput"
    shape: { dim:-1 dim: 1 dim: 28 }
  }
  variable {
    name: "Sigmoid_4"
    type: "Buffer"
    repeat_id: "RecurrentInput"
    shape: { dim:-1 dim: 1 dim: 28 }
  }
  variable {
    name: "Mul2_3"
    type: "Buffer"
    repeat_id: "RecurrentInput"
    shape: { dim:-1 dim: 1 dim: 28 }
  }
  variable {
    name: "Delay"
    type: "Buffer"
    repeat_id: "RecurrentInput"
    shape: { dim:-1 dim: 1 dim: 28 }
  }
  variable {
    name: "Delay_2"
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
    name: "Affine_2"
    type: "Buffer"
    shape: { dim:-1 dim: 1 }
  }
  variable {
    name: "y'"
    type: "Buffer"
    shape: { dim:-1 dim: 1 }
  }
  variable {
    name: "Slice"
    type: "Buffer"
    shape: { dim:-1 dim: 1 dim: 1 dim: 28 }
  }
  variable {
    name: "rnn_out"
    type: "Buffer"
    shape: { dim:-1 dim: 1 dim: 28 dim: 28 }
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
    name: "Affine_3"
    type: "Affine"
    repeat_id: "RecurrentInput"
    input: "Concatenate"
    input: "Affine_3/affine/W"
    input: "Affine_3/affine/b"
    output: "Affine_3"
    affine_param {
      base_axis: 1
    }
  }
  function {
    name: "Tanh"
    type: "Tanh"
    repeat_id: "RecurrentInput"
    input: "Affine_3"
    output: "Tanh"
  }
  function {
    name: "Affine_4"
    type: "Affine"
    repeat_id: "RecurrentInput"
    input: "Concatenate"
    input: "Affine_4/affine/W"
    input: "Affine_4/affine/b"
    output: "Affine_4"
    affine_param {
      base_axis: 1
    }
  }
  function {
    name: "Sigmoid"
    type: "Sigmoid"
    repeat_id: "RecurrentInput"
    input: "Affine_4"
    output: "Sigmoid"
  }
  function {
    name: "Mul2"
    type: "Mul2"
    repeat_id: "RecurrentInput"
    input: "Sigmoid"
    input: "Tanh"
    output: "Mul2"
  }
  function {
    name: "Concatenate"
    type: "Concatenate"
    repeat_id: "RecurrentInput"
    input: "Delay_2"
    input: "RecurrentInput"
    output: "Concatenate"
    concatenate_param {
      axis: 1
    }
  }
  function {
    name: "Add2"
    type: "Add2"
    repeat_id: "RecurrentInput"
    input: "Mul2"
    input: "Mul2_2"
    output: "Add2"
    add2_param {
      inplace: True
    }
  }
  function {
    name: "Mul2_2"
    type: "Mul2"
    repeat_id: "RecurrentInput"
    input: "Sigmoid_2"
    input: "Delay"
    output: "Mul2_2"
  }
  function {
    name: "Affine_5"
    type: "Affine"
    repeat_id: "RecurrentInput"
    input: "Concatenate"
    input: "Affine_5/affine/W"
    input: "Affine_5/affine/b"
    output: "Affine_5"
    affine_param {
      base_axis: 1
    }
  }
  function {
    name: "Sigmoid_2"
    type: "Sigmoid"
    repeat_id: "RecurrentInput"
    input: "Affine_5"
    output: "Sigmoid_2"
  }
  function {
    name: "Affine"
    type: "Affine"
    repeat_id: "RecurrentInput"
    input: "Add2"
    input: "Affine/affine/W"
    input: "Affine/affine/b"
    output: "Affine"
    affine_param {
      base_axis: 1
    }
  }
  function {
    name: "Tanh_2"
    type: "Tanh"
    repeat_id: "RecurrentInput"
    input: "Affine"
    output: "Tanh_2"
  }
  function {
    name: "Affine_6"
    type: "Affine"
    repeat_id: "RecurrentInput"
    input: "Concatenate"
    input: "Affine_6/affine/W"
    input: "Affine_6/affine/b"
    output: "Affine_6"
    affine_param {
      base_axis: 1
    }
  }
  function {
    name: "Sigmoid_4"
    type: "Sigmoid"
    repeat_id: "RecurrentInput"
    input: "Affine_6"
    output: "Sigmoid_4"
  }
  function {
    name: "Mul2_3"
    type: "Mul2"
    repeat_id: "RecurrentInput"
    input: "Sigmoid_4"
    input: "Tanh_2"
    output: "Mul2_3"
  }
  function {
    name: "Delay"
    type: "Delay"
    repeat_id: "RecurrentInput"
    input: "Affine"
    input: "Delay_Initial"
    output: "Delay"
    recurrent_param {
      repeat_id: "RecurrentInput"
    }
  }
  function {
    name: "Delay_2"
    type: "Delay"
    repeat_id: "RecurrentInput"
    input: "Mul2_3"
    input: "Delay_2_Initial"
    output: "Delay_2"
    recurrent_param {
      repeat_id: "RecurrentInput"
    }
  }
  function {
    name: "RecurrentOutput"
    type: "RecurrentOutput"
    input: "Mul2_3"
    output: "RecurrentOutput"
    recurrent_param {
      axis: 2
      repeat_id: "RecurrentInput"
      length: 28
    }
  }
  function {
    name: "Affine_2"
    type: "Affine"
    input: "Slice"
    input: "Affine_2/affine/W"
    input: "Affine_2/affine/b"
    output: "Affine_2"
    affine_param {
      base_axis: 1
    }
  }
  function {
    name: "y'"
    type: "Sigmoid"
    input: "Affine_2"
    output: "y'"
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
  function {
    name: "rnn_out"
    type: "Identity"
    input: "RecurrentOutput"
    output: "rnn_out"
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
  network_name: "Main_"
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
  generator_variable {
    variable_name: "Delay_2_Initial"
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
    variable_name: "Affine_3/affine/W"
    learning_rate_multiplier: 1.0
  }
  parameter_variable {
    variable_name: "Affine_3/affine/b"
    learning_rate_multiplier: 1.0
  }
  parameter_variable {
    variable_name: "Affine_4/affine/W"
    learning_rate_multiplier: 1.0
  }
  parameter_variable {
    variable_name: "Affine_4/affine/b"
    learning_rate_multiplier: 1.0
  }
  parameter_variable {
    variable_name: "Affine_5/affine/W"
    learning_rate_multiplier: 1.0
  }
  parameter_variable {
    variable_name: "Affine_5/affine/b"
    learning_rate_multiplier: 1.0
  }
  parameter_variable {
    variable_name: "Affine_6/affine/W"
    learning_rate_multiplier: 1.0
  }
  parameter_variable {
    variable_name: "Affine_6/affine/b"
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
  network_name: "Main_"
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
  generator_variable {
    variable_name: "Delay_2_Initial"
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
  network_name: "Main_"
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
  generator_variable {
    variable_name: "Delay_2_Initial"
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
  network_name: "Runtime"
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
  generator_variable {
    variable_name: "Delay_2_Initial"
    type: "Constant"
    multiplier: 0.0
  }
  output_variable {
    variable_name: "y'"
    data_name: "y'"
  }
  output_variable {
    variable_name: "rnn_out"
    data_name: "rnn_out"
  }
  parameter_variable {
    variable_name: "Affine_3/affine/W"
  }
  parameter_variable {
    variable_name: "Affine_3/affine/b"
  }
  parameter_variable {
    variable_name: "Affine_4/affine/W"
  }
  parameter_variable {
    variable_name: "Affine_4/affine/b"
  }
  parameter_variable {
    variable_name: "Affine_5/affine/W"
  }
  parameter_variable {
    variable_name: "Affine_5/affine/b"
  }
  parameter_variable {
    variable_name: "Affine/affine/W"
  }
  parameter_variable {
    variable_name: "Affine/affine/b"
  }
  parameter_variable {
    variable_name: "Affine_6/affine/W"
  }
  parameter_variable {
    variable_name: "Affine_6/affine/b"
  }
  parameter_variable {
    variable_name: "Affine_2/affine/W"
  }
  parameter_variable {
    variable_name: "Affine_2/affine/b"
  }
}
'''
# The nntxt is generated from /home/woody/aid/nnabla-sample-data/sample_project/tutorial/recurrent_neural_networks/long_short_term_memory(LSTM).files/20180720_192456/net.nntxt.
N00000013 = r'''global_config {
  default_context {
    array_class: "CudaCachedArray"
    device_id: "0"
    backends: "cudnn:float"
    backends: "cuda:float"
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
  name: "Main_"
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
    name: "C_Initial"
    type: "Buffer"
    shape: { dim:-1 dim: 1 dim: 28 }
  }
  variable {
    name: "H_Initial"
    type: "Buffer"
    shape: { dim:-1 dim: 1 dim: 28 }
  }
  variable {
    name: "Affine/affine/W"
    type: "Parameter"
    shape: { dim: 56 dim: 1 dim: 28 }
    initializer {
      type: "NormalAffineGlorot"
      multiplier: 1
    }
  }
  variable {
    name: "Affine/affine/b"
    type: "Parameter"
    shape: { dim: 1 dim: 28 }
    initializer {
      type: "Constant"
      multiplier: 0
    }
  }
  variable {
    name: "InputGate/affine/W"
    type: "Parameter"
    shape: { dim: 56 dim: 1 dim: 28 }
    initializer {
      type: "NormalAffineGlorot"
      multiplier: 1
    }
  }
  variable {
    name: "InputGate/affine/b"
    type: "Parameter"
    shape: { dim: 1 dim: 28 }
    initializer {
      type: "Constant"
      multiplier: 0
    }
  }
  variable {
    name: "ForgetGate/affine/W"
    type: "Parameter"
    shape: { dim: 56 dim: 1 dim: 28 }
    initializer {
      type: "NormalAffineGlorot"
      multiplier: 1
    }
  }
  variable {
    name: "ForgetGate/affine/b"
    type: "Parameter"
    shape: { dim: 1 dim: 28 }
    initializer {
      type: "Constant"
      multiplier: 0
    }
  }
  variable {
    name: "OutputGate/affine/W"
    type: "Parameter"
    shape: { dim: 56 dim: 1 dim: 28 }
    initializer {
      type: "NormalAffineGlorot"
      multiplier: 1
    }
  }
  variable {
    name: "OutputGate/affine/b"
    type: "Parameter"
    shape: { dim: 1 dim: 28 }
    initializer {
      type: "Constant"
      multiplier: 0
    }
  }
  variable {
    name: "Affine_2/affine/W"
    type: "Parameter"
    shape: { dim: 28 dim: 1 }
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
    name: "RecurrentInput"
    type: "Buffer"
    repeat_id: "RecurrentInput"
    shape: { dim:-1 dim: 1 dim: 28 }
  }
  variable {
    name: "C"
    type: "Buffer"
    repeat_id: "RecurrentInput"
    shape: { dim:-1 dim: 1 dim: 28 }
  }
  variable {
    name: "H"
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
    name: "Affine"
    type: "Buffer"
    repeat_id: "RecurrentInput"
    shape: { dim:-1 dim: 1 dim: 28 }
  }
  variable {
    name: "InputGate"
    type: "Buffer"
    repeat_id: "RecurrentInput"
    shape: { dim:-1 dim: 1 dim: 28 }
  }
  variable {
    name: "ForgetGate"
    type: "Buffer"
    repeat_id: "RecurrentInput"
    shape: { dim:-1 dim: 1 dim: 28 }
  }
  variable {
    name: "OutputGate"
    type: "Buffer"
    repeat_id: "RecurrentInput"
    shape: { dim:-1 dim: 1 dim: 28 }
  }
  variable {
    name: "Tanh"
    type: "Buffer"
    repeat_id: "RecurrentInput"
    shape: { dim:-1 dim: 1 dim: 28 }
  }
  variable {
    name: "Sigmoid_4"
    type: "Buffer"
    repeat_id: "RecurrentInput"
    shape: { dim:-1 dim: 1 dim: 28 }
  }
  variable {
    name: "Sigmoid_2"
    type: "Buffer"
    repeat_id: "RecurrentInput"
    shape: { dim:-1 dim: 1 dim: 28 }
  }
  variable {
    name: "Sigmoid_3"
    type: "Buffer"
    repeat_id: "RecurrentInput"
    shape: { dim:-1 dim: 1 dim: 28 }
  }
  variable {
    name: "Mul2"
    type: "Buffer"
    repeat_id: "RecurrentInput"
    shape: { dim:-1 dim: 1 dim: 28 }
  }
  variable {
    name: "Mul2_2"
    type: "Buffer"
    repeat_id: "RecurrentInput"
    shape: { dim:-1 dim: 1 dim: 28 }
  }
  variable {
    name: "Add2"
    type: "Buffer"
    repeat_id: "RecurrentInput"
    shape: { dim:-1 dim: 1 dim: 28 }
  }
  variable {
    name: "Tanh_2"
    type: "Buffer"
    repeat_id: "RecurrentInput"
    shape: { dim:-1 dim: 1 dim: 28 }
  }
  variable {
    name: "Mul2_3"
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
    name: "C"
    type: "Delay"
    repeat_id: "RecurrentInput"
    input: "Add2"
    input: "C_Initial"
    output: "C"
    recurrent_param {
      repeat_id: "RecurrentInput"
    }
  }
  function {
    name: "H"
    type: "Delay"
    repeat_id: "RecurrentInput"
    input: "Mul2_3"
    input: "H_Initial"
    output: "H"
    recurrent_param {
      repeat_id: "RecurrentInput"
    }
  }
  function {
    name: "Concatenate"
    type: "Concatenate"
    repeat_id: "RecurrentInput"
    input: "H"
    input: "RecurrentInput"
    output: "Concatenate"
    concatenate_param {
      axis: 1
    }
  }
  function {
    name: "Affine"
    type: "Affine"
    repeat_id: "RecurrentInput"
    input: "Concatenate"
    input: "Affine/affine/W"
    input: "Affine/affine/b"
    output: "Affine"
    affine_param {
      base_axis: 1
    }
  }
  function {
    name: "InputGate"
    type: "Affine"
    repeat_id: "RecurrentInput"
    input: "Concatenate"
    input: "InputGate/affine/W"
    input: "InputGate/affine/b"
    output: "InputGate"
    affine_param {
      base_axis: 1
    }
  }
  function {
    name: "ForgetGate"
    type: "Affine"
    repeat_id: "RecurrentInput"
    input: "Concatenate"
    input: "ForgetGate/affine/W"
    input: "ForgetGate/affine/b"
    output: "ForgetGate"
    affine_param {
      base_axis: 1
    }
  }
  function {
    name: "OutputGate"
    type: "Affine"
    repeat_id: "RecurrentInput"
    input: "Concatenate"
    input: "OutputGate/affine/W"
    input: "OutputGate/affine/b"
    output: "OutputGate"
    affine_param {
      base_axis: 1
    }
  }
  function {
    name: "Tanh"
    type: "Tanh"
    repeat_id: "RecurrentInput"
    input: "Affine"
    output: "Tanh"
  }
  function {
    name: "Sigmoid_4"
    type: "Sigmoid"
    repeat_id: "RecurrentInput"
    input: "InputGate"
    output: "Sigmoid_4"
  }
  function {
    name: "Sigmoid_2"
    type: "Sigmoid"
    repeat_id: "RecurrentInput"
    input: "ForgetGate"
    output: "Sigmoid_2"
  }
  function {
    name: "Sigmoid_3"
    type: "Sigmoid"
    repeat_id: "RecurrentInput"
    input: "OutputGate"
    output: "Sigmoid_3"
  }
  function {
    name: "Mul2"
    type: "Mul2"
    repeat_id: "RecurrentInput"
    input: "Sigmoid_4"
    input: "Tanh"
    output: "Mul2"
  }
  function {
    name: "Mul2_2"
    type: "Mul2"
    repeat_id: "RecurrentInput"
    input: "Sigmoid_2"
    input: "C"
    output: "Mul2_2"
  }
  function {
    name: "Add2"
    type: "Add2"
    repeat_id: "RecurrentInput"
    input: "Mul2"
    input: "Mul2_2"
    output: "Add2"
    add2_param {
      inplace: True
    }
  }
  function {
    name: "Tanh_2"
    type: "Tanh"
    repeat_id: "RecurrentInput"
    input: "Add2"
    output: "Tanh_2"
  }
  function {
    name: "Mul2_3"
    type: "Mul2"
    repeat_id: "RecurrentInput"
    input: "Sigmoid_3"
    input: "Tanh_2"
    output: "Mul2_3"
  }
  function {
    name: "RecurrentOutput"
    type: "RecurrentOutput"
    input: "Mul2_3"
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
  function {
    name: "Affine_2"
    type: "Affine"
    input: "Slice"
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
  name: "Runtime"
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
    name: "C_Initial"
    type: "Buffer"
    shape: { dim:-1 dim: 1 dim: 28 }
  }
  variable {
    name: "H_Initial"
    type: "Buffer"
    shape: { dim:-1 dim: 1 dim: 28 }
  }
  variable {
    name: "Affine/affine/W"
    type: "Parameter"
    shape: { dim: 56 dim: 1 dim: 28 }
    initializer {
      type: "NormalAffineGlorot"
      multiplier: 1
    }
  }
  variable {
    name: "Affine/affine/b"
    type: "Parameter"
    shape: { dim: 1 dim: 28 }
    initializer {
      type: "Constant"
      multiplier: 0
    }
  }
  variable {
    name: "InputGate/affine/W"
    type: "Parameter"
    shape: { dim: 56 dim: 1 dim: 28 }
    initializer {
      type: "NormalAffineGlorot"
      multiplier: 1
    }
  }
  variable {
    name: "InputGate/affine/b"
    type: "Parameter"
    shape: { dim: 1 dim: 28 }
    initializer {
      type: "Constant"
      multiplier: 0
    }
  }
  variable {
    name: "ForgetGate/affine/W"
    type: "Parameter"
    shape: { dim: 56 dim: 1 dim: 28 }
    initializer {
      type: "NormalAffineGlorot"
      multiplier: 1
    }
  }
  variable {
    name: "ForgetGate/affine/b"
    type: "Parameter"
    shape: { dim: 1 dim: 28 }
    initializer {
      type: "Constant"
      multiplier: 0
    }
  }
  variable {
    name: "OutputGate/affine/W"
    type: "Parameter"
    shape: { dim: 56 dim: 1 dim: 28 }
    initializer {
      type: "NormalAffineGlorot"
      multiplier: 1
    }
  }
  variable {
    name: "OutputGate/affine/b"
    type: "Parameter"
    shape: { dim: 1 dim: 28 }
    initializer {
      type: "Constant"
      multiplier: 0
    }
  }
  variable {
    name: "Affine_2/affine/W"
    type: "Parameter"
    shape: { dim: 28 dim: 1 }
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
    name: "RecurrentInput"
    type: "Buffer"
    repeat_id: "RecurrentInput"
    shape: { dim:-1 dim: 1 dim: 28 }
  }
  variable {
    name: "C"
    type: "Buffer"
    repeat_id: "RecurrentInput"
    shape: { dim:-1 dim: 1 dim: 28 }
  }
  variable {
    name: "H"
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
    name: "Affine"
    type: "Buffer"
    repeat_id: "RecurrentInput"
    shape: { dim:-1 dim: 1 dim: 28 }
  }
  variable {
    name: "InputGate"
    type: "Buffer"
    repeat_id: "RecurrentInput"
    shape: { dim:-1 dim: 1 dim: 28 }
  }
  variable {
    name: "ForgetGate"
    type: "Buffer"
    repeat_id: "RecurrentInput"
    shape: { dim:-1 dim: 1 dim: 28 }
  }
  variable {
    name: "OutputGate"
    type: "Buffer"
    repeat_id: "RecurrentInput"
    shape: { dim:-1 dim: 1 dim: 28 }
  }
  variable {
    name: "Tanh"
    type: "Buffer"
    repeat_id: "RecurrentInput"
    shape: { dim:-1 dim: 1 dim: 28 }
  }
  variable {
    name: "Sigmoid_4"
    type: "Buffer"
    repeat_id: "RecurrentInput"
    shape: { dim:-1 dim: 1 dim: 28 }
  }
  variable {
    name: "Sigmoid_2"
    type: "Buffer"
    repeat_id: "RecurrentInput"
    shape: { dim:-1 dim: 1 dim: 28 }
  }
  variable {
    name: "Sigmoid_3"
    type: "Buffer"
    repeat_id: "RecurrentInput"
    shape: { dim:-1 dim: 1 dim: 28 }
  }
  variable {
    name: "Mul2"
    type: "Buffer"
    repeat_id: "RecurrentInput"
    shape: { dim:-1 dim: 1 dim: 28 }
  }
  variable {
    name: "Mul2_2"
    type: "Buffer"
    repeat_id: "RecurrentInput"
    shape: { dim:-1 dim: 1 dim: 28 }
  }
  variable {
    name: "Add2"
    type: "Buffer"
    repeat_id: "RecurrentInput"
    shape: { dim:-1 dim: 1 dim: 28 }
  }
  variable {
    name: "Tanh_2"
    type: "Buffer"
    repeat_id: "RecurrentInput"
    shape: { dim:-1 dim: 1 dim: 28 }
  }
  variable {
    name: "Mul2_3"
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
  variable {
    name: "rnnout"
    type: "Buffer"
    shape: { dim:-1 dim: 1 dim: 28 dim: 28 }
  }
  variable {
    name: "Affine_2"
    type: "Buffer"
    shape: { dim:-1 dim: 1 }
  }
  variable {
    name: "y'"
    type: "Buffer"
    shape: { dim:-1 dim: 1 }
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
    name: "C"
    type: "Delay"
    repeat_id: "RecurrentInput"
    input: "Add2"
    input: "C_Initial"
    output: "C"
    recurrent_param {
      repeat_id: "RecurrentInput"
    }
  }
  function {
    name: "H"
    type: "Delay"
    repeat_id: "RecurrentInput"
    input: "Mul2_3"
    input: "H_Initial"
    output: "H"
    recurrent_param {
      repeat_id: "RecurrentInput"
    }
  }
  function {
    name: "Concatenate"
    type: "Concatenate"
    repeat_id: "RecurrentInput"
    input: "H"
    input: "RecurrentInput"
    output: "Concatenate"
    concatenate_param {
      axis: 1
    }
  }
  function {
    name: "Affine"
    type: "Affine"
    repeat_id: "RecurrentInput"
    input: "Concatenate"
    input: "Affine/affine/W"
    input: "Affine/affine/b"
    output: "Affine"
    affine_param {
      base_axis: 1
    }
  }
  function {
    name: "InputGate"
    type: "Affine"
    repeat_id: "RecurrentInput"
    input: "Concatenate"
    input: "InputGate/affine/W"
    input: "InputGate/affine/b"
    output: "InputGate"
    affine_param {
      base_axis: 1
    }
  }
  function {
    name: "ForgetGate"
    type: "Affine"
    repeat_id: "RecurrentInput"
    input: "Concatenate"
    input: "ForgetGate/affine/W"
    input: "ForgetGate/affine/b"
    output: "ForgetGate"
    affine_param {
      base_axis: 1
    }
  }
  function {
    name: "OutputGate"
    type: "Affine"
    repeat_id: "RecurrentInput"
    input: "Concatenate"
    input: "OutputGate/affine/W"
    input: "OutputGate/affine/b"
    output: "OutputGate"
    affine_param {
      base_axis: 1
    }
  }
  function {
    name: "Tanh"
    type: "Tanh"
    repeat_id: "RecurrentInput"
    input: "Affine"
    output: "Tanh"
  }
  function {
    name: "Sigmoid_4"
    type: "Sigmoid"
    repeat_id: "RecurrentInput"
    input: "InputGate"
    output: "Sigmoid_4"
  }
  function {
    name: "Sigmoid_2"
    type: "Sigmoid"
    repeat_id: "RecurrentInput"
    input: "ForgetGate"
    output: "Sigmoid_2"
  }
  function {
    name: "Sigmoid_3"
    type: "Sigmoid"
    repeat_id: "RecurrentInput"
    input: "OutputGate"
    output: "Sigmoid_3"
  }
  function {
    name: "Mul2"
    type: "Mul2"
    repeat_id: "RecurrentInput"
    input: "Sigmoid_4"
    input: "Tanh"
    output: "Mul2"
  }
  function {
    name: "Mul2_2"
    type: "Mul2"
    repeat_id: "RecurrentInput"
    input: "Sigmoid_2"
    input: "C"
    output: "Mul2_2"
  }
  function {
    name: "Add2"
    type: "Add2"
    repeat_id: "RecurrentInput"
    input: "Mul2"
    input: "Mul2_2"
    output: "Add2"
    add2_param {
      inplace: True
    }
  }
  function {
    name: "Tanh_2"
    type: "Tanh"
    repeat_id: "RecurrentInput"
    input: "Add2"
    output: "Tanh_2"
  }
  function {
    name: "Mul2_3"
    type: "Mul2"
    repeat_id: "RecurrentInput"
    input: "Sigmoid_3"
    input: "Tanh_2"
    output: "Mul2_3"
  }
  function {
    name: "RecurrentOutput"
    type: "RecurrentOutput"
    input: "Mul2_3"
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
  function {
    name: "rnnout"
    type: "Identity"
    input: "RecurrentOutput"
    output: "rnnout"
  }
  function {
    name: "Affine_2"
    type: "Affine"
    input: "Slice"
    input: "Affine_2/affine/W"
    input: "Affine_2/affine/b"
    output: "Affine_2"
    affine_param {
      base_axis: 1
    }
  }
  function {
    name: "y'"
    type: "Sigmoid"
    input: "Affine_2"
    output: "y'"
  }
}
dataset {
  name: "Training"
  uri: "C:\\Utils\\neural_network_console_internal_v120\\samples\\sample_dataset\\mnist\\small_mnist_4or9_training.csv"
  cache_dir: "C:\\Utils\\neural_network_console_internal_v120\\samples\\sample_dataset\\mnist\\small_mnist_4or9_training.cache"
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
  network_name: "Main_"
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
    variable_name: "C_Initial"
    type: "Constant"
    multiplier: 0
  }
  generator_variable {
    variable_name: "H_Initial"
    type: "Constant"
    multiplier: 0
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
  parameter_variable {
    variable_name: "InputGate/affine/W"
    learning_rate_multiplier: 1
  }
  parameter_variable {
    variable_name: "InputGate/affine/b"
    learning_rate_multiplier: 1
  }
  parameter_variable {
    variable_name: "ForgetGate/affine/W"
    learning_rate_multiplier: 1
  }
  parameter_variable {
    variable_name: "ForgetGate/affine/b"
    learning_rate_multiplier: 1
  }
  parameter_variable {
    variable_name: "OutputGate/affine/W"
    learning_rate_multiplier: 1
  }
  parameter_variable {
    variable_name: "OutputGate/affine/b"
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
  network_name: "Main_"
  dataset_name: "Training"
  data_variable {
    variable_name: "Input"
    data_name: "x"
  }
  generator_variable {
    variable_name: "C_Initial"
    type: "Constant"
    multiplier: 0
  }
  generator_variable {
    variable_name: "H_Initial"
    type: "Constant"
    multiplier: 0
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
  network_name: "Main_"
  dataset_name: "Validation"
  data_variable {
    variable_name: "Input"
    data_name: "x"
  }
  generator_variable {
    variable_name: "C_Initial"
    type: "Constant"
    multiplier: 0
  }
  generator_variable {
    variable_name: "H_Initial"
    type: "Constant"
    multiplier: 0
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
  network_name: "Runtime"
  num_evaluations: 1
  repeat_evaluation_type: "mean"
  need_back_propagation: false
  data_variable {
    variable_name: "Input"
    data_name: "x"
  }
  generator_variable {
    variable_name: "C_Initial"
    type: "Constant"
    multiplier: 0
  }
  generator_variable {
    variable_name: "H_Initial"
    type: "Constant"
    multiplier: 0
  }
  output_variable {
    variable_name: "rnnout"
    data_name: "rnnout"
  }
  output_variable {
    variable_name: "y'"
    data_name: "y'"
  }
  parameter_variable {
    variable_name: "Affine/affine/W"
  }
  parameter_variable {
    variable_name: "Affine/affine/b"
  }
  parameter_variable {
    variable_name: "InputGate/affine/W"
  }
  parameter_variable {
    variable_name: "InputGate/affine/b"
  }
  parameter_variable {
    variable_name: "ForgetGate/affine/W"
  }
  parameter_variable {
    variable_name: "ForgetGate/affine/b"
  }
  parameter_variable {
    variable_name: "OutputGate/affine/W"
  }
  parameter_variable {
    variable_name: "OutputGate/affine/b"
  }
  parameter_variable {
    variable_name: "Affine_2/affine/W"
  }
  parameter_variable {
    variable_name: "Affine_2/affine/b"
  }
}
'''
# The nntxt is generated from /home/woody/aid/nnabla-sample-data/sample_project/tutorial/binary_networks/binary_weight_mnist_MLP.files/20180723_154246/net.nntxt.
N00000014 = r'''global_config {
  default_context {
    array_class: "CudaCachedArray"
    device_id: "0"
    backends: "cudnn:float"
    backends: "cuda:float"
    backends: "cpu:float"
  }
}
training_config {
  max_epoch: 50
  iter_per_epoch: 937
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
    name: "BinaryWeightAffine/bwn_affine/W"
    type: "Parameter"
    shape: { dim: 784 dim: 2048 }
    initializer {
      type: "UniformAffineGlorot"
      multiplier: 1
    }
  }
  variable {
    name: "BinaryWeightAffine/bwn_affine/Wb"
    type: "Parameter"
    shape: { dim: 784 dim: 2048 }
    initializer {
      type: "Constant"
      multiplier: 0
    }
  }
  variable {
    name: "BinaryWeightAffine/bwn_affine/b"
    type: "Parameter"
    shape: { dim: 2048 }
    initializer {
      type: "Constant"
      multiplier: 0
    }
  }
  variable {
    name: "BinaryWeightAffine_2/bwn_affine/W"
    type: "Parameter"
    shape: { dim: 2048 dim: 2048 }
    initializer {
      type: "UniformAffineGlorot"
      multiplier: 1
    }
  }
  variable {
    name: "BinaryWeightAffine_2/bwn_affine/Wb"
    type: "Parameter"
    shape: { dim: 2048 dim: 2048 }
    initializer {
      type: "Constant"
      multiplier: 0
    }
  }
  variable {
    name: "BinaryWeightAffine_2/bwn_affine/b"
    type: "Parameter"
    shape: { dim: 2048 }
    initializer {
      type: "Constant"
      multiplier: 0
    }
  }
  variable {
    name: "BinaryWeightAffine_3/bwn_affine/W"
    type: "Parameter"
    shape: { dim: 2048 dim: 2048 }
    initializer {
      type: "UniformAffineGlorot"
      multiplier: 1
    }
  }
  variable {
    name: "BinaryWeightAffine_3/bwn_affine/Wb"
    type: "Parameter"
    shape: { dim: 2048 dim: 2048 }
    initializer {
      type: "Constant"
      multiplier: 0
    }
  }
  variable {
    name: "BinaryWeightAffine_3/bwn_affine/b"
    type: "Parameter"
    shape: { dim: 2048 }
    initializer {
      type: "Constant"
      multiplier: 0
    }
  }
  variable {
    name: "BinaryWeightAffine_4/bwn_affine/W"
    type: "Parameter"
    shape: { dim: 2048 dim: 10 }
    initializer {
      type: "UniformAffineGlorot"
      multiplier: 1
    }
  }
  variable {
    name: "BinaryWeightAffine_4/bwn_affine/Wb"
    type: "Parameter"
    shape: { dim: 2048 dim: 10 }
    initializer {
      type: "Constant"
      multiplier: 0
    }
  }
  variable {
    name: "BinaryWeightAffine_4/bwn_affine/b"
    type: "Parameter"
    shape: { dim: 10 }
    initializer {
      type: "Constant"
      multiplier: 0
    }
  }
  variable {
    name: "CategoricalCrossEntropy_T"
    type: "Buffer"
    shape: { dim:-1 dim: 1 }
  }
  variable {
    name: "BinaryWeightAffine"
    type: "Buffer"
    shape: { dim:-1 dim: 2048 }
  }
  variable {
    name: "ReLU"
    type: "Buffer"
    shape: { dim:-1 dim: 2048 }
  }
  variable {
    name: "BinaryWeightAffine_2"
    type: "Buffer"
    shape: { dim:-1 dim: 2048 }
  }
  variable {
    name: "ReLU_2"
    type: "Buffer"
    shape: { dim:-1 dim: 2048 }
  }
  variable {
    name: "BinaryWeightAffine_3"
    type: "Buffer"
    shape: { dim:-1 dim: 2048 }
  }
  variable {
    name: "ReLU_3"
    type: "Buffer"
    shape: { dim:-1 dim: 2048 }
  }
  variable {
    name: "BinaryWeightAffine_4"
    type: "Buffer"
    shape: { dim:-1 dim: 10 }
  }
  variable {
    name: "Softmax"
    type: "Buffer"
    shape: { dim:-1 dim: 10 }
  }
  variable {
    name: "CategoricalCrossEntropy"
    type: "Buffer"
    shape: { dim:-1 dim: 1 }
  }
  function {
    name: "BinaryWeightAffine"
    type: "BinaryWeightAffine"
    input: "Input"
    input: "BinaryWeightAffine/bwn_affine/W"
    input: "BinaryWeightAffine/bwn_affine/Wb"
    input: "BinaryWeightAffine/bwn_affine/b"
    output: "BinaryWeightAffine"
    binary_weight_affine_param {
      base_axis: 1
    }
  }
  function {
    name: "ReLU"
    type: "ReLU"
    input: "BinaryWeightAffine"
    output: "ReLU"
    relu_param {
      inplace: True
    }
  }
  function {
    name: "BinaryWeightAffine_2"
    type: "BinaryWeightAffine"
    input: "ReLU"
    input: "BinaryWeightAffine_2/bwn_affine/W"
    input: "BinaryWeightAffine_2/bwn_affine/Wb"
    input: "BinaryWeightAffine_2/bwn_affine/b"
    output: "BinaryWeightAffine_2"
    binary_weight_affine_param {
      base_axis: 1
    }
  }
  function {
    name: "ReLU_2"
    type: "ReLU"
    input: "BinaryWeightAffine_2"
    output: "ReLU_2"
    relu_param {
      inplace: True
    }
  }
  function {
    name: "BinaryWeightAffine_3"
    type: "BinaryWeightAffine"
    input: "ReLU_2"
    input: "BinaryWeightAffine_3/bwn_affine/W"
    input: "BinaryWeightAffine_3/bwn_affine/Wb"
    input: "BinaryWeightAffine_3/bwn_affine/b"
    output: "BinaryWeightAffine_3"
    binary_weight_affine_param {
      base_axis: 1
    }
  }
  function {
    name: "ReLU_3"
    type: "ReLU"
    input: "BinaryWeightAffine_3"
    output: "ReLU_3"
    relu_param {
      inplace: True
    }
  }
  function {
    name: "BinaryWeightAffine_4"
    type: "BinaryWeightAffine"
    input: "ReLU_3"
    input: "BinaryWeightAffine_4/bwn_affine/W"
    input: "BinaryWeightAffine_4/bwn_affine/Wb"
    input: "BinaryWeightAffine_4/bwn_affine/b"
    output: "BinaryWeightAffine_4"
    binary_weight_affine_param {
      base_axis: 1
    }
  }
  function {
    name: "Softmax"
    type: "Softmax"
    input: "BinaryWeightAffine_4"
    output: "Softmax"
    softmax_param {
      axis: 1
    }
  }
  function {
    name: "CategoricalCrossEntropy"
    type: "CategoricalCrossEntropy"
    input: "Softmax"
    input: "CategoricalCrossEntropy_T"
    output: "CategoricalCrossEntropy"
    categorical_cross_entropy_param {
      axis: 1
    }
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
    name: "BinaryWeightAffine/bwn_affine/W"
    type: "Parameter"
    shape: { dim: 784 dim: 2048 }
    initializer {
      type: "UniformAffineGlorot"
      multiplier: 1
    }
  }
  variable {
    name: "BinaryWeightAffine/bwn_affine/Wb"
    type: "Parameter"
    shape: { dim: 784 dim: 2048 }
    initializer {
      type: "Constant"
      multiplier: 0
    }
  }
  variable {
    name: "BinaryWeightAffine/bwn_affine/b"
    type: "Parameter"
    shape: { dim: 2048 }
    initializer {
      type: "Constant"
      multiplier: 0
    }
  }
  variable {
    name: "BinaryWeightAffine_2/bwn_affine/W"
    type: "Parameter"
    shape: { dim: 2048 dim: 2048 }
    initializer {
      type: "UniformAffineGlorot"
      multiplier: 1
    }
  }
  variable {
    name: "BinaryWeightAffine_2/bwn_affine/Wb"
    type: "Parameter"
    shape: { dim: 2048 dim: 2048 }
    initializer {
      type: "Constant"
      multiplier: 0
    }
  }
  variable {
    name: "BinaryWeightAffine_2/bwn_affine/b"
    type: "Parameter"
    shape: { dim: 2048 }
    initializer {
      type: "Constant"
      multiplier: 0
    }
  }
  variable {
    name: "BinaryWeightAffine_3/bwn_affine/W"
    type: "Parameter"
    shape: { dim: 2048 dim: 2048 }
    initializer {
      type: "UniformAffineGlorot"
      multiplier: 1
    }
  }
  variable {
    name: "BinaryWeightAffine_3/bwn_affine/Wb"
    type: "Parameter"
    shape: { dim: 2048 dim: 2048 }
    initializer {
      type: "Constant"
      multiplier: 0
    }
  }
  variable {
    name: "BinaryWeightAffine_3/bwn_affine/b"
    type: "Parameter"
    shape: { dim: 2048 }
    initializer {
      type: "Constant"
      multiplier: 0
    }
  }
  variable {
    name: "BinaryWeightAffine_4/bwn_affine/W"
    type: "Parameter"
    shape: { dim: 2048 dim: 10 }
    initializer {
      type: "UniformAffineGlorot"
      multiplier: 1
    }
  }
  variable {
    name: "BinaryWeightAffine_4/bwn_affine/Wb"
    type: "Parameter"
    shape: { dim: 2048 dim: 10 }
    initializer {
      type: "Constant"
      multiplier: 0
    }
  }
  variable {
    name: "BinaryWeightAffine_4/bwn_affine/b"
    type: "Parameter"
    shape: { dim: 10 }
    initializer {
      type: "Constant"
      multiplier: 0
    }
  }
  variable {
    name: "CategoricalCrossEntropy_T"
    type: "Buffer"
    shape: { dim:-1 dim: 1 }
  }
  variable {
    name: "BinaryWeightAffine"
    type: "Buffer"
    shape: { dim:-1 dim: 2048 }
  }
  variable {
    name: "ReLU"
    type: "Buffer"
    shape: { dim:-1 dim: 2048 }
  }
  variable {
    name: "BinaryWeightAffine_2"
    type: "Buffer"
    shape: { dim:-1 dim: 2048 }
  }
  variable {
    name: "ReLU_2"
    type: "Buffer"
    shape: { dim:-1 dim: 2048 }
  }
  variable {
    name: "BinaryWeightAffine_3"
    type: "Buffer"
    shape: { dim:-1 dim: 2048 }
  }
  variable {
    name: "ReLU_3"
    type: "Buffer"
    shape: { dim:-1 dim: 2048 }
  }
  variable {
    name: "BinaryWeightAffine_4"
    type: "Buffer"
    shape: { dim:-1 dim: 10 }
  }
  variable {
    name: "Softmax"
    type: "Buffer"
    shape: { dim:-1 dim: 10 }
  }
  variable {
    name: "CategoricalCrossEntropy"
    type: "Buffer"
    shape: { dim:-1 dim: 1 }
  }
  function {
    name: "BinaryWeightAffine"
    type: "BinaryWeightAffine"
    input: "Input"
    input: "BinaryWeightAffine/bwn_affine/W"
    input: "BinaryWeightAffine/bwn_affine/Wb"
    input: "BinaryWeightAffine/bwn_affine/b"
    output: "BinaryWeightAffine"
    binary_weight_affine_param {
      base_axis: 1
    }
  }
  function {
    name: "ReLU"
    type: "ReLU"
    input: "BinaryWeightAffine"
    output: "ReLU"
    relu_param {
      inplace: True
    }
  }
  function {
    name: "BinaryWeightAffine_2"
    type: "BinaryWeightAffine"
    input: "ReLU"
    input: "BinaryWeightAffine_2/bwn_affine/W"
    input: "BinaryWeightAffine_2/bwn_affine/Wb"
    input: "BinaryWeightAffine_2/bwn_affine/b"
    output: "BinaryWeightAffine_2"
    binary_weight_affine_param {
      base_axis: 1
    }
  }
  function {
    name: "ReLU_2"
    type: "ReLU"
    input: "BinaryWeightAffine_2"
    output: "ReLU_2"
    relu_param {
      inplace: True
    }
  }
  function {
    name: "BinaryWeightAffine_3"
    type: "BinaryWeightAffine"
    input: "ReLU_2"
    input: "BinaryWeightAffine_3/bwn_affine/W"
    input: "BinaryWeightAffine_3/bwn_affine/Wb"
    input: "BinaryWeightAffine_3/bwn_affine/b"
    output: "BinaryWeightAffine_3"
    binary_weight_affine_param {
      base_axis: 1
    }
  }
  function {
    name: "ReLU_3"
    type: "ReLU"
    input: "BinaryWeightAffine_3"
    output: "ReLU_3"
    relu_param {
      inplace: True
    }
  }
  function {
    name: "BinaryWeightAffine_4"
    type: "BinaryWeightAffine"
    input: "ReLU_3"
    input: "BinaryWeightAffine_4/bwn_affine/W"
    input: "BinaryWeightAffine_4/bwn_affine/Wb"
    input: "BinaryWeightAffine_4/bwn_affine/b"
    output: "BinaryWeightAffine_4"
    binary_weight_affine_param {
      base_axis: 1
    }
  }
  function {
    name: "Softmax"
    type: "Softmax"
    input: "BinaryWeightAffine_4"
    output: "Softmax"
    softmax_param {
      axis: 1
    }
  }
  function {
    name: "CategoricalCrossEntropy"
    type: "CategoricalCrossEntropy"
    input: "Softmax"
    input: "CategoricalCrossEntropy_T"
    output: "CategoricalCrossEntropy"
    categorical_cross_entropy_param {
      axis: 1
    }
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
    name: "BinaryWeightAffine/bwn_affine/W"
    type: "Parameter"
    shape: { dim: 784 dim: 2048 }
    initializer {
      type: "UniformAffineGlorot"
      multiplier: 1
    }
  }
  variable {
    name: "BinaryWeightAffine/bwn_affine/Wb"
    type: "Parameter"
    shape: { dim: 784 dim: 2048 }
    initializer {
      type: "Constant"
      multiplier: 0
    }
  }
  variable {
    name: "BinaryWeightAffine/bwn_affine/b"
    type: "Parameter"
    shape: { dim: 2048 }
    initializer {
      type: "Constant"
      multiplier: 0
    }
  }
  variable {
    name: "BinaryWeightAffine_2/bwn_affine/W"
    type: "Parameter"
    shape: { dim: 2048 dim: 2048 }
    initializer {
      type: "UniformAffineGlorot"
      multiplier: 1
    }
  }
  variable {
    name: "BinaryWeightAffine_2/bwn_affine/Wb"
    type: "Parameter"
    shape: { dim: 2048 dim: 2048 }
    initializer {
      type: "Constant"
      multiplier: 0
    }
  }
  variable {
    name: "BinaryWeightAffine_2/bwn_affine/b"
    type: "Parameter"
    shape: { dim: 2048 }
    initializer {
      type: "Constant"
      multiplier: 0
    }
  }
  variable {
    name: "BinaryWeightAffine_3/bwn_affine/W"
    type: "Parameter"
    shape: { dim: 2048 dim: 2048 }
    initializer {
      type: "UniformAffineGlorot"
      multiplier: 1
    }
  }
  variable {
    name: "BinaryWeightAffine_3/bwn_affine/Wb"
    type: "Parameter"
    shape: { dim: 2048 dim: 2048 }
    initializer {
      type: "Constant"
      multiplier: 0
    }
  }
  variable {
    name: "BinaryWeightAffine_3/bwn_affine/b"
    type: "Parameter"
    shape: { dim: 2048 }
    initializer {
      type: "Constant"
      multiplier: 0
    }
  }
  variable {
    name: "BinaryWeightAffine_4/bwn_affine/W"
    type: "Parameter"
    shape: { dim: 2048 dim: 10 }
    initializer {
      type: "UniformAffineGlorot"
      multiplier: 1
    }
  }
  variable {
    name: "BinaryWeightAffine_4/bwn_affine/Wb"
    type: "Parameter"
    shape: { dim: 2048 dim: 10 }
    initializer {
      type: "Constant"
      multiplier: 0
    }
  }
  variable {
    name: "BinaryWeightAffine_4/bwn_affine/b"
    type: "Parameter"
    shape: { dim: 10 }
    initializer {
      type: "Constant"
      multiplier: 0
    }
  }
  variable {
    name: "BinaryWeightAffine"
    type: "Buffer"
    shape: { dim:-1 dim: 2048 }
  }
  variable {
    name: "ReLU"
    type: "Buffer"
    shape: { dim:-1 dim: 2048 }
  }
  variable {
    name: "BinaryWeightAffine_2"
    type: "Buffer"
    shape: { dim:-1 dim: 2048 }
  }
  variable {
    name: "ReLU_2"
    type: "Buffer"
    shape: { dim:-1 dim: 2048 }
  }
  variable {
    name: "BinaryWeightAffine_3"
    type: "Buffer"
    shape: { dim:-1 dim: 2048 }
  }
  variable {
    name: "ReLU_3"
    type: "Buffer"
    shape: { dim:-1 dim: 2048 }
  }
  variable {
    name: "BinaryWeightAffine_4"
    type: "Buffer"
    shape: { dim:-1 dim: 10 }
  }
  variable {
    name: "Softmax"
    type: "Buffer"
    shape: { dim:-1 dim: 10 }
  }
  function {
    name: "BinaryWeightAffine"
    type: "BinaryWeightAffine"
    input: "Input"
    input: "BinaryWeightAffine/bwn_affine/W"
    input: "BinaryWeightAffine/bwn_affine/Wb"
    input: "BinaryWeightAffine/bwn_affine/b"
    output: "BinaryWeightAffine"
    binary_weight_affine_param {
      base_axis: 1
    }
  }
  function {
    name: "ReLU"
    type: "ReLU"
    input: "BinaryWeightAffine"
    output: "ReLU"
    relu_param {
      inplace: True
    }
  }
  function {
    name: "BinaryWeightAffine_2"
    type: "BinaryWeightAffine"
    input: "ReLU"
    input: "BinaryWeightAffine_2/bwn_affine/W"
    input: "BinaryWeightAffine_2/bwn_affine/Wb"
    input: "BinaryWeightAffine_2/bwn_affine/b"
    output: "BinaryWeightAffine_2"
    binary_weight_affine_param {
      base_axis: 1
    }
  }
  function {
    name: "ReLU_2"
    type: "ReLU"
    input: "BinaryWeightAffine_2"
    output: "ReLU_2"
    relu_param {
      inplace: True
    }
  }
  function {
    name: "BinaryWeightAffine_3"
    type: "BinaryWeightAffine"
    input: "ReLU_2"
    input: "BinaryWeightAffine_3/bwn_affine/W"
    input: "BinaryWeightAffine_3/bwn_affine/Wb"
    input: "BinaryWeightAffine_3/bwn_affine/b"
    output: "BinaryWeightAffine_3"
    binary_weight_affine_param {
      base_axis: 1
    }
  }
  function {
    name: "ReLU_3"
    type: "ReLU"
    input: "BinaryWeightAffine_3"
    output: "ReLU_3"
    relu_param {
      inplace: True
    }
  }
  function {
    name: "BinaryWeightAffine_4"
    type: "BinaryWeightAffine"
    input: "ReLU_3"
    input: "BinaryWeightAffine_4/bwn_affine/W"
    input: "BinaryWeightAffine_4/bwn_affine/Wb"
    input: "BinaryWeightAffine_4/bwn_affine/b"
    output: "BinaryWeightAffine_4"
    binary_weight_affine_param {
      base_axis: 1
    }
  }
  function {
    name: "Softmax"
    type: "Softmax"
    input: "BinaryWeightAffine_4"
    output: "Softmax"
    softmax_param {
      axis: 1
    }
  }
}
dataset {
  name: "Training"
  uri: "C:\\Utils\\neural_network_console_internal_v120\\samples\\sample_dataset\\mnist\\mnist_training.csv"
  cache_dir: "C:\\Utils\\neural_network_console_internal_v120\\samples\\sample_dataset\\mnist\\mnist_training.cache"
  overwrite_cache: True
  create_cache_explicitly: True
  shuffle: true
  no_image_normalization: False
  batch_size: 64
}
dataset {
  name: "Validation"
  uri: "C:\\Utils\\neural_network_console_internal_v120\\samples\\sample_dataset\\mnist\\mnist_test.csv"
  cache_dir: "C:\\Utils\\neural_network_console_internal_v120\\samples\\sample_dataset\\mnist\\mnist_test.cache"
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
      alpha: 0.0001
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
    variable_name: "CategoricalCrossEntropy_T"
    data_name: "y"
  }
  loss_variable {
    variable_name: "CategoricalCrossEntropy"
  }
  parameter_variable {
    variable_name: "BinaryWeightAffine/bwn_affine/W"
    learning_rate_multiplier: 1
  }
  parameter_variable {
    variable_name: "BinaryWeightAffine/bwn_affine/Wb"
    learning_rate_multiplier: 0
  }
  parameter_variable {
    variable_name: "BinaryWeightAffine/bwn_affine/b"
    learning_rate_multiplier: 1
  }
  parameter_variable {
    variable_name: "BinaryWeightAffine_2/bwn_affine/W"
    learning_rate_multiplier: 1
  }
  parameter_variable {
    variable_name: "BinaryWeightAffine_2/bwn_affine/Wb"
    learning_rate_multiplier: 0
  }
  parameter_variable {
    variable_name: "BinaryWeightAffine_2/bwn_affine/b"
    learning_rate_multiplier: 1
  }
  parameter_variable {
    variable_name: "BinaryWeightAffine_3/bwn_affine/W"
    learning_rate_multiplier: 1
  }
  parameter_variable {
    variable_name: "BinaryWeightAffine_3/bwn_affine/Wb"
    learning_rate_multiplier: 0
  }
  parameter_variable {
    variable_name: "BinaryWeightAffine_3/bwn_affine/b"
    learning_rate_multiplier: 1
  }
  parameter_variable {
    variable_name: "BinaryWeightAffine_4/bwn_affine/W"
    learning_rate_multiplier: 1
  }
  parameter_variable {
    variable_name: "BinaryWeightAffine_4/bwn_affine/Wb"
    learning_rate_multiplier: 0
  }
  parameter_variable {
    variable_name: "BinaryWeightAffine_4/bwn_affine/b"
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
    variable_name: "CategoricalCrossEntropy_T"
    data_name: "y"
  }
  monitor_variable {
    type: "Error"
    variable_name: "CategoricalCrossEntropy"
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
    variable_name: "CategoricalCrossEntropy_T"
    data_name: "y"
  }
  monitor_variable {
    type: "Error"
    variable_name: "CategoricalCrossEntropy"
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
    variable_name: "Softmax"
    data_name: "y'"
  }
  parameter_variable {
    variable_name: "BinaryWeightAffine/bwn_affine/W"
  }
  parameter_variable {
    variable_name: "BinaryWeightAffine/bwn_affine/Wb"
  }
  parameter_variable {
    variable_name: "BinaryWeightAffine/bwn_affine/b"
  }
  parameter_variable {
    variable_name: "BinaryWeightAffine_2/bwn_affine/W"
  }
  parameter_variable {
    variable_name: "BinaryWeightAffine_2/bwn_affine/Wb"
  }
  parameter_variable {
    variable_name: "BinaryWeightAffine_2/bwn_affine/b"
  }
  parameter_variable {
    variable_name: "BinaryWeightAffine_3/bwn_affine/W"
  }
  parameter_variable {
    variable_name: "BinaryWeightAffine_3/bwn_affine/Wb"
  }
  parameter_variable {
    variable_name: "BinaryWeightAffine_3/bwn_affine/b"
  }
  parameter_variable {
    variable_name: "BinaryWeightAffine_4/bwn_affine/W"
  }
  parameter_variable {
    variable_name: "BinaryWeightAffine_4/bwn_affine/Wb"
  }
  parameter_variable {
    variable_name: "BinaryWeightAffine_4/bwn_affine/b"
  }
}
'''
# The nntxt is generated from /home/woody/aid/nnabla-sample-data/sample_project/tutorial/binary_networks/binary_net_mnist_LeNet.files/20180723_150834/net.nntxt.
N00000015 = r'''global_config {
  default_context {
    array_class: "CudaCachedArray"
    device_id: "0"
    backends: "cudnn:float"
    backends: "cuda:float"
    backends: "cpu:float"
  }
}
training_config {
  max_epoch: 50
  iter_per_epoch: 937
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
    name: "BinaryConnectConvolution/bicon_conv/W"
    type: "Parameter"
    shape: { dim: 64 dim: 1 dim: 5 dim: 5 }
    initializer {
      type: "UniformConvolutionGlorot"
      multiplier: 1
    }
  }
  variable {
    name: "BinaryConnectConvolution/bicon_conv/Wb"
    type: "Parameter"
    shape: { dim: 64 dim: 1 dim: 5 dim: 5 }
    initializer {
      type: "Constant"
      multiplier: 0
    }
  }
  variable {
    name: "BinaryConnectConvolution/bicon_conv/b"
    type: "Parameter"
    shape: { dim: 64 }
    initializer {
      type: "Constant"
      multiplier: 0
    }
  }
  variable {
    name: "BatchNormalization/bn/beta"
    type: "Parameter"
    shape: { dim: 1 dim: 64 dim: 1 dim: 1 }
    initializer {
      type: "Constant"
      multiplier: 0
    }
  }
  variable {
    name: "BatchNormalization/bn/gamma"
    type: "Parameter"
    shape: { dim: 1 dim: 64 dim: 1 dim: 1 }
    initializer {
      type: "Constant"
      multiplier: 1
    }
  }
  variable {
    name: "BatchNormalization/bn/mean"
    type: "Parameter"
    shape: { dim: 1 dim: 64 dim: 1 dim: 1 }
    initializer {
      type: "Constant"
      multiplier: 0
    }
  }
  variable {
    name: "BatchNormalization/bn/var"
    type: "Parameter"
    shape: { dim: 1 dim: 64 dim: 1 dim: 1 }
    initializer {
      type: "Constant"
      multiplier: 0
    }
  }
  variable {
    name: "BinaryConnectConvolution_2/bicon_conv/W"
    type: "Parameter"
    shape: { dim: 64 dim: 64 dim: 5 dim: 5 }
    initializer {
      type: "UniformConvolutionGlorot"
      multiplier: 1
    }
  }
  variable {
    name: "BinaryConnectConvolution_2/bicon_conv/Wb"
    type: "Parameter"
    shape: { dim: 64 dim: 64 dim: 5 dim: 5 }
    initializer {
      type: "Constant"
      multiplier: 0
    }
  }
  variable {
    name: "BinaryConnectConvolution_2/bicon_conv/b"
    type: "Parameter"
    shape: { dim: 64 }
    initializer {
      type: "Constant"
      multiplier: 0
    }
  }
  variable {
    name: "BatchNormalization_2/bn/beta"
    type: "Parameter"
    shape: { dim: 1 dim: 64 dim: 1 dim: 1 }
    initializer {
      type: "Constant"
      multiplier: 0
    }
  }
  variable {
    name: "BatchNormalization_2/bn/gamma"
    type: "Parameter"
    shape: { dim: 1 dim: 64 dim: 1 dim: 1 }
    initializer {
      type: "Constant"
      multiplier: 1
    }
  }
  variable {
    name: "BatchNormalization_2/bn/mean"
    type: "Parameter"
    shape: { dim: 1 dim: 64 dim: 1 dim: 1 }
    initializer {
      type: "Constant"
      multiplier: 0
    }
  }
  variable {
    name: "BatchNormalization_2/bn/var"
    type: "Parameter"
    shape: { dim: 1 dim: 64 dim: 1 dim: 1 }
    initializer {
      type: "Constant"
      multiplier: 0
    }
  }
  variable {
    name: "BinaryConnectAffine/bicon_affine/W"
    type: "Parameter"
    shape: { dim: 1024 dim: 512 }
    initializer {
      type: "UniformAffineGlorot"
      multiplier: 1
    }
  }
  variable {
    name: "BinaryConnectAffine/bicon_affine/Wb"
    type: "Parameter"
    shape: { dim: 1024 dim: 512 }
    initializer {
      type: "Constant"
      multiplier: 0
    }
  }
  variable {
    name: "BinaryConnectAffine/bicon_affine/b"
    type: "Parameter"
    shape: { dim: 512 }
    initializer {
      type: "Constant"
      multiplier: 0
    }
  }
  variable {
    name: "BatchNormalization_3/bn/beta"
    type: "Parameter"
    shape: { dim: 1 dim: 512 }
    initializer {
      type: "Constant"
      multiplier: 0
    }
  }
  variable {
    name: "BatchNormalization_3/bn/gamma"
    type: "Parameter"
    shape: { dim: 1 dim: 512 }
    initializer {
      type: "Constant"
      multiplier: 1
    }
  }
  variable {
    name: "BatchNormalization_3/bn/mean"
    type: "Parameter"
    shape: { dim: 1 dim: 512 }
    initializer {
      type: "Constant"
      multiplier: 0
    }
  }
  variable {
    name: "BatchNormalization_3/bn/var"
    type: "Parameter"
    shape: { dim: 1 dim: 512 }
    initializer {
      type: "Constant"
      multiplier: 0
    }
  }
  variable {
    name: "BinaryConnectAffine_2/bicon_affine/W"
    type: "Parameter"
    shape: { dim: 512 dim: 10 }
    initializer {
      type: "UniformAffineGlorot"
      multiplier: 1
    }
  }
  variable {
    name: "BinaryConnectAffine_2/bicon_affine/Wb"
    type: "Parameter"
    shape: { dim: 512 dim: 10 }
    initializer {
      type: "Constant"
      multiplier: 0
    }
  }
  variable {
    name: "BinaryConnectAffine_2/bicon_affine/b"
    type: "Parameter"
    shape: { dim: 10 }
    initializer {
      type: "Constant"
      multiplier: 0
    }
  }
  variable {
    name: "BatchNormalization_4/bn/beta"
    type: "Parameter"
    shape: { dim: 1 dim: 10 }
    initializer {
      type: "Constant"
      multiplier: 0
    }
  }
  variable {
    name: "BatchNormalization_4/bn/gamma"
    type: "Parameter"
    shape: { dim: 1 dim: 10 }
    initializer {
      type: "Constant"
      multiplier: 1
    }
  }
  variable {
    name: "BatchNormalization_4/bn/mean"
    type: "Parameter"
    shape: { dim: 1 dim: 10 }
    initializer {
      type: "Constant"
      multiplier: 0
    }
  }
  variable {
    name: "BatchNormalization_4/bn/var"
    type: "Parameter"
    shape: { dim: 1 dim: 10 }
    initializer {
      type: "Constant"
      multiplier: 0
    }
  }
  variable {
    name: "CategoricalCrossEntropy_T"
    type: "Buffer"
    shape: { dim:-1 dim: 1 }
  }
  variable {
    name: "BinaryConnectConvolution"
    type: "Buffer"
    shape: { dim:-1 dim: 64 dim: 24 dim: 24 }
  }
  variable {
    name: "MaxPooling"
    type: "Buffer"
    shape: { dim:-1 dim: 64 dim: 12 dim: 12 }
  }
  variable {
    name: "BatchNormalization"
    type: "Buffer"
    shape: { dim:-1 dim: 64 dim: 12 dim: 12 }
  }
  variable {
    name: "BinarySigmoid"
    type: "Buffer"
    shape: { dim:-1 dim: 64 dim: 12 dim: 12 }
  }
  variable {
    name: "BinaryConnectConvolution_2"
    type: "Buffer"
    shape: { dim:-1 dim: 64 dim: 8 dim: 8 }
  }
  variable {
    name: "MaxPooling_2"
    type: "Buffer"
    shape: { dim:-1 dim: 64 dim: 4 dim: 4 }
  }
  variable {
    name: "BatchNormalization_2"
    type: "Buffer"
    shape: { dim:-1 dim: 64 dim: 4 dim: 4 }
  }
  variable {
    name: "BinarySigmoid_2"
    type: "Buffer"
    shape: { dim:-1 dim: 64 dim: 4 dim: 4 }
  }
  variable {
    name: "BinaryConnectAffine"
    type: "Buffer"
    shape: { dim:-1 dim: 512 }
  }
  variable {
    name: "BatchNormalization_3"
    type: "Buffer"
    shape: { dim:-1 dim: 512 }
  }
  variable {
    name: "BinarySigmoid_3"
    type: "Buffer"
    shape: { dim:-1 dim: 512 }
  }
  variable {
    name: "BinaryConnectAffine_2"
    type: "Buffer"
    shape: { dim:-1 dim: 10 }
  }
  variable {
    name: "BatchNormalization_4"
    type: "Buffer"
    shape: { dim:-1 dim: 10 }
  }
  variable {
    name: "Softmax"
    type: "Buffer"
    shape: { dim:-1 dim: 10 }
  }
  variable {
    name: "CategoricalCrossEntropy"
    type: "Buffer"
    shape: { dim:-1 dim: 1 }
  }
  function {
    name: "BinaryConnectConvolution"
    type: "BinaryConnectConvolution"
    input: "Input"
    input: "BinaryConnectConvolution/bicon_conv/W"
    input: "BinaryConnectConvolution/bicon_conv/Wb"
    input: "BinaryConnectConvolution/bicon_conv/b"
    output: "BinaryConnectConvolution"
    binary_connect_convolution_param {
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
    input: "BinaryConnectConvolution"
    output: "MaxPooling"
    max_pooling_param {
      kernel: { dim: 2 dim: 2 }
      stride: { dim: 2 dim: 2 }
      ignore_border: true
      pad: { dim: 0 dim: 0 }
    }
  }
  function {
    name: "BatchNormalization"
    type: "BatchNormalization"
    input: "MaxPooling"
    input: "BatchNormalization/bn/beta"
    input: "BatchNormalization/bn/gamma"
    input: "BatchNormalization/bn/mean"
    input: "BatchNormalization/bn/var"
    output: "BatchNormalization"
    batch_normalization_param {
      axes: 1
      decay_rate: 0.5
      eps: 0.01
      batch_stat: True
    }
  }
  function {
    name: "BinarySigmoid"
    type: "BinarySigmoid"
    input: "BatchNormalization"
    output: "BinarySigmoid"
  }
  function {
    name: "BinaryConnectConvolution_2"
    type: "BinaryConnectConvolution"
    input: "BinarySigmoid"
    input: "BinaryConnectConvolution_2/bicon_conv/W"
    input: "BinaryConnectConvolution_2/bicon_conv/Wb"
    input: "BinaryConnectConvolution_2/bicon_conv/b"
    output: "BinaryConnectConvolution_2"
    binary_connect_convolution_param {
      pad: { dim: 0 dim: 0 }
      stride: { dim: 1 dim: 1 }
      dilation: { dim: 1 dim: 1 }
      group: 1
      base_axis: 1
    }
  }
  function {
    name: "MaxPooling_2"
    type: "MaxPooling"
    input: "BinaryConnectConvolution_2"
    output: "MaxPooling_2"
    max_pooling_param {
      kernel: { dim: 2 dim: 2 }
      stride: { dim: 2 dim: 2 }
      ignore_border: true
      pad: { dim: 0 dim: 0 }
    }
  }
  function {
    name: "BatchNormalization_2"
    type: "BatchNormalization"
    input: "MaxPooling_2"
    input: "BatchNormalization_2/bn/beta"
    input: "BatchNormalization_2/bn/gamma"
    input: "BatchNormalization_2/bn/mean"
    input: "BatchNormalization_2/bn/var"
    output: "BatchNormalization_2"
    batch_normalization_param {
      axes: 1
      decay_rate: 0.5
      eps: 0.01
      batch_stat: True
    }
  }
  function {
    name: "BinarySigmoid_2"
    type: "BinarySigmoid"
    input: "BatchNormalization_2"
    output: "BinarySigmoid_2"
  }
  function {
    name: "BinaryConnectAffine"
    type: "BinaryConnectAffine"
    input: "BinarySigmoid_2"
    input: "BinaryConnectAffine/bicon_affine/W"
    input: "BinaryConnectAffine/bicon_affine/Wb"
    input: "BinaryConnectAffine/bicon_affine/b"
    output: "BinaryConnectAffine"
    binary_connect_affine_param {
      base_axis: 1
    }
  }
  function {
    name: "BatchNormalization_3"
    type: "BatchNormalization"
    input: "BinaryConnectAffine"
    input: "BatchNormalization_3/bn/beta"
    input: "BatchNormalization_3/bn/gamma"
    input: "BatchNormalization_3/bn/mean"
    input: "BatchNormalization_3/bn/var"
    output: "BatchNormalization_3"
    batch_normalization_param {
      axes: 1
      decay_rate: 0.5
      eps: 0.01
      batch_stat: True
    }
  }
  function {
    name: "BinarySigmoid_3"
    type: "BinarySigmoid"
    input: "BatchNormalization_3"
    output: "BinarySigmoid_3"
  }
  function {
    name: "BinaryConnectAffine_2"
    type: "BinaryConnectAffine"
    input: "BinarySigmoid_3"
    input: "BinaryConnectAffine_2/bicon_affine/W"
    input: "BinaryConnectAffine_2/bicon_affine/Wb"
    input: "BinaryConnectAffine_2/bicon_affine/b"
    output: "BinaryConnectAffine_2"
    binary_connect_affine_param {
      base_axis: 1
    }
  }
  function {
    name: "BatchNormalization_4"
    type: "BatchNormalization"
    input: "BinaryConnectAffine_2"
    input: "BatchNormalization_4/bn/beta"
    input: "BatchNormalization_4/bn/gamma"
    input: "BatchNormalization_4/bn/mean"
    input: "BatchNormalization_4/bn/var"
    output: "BatchNormalization_4"
    batch_normalization_param {
      axes: 1
      decay_rate: 0.5
      eps: 0.01
      batch_stat: True
    }
  }
  function {
    name: "Softmax"
    type: "Softmax"
    input: "BatchNormalization_4"
    output: "Softmax"
    softmax_param {
      axis: 1
    }
  }
  function {
    name: "CategoricalCrossEntropy"
    type: "CategoricalCrossEntropy"
    input: "Softmax"
    input: "CategoricalCrossEntropy_T"
    output: "CategoricalCrossEntropy"
    categorical_cross_entropy_param {
      axis: 1
    }
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
    name: "BinaryConnectConvolution/bicon_conv/W"
    type: "Parameter"
    shape: { dim: 64 dim: 1 dim: 5 dim: 5 }
    initializer {
      type: "UniformConvolutionGlorot"
      multiplier: 1
    }
  }
  variable {
    name: "BinaryConnectConvolution/bicon_conv/Wb"
    type: "Parameter"
    shape: { dim: 64 dim: 1 dim: 5 dim: 5 }
    initializer {
      type: "Constant"
      multiplier: 0
    }
  }
  variable {
    name: "BinaryConnectConvolution/bicon_conv/b"
    type: "Parameter"
    shape: { dim: 64 }
    initializer {
      type: "Constant"
      multiplier: 0
    }
  }
  variable {
    name: "BatchNormalization/bn/beta"
    type: "Parameter"
    shape: { dim: 1 dim: 64 dim: 1 dim: 1 }
    initializer {
      type: "Constant"
      multiplier: 0
    }
  }
  variable {
    name: "BatchNormalization/bn/gamma"
    type: "Parameter"
    shape: { dim: 1 dim: 64 dim: 1 dim: 1 }
    initializer {
      type: "Constant"
      multiplier: 1
    }
  }
  variable {
    name: "BatchNormalization/bn/mean"
    type: "Parameter"
    shape: { dim: 1 dim: 64 dim: 1 dim: 1 }
    initializer {
      type: "Constant"
      multiplier: 0
    }
  }
  variable {
    name: "BatchNormalization/bn/var"
    type: "Parameter"
    shape: { dim: 1 dim: 64 dim: 1 dim: 1 }
    initializer {
      type: "Constant"
      multiplier: 0
    }
  }
  variable {
    name: "BinaryConnectConvolution_2/bicon_conv/W"
    type: "Parameter"
    shape: { dim: 64 dim: 64 dim: 5 dim: 5 }
    initializer {
      type: "UniformConvolutionGlorot"
      multiplier: 1
    }
  }
  variable {
    name: "BinaryConnectConvolution_2/bicon_conv/Wb"
    type: "Parameter"
    shape: { dim: 64 dim: 64 dim: 5 dim: 5 }
    initializer {
      type: "Constant"
      multiplier: 0
    }
  }
  variable {
    name: "BinaryConnectConvolution_2/bicon_conv/b"
    type: "Parameter"
    shape: { dim: 64 }
    initializer {
      type: "Constant"
      multiplier: 0
    }
  }
  variable {
    name: "BatchNormalization_2/bn/beta"
    type: "Parameter"
    shape: { dim: 1 dim: 64 dim: 1 dim: 1 }
    initializer {
      type: "Constant"
      multiplier: 0
    }
  }
  variable {
    name: "BatchNormalization_2/bn/gamma"
    type: "Parameter"
    shape: { dim: 1 dim: 64 dim: 1 dim: 1 }
    initializer {
      type: "Constant"
      multiplier: 1
    }
  }
  variable {
    name: "BatchNormalization_2/bn/mean"
    type: "Parameter"
    shape: { dim: 1 dim: 64 dim: 1 dim: 1 }
    initializer {
      type: "Constant"
      multiplier: 0
    }
  }
  variable {
    name: "BatchNormalization_2/bn/var"
    type: "Parameter"
    shape: { dim: 1 dim: 64 dim: 1 dim: 1 }
    initializer {
      type: "Constant"
      multiplier: 0
    }
  }
  variable {
    name: "BinaryConnectAffine/bicon_affine/W"
    type: "Parameter"
    shape: { dim: 1024 dim: 512 }
    initializer {
      type: "UniformAffineGlorot"
      multiplier: 1
    }
  }
  variable {
    name: "BinaryConnectAffine/bicon_affine/Wb"
    type: "Parameter"
    shape: { dim: 1024 dim: 512 }
    initializer {
      type: "Constant"
      multiplier: 0
    }
  }
  variable {
    name: "BinaryConnectAffine/bicon_affine/b"
    type: "Parameter"
    shape: { dim: 512 }
    initializer {
      type: "Constant"
      multiplier: 0
    }
  }
  variable {
    name: "BatchNormalization_3/bn/beta"
    type: "Parameter"
    shape: { dim: 1 dim: 512 }
    initializer {
      type: "Constant"
      multiplier: 0
    }
  }
  variable {
    name: "BatchNormalization_3/bn/gamma"
    type: "Parameter"
    shape: { dim: 1 dim: 512 }
    initializer {
      type: "Constant"
      multiplier: 1
    }
  }
  variable {
    name: "BatchNormalization_3/bn/mean"
    type: "Parameter"
    shape: { dim: 1 dim: 512 }
    initializer {
      type: "Constant"
      multiplier: 0
    }
  }
  variable {
    name: "BatchNormalization_3/bn/var"
    type: "Parameter"
    shape: { dim: 1 dim: 512 }
    initializer {
      type: "Constant"
      multiplier: 0
    }
  }
  variable {
    name: "BinaryConnectAffine_2/bicon_affine/W"
    type: "Parameter"
    shape: { dim: 512 dim: 10 }
    initializer {
      type: "UniformAffineGlorot"
      multiplier: 1
    }
  }
  variable {
    name: "BinaryConnectAffine_2/bicon_affine/Wb"
    type: "Parameter"
    shape: { dim: 512 dim: 10 }
    initializer {
      type: "Constant"
      multiplier: 0
    }
  }
  variable {
    name: "BinaryConnectAffine_2/bicon_affine/b"
    type: "Parameter"
    shape: { dim: 10 }
    initializer {
      type: "Constant"
      multiplier: 0
    }
  }
  variable {
    name: "BatchNormalization_4/bn/beta"
    type: "Parameter"
    shape: { dim: 1 dim: 10 }
    initializer {
      type: "Constant"
      multiplier: 0
    }
  }
  variable {
    name: "BatchNormalization_4/bn/gamma"
    type: "Parameter"
    shape: { dim: 1 dim: 10 }
    initializer {
      type: "Constant"
      multiplier: 1
    }
  }
  variable {
    name: "BatchNormalization_4/bn/mean"
    type: "Parameter"
    shape: { dim: 1 dim: 10 }
    initializer {
      type: "Constant"
      multiplier: 0
    }
  }
  variable {
    name: "BatchNormalization_4/bn/var"
    type: "Parameter"
    shape: { dim: 1 dim: 10 }
    initializer {
      type: "Constant"
      multiplier: 0
    }
  }
  variable {
    name: "CategoricalCrossEntropy_T"
    type: "Buffer"
    shape: { dim:-1 dim: 1 }
  }
  variable {
    name: "BinaryConnectConvolution"
    type: "Buffer"
    shape: { dim:-1 dim: 64 dim: 24 dim: 24 }
  }
  variable {
    name: "MaxPooling"
    type: "Buffer"
    shape: { dim:-1 dim: 64 dim: 12 dim: 12 }
  }
  variable {
    name: "BatchNormalization"
    type: "Buffer"
    shape: { dim:-1 dim: 64 dim: 12 dim: 12 }
  }
  variable {
    name: "BinarySigmoid"
    type: "Buffer"
    shape: { dim:-1 dim: 64 dim: 12 dim: 12 }
  }
  variable {
    name: "BinaryConnectConvolution_2"
    type: "Buffer"
    shape: { dim:-1 dim: 64 dim: 8 dim: 8 }
  }
  variable {
    name: "MaxPooling_2"
    type: "Buffer"
    shape: { dim:-1 dim: 64 dim: 4 dim: 4 }
  }
  variable {
    name: "BatchNormalization_2"
    type: "Buffer"
    shape: { dim:-1 dim: 64 dim: 4 dim: 4 }
  }
  variable {
    name: "BinarySigmoid_2"
    type: "Buffer"
    shape: { dim:-1 dim: 64 dim: 4 dim: 4 }
  }
  variable {
    name: "BinaryConnectAffine"
    type: "Buffer"
    shape: { dim:-1 dim: 512 }
  }
  variable {
    name: "BatchNormalization_3"
    type: "Buffer"
    shape: { dim:-1 dim: 512 }
  }
  variable {
    name: "BinarySigmoid_3"
    type: "Buffer"
    shape: { dim:-1 dim: 512 }
  }
  variable {
    name: "BinaryConnectAffine_2"
    type: "Buffer"
    shape: { dim:-1 dim: 10 }
  }
  variable {
    name: "BatchNormalization_4"
    type: "Buffer"
    shape: { dim:-1 dim: 10 }
  }
  variable {
    name: "Softmax"
    type: "Buffer"
    shape: { dim:-1 dim: 10 }
  }
  variable {
    name: "CategoricalCrossEntropy"
    type: "Buffer"
    shape: { dim:-1 dim: 1 }
  }
  function {
    name: "BinaryConnectConvolution"
    type: "BinaryConnectConvolution"
    input: "Input"
    input: "BinaryConnectConvolution/bicon_conv/W"
    input: "BinaryConnectConvolution/bicon_conv/Wb"
    input: "BinaryConnectConvolution/bicon_conv/b"
    output: "BinaryConnectConvolution"
    binary_connect_convolution_param {
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
    input: "BinaryConnectConvolution"
    output: "MaxPooling"
    max_pooling_param {
      kernel: { dim: 2 dim: 2 }
      stride: { dim: 2 dim: 2 }
      ignore_border: true
      pad: { dim: 0 dim: 0 }
    }
  }
  function {
    name: "BatchNormalization"
    type: "BatchNormalization"
    input: "MaxPooling"
    input: "BatchNormalization/bn/beta"
    input: "BatchNormalization/bn/gamma"
    input: "BatchNormalization/bn/mean"
    input: "BatchNormalization/bn/var"
    output: "BatchNormalization"
    batch_normalization_param {
      axes: 1
      decay_rate: 0.5
      eps: 0.01
      batch_stat: False
    }
  }
  function {
    name: "BinarySigmoid"
    type: "BinarySigmoid"
    input: "BatchNormalization"
    output: "BinarySigmoid"
  }
  function {
    name: "BinaryConnectConvolution_2"
    type: "BinaryConnectConvolution"
    input: "BinarySigmoid"
    input: "BinaryConnectConvolution_2/bicon_conv/W"
    input: "BinaryConnectConvolution_2/bicon_conv/Wb"
    input: "BinaryConnectConvolution_2/bicon_conv/b"
    output: "BinaryConnectConvolution_2"
    binary_connect_convolution_param {
      pad: { dim: 0 dim: 0 }
      stride: { dim: 1 dim: 1 }
      dilation: { dim: 1 dim: 1 }
      group: 1
      base_axis: 1
    }
  }
  function {
    name: "MaxPooling_2"
    type: "MaxPooling"
    input: "BinaryConnectConvolution_2"
    output: "MaxPooling_2"
    max_pooling_param {
      kernel: { dim: 2 dim: 2 }
      stride: { dim: 2 dim: 2 }
      ignore_border: true
      pad: { dim: 0 dim: 0 }
    }
  }
  function {
    name: "BatchNormalization_2"
    type: "BatchNormalization"
    input: "MaxPooling_2"
    input: "BatchNormalization_2/bn/beta"
    input: "BatchNormalization_2/bn/gamma"
    input: "BatchNormalization_2/bn/mean"
    input: "BatchNormalization_2/bn/var"
    output: "BatchNormalization_2"
    batch_normalization_param {
      axes: 1
      decay_rate: 0.5
      eps: 0.01
      batch_stat: False
    }
  }
  function {
    name: "BinarySigmoid_2"
    type: "BinarySigmoid"
    input: "BatchNormalization_2"
    output: "BinarySigmoid_2"
  }
  function {
    name: "BinaryConnectAffine"
    type: "BinaryConnectAffine"
    input: "BinarySigmoid_2"
    input: "BinaryConnectAffine/bicon_affine/W"
    input: "BinaryConnectAffine/bicon_affine/Wb"
    input: "BinaryConnectAffine/bicon_affine/b"
    output: "BinaryConnectAffine"
    binary_connect_affine_param {
      base_axis: 1
    }
  }
  function {
    name: "BatchNormalization_3"
    type: "BatchNormalization"
    input: "BinaryConnectAffine"
    input: "BatchNormalization_3/bn/beta"
    input: "BatchNormalization_3/bn/gamma"
    input: "BatchNormalization_3/bn/mean"
    input: "BatchNormalization_3/bn/var"
    output: "BatchNormalization_3"
    batch_normalization_param {
      axes: 1
      decay_rate: 0.5
      eps: 0.01
      batch_stat: False
    }
  }
  function {
    name: "BinarySigmoid_3"
    type: "BinarySigmoid"
    input: "BatchNormalization_3"
    output: "BinarySigmoid_3"
  }
  function {
    name: "BinaryConnectAffine_2"
    type: "BinaryConnectAffine"
    input: "BinarySigmoid_3"
    input: "BinaryConnectAffine_2/bicon_affine/W"
    input: "BinaryConnectAffine_2/bicon_affine/Wb"
    input: "BinaryConnectAffine_2/bicon_affine/b"
    output: "BinaryConnectAffine_2"
    binary_connect_affine_param {
      base_axis: 1
    }
  }
  function {
    name: "BatchNormalization_4"
    type: "BatchNormalization"
    input: "BinaryConnectAffine_2"
    input: "BatchNormalization_4/bn/beta"
    input: "BatchNormalization_4/bn/gamma"
    input: "BatchNormalization_4/bn/mean"
    input: "BatchNormalization_4/bn/var"
    output: "BatchNormalization_4"
    batch_normalization_param {
      axes: 1
      decay_rate: 0.5
      eps: 0.01
      batch_stat: False
    }
  }
  function {
    name: "Softmax"
    type: "Softmax"
    input: "BatchNormalization_4"
    output: "Softmax"
    softmax_param {
      axis: 1
    }
  }
  function {
    name: "CategoricalCrossEntropy"
    type: "CategoricalCrossEntropy"
    input: "Softmax"
    input: "CategoricalCrossEntropy_T"
    output: "CategoricalCrossEntropy"
    categorical_cross_entropy_param {
      axis: 1
    }
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
    name: "BinaryConnectConvolution/bicon_conv/W"
    type: "Parameter"
    shape: { dim: 64 dim: 1 dim: 5 dim: 5 }
    initializer {
      type: "UniformConvolutionGlorot"
      multiplier: 1
    }
  }
  variable {
    name: "BinaryConnectConvolution/bicon_conv/Wb"
    type: "Parameter"
    shape: { dim: 64 dim: 1 dim: 5 dim: 5 }
    initializer {
      type: "Constant"
      multiplier: 0
    }
  }
  variable {
    name: "BinaryConnectConvolution/bicon_conv/b"
    type: "Parameter"
    shape: { dim: 64 }
    initializer {
      type: "Constant"
      multiplier: 0
    }
  }
  variable {
    name: "BatchNormalization/bn/beta"
    type: "Parameter"
    shape: { dim: 1 dim: 64 dim: 1 dim: 1 }
    initializer {
      type: "Constant"
      multiplier: 0
    }
  }
  variable {
    name: "BatchNormalization/bn/gamma"
    type: "Parameter"
    shape: { dim: 1 dim: 64 dim: 1 dim: 1 }
    initializer {
      type: "Constant"
      multiplier: 1
    }
  }
  variable {
    name: "BatchNormalization/bn/mean"
    type: "Parameter"
    shape: { dim: 1 dim: 64 dim: 1 dim: 1 }
    initializer {
      type: "Constant"
      multiplier: 0
    }
  }
  variable {
    name: "BatchNormalization/bn/var"
    type: "Parameter"
    shape: { dim: 1 dim: 64 dim: 1 dim: 1 }
    initializer {
      type: "Constant"
      multiplier: 0
    }
  }
  variable {
    name: "BinaryConnectConvolution_2/bicon_conv/W"
    type: "Parameter"
    shape: { dim: 64 dim: 64 dim: 5 dim: 5 }
    initializer {
      type: "UniformConvolutionGlorot"
      multiplier: 1
    }
  }
  variable {
    name: "BinaryConnectConvolution_2/bicon_conv/Wb"
    type: "Parameter"
    shape: { dim: 64 dim: 64 dim: 5 dim: 5 }
    initializer {
      type: "Constant"
      multiplier: 0
    }
  }
  variable {
    name: "BinaryConnectConvolution_2/bicon_conv/b"
    type: "Parameter"
    shape: { dim: 64 }
    initializer {
      type: "Constant"
      multiplier: 0
    }
  }
  variable {
    name: "BatchNormalization_2/bn/beta"
    type: "Parameter"
    shape: { dim: 1 dim: 64 dim: 1 dim: 1 }
    initializer {
      type: "Constant"
      multiplier: 0
    }
  }
  variable {
    name: "BatchNormalization_2/bn/gamma"
    type: "Parameter"
    shape: { dim: 1 dim: 64 dim: 1 dim: 1 }
    initializer {
      type: "Constant"
      multiplier: 1
    }
  }
  variable {
    name: "BatchNormalization_2/bn/mean"
    type: "Parameter"
    shape: { dim: 1 dim: 64 dim: 1 dim: 1 }
    initializer {
      type: "Constant"
      multiplier: 0
    }
  }
  variable {
    name: "BatchNormalization_2/bn/var"
    type: "Parameter"
    shape: { dim: 1 dim: 64 dim: 1 dim: 1 }
    initializer {
      type: "Constant"
      multiplier: 0
    }
  }
  variable {
    name: "BinaryConnectAffine/bicon_affine/W"
    type: "Parameter"
    shape: { dim: 1024 dim: 512 }
    initializer {
      type: "UniformAffineGlorot"
      multiplier: 1
    }
  }
  variable {
    name: "BinaryConnectAffine/bicon_affine/Wb"
    type: "Parameter"
    shape: { dim: 1024 dim: 512 }
    initializer {
      type: "Constant"
      multiplier: 0
    }
  }
  variable {
    name: "BinaryConnectAffine/bicon_affine/b"
    type: "Parameter"
    shape: { dim: 512 }
    initializer {
      type: "Constant"
      multiplier: 0
    }
  }
  variable {
    name: "BatchNormalization_3/bn/beta"
    type: "Parameter"
    shape: { dim: 1 dim: 512 }
    initializer {
      type: "Constant"
      multiplier: 0
    }
  }
  variable {
    name: "BatchNormalization_3/bn/gamma"
    type: "Parameter"
    shape: { dim: 1 dim: 512 }
    initializer {
      type: "Constant"
      multiplier: 1
    }
  }
  variable {
    name: "BatchNormalization_3/bn/mean"
    type: "Parameter"
    shape: { dim: 1 dim: 512 }
    initializer {
      type: "Constant"
      multiplier: 0
    }
  }
  variable {
    name: "BatchNormalization_3/bn/var"
    type: "Parameter"
    shape: { dim: 1 dim: 512 }
    initializer {
      type: "Constant"
      multiplier: 0
    }
  }
  variable {
    name: "BinaryConnectAffine_2/bicon_affine/W"
    type: "Parameter"
    shape: { dim: 512 dim: 10 }
    initializer {
      type: "UniformAffineGlorot"
      multiplier: 1
    }
  }
  variable {
    name: "BinaryConnectAffine_2/bicon_affine/Wb"
    type: "Parameter"
    shape: { dim: 512 dim: 10 }
    initializer {
      type: "Constant"
      multiplier: 0
    }
  }
  variable {
    name: "BinaryConnectAffine_2/bicon_affine/b"
    type: "Parameter"
    shape: { dim: 10 }
    initializer {
      type: "Constant"
      multiplier: 0
    }
  }
  variable {
    name: "BatchNormalization_4/bn/beta"
    type: "Parameter"
    shape: { dim: 1 dim: 10 }
    initializer {
      type: "Constant"
      multiplier: 0
    }
  }
  variable {
    name: "BatchNormalization_4/bn/gamma"
    type: "Parameter"
    shape: { dim: 1 dim: 10 }
    initializer {
      type: "Constant"
      multiplier: 1
    }
  }
  variable {
    name: "BatchNormalization_4/bn/mean"
    type: "Parameter"
    shape: { dim: 1 dim: 10 }
    initializer {
      type: "Constant"
      multiplier: 0
    }
  }
  variable {
    name: "BatchNormalization_4/bn/var"
    type: "Parameter"
    shape: { dim: 1 dim: 10 }
    initializer {
      type: "Constant"
      multiplier: 0
    }
  }
  variable {
    name: "BinaryConnectConvolution"
    type: "Buffer"
    shape: { dim:-1 dim: 64 dim: 24 dim: 24 }
  }
  variable {
    name: "MaxPooling"
    type: "Buffer"
    shape: { dim:-1 dim: 64 dim: 12 dim: 12 }
  }
  variable {
    name: "BatchNormalization"
    type: "Buffer"
    shape: { dim:-1 dim: 64 dim: 12 dim: 12 }
  }
  variable {
    name: "BinarySigmoid"
    type: "Buffer"
    shape: { dim:-1 dim: 64 dim: 12 dim: 12 }
  }
  variable {
    name: "BinaryConnectConvolution_2"
    type: "Buffer"
    shape: { dim:-1 dim: 64 dim: 8 dim: 8 }
  }
  variable {
    name: "MaxPooling_2"
    type: "Buffer"
    shape: { dim:-1 dim: 64 dim: 4 dim: 4 }
  }
  variable {
    name: "BatchNormalization_2"
    type: "Buffer"
    shape: { dim:-1 dim: 64 dim: 4 dim: 4 }
  }
  variable {
    name: "BinarySigmoid_2"
    type: "Buffer"
    shape: { dim:-1 dim: 64 dim: 4 dim: 4 }
  }
  variable {
    name: "BinaryConnectAffine"
    type: "Buffer"
    shape: { dim:-1 dim: 512 }
  }
  variable {
    name: "BatchNormalization_3"
    type: "Buffer"
    shape: { dim:-1 dim: 512 }
  }
  variable {
    name: "BinarySigmoid_3"
    type: "Buffer"
    shape: { dim:-1 dim: 512 }
  }
  variable {
    name: "BinaryConnectAffine_2"
    type: "Buffer"
    shape: { dim:-1 dim: 10 }
  }
  variable {
    name: "BatchNormalization_4"
    type: "Buffer"
    shape: { dim:-1 dim: 10 }
  }
  variable {
    name: "Softmax"
    type: "Buffer"
    shape: { dim:-1 dim: 10 }
  }
  function {
    name: "BinaryConnectConvolution"
    type: "BinaryConnectConvolution"
    input: "Input"
    input: "BinaryConnectConvolution/bicon_conv/W"
    input: "BinaryConnectConvolution/bicon_conv/Wb"
    input: "BinaryConnectConvolution/bicon_conv/b"
    output: "BinaryConnectConvolution"
    binary_connect_convolution_param {
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
    input: "BinaryConnectConvolution"
    output: "MaxPooling"
    max_pooling_param {
      kernel: { dim: 2 dim: 2 }
      stride: { dim: 2 dim: 2 }
      ignore_border: true
      pad: { dim: 0 dim: 0 }
    }
  }
  function {
    name: "BatchNormalization"
    type: "BatchNormalization"
    input: "MaxPooling"
    input: "BatchNormalization/bn/beta"
    input: "BatchNormalization/bn/gamma"
    input: "BatchNormalization/bn/mean"
    input: "BatchNormalization/bn/var"
    output: "BatchNormalization"
    batch_normalization_param {
      axes: 1
      decay_rate: 0.5
      eps: 0.01
      batch_stat: False
    }
  }
  function {
    name: "BinarySigmoid"
    type: "BinarySigmoid"
    input: "BatchNormalization"
    output: "BinarySigmoid"
  }
  function {
    name: "BinaryConnectConvolution_2"
    type: "BinaryConnectConvolution"
    input: "BinarySigmoid"
    input: "BinaryConnectConvolution_2/bicon_conv/W"
    input: "BinaryConnectConvolution_2/bicon_conv/Wb"
    input: "BinaryConnectConvolution_2/bicon_conv/b"
    output: "BinaryConnectConvolution_2"
    binary_connect_convolution_param {
      pad: { dim: 0 dim: 0 }
      stride: { dim: 1 dim: 1 }
      dilation: { dim: 1 dim: 1 }
      group: 1
      base_axis: 1
    }
  }
  function {
    name: "MaxPooling_2"
    type: "MaxPooling"
    input: "BinaryConnectConvolution_2"
    output: "MaxPooling_2"
    max_pooling_param {
      kernel: { dim: 2 dim: 2 }
      stride: { dim: 2 dim: 2 }
      ignore_border: true
      pad: { dim: 0 dim: 0 }
    }
  }
  function {
    name: "BatchNormalization_2"
    type: "BatchNormalization"
    input: "MaxPooling_2"
    input: "BatchNormalization_2/bn/beta"
    input: "BatchNormalization_2/bn/gamma"
    input: "BatchNormalization_2/bn/mean"
    input: "BatchNormalization_2/bn/var"
    output: "BatchNormalization_2"
    batch_normalization_param {
      axes: 1
      decay_rate: 0.5
      eps: 0.01
      batch_stat: False
    }
  }
  function {
    name: "BinarySigmoid_2"
    type: "BinarySigmoid"
    input: "BatchNormalization_2"
    output: "BinarySigmoid_2"
  }
  function {
    name: "BinaryConnectAffine"
    type: "BinaryConnectAffine"
    input: "BinarySigmoid_2"
    input: "BinaryConnectAffine/bicon_affine/W"
    input: "BinaryConnectAffine/bicon_affine/Wb"
    input: "BinaryConnectAffine/bicon_affine/b"
    output: "BinaryConnectAffine"
    binary_connect_affine_param {
      base_axis: 1
    }
  }
  function {
    name: "BatchNormalization_3"
    type: "BatchNormalization"
    input: "BinaryConnectAffine"
    input: "BatchNormalization_3/bn/beta"
    input: "BatchNormalization_3/bn/gamma"
    input: "BatchNormalization_3/bn/mean"
    input: "BatchNormalization_3/bn/var"
    output: "BatchNormalization_3"
    batch_normalization_param {
      axes: 1
      decay_rate: 0.5
      eps: 0.01
      batch_stat: False
    }
  }
  function {
    name: "BinarySigmoid_3"
    type: "BinarySigmoid"
    input: "BatchNormalization_3"
    output: "BinarySigmoid_3"
  }
  function {
    name: "BinaryConnectAffine_2"
    type: "BinaryConnectAffine"
    input: "BinarySigmoid_3"
    input: "BinaryConnectAffine_2/bicon_affine/W"
    input: "BinaryConnectAffine_2/bicon_affine/Wb"
    input: "BinaryConnectAffine_2/bicon_affine/b"
    output: "BinaryConnectAffine_2"
    binary_connect_affine_param {
      base_axis: 1
    }
  }
  function {
    name: "BatchNormalization_4"
    type: "BatchNormalization"
    input: "BinaryConnectAffine_2"
    input: "BatchNormalization_4/bn/beta"
    input: "BatchNormalization_4/bn/gamma"
    input: "BatchNormalization_4/bn/mean"
    input: "BatchNormalization_4/bn/var"
    output: "BatchNormalization_4"
    batch_normalization_param {
      axes: 1
      decay_rate: 0.5
      eps: 0.01
      batch_stat: False
    }
  }
  function {
    name: "Softmax"
    type: "Softmax"
    input: "BatchNormalization_4"
    output: "Softmax"
    softmax_param {
      axis: 1
    }
  }
}
dataset {
  name: "Training"
  uri: "C:\\Utils\\neural_network_console_internal_v120\\samples\\sample_dataset\\mnist\\mnist_training.csv"
  cache_dir: "C:\\Utils\\neural_network_console_internal_v120\\samples\\sample_dataset\\mnist\\mnist_training.cache"
  overwrite_cache: True
  create_cache_explicitly: True
  shuffle: true
  no_image_normalization: False
  batch_size: 64
}
dataset {
  name: "Validation"
  uri: "C:\\Utils\\neural_network_console_internal_v120\\samples\\sample_dataset\\mnist\\mnist_test.csv"
  cache_dir: "C:\\Utils\\neural_network_console_internal_v120\\samples\\sample_dataset\\mnist\\mnist_test.cache"
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
      alpha: 0.0001
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
    variable_name: "CategoricalCrossEntropy_T"
    data_name: "y"
  }
  loss_variable {
    variable_name: "CategoricalCrossEntropy"
  }
  parameter_variable {
    variable_name: "BinaryConnectConvolution/bicon_conv/W"
    learning_rate_multiplier: 1
  }
  parameter_variable {
    variable_name: "BinaryConnectConvolution/bicon_conv/Wb"
    learning_rate_multiplier: 0
  }
  parameter_variable {
    variable_name: "BinaryConnectConvolution/bicon_conv/b"
    learning_rate_multiplier: 1
  }
  parameter_variable {
    variable_name: "BatchNormalization/bn/beta"
    learning_rate_multiplier: 1
  }
  parameter_variable {
    variable_name: "BatchNormalization/bn/gamma"
    learning_rate_multiplier: 1
  }
  parameter_variable {
    variable_name: "BatchNormalization/bn/mean"
    learning_rate_multiplier: 0
  }
  parameter_variable {
    variable_name: "BatchNormalization/bn/var"
    learning_rate_multiplier: 0
  }
  parameter_variable {
    variable_name: "BinaryConnectConvolution_2/bicon_conv/W"
    learning_rate_multiplier: 1
  }
  parameter_variable {
    variable_name: "BinaryConnectConvolution_2/bicon_conv/Wb"
    learning_rate_multiplier: 0
  }
  parameter_variable {
    variable_name: "BinaryConnectConvolution_2/bicon_conv/b"
    learning_rate_multiplier: 1
  }
  parameter_variable {
    variable_name: "BatchNormalization_2/bn/beta"
    learning_rate_multiplier: 1
  }
  parameter_variable {
    variable_name: "BatchNormalization_2/bn/gamma"
    learning_rate_multiplier: 1
  }
  parameter_variable {
    variable_name: "BatchNormalization_2/bn/mean"
    learning_rate_multiplier: 0
  }
  parameter_variable {
    variable_name: "BatchNormalization_2/bn/var"
    learning_rate_multiplier: 0
  }
  parameter_variable {
    variable_name: "BinaryConnectAffine/bicon_affine/W"
    learning_rate_multiplier: 1
  }
  parameter_variable {
    variable_name: "BinaryConnectAffine/bicon_affine/Wb"
    learning_rate_multiplier: 0
  }
  parameter_variable {
    variable_name: "BinaryConnectAffine/bicon_affine/b"
    learning_rate_multiplier: 1
  }
  parameter_variable {
    variable_name: "BatchNormalization_3/bn/beta"
    learning_rate_multiplier: 1
  }
  parameter_variable {
    variable_name: "BatchNormalization_3/bn/gamma"
    learning_rate_multiplier: 1
  }
  parameter_variable {
    variable_name: "BatchNormalization_3/bn/mean"
    learning_rate_multiplier: 0
  }
  parameter_variable {
    variable_name: "BatchNormalization_3/bn/var"
    learning_rate_multiplier: 0
  }
  parameter_variable {
    variable_name: "BinaryConnectAffine_2/bicon_affine/W"
    learning_rate_multiplier: 1
  }
  parameter_variable {
    variable_name: "BinaryConnectAffine_2/bicon_affine/Wb"
    learning_rate_multiplier: 0
  }
  parameter_variable {
    variable_name: "BinaryConnectAffine_2/bicon_affine/b"
    learning_rate_multiplier: 1
  }
  parameter_variable {
    variable_name: "BatchNormalization_4/bn/beta"
    learning_rate_multiplier: 1
  }
  parameter_variable {
    variable_name: "BatchNormalization_4/bn/gamma"
    learning_rate_multiplier: 1
  }
  parameter_variable {
    variable_name: "BatchNormalization_4/bn/mean"
    learning_rate_multiplier: 0
  }
  parameter_variable {
    variable_name: "BatchNormalization_4/bn/var"
    learning_rate_multiplier: 0
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
    variable_name: "CategoricalCrossEntropy_T"
    data_name: "y"
  }
  monitor_variable {
    type: "Error"
    variable_name: "CategoricalCrossEntropy"
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
    variable_name: "CategoricalCrossEntropy_T"
    data_name: "y"
  }
  monitor_variable {
    type: "Error"
    variable_name: "CategoricalCrossEntropy"
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
    variable_name: "Softmax"
    data_name: "y'"
  }
  parameter_variable {
    variable_name: "BinaryConnectConvolution/bicon_conv/W"
  }
  parameter_variable {
    variable_name: "BinaryConnectConvolution/bicon_conv/Wb"
  }
  parameter_variable {
    variable_name: "BinaryConnectConvolution/bicon_conv/b"
  }
  parameter_variable {
    variable_name: "BatchNormalization/bn/beta"
  }
  parameter_variable {
    variable_name: "BatchNormalization/bn/gamma"
  }
  parameter_variable {
    variable_name: "BatchNormalization/bn/mean"
  }
  parameter_variable {
    variable_name: "BatchNormalization/bn/var"
  }
  parameter_variable {
    variable_name: "BinaryConnectConvolution_2/bicon_conv/W"
  }
  parameter_variable {
    variable_name: "BinaryConnectConvolution_2/bicon_conv/Wb"
  }
  parameter_variable {
    variable_name: "BinaryConnectConvolution_2/bicon_conv/b"
  }
  parameter_variable {
    variable_name: "BatchNormalization_2/bn/beta"
  }
  parameter_variable {
    variable_name: "BatchNormalization_2/bn/gamma"
  }
  parameter_variable {
    variable_name: "BatchNormalization_2/bn/mean"
  }
  parameter_variable {
    variable_name: "BatchNormalization_2/bn/var"
  }
  parameter_variable {
    variable_name: "BinaryConnectAffine/bicon_affine/W"
  }
  parameter_variable {
    variable_name: "BinaryConnectAffine/bicon_affine/Wb"
  }
  parameter_variable {
    variable_name: "BinaryConnectAffine/bicon_affine/b"
  }
  parameter_variable {
    variable_name: "BatchNormalization_3/bn/beta"
  }
  parameter_variable {
    variable_name: "BatchNormalization_3/bn/gamma"
  }
  parameter_variable {
    variable_name: "BatchNormalization_3/bn/mean"
  }
  parameter_variable {
    variable_name: "BatchNormalization_3/bn/var"
  }
  parameter_variable {
    variable_name: "BinaryConnectAffine_2/bicon_affine/W"
  }
  parameter_variable {
    variable_name: "BinaryConnectAffine_2/bicon_affine/Wb"
  }
  parameter_variable {
    variable_name: "BinaryConnectAffine_2/bicon_affine/b"
  }
  parameter_variable {
    variable_name: "BatchNormalization_4/bn/beta"
  }
  parameter_variable {
    variable_name: "BatchNormalization_4/bn/gamma"
  }
  parameter_variable {
    variable_name: "BatchNormalization_4/bn/mean"
  }
  parameter_variable {
    variable_name: "BatchNormalization_4/bn/var"
  }
}
'''
# The nntxt is generated from /home/woody/aid/nnabla-sample-data/sample_project/tutorial/binary_networks/binary_connect_mnist_LeNet.files/20180720_195907/net.nntxt.
N00000016 = r'''global_config {
  default_context {
    array_class: "CudaCachedArray"
    device_id: "0"
    backends: "cudnn:float"
    backends: "cuda:float"
    backends: "cpu:float"
  }
}
training_config {
  max_epoch: 100
  iter_per_epoch: 937
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
    name: "BinaryConnectConvolution/bicon_conv/W"
    type: "Parameter"
    shape: { dim: 64 dim: 1 dim: 5 dim: 5 }
    initializer {
      type: "UniformConvolutionGlorot"
      multiplier: 1
    }
  }
  variable {
    name: "BinaryConnectConvolution/bicon_conv/Wb"
    type: "Parameter"
    shape: { dim: 64 dim: 1 dim: 5 dim: 5 }
    initializer {
      type: "Constant"
      multiplier: 0
    }
  }
  variable {
    name: "BinaryConnectConvolution/bicon_conv/b"
    type: "Parameter"
    shape: { dim: 64 }
    initializer {
      type: "Constant"
      multiplier: 0
    }
  }
  variable {
    name: "BatchNormalization/bn/beta"
    type: "Parameter"
    shape: { dim: 1 dim: 64 dim: 1 dim: 1 }
    initializer {
      type: "Constant"
      multiplier: 0
    }
  }
  variable {
    name: "BatchNormalization/bn/gamma"
    type: "Parameter"
    shape: { dim: 1 dim: 64 dim: 1 dim: 1 }
    initializer {
      type: "Constant"
      multiplier: 1
    }
  }
  variable {
    name: "BatchNormalization/bn/mean"
    type: "Parameter"
    shape: { dim: 1 dim: 64 dim: 1 dim: 1 }
    initializer {
      type: "Constant"
      multiplier: 0
    }
  }
  variable {
    name: "BatchNormalization/bn/var"
    type: "Parameter"
    shape: { dim: 1 dim: 64 dim: 1 dim: 1 }
    initializer {
      type: "Constant"
      multiplier: 0
    }
  }
  variable {
    name: "BinaryConnectConvolution_2/bicon_conv/W"
    type: "Parameter"
    shape: { dim: 64 dim: 64 dim: 5 dim: 5 }
    initializer {
      type: "UniformConvolutionGlorot"
      multiplier: 1
    }
  }
  variable {
    name: "BinaryConnectConvolution_2/bicon_conv/Wb"
    type: "Parameter"
    shape: { dim: 64 dim: 64 dim: 5 dim: 5 }
    initializer {
      type: "Constant"
      multiplier: 0
    }
  }
  variable {
    name: "BinaryConnectConvolution_2/bicon_conv/b"
    type: "Parameter"
    shape: { dim: 64 }
    initializer {
      type: "Constant"
      multiplier: 0
    }
  }
  variable {
    name: "BatchNormalization_2/bn/beta"
    type: "Parameter"
    shape: { dim: 1 dim: 64 dim: 1 dim: 1 }
    initializer {
      type: "Constant"
      multiplier: 0
    }
  }
  variable {
    name: "BatchNormalization_2/bn/gamma"
    type: "Parameter"
    shape: { dim: 1 dim: 64 dim: 1 dim: 1 }
    initializer {
      type: "Constant"
      multiplier: 1
    }
  }
  variable {
    name: "BatchNormalization_2/bn/mean"
    type: "Parameter"
    shape: { dim: 1 dim: 64 dim: 1 dim: 1 }
    initializer {
      type: "Constant"
      multiplier: 0
    }
  }
  variable {
    name: "BatchNormalization_2/bn/var"
    type: "Parameter"
    shape: { dim: 1 dim: 64 dim: 1 dim: 1 }
    initializer {
      type: "Constant"
      multiplier: 0
    }
  }
  variable {
    name: "BinaryConnectAffine/bicon_affine/W"
    type: "Parameter"
    shape: { dim: 1024 dim: 512 }
    initializer {
      type: "UniformAffineGlorot"
      multiplier: 1
    }
  }
  variable {
    name: "BinaryConnectAffine/bicon_affine/Wb"
    type: "Parameter"
    shape: { dim: 1024 dim: 512 }
    initializer {
      type: "Constant"
      multiplier: 0
    }
  }
  variable {
    name: "BinaryConnectAffine/bicon_affine/b"
    type: "Parameter"
    shape: { dim: 512 }
    initializer {
      type: "Constant"
      multiplier: 0
    }
  }
  variable {
    name: "BatchNormalization_3/bn/beta"
    type: "Parameter"
    shape: { dim: 1 dim: 512 }
    initializer {
      type: "Constant"
      multiplier: 0
    }
  }
  variable {
    name: "BatchNormalization_3/bn/gamma"
    type: "Parameter"
    shape: { dim: 1 dim: 512 }
    initializer {
      type: "Constant"
      multiplier: 1
    }
  }
  variable {
    name: "BatchNormalization_3/bn/mean"
    type: "Parameter"
    shape: { dim: 1 dim: 512 }
    initializer {
      type: "Constant"
      multiplier: 0
    }
  }
  variable {
    name: "BatchNormalization_3/bn/var"
    type: "Parameter"
    shape: { dim: 1 dim: 512 }
    initializer {
      type: "Constant"
      multiplier: 0
    }
  }
  variable {
    name: "BinaryConnectAffine_2/bicon_affine/W"
    type: "Parameter"
    shape: { dim: 512 dim: 10 }
    initializer {
      type: "UniformAffineGlorot"
      multiplier: 1
    }
  }
  variable {
    name: "BinaryConnectAffine_2/bicon_affine/Wb"
    type: "Parameter"
    shape: { dim: 512 dim: 10 }
    initializer {
      type: "Constant"
      multiplier: 0
    }
  }
  variable {
    name: "BinaryConnectAffine_2/bicon_affine/b"
    type: "Parameter"
    shape: { dim: 10 }
    initializer {
      type: "Constant"
      multiplier: 0
    }
  }
  variable {
    name: "BatchNormalization_4/bn/beta"
    type: "Parameter"
    shape: { dim: 1 dim: 10 }
    initializer {
      type: "Constant"
      multiplier: 0
    }
  }
  variable {
    name: "BatchNormalization_4/bn/gamma"
    type: "Parameter"
    shape: { dim: 1 dim: 10 }
    initializer {
      type: "Constant"
      multiplier: 1
    }
  }
  variable {
    name: "BatchNormalization_4/bn/mean"
    type: "Parameter"
    shape: { dim: 1 dim: 10 }
    initializer {
      type: "Constant"
      multiplier: 0
    }
  }
  variable {
    name: "BatchNormalization_4/bn/var"
    type: "Parameter"
    shape: { dim: 1 dim: 10 }
    initializer {
      type: "Constant"
      multiplier: 0
    }
  }
  variable {
    name: "CategoricalCrossEntropy_T"
    type: "Buffer"
    shape: { dim:-1 dim: 1 }
  }
  variable {
    name: "BinaryConnectConvolution"
    type: "Buffer"
    shape: { dim:-1 dim: 64 dim: 24 dim: 24 }
  }
  variable {
    name: "MaxPooling"
    type: "Buffer"
    shape: { dim:-1 dim: 64 dim: 12 dim: 12 }
  }
  variable {
    name: "BatchNormalization"
    type: "Buffer"
    shape: { dim:-1 dim: 64 dim: 12 dim: 12 }
  }
  variable {
    name: "ReLU"
    type: "Buffer"
    shape: { dim:-1 dim: 64 dim: 12 dim: 12 }
  }
  variable {
    name: "BinaryConnectConvolution_2"
    type: "Buffer"
    shape: { dim:-1 dim: 64 dim: 8 dim: 8 }
  }
  variable {
    name: "MaxPooling_2"
    type: "Buffer"
    shape: { dim:-1 dim: 64 dim: 4 dim: 4 }
  }
  variable {
    name: "BatchNormalization_2"
    type: "Buffer"
    shape: { dim:-1 dim: 64 dim: 4 dim: 4 }
  }
  variable {
    name: "ReLU_3"
    type: "Buffer"
    shape: { dim:-1 dim: 64 dim: 4 dim: 4 }
  }
  variable {
    name: "BinaryConnectAffine"
    type: "Buffer"
    shape: { dim:-1 dim: 512 }
  }
  variable {
    name: "BatchNormalization_3"
    type: "Buffer"
    shape: { dim:-1 dim: 512 }
  }
  variable {
    name: "ReLU_2"
    type: "Buffer"
    shape: { dim:-1 dim: 512 }
  }
  variable {
    name: "BinaryConnectAffine_2"
    type: "Buffer"
    shape: { dim:-1 dim: 10 }
  }
  variable {
    name: "BatchNormalization_4"
    type: "Buffer"
    shape: { dim:-1 dim: 10 }
  }
  variable {
    name: "Softmax"
    type: "Buffer"
    shape: { dim:-1 dim: 10 }
  }
  variable {
    name: "CategoricalCrossEntropy"
    type: "Buffer"
    shape: { dim:-1 dim: 1 }
  }
  function {
    name: "BinaryConnectConvolution"
    type: "BinaryConnectConvolution"
    input: "Input"
    input: "BinaryConnectConvolution/bicon_conv/W"
    input: "BinaryConnectConvolution/bicon_conv/Wb"
    input: "BinaryConnectConvolution/bicon_conv/b"
    output: "BinaryConnectConvolution"
    binary_connect_convolution_param {
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
    input: "BinaryConnectConvolution"
    output: "MaxPooling"
    max_pooling_param {
      kernel: { dim: 2 dim: 2 }
      stride: { dim: 2 dim: 2 }
      ignore_border: true
      pad: { dim: 0 dim: 0 }
    }
  }
  function {
    name: "BatchNormalization"
    type: "BatchNormalization"
    input: "MaxPooling"
    input: "BatchNormalization/bn/beta"
    input: "BatchNormalization/bn/gamma"
    input: "BatchNormalization/bn/mean"
    input: "BatchNormalization/bn/var"
    output: "BatchNormalization"
    batch_normalization_param {
      axes: 1
      decay_rate: 0.5
      eps: 0.01
      batch_stat: True
    }
  }
  function {
    name: "ReLU"
    type: "ReLU"
    input: "BatchNormalization"
    output: "ReLU"
    relu_param {
      inplace: True
    }
  }
  function {
    name: "BinaryConnectConvolution_2"
    type: "BinaryConnectConvolution"
    input: "ReLU"
    input: "BinaryConnectConvolution_2/bicon_conv/W"
    input: "BinaryConnectConvolution_2/bicon_conv/Wb"
    input: "BinaryConnectConvolution_2/bicon_conv/b"
    output: "BinaryConnectConvolution_2"
    binary_connect_convolution_param {
      pad: { dim: 0 dim: 0 }
      stride: { dim: 1 dim: 1 }
      dilation: { dim: 1 dim: 1 }
      group: 1
      base_axis: 1
    }
  }
  function {
    name: "MaxPooling_2"
    type: "MaxPooling"
    input: "BinaryConnectConvolution_2"
    output: "MaxPooling_2"
    max_pooling_param {
      kernel: { dim: 2 dim: 2 }
      stride: { dim: 2 dim: 2 }
      ignore_border: true
      pad: { dim: 0 dim: 0 }
    }
  }
  function {
    name: "BatchNormalization_2"
    type: "BatchNormalization"
    input: "MaxPooling_2"
    input: "BatchNormalization_2/bn/beta"
    input: "BatchNormalization_2/bn/gamma"
    input: "BatchNormalization_2/bn/mean"
    input: "BatchNormalization_2/bn/var"
    output: "BatchNormalization_2"
    batch_normalization_param {
      axes: 1
      decay_rate: 0.5
      eps: 0.01
      batch_stat: True
    }
  }
  function {
    name: "ReLU_3"
    type: "ReLU"
    input: "BatchNormalization_2"
    output: "ReLU_3"
    relu_param {
      inplace: True
    }
  }
  function {
    name: "BinaryConnectAffine"
    type: "BinaryConnectAffine"
    input: "ReLU_3"
    input: "BinaryConnectAffine/bicon_affine/W"
    input: "BinaryConnectAffine/bicon_affine/Wb"
    input: "BinaryConnectAffine/bicon_affine/b"
    output: "BinaryConnectAffine"
    binary_connect_affine_param {
      base_axis: 1
    }
  }
  function {
    name: "BatchNormalization_3"
    type: "BatchNormalization"
    input: "BinaryConnectAffine"
    input: "BatchNormalization_3/bn/beta"
    input: "BatchNormalization_3/bn/gamma"
    input: "BatchNormalization_3/bn/mean"
    input: "BatchNormalization_3/bn/var"
    output: "BatchNormalization_3"
    batch_normalization_param {
      axes: 1
      decay_rate: 0.5
      eps: 0.01
      batch_stat: True
    }
  }
  function {
    name: "ReLU_2"
    type: "ReLU"
    input: "BatchNormalization_3"
    output: "ReLU_2"
    relu_param {
      inplace: True
    }
  }
  function {
    name: "BinaryConnectAffine_2"
    type: "BinaryConnectAffine"
    input: "ReLU_2"
    input: "BinaryConnectAffine_2/bicon_affine/W"
    input: "BinaryConnectAffine_2/bicon_affine/Wb"
    input: "BinaryConnectAffine_2/bicon_affine/b"
    output: "BinaryConnectAffine_2"
    binary_connect_affine_param {
      base_axis: 1
    }
  }
  function {
    name: "BatchNormalization_4"
    type: "BatchNormalization"
    input: "BinaryConnectAffine_2"
    input: "BatchNormalization_4/bn/beta"
    input: "BatchNormalization_4/bn/gamma"
    input: "BatchNormalization_4/bn/mean"
    input: "BatchNormalization_4/bn/var"
    output: "BatchNormalization_4"
    batch_normalization_param {
      axes: 1
      decay_rate: 0.5
      eps: 0.01
      batch_stat: True
    }
  }
  function {
    name: "Softmax"
    type: "Softmax"
    input: "BatchNormalization_4"
    output: "Softmax"
    softmax_param {
      axis: 1
    }
  }
  function {
    name: "CategoricalCrossEntropy"
    type: "CategoricalCrossEntropy"
    input: "Softmax"
    input: "CategoricalCrossEntropy_T"
    output: "CategoricalCrossEntropy"
    categorical_cross_entropy_param {
      axis: 1
    }
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
    name: "BinaryConnectConvolution/bicon_conv/W"
    type: "Parameter"
    shape: { dim: 64 dim: 1 dim: 5 dim: 5 }
    initializer {
      type: "UniformConvolutionGlorot"
      multiplier: 1
    }
  }
  variable {
    name: "BinaryConnectConvolution/bicon_conv/Wb"
    type: "Parameter"
    shape: { dim: 64 dim: 1 dim: 5 dim: 5 }
    initializer {
      type: "Constant"
      multiplier: 0
    }
  }
  variable {
    name: "BinaryConnectConvolution/bicon_conv/b"
    type: "Parameter"
    shape: { dim: 64 }
    initializer {
      type: "Constant"
      multiplier: 0
    }
  }
  variable {
    name: "BatchNormalization/bn/beta"
    type: "Parameter"
    shape: { dim: 1 dim: 64 dim: 1 dim: 1 }
    initializer {
      type: "Constant"
      multiplier: 0
    }
  }
  variable {
    name: "BatchNormalization/bn/gamma"
    type: "Parameter"
    shape: { dim: 1 dim: 64 dim: 1 dim: 1 }
    initializer {
      type: "Constant"
      multiplier: 1
    }
  }
  variable {
    name: "BatchNormalization/bn/mean"
    type: "Parameter"
    shape: { dim: 1 dim: 64 dim: 1 dim: 1 }
    initializer {
      type: "Constant"
      multiplier: 0
    }
  }
  variable {
    name: "BatchNormalization/bn/var"
    type: "Parameter"
    shape: { dim: 1 dim: 64 dim: 1 dim: 1 }
    initializer {
      type: "Constant"
      multiplier: 0
    }
  }
  variable {
    name: "BinaryConnectConvolution_2/bicon_conv/W"
    type: "Parameter"
    shape: { dim: 64 dim: 64 dim: 5 dim: 5 }
    initializer {
      type: "UniformConvolutionGlorot"
      multiplier: 1
    }
  }
  variable {
    name: "BinaryConnectConvolution_2/bicon_conv/Wb"
    type: "Parameter"
    shape: { dim: 64 dim: 64 dim: 5 dim: 5 }
    initializer {
      type: "Constant"
      multiplier: 0
    }
  }
  variable {
    name: "BinaryConnectConvolution_2/bicon_conv/b"
    type: "Parameter"
    shape: { dim: 64 }
    initializer {
      type: "Constant"
      multiplier: 0
    }
  }
  variable {
    name: "BatchNormalization_2/bn/beta"
    type: "Parameter"
    shape: { dim: 1 dim: 64 dim: 1 dim: 1 }
    initializer {
      type: "Constant"
      multiplier: 0
    }
  }
  variable {
    name: "BatchNormalization_2/bn/gamma"
    type: "Parameter"
    shape: { dim: 1 dim: 64 dim: 1 dim: 1 }
    initializer {
      type: "Constant"
      multiplier: 1
    }
  }
  variable {
    name: "BatchNormalization_2/bn/mean"
    type: "Parameter"
    shape: { dim: 1 dim: 64 dim: 1 dim: 1 }
    initializer {
      type: "Constant"
      multiplier: 0
    }
  }
  variable {
    name: "BatchNormalization_2/bn/var"
    type: "Parameter"
    shape: { dim: 1 dim: 64 dim: 1 dim: 1 }
    initializer {
      type: "Constant"
      multiplier: 0
    }
  }
  variable {
    name: "BinaryConnectAffine/bicon_affine/W"
    type: "Parameter"
    shape: { dim: 1024 dim: 512 }
    initializer {
      type: "UniformAffineGlorot"
      multiplier: 1
    }
  }
  variable {
    name: "BinaryConnectAffine/bicon_affine/Wb"
    type: "Parameter"
    shape: { dim: 1024 dim: 512 }
    initializer {
      type: "Constant"
      multiplier: 0
    }
  }
  variable {
    name: "BinaryConnectAffine/bicon_affine/b"
    type: "Parameter"
    shape: { dim: 512 }
    initializer {
      type: "Constant"
      multiplier: 0
    }
  }
  variable {
    name: "BatchNormalization_3/bn/beta"
    type: "Parameter"
    shape: { dim: 1 dim: 512 }
    initializer {
      type: "Constant"
      multiplier: 0
    }
  }
  variable {
    name: "BatchNormalization_3/bn/gamma"
    type: "Parameter"
    shape: { dim: 1 dim: 512 }
    initializer {
      type: "Constant"
      multiplier: 1
    }
  }
  variable {
    name: "BatchNormalization_3/bn/mean"
    type: "Parameter"
    shape: { dim: 1 dim: 512 }
    initializer {
      type: "Constant"
      multiplier: 0
    }
  }
  variable {
    name: "BatchNormalization_3/bn/var"
    type: "Parameter"
    shape: { dim: 1 dim: 512 }
    initializer {
      type: "Constant"
      multiplier: 0
    }
  }
  variable {
    name: "BinaryConnectAffine_2/bicon_affine/W"
    type: "Parameter"
    shape: { dim: 512 dim: 10 }
    initializer {
      type: "UniformAffineGlorot"
      multiplier: 1
    }
  }
  variable {
    name: "BinaryConnectAffine_2/bicon_affine/Wb"
    type: "Parameter"
    shape: { dim: 512 dim: 10 }
    initializer {
      type: "Constant"
      multiplier: 0
    }
  }
  variable {
    name: "BinaryConnectAffine_2/bicon_affine/b"
    type: "Parameter"
    shape: { dim: 10 }
    initializer {
      type: "Constant"
      multiplier: 0
    }
  }
  variable {
    name: "BatchNormalization_4/bn/beta"
    type: "Parameter"
    shape: { dim: 1 dim: 10 }
    initializer {
      type: "Constant"
      multiplier: 0
    }
  }
  variable {
    name: "BatchNormalization_4/bn/gamma"
    type: "Parameter"
    shape: { dim: 1 dim: 10 }
    initializer {
      type: "Constant"
      multiplier: 1
    }
  }
  variable {
    name: "BatchNormalization_4/bn/mean"
    type: "Parameter"
    shape: { dim: 1 dim: 10 }
    initializer {
      type: "Constant"
      multiplier: 0
    }
  }
  variable {
    name: "BatchNormalization_4/bn/var"
    type: "Parameter"
    shape: { dim: 1 dim: 10 }
    initializer {
      type: "Constant"
      multiplier: 0
    }
  }
  variable {
    name: "CategoricalCrossEntropy_T"
    type: "Buffer"
    shape: { dim:-1 dim: 1 }
  }
  variable {
    name: "BinaryConnectConvolution"
    type: "Buffer"
    shape: { dim:-1 dim: 64 dim: 24 dim: 24 }
  }
  variable {
    name: "MaxPooling"
    type: "Buffer"
    shape: { dim:-1 dim: 64 dim: 12 dim: 12 }
  }
  variable {
    name: "BatchNormalization"
    type: "Buffer"
    shape: { dim:-1 dim: 64 dim: 12 dim: 12 }
  }
  variable {
    name: "ReLU"
    type: "Buffer"
    shape: { dim:-1 dim: 64 dim: 12 dim: 12 }
  }
  variable {
    name: "BinaryConnectConvolution_2"
    type: "Buffer"
    shape: { dim:-1 dim: 64 dim: 8 dim: 8 }
  }
  variable {
    name: "MaxPooling_2"
    type: "Buffer"
    shape: { dim:-1 dim: 64 dim: 4 dim: 4 }
  }
  variable {
    name: "BatchNormalization_2"
    type: "Buffer"
    shape: { dim:-1 dim: 64 dim: 4 dim: 4 }
  }
  variable {
    name: "ReLU_3"
    type: "Buffer"
    shape: { dim:-1 dim: 64 dim: 4 dim: 4 }
  }
  variable {
    name: "BinaryConnectAffine"
    type: "Buffer"
    shape: { dim:-1 dim: 512 }
  }
  variable {
    name: "BatchNormalization_3"
    type: "Buffer"
    shape: { dim:-1 dim: 512 }
  }
  variable {
    name: "ReLU_2"
    type: "Buffer"
    shape: { dim:-1 dim: 512 }
  }
  variable {
    name: "BinaryConnectAffine_2"
    type: "Buffer"
    shape: { dim:-1 dim: 10 }
  }
  variable {
    name: "BatchNormalization_4"
    type: "Buffer"
    shape: { dim:-1 dim: 10 }
  }
  variable {
    name: "Softmax"
    type: "Buffer"
    shape: { dim:-1 dim: 10 }
  }
  variable {
    name: "CategoricalCrossEntropy"
    type: "Buffer"
    shape: { dim:-1 dim: 1 }
  }
  function {
    name: "BinaryConnectConvolution"
    type: "BinaryConnectConvolution"
    input: "Input"
    input: "BinaryConnectConvolution/bicon_conv/W"
    input: "BinaryConnectConvolution/bicon_conv/Wb"
    input: "BinaryConnectConvolution/bicon_conv/b"
    output: "BinaryConnectConvolution"
    binary_connect_convolution_param {
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
    input: "BinaryConnectConvolution"
    output: "MaxPooling"
    max_pooling_param {
      kernel: { dim: 2 dim: 2 }
      stride: { dim: 2 dim: 2 }
      ignore_border: true
      pad: { dim: 0 dim: 0 }
    }
  }
  function {
    name: "BatchNormalization"
    type: "BatchNormalization"
    input: "MaxPooling"
    input: "BatchNormalization/bn/beta"
    input: "BatchNormalization/bn/gamma"
    input: "BatchNormalization/bn/mean"
    input: "BatchNormalization/bn/var"
    output: "BatchNormalization"
    batch_normalization_param {
      axes: 1
      decay_rate: 0.5
      eps: 0.01
      batch_stat: False
    }
  }
  function {
    name: "ReLU"
    type: "ReLU"
    input: "BatchNormalization"
    output: "ReLU"
    relu_param {
      inplace: True
    }
  }
  function {
    name: "BinaryConnectConvolution_2"
    type: "BinaryConnectConvolution"
    input: "ReLU"
    input: "BinaryConnectConvolution_2/bicon_conv/W"
    input: "BinaryConnectConvolution_2/bicon_conv/Wb"
    input: "BinaryConnectConvolution_2/bicon_conv/b"
    output: "BinaryConnectConvolution_2"
    binary_connect_convolution_param {
      pad: { dim: 0 dim: 0 }
      stride: { dim: 1 dim: 1 }
      dilation: { dim: 1 dim: 1 }
      group: 1
      base_axis: 1
    }
  }
  function {
    name: "MaxPooling_2"
    type: "MaxPooling"
    input: "BinaryConnectConvolution_2"
    output: "MaxPooling_2"
    max_pooling_param {
      kernel: { dim: 2 dim: 2 }
      stride: { dim: 2 dim: 2 }
      ignore_border: true
      pad: { dim: 0 dim: 0 }
    }
  }
  function {
    name: "BatchNormalization_2"
    type: "BatchNormalization"
    input: "MaxPooling_2"
    input: "BatchNormalization_2/bn/beta"
    input: "BatchNormalization_2/bn/gamma"
    input: "BatchNormalization_2/bn/mean"
    input: "BatchNormalization_2/bn/var"
    output: "BatchNormalization_2"
    batch_normalization_param {
      axes: 1
      decay_rate: 0.5
      eps: 0.01
      batch_stat: False
    }
  }
  function {
    name: "ReLU_3"
    type: "ReLU"
    input: "BatchNormalization_2"
    output: "ReLU_3"
    relu_param {
      inplace: True
    }
  }
  function {
    name: "BinaryConnectAffine"
    type: "BinaryConnectAffine"
    input: "ReLU_3"
    input: "BinaryConnectAffine/bicon_affine/W"
    input: "BinaryConnectAffine/bicon_affine/Wb"
    input: "BinaryConnectAffine/bicon_affine/b"
    output: "BinaryConnectAffine"
    binary_connect_affine_param {
      base_axis: 1
    }
  }
  function {
    name: "BatchNormalization_3"
    type: "BatchNormalization"
    input: "BinaryConnectAffine"
    input: "BatchNormalization_3/bn/beta"
    input: "BatchNormalization_3/bn/gamma"
    input: "BatchNormalization_3/bn/mean"
    input: "BatchNormalization_3/bn/var"
    output: "BatchNormalization_3"
    batch_normalization_param {
      axes: 1
      decay_rate: 0.5
      eps: 0.01
      batch_stat: False
    }
  }
  function {
    name: "ReLU_2"
    type: "ReLU"
    input: "BatchNormalization_3"
    output: "ReLU_2"
    relu_param {
      inplace: True
    }
  }
  function {
    name: "BinaryConnectAffine_2"
    type: "BinaryConnectAffine"
    input: "ReLU_2"
    input: "BinaryConnectAffine_2/bicon_affine/W"
    input: "BinaryConnectAffine_2/bicon_affine/Wb"
    input: "BinaryConnectAffine_2/bicon_affine/b"
    output: "BinaryConnectAffine_2"
    binary_connect_affine_param {
      base_axis: 1
    }
  }
  function {
    name: "BatchNormalization_4"
    type: "BatchNormalization"
    input: "BinaryConnectAffine_2"
    input: "BatchNormalization_4/bn/beta"
    input: "BatchNormalization_4/bn/gamma"
    input: "BatchNormalization_4/bn/mean"
    input: "BatchNormalization_4/bn/var"
    output: "BatchNormalization_4"
    batch_normalization_param {
      axes: 1
      decay_rate: 0.5
      eps: 0.01
      batch_stat: False
    }
  }
  function {
    name: "Softmax"
    type: "Softmax"
    input: "BatchNormalization_4"
    output: "Softmax"
    softmax_param {
      axis: 1
    }
  }
  function {
    name: "CategoricalCrossEntropy"
    type: "CategoricalCrossEntropy"
    input: "Softmax"
    input: "CategoricalCrossEntropy_T"
    output: "CategoricalCrossEntropy"
    categorical_cross_entropy_param {
      axis: 1
    }
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
    name: "BinaryConnectConvolution/bicon_conv/W"
    type: "Parameter"
    shape: { dim: 64 dim: 1 dim: 5 dim: 5 }
    initializer {
      type: "UniformConvolutionGlorot"
      multiplier: 1
    }
  }
  variable {
    name: "BinaryConnectConvolution/bicon_conv/Wb"
    type: "Parameter"
    shape: { dim: 64 dim: 1 dim: 5 dim: 5 }
    initializer {
      type: "Constant"
      multiplier: 0
    }
  }
  variable {
    name: "BinaryConnectConvolution/bicon_conv/b"
    type: "Parameter"
    shape: { dim: 64 }
    initializer {
      type: "Constant"
      multiplier: 0
    }
  }
  variable {
    name: "BatchNormalization/bn/beta"
    type: "Parameter"
    shape: { dim: 1 dim: 64 dim: 1 dim: 1 }
    initializer {
      type: "Constant"
      multiplier: 0
    }
  }
  variable {
    name: "BatchNormalization/bn/gamma"
    type: "Parameter"
    shape: { dim: 1 dim: 64 dim: 1 dim: 1 }
    initializer {
      type: "Constant"
      multiplier: 1
    }
  }
  variable {
    name: "BatchNormalization/bn/mean"
    type: "Parameter"
    shape: { dim: 1 dim: 64 dim: 1 dim: 1 }
    initializer {
      type: "Constant"
      multiplier: 0
    }
  }
  variable {
    name: "BatchNormalization/bn/var"
    type: "Parameter"
    shape: { dim: 1 dim: 64 dim: 1 dim: 1 }
    initializer {
      type: "Constant"
      multiplier: 0
    }
  }
  variable {
    name: "BinaryConnectConvolution_2/bicon_conv/W"
    type: "Parameter"
    shape: { dim: 64 dim: 64 dim: 5 dim: 5 }
    initializer {
      type: "UniformConvolutionGlorot"
      multiplier: 1
    }
  }
  variable {
    name: "BinaryConnectConvolution_2/bicon_conv/Wb"
    type: "Parameter"
    shape: { dim: 64 dim: 64 dim: 5 dim: 5 }
    initializer {
      type: "Constant"
      multiplier: 0
    }
  }
  variable {
    name: "BinaryConnectConvolution_2/bicon_conv/b"
    type: "Parameter"
    shape: { dim: 64 }
    initializer {
      type: "Constant"
      multiplier: 0
    }
  }
  variable {
    name: "BatchNormalization_2/bn/beta"
    type: "Parameter"
    shape: { dim: 1 dim: 64 dim: 1 dim: 1 }
    initializer {
      type: "Constant"
      multiplier: 0
    }
  }
  variable {
    name: "BatchNormalization_2/bn/gamma"
    type: "Parameter"
    shape: { dim: 1 dim: 64 dim: 1 dim: 1 }
    initializer {
      type: "Constant"
      multiplier: 1
    }
  }
  variable {
    name: "BatchNormalization_2/bn/mean"
    type: "Parameter"
    shape: { dim: 1 dim: 64 dim: 1 dim: 1 }
    initializer {
      type: "Constant"
      multiplier: 0
    }
  }
  variable {
    name: "BatchNormalization_2/bn/var"
    type: "Parameter"
    shape: { dim: 1 dim: 64 dim: 1 dim: 1 }
    initializer {
      type: "Constant"
      multiplier: 0
    }
  }
  variable {
    name: "BinaryConnectAffine/bicon_affine/W"
    type: "Parameter"
    shape: { dim: 1024 dim: 512 }
    initializer {
      type: "UniformAffineGlorot"
      multiplier: 1
    }
  }
  variable {
    name: "BinaryConnectAffine/bicon_affine/Wb"
    type: "Parameter"
    shape: { dim: 1024 dim: 512 }
    initializer {
      type: "Constant"
      multiplier: 0
    }
  }
  variable {
    name: "BinaryConnectAffine/bicon_affine/b"
    type: "Parameter"
    shape: { dim: 512 }
    initializer {
      type: "Constant"
      multiplier: 0
    }
  }
  variable {
    name: "BatchNormalization_3/bn/beta"
    type: "Parameter"
    shape: { dim: 1 dim: 512 }
    initializer {
      type: "Constant"
      multiplier: 0
    }
  }
  variable {
    name: "BatchNormalization_3/bn/gamma"
    type: "Parameter"
    shape: { dim: 1 dim: 512 }
    initializer {
      type: "Constant"
      multiplier: 1
    }
  }
  variable {
    name: "BatchNormalization_3/bn/mean"
    type: "Parameter"
    shape: { dim: 1 dim: 512 }
    initializer {
      type: "Constant"
      multiplier: 0
    }
  }
  variable {
    name: "BatchNormalization_3/bn/var"
    type: "Parameter"
    shape: { dim: 1 dim: 512 }
    initializer {
      type: "Constant"
      multiplier: 0
    }
  }
  variable {
    name: "BinaryConnectAffine_2/bicon_affine/W"
    type: "Parameter"
    shape: { dim: 512 dim: 10 }
    initializer {
      type: "UniformAffineGlorot"
      multiplier: 1
    }
  }
  variable {
    name: "BinaryConnectAffine_2/bicon_affine/Wb"
    type: "Parameter"
    shape: { dim: 512 dim: 10 }
    initializer {
      type: "Constant"
      multiplier: 0
    }
  }
  variable {
    name: "BinaryConnectAffine_2/bicon_affine/b"
    type: "Parameter"
    shape: { dim: 10 }
    initializer {
      type: "Constant"
      multiplier: 0
    }
  }
  variable {
    name: "BatchNormalization_4/bn/beta"
    type: "Parameter"
    shape: { dim: 1 dim: 10 }
    initializer {
      type: "Constant"
      multiplier: 0
    }
  }
  variable {
    name: "BatchNormalization_4/bn/gamma"
    type: "Parameter"
    shape: { dim: 1 dim: 10 }
    initializer {
      type: "Constant"
      multiplier: 1
    }
  }
  variable {
    name: "BatchNormalization_4/bn/mean"
    type: "Parameter"
    shape: { dim: 1 dim: 10 }
    initializer {
      type: "Constant"
      multiplier: 0
    }
  }
  variable {
    name: "BatchNormalization_4/bn/var"
    type: "Parameter"
    shape: { dim: 1 dim: 10 }
    initializer {
      type: "Constant"
      multiplier: 0
    }
  }
  variable {
    name: "BinaryConnectConvolution"
    type: "Buffer"
    shape: { dim:-1 dim: 64 dim: 24 dim: 24 }
  }
  variable {
    name: "MaxPooling"
    type: "Buffer"
    shape: { dim:-1 dim: 64 dim: 12 dim: 12 }
  }
  variable {
    name: "BatchNormalization"
    type: "Buffer"
    shape: { dim:-1 dim: 64 dim: 12 dim: 12 }
  }
  variable {
    name: "ReLU"
    type: "Buffer"
    shape: { dim:-1 dim: 64 dim: 12 dim: 12 }
  }
  variable {
    name: "BinaryConnectConvolution_2"
    type: "Buffer"
    shape: { dim:-1 dim: 64 dim: 8 dim: 8 }
  }
  variable {
    name: "MaxPooling_2"
    type: "Buffer"
    shape: { dim:-1 dim: 64 dim: 4 dim: 4 }
  }
  variable {
    name: "BatchNormalization_2"
    type: "Buffer"
    shape: { dim:-1 dim: 64 dim: 4 dim: 4 }
  }
  variable {
    name: "ReLU_3"
    type: "Buffer"
    shape: { dim:-1 dim: 64 dim: 4 dim: 4 }
  }
  variable {
    name: "BinaryConnectAffine"
    type: "Buffer"
    shape: { dim:-1 dim: 512 }
  }
  variable {
    name: "BatchNormalization_3"
    type: "Buffer"
    shape: { dim:-1 dim: 512 }
  }
  variable {
    name: "ReLU_2"
    type: "Buffer"
    shape: { dim:-1 dim: 512 }
  }
  variable {
    name: "BinaryConnectAffine_2"
    type: "Buffer"
    shape: { dim:-1 dim: 10 }
  }
  variable {
    name: "BatchNormalization_4"
    type: "Buffer"
    shape: { dim:-1 dim: 10 }
  }
  variable {
    name: "Softmax"
    type: "Buffer"
    shape: { dim:-1 dim: 10 }
  }
  function {
    name: "BinaryConnectConvolution"
    type: "BinaryConnectConvolution"
    input: "Input"
    input: "BinaryConnectConvolution/bicon_conv/W"
    input: "BinaryConnectConvolution/bicon_conv/Wb"
    input: "BinaryConnectConvolution/bicon_conv/b"
    output: "BinaryConnectConvolution"
    binary_connect_convolution_param {
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
    input: "BinaryConnectConvolution"
    output: "MaxPooling"
    max_pooling_param {
      kernel: { dim: 2 dim: 2 }
      stride: { dim: 2 dim: 2 }
      ignore_border: true
      pad: { dim: 0 dim: 0 }
    }
  }
  function {
    name: "BatchNormalization"
    type: "BatchNormalization"
    input: "MaxPooling"
    input: "BatchNormalization/bn/beta"
    input: "BatchNormalization/bn/gamma"
    input: "BatchNormalization/bn/mean"
    input: "BatchNormalization/bn/var"
    output: "BatchNormalization"
    batch_normalization_param {
      axes: 1
      decay_rate: 0.5
      eps: 0.01
      batch_stat: False
    }
  }
  function {
    name: "ReLU"
    type: "ReLU"
    input: "BatchNormalization"
    output: "ReLU"
    relu_param {
      inplace: True
    }
  }
  function {
    name: "BinaryConnectConvolution_2"
    type: "BinaryConnectConvolution"
    input: "ReLU"
    input: "BinaryConnectConvolution_2/bicon_conv/W"
    input: "BinaryConnectConvolution_2/bicon_conv/Wb"
    input: "BinaryConnectConvolution_2/bicon_conv/b"
    output: "BinaryConnectConvolution_2"
    binary_connect_convolution_param {
      pad: { dim: 0 dim: 0 }
      stride: { dim: 1 dim: 1 }
      dilation: { dim: 1 dim: 1 }
      group: 1
      base_axis: 1
    }
  }
  function {
    name: "MaxPooling_2"
    type: "MaxPooling"
    input: "BinaryConnectConvolution_2"
    output: "MaxPooling_2"
    max_pooling_param {
      kernel: { dim: 2 dim: 2 }
      stride: { dim: 2 dim: 2 }
      ignore_border: true
      pad: { dim: 0 dim: 0 }
    }
  }
  function {
    name: "BatchNormalization_2"
    type: "BatchNormalization"
    input: "MaxPooling_2"
    input: "BatchNormalization_2/bn/beta"
    input: "BatchNormalization_2/bn/gamma"
    input: "BatchNormalization_2/bn/mean"
    input: "BatchNormalization_2/bn/var"
    output: "BatchNormalization_2"
    batch_normalization_param {
      axes: 1
      decay_rate: 0.5
      eps: 0.01
      batch_stat: False
    }
  }
  function {
    name: "ReLU_3"
    type: "ReLU"
    input: "BatchNormalization_2"
    output: "ReLU_3"
    relu_param {
      inplace: True
    }
  }
  function {
    name: "BinaryConnectAffine"
    type: "BinaryConnectAffine"
    input: "ReLU_3"
    input: "BinaryConnectAffine/bicon_affine/W"
    input: "BinaryConnectAffine/bicon_affine/Wb"
    input: "BinaryConnectAffine/bicon_affine/b"
    output: "BinaryConnectAffine"
    binary_connect_affine_param {
      base_axis: 1
    }
  }
  function {
    name: "BatchNormalization_3"
    type: "BatchNormalization"
    input: "BinaryConnectAffine"
    input: "BatchNormalization_3/bn/beta"
    input: "BatchNormalization_3/bn/gamma"
    input: "BatchNormalization_3/bn/mean"
    input: "BatchNormalization_3/bn/var"
    output: "BatchNormalization_3"
    batch_normalization_param {
      axes: 1
      decay_rate: 0.5
      eps: 0.01
      batch_stat: False
    }
  }
  function {
    name: "ReLU_2"
    type: "ReLU"
    input: "BatchNormalization_3"
    output: "ReLU_2"
    relu_param {
      inplace: True
    }
  }
  function {
    name: "BinaryConnectAffine_2"
    type: "BinaryConnectAffine"
    input: "ReLU_2"
    input: "BinaryConnectAffine_2/bicon_affine/W"
    input: "BinaryConnectAffine_2/bicon_affine/Wb"
    input: "BinaryConnectAffine_2/bicon_affine/b"
    output: "BinaryConnectAffine_2"
    binary_connect_affine_param {
      base_axis: 1
    }
  }
  function {
    name: "BatchNormalization_4"
    type: "BatchNormalization"
    input: "BinaryConnectAffine_2"
    input: "BatchNormalization_4/bn/beta"
    input: "BatchNormalization_4/bn/gamma"
    input: "BatchNormalization_4/bn/mean"
    input: "BatchNormalization_4/bn/var"
    output: "BatchNormalization_4"
    batch_normalization_param {
      axes: 1
      decay_rate: 0.5
      eps: 0.01
      batch_stat: False
    }
  }
  function {
    name: "Softmax"
    type: "Softmax"
    input: "BatchNormalization_4"
    output: "Softmax"
    softmax_param {
      axis: 1
    }
  }
}
dataset {
  name: "Training"
  uri: "C:\\Utils\\neural_network_console_internal_v120\\samples\\sample_dataset\\mnist\\mnist_training.csv"
  cache_dir: "C:\\Utils\\neural_network_console_internal_v120\\samples\\sample_dataset\\mnist\\mnist_training.cache"
  overwrite_cache: True
  create_cache_explicitly: True
  shuffle: true
  no_image_normalization: False
  batch_size: 64
}
dataset {
  name: "Validation"
  uri: "C:\\Utils\\neural_network_console_internal_v120\\samples\\sample_dataset\\mnist\\mnist_test.csv"
  cache_dir: "C:\\Utils\\neural_network_console_internal_v120\\samples\\sample_dataset\\mnist\\mnist_test.cache"
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
      alpha: 0.0001
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
    variable_name: "CategoricalCrossEntropy_T"
    data_name: "y"
  }
  loss_variable {
    variable_name: "CategoricalCrossEntropy"
  }
  parameter_variable {
    variable_name: "BinaryConnectConvolution/bicon_conv/W"
    learning_rate_multiplier: 1
  }
  parameter_variable {
    variable_name: "BinaryConnectConvolution/bicon_conv/Wb"
    learning_rate_multiplier: 0
  }
  parameter_variable {
    variable_name: "BinaryConnectConvolution/bicon_conv/b"
    learning_rate_multiplier: 1
  }
  parameter_variable {
    variable_name: "BatchNormalization/bn/beta"
    learning_rate_multiplier: 1
  }
  parameter_variable {
    variable_name: "BatchNormalization/bn/gamma"
    learning_rate_multiplier: 1
  }
  parameter_variable {
    variable_name: "BatchNormalization/bn/mean"
    learning_rate_multiplier: 0
  }
  parameter_variable {
    variable_name: "BatchNormalization/bn/var"
    learning_rate_multiplier: 0
  }
  parameter_variable {
    variable_name: "BinaryConnectConvolution_2/bicon_conv/W"
    learning_rate_multiplier: 1
  }
  parameter_variable {
    variable_name: "BinaryConnectConvolution_2/bicon_conv/Wb"
    learning_rate_multiplier: 0
  }
  parameter_variable {
    variable_name: "BinaryConnectConvolution_2/bicon_conv/b"
    learning_rate_multiplier: 1
  }
  parameter_variable {
    variable_name: "BatchNormalization_2/bn/beta"
    learning_rate_multiplier: 1
  }
  parameter_variable {
    variable_name: "BatchNormalization_2/bn/gamma"
    learning_rate_multiplier: 1
  }
  parameter_variable {
    variable_name: "BatchNormalization_2/bn/mean"
    learning_rate_multiplier: 0
  }
  parameter_variable {
    variable_name: "BatchNormalization_2/bn/var"
    learning_rate_multiplier: 0
  }
  parameter_variable {
    variable_name: "BinaryConnectAffine/bicon_affine/W"
    learning_rate_multiplier: 1
  }
  parameter_variable {
    variable_name: "BinaryConnectAffine/bicon_affine/Wb"
    learning_rate_multiplier: 0
  }
  parameter_variable {
    variable_name: "BinaryConnectAffine/bicon_affine/b"
    learning_rate_multiplier: 1
  }
  parameter_variable {
    variable_name: "BatchNormalization_3/bn/beta"
    learning_rate_multiplier: 1
  }
  parameter_variable {
    variable_name: "BatchNormalization_3/bn/gamma"
    learning_rate_multiplier: 1
  }
  parameter_variable {
    variable_name: "BatchNormalization_3/bn/mean"
    learning_rate_multiplier: 0
  }
  parameter_variable {
    variable_name: "BatchNormalization_3/bn/var"
    learning_rate_multiplier: 0
  }
  parameter_variable {
    variable_name: "BinaryConnectAffine_2/bicon_affine/W"
    learning_rate_multiplier: 1
  }
  parameter_variable {
    variable_name: "BinaryConnectAffine_2/bicon_affine/Wb"
    learning_rate_multiplier: 0
  }
  parameter_variable {
    variable_name: "BinaryConnectAffine_2/bicon_affine/b"
    learning_rate_multiplier: 1
  }
  parameter_variable {
    variable_name: "BatchNormalization_4/bn/beta"
    learning_rate_multiplier: 1
  }
  parameter_variable {
    variable_name: "BatchNormalization_4/bn/gamma"
    learning_rate_multiplier: 1
  }
  parameter_variable {
    variable_name: "BatchNormalization_4/bn/mean"
    learning_rate_multiplier: 0
  }
  parameter_variable {
    variable_name: "BatchNormalization_4/bn/var"
    learning_rate_multiplier: 0
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
    variable_name: "CategoricalCrossEntropy_T"
    data_name: "y"
  }
  monitor_variable {
    type: "Error"
    variable_name: "CategoricalCrossEntropy"
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
    variable_name: "CategoricalCrossEntropy_T"
    data_name: "y"
  }
  monitor_variable {
    type: "Error"
    variable_name: "CategoricalCrossEntropy"
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
    variable_name: "Softmax"
    data_name: "y'"
  }
  parameter_variable {
    variable_name: "BinaryConnectConvolution/bicon_conv/W"
  }
  parameter_variable {
    variable_name: "BinaryConnectConvolution/bicon_conv/Wb"
  }
  parameter_variable {
    variable_name: "BinaryConnectConvolution/bicon_conv/b"
  }
  parameter_variable {
    variable_name: "BatchNormalization/bn/beta"
  }
  parameter_variable {
    variable_name: "BatchNormalization/bn/gamma"
  }
  parameter_variable {
    variable_name: "BatchNormalization/bn/mean"
  }
  parameter_variable {
    variable_name: "BatchNormalization/bn/var"
  }
  parameter_variable {
    variable_name: "BinaryConnectConvolution_2/bicon_conv/W"
  }
  parameter_variable {
    variable_name: "BinaryConnectConvolution_2/bicon_conv/Wb"
  }
  parameter_variable {
    variable_name: "BinaryConnectConvolution_2/bicon_conv/b"
  }
  parameter_variable {
    variable_name: "BatchNormalization_2/bn/beta"
  }
  parameter_variable {
    variable_name: "BatchNormalization_2/bn/gamma"
  }
  parameter_variable {
    variable_name: "BatchNormalization_2/bn/mean"
  }
  parameter_variable {
    variable_name: "BatchNormalization_2/bn/var"
  }
  parameter_variable {
    variable_name: "BinaryConnectAffine/bicon_affine/W"
  }
  parameter_variable {
    variable_name: "BinaryConnectAffine/bicon_affine/Wb"
  }
  parameter_variable {
    variable_name: "BinaryConnectAffine/bicon_affine/b"
  }
  parameter_variable {
    variable_name: "BatchNormalization_3/bn/beta"
  }
  parameter_variable {
    variable_name: "BatchNormalization_3/bn/gamma"
  }
  parameter_variable {
    variable_name: "BatchNormalization_3/bn/mean"
  }
  parameter_variable {
    variable_name: "BatchNormalization_3/bn/var"
  }
  parameter_variable {
    variable_name: "BinaryConnectAffine_2/bicon_affine/W"
  }
  parameter_variable {
    variable_name: "BinaryConnectAffine_2/bicon_affine/Wb"
  }
  parameter_variable {
    variable_name: "BinaryConnectAffine_2/bicon_affine/b"
  }
  parameter_variable {
    variable_name: "BatchNormalization_4/bn/beta"
  }
  parameter_variable {
    variable_name: "BatchNormalization_4/bn/gamma"
  }
  parameter_variable {
    variable_name: "BatchNormalization_4/bn/mean"
  }
  parameter_variable {
    variable_name: "BatchNormalization_4/bn/var"
  }
}
'''
# The nntxt is generated from /home/woody/aid/nnabla-sample-data/sample_project/tutorial/binary_networks/binary_connect_mnist_MLP.files/20180723_144304/net.nntxt.
N00000017 = r'''global_config {
  default_context {
    array_class: "CudaCachedArray"
    device_id: "0"
    backends: "cudnn:float"
    backends: "cuda:float"
    backends: "cpu:float"
  }
}
training_config {
  max_epoch: 50
  iter_per_epoch: 937
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
    name: "BinaryConnectAffine/bicon_affine/W"
    type: "Parameter"
    shape: { dim: 784 dim: 2048 }
    initializer {
      type: "UniformAffineGlorot"
      multiplier: 1
    }
  }
  variable {
    name: "BinaryConnectAffine/bicon_affine/Wb"
    type: "Parameter"
    shape: { dim: 784 dim: 2048 }
    initializer {
      type: "Constant"
      multiplier: 0
    }
  }
  variable {
    name: "BinaryConnectAffine/bicon_affine/b"
    type: "Parameter"
    shape: { dim: 2048 }
    initializer {
      type: "Constant"
      multiplier: 0
    }
  }
  variable {
    name: "BatchNormalization/bn/beta"
    type: "Parameter"
    shape: { dim: 1 dim: 2048 }
    initializer {
      type: "Constant"
      multiplier: 0
    }
  }
  variable {
    name: "BatchNormalization/bn/gamma"
    type: "Parameter"
    shape: { dim: 1 dim: 2048 }
    initializer {
      type: "Constant"
      multiplier: 1
    }
  }
  variable {
    name: "BatchNormalization/bn/mean"
    type: "Parameter"
    shape: { dim: 1 dim: 2048 }
    initializer {
      type: "Constant"
      multiplier: 0
    }
  }
  variable {
    name: "BatchNormalization/bn/var"
    type: "Parameter"
    shape: { dim: 1 dim: 2048 }
    initializer {
      type: "Constant"
      multiplier: 0
    }
  }
  variable {
    name: "BinaryConnectAffine_2/bicon_affine/W"
    type: "Parameter"
    shape: { dim: 2048 dim: 2048 }
    initializer {
      type: "UniformAffineGlorot"
      multiplier: 1
    }
  }
  variable {
    name: "BinaryConnectAffine_2/bicon_affine/Wb"
    type: "Parameter"
    shape: { dim: 2048 dim: 2048 }
    initializer {
      type: "Constant"
      multiplier: 0
    }
  }
  variable {
    name: "BinaryConnectAffine_2/bicon_affine/b"
    type: "Parameter"
    shape: { dim: 2048 }
    initializer {
      type: "Constant"
      multiplier: 0
    }
  }
  variable {
    name: "BatchNormalization_2/bn/beta"
    type: "Parameter"
    shape: { dim: 1 dim: 2048 }
    initializer {
      type: "Constant"
      multiplier: 0
    }
  }
  variable {
    name: "BatchNormalization_2/bn/gamma"
    type: "Parameter"
    shape: { dim: 1 dim: 2048 }
    initializer {
      type: "Constant"
      multiplier: 1
    }
  }
  variable {
    name: "BatchNormalization_2/bn/mean"
    type: "Parameter"
    shape: { dim: 1 dim: 2048 }
    initializer {
      type: "Constant"
      multiplier: 0
    }
  }
  variable {
    name: "BatchNormalization_2/bn/var"
    type: "Parameter"
    shape: { dim: 1 dim: 2048 }
    initializer {
      type: "Constant"
      multiplier: 0
    }
  }
  variable {
    name: "BinaryConnectAffine_3/bicon_affine/W"
    type: "Parameter"
    shape: { dim: 2048 dim: 2048 }
    initializer {
      type: "UniformAffineGlorot"
      multiplier: 1
    }
  }
  variable {
    name: "BinaryConnectAffine_3/bicon_affine/Wb"
    type: "Parameter"
    shape: { dim: 2048 dim: 2048 }
    initializer {
      type: "Constant"
      multiplier: 0
    }
  }
  variable {
    name: "BinaryConnectAffine_3/bicon_affine/b"
    type: "Parameter"
    shape: { dim: 2048 }
    initializer {
      type: "Constant"
      multiplier: 0
    }
  }
  variable {
    name: "BatchNormalization_3/bn/beta"
    type: "Parameter"
    shape: { dim: 1 dim: 2048 }
    initializer {
      type: "Constant"
      multiplier: 0
    }
  }
  variable {
    name: "BatchNormalization_3/bn/gamma"
    type: "Parameter"
    shape: { dim: 1 dim: 2048 }
    initializer {
      type: "Constant"
      multiplier: 1
    }
  }
  variable {
    name: "BatchNormalization_3/bn/mean"
    type: "Parameter"
    shape: { dim: 1 dim: 2048 }
    initializer {
      type: "Constant"
      multiplier: 0
    }
  }
  variable {
    name: "BatchNormalization_3/bn/var"
    type: "Parameter"
    shape: { dim: 1 dim: 2048 }
    initializer {
      type: "Constant"
      multiplier: 0
    }
  }
  variable {
    name: "BinaryConnectAffine_4/bicon_affine/W"
    type: "Parameter"
    shape: { dim: 2048 dim: 10 }
    initializer {
      type: "UniformAffineGlorot"
      multiplier: 1
    }
  }
  variable {
    name: "BinaryConnectAffine_4/bicon_affine/Wb"
    type: "Parameter"
    shape: { dim: 2048 dim: 10 }
    initializer {
      type: "Constant"
      multiplier: 0
    }
  }
  variable {
    name: "BinaryConnectAffine_4/bicon_affine/b"
    type: "Parameter"
    shape: { dim: 10 }
    initializer {
      type: "Constant"
      multiplier: 0
    }
  }
  variable {
    name: "BatchNormalization_4/bn/beta"
    type: "Parameter"
    shape: { dim: 1 dim: 10 }
    initializer {
      type: "Constant"
      multiplier: 0
    }
  }
  variable {
    name: "BatchNormalization_4/bn/gamma"
    type: "Parameter"
    shape: { dim: 1 dim: 10 }
    initializer {
      type: "Constant"
      multiplier: 1
    }
  }
  variable {
    name: "BatchNormalization_4/bn/mean"
    type: "Parameter"
    shape: { dim: 1 dim: 10 }
    initializer {
      type: "Constant"
      multiplier: 0
    }
  }
  variable {
    name: "BatchNormalization_4/bn/var"
    type: "Parameter"
    shape: { dim: 1 dim: 10 }
    initializer {
      type: "Constant"
      multiplier: 0
    }
  }
  variable {
    name: "CategoricalCrossEntropy_T"
    type: "Buffer"
    shape: { dim:-1 dim: 1 }
  }
  variable {
    name: "BinaryConnectAffine"
    type: "Buffer"
    shape: { dim:-1 dim: 2048 }
  }
  variable {
    name: "BatchNormalization"
    type: "Buffer"
    shape: { dim:-1 dim: 2048 }
  }
  variable {
    name: "ReLU"
    type: "Buffer"
    shape: { dim:-1 dim: 2048 }
  }
  variable {
    name: "BinaryConnectAffine_2"
    type: "Buffer"
    shape: { dim:-1 dim: 2048 }
  }
  variable {
    name: "BatchNormalization_2"
    type: "Buffer"
    shape: { dim:-1 dim: 2048 }
  }
  variable {
    name: "ReLU_2"
    type: "Buffer"
    shape: { dim:-1 dim: 2048 }
  }
  variable {
    name: "BinaryConnectAffine_3"
    type: "Buffer"
    shape: { dim:-1 dim: 2048 }
  }
  variable {
    name: "BatchNormalization_3"
    type: "Buffer"
    shape: { dim:-1 dim: 2048 }
  }
  variable {
    name: "ReLU_3"
    type: "Buffer"
    shape: { dim:-1 dim: 2048 }
  }
  variable {
    name: "BinaryConnectAffine_4"
    type: "Buffer"
    shape: { dim:-1 dim: 10 }
  }
  variable {
    name: "BatchNormalization_4"
    type: "Buffer"
    shape: { dim:-1 dim: 10 }
  }
  variable {
    name: "Softmax"
    type: "Buffer"
    shape: { dim:-1 dim: 10 }
  }
  variable {
    name: "CategoricalCrossEntropy"
    type: "Buffer"
    shape: { dim:-1 dim: 1 }
  }
  function {
    name: "BinaryConnectAffine"
    type: "BinaryConnectAffine"
    input: "Input"
    input: "BinaryConnectAffine/bicon_affine/W"
    input: "BinaryConnectAffine/bicon_affine/Wb"
    input: "BinaryConnectAffine/bicon_affine/b"
    output: "BinaryConnectAffine"
    binary_connect_affine_param {
      base_axis: 1
    }
  }
  function {
    name: "BatchNormalization"
    type: "BatchNormalization"
    input: "BinaryConnectAffine"
    input: "BatchNormalization/bn/beta"
    input: "BatchNormalization/bn/gamma"
    input: "BatchNormalization/bn/mean"
    input: "BatchNormalization/bn/var"
    output: "BatchNormalization"
    batch_normalization_param {
      axes: 1
      decay_rate: 0.5
      eps: 0.01
      batch_stat: True
    }
  }
  function {
    name: "ReLU"
    type: "ReLU"
    input: "BatchNormalization"
    output: "ReLU"
    relu_param {
      inplace: True
    }
  }
  function {
    name: "BinaryConnectAffine_2"
    type: "BinaryConnectAffine"
    input: "ReLU"
    input: "BinaryConnectAffine_2/bicon_affine/W"
    input: "BinaryConnectAffine_2/bicon_affine/Wb"
    input: "BinaryConnectAffine_2/bicon_affine/b"
    output: "BinaryConnectAffine_2"
    binary_connect_affine_param {
      base_axis: 1
    }
  }
  function {
    name: "BatchNormalization_2"
    type: "BatchNormalization"
    input: "BinaryConnectAffine_2"
    input: "BatchNormalization_2/bn/beta"
    input: "BatchNormalization_2/bn/gamma"
    input: "BatchNormalization_2/bn/mean"
    input: "BatchNormalization_2/bn/var"
    output: "BatchNormalization_2"
    batch_normalization_param {
      axes: 1
      decay_rate: 0.5
      eps: 0.01
      batch_stat: True
    }
  }
  function {
    name: "ReLU_2"
    type: "ReLU"
    input: "BatchNormalization_2"
    output: "ReLU_2"
    relu_param {
      inplace: True
    }
  }
  function {
    name: "BinaryConnectAffine_3"
    type: "BinaryConnectAffine"
    input: "ReLU_2"
    input: "BinaryConnectAffine_3/bicon_affine/W"
    input: "BinaryConnectAffine_3/bicon_affine/Wb"
    input: "BinaryConnectAffine_3/bicon_affine/b"
    output: "BinaryConnectAffine_3"
    binary_connect_affine_param {
      base_axis: 1
    }
  }
  function {
    name: "BatchNormalization_3"
    type: "BatchNormalization"
    input: "BinaryConnectAffine_3"
    input: "BatchNormalization_3/bn/beta"
    input: "BatchNormalization_3/bn/gamma"
    input: "BatchNormalization_3/bn/mean"
    input: "BatchNormalization_3/bn/var"
    output: "BatchNormalization_3"
    batch_normalization_param {
      axes: 1
      decay_rate: 0.5
      eps: 0.01
      batch_stat: True
    }
  }
  function {
    name: "ReLU_3"
    type: "ReLU"
    input: "BatchNormalization_3"
    output: "ReLU_3"
    relu_param {
      inplace: True
    }
  }
  function {
    name: "BinaryConnectAffine_4"
    type: "BinaryConnectAffine"
    input: "ReLU_3"
    input: "BinaryConnectAffine_4/bicon_affine/W"
    input: "BinaryConnectAffine_4/bicon_affine/Wb"
    input: "BinaryConnectAffine_4/bicon_affine/b"
    output: "BinaryConnectAffine_4"
    binary_connect_affine_param {
      base_axis: 1
    }
  }
  function {
    name: "BatchNormalization_4"
    type: "BatchNormalization"
    input: "BinaryConnectAffine_4"
    input: "BatchNormalization_4/bn/beta"
    input: "BatchNormalization_4/bn/gamma"
    input: "BatchNormalization_4/bn/mean"
    input: "BatchNormalization_4/bn/var"
    output: "BatchNormalization_4"
    batch_normalization_param {
      axes: 1
      decay_rate: 0.5
      eps: 0.01
      batch_stat: True
    }
  }
  function {
    name: "Softmax"
    type: "Softmax"
    input: "BatchNormalization_4"
    output: "Softmax"
    softmax_param {
      axis: 1
    }
  }
  function {
    name: "CategoricalCrossEntropy"
    type: "CategoricalCrossEntropy"
    input: "Softmax"
    input: "CategoricalCrossEntropy_T"
    output: "CategoricalCrossEntropy"
    categorical_cross_entropy_param {
      axis: 1
    }
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
    name: "BinaryConnectAffine/bicon_affine/W"
    type: "Parameter"
    shape: { dim: 784 dim: 2048 }
    initializer {
      type: "UniformAffineGlorot"
      multiplier: 1
    }
  }
  variable {
    name: "BinaryConnectAffine/bicon_affine/Wb"
    type: "Parameter"
    shape: { dim: 784 dim: 2048 }
    initializer {
      type: "Constant"
      multiplier: 0
    }
  }
  variable {
    name: "BinaryConnectAffine/bicon_affine/b"
    type: "Parameter"
    shape: { dim: 2048 }
    initializer {
      type: "Constant"
      multiplier: 0
    }
  }
  variable {
    name: "BatchNormalization/bn/beta"
    type: "Parameter"
    shape: { dim: 1 dim: 2048 }
    initializer {
      type: "Constant"
      multiplier: 0
    }
  }
  variable {
    name: "BatchNormalization/bn/gamma"
    type: "Parameter"
    shape: { dim: 1 dim: 2048 }
    initializer {
      type: "Constant"
      multiplier: 1
    }
  }
  variable {
    name: "BatchNormalization/bn/mean"
    type: "Parameter"
    shape: { dim: 1 dim: 2048 }
    initializer {
      type: "Constant"
      multiplier: 0
    }
  }
  variable {
    name: "BatchNormalization/bn/var"
    type: "Parameter"
    shape: { dim: 1 dim: 2048 }
    initializer {
      type: "Constant"
      multiplier: 0
    }
  }
  variable {
    name: "BinaryConnectAffine_2/bicon_affine/W"
    type: "Parameter"
    shape: { dim: 2048 dim: 2048 }
    initializer {
      type: "UniformAffineGlorot"
      multiplier: 1
    }
  }
  variable {
    name: "BinaryConnectAffine_2/bicon_affine/Wb"
    type: "Parameter"
    shape: { dim: 2048 dim: 2048 }
    initializer {
      type: "Constant"
      multiplier: 0
    }
  }
  variable {
    name: "BinaryConnectAffine_2/bicon_affine/b"
    type: "Parameter"
    shape: { dim: 2048 }
    initializer {
      type: "Constant"
      multiplier: 0
    }
  }
  variable {
    name: "BatchNormalization_2/bn/beta"
    type: "Parameter"
    shape: { dim: 1 dim: 2048 }
    initializer {
      type: "Constant"
      multiplier: 0
    }
  }
  variable {
    name: "BatchNormalization_2/bn/gamma"
    type: "Parameter"
    shape: { dim: 1 dim: 2048 }
    initializer {
      type: "Constant"
      multiplier: 1
    }
  }
  variable {
    name: "BatchNormalization_2/bn/mean"
    type: "Parameter"
    shape: { dim: 1 dim: 2048 }
    initializer {
      type: "Constant"
      multiplier: 0
    }
  }
  variable {
    name: "BatchNormalization_2/bn/var"
    type: "Parameter"
    shape: { dim: 1 dim: 2048 }
    initializer {
      type: "Constant"
      multiplier: 0
    }
  }
  variable {
    name: "BinaryConnectAffine_3/bicon_affine/W"
    type: "Parameter"
    shape: { dim: 2048 dim: 2048 }
    initializer {
      type: "UniformAffineGlorot"
      multiplier: 1
    }
  }
  variable {
    name: "BinaryConnectAffine_3/bicon_affine/Wb"
    type: "Parameter"
    shape: { dim: 2048 dim: 2048 }
    initializer {
      type: "Constant"
      multiplier: 0
    }
  }
  variable {
    name: "BinaryConnectAffine_3/bicon_affine/b"
    type: "Parameter"
    shape: { dim: 2048 }
    initializer {
      type: "Constant"
      multiplier: 0
    }
  }
  variable {
    name: "BatchNormalization_3/bn/beta"
    type: "Parameter"
    shape: { dim: 1 dim: 2048 }
    initializer {
      type: "Constant"
      multiplier: 0
    }
  }
  variable {
    name: "BatchNormalization_3/bn/gamma"
    type: "Parameter"
    shape: { dim: 1 dim: 2048 }
    initializer {
      type: "Constant"
      multiplier: 1
    }
  }
  variable {
    name: "BatchNormalization_3/bn/mean"
    type: "Parameter"
    shape: { dim: 1 dim: 2048 }
    initializer {
      type: "Constant"
      multiplier: 0
    }
  }
  variable {
    name: "BatchNormalization_3/bn/var"
    type: "Parameter"
    shape: { dim: 1 dim: 2048 }
    initializer {
      type: "Constant"
      multiplier: 0
    }
  }
  variable {
    name: "BinaryConnectAffine_4/bicon_affine/W"
    type: "Parameter"
    shape: { dim: 2048 dim: 10 }
    initializer {
      type: "UniformAffineGlorot"
      multiplier: 1
    }
  }
  variable {
    name: "BinaryConnectAffine_4/bicon_affine/Wb"
    type: "Parameter"
    shape: { dim: 2048 dim: 10 }
    initializer {
      type: "Constant"
      multiplier: 0
    }
  }
  variable {
    name: "BinaryConnectAffine_4/bicon_affine/b"
    type: "Parameter"
    shape: { dim: 10 }
    initializer {
      type: "Constant"
      multiplier: 0
    }
  }
  variable {
    name: "BatchNormalization_4/bn/beta"
    type: "Parameter"
    shape: { dim: 1 dim: 10 }
    initializer {
      type: "Constant"
      multiplier: 0
    }
  }
  variable {
    name: "BatchNormalization_4/bn/gamma"
    type: "Parameter"
    shape: { dim: 1 dim: 10 }
    initializer {
      type: "Constant"
      multiplier: 1
    }
  }
  variable {
    name: "BatchNormalization_4/bn/mean"
    type: "Parameter"
    shape: { dim: 1 dim: 10 }
    initializer {
      type: "Constant"
      multiplier: 0
    }
  }
  variable {
    name: "BatchNormalization_4/bn/var"
    type: "Parameter"
    shape: { dim: 1 dim: 10 }
    initializer {
      type: "Constant"
      multiplier: 0
    }
  }
  variable {
    name: "CategoricalCrossEntropy_T"
    type: "Buffer"
    shape: { dim:-1 dim: 1 }
  }
  variable {
    name: "BinaryConnectAffine"
    type: "Buffer"
    shape: { dim:-1 dim: 2048 }
  }
  variable {
    name: "BatchNormalization"
    type: "Buffer"
    shape: { dim:-1 dim: 2048 }
  }
  variable {
    name: "ReLU"
    type: "Buffer"
    shape: { dim:-1 dim: 2048 }
  }
  variable {
    name: "BinaryConnectAffine_2"
    type: "Buffer"
    shape: { dim:-1 dim: 2048 }
  }
  variable {
    name: "BatchNormalization_2"
    type: "Buffer"
    shape: { dim:-1 dim: 2048 }
  }
  variable {
    name: "ReLU_2"
    type: "Buffer"
    shape: { dim:-1 dim: 2048 }
  }
  variable {
    name: "BinaryConnectAffine_3"
    type: "Buffer"
    shape: { dim:-1 dim: 2048 }
  }
  variable {
    name: "BatchNormalization_3"
    type: "Buffer"
    shape: { dim:-1 dim: 2048 }
  }
  variable {
    name: "ReLU_3"
    type: "Buffer"
    shape: { dim:-1 dim: 2048 }
  }
  variable {
    name: "BinaryConnectAffine_4"
    type: "Buffer"
    shape: { dim:-1 dim: 10 }
  }
  variable {
    name: "BatchNormalization_4"
    type: "Buffer"
    shape: { dim:-1 dim: 10 }
  }
  variable {
    name: "Softmax"
    type: "Buffer"
    shape: { dim:-1 dim: 10 }
  }
  variable {
    name: "CategoricalCrossEntropy"
    type: "Buffer"
    shape: { dim:-1 dim: 1 }
  }
  function {
    name: "BinaryConnectAffine"
    type: "BinaryConnectAffine"
    input: "Input"
    input: "BinaryConnectAffine/bicon_affine/W"
    input: "BinaryConnectAffine/bicon_affine/Wb"
    input: "BinaryConnectAffine/bicon_affine/b"
    output: "BinaryConnectAffine"
    binary_connect_affine_param {
      base_axis: 1
    }
  }
  function {
    name: "BatchNormalization"
    type: "BatchNormalization"
    input: "BinaryConnectAffine"
    input: "BatchNormalization/bn/beta"
    input: "BatchNormalization/bn/gamma"
    input: "BatchNormalization/bn/mean"
    input: "BatchNormalization/bn/var"
    output: "BatchNormalization"
    batch_normalization_param {
      axes: 1
      decay_rate: 0.5
      eps: 0.01
      batch_stat: False
    }
  }
  function {
    name: "ReLU"
    type: "ReLU"
    input: "BatchNormalization"
    output: "ReLU"
    relu_param {
      inplace: True
    }
  }
  function {
    name: "BinaryConnectAffine_2"
    type: "BinaryConnectAffine"
    input: "ReLU"
    input: "BinaryConnectAffine_2/bicon_affine/W"
    input: "BinaryConnectAffine_2/bicon_affine/Wb"
    input: "BinaryConnectAffine_2/bicon_affine/b"
    output: "BinaryConnectAffine_2"
    binary_connect_affine_param {
      base_axis: 1
    }
  }
  function {
    name: "BatchNormalization_2"
    type: "BatchNormalization"
    input: "BinaryConnectAffine_2"
    input: "BatchNormalization_2/bn/beta"
    input: "BatchNormalization_2/bn/gamma"
    input: "BatchNormalization_2/bn/mean"
    input: "BatchNormalization_2/bn/var"
    output: "BatchNormalization_2"
    batch_normalization_param {
      axes: 1
      decay_rate: 0.5
      eps: 0.01
      batch_stat: False
    }
  }
  function {
    name: "ReLU_2"
    type: "ReLU"
    input: "BatchNormalization_2"
    output: "ReLU_2"
    relu_param {
      inplace: True
    }
  }
  function {
    name: "BinaryConnectAffine_3"
    type: "BinaryConnectAffine"
    input: "ReLU_2"
    input: "BinaryConnectAffine_3/bicon_affine/W"
    input: "BinaryConnectAffine_3/bicon_affine/Wb"
    input: "BinaryConnectAffine_3/bicon_affine/b"
    output: "BinaryConnectAffine_3"
    binary_connect_affine_param {
      base_axis: 1
    }
  }
  function {
    name: "BatchNormalization_3"
    type: "BatchNormalization"
    input: "BinaryConnectAffine_3"
    input: "BatchNormalization_3/bn/beta"
    input: "BatchNormalization_3/bn/gamma"
    input: "BatchNormalization_3/bn/mean"
    input: "BatchNormalization_3/bn/var"
    output: "BatchNormalization_3"
    batch_normalization_param {
      axes: 1
      decay_rate: 0.5
      eps: 0.01
      batch_stat: False
    }
  }
  function {
    name: "ReLU_3"
    type: "ReLU"
    input: "BatchNormalization_3"
    output: "ReLU_3"
    relu_param {
      inplace: True
    }
  }
  function {
    name: "BinaryConnectAffine_4"
    type: "BinaryConnectAffine"
    input: "ReLU_3"
    input: "BinaryConnectAffine_4/bicon_affine/W"
    input: "BinaryConnectAffine_4/bicon_affine/Wb"
    input: "BinaryConnectAffine_4/bicon_affine/b"
    output: "BinaryConnectAffine_4"
    binary_connect_affine_param {
      base_axis: 1
    }
  }
  function {
    name: "BatchNormalization_4"
    type: "BatchNormalization"
    input: "BinaryConnectAffine_4"
    input: "BatchNormalization_4/bn/beta"
    input: "BatchNormalization_4/bn/gamma"
    input: "BatchNormalization_4/bn/mean"
    input: "BatchNormalization_4/bn/var"
    output: "BatchNormalization_4"
    batch_normalization_param {
      axes: 1
      decay_rate: 0.5
      eps: 0.01
      batch_stat: False
    }
  }
  function {
    name: "Softmax"
    type: "Softmax"
    input: "BatchNormalization_4"
    output: "Softmax"
    softmax_param {
      axis: 1
    }
  }
  function {
    name: "CategoricalCrossEntropy"
    type: "CategoricalCrossEntropy"
    input: "Softmax"
    input: "CategoricalCrossEntropy_T"
    output: "CategoricalCrossEntropy"
    categorical_cross_entropy_param {
      axis: 1
    }
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
    name: "BinaryConnectAffine/bicon_affine/W"
    type: "Parameter"
    shape: { dim: 784 dim: 2048 }
    initializer {
      type: "UniformAffineGlorot"
      multiplier: 1
    }
  }
  variable {
    name: "BinaryConnectAffine/bicon_affine/Wb"
    type: "Parameter"
    shape: { dim: 784 dim: 2048 }
    initializer {
      type: "Constant"
      multiplier: 0
    }
  }
  variable {
    name: "BinaryConnectAffine/bicon_affine/b"
    type: "Parameter"
    shape: { dim: 2048 }
    initializer {
      type: "Constant"
      multiplier: 0
    }
  }
  variable {
    name: "BatchNormalization/bn/beta"
    type: "Parameter"
    shape: { dim: 1 dim: 2048 }
    initializer {
      type: "Constant"
      multiplier: 0
    }
  }
  variable {
    name: "BatchNormalization/bn/gamma"
    type: "Parameter"
    shape: { dim: 1 dim: 2048 }
    initializer {
      type: "Constant"
      multiplier: 1
    }
  }
  variable {
    name: "BatchNormalization/bn/mean"
    type: "Parameter"
    shape: { dim: 1 dim: 2048 }
    initializer {
      type: "Constant"
      multiplier: 0
    }
  }
  variable {
    name: "BatchNormalization/bn/var"
    type: "Parameter"
    shape: { dim: 1 dim: 2048 }
    initializer {
      type: "Constant"
      multiplier: 0
    }
  }
  variable {
    name: "BinaryConnectAffine_2/bicon_affine/W"
    type: "Parameter"
    shape: { dim: 2048 dim: 2048 }
    initializer {
      type: "UniformAffineGlorot"
      multiplier: 1
    }
  }
  variable {
    name: "BinaryConnectAffine_2/bicon_affine/Wb"
    type: "Parameter"
    shape: { dim: 2048 dim: 2048 }
    initializer {
      type: "Constant"
      multiplier: 0
    }
  }
  variable {
    name: "BinaryConnectAffine_2/bicon_affine/b"
    type: "Parameter"
    shape: { dim: 2048 }
    initializer {
      type: "Constant"
      multiplier: 0
    }
  }
  variable {
    name: "BatchNormalization_2/bn/beta"
    type: "Parameter"
    shape: { dim: 1 dim: 2048 }
    initializer {
      type: "Constant"
      multiplier: 0
    }
  }
  variable {
    name: "BatchNormalization_2/bn/gamma"
    type: "Parameter"
    shape: { dim: 1 dim: 2048 }
    initializer {
      type: "Constant"
      multiplier: 1
    }
  }
  variable {
    name: "BatchNormalization_2/bn/mean"
    type: "Parameter"
    shape: { dim: 1 dim: 2048 }
    initializer {
      type: "Constant"
      multiplier: 0
    }
  }
  variable {
    name: "BatchNormalization_2/bn/var"
    type: "Parameter"
    shape: { dim: 1 dim: 2048 }
    initializer {
      type: "Constant"
      multiplier: 0
    }
  }
  variable {
    name: "BinaryConnectAffine_3/bicon_affine/W"
    type: "Parameter"
    shape: { dim: 2048 dim: 2048 }
    initializer {
      type: "UniformAffineGlorot"
      multiplier: 1
    }
  }
  variable {
    name: "BinaryConnectAffine_3/bicon_affine/Wb"
    type: "Parameter"
    shape: { dim: 2048 dim: 2048 }
    initializer {
      type: "Constant"
      multiplier: 0
    }
  }
  variable {
    name: "BinaryConnectAffine_3/bicon_affine/b"
    type: "Parameter"
    shape: { dim: 2048 }
    initializer {
      type: "Constant"
      multiplier: 0
    }
  }
  variable {
    name: "BatchNormalization_3/bn/beta"
    type: "Parameter"
    shape: { dim: 1 dim: 2048 }
    initializer {
      type: "Constant"
      multiplier: 0
    }
  }
  variable {
    name: "BatchNormalization_3/bn/gamma"
    type: "Parameter"
    shape: { dim: 1 dim: 2048 }
    initializer {
      type: "Constant"
      multiplier: 1
    }
  }
  variable {
    name: "BatchNormalization_3/bn/mean"
    type: "Parameter"
    shape: { dim: 1 dim: 2048 }
    initializer {
      type: "Constant"
      multiplier: 0
    }
  }
  variable {
    name: "BatchNormalization_3/bn/var"
    type: "Parameter"
    shape: { dim: 1 dim: 2048 }
    initializer {
      type: "Constant"
      multiplier: 0
    }
  }
  variable {
    name: "BinaryConnectAffine_4/bicon_affine/W"
    type: "Parameter"
    shape: { dim: 2048 dim: 10 }
    initializer {
      type: "UniformAffineGlorot"
      multiplier: 1
    }
  }
  variable {
    name: "BinaryConnectAffine_4/bicon_affine/Wb"
    type: "Parameter"
    shape: { dim: 2048 dim: 10 }
    initializer {
      type: "Constant"
      multiplier: 0
    }
  }
  variable {
    name: "BinaryConnectAffine_4/bicon_affine/b"
    type: "Parameter"
    shape: { dim: 10 }
    initializer {
      type: "Constant"
      multiplier: 0
    }
  }
  variable {
    name: "BatchNormalization_4/bn/beta"
    type: "Parameter"
    shape: { dim: 1 dim: 10 }
    initializer {
      type: "Constant"
      multiplier: 0
    }
  }
  variable {
    name: "BatchNormalization_4/bn/gamma"
    type: "Parameter"
    shape: { dim: 1 dim: 10 }
    initializer {
      type: "Constant"
      multiplier: 1
    }
  }
  variable {
    name: "BatchNormalization_4/bn/mean"
    type: "Parameter"
    shape: { dim: 1 dim: 10 }
    initializer {
      type: "Constant"
      multiplier: 0
    }
  }
  variable {
    name: "BatchNormalization_4/bn/var"
    type: "Parameter"
    shape: { dim: 1 dim: 10 }
    initializer {
      type: "Constant"
      multiplier: 0
    }
  }
  variable {
    name: "BinaryConnectAffine"
    type: "Buffer"
    shape: { dim:-1 dim: 2048 }
  }
  variable {
    name: "BatchNormalization"
    type: "Buffer"
    shape: { dim:-1 dim: 2048 }
  }
  variable {
    name: "ReLU"
    type: "Buffer"
    shape: { dim:-1 dim: 2048 }
  }
  variable {
    name: "BinaryConnectAffine_2"
    type: "Buffer"
    shape: { dim:-1 dim: 2048 }
  }
  variable {
    name: "BatchNormalization_2"
    type: "Buffer"
    shape: { dim:-1 dim: 2048 }
  }
  variable {
    name: "ReLU_2"
    type: "Buffer"
    shape: { dim:-1 dim: 2048 }
  }
  variable {
    name: "BinaryConnectAffine_3"
    type: "Buffer"
    shape: { dim:-1 dim: 2048 }
  }
  variable {
    name: "BatchNormalization_3"
    type: "Buffer"
    shape: { dim:-1 dim: 2048 }
  }
  variable {
    name: "ReLU_3"
    type: "Buffer"
    shape: { dim:-1 dim: 2048 }
  }
  variable {
    name: "BinaryConnectAffine_4"
    type: "Buffer"
    shape: { dim:-1 dim: 10 }
  }
  variable {
    name: "BatchNormalization_4"
    type: "Buffer"
    shape: { dim:-1 dim: 10 }
  }
  variable {
    name: "Softmax"
    type: "Buffer"
    shape: { dim:-1 dim: 10 }
  }
  function {
    name: "BinaryConnectAffine"
    type: "BinaryConnectAffine"
    input: "Input"
    input: "BinaryConnectAffine/bicon_affine/W"
    input: "BinaryConnectAffine/bicon_affine/Wb"
    input: "BinaryConnectAffine/bicon_affine/b"
    output: "BinaryConnectAffine"
    binary_connect_affine_param {
      base_axis: 1
    }
  }
  function {
    name: "BatchNormalization"
    type: "BatchNormalization"
    input: "BinaryConnectAffine"
    input: "BatchNormalization/bn/beta"
    input: "BatchNormalization/bn/gamma"
    input: "BatchNormalization/bn/mean"
    input: "BatchNormalization/bn/var"
    output: "BatchNormalization"
    batch_normalization_param {
      axes: 1
      decay_rate: 0.5
      eps: 0.01
      batch_stat: False
    }
  }
  function {
    name: "ReLU"
    type: "ReLU"
    input: "BatchNormalization"
    output: "ReLU"
    relu_param {
      inplace: True
    }
  }
  function {
    name: "BinaryConnectAffine_2"
    type: "BinaryConnectAffine"
    input: "ReLU"
    input: "BinaryConnectAffine_2/bicon_affine/W"
    input: "BinaryConnectAffine_2/bicon_affine/Wb"
    input: "BinaryConnectAffine_2/bicon_affine/b"
    output: "BinaryConnectAffine_2"
    binary_connect_affine_param {
      base_axis: 1
    }
  }
  function {
    name: "BatchNormalization_2"
    type: "BatchNormalization"
    input: "BinaryConnectAffine_2"
    input: "BatchNormalization_2/bn/beta"
    input: "BatchNormalization_2/bn/gamma"
    input: "BatchNormalization_2/bn/mean"
    input: "BatchNormalization_2/bn/var"
    output: "BatchNormalization_2"
    batch_normalization_param {
      axes: 1
      decay_rate: 0.5
      eps: 0.01
      batch_stat: False
    }
  }
  function {
    name: "ReLU_2"
    type: "ReLU"
    input: "BatchNormalization_2"
    output: "ReLU_2"
    relu_param {
      inplace: True
    }
  }
  function {
    name: "BinaryConnectAffine_3"
    type: "BinaryConnectAffine"
    input: "ReLU_2"
    input: "BinaryConnectAffine_3/bicon_affine/W"
    input: "BinaryConnectAffine_3/bicon_affine/Wb"
    input: "BinaryConnectAffine_3/bicon_affine/b"
    output: "BinaryConnectAffine_3"
    binary_connect_affine_param {
      base_axis: 1
    }
  }
  function {
    name: "BatchNormalization_3"
    type: "BatchNormalization"
    input: "BinaryConnectAffine_3"
    input: "BatchNormalization_3/bn/beta"
    input: "BatchNormalization_3/bn/gamma"
    input: "BatchNormalization_3/bn/mean"
    input: "BatchNormalization_3/bn/var"
    output: "BatchNormalization_3"
    batch_normalization_param {
      axes: 1
      decay_rate: 0.5
      eps: 0.01
      batch_stat: False
    }
  }
  function {
    name: "ReLU_3"
    type: "ReLU"
    input: "BatchNormalization_3"
    output: "ReLU_3"
    relu_param {
      inplace: True
    }
  }
  function {
    name: "BinaryConnectAffine_4"
    type: "BinaryConnectAffine"
    input: "ReLU_3"
    input: "BinaryConnectAffine_4/bicon_affine/W"
    input: "BinaryConnectAffine_4/bicon_affine/Wb"
    input: "BinaryConnectAffine_4/bicon_affine/b"
    output: "BinaryConnectAffine_4"
    binary_connect_affine_param {
      base_axis: 1
    }
  }
  function {
    name: "BatchNormalization_4"
    type: "BatchNormalization"
    input: "BinaryConnectAffine_4"
    input: "BatchNormalization_4/bn/beta"
    input: "BatchNormalization_4/bn/gamma"
    input: "BatchNormalization_4/bn/mean"
    input: "BatchNormalization_4/bn/var"
    output: "BatchNormalization_4"
    batch_normalization_param {
      axes: 1
      decay_rate: 0.5
      eps: 0.01
      batch_stat: False
    }
  }
  function {
    name: "Softmax"
    type: "Softmax"
    input: "BatchNormalization_4"
    output: "Softmax"
    softmax_param {
      axis: 1
    }
  }
}
dataset {
  name: "Training"
  uri: "C:\\Utils\\neural_network_console_internal_v120\\samples\\sample_dataset\\mnist\\mnist_training.csv"
  cache_dir: "C:\\Utils\\neural_network_console_internal_v120\\samples\\sample_dataset\\mnist\\mnist_training.cache"
  overwrite_cache: True
  create_cache_explicitly: True
  shuffle: true
  no_image_normalization: False
  batch_size: 64
}
dataset {
  name: "Validation"
  uri: "C:\\Utils\\neural_network_console_internal_v120\\samples\\sample_dataset\\mnist\\mnist_test.csv"
  cache_dir: "C:\\Utils\\neural_network_console_internal_v120\\samples\\sample_dataset\\mnist\\mnist_test.cache"
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
      alpha: 0.0001
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
    variable_name: "CategoricalCrossEntropy_T"
    data_name: "y"
  }
  loss_variable {
    variable_name: "CategoricalCrossEntropy"
  }
  parameter_variable {
    variable_name: "BinaryConnectAffine/bicon_affine/W"
    learning_rate_multiplier: 1
  }
  parameter_variable {
    variable_name: "BinaryConnectAffine/bicon_affine/Wb"
    learning_rate_multiplier: 0
  }
  parameter_variable {
    variable_name: "BinaryConnectAffine/bicon_affine/b"
    learning_rate_multiplier: 1
  }
  parameter_variable {
    variable_name: "BatchNormalization/bn/beta"
    learning_rate_multiplier: 1
  }
  parameter_variable {
    variable_name: "BatchNormalization/bn/gamma"
    learning_rate_multiplier: 1
  }
  parameter_variable {
    variable_name: "BatchNormalization/bn/mean"
    learning_rate_multiplier: 0
  }
  parameter_variable {
    variable_name: "BatchNormalization/bn/var"
    learning_rate_multiplier: 0
  }
  parameter_variable {
    variable_name: "BinaryConnectAffine_2/bicon_affine/W"
    learning_rate_multiplier: 1
  }
  parameter_variable {
    variable_name: "BinaryConnectAffine_2/bicon_affine/Wb"
    learning_rate_multiplier: 0
  }
  parameter_variable {
    variable_name: "BinaryConnectAffine_2/bicon_affine/b"
    learning_rate_multiplier: 1
  }
  parameter_variable {
    variable_name: "BatchNormalization_2/bn/beta"
    learning_rate_multiplier: 1
  }
  parameter_variable {
    variable_name: "BatchNormalization_2/bn/gamma"
    learning_rate_multiplier: 1
  }
  parameter_variable {
    variable_name: "BatchNormalization_2/bn/mean"
    learning_rate_multiplier: 0
  }
  parameter_variable {
    variable_name: "BatchNormalization_2/bn/var"
    learning_rate_multiplier: 0
  }
  parameter_variable {
    variable_name: "BinaryConnectAffine_3/bicon_affine/W"
    learning_rate_multiplier: 1
  }
  parameter_variable {
    variable_name: "BinaryConnectAffine_3/bicon_affine/Wb"
    learning_rate_multiplier: 0
  }
  parameter_variable {
    variable_name: "BinaryConnectAffine_3/bicon_affine/b"
    learning_rate_multiplier: 1
  }
  parameter_variable {
    variable_name: "BatchNormalization_3/bn/beta"
    learning_rate_multiplier: 1
  }
  parameter_variable {
    variable_name: "BatchNormalization_3/bn/gamma"
    learning_rate_multiplier: 1
  }
  parameter_variable {
    variable_name: "BatchNormalization_3/bn/mean"
    learning_rate_multiplier: 0
  }
  parameter_variable {
    variable_name: "BatchNormalization_3/bn/var"
    learning_rate_multiplier: 0
  }
  parameter_variable {
    variable_name: "BinaryConnectAffine_4/bicon_affine/W"
    learning_rate_multiplier: 1
  }
  parameter_variable {
    variable_name: "BinaryConnectAffine_4/bicon_affine/Wb"
    learning_rate_multiplier: 0
  }
  parameter_variable {
    variable_name: "BinaryConnectAffine_4/bicon_affine/b"
    learning_rate_multiplier: 1
  }
  parameter_variable {
    variable_name: "BatchNormalization_4/bn/beta"
    learning_rate_multiplier: 1
  }
  parameter_variable {
    variable_name: "BatchNormalization_4/bn/gamma"
    learning_rate_multiplier: 1
  }
  parameter_variable {
    variable_name: "BatchNormalization_4/bn/mean"
    learning_rate_multiplier: 0
  }
  parameter_variable {
    variable_name: "BatchNormalization_4/bn/var"
    learning_rate_multiplier: 0
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
    variable_name: "CategoricalCrossEntropy_T"
    data_name: "y"
  }
  monitor_variable {
    type: "Error"
    variable_name: "CategoricalCrossEntropy"
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
    variable_name: "CategoricalCrossEntropy_T"
    data_name: "y"
  }
  monitor_variable {
    type: "Error"
    variable_name: "CategoricalCrossEntropy"
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
    variable_name: "Softmax"
    data_name: "y'"
  }
  parameter_variable {
    variable_name: "BinaryConnectAffine/bicon_affine/W"
  }
  parameter_variable {
    variable_name: "BinaryConnectAffine/bicon_affine/Wb"
  }
  parameter_variable {
    variable_name: "BinaryConnectAffine/bicon_affine/b"
  }
  parameter_variable {
    variable_name: "BatchNormalization/bn/beta"
  }
  parameter_variable {
    variable_name: "BatchNormalization/bn/gamma"
  }
  parameter_variable {
    variable_name: "BatchNormalization/bn/mean"
  }
  parameter_variable {
    variable_name: "BatchNormalization/bn/var"
  }
  parameter_variable {
    variable_name: "BinaryConnectAffine_2/bicon_affine/W"
  }
  parameter_variable {
    variable_name: "BinaryConnectAffine_2/bicon_affine/Wb"
  }
  parameter_variable {
    variable_name: "BinaryConnectAffine_2/bicon_affine/b"
  }
  parameter_variable {
    variable_name: "BatchNormalization_2/bn/beta"
  }
  parameter_variable {
    variable_name: "BatchNormalization_2/bn/gamma"
  }
  parameter_variable {
    variable_name: "BatchNormalization_2/bn/mean"
  }
  parameter_variable {
    variable_name: "BatchNormalization_2/bn/var"
  }
  parameter_variable {
    variable_name: "BinaryConnectAffine_3/bicon_affine/W"
  }
  parameter_variable {
    variable_name: "BinaryConnectAffine_3/bicon_affine/Wb"
  }
  parameter_variable {
    variable_name: "BinaryConnectAffine_3/bicon_affine/b"
  }
  parameter_variable {
    variable_name: "BatchNormalization_3/bn/beta"
  }
  parameter_variable {
    variable_name: "BatchNormalization_3/bn/gamma"
  }
  parameter_variable {
    variable_name: "BatchNormalization_3/bn/mean"
  }
  parameter_variable {
    variable_name: "BatchNormalization_3/bn/var"
  }
  parameter_variable {
    variable_name: "BinaryConnectAffine_4/bicon_affine/W"
  }
  parameter_variable {
    variable_name: "BinaryConnectAffine_4/bicon_affine/Wb"
  }
  parameter_variable {
    variable_name: "BinaryConnectAffine_4/bicon_affine/b"
  }
  parameter_variable {
    variable_name: "BatchNormalization_4/bn/beta"
  }
  parameter_variable {
    variable_name: "BatchNormalization_4/bn/gamma"
  }
  parameter_variable {
    variable_name: "BatchNormalization_4/bn/mean"
  }
  parameter_variable {
    variable_name: "BatchNormalization_4/bn/var"
  }
}
'''
# The nntxt is generated from /home/woody/aid/nnabla-sample-data/sample_project/tutorial/binary_networks/binary_net_mnist_MLP.files/20180723_152007/net.nntxt.
N00000018 = r'''global_config {
  default_context {
    array_class: "CudaCachedArray"
    device_id: "0"
    backends: "cudnn:float"
    backends: "cuda:float"
    backends: "cpu:float"
  }
}
training_config {
  max_epoch: 50
  iter_per_epoch: 937
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
    name: "BinaryConnectAffine/bicon_affine/W"
    type: "Parameter"
    shape: { dim: 784 dim: 2048 }
    initializer {
      type: "UniformAffineGlorot"
      multiplier: 1
    }
  }
  variable {
    name: "BinaryConnectAffine/bicon_affine/Wb"
    type: "Parameter"
    shape: { dim: 784 dim: 2048 }
    initializer {
      type: "Constant"
      multiplier: 0
    }
  }
  variable {
    name: "BinaryConnectAffine/bicon_affine/b"
    type: "Parameter"
    shape: { dim: 2048 }
    initializer {
      type: "Constant"
      multiplier: 0
    }
  }
  variable {
    name: "BatchNormalization/bn/beta"
    type: "Parameter"
    shape: { dim: 1 dim: 2048 }
    initializer {
      type: "Constant"
      multiplier: 0
    }
  }
  variable {
    name: "BatchNormalization/bn/gamma"
    type: "Parameter"
    shape: { dim: 1 dim: 2048 }
    initializer {
      type: "Constant"
      multiplier: 1
    }
  }
  variable {
    name: "BatchNormalization/bn/mean"
    type: "Parameter"
    shape: { dim: 1 dim: 2048 }
    initializer {
      type: "Constant"
      multiplier: 0
    }
  }
  variable {
    name: "BatchNormalization/bn/var"
    type: "Parameter"
    shape: { dim: 1 dim: 2048 }
    initializer {
      type: "Constant"
      multiplier: 0
    }
  }
  variable {
    name: "BinaryConnectAffine_2/bicon_affine/W"
    type: "Parameter"
    shape: { dim: 2048 dim: 2048 }
    initializer {
      type: "UniformAffineGlorot"
      multiplier: 1
    }
  }
  variable {
    name: "BinaryConnectAffine_2/bicon_affine/Wb"
    type: "Parameter"
    shape: { dim: 2048 dim: 2048 }
    initializer {
      type: "Constant"
      multiplier: 0
    }
  }
  variable {
    name: "BinaryConnectAffine_2/bicon_affine/b"
    type: "Parameter"
    shape: { dim: 2048 }
    initializer {
      type: "Constant"
      multiplier: 0
    }
  }
  variable {
    name: "BatchNormalization_2/bn/beta"
    type: "Parameter"
    shape: { dim: 1 dim: 2048 }
    initializer {
      type: "Constant"
      multiplier: 0
    }
  }
  variable {
    name: "BatchNormalization_2/bn/gamma"
    type: "Parameter"
    shape: { dim: 1 dim: 2048 }
    initializer {
      type: "Constant"
      multiplier: 1
    }
  }
  variable {
    name: "BatchNormalization_2/bn/mean"
    type: "Parameter"
    shape: { dim: 1 dim: 2048 }
    initializer {
      type: "Constant"
      multiplier: 0
    }
  }
  variable {
    name: "BatchNormalization_2/bn/var"
    type: "Parameter"
    shape: { dim: 1 dim: 2048 }
    initializer {
      type: "Constant"
      multiplier: 0
    }
  }
  variable {
    name: "BinaryConnectAffine_3/bicon_affine/W"
    type: "Parameter"
    shape: { dim: 2048 dim: 2048 }
    initializer {
      type: "UniformAffineGlorot"
      multiplier: 1
    }
  }
  variable {
    name: "BinaryConnectAffine_3/bicon_affine/Wb"
    type: "Parameter"
    shape: { dim: 2048 dim: 2048 }
    initializer {
      type: "Constant"
      multiplier: 0
    }
  }
  variable {
    name: "BinaryConnectAffine_3/bicon_affine/b"
    type: "Parameter"
    shape: { dim: 2048 }
    initializer {
      type: "Constant"
      multiplier: 0
    }
  }
  variable {
    name: "BatchNormalization_3/bn/beta"
    type: "Parameter"
    shape: { dim: 1 dim: 2048 }
    initializer {
      type: "Constant"
      multiplier: 0
    }
  }
  variable {
    name: "BatchNormalization_3/bn/gamma"
    type: "Parameter"
    shape: { dim: 1 dim: 2048 }
    initializer {
      type: "Constant"
      multiplier: 1
    }
  }
  variable {
    name: "BatchNormalization_3/bn/mean"
    type: "Parameter"
    shape: { dim: 1 dim: 2048 }
    initializer {
      type: "Constant"
      multiplier: 0
    }
  }
  variable {
    name: "BatchNormalization_3/bn/var"
    type: "Parameter"
    shape: { dim: 1 dim: 2048 }
    initializer {
      type: "Constant"
      multiplier: 0
    }
  }
  variable {
    name: "BinaryConnectAffine_4/bicon_affine/W"
    type: "Parameter"
    shape: { dim: 2048 dim: 10 }
    initializer {
      type: "UniformAffineGlorot"
      multiplier: 1
    }
  }
  variable {
    name: "BinaryConnectAffine_4/bicon_affine/Wb"
    type: "Parameter"
    shape: { dim: 2048 dim: 10 }
    initializer {
      type: "Constant"
      multiplier: 0
    }
  }
  variable {
    name: "BinaryConnectAffine_4/bicon_affine/b"
    type: "Parameter"
    shape: { dim: 10 }
    initializer {
      type: "Constant"
      multiplier: 0
    }
  }
  variable {
    name: "BatchNormalization_4/bn/beta"
    type: "Parameter"
    shape: { dim: 1 dim: 10 }
    initializer {
      type: "Constant"
      multiplier: 0
    }
  }
  variable {
    name: "BatchNormalization_4/bn/gamma"
    type: "Parameter"
    shape: { dim: 1 dim: 10 }
    initializer {
      type: "Constant"
      multiplier: 1
    }
  }
  variable {
    name: "BatchNormalization_4/bn/mean"
    type: "Parameter"
    shape: { dim: 1 dim: 10 }
    initializer {
      type: "Constant"
      multiplier: 0
    }
  }
  variable {
    name: "BatchNormalization_4/bn/var"
    type: "Parameter"
    shape: { dim: 1 dim: 10 }
    initializer {
      type: "Constant"
      multiplier: 0
    }
  }
  variable {
    name: "CategoricalCrossEntropy_T"
    type: "Buffer"
    shape: { dim:-1 dim: 1 }
  }
  variable {
    name: "BinaryConnectAffine"
    type: "Buffer"
    shape: { dim:-1 dim: 2048 }
  }
  variable {
    name: "BatchNormalization"
    type: "Buffer"
    shape: { dim:-1 dim: 2048 }
  }
  variable {
    name: "BinarySigmoid"
    type: "Buffer"
    shape: { dim:-1 dim: 2048 }
  }
  variable {
    name: "BinaryConnectAffine_2"
    type: "Buffer"
    shape: { dim:-1 dim: 2048 }
  }
  variable {
    name: "BatchNormalization_2"
    type: "Buffer"
    shape: { dim:-1 dim: 2048 }
  }
  variable {
    name: "BinarySigmoid_2"
    type: "Buffer"
    shape: { dim:-1 dim: 2048 }
  }
  variable {
    name: "BinaryConnectAffine_3"
    type: "Buffer"
    shape: { dim:-1 dim: 2048 }
  }
  variable {
    name: "BatchNormalization_3"
    type: "Buffer"
    shape: { dim:-1 dim: 2048 }
  }
  variable {
    name: "BinarySigmoid_3"
    type: "Buffer"
    shape: { dim:-1 dim: 2048 }
  }
  variable {
    name: "BinaryConnectAffine_4"
    type: "Buffer"
    shape: { dim:-1 dim: 10 }
  }
  variable {
    name: "BatchNormalization_4"
    type: "Buffer"
    shape: { dim:-1 dim: 10 }
  }
  variable {
    name: "Softmax"
    type: "Buffer"
    shape: { dim:-1 dim: 10 }
  }
  variable {
    name: "CategoricalCrossEntropy"
    type: "Buffer"
    shape: { dim:-1 dim: 1 }
  }
  function {
    name: "BinaryConnectAffine"
    type: "BinaryConnectAffine"
    input: "Input"
    input: "BinaryConnectAffine/bicon_affine/W"
    input: "BinaryConnectAffine/bicon_affine/Wb"
    input: "BinaryConnectAffine/bicon_affine/b"
    output: "BinaryConnectAffine"
    binary_connect_affine_param {
      base_axis: 1
    }
  }
  function {
    name: "BatchNormalization"
    type: "BatchNormalization"
    input: "BinaryConnectAffine"
    input: "BatchNormalization/bn/beta"
    input: "BatchNormalization/bn/gamma"
    input: "BatchNormalization/bn/mean"
    input: "BatchNormalization/bn/var"
    output: "BatchNormalization"
    batch_normalization_param {
      axes: 1
      decay_rate: 0.5
      eps: 0.01
      batch_stat: True
    }
  }
  function {
    name: "BinarySigmoid"
    type: "BinarySigmoid"
    input: "BatchNormalization"
    output: "BinarySigmoid"
  }
  function {
    name: "BinaryConnectAffine_2"
    type: "BinaryConnectAffine"
    input: "BinarySigmoid"
    input: "BinaryConnectAffine_2/bicon_affine/W"
    input: "BinaryConnectAffine_2/bicon_affine/Wb"
    input: "BinaryConnectAffine_2/bicon_affine/b"
    output: "BinaryConnectAffine_2"
    binary_connect_affine_param {
      base_axis: 1
    }
  }
  function {
    name: "BatchNormalization_2"
    type: "BatchNormalization"
    input: "BinaryConnectAffine_2"
    input: "BatchNormalization_2/bn/beta"
    input: "BatchNormalization_2/bn/gamma"
    input: "BatchNormalization_2/bn/mean"
    input: "BatchNormalization_2/bn/var"
    output: "BatchNormalization_2"
    batch_normalization_param {
      axes: 1
      decay_rate: 0.5
      eps: 0.01
      batch_stat: True
    }
  }
  function {
    name: "BinarySigmoid_2"
    type: "BinarySigmoid"
    input: "BatchNormalization_2"
    output: "BinarySigmoid_2"
  }
  function {
    name: "BinaryConnectAffine_3"
    type: "BinaryConnectAffine"
    input: "BinarySigmoid_2"
    input: "BinaryConnectAffine_3/bicon_affine/W"
    input: "BinaryConnectAffine_3/bicon_affine/Wb"
    input: "BinaryConnectAffine_3/bicon_affine/b"
    output: "BinaryConnectAffine_3"
    binary_connect_affine_param {
      base_axis: 1
    }
  }
  function {
    name: "BatchNormalization_3"
    type: "BatchNormalization"
    input: "BinaryConnectAffine_3"
    input: "BatchNormalization_3/bn/beta"
    input: "BatchNormalization_3/bn/gamma"
    input: "BatchNormalization_3/bn/mean"
    input: "BatchNormalization_3/bn/var"
    output: "BatchNormalization_3"
    batch_normalization_param {
      axes: 1
      decay_rate: 0.5
      eps: 0.01
      batch_stat: True
    }
  }
  function {
    name: "BinarySigmoid_3"
    type: "BinarySigmoid"
    input: "BatchNormalization_3"
    output: "BinarySigmoid_3"
  }
  function {
    name: "BinaryConnectAffine_4"
    type: "BinaryConnectAffine"
    input: "BinarySigmoid_3"
    input: "BinaryConnectAffine_4/bicon_affine/W"
    input: "BinaryConnectAffine_4/bicon_affine/Wb"
    input: "BinaryConnectAffine_4/bicon_affine/b"
    output: "BinaryConnectAffine_4"
    binary_connect_affine_param {
      base_axis: 1
    }
  }
  function {
    name: "BatchNormalization_4"
    type: "BatchNormalization"
    input: "BinaryConnectAffine_4"
    input: "BatchNormalization_4/bn/beta"
    input: "BatchNormalization_4/bn/gamma"
    input: "BatchNormalization_4/bn/mean"
    input: "BatchNormalization_4/bn/var"
    output: "BatchNormalization_4"
    batch_normalization_param {
      axes: 1
      decay_rate: 0.5
      eps: 0.01
      batch_stat: True
    }
  }
  function {
    name: "Softmax"
    type: "Softmax"
    input: "BatchNormalization_4"
    output: "Softmax"
    softmax_param {
      axis: 1
    }
  }
  function {
    name: "CategoricalCrossEntropy"
    type: "CategoricalCrossEntropy"
    input: "Softmax"
    input: "CategoricalCrossEntropy_T"
    output: "CategoricalCrossEntropy"
    categorical_cross_entropy_param {
      axis: 1
    }
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
    name: "BinaryConnectAffine/bicon_affine/W"
    type: "Parameter"
    shape: { dim: 784 dim: 2048 }
    initializer {
      type: "UniformAffineGlorot"
      multiplier: 1
    }
  }
  variable {
    name: "BinaryConnectAffine/bicon_affine/Wb"
    type: "Parameter"
    shape: { dim: 784 dim: 2048 }
    initializer {
      type: "Constant"
      multiplier: 0
    }
  }
  variable {
    name: "BinaryConnectAffine/bicon_affine/b"
    type: "Parameter"
    shape: { dim: 2048 }
    initializer {
      type: "Constant"
      multiplier: 0
    }
  }
  variable {
    name: "BatchNormalization/bn/beta"
    type: "Parameter"
    shape: { dim: 1 dim: 2048 }
    initializer {
      type: "Constant"
      multiplier: 0
    }
  }
  variable {
    name: "BatchNormalization/bn/gamma"
    type: "Parameter"
    shape: { dim: 1 dim: 2048 }
    initializer {
      type: "Constant"
      multiplier: 1
    }
  }
  variable {
    name: "BatchNormalization/bn/mean"
    type: "Parameter"
    shape: { dim: 1 dim: 2048 }
    initializer {
      type: "Constant"
      multiplier: 0
    }
  }
  variable {
    name: "BatchNormalization/bn/var"
    type: "Parameter"
    shape: { dim: 1 dim: 2048 }
    initializer {
      type: "Constant"
      multiplier: 0
    }
  }
  variable {
    name: "BinaryConnectAffine_2/bicon_affine/W"
    type: "Parameter"
    shape: { dim: 2048 dim: 2048 }
    initializer {
      type: "UniformAffineGlorot"
      multiplier: 1
    }
  }
  variable {
    name: "BinaryConnectAffine_2/bicon_affine/Wb"
    type: "Parameter"
    shape: { dim: 2048 dim: 2048 }
    initializer {
      type: "Constant"
      multiplier: 0
    }
  }
  variable {
    name: "BinaryConnectAffine_2/bicon_affine/b"
    type: "Parameter"
    shape: { dim: 2048 }
    initializer {
      type: "Constant"
      multiplier: 0
    }
  }
  variable {
    name: "BatchNormalization_2/bn/beta"
    type: "Parameter"
    shape: { dim: 1 dim: 2048 }
    initializer {
      type: "Constant"
      multiplier: 0
    }
  }
  variable {
    name: "BatchNormalization_2/bn/gamma"
    type: "Parameter"
    shape: { dim: 1 dim: 2048 }
    initializer {
      type: "Constant"
      multiplier: 1
    }
  }
  variable {
    name: "BatchNormalization_2/bn/mean"
    type: "Parameter"
    shape: { dim: 1 dim: 2048 }
    initializer {
      type: "Constant"
      multiplier: 0
    }
  }
  variable {
    name: "BatchNormalization_2/bn/var"
    type: "Parameter"
    shape: { dim: 1 dim: 2048 }
    initializer {
      type: "Constant"
      multiplier: 0
    }
  }
  variable {
    name: "BinaryConnectAffine_3/bicon_affine/W"
    type: "Parameter"
    shape: { dim: 2048 dim: 2048 }
    initializer {
      type: "UniformAffineGlorot"
      multiplier: 1
    }
  }
  variable {
    name: "BinaryConnectAffine_3/bicon_affine/Wb"
    type: "Parameter"
    shape: { dim: 2048 dim: 2048 }
    initializer {
      type: "Constant"
      multiplier: 0
    }
  }
  variable {
    name: "BinaryConnectAffine_3/bicon_affine/b"
    type: "Parameter"
    shape: { dim: 2048 }
    initializer {
      type: "Constant"
      multiplier: 0
    }
  }
  variable {
    name: "BatchNormalization_3/bn/beta"
    type: "Parameter"
    shape: { dim: 1 dim: 2048 }
    initializer {
      type: "Constant"
      multiplier: 0
    }
  }
  variable {
    name: "BatchNormalization_3/bn/gamma"
    type: "Parameter"
    shape: { dim: 1 dim: 2048 }
    initializer {
      type: "Constant"
      multiplier: 1
    }
  }
  variable {
    name: "BatchNormalization_3/bn/mean"
    type: "Parameter"
    shape: { dim: 1 dim: 2048 }
    initializer {
      type: "Constant"
      multiplier: 0
    }
  }
  variable {
    name: "BatchNormalization_3/bn/var"
    type: "Parameter"
    shape: { dim: 1 dim: 2048 }
    initializer {
      type: "Constant"
      multiplier: 0
    }
  }
  variable {
    name: "BinaryConnectAffine_4/bicon_affine/W"
    type: "Parameter"
    shape: { dim: 2048 dim: 10 }
    initializer {
      type: "UniformAffineGlorot"
      multiplier: 1
    }
  }
  variable {
    name: "BinaryConnectAffine_4/bicon_affine/Wb"
    type: "Parameter"
    shape: { dim: 2048 dim: 10 }
    initializer {
      type: "Constant"
      multiplier: 0
    }
  }
  variable {
    name: "BinaryConnectAffine_4/bicon_affine/b"
    type: "Parameter"
    shape: { dim: 10 }
    initializer {
      type: "Constant"
      multiplier: 0
    }
  }
  variable {
    name: "BatchNormalization_4/bn/beta"
    type: "Parameter"
    shape: { dim: 1 dim: 10 }
    initializer {
      type: "Constant"
      multiplier: 0
    }
  }
  variable {
    name: "BatchNormalization_4/bn/gamma"
    type: "Parameter"
    shape: { dim: 1 dim: 10 }
    initializer {
      type: "Constant"
      multiplier: 1
    }
  }
  variable {
    name: "BatchNormalization_4/bn/mean"
    type: "Parameter"
    shape: { dim: 1 dim: 10 }
    initializer {
      type: "Constant"
      multiplier: 0
    }
  }
  variable {
    name: "BatchNormalization_4/bn/var"
    type: "Parameter"
    shape: { dim: 1 dim: 10 }
    initializer {
      type: "Constant"
      multiplier: 0
    }
  }
  variable {
    name: "CategoricalCrossEntropy_T"
    type: "Buffer"
    shape: { dim:-1 dim: 1 }
  }
  variable {
    name: "BinaryConnectAffine"
    type: "Buffer"
    shape: { dim:-1 dim: 2048 }
  }
  variable {
    name: "BatchNormalization"
    type: "Buffer"
    shape: { dim:-1 dim: 2048 }
  }
  variable {
    name: "BinarySigmoid"
    type: "Buffer"
    shape: { dim:-1 dim: 2048 }
  }
  variable {
    name: "BinaryConnectAffine_2"
    type: "Buffer"
    shape: { dim:-1 dim: 2048 }
  }
  variable {
    name: "BatchNormalization_2"
    type: "Buffer"
    shape: { dim:-1 dim: 2048 }
  }
  variable {
    name: "BinarySigmoid_2"
    type: "Buffer"
    shape: { dim:-1 dim: 2048 }
  }
  variable {
    name: "BinaryConnectAffine_3"
    type: "Buffer"
    shape: { dim:-1 dim: 2048 }
  }
  variable {
    name: "BatchNormalization_3"
    type: "Buffer"
    shape: { dim:-1 dim: 2048 }
  }
  variable {
    name: "BinarySigmoid_3"
    type: "Buffer"
    shape: { dim:-1 dim: 2048 }
  }
  variable {
    name: "BinaryConnectAffine_4"
    type: "Buffer"
    shape: { dim:-1 dim: 10 }
  }
  variable {
    name: "BatchNormalization_4"
    type: "Buffer"
    shape: { dim:-1 dim: 10 }
  }
  variable {
    name: "Softmax"
    type: "Buffer"
    shape: { dim:-1 dim: 10 }
  }
  variable {
    name: "CategoricalCrossEntropy"
    type: "Buffer"
    shape: { dim:-1 dim: 1 }
  }
  function {
    name: "BinaryConnectAffine"
    type: "BinaryConnectAffine"
    input: "Input"
    input: "BinaryConnectAffine/bicon_affine/W"
    input: "BinaryConnectAffine/bicon_affine/Wb"
    input: "BinaryConnectAffine/bicon_affine/b"
    output: "BinaryConnectAffine"
    binary_connect_affine_param {
      base_axis: 1
    }
  }
  function {
    name: "BatchNormalization"
    type: "BatchNormalization"
    input: "BinaryConnectAffine"
    input: "BatchNormalization/bn/beta"
    input: "BatchNormalization/bn/gamma"
    input: "BatchNormalization/bn/mean"
    input: "BatchNormalization/bn/var"
    output: "BatchNormalization"
    batch_normalization_param {
      axes: 1
      decay_rate: 0.5
      eps: 0.01
      batch_stat: False
    }
  }
  function {
    name: "BinarySigmoid"
    type: "BinarySigmoid"
    input: "BatchNormalization"
    output: "BinarySigmoid"
  }
  function {
    name: "BinaryConnectAffine_2"
    type: "BinaryConnectAffine"
    input: "BinarySigmoid"
    input: "BinaryConnectAffine_2/bicon_affine/W"
    input: "BinaryConnectAffine_2/bicon_affine/Wb"
    input: "BinaryConnectAffine_2/bicon_affine/b"
    output: "BinaryConnectAffine_2"
    binary_connect_affine_param {
      base_axis: 1
    }
  }
  function {
    name: "BatchNormalization_2"
    type: "BatchNormalization"
    input: "BinaryConnectAffine_2"
    input: "BatchNormalization_2/bn/beta"
    input: "BatchNormalization_2/bn/gamma"
    input: "BatchNormalization_2/bn/mean"
    input: "BatchNormalization_2/bn/var"
    output: "BatchNormalization_2"
    batch_normalization_param {
      axes: 1
      decay_rate: 0.5
      eps: 0.01
      batch_stat: False
    }
  }
  function {
    name: "BinarySigmoid_2"
    type: "BinarySigmoid"
    input: "BatchNormalization_2"
    output: "BinarySigmoid_2"
  }
  function {
    name: "BinaryConnectAffine_3"
    type: "BinaryConnectAffine"
    input: "BinarySigmoid_2"
    input: "BinaryConnectAffine_3/bicon_affine/W"
    input: "BinaryConnectAffine_3/bicon_affine/Wb"
    input: "BinaryConnectAffine_3/bicon_affine/b"
    output: "BinaryConnectAffine_3"
    binary_connect_affine_param {
      base_axis: 1
    }
  }
  function {
    name: "BatchNormalization_3"
    type: "BatchNormalization"
    input: "BinaryConnectAffine_3"
    input: "BatchNormalization_3/bn/beta"
    input: "BatchNormalization_3/bn/gamma"
    input: "BatchNormalization_3/bn/mean"
    input: "BatchNormalization_3/bn/var"
    output: "BatchNormalization_3"
    batch_normalization_param {
      axes: 1
      decay_rate: 0.5
      eps: 0.01
      batch_stat: False
    }
  }
  function {
    name: "BinarySigmoid_3"
    type: "BinarySigmoid"
    input: "BatchNormalization_3"
    output: "BinarySigmoid_3"
  }
  function {
    name: "BinaryConnectAffine_4"
    type: "BinaryConnectAffine"
    input: "BinarySigmoid_3"
    input: "BinaryConnectAffine_4/bicon_affine/W"
    input: "BinaryConnectAffine_4/bicon_affine/Wb"
    input: "BinaryConnectAffine_4/bicon_affine/b"
    output: "BinaryConnectAffine_4"
    binary_connect_affine_param {
      base_axis: 1
    }
  }
  function {
    name: "BatchNormalization_4"
    type: "BatchNormalization"
    input: "BinaryConnectAffine_4"
    input: "BatchNormalization_4/bn/beta"
    input: "BatchNormalization_4/bn/gamma"
    input: "BatchNormalization_4/bn/mean"
    input: "BatchNormalization_4/bn/var"
    output: "BatchNormalization_4"
    batch_normalization_param {
      axes: 1
      decay_rate: 0.5
      eps: 0.01
      batch_stat: False
    }
  }
  function {
    name: "Softmax"
    type: "Softmax"
    input: "BatchNormalization_4"
    output: "Softmax"
    softmax_param {
      axis: 1
    }
  }
  function {
    name: "CategoricalCrossEntropy"
    type: "CategoricalCrossEntropy"
    input: "Softmax"
    input: "CategoricalCrossEntropy_T"
    output: "CategoricalCrossEntropy"
    categorical_cross_entropy_param {
      axis: 1
    }
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
    name: "BinaryConnectAffine/bicon_affine/W"
    type: "Parameter"
    shape: { dim: 784 dim: 2048 }
    initializer {
      type: "UniformAffineGlorot"
      multiplier: 1
    }
  }
  variable {
    name: "BinaryConnectAffine/bicon_affine/Wb"
    type: "Parameter"
    shape: { dim: 784 dim: 2048 }
    initializer {
      type: "Constant"
      multiplier: 0
    }
  }
  variable {
    name: "BinaryConnectAffine/bicon_affine/b"
    type: "Parameter"
    shape: { dim: 2048 }
    initializer {
      type: "Constant"
      multiplier: 0
    }
  }
  variable {
    name: "BatchNormalization/bn/beta"
    type: "Parameter"
    shape: { dim: 1 dim: 2048 }
    initializer {
      type: "Constant"
      multiplier: 0
    }
  }
  variable {
    name: "BatchNormalization/bn/gamma"
    type: "Parameter"
    shape: { dim: 1 dim: 2048 }
    initializer {
      type: "Constant"
      multiplier: 1
    }
  }
  variable {
    name: "BatchNormalization/bn/mean"
    type: "Parameter"
    shape: { dim: 1 dim: 2048 }
    initializer {
      type: "Constant"
      multiplier: 0
    }
  }
  variable {
    name: "BatchNormalization/bn/var"
    type: "Parameter"
    shape: { dim: 1 dim: 2048 }
    initializer {
      type: "Constant"
      multiplier: 0
    }
  }
  variable {
    name: "BinaryConnectAffine_2/bicon_affine/W"
    type: "Parameter"
    shape: { dim: 2048 dim: 2048 }
    initializer {
      type: "UniformAffineGlorot"
      multiplier: 1
    }
  }
  variable {
    name: "BinaryConnectAffine_2/bicon_affine/Wb"
    type: "Parameter"
    shape: { dim: 2048 dim: 2048 }
    initializer {
      type: "Constant"
      multiplier: 0
    }
  }
  variable {
    name: "BinaryConnectAffine_2/bicon_affine/b"
    type: "Parameter"
    shape: { dim: 2048 }
    initializer {
      type: "Constant"
      multiplier: 0
    }
  }
  variable {
    name: "BatchNormalization_2/bn/beta"
    type: "Parameter"
    shape: { dim: 1 dim: 2048 }
    initializer {
      type: "Constant"
      multiplier: 0
    }
  }
  variable {
    name: "BatchNormalization_2/bn/gamma"
    type: "Parameter"
    shape: { dim: 1 dim: 2048 }
    initializer {
      type: "Constant"
      multiplier: 1
    }
  }
  variable {
    name: "BatchNormalization_2/bn/mean"
    type: "Parameter"
    shape: { dim: 1 dim: 2048 }
    initializer {
      type: "Constant"
      multiplier: 0
    }
  }
  variable {
    name: "BatchNormalization_2/bn/var"
    type: "Parameter"
    shape: { dim: 1 dim: 2048 }
    initializer {
      type: "Constant"
      multiplier: 0
    }
  }
  variable {
    name: "BinaryConnectAffine_3/bicon_affine/W"
    type: "Parameter"
    shape: { dim: 2048 dim: 2048 }
    initializer {
      type: "UniformAffineGlorot"
      multiplier: 1
    }
  }
  variable {
    name: "BinaryConnectAffine_3/bicon_affine/Wb"
    type: "Parameter"
    shape: { dim: 2048 dim: 2048 }
    initializer {
      type: "Constant"
      multiplier: 0
    }
  }
  variable {
    name: "BinaryConnectAffine_3/bicon_affine/b"
    type: "Parameter"
    shape: { dim: 2048 }
    initializer {
      type: "Constant"
      multiplier: 0
    }
  }
  variable {
    name: "BatchNormalization_3/bn/beta"
    type: "Parameter"
    shape: { dim: 1 dim: 2048 }
    initializer {
      type: "Constant"
      multiplier: 0
    }
  }
  variable {
    name: "BatchNormalization_3/bn/gamma"
    type: "Parameter"
    shape: { dim: 1 dim: 2048 }
    initializer {
      type: "Constant"
      multiplier: 1
    }
  }
  variable {
    name: "BatchNormalization_3/bn/mean"
    type: "Parameter"
    shape: { dim: 1 dim: 2048 }
    initializer {
      type: "Constant"
      multiplier: 0
    }
  }
  variable {
    name: "BatchNormalization_3/bn/var"
    type: "Parameter"
    shape: { dim: 1 dim: 2048 }
    initializer {
      type: "Constant"
      multiplier: 0
    }
  }
  variable {
    name: "BinaryConnectAffine_4/bicon_affine/W"
    type: "Parameter"
    shape: { dim: 2048 dim: 10 }
    initializer {
      type: "UniformAffineGlorot"
      multiplier: 1
    }
  }
  variable {
    name: "BinaryConnectAffine_4/bicon_affine/Wb"
    type: "Parameter"
    shape: { dim: 2048 dim: 10 }
    initializer {
      type: "Constant"
      multiplier: 0
    }
  }
  variable {
    name: "BinaryConnectAffine_4/bicon_affine/b"
    type: "Parameter"
    shape: { dim: 10 }
    initializer {
      type: "Constant"
      multiplier: 0
    }
  }
  variable {
    name: "BatchNormalization_4/bn/beta"
    type: "Parameter"
    shape: { dim: 1 dim: 10 }
    initializer {
      type: "Constant"
      multiplier: 0
    }
  }
  variable {
    name: "BatchNormalization_4/bn/gamma"
    type: "Parameter"
    shape: { dim: 1 dim: 10 }
    initializer {
      type: "Constant"
      multiplier: 1
    }
  }
  variable {
    name: "BatchNormalization_4/bn/mean"
    type: "Parameter"
    shape: { dim: 1 dim: 10 }
    initializer {
      type: "Constant"
      multiplier: 0
    }
  }
  variable {
    name: "BatchNormalization_4/bn/var"
    type: "Parameter"
    shape: { dim: 1 dim: 10 }
    initializer {
      type: "Constant"
      multiplier: 0
    }
  }
  variable {
    name: "BinaryConnectAffine"
    type: "Buffer"
    shape: { dim:-1 dim: 2048 }
  }
  variable {
    name: "BatchNormalization"
    type: "Buffer"
    shape: { dim:-1 dim: 2048 }
  }
  variable {
    name: "BinarySigmoid"
    type: "Buffer"
    shape: { dim:-1 dim: 2048 }
  }
  variable {
    name: "BinaryConnectAffine_2"
    type: "Buffer"
    shape: { dim:-1 dim: 2048 }
  }
  variable {
    name: "BatchNormalization_2"
    type: "Buffer"
    shape: { dim:-1 dim: 2048 }
  }
  variable {
    name: "BinarySigmoid_2"
    type: "Buffer"
    shape: { dim:-1 dim: 2048 }
  }
  variable {
    name: "BinaryConnectAffine_3"
    type: "Buffer"
    shape: { dim:-1 dim: 2048 }
  }
  variable {
    name: "BatchNormalization_3"
    type: "Buffer"
    shape: { dim:-1 dim: 2048 }
  }
  variable {
    name: "BinarySigmoid_3"
    type: "Buffer"
    shape: { dim:-1 dim: 2048 }
  }
  variable {
    name: "BinaryConnectAffine_4"
    type: "Buffer"
    shape: { dim:-1 dim: 10 }
  }
  variable {
    name: "BatchNormalization_4"
    type: "Buffer"
    shape: { dim:-1 dim: 10 }
  }
  variable {
    name: "Softmax"
    type: "Buffer"
    shape: { dim:-1 dim: 10 }
  }
  function {
    name: "BinaryConnectAffine"
    type: "BinaryConnectAffine"
    input: "Input"
    input: "BinaryConnectAffine/bicon_affine/W"
    input: "BinaryConnectAffine/bicon_affine/Wb"
    input: "BinaryConnectAffine/bicon_affine/b"
    output: "BinaryConnectAffine"
    binary_connect_affine_param {
      base_axis: 1
    }
  }
  function {
    name: "BatchNormalization"
    type: "BatchNormalization"
    input: "BinaryConnectAffine"
    input: "BatchNormalization/bn/beta"
    input: "BatchNormalization/bn/gamma"
    input: "BatchNormalization/bn/mean"
    input: "BatchNormalization/bn/var"
    output: "BatchNormalization"
    batch_normalization_param {
      axes: 1
      decay_rate: 0.5
      eps: 0.01
      batch_stat: False
    }
  }
  function {
    name: "BinarySigmoid"
    type: "BinarySigmoid"
    input: "BatchNormalization"
    output: "BinarySigmoid"
  }
  function {
    name: "BinaryConnectAffine_2"
    type: "BinaryConnectAffine"
    input: "BinarySigmoid"
    input: "BinaryConnectAffine_2/bicon_affine/W"
    input: "BinaryConnectAffine_2/bicon_affine/Wb"
    input: "BinaryConnectAffine_2/bicon_affine/b"
    output: "BinaryConnectAffine_2"
    binary_connect_affine_param {
      base_axis: 1
    }
  }
  function {
    name: "BatchNormalization_2"
    type: "BatchNormalization"
    input: "BinaryConnectAffine_2"
    input: "BatchNormalization_2/bn/beta"
    input: "BatchNormalization_2/bn/gamma"
    input: "BatchNormalization_2/bn/mean"
    input: "BatchNormalization_2/bn/var"
    output: "BatchNormalization_2"
    batch_normalization_param {
      axes: 1
      decay_rate: 0.5
      eps: 0.01
      batch_stat: False
    }
  }
  function {
    name: "BinarySigmoid_2"
    type: "BinarySigmoid"
    input: "BatchNormalization_2"
    output: "BinarySigmoid_2"
  }
  function {
    name: "BinaryConnectAffine_3"
    type: "BinaryConnectAffine"
    input: "BinarySigmoid_2"
    input: "BinaryConnectAffine_3/bicon_affine/W"
    input: "BinaryConnectAffine_3/bicon_affine/Wb"
    input: "BinaryConnectAffine_3/bicon_affine/b"
    output: "BinaryConnectAffine_3"
    binary_connect_affine_param {
      base_axis: 1
    }
  }
  function {
    name: "BatchNormalization_3"
    type: "BatchNormalization"
    input: "BinaryConnectAffine_3"
    input: "BatchNormalization_3/bn/beta"
    input: "BatchNormalization_3/bn/gamma"
    input: "BatchNormalization_3/bn/mean"
    input: "BatchNormalization_3/bn/var"
    output: "BatchNormalization_3"
    batch_normalization_param {
      axes: 1
      decay_rate: 0.5
      eps: 0.01
      batch_stat: False
    }
  }
  function {
    name: "BinarySigmoid_3"
    type: "BinarySigmoid"
    input: "BatchNormalization_3"
    output: "BinarySigmoid_3"
  }
  function {
    name: "BinaryConnectAffine_4"
    type: "BinaryConnectAffine"
    input: "BinarySigmoid_3"
    input: "BinaryConnectAffine_4/bicon_affine/W"
    input: "BinaryConnectAffine_4/bicon_affine/Wb"
    input: "BinaryConnectAffine_4/bicon_affine/b"
    output: "BinaryConnectAffine_4"
    binary_connect_affine_param {
      base_axis: 1
    }
  }
  function {
    name: "BatchNormalization_4"
    type: "BatchNormalization"
    input: "BinaryConnectAffine_4"
    input: "BatchNormalization_4/bn/beta"
    input: "BatchNormalization_4/bn/gamma"
    input: "BatchNormalization_4/bn/mean"
    input: "BatchNormalization_4/bn/var"
    output: "BatchNormalization_4"
    batch_normalization_param {
      axes: 1
      decay_rate: 0.5
      eps: 0.01
      batch_stat: False
    }
  }
  function {
    name: "Softmax"
    type: "Softmax"
    input: "BatchNormalization_4"
    output: "Softmax"
    softmax_param {
      axis: 1
    }
  }
}
dataset {
  name: "Training"
  uri: "C:\\Utils\\neural_network_console_internal_v120\\samples\\sample_dataset\\mnist\\mnist_training.csv"
  cache_dir: "C:\\Utils\\neural_network_console_internal_v120\\samples\\sample_dataset\\mnist\\mnist_training.cache"
  overwrite_cache: True
  create_cache_explicitly: True
  shuffle: true
  no_image_normalization: False
  batch_size: 64
}
dataset {
  name: "Validation"
  uri: "C:\\Utils\\neural_network_console_internal_v120\\samples\\sample_dataset\\mnist\\mnist_test.csv"
  cache_dir: "C:\\Utils\\neural_network_console_internal_v120\\samples\\sample_dataset\\mnist\\mnist_test.cache"
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
      alpha: 0.0001
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
    variable_name: "CategoricalCrossEntropy_T"
    data_name: "y"
  }
  loss_variable {
    variable_name: "CategoricalCrossEntropy"
  }
  parameter_variable {
    variable_name: "BinaryConnectAffine/bicon_affine/W"
    learning_rate_multiplier: 1
  }
  parameter_variable {
    variable_name: "BinaryConnectAffine/bicon_affine/Wb"
    learning_rate_multiplier: 0
  }
  parameter_variable {
    variable_name: "BinaryConnectAffine/bicon_affine/b"
    learning_rate_multiplier: 1
  }
  parameter_variable {
    variable_name: "BatchNormalization/bn/beta"
    learning_rate_multiplier: 1
  }
  parameter_variable {
    variable_name: "BatchNormalization/bn/gamma"
    learning_rate_multiplier: 1
  }
  parameter_variable {
    variable_name: "BatchNormalization/bn/mean"
    learning_rate_multiplier: 0
  }
  parameter_variable {
    variable_name: "BatchNormalization/bn/var"
    learning_rate_multiplier: 0
  }
  parameter_variable {
    variable_name: "BinaryConnectAffine_2/bicon_affine/W"
    learning_rate_multiplier: 1
  }
  parameter_variable {
    variable_name: "BinaryConnectAffine_2/bicon_affine/Wb"
    learning_rate_multiplier: 0
  }
  parameter_variable {
    variable_name: "BinaryConnectAffine_2/bicon_affine/b"
    learning_rate_multiplier: 1
  }
  parameter_variable {
    variable_name: "BatchNormalization_2/bn/beta"
    learning_rate_multiplier: 1
  }
  parameter_variable {
    variable_name: "BatchNormalization_2/bn/gamma"
    learning_rate_multiplier: 1
  }
  parameter_variable {
    variable_name: "BatchNormalization_2/bn/mean"
    learning_rate_multiplier: 0
  }
  parameter_variable {
    variable_name: "BatchNormalization_2/bn/var"
    learning_rate_multiplier: 0
  }
  parameter_variable {
    variable_name: "BinaryConnectAffine_3/bicon_affine/W"
    learning_rate_multiplier: 1
  }
  parameter_variable {
    variable_name: "BinaryConnectAffine_3/bicon_affine/Wb"
    learning_rate_multiplier: 0
  }
  parameter_variable {
    variable_name: "BinaryConnectAffine_3/bicon_affine/b"
    learning_rate_multiplier: 1
  }
  parameter_variable {
    variable_name: "BatchNormalization_3/bn/beta"
    learning_rate_multiplier: 1
  }
  parameter_variable {
    variable_name: "BatchNormalization_3/bn/gamma"
    learning_rate_multiplier: 1
  }
  parameter_variable {
    variable_name: "BatchNormalization_3/bn/mean"
    learning_rate_multiplier: 0
  }
  parameter_variable {
    variable_name: "BatchNormalization_3/bn/var"
    learning_rate_multiplier: 0
  }
  parameter_variable {
    variable_name: "BinaryConnectAffine_4/bicon_affine/W"
    learning_rate_multiplier: 1
  }
  parameter_variable {
    variable_name: "BinaryConnectAffine_4/bicon_affine/Wb"
    learning_rate_multiplier: 0
  }
  parameter_variable {
    variable_name: "BinaryConnectAffine_4/bicon_affine/b"
    learning_rate_multiplier: 1
  }
  parameter_variable {
    variable_name: "BatchNormalization_4/bn/beta"
    learning_rate_multiplier: 1
  }
  parameter_variable {
    variable_name: "BatchNormalization_4/bn/gamma"
    learning_rate_multiplier: 1
  }
  parameter_variable {
    variable_name: "BatchNormalization_4/bn/mean"
    learning_rate_multiplier: 0
  }
  parameter_variable {
    variable_name: "BatchNormalization_4/bn/var"
    learning_rate_multiplier: 0
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
    variable_name: "CategoricalCrossEntropy_T"
    data_name: "y"
  }
  monitor_variable {
    type: "Error"
    variable_name: "CategoricalCrossEntropy"
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
    variable_name: "CategoricalCrossEntropy_T"
    data_name: "y"
  }
  monitor_variable {
    type: "Error"
    variable_name: "CategoricalCrossEntropy"
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
    variable_name: "Softmax"
    data_name: "y'"
  }
  parameter_variable {
    variable_name: "BinaryConnectAffine/bicon_affine/W"
  }
  parameter_variable {
    variable_name: "BinaryConnectAffine/bicon_affine/Wb"
  }
  parameter_variable {
    variable_name: "BinaryConnectAffine/bicon_affine/b"
  }
  parameter_variable {
    variable_name: "BatchNormalization/bn/beta"
  }
  parameter_variable {
    variable_name: "BatchNormalization/bn/gamma"
  }
  parameter_variable {
    variable_name: "BatchNormalization/bn/mean"
  }
  parameter_variable {
    variable_name: "BatchNormalization/bn/var"
  }
  parameter_variable {
    variable_name: "BinaryConnectAffine_2/bicon_affine/W"
  }
  parameter_variable {
    variable_name: "BinaryConnectAffine_2/bicon_affine/Wb"
  }
  parameter_variable {
    variable_name: "BinaryConnectAffine_2/bicon_affine/b"
  }
  parameter_variable {
    variable_name: "BatchNormalization_2/bn/beta"
  }
  parameter_variable {
    variable_name: "BatchNormalization_2/bn/gamma"
  }
  parameter_variable {
    variable_name: "BatchNormalization_2/bn/mean"
  }
  parameter_variable {
    variable_name: "BatchNormalization_2/bn/var"
  }
  parameter_variable {
    variable_name: "BinaryConnectAffine_3/bicon_affine/W"
  }
  parameter_variable {
    variable_name: "BinaryConnectAffine_3/bicon_affine/Wb"
  }
  parameter_variable {
    variable_name: "BinaryConnectAffine_3/bicon_affine/b"
  }
  parameter_variable {
    variable_name: "BatchNormalization_3/bn/beta"
  }
  parameter_variable {
    variable_name: "BatchNormalization_3/bn/gamma"
  }
  parameter_variable {
    variable_name: "BatchNormalization_3/bn/mean"
  }
  parameter_variable {
    variable_name: "BatchNormalization_3/bn/var"
  }
  parameter_variable {
    variable_name: "BinaryConnectAffine_4/bicon_affine/W"
  }
  parameter_variable {
    variable_name: "BinaryConnectAffine_4/bicon_affine/Wb"
  }
  parameter_variable {
    variable_name: "BinaryConnectAffine_4/bicon_affine/b"
  }
  parameter_variable {
    variable_name: "BatchNormalization_4/bn/beta"
  }
  parameter_variable {
    variable_name: "BatchNormalization_4/bn/gamma"
  }
  parameter_variable {
    variable_name: "BatchNormalization_4/bn/mean"
  }
  parameter_variable {
    variable_name: "BatchNormalization_4/bn/var"
  }
}
'''
# The nntxt is generated from /home/woody/aid/nnabla-sample-data/sample_project/tutorial/basics/12_residual_learning.files/20170804_140212/net.nntxt.
N00000019 = r'''global_config {
  default_context {
    backend: "cpu|cuda"
    array_class: "CudaArray"
    device_id: "0"
    compute_backend: "default|cudnn"
  }
}
training_config {
  max_epoch: 10
  iter_per_epoch: 937
  save_best: true
}
network {
  name: "Main"
  batch_size: 64
  repeat_info {
    id: "RepeatStart"
    times: 2
  }
  repeat_info {
    id: "RepeatStart_2"
    times: 2
  }
  variable {
    name: "Input"
    type: "Buffer"
    shape: { dim:-1 dim: 1 dim: 28 dim: 28 }
  }
  variable {
    name: "Convolution"
    type: "Buffer"
    shape: { dim:-1 dim: 64 dim: 32 dim: 32 }
  }
  variable {
    name: "BatchNormalization"
    type: "Buffer"
    shape: { dim:-1 dim: 64 dim: 32 dim: 32 }
  }
  variable {
    name: "ReLU"
    type: "Buffer"
    shape: { dim:-1 dim: 64 dim: 32 dim: 32 }
  }
  variable {
    name: "Convolution_2"
    type: "Buffer"
    shape: { dim:-1 dim: 64 dim: 32 dim: 32 }
  }
  variable {
    name: "BatchNormalization_2"
    type: "Buffer"
    shape: { dim:-1 dim: 64 dim: 32 dim: 32 }
  }
  variable {
    name: "ReLU_2"
    type: "Buffer"
    shape: { dim:-1 dim: 64 dim: 32 dim: 32 }
  }
  variable {
    name: "Convolution_3"
    type: "Buffer"
    shape: { dim:-1 dim: 64 dim: 32 dim: 32 }
  }
  variable {
    name: "BatchNormalization_3"
    type: "Buffer"
    shape: { dim:-1 dim: 64 dim: 32 dim: 32 }
  }
  variable {
    name: "ReLU_3"
    type: "Buffer"
    shape: { dim:-1 dim: 64 dim: 32 dim: 32 }
  }
  variable {
    name: "Convolution_4"
    type: "Buffer"
    shape: { dim:-1 dim: 64 dim: 32 dim: 32 }
  }
  variable {
    name: "BatchNormalization_4"
    type: "Buffer"
    shape: { dim:-1 dim: 64 dim: 32 dim: 32 }
  }
  variable {
    name: "Add2"
    type: "Buffer"
    shape: { dim:-1 dim: 64 dim: 32 dim: 32 }
  }
  variable {
    name: "ReLU_4"
    type: "Buffer"
    shape: { dim:-1 dim: 64 dim: 32 dim: 32 }
  }
  variable {
    name: "MaxPooling"
    type: "Buffer"
    shape: { dim:-1 dim: 64 dim: 16 dim: 16 }
  }
  variable {
    name: "RepeatStart"
    type: "Buffer"
    repeat_id: "RepeatStart"
    shape: { dim:-1 dim: 64 dim: 16 dim: 16 }
  }
  variable {
    name: "Convolution_5"
    type: "Buffer"
    repeat_id: "RepeatStart"
    shape: { dim:-1 dim: 64 dim: 16 dim: 16 }
  }
  variable {
    name: "BatchNormalization_5"
    type: "Buffer"
    repeat_id: "RepeatStart"
    shape: { dim:-1 dim: 64 dim: 16 dim: 16 }
  }
  variable {
    name: "ReLU_5"
    type: "Buffer"
    repeat_id: "RepeatStart"
    shape: { dim:-1 dim: 64 dim: 16 dim: 16 }
  }
  variable {
    name: "Convolution_6"
    type: "Buffer"
    repeat_id: "RepeatStart"
    shape: { dim:-1 dim: 64 dim: 16 dim: 16 }
  }
  variable {
    name: "BatchNormalization_6"
    type: "Buffer"
    repeat_id: "RepeatStart"
    shape: { dim:-1 dim: 64 dim: 16 dim: 16 }
  }
  variable {
    name: "ReLU_6"
    type: "Buffer"
    repeat_id: "RepeatStart"
    shape: { dim:-1 dim: 64 dim: 16 dim: 16 }
  }
  variable {
    name: "Convolution_7"
    type: "Buffer"
    repeat_id: "RepeatStart"
    shape: { dim:-1 dim: 64 dim: 16 dim: 16 }
  }
  variable {
    name: "BatchNormalization_7"
    type: "Buffer"
    repeat_id: "RepeatStart"
    shape: { dim:-1 dim: 64 dim: 16 dim: 16 }
  }
  variable {
    name: "Add2_2"
    type: "Buffer"
    repeat_id: "RepeatStart"
    shape: { dim:-1 dim: 64 dim: 16 dim: 16 }
  }
  variable {
    name: "ReLU_7"
    type: "Buffer"
    repeat_id: "RepeatStart"
    shape: { dim:-1 dim: 64 dim: 16 dim: 16 }
  }
  variable {
    name: "RepeatEnd"
    type: "Buffer"
    shape: { dim:-1 dim: 64 dim: 16 dim: 16 }
  }
  variable {
    name: "MaxPooling_2"
    type: "Buffer"
    shape: { dim:-1 dim: 64 dim: 8 dim: 8 }
  }
  variable {
    name: "RepeatStart_2"
    type: "Buffer"
    repeat_id: "RepeatStart_2"
    shape: { dim:-1 dim: 64 dim: 8 dim: 8 }
  }
  variable {
    name: "Convolution_8"
    type: "Buffer"
    repeat_id: "RepeatStart_2"
    shape: { dim:-1 dim: 64 dim: 8 dim: 8 }
  }
  variable {
    name: "BatchNormalization_8"
    type: "Buffer"
    repeat_id: "RepeatStart_2"
    shape: { dim:-1 dim: 64 dim: 8 dim: 8 }
  }
  variable {
    name: "ReLU_8"
    type: "Buffer"
    repeat_id: "RepeatStart_2"
    shape: { dim:-1 dim: 64 dim: 8 dim: 8 }
  }
  variable {
    name: "Convolution_9"
    type: "Buffer"
    repeat_id: "RepeatStart_2"
    shape: { dim:-1 dim: 64 dim: 8 dim: 8 }
  }
  variable {
    name: "BatchNormalization_9"
    type: "Buffer"
    repeat_id: "RepeatStart_2"
    shape: { dim:-1 dim: 64 dim: 8 dim: 8 }
  }
  variable {
    name: "ReLU_9"
    type: "Buffer"
    repeat_id: "RepeatStart_2"
    shape: { dim:-1 dim: 64 dim: 8 dim: 8 }
  }
  variable {
    name: "Convolution_10"
    type: "Buffer"
    repeat_id: "RepeatStart_2"
    shape: { dim:-1 dim: 64 dim: 8 dim: 8 }
  }
  variable {
    name: "BatchNormalization_10"
    type: "Buffer"
    repeat_id: "RepeatStart_2"
    shape: { dim:-1 dim: 64 dim: 8 dim: 8 }
  }
  variable {
    name: "Add2_3"
    type: "Buffer"
    repeat_id: "RepeatStart_2"
    shape: { dim:-1 dim: 64 dim: 8 dim: 8 }
  }
  variable {
    name: "ReLU_10"
    type: "Buffer"
    repeat_id: "RepeatStart_2"
    shape: { dim:-1 dim: 64 dim: 8 dim: 8 }
  }
  variable {
    name: "RepeatEnd_2"
    type: "Buffer"
    shape: { dim:-1 dim: 64 dim: 8 dim: 8 }
  }
  variable {
    name: "MaxPooling_3"
    type: "Buffer"
    shape: { dim:-1 dim: 64 dim: 4 dim: 4 }
  }
  variable {
    name: "Convolution_11"
    type: "Buffer"
    shape: { dim:-1 dim: 64 dim: 4 dim: 4 }
  }
  variable {
    name: "BatchNormalization_11"
    type: "Buffer"
    shape: { dim:-1 dim: 64 dim: 4 dim: 4 }
  }
  variable {
    name: "ReLU_11"
    type: "Buffer"
    shape: { dim:-1 dim: 64 dim: 4 dim: 4 }
  }
  variable {
    name: "Convolution_12"
    type: "Buffer"
    shape: { dim:-1 dim: 64 dim: 4 dim: 4 }
  }
  variable {
    name: "BatchNormalization_12"
    type: "Buffer"
    shape: { dim:-1 dim: 64 dim: 4 dim: 4 }
  }
  variable {
    name: "ReLU_12"
    type: "Buffer"
    shape: { dim:-1 dim: 64 dim: 4 dim: 4 }
  }
  variable {
    name: "Convolution_13"
    type: "Buffer"
    shape: { dim:-1 dim: 64 dim: 4 dim: 4 }
  }
  variable {
    name: "BatchNormalization_13"
    type: "Buffer"
    shape: { dim:-1 dim: 64 dim: 4 dim: 4 }
  }
  variable {
    name: "Add2_4"
    type: "Buffer"
    shape: { dim:-1 dim: 64 dim: 4 dim: 4 }
  }
  variable {
    name: "ReLU_13"
    type: "Buffer"
    shape: { dim:-1 dim: 64 dim: 4 dim: 4 }
  }
  variable {
    name: "AveragePooling"
    type: "Buffer"
    shape: { dim:-1 dim: 64 dim: 1 dim: 1 }
  }
  variable {
    name: "Affine"
    type: "Buffer"
    shape: { dim:-1 dim: 10 }
  }
  variable {
    name: "Softmax"
    type: "Buffer"
    shape: { dim:-1 dim: 10 }
  }
  variable {
    name: "CategoricalCrossEntropy"
    type: "Buffer"
    shape: { dim:-1 dim: 1 }
  }
  variable {
    name: "Convolution/conv/W"
    type: "Parameter"
    shape: { dim: 64 dim: 1 dim: 5 dim: 5 }
    initializer {
      type: "NormalConvolutionGlorot"
      multiplier: 1.0
    }
  }
  variable {
    name: "BatchNormalization/bn/beta"
    type: "Parameter"
    shape: { dim: 1 dim: 64 dim: 1 dim: 1 }
    initializer {
      type: "Constant"
      multiplier: 0.0
    }
  }
  variable {
    name: "BatchNormalization/bn/gamma"
    type: "Parameter"
    shape: { dim: 1 dim: 64 dim: 1 dim: 1 }
    initializer {
      type: "Constant"
      multiplier: 1.0
    }
  }
  variable {
    name: "BatchNormalization/bn/mean"
    type: "Parameter"
    shape: { dim: 1 dim: 64 dim: 1 dim: 1 }
    initializer {
      type: "Constant"
      multiplier: 0.0
    }
  }
  variable {
    name: "BatchNormalization/bn/var"
    type: "Parameter"
    shape: { dim: 1 dim: 64 dim: 1 dim: 1 }
    initializer {
      type: "Constant"
      multiplier: 0.0
    }
  }
  variable {
    name: "Convolution_2/conv/W"
    type: "Parameter"
    shape: { dim: 64 dim: 64 dim: 3 dim: 3 }
    initializer {
      type: "NormalConvolutionGlorot"
      multiplier: 1.0
    }
  }
  variable {
    name: "BatchNormalization_2/bn/beta"
    type: "Parameter"
    shape: { dim: 1 dim: 64 dim: 1 dim: 1 }
    initializer {
      type: "Constant"
      multiplier: 0.0
    }
  }
  variable {
    name: "BatchNormalization_2/bn/gamma"
    type: "Parameter"
    shape: { dim: 1 dim: 64 dim: 1 dim: 1 }
    initializer {
      type: "Constant"
      multiplier: 1.0
    }
  }
  variable {
    name: "BatchNormalization_2/bn/mean"
    type: "Parameter"
    shape: { dim: 1 dim: 64 dim: 1 dim: 1 }
    initializer {
      type: "Constant"
      multiplier: 0.0
    }
  }
  variable {
    name: "BatchNormalization_2/bn/var"
    type: "Parameter"
    shape: { dim: 1 dim: 64 dim: 1 dim: 1 }
    initializer {
      type: "Constant"
      multiplier: 0.0
    }
  }
  variable {
    name: "Convolution_3/conv/W"
    type: "Parameter"
    shape: { dim: 64 dim: 64 dim: 1 dim: 1 }
    initializer {
      type: "NormalConvolutionGlorot"
      multiplier: 1.0
    }
  }
  variable {
    name: "BatchNormalization_3/bn/beta"
    type: "Parameter"
    shape: { dim: 1 dim: 64 dim: 1 dim: 1 }
    initializer {
      type: "Constant"
      multiplier: 0.0
    }
  }
  variable {
    name: "BatchNormalization_3/bn/gamma"
    type: "Parameter"
    shape: { dim: 1 dim: 64 dim: 1 dim: 1 }
    initializer {
      type: "Constant"
      multiplier: 1.0
    }
  }
  variable {
    name: "BatchNormalization_3/bn/mean"
    type: "Parameter"
    shape: { dim: 1 dim: 64 dim: 1 dim: 1 }
    initializer {
      type: "Constant"
      multiplier: 0.0
    }
  }
  variable {
    name: "BatchNormalization_3/bn/var"
    type: "Parameter"
    shape: { dim: 1 dim: 64 dim: 1 dim: 1 }
    initializer {
      type: "Constant"
      multiplier: 0.0
    }
  }
  variable {
    name: "Convolution_4/conv/W"
    type: "Parameter"
    shape: { dim: 64 dim: 64 dim: 3 dim: 3 }
    initializer {
      type: "NormalConvolutionGlorot"
      multiplier: 1.0
    }
  }
  variable {
    name: "BatchNormalization_4/bn/beta"
    type: "Parameter"
    shape: { dim: 1 dim: 64 dim: 1 dim: 1 }
    initializer {
      type: "Constant"
      multiplier: 0.0
    }
  }
  variable {
    name: "BatchNormalization_4/bn/gamma"
    type: "Parameter"
    shape: { dim: 1 dim: 64 dim: 1 dim: 1 }
    initializer {
      type: "Constant"
      multiplier: 1.0
    }
  }
  variable {
    name: "BatchNormalization_4/bn/mean"
    type: "Parameter"
    shape: { dim: 1 dim: 64 dim: 1 dim: 1 }
    initializer {
      type: "Constant"
      multiplier: 0.0
    }
  }
  variable {
    name: "BatchNormalization_4/bn/var"
    type: "Parameter"
    shape: { dim: 1 dim: 64 dim: 1 dim: 1 }
    initializer {
      type: "Constant"
      multiplier: 0.0
    }
  }
  variable {
    name: "Convolution_5/conv/W"
    type: "Parameter"
    repeat_id: "RepeatStart"
    shape: { dim: 64 dim: 64 dim: 3 dim: 3 }
    initializer {
      type: "NormalConvolutionGlorot"
      multiplier: 1.0
    }
  }
  variable {
    name: "BatchNormalization_5/bn/beta"
    type: "Parameter"
    repeat_id: "RepeatStart"
    shape: { dim: 1 dim: 64 dim: 1 dim: 1 }
    initializer {
      type: "Constant"
      multiplier: 0.0
    }
  }
  variable {
    name: "BatchNormalization_5/bn/gamma"
    type: "Parameter"
    repeat_id: "RepeatStart"
    shape: { dim: 1 dim: 64 dim: 1 dim: 1 }
    initializer {
      type: "Constant"
      multiplier: 1.0
    }
  }
  variable {
    name: "BatchNormalization_5/bn/mean"
    type: "Parameter"
    repeat_id: "RepeatStart"
    shape: { dim: 1 dim: 64 dim: 1 dim: 1 }
    initializer {
      type: "Constant"
      multiplier: 0.0
    }
  }
  variable {
    name: "BatchNormalization_5/bn/var"
    type: "Parameter"
    repeat_id: "RepeatStart"
    shape: { dim: 1 dim: 64 dim: 1 dim: 1 }
    initializer {
      type: "Constant"
      multiplier: 0.0
    }
  }
  variable {
    name: "Convolution_6/conv/W"
    type: "Parameter"
    repeat_id: "RepeatStart"
    shape: { dim: 64 dim: 64 dim: 1 dim: 1 }
    initializer {
      type: "NormalConvolutionGlorot"
      multiplier: 1.0
    }
  }
  variable {
    name: "BatchNormalization_6/bn/beta"
    type: "Parameter"
    repeat_id: "RepeatStart"
    shape: { dim: 1 dim: 64 dim: 1 dim: 1 }
    initializer {
      type: "Constant"
      multiplier: 0.0
    }
  }
  variable {
    name: "BatchNormalization_6/bn/gamma"
    type: "Parameter"
    repeat_id: "RepeatStart"
    shape: { dim: 1 dim: 64 dim: 1 dim: 1 }
    initializer {
      type: "Constant"
      multiplier: 1.0
    }
  }
  variable {
    name: "BatchNormalization_6/bn/mean"
    type: "Parameter"
    repeat_id: "RepeatStart"
    shape: { dim: 1 dim: 64 dim: 1 dim: 1 }
    initializer {
      type: "Constant"
      multiplier: 0.0
    }
  }
  variable {
    name: "BatchNormalization_6/bn/var"
    type: "Parameter"
    repeat_id: "RepeatStart"
    shape: { dim: 1 dim: 64 dim: 1 dim: 1 }
    initializer {
      type: "Constant"
      multiplier: 0.0
    }
  }
  variable {
    name: "Convolution_7/conv/W"
    type: "Parameter"
    repeat_id: "RepeatStart"
    shape: { dim: 64 dim: 64 dim: 3 dim: 3 }
    initializer {
      type: "NormalConvolutionGlorot"
      multiplier: 1.0
    }
  }
  variable {
    name: "BatchNormalization_7/bn/beta"
    type: "Parameter"
    repeat_id: "RepeatStart"
    shape: { dim: 1 dim: 64 dim: 1 dim: 1 }
    initializer {
      type: "Constant"
      multiplier: 0.0
    }
  }
  variable {
    name: "BatchNormalization_7/bn/gamma"
    type: "Parameter"
    repeat_id: "RepeatStart"
    shape: { dim: 1 dim: 64 dim: 1 dim: 1 }
    initializer {
      type: "Constant"
      multiplier: 1.0
    }
  }
  variable {
    name: "BatchNormalization_7/bn/mean"
    type: "Parameter"
    repeat_id: "RepeatStart"
    shape: { dim: 1 dim: 64 dim: 1 dim: 1 }
    initializer {
      type: "Constant"
      multiplier: 0.0
    }
  }
  variable {
    name: "BatchNormalization_7/bn/var"
    type: "Parameter"
    repeat_id: "RepeatStart"
    shape: { dim: 1 dim: 64 dim: 1 dim: 1 }
    initializer {
      type: "Constant"
      multiplier: 0.0
    }
  }
  variable {
    name: "Convolution_8/conv/W"
    type: "Parameter"
    repeat_id: "RepeatStart_2"
    shape: { dim: 64 dim: 64 dim: 3 dim: 3 }
    initializer {
      type: "NormalConvolutionGlorot"
      multiplier: 1.0
    }
  }
  variable {
    name: "BatchNormalization_8/bn/beta"
    type: "Parameter"
    repeat_id: "RepeatStart_2"
    shape: { dim: 1 dim: 64 dim: 1 dim: 1 }
    initializer {
      type: "Constant"
      multiplier: 0.0
    }
  }
  variable {
    name: "BatchNormalization_8/bn/gamma"
    type: "Parameter"
    repeat_id: "RepeatStart_2"
    shape: { dim: 1 dim: 64 dim: 1 dim: 1 }
    initializer {
      type: "Constant"
      multiplier: 1.0
    }
  }
  variable {
    name: "BatchNormalization_8/bn/mean"
    type: "Parameter"
    repeat_id: "RepeatStart_2"
    shape: { dim: 1 dim: 64 dim: 1 dim: 1 }
    initializer {
      type: "Constant"
      multiplier: 0.0
    }
  }
  variable {
    name: "BatchNormalization_8/bn/var"
    type: "Parameter"
    repeat_id: "RepeatStart_2"
    shape: { dim: 1 dim: 64 dim: 1 dim: 1 }
    initializer {
      type: "Constant"
      multiplier: 0.0
    }
  }
  variable {
    name: "Convolution_9/conv/W"
    type: "Parameter"
    repeat_id: "RepeatStart_2"
    shape: { dim: 64 dim: 64 dim: 1 dim: 1 }
    initializer {
      type: "NormalConvolutionGlorot"
      multiplier: 1.0
    }
  }
  variable {
    name: "BatchNormalization_9/bn/beta"
    type: "Parameter"
    repeat_id: "RepeatStart_2"
    shape: { dim: 1 dim: 64 dim: 1 dim: 1 }
    initializer {
      type: "Constant"
      multiplier: 0.0
    }
  }
  variable {
    name: "BatchNormalization_9/bn/gamma"
    type: "Parameter"
    repeat_id: "RepeatStart_2"
    shape: { dim: 1 dim: 64 dim: 1 dim: 1 }
    initializer {
      type: "Constant"
      multiplier: 1.0
    }
  }
  variable {
    name: "BatchNormalization_9/bn/mean"
    type: "Parameter"
    repeat_id: "RepeatStart_2"
    shape: { dim: 1 dim: 64 dim: 1 dim: 1 }
    initializer {
      type: "Constant"
      multiplier: 0.0
    }
  }
  variable {
    name: "BatchNormalization_9/bn/var"
    type: "Parameter"
    repeat_id: "RepeatStart_2"
    shape: { dim: 1 dim: 64 dim: 1 dim: 1 }
    initializer {
      type: "Constant"
      multiplier: 0.0
    }
  }
  variable {
    name: "Convolution_10/conv/W"
    type: "Parameter"
    repeat_id: "RepeatStart_2"
    shape: { dim: 64 dim: 64 dim: 3 dim: 3 }
    initializer {
      type: "NormalConvolutionGlorot"
      multiplier: 1.0
    }
  }
  variable {
    name: "BatchNormalization_10/bn/beta"
    type: "Parameter"
    repeat_id: "RepeatStart_2"
    shape: { dim: 1 dim: 64 dim: 1 dim: 1 }
    initializer {
      type: "Constant"
      multiplier: 0.0
    }
  }
  variable {
    name: "BatchNormalization_10/bn/gamma"
    type: "Parameter"
    repeat_id: "RepeatStart_2"
    shape: { dim: 1 dim: 64 dim: 1 dim: 1 }
    initializer {
      type: "Constant"
      multiplier: 1.0
    }
  }
  variable {
    name: "BatchNormalization_10/bn/mean"
    type: "Parameter"
    repeat_id: "RepeatStart_2"
    shape: { dim: 1 dim: 64 dim: 1 dim: 1 }
    initializer {
      type: "Constant"
      multiplier: 0.0
    }
  }
  variable {
    name: "BatchNormalization_10/bn/var"
    type: "Parameter"
    repeat_id: "RepeatStart_2"
    shape: { dim: 1 dim: 64 dim: 1 dim: 1 }
    initializer {
      type: "Constant"
      multiplier: 0.0
    }
  }
  variable {
    name: "Convolution_11/conv/W"
    type: "Parameter"
    shape: { dim: 64 dim: 64 dim: 3 dim: 3 }
    initializer {
      type: "NormalConvolutionGlorot"
      multiplier: 1.0
    }
  }
  variable {
    name: "BatchNormalization_11/bn/beta"
    type: "Parameter"
    shape: { dim: 1 dim: 64 dim: 1 dim: 1 }
    initializer {
      type: "Constant"
      multiplier: 0.0
    }
  }
  variable {
    name: "BatchNormalization_11/bn/gamma"
    type: "Parameter"
    shape: { dim: 1 dim: 64 dim: 1 dim: 1 }
    initializer {
      type: "Constant"
      multiplier: 1.0
    }
  }
  variable {
    name: "BatchNormalization_11/bn/mean"
    type: "Parameter"
    shape: { dim: 1 dim: 64 dim: 1 dim: 1 }
    initializer {
      type: "Constant"
      multiplier: 0.0
    }
  }
  variable {
    name: "BatchNormalization_11/bn/var"
    type: "Parameter"
    shape: { dim: 1 dim: 64 dim: 1 dim: 1 }
    initializer {
      type: "Constant"
      multiplier: 0.0
    }
  }
  variable {
    name: "Convolution_12/conv/W"
    type: "Parameter"
    shape: { dim: 64 dim: 64 dim: 1 dim: 1 }
    initializer {
      type: "NormalConvolutionGlorot"
      multiplier: 1.0
    }
  }
  variable {
    name: "BatchNormalization_12/bn/beta"
    type: "Parameter"
    shape: { dim: 1 dim: 64 dim: 1 dim: 1 }
    initializer {
      type: "Constant"
      multiplier: 0.0
    }
  }
  variable {
    name: "BatchNormalization_12/bn/gamma"
    type: "Parameter"
    shape: { dim: 1 dim: 64 dim: 1 dim: 1 }
    initializer {
      type: "Constant"
      multiplier: 1.0
    }
  }
  variable {
    name: "BatchNormalization_12/bn/mean"
    type: "Parameter"
    shape: { dim: 1 dim: 64 dim: 1 dim: 1 }
    initializer {
      type: "Constant"
      multiplier: 0.0
    }
  }
  variable {
    name: "BatchNormalization_12/bn/var"
    type: "Parameter"
    shape: { dim: 1 dim: 64 dim: 1 dim: 1 }
    initializer {
      type: "Constant"
      multiplier: 0.0
    }
  }
  variable {
    name: "Convolution_13/conv/W"
    type: "Parameter"
    shape: { dim: 64 dim: 64 dim: 3 dim: 3 }
    initializer {
      type: "NormalConvolutionGlorot"
      multiplier: 1.0
    }
  }
  variable {
    name: "BatchNormalization_13/bn/beta"
    type: "Parameter"
    shape: { dim: 1 dim: 64 dim: 1 dim: 1 }
    initializer {
      type: "Constant"
      multiplier: 0.0
    }
  }
  variable {
    name: "BatchNormalization_13/bn/gamma"
    type: "Parameter"
    shape: { dim: 1 dim: 64 dim: 1 dim: 1 }
    initializer {
      type: "Constant"
      multiplier: 1.0
    }
  }
  variable {
    name: "BatchNormalization_13/bn/mean"
    type: "Parameter"
    shape: { dim: 1 dim: 64 dim: 1 dim: 1 }
    initializer {
      type: "Constant"
      multiplier: 0.0
    }
  }
  variable {
    name: "BatchNormalization_13/bn/var"
    type: "Parameter"
    shape: { dim: 1 dim: 64 dim: 1 dim: 1 }
    initializer {
      type: "Constant"
      multiplier: 0.0
    }
  }
  variable {
    name: "Affine/affine/W"
    type: "Parameter"
    shape: { dim: 64 dim: 10 }
    initializer {
      type: "NormalAffineGlorot"
      multiplier: 1.0
    }
  }
  variable {
    name: "Affine/affine/b"
    type: "Parameter"
    shape: { dim: 10 }
    initializer {
      type: "Constant"
      multiplier: 0.0
    }
  }
  variable {
    name: "CategoricalCrossEntropy_T"
    type: "Buffer"
    shape: { dim:-1 dim: 1 }
  }
  function {
    name: "Convolution"
    type: "Convolution"
    input: "Input"
    input: "Convolution/conv/W"
    output: "Convolution"
    convolution_param {
      pad: { dim: 4 dim: 4 }
      stride: { dim: 1 dim: 1 }
      dilation: { dim: 1 dim: 1 }
      group: 1
      base_axis: 1
    }
  }
  function {
    name: "BatchNormalization"
    type: "BatchNormalization"
    input: "Convolution"
    input: "BatchNormalization/bn/beta"
    input: "BatchNormalization/bn/gamma"
    input: "BatchNormalization/bn/mean"
    input: "BatchNormalization/bn/var"
    output: "BatchNormalization"
    batch_normalization_param {
      axes: 1
      decay_rate: 0.5
      eps: 0.01
      batch_stat: true
    }
  }
  function {
    name: "ReLU"
    type: "ReLU"
    input: "BatchNormalization"
    output: "ReLU"
    relu_param {
      inplace: True
    }
  }
  function {
    name: "Convolution_2"
    type: "Convolution"
    input: "ReLU"
    input: "Convolution_2/conv/W"
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
    name: "BatchNormalization_2"
    type: "BatchNormalization"
    input: "Convolution_2"
    input: "BatchNormalization_2/bn/beta"
    input: "BatchNormalization_2/bn/gamma"
    input: "BatchNormalization_2/bn/mean"
    input: "BatchNormalization_2/bn/var"
    output: "BatchNormalization_2"
    batch_normalization_param {
      axes: 1
      decay_rate: 0.5
      eps: 0.01
      batch_stat: true
    }
  }
  function {
    name: "ReLU_2"
    type: "ReLU"
    input: "BatchNormalization_2"
    output: "ReLU_2"
    relu_param {
      inplace: True
    }
  }
  function {
    name: "Convolution_3"
    type: "Convolution"
    input: "ReLU_2"
    input: "Convolution_3/conv/W"
    output: "Convolution_3"
    convolution_param {
      pad: { dim: 0 dim: 0 }
      stride: { dim: 1 dim: 1 }
      dilation: { dim: 1 dim: 1 }
      group: 1
      base_axis: 1
    }
  }
  function {
    name: "BatchNormalization_3"
    type: "BatchNormalization"
    input: "Convolution_3"
    input: "BatchNormalization_3/bn/beta"
    input: "BatchNormalization_3/bn/gamma"
    input: "BatchNormalization_3/bn/mean"
    input: "BatchNormalization_3/bn/var"
    output: "BatchNormalization_3"
    batch_normalization_param {
      axes: 1
      decay_rate: 0.5
      eps: 0.01
      batch_stat: true
    }
  }
  function {
    name: "ReLU_3"
    type: "ReLU"
    input: "BatchNormalization_3"
    output: "ReLU_3"
    relu_param {
      inplace: True
    }
  }
  function {
    name: "Convolution_4"
    type: "Convolution"
    input: "ReLU_3"
    input: "Convolution_4/conv/W"
    output: "Convolution_4"
    convolution_param {
      pad: { dim: 1 dim: 1 }
      stride: { dim: 1 dim: 1 }
      dilation: { dim: 1 dim: 1 }
      group: 1
      base_axis: 1
    }
  }
  function {
    name: "BatchNormalization_4"
    type: "BatchNormalization"
    input: "Convolution_4"
    input: "BatchNormalization_4/bn/beta"
    input: "BatchNormalization_4/bn/gamma"
    input: "BatchNormalization_4/bn/mean"
    input: "BatchNormalization_4/bn/var"
    output: "BatchNormalization_4"
    batch_normalization_param {
      axes: 1
      decay_rate: 0.5
      eps: 0.01
      batch_stat: true
    }
  }
  function {
    name: "Add2"
    type: "Add2"
    input: "BatchNormalization_4"
    input: "ReLU"
    output: "Add2"
    add2_param {
      inplace: False
    }
  }
  function {
    name: "ReLU_4"
    type: "ReLU"
    input: "Add2"
    output: "ReLU_4"
    relu_param {
      inplace: True
    }
  }
  function {
    name: "MaxPooling"
    type: "MaxPooling"
    input: "ReLU_4"
    output: "MaxPooling"
    max_pooling_param {
      kernel: { dim: 2 dim: 2 }
      stride: { dim: 2 dim: 2 }
      ignore_border: true
      pad: { dim: 0 dim: 0 }
    }
  }
  function {
    name: "RepeatStart"
    type: "RepeatStart"
    repeat_id: "RepeatStart"
    input: "MaxPooling"
    input: "ReLU_7"
    output: "RepeatStart"
    repeat_param {
      repeat_id: "RepeatStart"
    }
  }
  function {
    name: "Convolution_5"
    type: "Convolution"
    repeat_id: "RepeatStart"
    input: "RepeatStart"
    input: "Convolution_5/conv/W"
    output: "Convolution_5"
    convolution_param {
      pad: { dim: 1 dim: 1 }
      stride: { dim: 1 dim: 1 }
      dilation: { dim: 1 dim: 1 }
      group: 1
      base_axis: 1
    }
  }
  function {
    name: "BatchNormalization_5"
    type: "BatchNormalization"
    repeat_id: "RepeatStart"
    input: "Convolution_5"
    input: "BatchNormalization_5/bn/beta"
    input: "BatchNormalization_5/bn/gamma"
    input: "BatchNormalization_5/bn/mean"
    input: "BatchNormalization_5/bn/var"
    output: "BatchNormalization_5"
    batch_normalization_param {
      axes: 1
      decay_rate: 0.5
      eps: 0.01
      batch_stat: true
    }
  }
  function {
    name: "ReLU_5"
    type: "ReLU"
    repeat_id: "RepeatStart"
    input: "BatchNormalization_5"
    output: "ReLU_5"
    relu_param {
      inplace: True
    }
  }
  function {
    name: "Convolution_6"
    type: "Convolution"
    repeat_id: "RepeatStart"
    input: "ReLU_5"
    input: "Convolution_6/conv/W"
    output: "Convolution_6"
    convolution_param {
      pad: { dim: 0 dim: 0 }
      stride: { dim: 1 dim: 1 }
      dilation: { dim: 1 dim: 1 }
      group: 1
      base_axis: 1
    }
  }
  function {
    name: "BatchNormalization_6"
    type: "BatchNormalization"
    repeat_id: "RepeatStart"
    input: "Convolution_6"
    input: "BatchNormalization_6/bn/beta"
    input: "BatchNormalization_6/bn/gamma"
    input: "BatchNormalization_6/bn/mean"
    input: "BatchNormalization_6/bn/var"
    output: "BatchNormalization_6"
    batch_normalization_param {
      axes: 1
      decay_rate: 0.5
      eps: 0.01
      batch_stat: true
    }
  }
  function {
    name: "ReLU_6"
    type: "ReLU"
    repeat_id: "RepeatStart"
    input: "BatchNormalization_6"
    output: "ReLU_6"
    relu_param {
      inplace: True
    }
  }
  function {
    name: "Convolution_7"
    type: "Convolution"
    repeat_id: "RepeatStart"
    input: "ReLU_6"
    input: "Convolution_7/conv/W"
    output: "Convolution_7"
    convolution_param {
      pad: { dim: 1 dim: 1 }
      stride: { dim: 1 dim: 1 }
      dilation: { dim: 1 dim: 1 }
      group: 1
      base_axis: 1
    }
  }
  function {
    name: "BatchNormalization_7"
    type: "BatchNormalization"
    repeat_id: "RepeatStart"
    input: "Convolution_7"
    input: "BatchNormalization_7/bn/beta"
    input: "BatchNormalization_7/bn/gamma"
    input: "BatchNormalization_7/bn/mean"
    input: "BatchNormalization_7/bn/var"
    output: "BatchNormalization_7"
    batch_normalization_param {
      axes: 1
      decay_rate: 0.5
      eps: 0.01
      batch_stat: true
    }
  }
  function {
    name: "Add2_2"
    type: "Add2"
    repeat_id: "RepeatStart"
    input: "BatchNormalization_7"
    input: "RepeatStart"
    output: "Add2_2"
    add2_param {
      inplace: False
    }
  }
  function {
    name: "ReLU_7"
    type: "ReLU"
    repeat_id: "RepeatStart"
    input: "Add2_2"
    output: "ReLU_7"
    relu_param {
      inplace: True
    }
  }
  function {
    name: "RepeatEnd"
    type: "RepeatEnd"
    input: "ReLU_7"
    output: "RepeatEnd"
    repeat_param {
      repeat_id: "RepeatStart"
      times: 2
    }
  }
  function {
    name: "MaxPooling_2"
    type: "MaxPooling"
    input: "RepeatEnd"
    output: "MaxPooling_2"
    max_pooling_param {
      kernel: { dim: 2 dim: 2 }
      stride: { dim: 2 dim: 2 }
      ignore_border: true
      pad: { dim: 0 dim: 0 }
    }
  }
  function {
    name: "RepeatStart_2"
    type: "RepeatStart"
    repeat_id: "RepeatStart_2"
    input: "MaxPooling_2"
    input: "ReLU_10"
    output: "RepeatStart_2"
    repeat_param {
      repeat_id: "RepeatStart_2"
    }
  }
  function {
    name: "Convolution_8"
    type: "Convolution"
    repeat_id: "RepeatStart_2"
    input: "RepeatStart_2"
    input: "Convolution_8/conv/W"
    output: "Convolution_8"
    convolution_param {
      pad: { dim: 1 dim: 1 }
      stride: { dim: 1 dim: 1 }
      dilation: { dim: 1 dim: 1 }
      group: 1
      base_axis: 1
    }
  }
  function {
    name: "BatchNormalization_8"
    type: "BatchNormalization"
    repeat_id: "RepeatStart_2"
    input: "Convolution_8"
    input: "BatchNormalization_8/bn/beta"
    input: "BatchNormalization_8/bn/gamma"
    input: "BatchNormalization_8/bn/mean"
    input: "BatchNormalization_8/bn/var"
    output: "BatchNormalization_8"
    batch_normalization_param {
      axes: 1
      decay_rate: 0.5
      eps: 0.01
      batch_stat: true
    }
  }
  function {
    name: "ReLU_8"
    type: "ReLU"
    repeat_id: "RepeatStart_2"
    input: "BatchNormalization_8"
    output: "ReLU_8"
    relu_param {
      inplace: True
    }
  }
  function {
    name: "Convolution_9"
    type: "Convolution"
    repeat_id: "RepeatStart_2"
    input: "ReLU_8"
    input: "Convolution_9/conv/W"
    output: "Convolution_9"
    convolution_param {
      pad: { dim: 0 dim: 0 }
      stride: { dim: 1 dim: 1 }
      dilation: { dim: 1 dim: 1 }
      group: 1
      base_axis: 1
    }
  }
  function {
    name: "BatchNormalization_9"
    type: "BatchNormalization"
    repeat_id: "RepeatStart_2"
    input: "Convolution_9"
    input: "BatchNormalization_9/bn/beta"
    input: "BatchNormalization_9/bn/gamma"
    input: "BatchNormalization_9/bn/mean"
    input: "BatchNormalization_9/bn/var"
    output: "BatchNormalization_9"
    batch_normalization_param {
      axes: 1
      decay_rate: 0.5
      eps: 0.01
      batch_stat: true
    }
  }
  function {
    name: "ReLU_9"
    type: "ReLU"
    repeat_id: "RepeatStart_2"
    input: "BatchNormalization_9"
    output: "ReLU_9"
    relu_param {
      inplace: True
    }
  }
  function {
    name: "Convolution_10"
    type: "Convolution"
    repeat_id: "RepeatStart_2"
    input: "ReLU_9"
    input: "Convolution_10/conv/W"
    output: "Convolution_10"
    convolution_param {
      pad: { dim: 1 dim: 1 }
      stride: { dim: 1 dim: 1 }
      dilation: { dim: 1 dim: 1 }
      group: 1
      base_axis: 1
    }
  }
  function {
    name: "BatchNormalization_10"
    type: "BatchNormalization"
    repeat_id: "RepeatStart_2"
    input: "Convolution_10"
    input: "BatchNormalization_10/bn/beta"
    input: "BatchNormalization_10/bn/gamma"
    input: "BatchNormalization_10/bn/mean"
    input: "BatchNormalization_10/bn/var"
    output: "BatchNormalization_10"
    batch_normalization_param {
      axes: 1
      decay_rate: 0.5
      eps: 0.01
      batch_stat: true
    }
  }
  function {
    name: "Add2_3"
    type: "Add2"
    repeat_id: "RepeatStart_2"
    input: "BatchNormalization_10"
    input: "RepeatStart_2"
    output: "Add2_3"
    add2_param {
      inplace: False
    }
  }
  function {
    name: "ReLU_10"
    type: "ReLU"
    repeat_id: "RepeatStart_2"
    input: "Add2_3"
    output: "ReLU_10"
    relu_param {
      inplace: True
    }
  }
  function {
    name: "RepeatEnd_2"
    type: "RepeatEnd"
    input: "ReLU_10"
    output: "RepeatEnd_2"
    repeat_param {
      repeat_id: "RepeatStart_2"
      times: 2
    }
  }
  function {
    name: "MaxPooling_3"
    type: "MaxPooling"
    input: "RepeatEnd_2"
    output: "MaxPooling_3"
    max_pooling_param {
      kernel: { dim: 2 dim: 2 }
      stride: { dim: 2 dim: 2 }
      ignore_border: true
      pad: { dim: 0 dim: 0 }
    }
  }
  function {
    name: "Convolution_11"
    type: "Convolution"
    input: "MaxPooling_3"
    input: "Convolution_11/conv/W"
    output: "Convolution_11"
    convolution_param {
      pad: { dim: 1 dim: 1 }
      stride: { dim: 1 dim: 1 }
      dilation: { dim: 1 dim: 1 }
      group: 1
      base_axis: 1
    }
  }
  function {
    name: "BatchNormalization_11"
    type: "BatchNormalization"
    input: "Convolution_11"
    input: "BatchNormalization_11/bn/beta"
    input: "BatchNormalization_11/bn/gamma"
    input: "BatchNormalization_11/bn/mean"
    input: "BatchNormalization_11/bn/var"
    output: "BatchNormalization_11"
    batch_normalization_param {
      axes: 1
      decay_rate: 0.5
      eps: 0.01
      batch_stat: true
    }
  }
  function {
    name: "ReLU_11"
    type: "ReLU"
    input: "BatchNormalization_11"
    output: "ReLU_11"
    relu_param {
      inplace: True
    }
  }
  function {
    name: "Convolution_12"
    type: "Convolution"
    input: "ReLU_11"
    input: "Convolution_12/conv/W"
    output: "Convolution_12"
    convolution_param {
      pad: { dim: 0 dim: 0 }
      stride: { dim: 1 dim: 1 }
      dilation: { dim: 1 dim: 1 }
      group: 1
      base_axis: 1
    }
  }
  function {
    name: "BatchNormalization_12"
    type: "BatchNormalization"
    input: "Convolution_12"
    input: "BatchNormalization_12/bn/beta"
    input: "BatchNormalization_12/bn/gamma"
    input: "BatchNormalization_12/bn/mean"
    input: "BatchNormalization_12/bn/var"
    output: "BatchNormalization_12"
    batch_normalization_param {
      axes: 1
      decay_rate: 0.5
      eps: 0.01
      batch_stat: true
    }
  }
  function {
    name: "ReLU_12"
    type: "ReLU"
    input: "BatchNormalization_12"
    output: "ReLU_12"
    relu_param {
      inplace: True
    }
  }
  function {
    name: "Convolution_13"
    type: "Convolution"
    input: "ReLU_12"
    input: "Convolution_13/conv/W"
    output: "Convolution_13"
    convolution_param {
      pad: { dim: 1 dim: 1 }
      stride: { dim: 1 dim: 1 }
      dilation: { dim: 1 dim: 1 }
      group: 1
      base_axis: 1
    }
  }
  function {
    name: "BatchNormalization_13"
    type: "BatchNormalization"
    input: "Convolution_13"
    input: "BatchNormalization_13/bn/beta"
    input: "BatchNormalization_13/bn/gamma"
    input: "BatchNormalization_13/bn/mean"
    input: "BatchNormalization_13/bn/var"
    output: "BatchNormalization_13"
    batch_normalization_param {
      axes: 1
      decay_rate: 0.5
      eps: 0.01
      batch_stat: true
    }
  }
  function {
    name: "Add2_4"
    type: "Add2"
    input: "BatchNormalization_13"
    input: "MaxPooling_3"
    output: "Add2_4"
    add2_param {
      inplace: False
    }
  }
  function {
    name: "ReLU_13"
    type: "ReLU"
    input: "Add2_4"
    output: "ReLU_13"
    relu_param {
      inplace: True
    }
  }
  function {
    name: "AveragePooling"
    type: "AveragePooling"
    input: "ReLU_13"
    output: "AveragePooling"
    average_pooling_param {
      kernel: { dim: 4 dim: 4 }
      stride: { dim: 4 dim: 4 }
      ignore_border: true
      pad: { dim: 0 dim: 0 }
      including_pad: true
    }
  }
  function {
    name: "Affine"
    type: "Affine"
    input: "AveragePooling"
    input: "Affine/affine/W"
    input: "Affine/affine/b"
    output: "Affine"
    affine_param {
      base_axis: 1
    }
  }
  function {
    name: "Softmax"
    type: "Softmax"
    input: "Affine"
    output: "Softmax"
    softmax_param {
      axis: 1
    }
  }
  function {
    name: "CategoricalCrossEntropy"
    type: "CategoricalCrossEntropy"
    input: "Softmax"
    input: "CategoricalCrossEntropy_T"
    output: "CategoricalCrossEntropy"
    categorical_cross_entropy_param {
      axis: 1
    }
  }
}
network {
  name: "MainValidation"
  batch_size: 64
  repeat_info {
    id: "RepeatStart"
    times: 2
  }
  repeat_info {
    id: "RepeatStart_2"
    times: 2
  }
  variable {
    name: "Input"
    type: "Buffer"
    shape: { dim:-1 dim: 1 dim: 28 dim: 28 }
  }
  variable {
    name: "Convolution"
    type: "Buffer"
    shape: { dim:-1 dim: 64 dim: 32 dim: 32 }
  }
  variable {
    name: "BatchNormalization"
    type: "Buffer"
    shape: { dim:-1 dim: 64 dim: 32 dim: 32 }
  }
  variable {
    name: "ReLU"
    type: "Buffer"
    shape: { dim:-1 dim: 64 dim: 32 dim: 32 }
  }
  variable {
    name: "Convolution_2"
    type: "Buffer"
    shape: { dim:-1 dim: 64 dim: 32 dim: 32 }
  }
  variable {
    name: "BatchNormalization_2"
    type: "Buffer"
    shape: { dim:-1 dim: 64 dim: 32 dim: 32 }
  }
  variable {
    name: "ReLU_2"
    type: "Buffer"
    shape: { dim:-1 dim: 64 dim: 32 dim: 32 }
  }
  variable {
    name: "Convolution_3"
    type: "Buffer"
    shape: { dim:-1 dim: 64 dim: 32 dim: 32 }
  }
  variable {
    name: "BatchNormalization_3"
    type: "Buffer"
    shape: { dim:-1 dim: 64 dim: 32 dim: 32 }
  }
  variable {
    name: "ReLU_3"
    type: "Buffer"
    shape: { dim:-1 dim: 64 dim: 32 dim: 32 }
  }
  variable {
    name: "Convolution_4"
    type: "Buffer"
    shape: { dim:-1 dim: 64 dim: 32 dim: 32 }
  }
  variable {
    name: "BatchNormalization_4"
    type: "Buffer"
    shape: { dim:-1 dim: 64 dim: 32 dim: 32 }
  }
  variable {
    name: "Add2"
    type: "Buffer"
    shape: { dim:-1 dim: 64 dim: 32 dim: 32 }
  }
  variable {
    name: "ReLU_4"
    type: "Buffer"
    shape: { dim:-1 dim: 64 dim: 32 dim: 32 }
  }
  variable {
    name: "MaxPooling"
    type: "Buffer"
    shape: { dim:-1 dim: 64 dim: 16 dim: 16 }
  }
  variable {
    name: "RepeatStart"
    type: "Buffer"
    repeat_id: "RepeatStart"
    shape: { dim:-1 dim: 64 dim: 16 dim: 16 }
  }
  variable {
    name: "Convolution_5"
    type: "Buffer"
    repeat_id: "RepeatStart"
    shape: { dim:-1 dim: 64 dim: 16 dim: 16 }
  }
  variable {
    name: "BatchNormalization_5"
    type: "Buffer"
    repeat_id: "RepeatStart"
    shape: { dim:-1 dim: 64 dim: 16 dim: 16 }
  }
  variable {
    name: "ReLU_5"
    type: "Buffer"
    repeat_id: "RepeatStart"
    shape: { dim:-1 dim: 64 dim: 16 dim: 16 }
  }
  variable {
    name: "Convolution_6"
    type: "Buffer"
    repeat_id: "RepeatStart"
    shape: { dim:-1 dim: 64 dim: 16 dim: 16 }
  }
  variable {
    name: "BatchNormalization_6"
    type: "Buffer"
    repeat_id: "RepeatStart"
    shape: { dim:-1 dim: 64 dim: 16 dim: 16 }
  }
  variable {
    name: "ReLU_6"
    type: "Buffer"
    repeat_id: "RepeatStart"
    shape: { dim:-1 dim: 64 dim: 16 dim: 16 }
  }
  variable {
    name: "Convolution_7"
    type: "Buffer"
    repeat_id: "RepeatStart"
    shape: { dim:-1 dim: 64 dim: 16 dim: 16 }
  }
  variable {
    name: "BatchNormalization_7"
    type: "Buffer"
    repeat_id: "RepeatStart"
    shape: { dim:-1 dim: 64 dim: 16 dim: 16 }
  }
  variable {
    name: "Add2_2"
    type: "Buffer"
    repeat_id: "RepeatStart"
    shape: { dim:-1 dim: 64 dim: 16 dim: 16 }
  }
  variable {
    name: "ReLU_7"
    type: "Buffer"
    repeat_id: "RepeatStart"
    shape: { dim:-1 dim: 64 dim: 16 dim: 16 }
  }
  variable {
    name: "RepeatEnd"
    type: "Buffer"
    shape: { dim:-1 dim: 64 dim: 16 dim: 16 }
  }
  variable {
    name: "MaxPooling_2"
    type: "Buffer"
    shape: { dim:-1 dim: 64 dim: 8 dim: 8 }
  }
  variable {
    name: "RepeatStart_2"
    type: "Buffer"
    repeat_id: "RepeatStart_2"
    shape: { dim:-1 dim: 64 dim: 8 dim: 8 }
  }
  variable {
    name: "Convolution_8"
    type: "Buffer"
    repeat_id: "RepeatStart_2"
    shape: { dim:-1 dim: 64 dim: 8 dim: 8 }
  }
  variable {
    name: "BatchNormalization_8"
    type: "Buffer"
    repeat_id: "RepeatStart_2"
    shape: { dim:-1 dim: 64 dim: 8 dim: 8 }
  }
  variable {
    name: "ReLU_8"
    type: "Buffer"
    repeat_id: "RepeatStart_2"
    shape: { dim:-1 dim: 64 dim: 8 dim: 8 }
  }
  variable {
    name: "Convolution_9"
    type: "Buffer"
    repeat_id: "RepeatStart_2"
    shape: { dim:-1 dim: 64 dim: 8 dim: 8 }
  }
  variable {
    name: "BatchNormalization_9"
    type: "Buffer"
    repeat_id: "RepeatStart_2"
    shape: { dim:-1 dim: 64 dim: 8 dim: 8 }
  }
  variable {
    name: "ReLU_9"
    type: "Buffer"
    repeat_id: "RepeatStart_2"
    shape: { dim:-1 dim: 64 dim: 8 dim: 8 }
  }
  variable {
    name: "Convolution_10"
    type: "Buffer"
    repeat_id: "RepeatStart_2"
    shape: { dim:-1 dim: 64 dim: 8 dim: 8 }
  }
  variable {
    name: "BatchNormalization_10"
    type: "Buffer"
    repeat_id: "RepeatStart_2"
    shape: { dim:-1 dim: 64 dim: 8 dim: 8 }
  }
  variable {
    name: "Add2_3"
    type: "Buffer"
    repeat_id: "RepeatStart_2"
    shape: { dim:-1 dim: 64 dim: 8 dim: 8 }
  }
  variable {
    name: "ReLU_10"
    type: "Buffer"
    repeat_id: "RepeatStart_2"
    shape: { dim:-1 dim: 64 dim: 8 dim: 8 }
  }
  variable {
    name: "RepeatEnd_2"
    type: "Buffer"
    shape: { dim:-1 dim: 64 dim: 8 dim: 8 }
  }
  variable {
    name: "MaxPooling_3"
    type: "Buffer"
    shape: { dim:-1 dim: 64 dim: 4 dim: 4 }
  }
  variable {
    name: "Convolution_11"
    type: "Buffer"
    shape: { dim:-1 dim: 64 dim: 4 dim: 4 }
  }
  variable {
    name: "BatchNormalization_11"
    type: "Buffer"
    shape: { dim:-1 dim: 64 dim: 4 dim: 4 }
  }
  variable {
    name: "ReLU_11"
    type: "Buffer"
    shape: { dim:-1 dim: 64 dim: 4 dim: 4 }
  }
  variable {
    name: "Convolution_12"
    type: "Buffer"
    shape: { dim:-1 dim: 64 dim: 4 dim: 4 }
  }
  variable {
    name: "BatchNormalization_12"
    type: "Buffer"
    shape: { dim:-1 dim: 64 dim: 4 dim: 4 }
  }
  variable {
    name: "ReLU_12"
    type: "Buffer"
    shape: { dim:-1 dim: 64 dim: 4 dim: 4 }
  }
  variable {
    name: "Convolution_13"
    type: "Buffer"
    shape: { dim:-1 dim: 64 dim: 4 dim: 4 }
  }
  variable {
    name: "BatchNormalization_13"
    type: "Buffer"
    shape: { dim:-1 dim: 64 dim: 4 dim: 4 }
  }
  variable {
    name: "Add2_4"
    type: "Buffer"
    shape: { dim:-1 dim: 64 dim: 4 dim: 4 }
  }
  variable {
    name: "ReLU_13"
    type: "Buffer"
    shape: { dim:-1 dim: 64 dim: 4 dim: 4 }
  }
  variable {
    name: "AveragePooling"
    type: "Buffer"
    shape: { dim:-1 dim: 64 dim: 1 dim: 1 }
  }
  variable {
    name: "Affine"
    type: "Buffer"
    shape: { dim:-1 dim: 10 }
  }
  variable {
    name: "Softmax"
    type: "Buffer"
    shape: { dim:-1 dim: 10 }
  }
  variable {
    name: "CategoricalCrossEntropy"
    type: "Buffer"
    shape: { dim:-1 dim: 1 }
  }
  variable {
    name: "Convolution/conv/W"
    type: "Parameter"
    shape: { dim: 64 dim: 1 dim: 5 dim: 5 }
    initializer {
      type: "NormalConvolutionGlorot"
      multiplier: 1.0
    }
  }
  variable {
    name: "BatchNormalization/bn/beta"
    type: "Parameter"
    shape: { dim: 1 dim: 64 dim: 1 dim: 1 }
    initializer {
      type: "Constant"
      multiplier: 0.0
    }
  }
  variable {
    name: "BatchNormalization/bn/gamma"
    type: "Parameter"
    shape: { dim: 1 dim: 64 dim: 1 dim: 1 }
    initializer {
      type: "Constant"
      multiplier: 1.0
    }
  }
  variable {
    name: "BatchNormalization/bn/mean"
    type: "Parameter"
    shape: { dim: 1 dim: 64 dim: 1 dim: 1 }
    initializer {
      type: "Constant"
      multiplier: 0.0
    }
  }
  variable {
    name: "BatchNormalization/bn/var"
    type: "Parameter"
    shape: { dim: 1 dim: 64 dim: 1 dim: 1 }
    initializer {
      type: "Constant"
      multiplier: 0.0
    }
  }
  variable {
    name: "Convolution_2/conv/W"
    type: "Parameter"
    shape: { dim: 64 dim: 64 dim: 3 dim: 3 }
    initializer {
      type: "NormalConvolutionGlorot"
      multiplier: 1.0
    }
  }
  variable {
    name: "BatchNormalization_2/bn/beta"
    type: "Parameter"
    shape: { dim: 1 dim: 64 dim: 1 dim: 1 }
    initializer {
      type: "Constant"
      multiplier: 0.0
    }
  }
  variable {
    name: "BatchNormalization_2/bn/gamma"
    type: "Parameter"
    shape: { dim: 1 dim: 64 dim: 1 dim: 1 }
    initializer {
      type: "Constant"
      multiplier: 1.0
    }
  }
  variable {
    name: "BatchNormalization_2/bn/mean"
    type: "Parameter"
    shape: { dim: 1 dim: 64 dim: 1 dim: 1 }
    initializer {
      type: "Constant"
      multiplier: 0.0
    }
  }
  variable {
    name: "BatchNormalization_2/bn/var"
    type: "Parameter"
    shape: { dim: 1 dim: 64 dim: 1 dim: 1 }
    initializer {
      type: "Constant"
      multiplier: 0.0
    }
  }
  variable {
    name: "Convolution_3/conv/W"
    type: "Parameter"
    shape: { dim: 64 dim: 64 dim: 1 dim: 1 }
    initializer {
      type: "NormalConvolutionGlorot"
      multiplier: 1.0
    }
  }
  variable {
    name: "BatchNormalization_3/bn/beta"
    type: "Parameter"
    shape: { dim: 1 dim: 64 dim: 1 dim: 1 }
    initializer {
      type: "Constant"
      multiplier: 0.0
    }
  }
  variable {
    name: "BatchNormalization_3/bn/gamma"
    type: "Parameter"
    shape: { dim: 1 dim: 64 dim: 1 dim: 1 }
    initializer {
      type: "Constant"
      multiplier: 1.0
    }
  }
  variable {
    name: "BatchNormalization_3/bn/mean"
    type: "Parameter"
    shape: { dim: 1 dim: 64 dim: 1 dim: 1 }
    initializer {
      type: "Constant"
      multiplier: 0.0
    }
  }
  variable {
    name: "BatchNormalization_3/bn/var"
    type: "Parameter"
    shape: { dim: 1 dim: 64 dim: 1 dim: 1 }
    initializer {
      type: "Constant"
      multiplier: 0.0
    }
  }
  variable {
    name: "Convolution_4/conv/W"
    type: "Parameter"
    shape: { dim: 64 dim: 64 dim: 3 dim: 3 }
    initializer {
      type: "NormalConvolutionGlorot"
      multiplier: 1.0
    }
  }
  variable {
    name: "BatchNormalization_4/bn/beta"
    type: "Parameter"
    shape: { dim: 1 dim: 64 dim: 1 dim: 1 }
    initializer {
      type: "Constant"
      multiplier: 0.0
    }
  }
  variable {
    name: "BatchNormalization_4/bn/gamma"
    type: "Parameter"
    shape: { dim: 1 dim: 64 dim: 1 dim: 1 }
    initializer {
      type: "Constant"
      multiplier: 1.0
    }
  }
  variable {
    name: "BatchNormalization_4/bn/mean"
    type: "Parameter"
    shape: { dim: 1 dim: 64 dim: 1 dim: 1 }
    initializer {
      type: "Constant"
      multiplier: 0.0
    }
  }
  variable {
    name: "BatchNormalization_4/bn/var"
    type: "Parameter"
    shape: { dim: 1 dim: 64 dim: 1 dim: 1 }
    initializer {
      type: "Constant"
      multiplier: 0.0
    }
  }
  variable {
    name: "Convolution_5/conv/W"
    type: "Parameter"
    repeat_id: "RepeatStart"
    shape: { dim: 64 dim: 64 dim: 3 dim: 3 }
    initializer {
      type: "NormalConvolutionGlorot"
      multiplier: 1.0
    }
  }
  variable {
    name: "BatchNormalization_5/bn/beta"
    type: "Parameter"
    repeat_id: "RepeatStart"
    shape: { dim: 1 dim: 64 dim: 1 dim: 1 }
    initializer {
      type: "Constant"
      multiplier: 0.0
    }
  }
  variable {
    name: "BatchNormalization_5/bn/gamma"
    type: "Parameter"
    repeat_id: "RepeatStart"
    shape: { dim: 1 dim: 64 dim: 1 dim: 1 }
    initializer {
      type: "Constant"
      multiplier: 1.0
    }
  }
  variable {
    name: "BatchNormalization_5/bn/mean"
    type: "Parameter"
    repeat_id: "RepeatStart"
    shape: { dim: 1 dim: 64 dim: 1 dim: 1 }
    initializer {
      type: "Constant"
      multiplier: 0.0
    }
  }
  variable {
    name: "BatchNormalization_5/bn/var"
    type: "Parameter"
    repeat_id: "RepeatStart"
    shape: { dim: 1 dim: 64 dim: 1 dim: 1 }
    initializer {
      type: "Constant"
      multiplier: 0.0
    }
  }
  variable {
    name: "Convolution_6/conv/W"
    type: "Parameter"
    repeat_id: "RepeatStart"
    shape: { dim: 64 dim: 64 dim: 1 dim: 1 }
    initializer {
      type: "NormalConvolutionGlorot"
      multiplier: 1.0
    }
  }
  variable {
    name: "BatchNormalization_6/bn/beta"
    type: "Parameter"
    repeat_id: "RepeatStart"
    shape: { dim: 1 dim: 64 dim: 1 dim: 1 }
    initializer {
      type: "Constant"
      multiplier: 0.0
    }
  }
  variable {
    name: "BatchNormalization_6/bn/gamma"
    type: "Parameter"
    repeat_id: "RepeatStart"
    shape: { dim: 1 dim: 64 dim: 1 dim: 1 }
    initializer {
      type: "Constant"
      multiplier: 1.0
    }
  }
  variable {
    name: "BatchNormalization_6/bn/mean"
    type: "Parameter"
    repeat_id: "RepeatStart"
    shape: { dim: 1 dim: 64 dim: 1 dim: 1 }
    initializer {
      type: "Constant"
      multiplier: 0.0
    }
  }
  variable {
    name: "BatchNormalization_6/bn/var"
    type: "Parameter"
    repeat_id: "RepeatStart"
    shape: { dim: 1 dim: 64 dim: 1 dim: 1 }
    initializer {
      type: "Constant"
      multiplier: 0.0
    }
  }
  variable {
    name: "Convolution_7/conv/W"
    type: "Parameter"
    repeat_id: "RepeatStart"
    shape: { dim: 64 dim: 64 dim: 3 dim: 3 }
    initializer {
      type: "NormalConvolutionGlorot"
      multiplier: 1.0
    }
  }
  variable {
    name: "BatchNormalization_7/bn/beta"
    type: "Parameter"
    repeat_id: "RepeatStart"
    shape: { dim: 1 dim: 64 dim: 1 dim: 1 }
    initializer {
      type: "Constant"
      multiplier: 0.0
    }
  }
  variable {
    name: "BatchNormalization_7/bn/gamma"
    type: "Parameter"
    repeat_id: "RepeatStart"
    shape: { dim: 1 dim: 64 dim: 1 dim: 1 }
    initializer {
      type: "Constant"
      multiplier: 1.0
    }
  }
  variable {
    name: "BatchNormalization_7/bn/mean"
    type: "Parameter"
    repeat_id: "RepeatStart"
    shape: { dim: 1 dim: 64 dim: 1 dim: 1 }
    initializer {
      type: "Constant"
      multiplier: 0.0
    }
  }
  variable {
    name: "BatchNormalization_7/bn/var"
    type: "Parameter"
    repeat_id: "RepeatStart"
    shape: { dim: 1 dim: 64 dim: 1 dim: 1 }
    initializer {
      type: "Constant"
      multiplier: 0.0
    }
  }
  variable {
    name: "Convolution_8/conv/W"
    type: "Parameter"
    repeat_id: "RepeatStart_2"
    shape: { dim: 64 dim: 64 dim: 3 dim: 3 }
    initializer {
      type: "NormalConvolutionGlorot"
      multiplier: 1.0
    }
  }
  variable {
    name: "BatchNormalization_8/bn/beta"
    type: "Parameter"
    repeat_id: "RepeatStart_2"
    shape: { dim: 1 dim: 64 dim: 1 dim: 1 }
    initializer {
      type: "Constant"
      multiplier: 0.0
    }
  }
  variable {
    name: "BatchNormalization_8/bn/gamma"
    type: "Parameter"
    repeat_id: "RepeatStart_2"
    shape: { dim: 1 dim: 64 dim: 1 dim: 1 }
    initializer {
      type: "Constant"
      multiplier: 1.0
    }
  }
  variable {
    name: "BatchNormalization_8/bn/mean"
    type: "Parameter"
    repeat_id: "RepeatStart_2"
    shape: { dim: 1 dim: 64 dim: 1 dim: 1 }
    initializer {
      type: "Constant"
      multiplier: 0.0
    }
  }
  variable {
    name: "BatchNormalization_8/bn/var"
    type: "Parameter"
    repeat_id: "RepeatStart_2"
    shape: { dim: 1 dim: 64 dim: 1 dim: 1 }
    initializer {
      type: "Constant"
      multiplier: 0.0
    }
  }
  variable {
    name: "Convolution_9/conv/W"
    type: "Parameter"
    repeat_id: "RepeatStart_2"
    shape: { dim: 64 dim: 64 dim: 1 dim: 1 }
    initializer {
      type: "NormalConvolutionGlorot"
      multiplier: 1.0
    }
  }
  variable {
    name: "BatchNormalization_9/bn/beta"
    type: "Parameter"
    repeat_id: "RepeatStart_2"
    shape: { dim: 1 dim: 64 dim: 1 dim: 1 }
    initializer {
      type: "Constant"
      multiplier: 0.0
    }
  }
  variable {
    name: "BatchNormalization_9/bn/gamma"
    type: "Parameter"
    repeat_id: "RepeatStart_2"
    shape: { dim: 1 dim: 64 dim: 1 dim: 1 }
    initializer {
      type: "Constant"
      multiplier: 1.0
    }
  }
  variable {
    name: "BatchNormalization_9/bn/mean"
    type: "Parameter"
    repeat_id: "RepeatStart_2"
    shape: { dim: 1 dim: 64 dim: 1 dim: 1 }
    initializer {
      type: "Constant"
      multiplier: 0.0
    }
  }
  variable {
    name: "BatchNormalization_9/bn/var"
    type: "Parameter"
    repeat_id: "RepeatStart_2"
    shape: { dim: 1 dim: 64 dim: 1 dim: 1 }
    initializer {
      type: "Constant"
      multiplier: 0.0
    }
  }
  variable {
    name: "Convolution_10/conv/W"
    type: "Parameter"
    repeat_id: "RepeatStart_2"
    shape: { dim: 64 dim: 64 dim: 3 dim: 3 }
    initializer {
      type: "NormalConvolutionGlorot"
      multiplier: 1.0
    }
  }
  variable {
    name: "BatchNormalization_10/bn/beta"
    type: "Parameter"
    repeat_id: "RepeatStart_2"
    shape: { dim: 1 dim: 64 dim: 1 dim: 1 }
    initializer {
      type: "Constant"
      multiplier: 0.0
    }
  }
  variable {
    name: "BatchNormalization_10/bn/gamma"
    type: "Parameter"
    repeat_id: "RepeatStart_2"
    shape: { dim: 1 dim: 64 dim: 1 dim: 1 }
    initializer {
      type: "Constant"
      multiplier: 1.0
    }
  }
  variable {
    name: "BatchNormalization_10/bn/mean"
    type: "Parameter"
    repeat_id: "RepeatStart_2"
    shape: { dim: 1 dim: 64 dim: 1 dim: 1 }
    initializer {
      type: "Constant"
      multiplier: 0.0
    }
  }
  variable {
    name: "BatchNormalization_10/bn/var"
    type: "Parameter"
    repeat_id: "RepeatStart_2"
    shape: { dim: 1 dim: 64 dim: 1 dim: 1 }
    initializer {
      type: "Constant"
      multiplier: 0.0
    }
  }
  variable {
    name: "Convolution_11/conv/W"
    type: "Parameter"
    shape: { dim: 64 dim: 64 dim: 3 dim: 3 }
    initializer {
      type: "NormalConvolutionGlorot"
      multiplier: 1.0
    }
  }
  variable {
    name: "BatchNormalization_11/bn/beta"
    type: "Parameter"
    shape: { dim: 1 dim: 64 dim: 1 dim: 1 }
    initializer {
      type: "Constant"
      multiplier: 0.0
    }
  }
  variable {
    name: "BatchNormalization_11/bn/gamma"
    type: "Parameter"
    shape: { dim: 1 dim: 64 dim: 1 dim: 1 }
    initializer {
      type: "Constant"
      multiplier: 1.0
    }
  }
  variable {
    name: "BatchNormalization_11/bn/mean"
    type: "Parameter"
    shape: { dim: 1 dim: 64 dim: 1 dim: 1 }
    initializer {
      type: "Constant"
      multiplier: 0.0
    }
  }
  variable {
    name: "BatchNormalization_11/bn/var"
    type: "Parameter"
    shape: { dim: 1 dim: 64 dim: 1 dim: 1 }
    initializer {
      type: "Constant"
      multiplier: 0.0
    }
  }
  variable {
    name: "Convolution_12/conv/W"
    type: "Parameter"
    shape: { dim: 64 dim: 64 dim: 1 dim: 1 }
    initializer {
      type: "NormalConvolutionGlorot"
      multiplier: 1.0
    }
  }
  variable {
    name: "BatchNormalization_12/bn/beta"
    type: "Parameter"
    shape: { dim: 1 dim: 64 dim: 1 dim: 1 }
    initializer {
      type: "Constant"
      multiplier: 0.0
    }
  }
  variable {
    name: "BatchNormalization_12/bn/gamma"
    type: "Parameter"
    shape: { dim: 1 dim: 64 dim: 1 dim: 1 }
    initializer {
      type: "Constant"
      multiplier: 1.0
    }
  }
  variable {
    name: "BatchNormalization_12/bn/mean"
    type: "Parameter"
    shape: { dim: 1 dim: 64 dim: 1 dim: 1 }
    initializer {
      type: "Constant"
      multiplier: 0.0
    }
  }
  variable {
    name: "BatchNormalization_12/bn/var"
    type: "Parameter"
    shape: { dim: 1 dim: 64 dim: 1 dim: 1 }
    initializer {
      type: "Constant"
      multiplier: 0.0
    }
  }
  variable {
    name: "Convolution_13/conv/W"
    type: "Parameter"
    shape: { dim: 64 dim: 64 dim: 3 dim: 3 }
    initializer {
      type: "NormalConvolutionGlorot"
      multiplier: 1.0
    }
  }
  variable {
    name: "BatchNormalization_13/bn/beta"
    type: "Parameter"
    shape: { dim: 1 dim: 64 dim: 1 dim: 1 }
    initializer {
      type: "Constant"
      multiplier: 0.0
    }
  }
  variable {
    name: "BatchNormalization_13/bn/gamma"
    type: "Parameter"
    shape: { dim: 1 dim: 64 dim: 1 dim: 1 }
    initializer {
      type: "Constant"
      multiplier: 1.0
    }
  }
  variable {
    name: "BatchNormalization_13/bn/mean"
    type: "Parameter"
    shape: { dim: 1 dim: 64 dim: 1 dim: 1 }
    initializer {
      type: "Constant"
      multiplier: 0.0
    }
  }
  variable {
    name: "BatchNormalization_13/bn/var"
    type: "Parameter"
    shape: { dim: 1 dim: 64 dim: 1 dim: 1 }
    initializer {
      type: "Constant"
      multiplier: 0.0
    }
  }
  variable {
    name: "Affine/affine/W"
    type: "Parameter"
    shape: { dim: 64 dim: 10 }
    initializer {
      type: "NormalAffineGlorot"
      multiplier: 1.0
    }
  }
  variable {
    name: "Affine/affine/b"
    type: "Parameter"
    shape: { dim: 10 }
    initializer {
      type: "Constant"
      multiplier: 0.0
    }
  }
  variable {
    name: "CategoricalCrossEntropy_T"
    type: "Buffer"
    shape: { dim:-1 dim: 1 }
  }
  function {
    name: "Convolution"
    type: "Convolution"
    input: "Input"
    input: "Convolution/conv/W"
    output: "Convolution"
    convolution_param {
      pad: { dim: 4 dim: 4 }
      stride: { dim: 1 dim: 1 }
      dilation: { dim: 1 dim: 1 }
      group: 1
      base_axis: 1
    }
  }
  function {
    name: "BatchNormalization"
    type: "BatchNormalization"
    input: "Convolution"
    input: "BatchNormalization/bn/beta"
    input: "BatchNormalization/bn/gamma"
    input: "BatchNormalization/bn/mean"
    input: "BatchNormalization/bn/var"
    output: "BatchNormalization"
    batch_normalization_param {
      axes: 1
      decay_rate: 0.5
      eps: 0.01
      batch_stat: false
    }
  }
  function {
    name: "ReLU"
    type: "ReLU"
    input: "BatchNormalization"
    output: "ReLU"
    relu_param {
      inplace: True
    }
  }
  function {
    name: "Convolution_2"
    type: "Convolution"
    input: "ReLU"
    input: "Convolution_2/conv/W"
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
    name: "BatchNormalization_2"
    type: "BatchNormalization"
    input: "Convolution_2"
    input: "BatchNormalization_2/bn/beta"
    input: "BatchNormalization_2/bn/gamma"
    input: "BatchNormalization_2/bn/mean"
    input: "BatchNormalization_2/bn/var"
    output: "BatchNormalization_2"
    batch_normalization_param {
      axes: 1
      decay_rate: 0.5
      eps: 0.01
      batch_stat: false
    }
  }
  function {
    name: "ReLU_2"
    type: "ReLU"
    input: "BatchNormalization_2"
    output: "ReLU_2"
    relu_param {
      inplace: True
    }
  }
  function {
    name: "Convolution_3"
    type: "Convolution"
    input: "ReLU_2"
    input: "Convolution_3/conv/W"
    output: "Convolution_3"
    convolution_param {
      pad: { dim: 0 dim: 0 }
      stride: { dim: 1 dim: 1 }
      dilation: { dim: 1 dim: 1 }
      group: 1
      base_axis: 1
    }
  }
  function {
    name: "BatchNormalization_3"
    type: "BatchNormalization"
    input: "Convolution_3"
    input: "BatchNormalization_3/bn/beta"
    input: "BatchNormalization_3/bn/gamma"
    input: "BatchNormalization_3/bn/mean"
    input: "BatchNormalization_3/bn/var"
    output: "BatchNormalization_3"
    batch_normalization_param {
      axes: 1
      decay_rate: 0.5
      eps: 0.01
      batch_stat: false
    }
  }
  function {
    name: "ReLU_3"
    type: "ReLU"
    input: "BatchNormalization_3"
    output: "ReLU_3"
    relu_param {
      inplace: True
    }
  }
  function {
    name: "Convolution_4"
    type: "Convolution"
    input: "ReLU_3"
    input: "Convolution_4/conv/W"
    output: "Convolution_4"
    convolution_param {
      pad: { dim: 1 dim: 1 }
      stride: { dim: 1 dim: 1 }
      dilation: { dim: 1 dim: 1 }
      group: 1
      base_axis: 1
    }
  }
  function {
    name: "BatchNormalization_4"
    type: "BatchNormalization"
    input: "Convolution_4"
    input: "BatchNormalization_4/bn/beta"
    input: "BatchNormalization_4/bn/gamma"
    input: "BatchNormalization_4/bn/mean"
    input: "BatchNormalization_4/bn/var"
    output: "BatchNormalization_4"
    batch_normalization_param {
      axes: 1
      decay_rate: 0.5
      eps: 0.01
      batch_stat: false
    }
  }
  function {
    name: "Add2"
    type: "Add2"
    input: "BatchNormalization_4"
    input: "ReLU"
    output: "Add2"
    add2_param {
      inplace: False
    }
  }
  function {
    name: "ReLU_4"
    type: "ReLU"
    input: "Add2"
    output: "ReLU_4"
    relu_param {
      inplace: True
    }
  }
  function {
    name: "MaxPooling"
    type: "MaxPooling"
    input: "ReLU_4"
    output: "MaxPooling"
    max_pooling_param {
      kernel: { dim: 2 dim: 2 }
      stride: { dim: 2 dim: 2 }
      ignore_border: true
      pad: { dim: 0 dim: 0 }
    }
  }
  function {
    name: "RepeatStart"
    type: "RepeatStart"
    repeat_id: "RepeatStart"
    input: "MaxPooling"
    input: "ReLU_7"
    output: "RepeatStart"
    repeat_param {
      repeat_id: "RepeatStart"
    }
  }
  function {
    name: "Convolution_5"
    type: "Convolution"
    repeat_id: "RepeatStart"
    input: "RepeatStart"
    input: "Convolution_5/conv/W"
    output: "Convolution_5"
    convolution_param {
      pad: { dim: 1 dim: 1 }
      stride: { dim: 1 dim: 1 }
      dilation: { dim: 1 dim: 1 }
      group: 1
      base_axis: 1
    }
  }
  function {
    name: "BatchNormalization_5"
    type: "BatchNormalization"
    repeat_id: "RepeatStart"
    input: "Convolution_5"
    input: "BatchNormalization_5/bn/beta"
    input: "BatchNormalization_5/bn/gamma"
    input: "BatchNormalization_5/bn/mean"
    input: "BatchNormalization_5/bn/var"
    output: "BatchNormalization_5"
    batch_normalization_param {
      axes: 1
      decay_rate: 0.5
      eps: 0.01
      batch_stat: false
    }
  }
  function {
    name: "ReLU_5"
    type: "ReLU"
    repeat_id: "RepeatStart"
    input: "BatchNormalization_5"
    output: "ReLU_5"
    relu_param {
      inplace: True
    }
  }
  function {
    name: "Convolution_6"
    type: "Convolution"
    repeat_id: "RepeatStart"
    input: "ReLU_5"
    input: "Convolution_6/conv/W"
    output: "Convolution_6"
    convolution_param {
      pad: { dim: 0 dim: 0 }
      stride: { dim: 1 dim: 1 }
      dilation: { dim: 1 dim: 1 }
      group: 1
      base_axis: 1
    }
  }
  function {
    name: "BatchNormalization_6"
    type: "BatchNormalization"
    repeat_id: "RepeatStart"
    input: "Convolution_6"
    input: "BatchNormalization_6/bn/beta"
    input: "BatchNormalization_6/bn/gamma"
    input: "BatchNormalization_6/bn/mean"
    input: "BatchNormalization_6/bn/var"
    output: "BatchNormalization_6"
    batch_normalization_param {
      axes: 1
      decay_rate: 0.5
      eps: 0.01
      batch_stat: false
    }
  }
  function {
    name: "ReLU_6"
    type: "ReLU"
    repeat_id: "RepeatStart"
    input: "BatchNormalization_6"
    output: "ReLU_6"
    relu_param {
      inplace: True
    }
  }
  function {
    name: "Convolution_7"
    type: "Convolution"
    repeat_id: "RepeatStart"
    input: "ReLU_6"
    input: "Convolution_7/conv/W"
    output: "Convolution_7"
    convolution_param {
      pad: { dim: 1 dim: 1 }
      stride: { dim: 1 dim: 1 }
      dilation: { dim: 1 dim: 1 }
      group: 1
      base_axis: 1
    }
  }
  function {
    name: "BatchNormalization_7"
    type: "BatchNormalization"
    repeat_id: "RepeatStart"
    input: "Convolution_7"
    input: "BatchNormalization_7/bn/beta"
    input: "BatchNormalization_7/bn/gamma"
    input: "BatchNormalization_7/bn/mean"
    input: "BatchNormalization_7/bn/var"
    output: "BatchNormalization_7"
    batch_normalization_param {
      axes: 1
      decay_rate: 0.5
      eps: 0.01
      batch_stat: false
    }
  }
  function {
    name: "Add2_2"
    type: "Add2"
    repeat_id: "RepeatStart"
    input: "BatchNormalization_7"
    input: "RepeatStart"
    output: "Add2_2"
    add2_param {
      inplace: False
    }
  }
  function {
    name: "ReLU_7"
    type: "ReLU"
    repeat_id: "RepeatStart"
    input: "Add2_2"
    output: "ReLU_7"
    relu_param {
      inplace: True
    }
  }
  function {
    name: "RepeatEnd"
    type: "RepeatEnd"
    input: "ReLU_7"
    output: "RepeatEnd"
    repeat_param {
      repeat_id: "RepeatStart"
      times: 2
    }
  }
  function {
    name: "MaxPooling_2"
    type: "MaxPooling"
    input: "RepeatEnd"
    output: "MaxPooling_2"
    max_pooling_param {
      kernel: { dim: 2 dim: 2 }
      stride: { dim: 2 dim: 2 }
      ignore_border: true
      pad: { dim: 0 dim: 0 }
    }
  }
  function {
    name: "RepeatStart_2"
    type: "RepeatStart"
    repeat_id: "RepeatStart_2"
    input: "MaxPooling_2"
    input: "ReLU_10"
    output: "RepeatStart_2"
    repeat_param {
      repeat_id: "RepeatStart_2"
    }
  }
  function {
    name: "Convolution_8"
    type: "Convolution"
    repeat_id: "RepeatStart_2"
    input: "RepeatStart_2"
    input: "Convolution_8/conv/W"
    output: "Convolution_8"
    convolution_param {
      pad: { dim: 1 dim: 1 }
      stride: { dim: 1 dim: 1 }
      dilation: { dim: 1 dim: 1 }
      group: 1
      base_axis: 1
    }
  }
  function {
    name: "BatchNormalization_8"
    type: "BatchNormalization"
    repeat_id: "RepeatStart_2"
    input: "Convolution_8"
    input: "BatchNormalization_8/bn/beta"
    input: "BatchNormalization_8/bn/gamma"
    input: "BatchNormalization_8/bn/mean"
    input: "BatchNormalization_8/bn/var"
    output: "BatchNormalization_8"
    batch_normalization_param {
      axes: 1
      decay_rate: 0.5
      eps: 0.01
      batch_stat: false
    }
  }
  function {
    name: "ReLU_8"
    type: "ReLU"
    repeat_id: "RepeatStart_2"
    input: "BatchNormalization_8"
    output: "ReLU_8"
    relu_param {
      inplace: True
    }
  }
  function {
    name: "Convolution_9"
    type: "Convolution"
    repeat_id: "RepeatStart_2"
    input: "ReLU_8"
    input: "Convolution_9/conv/W"
    output: "Convolution_9"
    convolution_param {
      pad: { dim: 0 dim: 0 }
      stride: { dim: 1 dim: 1 }
      dilation: { dim: 1 dim: 1 }
      group: 1
      base_axis: 1
    }
  }
  function {
    name: "BatchNormalization_9"
    type: "BatchNormalization"
    repeat_id: "RepeatStart_2"
    input: "Convolution_9"
    input: "BatchNormalization_9/bn/beta"
    input: "BatchNormalization_9/bn/gamma"
    input: "BatchNormalization_9/bn/mean"
    input: "BatchNormalization_9/bn/var"
    output: "BatchNormalization_9"
    batch_normalization_param {
      axes: 1
      decay_rate: 0.5
      eps: 0.01
      batch_stat: false
    }
  }
  function {
    name: "ReLU_9"
    type: "ReLU"
    repeat_id: "RepeatStart_2"
    input: "BatchNormalization_9"
    output: "ReLU_9"
    relu_param {
      inplace: True
    }
  }
  function {
    name: "Convolution_10"
    type: "Convolution"
    repeat_id: "RepeatStart_2"
    input: "ReLU_9"
    input: "Convolution_10/conv/W"
    output: "Convolution_10"
    convolution_param {
      pad: { dim: 1 dim: 1 }
      stride: { dim: 1 dim: 1 }
      dilation: { dim: 1 dim: 1 }
      group: 1
      base_axis: 1
    }
  }
  function {
    name: "BatchNormalization_10"
    type: "BatchNormalization"
    repeat_id: "RepeatStart_2"
    input: "Convolution_10"
    input: "BatchNormalization_10/bn/beta"
    input: "BatchNormalization_10/bn/gamma"
    input: "BatchNormalization_10/bn/mean"
    input: "BatchNormalization_10/bn/var"
    output: "BatchNormalization_10"
    batch_normalization_param {
      axes: 1
      decay_rate: 0.5
      eps: 0.01
      batch_stat: false
    }
  }
  function {
    name: "Add2_3"
    type: "Add2"
    repeat_id: "RepeatStart_2"
    input: "BatchNormalization_10"
    input: "RepeatStart_2"
    output: "Add2_3"
    add2_param {
      inplace: False
    }
  }
  function {
    name: "ReLU_10"
    type: "ReLU"
    repeat_id: "RepeatStart_2"
    input: "Add2_3"
    output: "ReLU_10"
    relu_param {
      inplace: True
    }
  }
  function {
    name: "RepeatEnd_2"
    type: "RepeatEnd"
    input: "ReLU_10"
    output: "RepeatEnd_2"
    repeat_param {
      repeat_id: "RepeatStart_2"
      times: 2
    }
  }
  function {
    name: "MaxPooling_3"
    type: "MaxPooling"
    input: "RepeatEnd_2"
    output: "MaxPooling_3"
    max_pooling_param {
      kernel: { dim: 2 dim: 2 }
      stride: { dim: 2 dim: 2 }
      ignore_border: true
      pad: { dim: 0 dim: 0 }
    }
  }
  function {
    name: "Convolution_11"
    type: "Convolution"
    input: "MaxPooling_3"
    input: "Convolution_11/conv/W"
    output: "Convolution_11"
    convolution_param {
      pad: { dim: 1 dim: 1 }
      stride: { dim: 1 dim: 1 }
      dilation: { dim: 1 dim: 1 }
      group: 1
      base_axis: 1
    }
  }
  function {
    name: "BatchNormalization_11"
    type: "BatchNormalization"
    input: "Convolution_11"
    input: "BatchNormalization_11/bn/beta"
    input: "BatchNormalization_11/bn/gamma"
    input: "BatchNormalization_11/bn/mean"
    input: "BatchNormalization_11/bn/var"
    output: "BatchNormalization_11"
    batch_normalization_param {
      axes: 1
      decay_rate: 0.5
      eps: 0.01
      batch_stat: false
    }
  }
  function {
    name: "ReLU_11"
    type: "ReLU"
    input: "BatchNormalization_11"
    output: "ReLU_11"
    relu_param {
      inplace: True
    }
  }
  function {
    name: "Convolution_12"
    type: "Convolution"
    input: "ReLU_11"
    input: "Convolution_12/conv/W"
    output: "Convolution_12"
    convolution_param {
      pad: { dim: 0 dim: 0 }
      stride: { dim: 1 dim: 1 }
      dilation: { dim: 1 dim: 1 }
      group: 1
      base_axis: 1
    }
  }
  function {
    name: "BatchNormalization_12"
    type: "BatchNormalization"
    input: "Convolution_12"
    input: "BatchNormalization_12/bn/beta"
    input: "BatchNormalization_12/bn/gamma"
    input: "BatchNormalization_12/bn/mean"
    input: "BatchNormalization_12/bn/var"
    output: "BatchNormalization_12"
    batch_normalization_param {
      axes: 1
      decay_rate: 0.5
      eps: 0.01
      batch_stat: false
    }
  }
  function {
    name: "ReLU_12"
    type: "ReLU"
    input: "BatchNormalization_12"
    output: "ReLU_12"
    relu_param {
      inplace: True
    }
  }
  function {
    name: "Convolution_13"
    type: "Convolution"
    input: "ReLU_12"
    input: "Convolution_13/conv/W"
    output: "Convolution_13"
    convolution_param {
      pad: { dim: 1 dim: 1 }
      stride: { dim: 1 dim: 1 }
      dilation: { dim: 1 dim: 1 }
      group: 1
      base_axis: 1
    }
  }
  function {
    name: "BatchNormalization_13"
    type: "BatchNormalization"
    input: "Convolution_13"
    input: "BatchNormalization_13/bn/beta"
    input: "BatchNormalization_13/bn/gamma"
    input: "BatchNormalization_13/bn/mean"
    input: "BatchNormalization_13/bn/var"
    output: "BatchNormalization_13"
    batch_normalization_param {
      axes: 1
      decay_rate: 0.5
      eps: 0.01
      batch_stat: false
    }
  }
  function {
    name: "Add2_4"
    type: "Add2"
    input: "BatchNormalization_13"
    input: "MaxPooling_3"
    output: "Add2_4"
    add2_param {
      inplace: False
    }
  }
  function {
    name: "ReLU_13"
    type: "ReLU"
    input: "Add2_4"
    output: "ReLU_13"
    relu_param {
      inplace: True
    }
  }
  function {
    name: "AveragePooling"
    type: "AveragePooling"
    input: "ReLU_13"
    output: "AveragePooling"
    average_pooling_param {
      kernel: { dim: 4 dim: 4 }
      stride: { dim: 4 dim: 4 }
      ignore_border: true
      pad: { dim: 0 dim: 0 }
      including_pad: true
    }
  }
  function {
    name: "Affine"
    type: "Affine"
    input: "AveragePooling"
    input: "Affine/affine/W"
    input: "Affine/affine/b"
    output: "Affine"
    affine_param {
      base_axis: 1
    }
  }
  function {
    name: "Softmax"
    type: "Softmax"
    input: "Affine"
    output: "Softmax"
    softmax_param {
      axis: 1
    }
  }
  function {
    name: "CategoricalCrossEntropy"
    type: "CategoricalCrossEntropy"
    input: "Softmax"
    input: "CategoricalCrossEntropy_T"
    output: "CategoricalCrossEntropy"
    categorical_cross_entropy_param {
      axis: 1
    }
  }
}
network {
  name: "MainRuntime"
  batch_size: 64
  repeat_info {
    id: "RepeatStart"
    times: 2
  }
  repeat_info {
    id: "RepeatStart_2"
    times: 2
  }
  variable {
    name: "Input"
    type: "Buffer"
    shape: { dim:-1 dim: 1 dim: 28 dim: 28 }
  }
  variable {
    name: "Convolution"
    type: "Buffer"
    shape: { dim:-1 dim: 64 dim: 32 dim: 32 }
  }
  variable {
    name: "BatchNormalization"
    type: "Buffer"
    shape: { dim:-1 dim: 64 dim: 32 dim: 32 }
  }
  variable {
    name: "ReLU"
    type: "Buffer"
    shape: { dim:-1 dim: 64 dim: 32 dim: 32 }
  }
  variable {
    name: "Convolution_2"
    type: "Buffer"
    shape: { dim:-1 dim: 64 dim: 32 dim: 32 }
  }
  variable {
    name: "BatchNormalization_2"
    type: "Buffer"
    shape: { dim:-1 dim: 64 dim: 32 dim: 32 }
  }
  variable {
    name: "ReLU_2"
    type: "Buffer"
    shape: { dim:-1 dim: 64 dim: 32 dim: 32 }
  }
  variable {
    name: "Convolution_3"
    type: "Buffer"
    shape: { dim:-1 dim: 64 dim: 32 dim: 32 }
  }
  variable {
    name: "BatchNormalization_3"
    type: "Buffer"
    shape: { dim:-1 dim: 64 dim: 32 dim: 32 }
  }
  variable {
    name: "ReLU_3"
    type: "Buffer"
    shape: { dim:-1 dim: 64 dim: 32 dim: 32 }
  }
  variable {
    name: "Convolution_4"
    type: "Buffer"
    shape: { dim:-1 dim: 64 dim: 32 dim: 32 }
  }
  variable {
    name: "BatchNormalization_4"
    type: "Buffer"
    shape: { dim:-1 dim: 64 dim: 32 dim: 32 }
  }
  variable {
    name: "Add2"
    type: "Buffer"
    shape: { dim:-1 dim: 64 dim: 32 dim: 32 }
  }
  variable {
    name: "ReLU_4"
    type: "Buffer"
    shape: { dim:-1 dim: 64 dim: 32 dim: 32 }
  }
  variable {
    name: "MaxPooling"
    type: "Buffer"
    shape: { dim:-1 dim: 64 dim: 16 dim: 16 }
  }
  variable {
    name: "RepeatStart"
    type: "Buffer"
    repeat_id: "RepeatStart"
    shape: { dim:-1 dim: 64 dim: 16 dim: 16 }
  }
  variable {
    name: "Convolution_5"
    type: "Buffer"
    repeat_id: "RepeatStart"
    shape: { dim:-1 dim: 64 dim: 16 dim: 16 }
  }
  variable {
    name: "BatchNormalization_5"
    type: "Buffer"
    repeat_id: "RepeatStart"
    shape: { dim:-1 dim: 64 dim: 16 dim: 16 }
  }
  variable {
    name: "ReLU_5"
    type: "Buffer"
    repeat_id: "RepeatStart"
    shape: { dim:-1 dim: 64 dim: 16 dim: 16 }
  }
  variable {
    name: "Convolution_6"
    type: "Buffer"
    repeat_id: "RepeatStart"
    shape: { dim:-1 dim: 64 dim: 16 dim: 16 }
  }
  variable {
    name: "BatchNormalization_6"
    type: "Buffer"
    repeat_id: "RepeatStart"
    shape: { dim:-1 dim: 64 dim: 16 dim: 16 }
  }
  variable {
    name: "ReLU_6"
    type: "Buffer"
    repeat_id: "RepeatStart"
    shape: { dim:-1 dim: 64 dim: 16 dim: 16 }
  }
  variable {
    name: "Convolution_7"
    type: "Buffer"
    repeat_id: "RepeatStart"
    shape: { dim:-1 dim: 64 dim: 16 dim: 16 }
  }
  variable {
    name: "BatchNormalization_7"
    type: "Buffer"
    repeat_id: "RepeatStart"
    shape: { dim:-1 dim: 64 dim: 16 dim: 16 }
  }
  variable {
    name: "Add2_2"
    type: "Buffer"
    repeat_id: "RepeatStart"
    shape: { dim:-1 dim: 64 dim: 16 dim: 16 }
  }
  variable {
    name: "ReLU_7"
    type: "Buffer"
    repeat_id: "RepeatStart"
    shape: { dim:-1 dim: 64 dim: 16 dim: 16 }
  }
  variable {
    name: "RepeatEnd"
    type: "Buffer"
    shape: { dim:-1 dim: 64 dim: 16 dim: 16 }
  }
  variable {
    name: "MaxPooling_2"
    type: "Buffer"
    shape: { dim:-1 dim: 64 dim: 8 dim: 8 }
  }
  variable {
    name: "RepeatStart_2"
    type: "Buffer"
    repeat_id: "RepeatStart_2"
    shape: { dim:-1 dim: 64 dim: 8 dim: 8 }
  }
  variable {
    name: "Convolution_8"
    type: "Buffer"
    repeat_id: "RepeatStart_2"
    shape: { dim:-1 dim: 64 dim: 8 dim: 8 }
  }
  variable {
    name: "BatchNormalization_8"
    type: "Buffer"
    repeat_id: "RepeatStart_2"
    shape: { dim:-1 dim: 64 dim: 8 dim: 8 }
  }
  variable {
    name: "ReLU_8"
    type: "Buffer"
    repeat_id: "RepeatStart_2"
    shape: { dim:-1 dim: 64 dim: 8 dim: 8 }
  }
  variable {
    name: "Convolution_9"
    type: "Buffer"
    repeat_id: "RepeatStart_2"
    shape: { dim:-1 dim: 64 dim: 8 dim: 8 }
  }
  variable {
    name: "BatchNormalization_9"
    type: "Buffer"
    repeat_id: "RepeatStart_2"
    shape: { dim:-1 dim: 64 dim: 8 dim: 8 }
  }
  variable {
    name: "ReLU_9"
    type: "Buffer"
    repeat_id: "RepeatStart_2"
    shape: { dim:-1 dim: 64 dim: 8 dim: 8 }
  }
  variable {
    name: "Convolution_10"
    type: "Buffer"
    repeat_id: "RepeatStart_2"
    shape: { dim:-1 dim: 64 dim: 8 dim: 8 }
  }
  variable {
    name: "BatchNormalization_10"
    type: "Buffer"
    repeat_id: "RepeatStart_2"
    shape: { dim:-1 dim: 64 dim: 8 dim: 8 }
  }
  variable {
    name: "Add2_3"
    type: "Buffer"
    repeat_id: "RepeatStart_2"
    shape: { dim:-1 dim: 64 dim: 8 dim: 8 }
  }
  variable {
    name: "ReLU_10"
    type: "Buffer"
    repeat_id: "RepeatStart_2"
    shape: { dim:-1 dim: 64 dim: 8 dim: 8 }
  }
  variable {
    name: "RepeatEnd_2"
    type: "Buffer"
    shape: { dim:-1 dim: 64 dim: 8 dim: 8 }
  }
  variable {
    name: "MaxPooling_3"
    type: "Buffer"
    shape: { dim:-1 dim: 64 dim: 4 dim: 4 }
  }
  variable {
    name: "Convolution_11"
    type: "Buffer"
    shape: { dim:-1 dim: 64 dim: 4 dim: 4 }
  }
  variable {
    name: "BatchNormalization_11"
    type: "Buffer"
    shape: { dim:-1 dim: 64 dim: 4 dim: 4 }
  }
  variable {
    name: "ReLU_11"
    type: "Buffer"
    shape: { dim:-1 dim: 64 dim: 4 dim: 4 }
  }
  variable {
    name: "Convolution_12"
    type: "Buffer"
    shape: { dim:-1 dim: 64 dim: 4 dim: 4 }
  }
  variable {
    name: "BatchNormalization_12"
    type: "Buffer"
    shape: { dim:-1 dim: 64 dim: 4 dim: 4 }
  }
  variable {
    name: "ReLU_12"
    type: "Buffer"
    shape: { dim:-1 dim: 64 dim: 4 dim: 4 }
  }
  variable {
    name: "Convolution_13"
    type: "Buffer"
    shape: { dim:-1 dim: 64 dim: 4 dim: 4 }
  }
  variable {
    name: "BatchNormalization_13"
    type: "Buffer"
    shape: { dim:-1 dim: 64 dim: 4 dim: 4 }
  }
  variable {
    name: "Add2_4"
    type: "Buffer"
    shape: { dim:-1 dim: 64 dim: 4 dim: 4 }
  }
  variable {
    name: "ReLU_13"
    type: "Buffer"
    shape: { dim:-1 dim: 64 dim: 4 dim: 4 }
  }
  variable {
    name: "AveragePooling"
    type: "Buffer"
    shape: { dim:-1 dim: 64 dim: 1 dim: 1 }
  }
  variable {
    name: "Affine"
    type: "Buffer"
    shape: { dim:-1 dim: 10 }
  }
  variable {
    name: "Softmax"
    type: "Buffer"
    shape: { dim:-1 dim: 10 }
  }
  variable {
    name: "Convolution/conv/W"
    type: "Parameter"
    shape: { dim: 64 dim: 1 dim: 5 dim: 5 }
    initializer {
      type: "NormalConvolutionGlorot"
      multiplier: 1.0
    }
  }
  variable {
    name: "BatchNormalization/bn/beta"
    type: "Parameter"
    shape: { dim: 1 dim: 64 dim: 1 dim: 1 }
    initializer {
      type: "Constant"
      multiplier: 0.0
    }
  }
  variable {
    name: "BatchNormalization/bn/gamma"
    type: "Parameter"
    shape: { dim: 1 dim: 64 dim: 1 dim: 1 }
    initializer {
      type: "Constant"
      multiplier: 1.0
    }
  }
  variable {
    name: "BatchNormalization/bn/mean"
    type: "Parameter"
    shape: { dim: 1 dim: 64 dim: 1 dim: 1 }
    initializer {
      type: "Constant"
      multiplier: 0.0
    }
  }
  variable {
    name: "BatchNormalization/bn/var"
    type: "Parameter"
    shape: { dim: 1 dim: 64 dim: 1 dim: 1 }
    initializer {
      type: "Constant"
      multiplier: 0.0
    }
  }
  variable {
    name: "Convolution_2/conv/W"
    type: "Parameter"
    shape: { dim: 64 dim: 64 dim: 3 dim: 3 }
    initializer {
      type: "NormalConvolutionGlorot"
      multiplier: 1.0
    }
  }
  variable {
    name: "BatchNormalization_2/bn/beta"
    type: "Parameter"
    shape: { dim: 1 dim: 64 dim: 1 dim: 1 }
    initializer {
      type: "Constant"
      multiplier: 0.0
    }
  }
  variable {
    name: "BatchNormalization_2/bn/gamma"
    type: "Parameter"
    shape: { dim: 1 dim: 64 dim: 1 dim: 1 }
    initializer {
      type: "Constant"
      multiplier: 1.0
    }
  }
  variable {
    name: "BatchNormalization_2/bn/mean"
    type: "Parameter"
    shape: { dim: 1 dim: 64 dim: 1 dim: 1 }
    initializer {
      type: "Constant"
      multiplier: 0.0
    }
  }
  variable {
    name: "BatchNormalization_2/bn/var"
    type: "Parameter"
    shape: { dim: 1 dim: 64 dim: 1 dim: 1 }
    initializer {
      type: "Constant"
      multiplier: 0.0
    }
  }
  variable {
    name: "Convolution_3/conv/W"
    type: "Parameter"
    shape: { dim: 64 dim: 64 dim: 1 dim: 1 }
    initializer {
      type: "NormalConvolutionGlorot"
      multiplier: 1.0
    }
  }
  variable {
    name: "BatchNormalization_3/bn/beta"
    type: "Parameter"
    shape: { dim: 1 dim: 64 dim: 1 dim: 1 }
    initializer {
      type: "Constant"
      multiplier: 0.0
    }
  }
  variable {
    name: "BatchNormalization_3/bn/gamma"
    type: "Parameter"
    shape: { dim: 1 dim: 64 dim: 1 dim: 1 }
    initializer {
      type: "Constant"
      multiplier: 1.0
    }
  }
  variable {
    name: "BatchNormalization_3/bn/mean"
    type: "Parameter"
    shape: { dim: 1 dim: 64 dim: 1 dim: 1 }
    initializer {
      type: "Constant"
      multiplier: 0.0
    }
  }
  variable {
    name: "BatchNormalization_3/bn/var"
    type: "Parameter"
    shape: { dim: 1 dim: 64 dim: 1 dim: 1 }
    initializer {
      type: "Constant"
      multiplier: 0.0
    }
  }
  variable {
    name: "Convolution_4/conv/W"
    type: "Parameter"
    shape: { dim: 64 dim: 64 dim: 3 dim: 3 }
    initializer {
      type: "NormalConvolutionGlorot"
      multiplier: 1.0
    }
  }
  variable {
    name: "BatchNormalization_4/bn/beta"
    type: "Parameter"
    shape: { dim: 1 dim: 64 dim: 1 dim: 1 }
    initializer {
      type: "Constant"
      multiplier: 0.0
    }
  }
  variable {
    name: "BatchNormalization_4/bn/gamma"
    type: "Parameter"
    shape: { dim: 1 dim: 64 dim: 1 dim: 1 }
    initializer {
      type: "Constant"
      multiplier: 1.0
    }
  }
  variable {
    name: "BatchNormalization_4/bn/mean"
    type: "Parameter"
    shape: { dim: 1 dim: 64 dim: 1 dim: 1 }
    initializer {
      type: "Constant"
      multiplier: 0.0
    }
  }
  variable {
    name: "BatchNormalization_4/bn/var"
    type: "Parameter"
    shape: { dim: 1 dim: 64 dim: 1 dim: 1 }
    initializer {
      type: "Constant"
      multiplier: 0.0
    }
  }
  variable {
    name: "Convolution_5/conv/W"
    type: "Parameter"
    repeat_id: "RepeatStart"
    shape: { dim: 64 dim: 64 dim: 3 dim: 3 }
    initializer {
      type: "NormalConvolutionGlorot"
      multiplier: 1.0
    }
  }
  variable {
    name: "BatchNormalization_5/bn/beta"
    type: "Parameter"
    repeat_id: "RepeatStart"
    shape: { dim: 1 dim: 64 dim: 1 dim: 1 }
    initializer {
      type: "Constant"
      multiplier: 0.0
    }
  }
  variable {
    name: "BatchNormalization_5/bn/gamma"
    type: "Parameter"
    repeat_id: "RepeatStart"
    shape: { dim: 1 dim: 64 dim: 1 dim: 1 }
    initializer {
      type: "Constant"
      multiplier: 1.0
    }
  }
  variable {
    name: "BatchNormalization_5/bn/mean"
    type: "Parameter"
    repeat_id: "RepeatStart"
    shape: { dim: 1 dim: 64 dim: 1 dim: 1 }
    initializer {
      type: "Constant"
      multiplier: 0.0
    }
  }
  variable {
    name: "BatchNormalization_5/bn/var"
    type: "Parameter"
    repeat_id: "RepeatStart"
    shape: { dim: 1 dim: 64 dim: 1 dim: 1 }
    initializer {
      type: "Constant"
      multiplier: 0.0
    }
  }
  variable {
    name: "Convolution_6/conv/W"
    type: "Parameter"
    repeat_id: "RepeatStart"
    shape: { dim: 64 dim: 64 dim: 1 dim: 1 }
    initializer {
      type: "NormalConvolutionGlorot"
      multiplier: 1.0
    }
  }
  variable {
    name: "BatchNormalization_6/bn/beta"
    type: "Parameter"
    repeat_id: "RepeatStart"
    shape: { dim: 1 dim: 64 dim: 1 dim: 1 }
    initializer {
      type: "Constant"
      multiplier: 0.0
    }
  }
  variable {
    name: "BatchNormalization_6/bn/gamma"
    type: "Parameter"
    repeat_id: "RepeatStart"
    shape: { dim: 1 dim: 64 dim: 1 dim: 1 }
    initializer {
      type: "Constant"
      multiplier: 1.0
    }
  }
  variable {
    name: "BatchNormalization_6/bn/mean"
    type: "Parameter"
    repeat_id: "RepeatStart"
    shape: { dim: 1 dim: 64 dim: 1 dim: 1 }
    initializer {
      type: "Constant"
      multiplier: 0.0
    }
  }
  variable {
    name: "BatchNormalization_6/bn/var"
    type: "Parameter"
    repeat_id: "RepeatStart"
    shape: { dim: 1 dim: 64 dim: 1 dim: 1 }
    initializer {
      type: "Constant"
      multiplier: 0.0
    }
  }
  variable {
    name: "Convolution_7/conv/W"
    type: "Parameter"
    repeat_id: "RepeatStart"
    shape: { dim: 64 dim: 64 dim: 3 dim: 3 }
    initializer {
      type: "NormalConvolutionGlorot"
      multiplier: 1.0
    }
  }
  variable {
    name: "BatchNormalization_7/bn/beta"
    type: "Parameter"
    repeat_id: "RepeatStart"
    shape: { dim: 1 dim: 64 dim: 1 dim: 1 }
    initializer {
      type: "Constant"
      multiplier: 0.0
    }
  }
  variable {
    name: "BatchNormalization_7/bn/gamma"
    type: "Parameter"
    repeat_id: "RepeatStart"
    shape: { dim: 1 dim: 64 dim: 1 dim: 1 }
    initializer {
      type: "Constant"
      multiplier: 1.0
    }
  }
  variable {
    name: "BatchNormalization_7/bn/mean"
    type: "Parameter"
    repeat_id: "RepeatStart"
    shape: { dim: 1 dim: 64 dim: 1 dim: 1 }
    initializer {
      type: "Constant"
      multiplier: 0.0
    }
  }
  variable {
    name: "BatchNormalization_7/bn/var"
    type: "Parameter"
    repeat_id: "RepeatStart"
    shape: { dim: 1 dim: 64 dim: 1 dim: 1 }
    initializer {
      type: "Constant"
      multiplier: 0.0
    }
  }
  variable {
    name: "Convolution_8/conv/W"
    type: "Parameter"
    repeat_id: "RepeatStart_2"
    shape: { dim: 64 dim: 64 dim: 3 dim: 3 }
    initializer {
      type: "NormalConvolutionGlorot"
      multiplier: 1.0
    }
  }
  variable {
    name: "BatchNormalization_8/bn/beta"
    type: "Parameter"
    repeat_id: "RepeatStart_2"
    shape: { dim: 1 dim: 64 dim: 1 dim: 1 }
    initializer {
      type: "Constant"
      multiplier: 0.0
    }
  }
  variable {
    name: "BatchNormalization_8/bn/gamma"
    type: "Parameter"
    repeat_id: "RepeatStart_2"
    shape: { dim: 1 dim: 64 dim: 1 dim: 1 }
    initializer {
      type: "Constant"
      multiplier: 1.0
    }
  }
  variable {
    name: "BatchNormalization_8/bn/mean"
    type: "Parameter"
    repeat_id: "RepeatStart_2"
    shape: { dim: 1 dim: 64 dim: 1 dim: 1 }
    initializer {
      type: "Constant"
      multiplier: 0.0
    }
  }
  variable {
    name: "BatchNormalization_8/bn/var"
    type: "Parameter"
    repeat_id: "RepeatStart_2"
    shape: { dim: 1 dim: 64 dim: 1 dim: 1 }
    initializer {
      type: "Constant"
      multiplier: 0.0
    }
  }
  variable {
    name: "Convolution_9/conv/W"
    type: "Parameter"
    repeat_id: "RepeatStart_2"
    shape: { dim: 64 dim: 64 dim: 1 dim: 1 }
    initializer {
      type: "NormalConvolutionGlorot"
      multiplier: 1.0
    }
  }
  variable {
    name: "BatchNormalization_9/bn/beta"
    type: "Parameter"
    repeat_id: "RepeatStart_2"
    shape: { dim: 1 dim: 64 dim: 1 dim: 1 }
    initializer {
      type: "Constant"
      multiplier: 0.0
    }
  }
  variable {
    name: "BatchNormalization_9/bn/gamma"
    type: "Parameter"
    repeat_id: "RepeatStart_2"
    shape: { dim: 1 dim: 64 dim: 1 dim: 1 }
    initializer {
      type: "Constant"
      multiplier: 1.0
    }
  }
  variable {
    name: "BatchNormalization_9/bn/mean"
    type: "Parameter"
    repeat_id: "RepeatStart_2"
    shape: { dim: 1 dim: 64 dim: 1 dim: 1 }
    initializer {
      type: "Constant"
      multiplier: 0.0
    }
  }
  variable {
    name: "BatchNormalization_9/bn/var"
    type: "Parameter"
    repeat_id: "RepeatStart_2"
    shape: { dim: 1 dim: 64 dim: 1 dim: 1 }
    initializer {
      type: "Constant"
      multiplier: 0.0
    }
  }
  variable {
    name: "Convolution_10/conv/W"
    type: "Parameter"
    repeat_id: "RepeatStart_2"
    shape: { dim: 64 dim: 64 dim: 3 dim: 3 }
    initializer {
      type: "NormalConvolutionGlorot"
      multiplier: 1.0
    }
  }
  variable {
    name: "BatchNormalization_10/bn/beta"
    type: "Parameter"
    repeat_id: "RepeatStart_2"
    shape: { dim: 1 dim: 64 dim: 1 dim: 1 }
    initializer {
      type: "Constant"
      multiplier: 0.0
    }
  }
  variable {
    name: "BatchNormalization_10/bn/gamma"
    type: "Parameter"
    repeat_id: "RepeatStart_2"
    shape: { dim: 1 dim: 64 dim: 1 dim: 1 }
    initializer {
      type: "Constant"
      multiplier: 1.0
    }
  }
  variable {
    name: "BatchNormalization_10/bn/mean"
    type: "Parameter"
    repeat_id: "RepeatStart_2"
    shape: { dim: 1 dim: 64 dim: 1 dim: 1 }
    initializer {
      type: "Constant"
      multiplier: 0.0
    }
  }
  variable {
    name: "BatchNormalization_10/bn/var"
    type: "Parameter"
    repeat_id: "RepeatStart_2"
    shape: { dim: 1 dim: 64 dim: 1 dim: 1 }
    initializer {
      type: "Constant"
      multiplier: 0.0
    }
  }
  variable {
    name: "Convolution_11/conv/W"
    type: "Parameter"
    shape: { dim: 64 dim: 64 dim: 3 dim: 3 }
    initializer {
      type: "NormalConvolutionGlorot"
      multiplier: 1.0
    }
  }
  variable {
    name: "BatchNormalization_11/bn/beta"
    type: "Parameter"
    shape: { dim: 1 dim: 64 dim: 1 dim: 1 }
    initializer {
      type: "Constant"
      multiplier: 0.0
    }
  }
  variable {
    name: "BatchNormalization_11/bn/gamma"
    type: "Parameter"
    shape: { dim: 1 dim: 64 dim: 1 dim: 1 }
    initializer {
      type: "Constant"
      multiplier: 1.0
    }
  }
  variable {
    name: "BatchNormalization_11/bn/mean"
    type: "Parameter"
    shape: { dim: 1 dim: 64 dim: 1 dim: 1 }
    initializer {
      type: "Constant"
      multiplier: 0.0
    }
  }
  variable {
    name: "BatchNormalization_11/bn/var"
    type: "Parameter"
    shape: { dim: 1 dim: 64 dim: 1 dim: 1 }
    initializer {
      type: "Constant"
      multiplier: 0.0
    }
  }
  variable {
    name: "Convolution_12/conv/W"
    type: "Parameter"
    shape: { dim: 64 dim: 64 dim: 1 dim: 1 }
    initializer {
      type: "NormalConvolutionGlorot"
      multiplier: 1.0
    }
  }
  variable {
    name: "BatchNormalization_12/bn/beta"
    type: "Parameter"
    shape: { dim: 1 dim: 64 dim: 1 dim: 1 }
    initializer {
      type: "Constant"
      multiplier: 0.0
    }
  }
  variable {
    name: "BatchNormalization_12/bn/gamma"
    type: "Parameter"
    shape: { dim: 1 dim: 64 dim: 1 dim: 1 }
    initializer {
      type: "Constant"
      multiplier: 1.0
    }
  }
  variable {
    name: "BatchNormalization_12/bn/mean"
    type: "Parameter"
    shape: { dim: 1 dim: 64 dim: 1 dim: 1 }
    initializer {
      type: "Constant"
      multiplier: 0.0
    }
  }
  variable {
    name: "BatchNormalization_12/bn/var"
    type: "Parameter"
    shape: { dim: 1 dim: 64 dim: 1 dim: 1 }
    initializer {
      type: "Constant"
      multiplier: 0.0
    }
  }
  variable {
    name: "Convolution_13/conv/W"
    type: "Parameter"
    shape: { dim: 64 dim: 64 dim: 3 dim: 3 }
    initializer {
      type: "NormalConvolutionGlorot"
      multiplier: 1.0
    }
  }
  variable {
    name: "BatchNormalization_13/bn/beta"
    type: "Parameter"
    shape: { dim: 1 dim: 64 dim: 1 dim: 1 }
    initializer {
      type: "Constant"
      multiplier: 0.0
    }
  }
  variable {
    name: "BatchNormalization_13/bn/gamma"
    type: "Parameter"
    shape: { dim: 1 dim: 64 dim: 1 dim: 1 }
    initializer {
      type: "Constant"
      multiplier: 1.0
    }
  }
  variable {
    name: "BatchNormalization_13/bn/mean"
    type: "Parameter"
    shape: { dim: 1 dim: 64 dim: 1 dim: 1 }
    initializer {
      type: "Constant"
      multiplier: 0.0
    }
  }
  variable {
    name: "BatchNormalization_13/bn/var"
    type: "Parameter"
    shape: { dim: 1 dim: 64 dim: 1 dim: 1 }
    initializer {
      type: "Constant"
      multiplier: 0.0
    }
  }
  variable {
    name: "Affine/affine/W"
    type: "Parameter"
    shape: { dim: 64 dim: 10 }
    initializer {
      type: "NormalAffineGlorot"
      multiplier: 1.0
    }
  }
  variable {
    name: "Affine/affine/b"
    type: "Parameter"
    shape: { dim: 10 }
    initializer {
      type: "Constant"
      multiplier: 0.0
    }
  }
  function {
    name: "Convolution"
    type: "Convolution"
    input: "Input"
    input: "Convolution/conv/W"
    output: "Convolution"
    convolution_param {
      pad: { dim: 4 dim: 4 }
      stride: { dim: 1 dim: 1 }
      dilation: { dim: 1 dim: 1 }
      group: 1
      base_axis: 1
    }
  }
  function {
    name: "BatchNormalization"
    type: "BatchNormalization"
    input: "Convolution"
    input: "BatchNormalization/bn/beta"
    input: "BatchNormalization/bn/gamma"
    input: "BatchNormalization/bn/mean"
    input: "BatchNormalization/bn/var"
    output: "BatchNormalization"
    batch_normalization_param {
      axes: 1
      decay_rate: 0.5
      eps: 0.01
      batch_stat: false
    }
  }
  function {
    name: "ReLU"
    type: "ReLU"
    input: "BatchNormalization"
    output: "ReLU"
    relu_param {
      inplace: True
    }
  }
  function {
    name: "Convolution_2"
    type: "Convolution"
    input: "ReLU"
    input: "Convolution_2/conv/W"
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
    name: "BatchNormalization_2"
    type: "BatchNormalization"
    input: "Convolution_2"
    input: "BatchNormalization_2/bn/beta"
    input: "BatchNormalization_2/bn/gamma"
    input: "BatchNormalization_2/bn/mean"
    input: "BatchNormalization_2/bn/var"
    output: "BatchNormalization_2"
    batch_normalization_param {
      axes: 1
      decay_rate: 0.5
      eps: 0.01
      batch_stat: false
    }
  }
  function {
    name: "ReLU_2"
    type: "ReLU"
    input: "BatchNormalization_2"
    output: "ReLU_2"
    relu_param {
      inplace: True
    }
  }
  function {
    name: "Convolution_3"
    type: "Convolution"
    input: "ReLU_2"
    input: "Convolution_3/conv/W"
    output: "Convolution_3"
    convolution_param {
      pad: { dim: 0 dim: 0 }
      stride: { dim: 1 dim: 1 }
      dilation: { dim: 1 dim: 1 }
      group: 1
      base_axis: 1
    }
  }
  function {
    name: "BatchNormalization_3"
    type: "BatchNormalization"
    input: "Convolution_3"
    input: "BatchNormalization_3/bn/beta"
    input: "BatchNormalization_3/bn/gamma"
    input: "BatchNormalization_3/bn/mean"
    input: "BatchNormalization_3/bn/var"
    output: "BatchNormalization_3"
    batch_normalization_param {
      axes: 1
      decay_rate: 0.5
      eps: 0.01
      batch_stat: false
    }
  }
  function {
    name: "ReLU_3"
    type: "ReLU"
    input: "BatchNormalization_3"
    output: "ReLU_3"
    relu_param {
      inplace: True
    }
  }
  function {
    name: "Convolution_4"
    type: "Convolution"
    input: "ReLU_3"
    input: "Convolution_4/conv/W"
    output: "Convolution_4"
    convolution_param {
      pad: { dim: 1 dim: 1 }
      stride: { dim: 1 dim: 1 }
      dilation: { dim: 1 dim: 1 }
      group: 1
      base_axis: 1
    }
  }
  function {
    name: "BatchNormalization_4"
    type: "BatchNormalization"
    input: "Convolution_4"
    input: "BatchNormalization_4/bn/beta"
    input: "BatchNormalization_4/bn/gamma"
    input: "BatchNormalization_4/bn/mean"
    input: "BatchNormalization_4/bn/var"
    output: "BatchNormalization_4"
    batch_normalization_param {
      axes: 1
      decay_rate: 0.5
      eps: 0.01
      batch_stat: false
    }
  }
  function {
    name: "Add2"
    type: "Add2"
    input: "BatchNormalization_4"
    input: "ReLU"
    output: "Add2"
    add2_param {
      inplace: False
    }
  }
  function {
    name: "ReLU_4"
    type: "ReLU"
    input: "Add2"
    output: "ReLU_4"
    relu_param {
      inplace: True
    }
  }
  function {
    name: "MaxPooling"
    type: "MaxPooling"
    input: "ReLU_4"
    output: "MaxPooling"
    max_pooling_param {
      kernel: { dim: 2 dim: 2 }
      stride: { dim: 2 dim: 2 }
      ignore_border: true
      pad: { dim: 0 dim: 0 }
    }
  }
  function {
    name: "RepeatStart"
    type: "RepeatStart"
    repeat_id: "RepeatStart"
    input: "MaxPooling"
    input: "ReLU_7"
    output: "RepeatStart"
    repeat_param {
      repeat_id: "RepeatStart"
    }
  }
  function {
    name: "Convolution_5"
    type: "Convolution"
    repeat_id: "RepeatStart"
    input: "RepeatStart"
    input: "Convolution_5/conv/W"
    output: "Convolution_5"
    convolution_param {
      pad: { dim: 1 dim: 1 }
      stride: { dim: 1 dim: 1 }
      dilation: { dim: 1 dim: 1 }
      group: 1
      base_axis: 1
    }
  }
  function {
    name: "BatchNormalization_5"
    type: "BatchNormalization"
    repeat_id: "RepeatStart"
    input: "Convolution_5"
    input: "BatchNormalization_5/bn/beta"
    input: "BatchNormalization_5/bn/gamma"
    input: "BatchNormalization_5/bn/mean"
    input: "BatchNormalization_5/bn/var"
    output: "BatchNormalization_5"
    batch_normalization_param {
      axes: 1
      decay_rate: 0.5
      eps: 0.01
      batch_stat: false
    }
  }
  function {
    name: "ReLU_5"
    type: "ReLU"
    repeat_id: "RepeatStart"
    input: "BatchNormalization_5"
    output: "ReLU_5"
    relu_param {
      inplace: True
    }
  }
  function {
    name: "Convolution_6"
    type: "Convolution"
    repeat_id: "RepeatStart"
    input: "ReLU_5"
    input: "Convolution_6/conv/W"
    output: "Convolution_6"
    convolution_param {
      pad: { dim: 0 dim: 0 }
      stride: { dim: 1 dim: 1 }
      dilation: { dim: 1 dim: 1 }
      group: 1
      base_axis: 1
    }
  }
  function {
    name: "BatchNormalization_6"
    type: "BatchNormalization"
    repeat_id: "RepeatStart"
    input: "Convolution_6"
    input: "BatchNormalization_6/bn/beta"
    input: "BatchNormalization_6/bn/gamma"
    input: "BatchNormalization_6/bn/mean"
    input: "BatchNormalization_6/bn/var"
    output: "BatchNormalization_6"
    batch_normalization_param {
      axes: 1
      decay_rate: 0.5
      eps: 0.01
      batch_stat: false
    }
  }
  function {
    name: "ReLU_6"
    type: "ReLU"
    repeat_id: "RepeatStart"
    input: "BatchNormalization_6"
    output: "ReLU_6"
    relu_param {
      inplace: True
    }
  }
  function {
    name: "Convolution_7"
    type: "Convolution"
    repeat_id: "RepeatStart"
    input: "ReLU_6"
    input: "Convolution_7/conv/W"
    output: "Convolution_7"
    convolution_param {
      pad: { dim: 1 dim: 1 }
      stride: { dim: 1 dim: 1 }
      dilation: { dim: 1 dim: 1 }
      group: 1
      base_axis: 1
    }
  }
  function {
    name: "BatchNormalization_7"
    type: "BatchNormalization"
    repeat_id: "RepeatStart"
    input: "Convolution_7"
    input: "BatchNormalization_7/bn/beta"
    input: "BatchNormalization_7/bn/gamma"
    input: "BatchNormalization_7/bn/mean"
    input: "BatchNormalization_7/bn/var"
    output: "BatchNormalization_7"
    batch_normalization_param {
      axes: 1
      decay_rate: 0.5
      eps: 0.01
      batch_stat: false
    }
  }
  function {
    name: "Add2_2"
    type: "Add2"
    repeat_id: "RepeatStart"
    input: "BatchNormalization_7"
    input: "RepeatStart"
    output: "Add2_2"
    add2_param {
      inplace: False
    }
  }
  function {
    name: "ReLU_7"
    type: "ReLU"
    repeat_id: "RepeatStart"
    input: "Add2_2"
    output: "ReLU_7"
    relu_param {
      inplace: True
    }
  }
  function {
    name: "RepeatEnd"
    type: "RepeatEnd"
    input: "ReLU_7"
    output: "RepeatEnd"
    repeat_param {
      repeat_id: "RepeatStart"
      times: 2
    }
  }
  function {
    name: "MaxPooling_2"
    type: "MaxPooling"
    input: "RepeatEnd"
    output: "MaxPooling_2"
    max_pooling_param {
      kernel: { dim: 2 dim: 2 }
      stride: { dim: 2 dim: 2 }
      ignore_border: true
      pad: { dim: 0 dim: 0 }
    }
  }
  function {
    name: "RepeatStart_2"
    type: "RepeatStart"
    repeat_id: "RepeatStart_2"
    input: "MaxPooling_2"
    input: "ReLU_10"
    output: "RepeatStart_2"
    repeat_param {
      repeat_id: "RepeatStart_2"
    }
  }
  function {
    name: "Convolution_8"
    type: "Convolution"
    repeat_id: "RepeatStart_2"
    input: "RepeatStart_2"
    input: "Convolution_8/conv/W"
    output: "Convolution_8"
    convolution_param {
      pad: { dim: 1 dim: 1 }
      stride: { dim: 1 dim: 1 }
      dilation: { dim: 1 dim: 1 }
      group: 1
      base_axis: 1
    }
  }
  function {
    name: "BatchNormalization_8"
    type: "BatchNormalization"
    repeat_id: "RepeatStart_2"
    input: "Convolution_8"
    input: "BatchNormalization_8/bn/beta"
    input: "BatchNormalization_8/bn/gamma"
    input: "BatchNormalization_8/bn/mean"
    input: "BatchNormalization_8/bn/var"
    output: "BatchNormalization_8"
    batch_normalization_param {
      axes: 1
      decay_rate: 0.5
      eps: 0.01
      batch_stat: false
    }
  }
  function {
    name: "ReLU_8"
    type: "ReLU"
    repeat_id: "RepeatStart_2"
    input: "BatchNormalization_8"
    output: "ReLU_8"
    relu_param {
      inplace: True
    }
  }
  function {
    name: "Convolution_9"
    type: "Convolution"
    repeat_id: "RepeatStart_2"
    input: "ReLU_8"
    input: "Convolution_9/conv/W"
    output: "Convolution_9"
    convolution_param {
      pad: { dim: 0 dim: 0 }
      stride: { dim: 1 dim: 1 }
      dilation: { dim: 1 dim: 1 }
      group: 1
      base_axis: 1
    }
  }
  function {
    name: "BatchNormalization_9"
    type: "BatchNormalization"
    repeat_id: "RepeatStart_2"
    input: "Convolution_9"
    input: "BatchNormalization_9/bn/beta"
    input: "BatchNormalization_9/bn/gamma"
    input: "BatchNormalization_9/bn/mean"
    input: "BatchNormalization_9/bn/var"
    output: "BatchNormalization_9"
    batch_normalization_param {
      axes: 1
      decay_rate: 0.5
      eps: 0.01
      batch_stat: false
    }
  }
  function {
    name: "ReLU_9"
    type: "ReLU"
    repeat_id: "RepeatStart_2"
    input: "BatchNormalization_9"
    output: "ReLU_9"
    relu_param {
      inplace: True
    }
  }
  function {
    name: "Convolution_10"
    type: "Convolution"
    repeat_id: "RepeatStart_2"
    input: "ReLU_9"
    input: "Convolution_10/conv/W"
    output: "Convolution_10"
    convolution_param {
      pad: { dim: 1 dim: 1 }
      stride: { dim: 1 dim: 1 }
      dilation: { dim: 1 dim: 1 }
      group: 1
      base_axis: 1
    }
  }
  function {
    name: "BatchNormalization_10"
    type: "BatchNormalization"
    repeat_id: "RepeatStart_2"
    input: "Convolution_10"
    input: "BatchNormalization_10/bn/beta"
    input: "BatchNormalization_10/bn/gamma"
    input: "BatchNormalization_10/bn/mean"
    input: "BatchNormalization_10/bn/var"
    output: "BatchNormalization_10"
    batch_normalization_param {
      axes: 1
      decay_rate: 0.5
      eps: 0.01
      batch_stat: false
    }
  }
  function {
    name: "Add2_3"
    type: "Add2"
    repeat_id: "RepeatStart_2"
    input: "BatchNormalization_10"
    input: "RepeatStart_2"
    output: "Add2_3"
    add2_param {
      inplace: False
    }
  }
  function {
    name: "ReLU_10"
    type: "ReLU"
    repeat_id: "RepeatStart_2"
    input: "Add2_3"
    output: "ReLU_10"
    relu_param {
      inplace: True
    }
  }
  function {
    name: "RepeatEnd_2"
    type: "RepeatEnd"
    input: "ReLU_10"
    output: "RepeatEnd_2"
    repeat_param {
      repeat_id: "RepeatStart_2"
      times: 2
    }
  }
  function {
    name: "MaxPooling_3"
    type: "MaxPooling"
    input: "RepeatEnd_2"
    output: "MaxPooling_3"
    max_pooling_param {
      kernel: { dim: 2 dim: 2 }
      stride: { dim: 2 dim: 2 }
      ignore_border: true
      pad: { dim: 0 dim: 0 }
    }
  }
  function {
    name: "Convolution_11"
    type: "Convolution"
    input: "MaxPooling_3"
    input: "Convolution_11/conv/W"
    output: "Convolution_11"
    convolution_param {
      pad: { dim: 1 dim: 1 }
      stride: { dim: 1 dim: 1 }
      dilation: { dim: 1 dim: 1 }
      group: 1
      base_axis: 1
    }
  }
  function {
    name: "BatchNormalization_11"
    type: "BatchNormalization"
    input: "Convolution_11"
    input: "BatchNormalization_11/bn/beta"
    input: "BatchNormalization_11/bn/gamma"
    input: "BatchNormalization_11/bn/mean"
    input: "BatchNormalization_11/bn/var"
    output: "BatchNormalization_11"
    batch_normalization_param {
      axes: 1
      decay_rate: 0.5
      eps: 0.01
      batch_stat: false
    }
  }
  function {
    name: "ReLU_11"
    type: "ReLU"
    input: "BatchNormalization_11"
    output: "ReLU_11"
    relu_param {
      inplace: True
    }
  }
  function {
    name: "Convolution_12"
    type: "Convolution"
    input: "ReLU_11"
    input: "Convolution_12/conv/W"
    output: "Convolution_12"
    convolution_param {
      pad: { dim: 0 dim: 0 }
      stride: { dim: 1 dim: 1 }
      dilation: { dim: 1 dim: 1 }
      group: 1
      base_axis: 1
    }
  }
  function {
    name: "BatchNormalization_12"
    type: "BatchNormalization"
    input: "Convolution_12"
    input: "BatchNormalization_12/bn/beta"
    input: "BatchNormalization_12/bn/gamma"
    input: "BatchNormalization_12/bn/mean"
    input: "BatchNormalization_12/bn/var"
    output: "BatchNormalization_12"
    batch_normalization_param {
      axes: 1
      decay_rate: 0.5
      eps: 0.01
      batch_stat: false
    }
  }
  function {
    name: "ReLU_12"
    type: "ReLU"
    input: "BatchNormalization_12"
    output: "ReLU_12"
    relu_param {
      inplace: True
    }
  }
  function {
    name: "Convolution_13"
    type: "Convolution"
    input: "ReLU_12"
    input: "Convolution_13/conv/W"
    output: "Convolution_13"
    convolution_param {
      pad: { dim: 1 dim: 1 }
      stride: { dim: 1 dim: 1 }
      dilation: { dim: 1 dim: 1 }
      group: 1
      base_axis: 1
    }
  }
  function {
    name: "BatchNormalization_13"
    type: "BatchNormalization"
    input: "Convolution_13"
    input: "BatchNormalization_13/bn/beta"
    input: "BatchNormalization_13/bn/gamma"
    input: "BatchNormalization_13/bn/mean"
    input: "BatchNormalization_13/bn/var"
    output: "BatchNormalization_13"
    batch_normalization_param {
      axes: 1
      decay_rate: 0.5
      eps: 0.01
      batch_stat: false
    }
  }
  function {
    name: "Add2_4"
    type: "Add2"
    input: "BatchNormalization_13"
    input: "MaxPooling_3"
    output: "Add2_4"
    add2_param {
      inplace: False
    }
  }
  function {
    name: "ReLU_13"
    type: "ReLU"
    input: "Add2_4"
    output: "ReLU_13"
    relu_param {
      inplace: True
    }
  }
  function {
    name: "AveragePooling"
    type: "AveragePooling"
    input: "ReLU_13"
    output: "AveragePooling"
    average_pooling_param {
      kernel: { dim: 4 dim: 4 }
      stride: { dim: 4 dim: 4 }
      ignore_border: true
      pad: { dim: 0 dim: 0 }
      including_pad: true
    }
  }
  function {
    name: "Affine"
    type: "Affine"
    input: "AveragePooling"
    input: "Affine/affine/W"
    input: "Affine/affine/b"
    output: "Affine"
    affine_param {
      base_axis: 1
    }
  }
  function {
    name: "Softmax"
    type: "Softmax"
    input: "Affine"
    output: "Softmax"
    softmax_param {
      axis: 1
    }
  }
}
dataset {
  name: "Training"
  uri: "K:\\neural_network_console_170725\\samples\\sample_dataset\\mnist\\mnist_training.csv"
  cache_dir: "K:\\neural_network_console_170725\\samples\\sample_dataset\\mnist\\mnist_training.cache"
  overwrite_cache: False
  create_cache_explicitly: True
  shuffle: true
  no_image_normalization: False
  batch_size: 64
}
dataset {
  name: "Validation"
  uri: "K:\\neural_network_console_170725\\samples\\sample_dataset\\mnist\\mnist_test.csv"
  cache_dir: "K:\\neural_network_console_170725\\samples\\sample_dataset\\mnist\\mnist_test.cache"
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
  data_variable {
    variable_name: "CategoricalCrossEntropy_T"
    data_name: "y"
  }
  loss_variable {
    variable_name: "CategoricalCrossEntropy"
  }
  parameter_variable {
    variable_name: "Convolution/conv/W"
    learning_rate_multiplier: 1.0
  }
  parameter_variable {
    variable_name: "BatchNormalization/bn/beta"
    learning_rate_multiplier: 1.0
  }
  parameter_variable {
    variable_name: "BatchNormalization/bn/gamma"
    learning_rate_multiplier: 1.0
  }
  parameter_variable {
    variable_name: "BatchNormalization/bn/mean"
    learning_rate_multiplier: 0.0
  }
  parameter_variable {
    variable_name: "BatchNormalization/bn/var"
    learning_rate_multiplier: 0.0
  }
  parameter_variable {
    variable_name: "Convolution_2/conv/W"
    learning_rate_multiplier: 1.0
  }
  parameter_variable {
    variable_name: "BatchNormalization_2/bn/beta"
    learning_rate_multiplier: 1.0
  }
  parameter_variable {
    variable_name: "BatchNormalization_2/bn/gamma"
    learning_rate_multiplier: 1.0
  }
  parameter_variable {
    variable_name: "BatchNormalization_2/bn/mean"
    learning_rate_multiplier: 0.0
  }
  parameter_variable {
    variable_name: "BatchNormalization_2/bn/var"
    learning_rate_multiplier: 0.0
  }
  parameter_variable {
    variable_name: "Convolution_3/conv/W"
    learning_rate_multiplier: 1.0
  }
  parameter_variable {
    variable_name: "BatchNormalization_3/bn/beta"
    learning_rate_multiplier: 1.0
  }
  parameter_variable {
    variable_name: "BatchNormalization_3/bn/gamma"
    learning_rate_multiplier: 1.0
  }
  parameter_variable {
    variable_name: "BatchNormalization_3/bn/mean"
    learning_rate_multiplier: 0.0
  }
  parameter_variable {
    variable_name: "BatchNormalization_3/bn/var"
    learning_rate_multiplier: 0.0
  }
  parameter_variable {
    variable_name: "Convolution_4/conv/W"
    learning_rate_multiplier: 1.0
  }
  parameter_variable {
    variable_name: "BatchNormalization_4/bn/beta"
    learning_rate_multiplier: 1.0
  }
  parameter_variable {
    variable_name: "BatchNormalization_4/bn/gamma"
    learning_rate_multiplier: 1.0
  }
  parameter_variable {
    variable_name: "BatchNormalization_4/bn/mean"
    learning_rate_multiplier: 0.0
  }
  parameter_variable {
    variable_name: "BatchNormalization_4/bn/var"
    learning_rate_multiplier: 0.0
  }
  parameter_variable {
    variable_name: "Convolution_5/conv/W"
    learning_rate_multiplier: 1.0
  }
  parameter_variable {
    variable_name: "BatchNormalization_5/bn/beta"
    learning_rate_multiplier: 1.0
  }
  parameter_variable {
    variable_name: "BatchNormalization_5/bn/gamma"
    learning_rate_multiplier: 1.0
  }
  parameter_variable {
    variable_name: "BatchNormalization_5/bn/mean"
    learning_rate_multiplier: 0.0
  }
  parameter_variable {
    variable_name: "BatchNormalization_5/bn/var"
    learning_rate_multiplier: 0.0
  }
  parameter_variable {
    variable_name: "Convolution_6/conv/W"
    learning_rate_multiplier: 1.0
  }
  parameter_variable {
    variable_name: "BatchNormalization_6/bn/beta"
    learning_rate_multiplier: 1.0
  }
  parameter_variable {
    variable_name: "BatchNormalization_6/bn/gamma"
    learning_rate_multiplier: 1.0
  }
  parameter_variable {
    variable_name: "BatchNormalization_6/bn/mean"
    learning_rate_multiplier: 0.0
  }
  parameter_variable {
    variable_name: "BatchNormalization_6/bn/var"
    learning_rate_multiplier: 0.0
  }
  parameter_variable {
    variable_name: "Convolution_7/conv/W"
    learning_rate_multiplier: 1.0
  }
  parameter_variable {
    variable_name: "BatchNormalization_7/bn/beta"
    learning_rate_multiplier: 1.0
  }
  parameter_variable {
    variable_name: "BatchNormalization_7/bn/gamma"
    learning_rate_multiplier: 1.0
  }
  parameter_variable {
    variable_name: "BatchNormalization_7/bn/mean"
    learning_rate_multiplier: 0.0
  }
  parameter_variable {
    variable_name: "BatchNormalization_7/bn/var"
    learning_rate_multiplier: 0.0
  }
  parameter_variable {
    variable_name: "Convolution_8/conv/W"
    learning_rate_multiplier: 1.0
  }
  parameter_variable {
    variable_name: "BatchNormalization_8/bn/beta"
    learning_rate_multiplier: 1.0
  }
  parameter_variable {
    variable_name: "BatchNormalization_8/bn/gamma"
    learning_rate_multiplier: 1.0
  }
  parameter_variable {
    variable_name: "BatchNormalization_8/bn/mean"
    learning_rate_multiplier: 0.0
  }
  parameter_variable {
    variable_name: "BatchNormalization_8/bn/var"
    learning_rate_multiplier: 0.0
  }
  parameter_variable {
    variable_name: "Convolution_9/conv/W"
    learning_rate_multiplier: 1.0
  }
  parameter_variable {
    variable_name: "BatchNormalization_9/bn/beta"
    learning_rate_multiplier: 1.0
  }
  parameter_variable {
    variable_name: "BatchNormalization_9/bn/gamma"
    learning_rate_multiplier: 1.0
  }
  parameter_variable {
    variable_name: "BatchNormalization_9/bn/mean"
    learning_rate_multiplier: 0.0
  }
  parameter_variable {
    variable_name: "BatchNormalization_9/bn/var"
    learning_rate_multiplier: 0.0
  }
  parameter_variable {
    variable_name: "Convolution_10/conv/W"
    learning_rate_multiplier: 1.0
  }
  parameter_variable {
    variable_name: "BatchNormalization_10/bn/beta"
    learning_rate_multiplier: 1.0
  }
  parameter_variable {
    variable_name: "BatchNormalization_10/bn/gamma"
    learning_rate_multiplier: 1.0
  }
  parameter_variable {
    variable_name: "BatchNormalization_10/bn/mean"
    learning_rate_multiplier: 0.0
  }
  parameter_variable {
    variable_name: "BatchNormalization_10/bn/var"
    learning_rate_multiplier: 0.0
  }
  parameter_variable {
    variable_name: "Convolution_11/conv/W"
    learning_rate_multiplier: 1.0
  }
  parameter_variable {
    variable_name: "BatchNormalization_11/bn/beta"
    learning_rate_multiplier: 1.0
  }
  parameter_variable {
    variable_name: "BatchNormalization_11/bn/gamma"
    learning_rate_multiplier: 1.0
  }
  parameter_variable {
    variable_name: "BatchNormalization_11/bn/mean"
    learning_rate_multiplier: 0.0
  }
  parameter_variable {
    variable_name: "BatchNormalization_11/bn/var"
    learning_rate_multiplier: 0.0
  }
  parameter_variable {
    variable_name: "Convolution_12/conv/W"
    learning_rate_multiplier: 1.0
  }
  parameter_variable {
    variable_name: "BatchNormalization_12/bn/beta"
    learning_rate_multiplier: 1.0
  }
  parameter_variable {
    variable_name: "BatchNormalization_12/bn/gamma"
    learning_rate_multiplier: 1.0
  }
  parameter_variable {
    variable_name: "BatchNormalization_12/bn/mean"
    learning_rate_multiplier: 0.0
  }
  parameter_variable {
    variable_name: "BatchNormalization_12/bn/var"
    learning_rate_multiplier: 0.0
  }
  parameter_variable {
    variable_name: "Convolution_13/conv/W"
    learning_rate_multiplier: 1.0
  }
  parameter_variable {
    variable_name: "BatchNormalization_13/bn/beta"
    learning_rate_multiplier: 1.0
  }
  parameter_variable {
    variable_name: "BatchNormalization_13/bn/gamma"
    learning_rate_multiplier: 1.0
  }
  parameter_variable {
    variable_name: "BatchNormalization_13/bn/mean"
    learning_rate_multiplier: 0.0
  }
  parameter_variable {
    variable_name: "BatchNormalization_13/bn/var"
    learning_rate_multiplier: 0.0
  }
  parameter_variable {
    variable_name: "Affine/affine/W"
    learning_rate_multiplier: 1.0
  }
  parameter_variable {
    variable_name: "Affine/affine/b"
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
  data_variable {
    variable_name: "CategoricalCrossEntropy_T"
    data_name: "y"
  }
  monitor_variable {
    type: "Error"
    variable_name: "CategoricalCrossEntropy"
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
    variable_name: "CategoricalCrossEntropy_T"
    data_name: "y"
  }
  monitor_variable {
    type: "Error"
    variable_name: "CategoricalCrossEntropy"
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
    variable_name: "Softmax"
    data_name: "y'"
  }
  parameter_variable {
    variable_name: "Convolution/conv/W"
  }
  parameter_variable {
    variable_name: "BatchNormalization/bn/beta"
  }
  parameter_variable {
    variable_name: "BatchNormalization/bn/gamma"
  }
  parameter_variable {
    variable_name: "BatchNormalization/bn/mean"
  }
  parameter_variable {
    variable_name: "BatchNormalization/bn/var"
  }
  parameter_variable {
    variable_name: "Convolution_2/conv/W"
  }
  parameter_variable {
    variable_name: "BatchNormalization_2/bn/beta"
  }
  parameter_variable {
    variable_name: "BatchNormalization_2/bn/gamma"
  }
  parameter_variable {
    variable_name: "BatchNormalization_2/bn/mean"
  }
  parameter_variable {
    variable_name: "BatchNormalization_2/bn/var"
  }
  parameter_variable {
    variable_name: "Convolution_3/conv/W"
  }
  parameter_variable {
    variable_name: "BatchNormalization_3/bn/beta"
  }
  parameter_variable {
    variable_name: "BatchNormalization_3/bn/gamma"
  }
  parameter_variable {
    variable_name: "BatchNormalization_3/bn/mean"
  }
  parameter_variable {
    variable_name: "BatchNormalization_3/bn/var"
  }
  parameter_variable {
    variable_name: "Convolution_4/conv/W"
  }
  parameter_variable {
    variable_name: "BatchNormalization_4/bn/beta"
  }
  parameter_variable {
    variable_name: "BatchNormalization_4/bn/gamma"
  }
  parameter_variable {
    variable_name: "BatchNormalization_4/bn/mean"
  }
  parameter_variable {
    variable_name: "BatchNormalization_4/bn/var"
  }
  parameter_variable {
    variable_name: "Convolution_5/conv/W"
  }
  parameter_variable {
    variable_name: "BatchNormalization_5/bn/beta"
  }
  parameter_variable {
    variable_name: "BatchNormalization_5/bn/gamma"
  }
  parameter_variable {
    variable_name: "BatchNormalization_5/bn/mean"
  }
  parameter_variable {
    variable_name: "BatchNormalization_5/bn/var"
  }
  parameter_variable {
    variable_name: "Convolution_6/conv/W"
  }
  parameter_variable {
    variable_name: "BatchNormalization_6/bn/beta"
  }
  parameter_variable {
    variable_name: "BatchNormalization_6/bn/gamma"
  }
  parameter_variable {
    variable_name: "BatchNormalization_6/bn/mean"
  }
  parameter_variable {
    variable_name: "BatchNormalization_6/bn/var"
  }
  parameter_variable {
    variable_name: "Convolution_7/conv/W"
  }
  parameter_variable {
    variable_name: "BatchNormalization_7/bn/beta"
  }
  parameter_variable {
    variable_name: "BatchNormalization_7/bn/gamma"
  }
  parameter_variable {
    variable_name: "BatchNormalization_7/bn/mean"
  }
  parameter_variable {
    variable_name: "BatchNormalization_7/bn/var"
  }
  parameter_variable {
    variable_name: "Convolution_8/conv/W"
  }
  parameter_variable {
    variable_name: "BatchNormalization_8/bn/beta"
  }
  parameter_variable {
    variable_name: "BatchNormalization_8/bn/gamma"
  }
  parameter_variable {
    variable_name: "BatchNormalization_8/bn/mean"
  }
  parameter_variable {
    variable_name: "BatchNormalization_8/bn/var"
  }
  parameter_variable {
    variable_name: "Convolution_9/conv/W"
  }
  parameter_variable {
    variable_name: "BatchNormalization_9/bn/beta"
  }
  parameter_variable {
    variable_name: "BatchNormalization_9/bn/gamma"
  }
  parameter_variable {
    variable_name: "BatchNormalization_9/bn/mean"
  }
  parameter_variable {
    variable_name: "BatchNormalization_9/bn/var"
  }
  parameter_variable {
    variable_name: "Convolution_10/conv/W"
  }
  parameter_variable {
    variable_name: "BatchNormalization_10/bn/beta"
  }
  parameter_variable {
    variable_name: "BatchNormalization_10/bn/gamma"
  }
  parameter_variable {
    variable_name: "BatchNormalization_10/bn/mean"
  }
  parameter_variable {
    variable_name: "BatchNormalization_10/bn/var"
  }
  parameter_variable {
    variable_name: "Convolution_11/conv/W"
  }
  parameter_variable {
    variable_name: "BatchNormalization_11/bn/beta"
  }
  parameter_variable {
    variable_name: "BatchNormalization_11/bn/gamma"
  }
  parameter_variable {
    variable_name: "BatchNormalization_11/bn/mean"
  }
  parameter_variable {
    variable_name: "BatchNormalization_11/bn/var"
  }
  parameter_variable {
    variable_name: "Convolution_12/conv/W"
  }
  parameter_variable {
    variable_name: "BatchNormalization_12/bn/beta"
  }
  parameter_variable {
    variable_name: "BatchNormalization_12/bn/gamma"
  }
  parameter_variable {
    variable_name: "BatchNormalization_12/bn/mean"
  }
  parameter_variable {
    variable_name: "BatchNormalization_12/bn/var"
  }
  parameter_variable {
    variable_name: "Convolution_13/conv/W"
  }
  parameter_variable {
    variable_name: "BatchNormalization_13/bn/beta"
  }
  parameter_variable {
    variable_name: "BatchNormalization_13/bn/gamma"
  }
  parameter_variable {
    variable_name: "BatchNormalization_13/bn/mean"
  }
  parameter_variable {
    variable_name: "BatchNormalization_13/bn/var"
  }
  parameter_variable {
    variable_name: "Affine/affine/W"
  }
  parameter_variable {
    variable_name: "Affine/affine/b"
  }
}
'''
# The nntxt is generated from /home/woody/aid/nnabla-sample-data/sample_project/tutorial/basics/12_residual_learning.files/20180720_182255/net.nntxt.
N00000020 = r'''global_config {
  default_context {
    array_class: "CudaCachedArray"
    device_id: "0"
    backends: "cudnn:float"
    backends: "cuda:float"
    backends: "cpu:float"
  }
}
training_config {
  max_epoch: 10
  iter_per_epoch: 937
  save_best: true
  monitor_interval: 10
}
network {
  name: "Main"
  batch_size: 64
  repeat_info {
    id: "RepeatStart"
    times: 2
  }
  repeat_info {
    id: "RepeatStart_2"
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
    shape: { dim: 64 dim: 1 dim: 5 dim: 5 }
    initializer {
      type: "NormalConvolutionGlorot"
      multiplier: 1
    }
  }
  variable {
    name: "BatchNormalization/bn/beta"
    type: "Parameter"
    shape: { dim: 1 dim: 64 dim: 1 dim: 1 }
    initializer {
      type: "Constant"
      multiplier: 0
    }
  }
  variable {
    name: "BatchNormalization/bn/gamma"
    type: "Parameter"
    shape: { dim: 1 dim: 64 dim: 1 dim: 1 }
    initializer {
      type: "Constant"
      multiplier: 1
    }
  }
  variable {
    name: "BatchNormalization/bn/mean"
    type: "Parameter"
    shape: { dim: 1 dim: 64 dim: 1 dim: 1 }
    initializer {
      type: "Constant"
      multiplier: 0
    }
  }
  variable {
    name: "BatchNormalization/bn/var"
    type: "Parameter"
    shape: { dim: 1 dim: 64 dim: 1 dim: 1 }
    initializer {
      type: "Constant"
      multiplier: 0
    }
  }
  variable {
    name: "Convolution_2/conv/W"
    type: "Parameter"
    shape: { dim: 64 dim: 64 dim: 1 dim: 1 }
    initializer {
      type: "NormalConvolutionGlorot"
      multiplier: 1
    }
  }
  variable {
    name: "BatchNormalization_2/bn/beta"
    type: "Parameter"
    shape: { dim: 1 dim: 64 dim: 1 dim: 1 }
    initializer {
      type: "Constant"
      multiplier: 0
    }
  }
  variable {
    name: "BatchNormalization_2/bn/gamma"
    type: "Parameter"
    shape: { dim: 1 dim: 64 dim: 1 dim: 1 }
    initializer {
      type: "Constant"
      multiplier: 1
    }
  }
  variable {
    name: "BatchNormalization_2/bn/mean"
    type: "Parameter"
    shape: { dim: 1 dim: 64 dim: 1 dim: 1 }
    initializer {
      type: "Constant"
      multiplier: 0
    }
  }
  variable {
    name: "BatchNormalization_2/bn/var"
    type: "Parameter"
    shape: { dim: 1 dim: 64 dim: 1 dim: 1 }
    initializer {
      type: "Constant"
      multiplier: 0
    }
  }
  variable {
    name: "Convolution_3/conv/W"
    type: "Parameter"
    shape: { dim: 64 dim: 64 dim: 3 dim: 3 }
    initializer {
      type: "NormalConvolutionGlorot"
      multiplier: 1
    }
  }
  variable {
    name: "BatchNormalization_3/bn/beta"
    type: "Parameter"
    shape: { dim: 1 dim: 64 dim: 1 dim: 1 }
    initializer {
      type: "Constant"
      multiplier: 0
    }
  }
  variable {
    name: "BatchNormalization_3/bn/gamma"
    type: "Parameter"
    shape: { dim: 1 dim: 64 dim: 1 dim: 1 }
    initializer {
      type: "Constant"
      multiplier: 1
    }
  }
  variable {
    name: "BatchNormalization_3/bn/mean"
    type: "Parameter"
    shape: { dim: 1 dim: 64 dim: 1 dim: 1 }
    initializer {
      type: "Constant"
      multiplier: 0
    }
  }
  variable {
    name: "BatchNormalization_3/bn/var"
    type: "Parameter"
    shape: { dim: 1 dim: 64 dim: 1 dim: 1 }
    initializer {
      type: "Constant"
      multiplier: 0
    }
  }
  variable {
    name: "Convolution_4/conv/W"
    type: "Parameter"
    shape: { dim: 64 dim: 64 dim: 1 dim: 1 }
    initializer {
      type: "NormalConvolutionGlorot"
      multiplier: 1
    }
  }
  variable {
    name: "BatchNormalization_4/bn/beta"
    type: "Parameter"
    shape: { dim: 1 dim: 64 dim: 1 dim: 1 }
    initializer {
      type: "Constant"
      multiplier: 0
    }
  }
  variable {
    name: "BatchNormalization_4/bn/gamma"
    type: "Parameter"
    shape: { dim: 1 dim: 64 dim: 1 dim: 1 }
    initializer {
      type: "Constant"
      multiplier: 1
    }
  }
  variable {
    name: "BatchNormalization_4/bn/mean"
    type: "Parameter"
    shape: { dim: 1 dim: 64 dim: 1 dim: 1 }
    initializer {
      type: "Constant"
      multiplier: 0
    }
  }
  variable {
    name: "BatchNormalization_4/bn/var"
    type: "Parameter"
    shape: { dim: 1 dim: 64 dim: 1 dim: 1 }
    initializer {
      type: "Constant"
      multiplier: 0
    }
  }
  variable {
    name: "Convolution_5{RepeatStart}/conv/W"
    type: "Parameter"
    repeat_id: "RepeatStart"
    shape: { dim: 64 dim: 64 dim: 1 dim: 1 }
    initializer {
      type: "NormalConvolutionGlorot"
      multiplier: 1
    }
  }
  variable {
    name: "BatchNormalization_5{RepeatStart}/bn/beta"
    type: "Parameter"
    repeat_id: "RepeatStart"
    shape: { dim: 1 dim: 64 dim: 1 dim: 1 }
    initializer {
      type: "Constant"
      multiplier: 0
    }
  }
  variable {
    name: "BatchNormalization_5{RepeatStart}/bn/gamma"
    type: "Parameter"
    repeat_id: "RepeatStart"
    shape: { dim: 1 dim: 64 dim: 1 dim: 1 }
    initializer {
      type: "Constant"
      multiplier: 1
    }
  }
  variable {
    name: "BatchNormalization_5{RepeatStart}/bn/mean"
    type: "Parameter"
    repeat_id: "RepeatStart"
    shape: { dim: 1 dim: 64 dim: 1 dim: 1 }
    initializer {
      type: "Constant"
      multiplier: 0
    }
  }
  variable {
    name: "BatchNormalization_5{RepeatStart}/bn/var"
    type: "Parameter"
    repeat_id: "RepeatStart"
    shape: { dim: 1 dim: 64 dim: 1 dim: 1 }
    initializer {
      type: "Constant"
      multiplier: 0
    }
  }
  variable {
    name: "Convolution_6{RepeatStart}/conv/W"
    type: "Parameter"
    repeat_id: "RepeatStart"
    shape: { dim: 64 dim: 64 dim: 3 dim: 3 }
    initializer {
      type: "NormalConvolutionGlorot"
      multiplier: 1
    }
  }
  variable {
    name: "BatchNormalization_6{RepeatStart}/bn/beta"
    type: "Parameter"
    repeat_id: "RepeatStart"
    shape: { dim: 1 dim: 64 dim: 1 dim: 1 }
    initializer {
      type: "Constant"
      multiplier: 0
    }
  }
  variable {
    name: "BatchNormalization_6{RepeatStart}/bn/gamma"
    type: "Parameter"
    repeat_id: "RepeatStart"
    shape: { dim: 1 dim: 64 dim: 1 dim: 1 }
    initializer {
      type: "Constant"
      multiplier: 1
    }
  }
  variable {
    name: "BatchNormalization_6{RepeatStart}/bn/mean"
    type: "Parameter"
    repeat_id: "RepeatStart"
    shape: { dim: 1 dim: 64 dim: 1 dim: 1 }
    initializer {
      type: "Constant"
      multiplier: 0
    }
  }
  variable {
    name: "BatchNormalization_6{RepeatStart}/bn/var"
    type: "Parameter"
    repeat_id: "RepeatStart"
    shape: { dim: 1 dim: 64 dim: 1 dim: 1 }
    initializer {
      type: "Constant"
      multiplier: 0
    }
  }
  variable {
    name: "Convolution_7{RepeatStart}/conv/W"
    type: "Parameter"
    repeat_id: "RepeatStart"
    shape: { dim: 64 dim: 64 dim: 1 dim: 1 }
    initializer {
      type: "NormalConvolutionGlorot"
      multiplier: 1
    }
  }
  variable {
    name: "BatchNormalization_7{RepeatStart}/bn/beta"
    type: "Parameter"
    repeat_id: "RepeatStart"
    shape: { dim: 1 dim: 64 dim: 1 dim: 1 }
    initializer {
      type: "Constant"
      multiplier: 0
    }
  }
  variable {
    name: "BatchNormalization_7{RepeatStart}/bn/gamma"
    type: "Parameter"
    repeat_id: "RepeatStart"
    shape: { dim: 1 dim: 64 dim: 1 dim: 1 }
    initializer {
      type: "Constant"
      multiplier: 1
    }
  }
  variable {
    name: "BatchNormalization_7{RepeatStart}/bn/mean"
    type: "Parameter"
    repeat_id: "RepeatStart"
    shape: { dim: 1 dim: 64 dim: 1 dim: 1 }
    initializer {
      type: "Constant"
      multiplier: 0
    }
  }
  variable {
    name: "BatchNormalization_7{RepeatStart}/bn/var"
    type: "Parameter"
    repeat_id: "RepeatStart"
    shape: { dim: 1 dim: 64 dim: 1 dim: 1 }
    initializer {
      type: "Constant"
      multiplier: 0
    }
  }
  variable {
    name: "Convolution_8{RepeatStart_2}/conv/W"
    type: "Parameter"
    repeat_id: "RepeatStart_2"
    shape: { dim: 64 dim: 64 dim: 1 dim: 1 }
    initializer {
      type: "NormalConvolutionGlorot"
      multiplier: 1
    }
  }
  variable {
    name: "BatchNormalization_8{RepeatStart_2}/bn/beta"
    type: "Parameter"
    repeat_id: "RepeatStart_2"
    shape: { dim: 1 dim: 64 dim: 1 dim: 1 }
    initializer {
      type: "Constant"
      multiplier: 0
    }
  }
  variable {
    name: "BatchNormalization_8{RepeatStart_2}/bn/gamma"
    type: "Parameter"
    repeat_id: "RepeatStart_2"
    shape: { dim: 1 dim: 64 dim: 1 dim: 1 }
    initializer {
      type: "Constant"
      multiplier: 1
    }
  }
  variable {
    name: "BatchNormalization_8{RepeatStart_2}/bn/mean"
    type: "Parameter"
    repeat_id: "RepeatStart_2"
    shape: { dim: 1 dim: 64 dim: 1 dim: 1 }
    initializer {
      type: "Constant"
      multiplier: 0
    }
  }
  variable {
    name: "BatchNormalization_8{RepeatStart_2}/bn/var"
    type: "Parameter"
    repeat_id: "RepeatStart_2"
    shape: { dim: 1 dim: 64 dim: 1 dim: 1 }
    initializer {
      type: "Constant"
      multiplier: 0
    }
  }
  variable {
    name: "Convolution_9{RepeatStart_2}/conv/W"
    type: "Parameter"
    repeat_id: "RepeatStart_2"
    shape: { dim: 64 dim: 64 dim: 3 dim: 3 }
    initializer {
      type: "NormalConvolutionGlorot"
      multiplier: 1
    }
  }
  variable {
    name: "BatchNormalization_9{RepeatStart_2}/bn/beta"
    type: "Parameter"
    repeat_id: "RepeatStart_2"
    shape: { dim: 1 dim: 64 dim: 1 dim: 1 }
    initializer {
      type: "Constant"
      multiplier: 0
    }
  }
  variable {
    name: "BatchNormalization_9{RepeatStart_2}/bn/gamma"
    type: "Parameter"
    repeat_id: "RepeatStart_2"
    shape: { dim: 1 dim: 64 dim: 1 dim: 1 }
    initializer {
      type: "Constant"
      multiplier: 1
    }
  }
  variable {
    name: "BatchNormalization_9{RepeatStart_2}/bn/mean"
    type: "Parameter"
    repeat_id: "RepeatStart_2"
    shape: { dim: 1 dim: 64 dim: 1 dim: 1 }
    initializer {
      type: "Constant"
      multiplier: 0
    }
  }
  variable {
    name: "BatchNormalization_9{RepeatStart_2}/bn/var"
    type: "Parameter"
    repeat_id: "RepeatStart_2"
    shape: { dim: 1 dim: 64 dim: 1 dim: 1 }
    initializer {
      type: "Constant"
      multiplier: 0
    }
  }
  variable {
    name: "Convolution_10{RepeatStart_2}/conv/W"
    type: "Parameter"
    repeat_id: "RepeatStart_2"
    shape: { dim: 64 dim: 64 dim: 1 dim: 1 }
    initializer {
      type: "NormalConvolutionGlorot"
      multiplier: 1
    }
  }
  variable {
    name: "BatchNormalization_10{RepeatStart_2}/bn/beta"
    type: "Parameter"
    repeat_id: "RepeatStart_2"
    shape: { dim: 1 dim: 64 dim: 1 dim: 1 }
    initializer {
      type: "Constant"
      multiplier: 0
    }
  }
  variable {
    name: "BatchNormalization_10{RepeatStart_2}/bn/gamma"
    type: "Parameter"
    repeat_id: "RepeatStart_2"
    shape: { dim: 1 dim: 64 dim: 1 dim: 1 }
    initializer {
      type: "Constant"
      multiplier: 1
    }
  }
  variable {
    name: "BatchNormalization_10{RepeatStart_2}/bn/mean"
    type: "Parameter"
    repeat_id: "RepeatStart_2"
    shape: { dim: 1 dim: 64 dim: 1 dim: 1 }
    initializer {
      type: "Constant"
      multiplier: 0
    }
  }
  variable {
    name: "BatchNormalization_10{RepeatStart_2}/bn/var"
    type: "Parameter"
    repeat_id: "RepeatStart_2"
    shape: { dim: 1 dim: 64 dim: 1 dim: 1 }
    initializer {
      type: "Constant"
      multiplier: 0
    }
  }
  variable {
    name: "Convolution_11/conv/W"
    type: "Parameter"
    shape: { dim: 64 dim: 64 dim: 1 dim: 1 }
    initializer {
      type: "NormalConvolutionGlorot"
      multiplier: 1
    }
  }
  variable {
    name: "BatchNormalization_11/bn/beta"
    type: "Parameter"
    shape: { dim: 1 dim: 64 dim: 1 dim: 1 }
    initializer {
      type: "Constant"
      multiplier: 0
    }
  }
  variable {
    name: "BatchNormalization_11/bn/gamma"
    type: "Parameter"
    shape: { dim: 1 dim: 64 dim: 1 dim: 1 }
    initializer {
      type: "Constant"
      multiplier: 1
    }
  }
  variable {
    name: "BatchNormalization_11/bn/mean"
    type: "Parameter"
    shape: { dim: 1 dim: 64 dim: 1 dim: 1 }
    initializer {
      type: "Constant"
      multiplier: 0
    }
  }
  variable {
    name: "BatchNormalization_11/bn/var"
    type: "Parameter"
    shape: { dim: 1 dim: 64 dim: 1 dim: 1 }
    initializer {
      type: "Constant"
      multiplier: 0
    }
  }
  variable {
    name: "Convolution_12/conv/W"
    type: "Parameter"
    shape: { dim: 64 dim: 64 dim: 3 dim: 3 }
    initializer {
      type: "NormalConvolutionGlorot"
      multiplier: 1
    }
  }
  variable {
    name: "BatchNormalization_12/bn/beta"
    type: "Parameter"
    shape: { dim: 1 dim: 64 dim: 1 dim: 1 }
    initializer {
      type: "Constant"
      multiplier: 0
    }
  }
  variable {
    name: "BatchNormalization_12/bn/gamma"
    type: "Parameter"
    shape: { dim: 1 dim: 64 dim: 1 dim: 1 }
    initializer {
      type: "Constant"
      multiplier: 1
    }
  }
  variable {
    name: "BatchNormalization_12/bn/mean"
    type: "Parameter"
    shape: { dim: 1 dim: 64 dim: 1 dim: 1 }
    initializer {
      type: "Constant"
      multiplier: 0
    }
  }
  variable {
    name: "BatchNormalization_12/bn/var"
    type: "Parameter"
    shape: { dim: 1 dim: 64 dim: 1 dim: 1 }
    initializer {
      type: "Constant"
      multiplier: 0
    }
  }
  variable {
    name: "Convolution_13/conv/W"
    type: "Parameter"
    shape: { dim: 64 dim: 64 dim: 1 dim: 1 }
    initializer {
      type: "NormalConvolutionGlorot"
      multiplier: 1
    }
  }
  variable {
    name: "BatchNormalization_13/bn/beta"
    type: "Parameter"
    shape: { dim: 1 dim: 64 dim: 1 dim: 1 }
    initializer {
      type: "Constant"
      multiplier: 0
    }
  }
  variable {
    name: "BatchNormalization_13/bn/gamma"
    type: "Parameter"
    shape: { dim: 1 dim: 64 dim: 1 dim: 1 }
    initializer {
      type: "Constant"
      multiplier: 1
    }
  }
  variable {
    name: "BatchNormalization_13/bn/mean"
    type: "Parameter"
    shape: { dim: 1 dim: 64 dim: 1 dim: 1 }
    initializer {
      type: "Constant"
      multiplier: 0
    }
  }
  variable {
    name: "BatchNormalization_13/bn/var"
    type: "Parameter"
    shape: { dim: 1 dim: 64 dim: 1 dim: 1 }
    initializer {
      type: "Constant"
      multiplier: 0
    }
  }
  variable {
    name: "Affine/affine/W"
    type: "Parameter"
    shape: { dim: 64 dim: 10 }
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
    name: "CategoricalCrossEntropy_T"
    type: "Buffer"
    shape: { dim:-1 dim: 1 }
  }
  variable {
    name: "Convolution"
    type: "Buffer"
    shape: { dim:-1 dim: 64 dim: 32 dim: 32 }
  }
  variable {
    name: "BatchNormalization"
    type: "Buffer"
    shape: { dim:-1 dim: 64 dim: 32 dim: 32 }
  }
  variable {
    name: "ReLU"
    type: "Buffer"
    shape: { dim:-1 dim: 64 dim: 32 dim: 32 }
  }
  variable {
    name: "Convolution_2"
    type: "Buffer"
    shape: { dim:-1 dim: 64 dim: 32 dim: 32 }
  }
  variable {
    name: "BatchNormalization_2"
    type: "Buffer"
    shape: { dim:-1 dim: 64 dim: 32 dim: 32 }
  }
  variable {
    name: "ReLU_2"
    type: "Buffer"
    shape: { dim:-1 dim: 64 dim: 32 dim: 32 }
  }
  variable {
    name: "Convolution_3"
    type: "Buffer"
    shape: { dim:-1 dim: 64 dim: 32 dim: 32 }
  }
  variable {
    name: "BatchNormalization_3"
    type: "Buffer"
    shape: { dim:-1 dim: 64 dim: 32 dim: 32 }
  }
  variable {
    name: "ReLU_3"
    type: "Buffer"
    shape: { dim:-1 dim: 64 dim: 32 dim: 32 }
  }
  variable {
    name: "Convolution_4"
    type: "Buffer"
    shape: { dim:-1 dim: 64 dim: 32 dim: 32 }
  }
  variable {
    name: "BatchNormalization_4"
    type: "Buffer"
    shape: { dim:-1 dim: 64 dim: 32 dim: 32 }
  }
  variable {
    name: "Add2"
    type: "Buffer"
    shape: { dim:-1 dim: 64 dim: 32 dim: 32 }
  }
  variable {
    name: "ReLU_4"
    type: "Buffer"
    shape: { dim:-1 dim: 64 dim: 32 dim: 32 }
  }
  variable {
    name: "MaxPooling"
    type: "Buffer"
    shape: { dim:-1 dim: 64 dim: 16 dim: 16 }
  }
  variable {
    name: "RepeatStart"
    type: "Buffer"
    repeat_id: "RepeatStart"
    shape: { dim:-1 dim: 64 dim: 16 dim: 16 }
  }
  variable {
    name: "Convolution_5"
    type: "Buffer"
    repeat_id: "RepeatStart"
    shape: { dim:-1 dim: 64 dim: 16 dim: 16 }
  }
  variable {
    name: "BatchNormalization_5"
    type: "Buffer"
    repeat_id: "RepeatStart"
    shape: { dim:-1 dim: 64 dim: 16 dim: 16 }
  }
  variable {
    name: "ReLU_5"
    type: "Buffer"
    repeat_id: "RepeatStart"
    shape: { dim:-1 dim: 64 dim: 16 dim: 16 }
  }
  variable {
    name: "Convolution_6"
    type: "Buffer"
    repeat_id: "RepeatStart"
    shape: { dim:-1 dim: 64 dim: 16 dim: 16 }
  }
  variable {
    name: "BatchNormalization_6"
    type: "Buffer"
    repeat_id: "RepeatStart"
    shape: { dim:-1 dim: 64 dim: 16 dim: 16 }
  }
  variable {
    name: "ReLU_6"
    type: "Buffer"
    repeat_id: "RepeatStart"
    shape: { dim:-1 dim: 64 dim: 16 dim: 16 }
  }
  variable {
    name: "Convolution_7"
    type: "Buffer"
    repeat_id: "RepeatStart"
    shape: { dim:-1 dim: 64 dim: 16 dim: 16 }
  }
  variable {
    name: "BatchNormalization_7"
    type: "Buffer"
    repeat_id: "RepeatStart"
    shape: { dim:-1 dim: 64 dim: 16 dim: 16 }
  }
  variable {
    name: "Add2_2"
    type: "Buffer"
    repeat_id: "RepeatStart"
    shape: { dim:-1 dim: 64 dim: 16 dim: 16 }
  }
  variable {
    name: "ReLU_7"
    type: "Buffer"
    repeat_id: "RepeatStart"
    shape: { dim:-1 dim: 64 dim: 16 dim: 16 }
  }
  variable {
    name: "RepeatEnd"
    type: "Buffer"
    shape: { dim:-1 dim: 64 dim: 16 dim: 16 }
  }
  variable {
    name: "MaxPooling_2"
    type: "Buffer"
    shape: { dim:-1 dim: 64 dim: 8 dim: 8 }
  }
  variable {
    name: "RepeatStart_2"
    type: "Buffer"
    repeat_id: "RepeatStart_2"
    shape: { dim:-1 dim: 64 dim: 8 dim: 8 }
  }
  variable {
    name: "Convolution_8"
    type: "Buffer"
    repeat_id: "RepeatStart_2"
    shape: { dim:-1 dim: 64 dim: 8 dim: 8 }
  }
  variable {
    name: "BatchNormalization_8"
    type: "Buffer"
    repeat_id: "RepeatStart_2"
    shape: { dim:-1 dim: 64 dim: 8 dim: 8 }
  }
  variable {
    name: "ReLU_8"
    type: "Buffer"
    repeat_id: "RepeatStart_2"
    shape: { dim:-1 dim: 64 dim: 8 dim: 8 }
  }
  variable {
    name: "Convolution_9"
    type: "Buffer"
    repeat_id: "RepeatStart_2"
    shape: { dim:-1 dim: 64 dim: 8 dim: 8 }
  }
  variable {
    name: "BatchNormalization_9"
    type: "Buffer"
    repeat_id: "RepeatStart_2"
    shape: { dim:-1 dim: 64 dim: 8 dim: 8 }
  }
  variable {
    name: "ReLU_9"
    type: "Buffer"
    repeat_id: "RepeatStart_2"
    shape: { dim:-1 dim: 64 dim: 8 dim: 8 }
  }
  variable {
    name: "Convolution_10"
    type: "Buffer"
    repeat_id: "RepeatStart_2"
    shape: { dim:-1 dim: 64 dim: 8 dim: 8 }
  }
  variable {
    name: "BatchNormalization_10"
    type: "Buffer"
    repeat_id: "RepeatStart_2"
    shape: { dim:-1 dim: 64 dim: 8 dim: 8 }
  }
  variable {
    name: "Add2_3"
    type: "Buffer"
    repeat_id: "RepeatStart_2"
    shape: { dim:-1 dim: 64 dim: 8 dim: 8 }
  }
  variable {
    name: "ReLU_10"
    type: "Buffer"
    repeat_id: "RepeatStart_2"
    shape: { dim:-1 dim: 64 dim: 8 dim: 8 }
  }
  variable {
    name: "RepeatEnd_2"
    type: "Buffer"
    shape: { dim:-1 dim: 64 dim: 8 dim: 8 }
  }
  variable {
    name: "MaxPooling_3"
    type: "Buffer"
    shape: { dim:-1 dim: 64 dim: 4 dim: 4 }
  }
  variable {
    name: "Convolution_11"
    type: "Buffer"
    shape: { dim:-1 dim: 64 dim: 4 dim: 4 }
  }
  variable {
    name: "BatchNormalization_11"
    type: "Buffer"
    shape: { dim:-1 dim: 64 dim: 4 dim: 4 }
  }
  variable {
    name: "ReLU_11"
    type: "Buffer"
    shape: { dim:-1 dim: 64 dim: 4 dim: 4 }
  }
  variable {
    name: "Convolution_12"
    type: "Buffer"
    shape: { dim:-1 dim: 64 dim: 4 dim: 4 }
  }
  variable {
    name: "BatchNormalization_12"
    type: "Buffer"
    shape: { dim:-1 dim: 64 dim: 4 dim: 4 }
  }
  variable {
    name: "ReLU_12"
    type: "Buffer"
    shape: { dim:-1 dim: 64 dim: 4 dim: 4 }
  }
  variable {
    name: "Convolution_13"
    type: "Buffer"
    shape: { dim:-1 dim: 64 dim: 4 dim: 4 }
  }
  variable {
    name: "BatchNormalization_13"
    type: "Buffer"
    shape: { dim:-1 dim: 64 dim: 4 dim: 4 }
  }
  variable {
    name: "Add2_4"
    type: "Buffer"
    shape: { dim:-1 dim: 64 dim: 4 dim: 4 }
  }
  variable {
    name: "ReLU_13"
    type: "Buffer"
    shape: { dim:-1 dim: 64 dim: 4 dim: 4 }
  }
  variable {
    name: "AveragePooling"
    type: "Buffer"
    shape: { dim:-1 dim: 64 dim: 1 dim: 1 }
  }
  variable {
    name: "Affine"
    type: "Buffer"
    shape: { dim:-1 dim: 10 }
  }
  variable {
    name: "Softmax"
    type: "Buffer"
    shape: { dim:-1 dim: 10 }
  }
  variable {
    name: "CategoricalCrossEntropy"
    type: "Buffer"
    shape: { dim:-1 dim: 1 }
  }
  function {
    name: "Convolution"
    type: "Convolution"
    input: "Input"
    input: "Convolution/conv/W"
    output: "Convolution"
    convolution_param {
      pad: { dim: 4 dim: 4 }
      stride: { dim: 1 dim: 1 }
      dilation: { dim: 1 dim: 1 }
      group: 1
      base_axis: 1
    }
  }
  function {
    name: "BatchNormalization"
    type: "BatchNormalization"
    input: "Convolution"
    input: "BatchNormalization/bn/beta"
    input: "BatchNormalization/bn/gamma"
    input: "BatchNormalization/bn/mean"
    input: "BatchNormalization/bn/var"
    output: "BatchNormalization"
    batch_normalization_param {
      axes: 1
      decay_rate: 0.5
      eps: 0.01
      batch_stat: True
    }
  }
  function {
    name: "ReLU"
    type: "ReLU"
    input: "BatchNormalization"
    output: "ReLU"
    relu_param {
      inplace: True
    }
  }
  function {
    name: "Convolution_2"
    type: "Convolution"
    input: "ReLU"
    input: "Convolution_2/conv/W"
    output: "Convolution_2"
    convolution_param {
      pad: { dim: 0 dim: 0 }
      stride: { dim: 1 dim: 1 }
      dilation: { dim: 1 dim: 1 }
      group: 1
      base_axis: 1
    }
  }
  function {
    name: "BatchNormalization_2"
    type: "BatchNormalization"
    input: "Convolution_2"
    input: "BatchNormalization_2/bn/beta"
    input: "BatchNormalization_2/bn/gamma"
    input: "BatchNormalization_2/bn/mean"
    input: "BatchNormalization_2/bn/var"
    output: "BatchNormalization_2"
    batch_normalization_param {
      axes: 1
      decay_rate: 0.5
      eps: 0.01
      batch_stat: True
    }
  }
  function {
    name: "ReLU_2"
    type: "ReLU"
    input: "BatchNormalization_2"
    output: "ReLU_2"
    relu_param {
      inplace: True
    }
  }
  function {
    name: "Convolution_3"
    type: "Convolution"
    input: "ReLU_2"
    input: "Convolution_3/conv/W"
    output: "Convolution_3"
    convolution_param {
      pad: { dim: 1 dim: 1 }
      stride: { dim: 1 dim: 1 }
      dilation: { dim: 1 dim: 1 }
      group: 1
      base_axis: 1
    }
  }
  function {
    name: "BatchNormalization_3"
    type: "BatchNormalization"
    input: "Convolution_3"
    input: "BatchNormalization_3/bn/beta"
    input: "BatchNormalization_3/bn/gamma"
    input: "BatchNormalization_3/bn/mean"
    input: "BatchNormalization_3/bn/var"
    output: "BatchNormalization_3"
    batch_normalization_param {
      axes: 1
      decay_rate: 0.5
      eps: 0.01
      batch_stat: True
    }
  }
  function {
    name: "ReLU_3"
    type: "ReLU"
    input: "BatchNormalization_3"
    output: "ReLU_3"
    relu_param {
      inplace: True
    }
  }
  function {
    name: "Convolution_4"
    type: "Convolution"
    input: "ReLU_3"
    input: "Convolution_4/conv/W"
    output: "Convolution_4"
    convolution_param {
      pad: { dim: 0 dim: 0 }
      stride: { dim: 1 dim: 1 }
      dilation: { dim: 1 dim: 1 }
      group: 1
      base_axis: 1
    }
  }
  function {
    name: "BatchNormalization_4"
    type: "BatchNormalization"
    input: "Convolution_4"
    input: "BatchNormalization_4/bn/beta"
    input: "BatchNormalization_4/bn/gamma"
    input: "BatchNormalization_4/bn/mean"
    input: "BatchNormalization_4/bn/var"
    output: "BatchNormalization_4"
    batch_normalization_param {
      axes: 1
      decay_rate: 0.5
      eps: 0.01
      batch_stat: True
    }
  }
  function {
    name: "Add2"
    type: "Add2"
    input: "BatchNormalization_4"
    input: "ReLU"
    output: "Add2"
    add2_param {
      inplace: False
    }
  }
  function {
    name: "ReLU_4"
    type: "ReLU"
    input: "Add2"
    output: "ReLU_4"
    relu_param {
      inplace: True
    }
  }
  function {
    name: "MaxPooling"
    type: "MaxPooling"
    input: "ReLU_4"
    output: "MaxPooling"
    max_pooling_param {
      kernel: { dim: 2 dim: 2 }
      stride: { dim: 2 dim: 2 }
      ignore_border: true
      pad: { dim: 0 dim: 0 }
    }
  }
  function {
    name: "RepeatStart"
    type: "RepeatStart"
    repeat_id: "RepeatStart"
    input: "MaxPooling"
    input: "ReLU_7"
    output: "RepeatStart"
    repeat_param {
      repeat_id: "RepeatStart"
    }
  }
  function {
    name: "Convolution_5"
    type: "Convolution"
    repeat_id: "RepeatStart"
    input: "RepeatStart"
    input: "Convolution_5{RepeatStart}/conv/W"
    output: "Convolution_5"
    convolution_param {
      pad: { dim: 0 dim: 0 }
      stride: { dim: 1 dim: 1 }
      dilation: { dim: 1 dim: 1 }
      group: 1
      base_axis: 1
    }
  }
  function {
    name: "BatchNormalization_5"
    type: "BatchNormalization"
    repeat_id: "RepeatStart"
    input: "Convolution_5"
    input: "BatchNormalization_5{RepeatStart}/bn/beta"
    input: "BatchNormalization_5{RepeatStart}/bn/gamma"
    input: "BatchNormalization_5{RepeatStart}/bn/mean"
    input: "BatchNormalization_5{RepeatStart}/bn/var"
    output: "BatchNormalization_5"
    batch_normalization_param {
      axes: 1
      decay_rate: 0.5
      eps: 0.01
      batch_stat: True
    }
  }
  function {
    name: "ReLU_5"
    type: "ReLU"
    repeat_id: "RepeatStart"
    input: "BatchNormalization_5"
    output: "ReLU_5"
    relu_param {
      inplace: True
    }
  }
  function {
    name: "Convolution_6"
    type: "Convolution"
    repeat_id: "RepeatStart"
    input: "ReLU_5"
    input: "Convolution_6{RepeatStart}/conv/W"
    output: "Convolution_6"
    convolution_param {
      pad: { dim: 1 dim: 1 }
      stride: { dim: 1 dim: 1 }
      dilation: { dim: 1 dim: 1 }
      group: 1
      base_axis: 1
    }
  }
  function {
    name: "BatchNormalization_6"
    type: "BatchNormalization"
    repeat_id: "RepeatStart"
    input: "Convolution_6"
    input: "BatchNormalization_6{RepeatStart}/bn/beta"
    input: "BatchNormalization_6{RepeatStart}/bn/gamma"
    input: "BatchNormalization_6{RepeatStart}/bn/mean"
    input: "BatchNormalization_6{RepeatStart}/bn/var"
    output: "BatchNormalization_6"
    batch_normalization_param {
      axes: 1
      decay_rate: 0.5
      eps: 0.01
      batch_stat: True
    }
  }
  function {
    name: "ReLU_6"
    type: "ReLU"
    repeat_id: "RepeatStart"
    input: "BatchNormalization_6"
    output: "ReLU_6"
    relu_param {
      inplace: True
    }
  }
  function {
    name: "Convolution_7"
    type: "Convolution"
    repeat_id: "RepeatStart"
    input: "ReLU_6"
    input: "Convolution_7{RepeatStart}/conv/W"
    output: "Convolution_7"
    convolution_param {
      pad: { dim: 0 dim: 0 }
      stride: { dim: 1 dim: 1 }
      dilation: { dim: 1 dim: 1 }
      group: 1
      base_axis: 1
    }
  }
  function {
    name: "BatchNormalization_7"
    type: "BatchNormalization"
    repeat_id: "RepeatStart"
    input: "Convolution_7"
    input: "BatchNormalization_7{RepeatStart}/bn/beta"
    input: "BatchNormalization_7{RepeatStart}/bn/gamma"
    input: "BatchNormalization_7{RepeatStart}/bn/mean"
    input: "BatchNormalization_7{RepeatStart}/bn/var"
    output: "BatchNormalization_7"
    batch_normalization_param {
      axes: 1
      decay_rate: 0.5
      eps: 0.01
      batch_stat: True
    }
  }
  function {
    name: "Add2_2"
    type: "Add2"
    repeat_id: "RepeatStart"
    input: "BatchNormalization_7"
    input: "RepeatStart"
    output: "Add2_2"
    add2_param {
      inplace: False
    }
  }
  function {
    name: "ReLU_7"
    type: "ReLU"
    repeat_id: "RepeatStart"
    input: "Add2_2"
    output: "ReLU_7"
    relu_param {
      inplace: True
    }
  }
  function {
    name: "RepeatEnd"
    type: "RepeatEnd"
    input: "ReLU_7"
    output: "RepeatEnd"
    repeat_param {
      repeat_id: "RepeatStart"
      times: 2
    }
  }
  function {
    name: "MaxPooling_2"
    type: "MaxPooling"
    input: "RepeatEnd"
    output: "MaxPooling_2"
    max_pooling_param {
      kernel: { dim: 2 dim: 2 }
      stride: { dim: 2 dim: 2 }
      ignore_border: true
      pad: { dim: 0 dim: 0 }
    }
  }
  function {
    name: "RepeatStart_2"
    type: "RepeatStart"
    repeat_id: "RepeatStart_2"
    input: "MaxPooling_2"
    input: "ReLU_10"
    output: "RepeatStart_2"
    repeat_param {
      repeat_id: "RepeatStart_2"
    }
  }
  function {
    name: "Convolution_8"
    type: "Convolution"
    repeat_id: "RepeatStart_2"
    input: "RepeatStart_2"
    input: "Convolution_8{RepeatStart_2}/conv/W"
    output: "Convolution_8"
    convolution_param {
      pad: { dim: 0 dim: 0 }
      stride: { dim: 1 dim: 1 }
      dilation: { dim: 1 dim: 1 }
      group: 1
      base_axis: 1
    }
  }
  function {
    name: "BatchNormalization_8"
    type: "BatchNormalization"
    repeat_id: "RepeatStart_2"
    input: "Convolution_8"
    input: "BatchNormalization_8{RepeatStart_2}/bn/beta"
    input: "BatchNormalization_8{RepeatStart_2}/bn/gamma"
    input: "BatchNormalization_8{RepeatStart_2}/bn/mean"
    input: "BatchNormalization_8{RepeatStart_2}/bn/var"
    output: "BatchNormalization_8"
    batch_normalization_param {
      axes: 1
      decay_rate: 0.5
      eps: 0.01
      batch_stat: True
    }
  }
  function {
    name: "ReLU_8"
    type: "ReLU"
    repeat_id: "RepeatStart_2"
    input: "BatchNormalization_8"
    output: "ReLU_8"
    relu_param {
      inplace: True
    }
  }
  function {
    name: "Convolution_9"
    type: "Convolution"
    repeat_id: "RepeatStart_2"
    input: "ReLU_8"
    input: "Convolution_9{RepeatStart_2}/conv/W"
    output: "Convolution_9"
    convolution_param {
      pad: { dim: 1 dim: 1 }
      stride: { dim: 1 dim: 1 }
      dilation: { dim: 1 dim: 1 }
      group: 1
      base_axis: 1
    }
  }
  function {
    name: "BatchNormalization_9"
    type: "BatchNormalization"
    repeat_id: "RepeatStart_2"
    input: "Convolution_9"
    input: "BatchNormalization_9{RepeatStart_2}/bn/beta"
    input: "BatchNormalization_9{RepeatStart_2}/bn/gamma"
    input: "BatchNormalization_9{RepeatStart_2}/bn/mean"
    input: "BatchNormalization_9{RepeatStart_2}/bn/var"
    output: "BatchNormalization_9"
    batch_normalization_param {
      axes: 1
      decay_rate: 0.5
      eps: 0.01
      batch_stat: True
    }
  }
  function {
    name: "ReLU_9"
    type: "ReLU"
    repeat_id: "RepeatStart_2"
    input: "BatchNormalization_9"
    output: "ReLU_9"
    relu_param {
      inplace: True
    }
  }
  function {
    name: "Convolution_10"
    type: "Convolution"
    repeat_id: "RepeatStart_2"
    input: "ReLU_9"
    input: "Convolution_10{RepeatStart_2}/conv/W"
    output: "Convolution_10"
    convolution_param {
      pad: { dim: 0 dim: 0 }
      stride: { dim: 1 dim: 1 }
      dilation: { dim: 1 dim: 1 }
      group: 1
      base_axis: 1
    }
  }
  function {
    name: "BatchNormalization_10"
    type: "BatchNormalization"
    repeat_id: "RepeatStart_2"
    input: "Convolution_10"
    input: "BatchNormalization_10{RepeatStart_2}/bn/beta"
    input: "BatchNormalization_10{RepeatStart_2}/bn/gamma"
    input: "BatchNormalization_10{RepeatStart_2}/bn/mean"
    input: "BatchNormalization_10{RepeatStart_2}/bn/var"
    output: "BatchNormalization_10"
    batch_normalization_param {
      axes: 1
      decay_rate: 0.5
      eps: 0.01
      batch_stat: True
    }
  }
  function {
    name: "Add2_3"
    type: "Add2"
    repeat_id: "RepeatStart_2"
    input: "BatchNormalization_10"
    input: "RepeatStart_2"
    output: "Add2_3"
    add2_param {
      inplace: False
    }
  }
  function {
    name: "ReLU_10"
    type: "ReLU"
    repeat_id: "RepeatStart_2"
    input: "Add2_3"
    output: "ReLU_10"
    relu_param {
      inplace: True
    }
  }
  function {
    name: "RepeatEnd_2"
    type: "RepeatEnd"
    input: "ReLU_10"
    output: "RepeatEnd_2"
    repeat_param {
      repeat_id: "RepeatStart_2"
      times: 2
    }
  }
  function {
    name: "MaxPooling_3"
    type: "MaxPooling"
    input: "RepeatEnd_2"
    output: "MaxPooling_3"
    max_pooling_param {
      kernel: { dim: 2 dim: 2 }
      stride: { dim: 2 dim: 2 }
      ignore_border: true
      pad: { dim: 0 dim: 0 }
    }
  }
  function {
    name: "Convolution_11"
    type: "Convolution"
    input: "MaxPooling_3"
    input: "Convolution_11/conv/W"
    output: "Convolution_11"
    convolution_param {
      pad: { dim: 0 dim: 0 }
      stride: { dim: 1 dim: 1 }
      dilation: { dim: 1 dim: 1 }
      group: 1
      base_axis: 1
    }
  }
  function {
    name: "BatchNormalization_11"
    type: "BatchNormalization"
    input: "Convolution_11"
    input: "BatchNormalization_11/bn/beta"
    input: "BatchNormalization_11/bn/gamma"
    input: "BatchNormalization_11/bn/mean"
    input: "BatchNormalization_11/bn/var"
    output: "BatchNormalization_11"
    batch_normalization_param {
      axes: 1
      decay_rate: 0.5
      eps: 0.01
      batch_stat: True
    }
  }
  function {
    name: "ReLU_11"
    type: "ReLU"
    input: "BatchNormalization_11"
    output: "ReLU_11"
    relu_param {
      inplace: True
    }
  }
  function {
    name: "Convolution_12"
    type: "Convolution"
    input: "ReLU_11"
    input: "Convolution_12/conv/W"
    output: "Convolution_12"
    convolution_param {
      pad: { dim: 1 dim: 1 }
      stride: { dim: 1 dim: 1 }
      dilation: { dim: 1 dim: 1 }
      group: 1
      base_axis: 1
    }
  }
  function {
    name: "BatchNormalization_12"
    type: "BatchNormalization"
    input: "Convolution_12"
    input: "BatchNormalization_12/bn/beta"
    input: "BatchNormalization_12/bn/gamma"
    input: "BatchNormalization_12/bn/mean"
    input: "BatchNormalization_12/bn/var"
    output: "BatchNormalization_12"
    batch_normalization_param {
      axes: 1
      decay_rate: 0.5
      eps: 0.01
      batch_stat: True
    }
  }
  function {
    name: "ReLU_12"
    type: "ReLU"
    input: "BatchNormalization_12"
    output: "ReLU_12"
    relu_param {
      inplace: True
    }
  }
  function {
    name: "Convolution_13"
    type: "Convolution"
    input: "ReLU_12"
    input: "Convolution_13/conv/W"
    output: "Convolution_13"
    convolution_param {
      pad: { dim: 0 dim: 0 }
      stride: { dim: 1 dim: 1 }
      dilation: { dim: 1 dim: 1 }
      group: 1
      base_axis: 1
    }
  }
  function {
    name: "BatchNormalization_13"
    type: "BatchNormalization"
    input: "Convolution_13"
    input: "BatchNormalization_13/bn/beta"
    input: "BatchNormalization_13/bn/gamma"
    input: "BatchNormalization_13/bn/mean"
    input: "BatchNormalization_13/bn/var"
    output: "BatchNormalization_13"
    batch_normalization_param {
      axes: 1
      decay_rate: 0.5
      eps: 0.01
      batch_stat: True
    }
  }
  function {
    name: "Add2_4"
    type: "Add2"
    input: "BatchNormalization_13"
    input: "MaxPooling_3"
    output: "Add2_4"
    add2_param {
      inplace: False
    }
  }
  function {
    name: "ReLU_13"
    type: "ReLU"
    input: "Add2_4"
    output: "ReLU_13"
    relu_param {
      inplace: True
    }
  }
  function {
    name: "AveragePooling"
    type: "AveragePooling"
    input: "ReLU_13"
    output: "AveragePooling"
    average_pooling_param {
      kernel: { dim: 4 dim: 4 }
      stride: { dim: 4 dim: 4 }
      ignore_border: true
      pad: { dim: 0 dim: 0 }
      including_pad: true
    }
  }
  function {
    name: "Affine"
    type: "Affine"
    input: "AveragePooling"
    input: "Affine/affine/W"
    input: "Affine/affine/b"
    output: "Affine"
    affine_param {
      base_axis: 1
    }
  }
  function {
    name: "Softmax"
    type: "Softmax"
    input: "Affine"
    output: "Softmax"
    softmax_param {
      axis: 1
    }
  }
  function {
    name: "CategoricalCrossEntropy"
    type: "CategoricalCrossEntropy"
    input: "Softmax"
    input: "CategoricalCrossEntropy_T"
    output: "CategoricalCrossEntropy"
    categorical_cross_entropy_param {
      axis: 1
    }
  }
}
network {
  name: "MainValidation"
  batch_size: 64
  repeat_info {
    id: "RepeatStart"
    times: 2
  }
  repeat_info {
    id: "RepeatStart_2"
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
    shape: { dim: 64 dim: 1 dim: 5 dim: 5 }
    initializer {
      type: "NormalConvolutionGlorot"
      multiplier: 1
    }
  }
  variable {
    name: "BatchNormalization/bn/beta"
    type: "Parameter"
    shape: { dim: 1 dim: 64 dim: 1 dim: 1 }
    initializer {
      type: "Constant"
      multiplier: 0
    }
  }
  variable {
    name: "BatchNormalization/bn/gamma"
    type: "Parameter"
    shape: { dim: 1 dim: 64 dim: 1 dim: 1 }
    initializer {
      type: "Constant"
      multiplier: 1
    }
  }
  variable {
    name: "BatchNormalization/bn/mean"
    type: "Parameter"
    shape: { dim: 1 dim: 64 dim: 1 dim: 1 }
    initializer {
      type: "Constant"
      multiplier: 0
    }
  }
  variable {
    name: "BatchNormalization/bn/var"
    type: "Parameter"
    shape: { dim: 1 dim: 64 dim: 1 dim: 1 }
    initializer {
      type: "Constant"
      multiplier: 0
    }
  }
  variable {
    name: "Convolution_2/conv/W"
    type: "Parameter"
    shape: { dim: 64 dim: 64 dim: 1 dim: 1 }
    initializer {
      type: "NormalConvolutionGlorot"
      multiplier: 1
    }
  }
  variable {
    name: "BatchNormalization_2/bn/beta"
    type: "Parameter"
    shape: { dim: 1 dim: 64 dim: 1 dim: 1 }
    initializer {
      type: "Constant"
      multiplier: 0
    }
  }
  variable {
    name: "BatchNormalization_2/bn/gamma"
    type: "Parameter"
    shape: { dim: 1 dim: 64 dim: 1 dim: 1 }
    initializer {
      type: "Constant"
      multiplier: 1
    }
  }
  variable {
    name: "BatchNormalization_2/bn/mean"
    type: "Parameter"
    shape: { dim: 1 dim: 64 dim: 1 dim: 1 }
    initializer {
      type: "Constant"
      multiplier: 0
    }
  }
  variable {
    name: "BatchNormalization_2/bn/var"
    type: "Parameter"
    shape: { dim: 1 dim: 64 dim: 1 dim: 1 }
    initializer {
      type: "Constant"
      multiplier: 0
    }
  }
  variable {
    name: "Convolution_3/conv/W"
    type: "Parameter"
    shape: { dim: 64 dim: 64 dim: 3 dim: 3 }
    initializer {
      type: "NormalConvolutionGlorot"
      multiplier: 1
    }
  }
  variable {
    name: "BatchNormalization_3/bn/beta"
    type: "Parameter"
    shape: { dim: 1 dim: 64 dim: 1 dim: 1 }
    initializer {
      type: "Constant"
      multiplier: 0
    }
  }
  variable {
    name: "BatchNormalization_3/bn/gamma"
    type: "Parameter"
    shape: { dim: 1 dim: 64 dim: 1 dim: 1 }
    initializer {
      type: "Constant"
      multiplier: 1
    }
  }
  variable {
    name: "BatchNormalization_3/bn/mean"
    type: "Parameter"
    shape: { dim: 1 dim: 64 dim: 1 dim: 1 }
    initializer {
      type: "Constant"
      multiplier: 0
    }
  }
  variable {
    name: "BatchNormalization_3/bn/var"
    type: "Parameter"
    shape: { dim: 1 dim: 64 dim: 1 dim: 1 }
    initializer {
      type: "Constant"
      multiplier: 0
    }
  }
  variable {
    name: "Convolution_4/conv/W"
    type: "Parameter"
    shape: { dim: 64 dim: 64 dim: 1 dim: 1 }
    initializer {
      type: "NormalConvolutionGlorot"
      multiplier: 1
    }
  }
  variable {
    name: "BatchNormalization_4/bn/beta"
    type: "Parameter"
    shape: { dim: 1 dim: 64 dim: 1 dim: 1 }
    initializer {
      type: "Constant"
      multiplier: 0
    }
  }
  variable {
    name: "BatchNormalization_4/bn/gamma"
    type: "Parameter"
    shape: { dim: 1 dim: 64 dim: 1 dim: 1 }
    initializer {
      type: "Constant"
      multiplier: 1
    }
  }
  variable {
    name: "BatchNormalization_4/bn/mean"
    type: "Parameter"
    shape: { dim: 1 dim: 64 dim: 1 dim: 1 }
    initializer {
      type: "Constant"
      multiplier: 0
    }
  }
  variable {
    name: "BatchNormalization_4/bn/var"
    type: "Parameter"
    shape: { dim: 1 dim: 64 dim: 1 dim: 1 }
    initializer {
      type: "Constant"
      multiplier: 0
    }
  }
  variable {
    name: "Convolution_5{RepeatStart}/conv/W"
    type: "Parameter"
    repeat_id: "RepeatStart"
    shape: { dim: 64 dim: 64 dim: 1 dim: 1 }
    initializer {
      type: "NormalConvolutionGlorot"
      multiplier: 1
    }
  }
  variable {
    name: "BatchNormalization_5{RepeatStart}/bn/beta"
    type: "Parameter"
    repeat_id: "RepeatStart"
    shape: { dim: 1 dim: 64 dim: 1 dim: 1 }
    initializer {
      type: "Constant"
      multiplier: 0
    }
  }
  variable {
    name: "BatchNormalization_5{RepeatStart}/bn/gamma"
    type: "Parameter"
    repeat_id: "RepeatStart"
    shape: { dim: 1 dim: 64 dim: 1 dim: 1 }
    initializer {
      type: "Constant"
      multiplier: 1
    }
  }
  variable {
    name: "BatchNormalization_5{RepeatStart}/bn/mean"
    type: "Parameter"
    repeat_id: "RepeatStart"
    shape: { dim: 1 dim: 64 dim: 1 dim: 1 }
    initializer {
      type: "Constant"
      multiplier: 0
    }
  }
  variable {
    name: "BatchNormalization_5{RepeatStart}/bn/var"
    type: "Parameter"
    repeat_id: "RepeatStart"
    shape: { dim: 1 dim: 64 dim: 1 dim: 1 }
    initializer {
      type: "Constant"
      multiplier: 0
    }
  }
  variable {
    name: "Convolution_6{RepeatStart}/conv/W"
    type: "Parameter"
    repeat_id: "RepeatStart"
    shape: { dim: 64 dim: 64 dim: 3 dim: 3 }
    initializer {
      type: "NormalConvolutionGlorot"
      multiplier: 1
    }
  }
  variable {
    name: "BatchNormalization_6{RepeatStart}/bn/beta"
    type: "Parameter"
    repeat_id: "RepeatStart"
    shape: { dim: 1 dim: 64 dim: 1 dim: 1 }
    initializer {
      type: "Constant"
      multiplier: 0
    }
  }
  variable {
    name: "BatchNormalization_6{RepeatStart}/bn/gamma"
    type: "Parameter"
    repeat_id: "RepeatStart"
    shape: { dim: 1 dim: 64 dim: 1 dim: 1 }
    initializer {
      type: "Constant"
      multiplier: 1
    }
  }
  variable {
    name: "BatchNormalization_6{RepeatStart}/bn/mean"
    type: "Parameter"
    repeat_id: "RepeatStart"
    shape: { dim: 1 dim: 64 dim: 1 dim: 1 }
    initializer {
      type: "Constant"
      multiplier: 0
    }
  }
  variable {
    name: "BatchNormalization_6{RepeatStart}/bn/var"
    type: "Parameter"
    repeat_id: "RepeatStart"
    shape: { dim: 1 dim: 64 dim: 1 dim: 1 }
    initializer {
      type: "Constant"
      multiplier: 0
    }
  }
  variable {
    name: "Convolution_7{RepeatStart}/conv/W"
    type: "Parameter"
    repeat_id: "RepeatStart"
    shape: { dim: 64 dim: 64 dim: 1 dim: 1 }
    initializer {
      type: "NormalConvolutionGlorot"
      multiplier: 1
    }
  }
  variable {
    name: "BatchNormalization_7{RepeatStart}/bn/beta"
    type: "Parameter"
    repeat_id: "RepeatStart"
    shape: { dim: 1 dim: 64 dim: 1 dim: 1 }
    initializer {
      type: "Constant"
      multiplier: 0
    }
  }
  variable {
    name: "BatchNormalization_7{RepeatStart}/bn/gamma"
    type: "Parameter"
    repeat_id: "RepeatStart"
    shape: { dim: 1 dim: 64 dim: 1 dim: 1 }
    initializer {
      type: "Constant"
      multiplier: 1
    }
  }
  variable {
    name: "BatchNormalization_7{RepeatStart}/bn/mean"
    type: "Parameter"
    repeat_id: "RepeatStart"
    shape: { dim: 1 dim: 64 dim: 1 dim: 1 }
    initializer {
      type: "Constant"
      multiplier: 0
    }
  }
  variable {
    name: "BatchNormalization_7{RepeatStart}/bn/var"
    type: "Parameter"
    repeat_id: "RepeatStart"
    shape: { dim: 1 dim: 64 dim: 1 dim: 1 }
    initializer {
      type: "Constant"
      multiplier: 0
    }
  }
  variable {
    name: "Convolution_8{RepeatStart_2}/conv/W"
    type: "Parameter"
    repeat_id: "RepeatStart_2"
    shape: { dim: 64 dim: 64 dim: 1 dim: 1 }
    initializer {
      type: "NormalConvolutionGlorot"
      multiplier: 1
    }
  }
  variable {
    name: "BatchNormalization_8{RepeatStart_2}/bn/beta"
    type: "Parameter"
    repeat_id: "RepeatStart_2"
    shape: { dim: 1 dim: 64 dim: 1 dim: 1 }
    initializer {
      type: "Constant"
      multiplier: 0
    }
  }
  variable {
    name: "BatchNormalization_8{RepeatStart_2}/bn/gamma"
    type: "Parameter"
    repeat_id: "RepeatStart_2"
    shape: { dim: 1 dim: 64 dim: 1 dim: 1 }
    initializer {
      type: "Constant"
      multiplier: 1
    }
  }
  variable {
    name: "BatchNormalization_8{RepeatStart_2}/bn/mean"
    type: "Parameter"
    repeat_id: "RepeatStart_2"
    shape: { dim: 1 dim: 64 dim: 1 dim: 1 }
    initializer {
      type: "Constant"
      multiplier: 0
    }
  }
  variable {
    name: "BatchNormalization_8{RepeatStart_2}/bn/var"
    type: "Parameter"
    repeat_id: "RepeatStart_2"
    shape: { dim: 1 dim: 64 dim: 1 dim: 1 }
    initializer {
      type: "Constant"
      multiplier: 0
    }
  }
  variable {
    name: "Convolution_9{RepeatStart_2}/conv/W"
    type: "Parameter"
    repeat_id: "RepeatStart_2"
    shape: { dim: 64 dim: 64 dim: 3 dim: 3 }
    initializer {
      type: "NormalConvolutionGlorot"
      multiplier: 1
    }
  }
  variable {
    name: "BatchNormalization_9{RepeatStart_2}/bn/beta"
    type: "Parameter"
    repeat_id: "RepeatStart_2"
    shape: { dim: 1 dim: 64 dim: 1 dim: 1 }
    initializer {
      type: "Constant"
      multiplier: 0
    }
  }
  variable {
    name: "BatchNormalization_9{RepeatStart_2}/bn/gamma"
    type: "Parameter"
    repeat_id: "RepeatStart_2"
    shape: { dim: 1 dim: 64 dim: 1 dim: 1 }
    initializer {
      type: "Constant"
      multiplier: 1
    }
  }
  variable {
    name: "BatchNormalization_9{RepeatStart_2}/bn/mean"
    type: "Parameter"
    repeat_id: "RepeatStart_2"
    shape: { dim: 1 dim: 64 dim: 1 dim: 1 }
    initializer {
      type: "Constant"
      multiplier: 0
    }
  }
  variable {
    name: "BatchNormalization_9{RepeatStart_2}/bn/var"
    type: "Parameter"
    repeat_id: "RepeatStart_2"
    shape: { dim: 1 dim: 64 dim: 1 dim: 1 }
    initializer {
      type: "Constant"
      multiplier: 0
    }
  }
  variable {
    name: "Convolution_10{RepeatStart_2}/conv/W"
    type: "Parameter"
    repeat_id: "RepeatStart_2"
    shape: { dim: 64 dim: 64 dim: 1 dim: 1 }
    initializer {
      type: "NormalConvolutionGlorot"
      multiplier: 1
    }
  }
  variable {
    name: "BatchNormalization_10{RepeatStart_2}/bn/beta"
    type: "Parameter"
    repeat_id: "RepeatStart_2"
    shape: { dim: 1 dim: 64 dim: 1 dim: 1 }
    initializer {
      type: "Constant"
      multiplier: 0
    }
  }
  variable {
    name: "BatchNormalization_10{RepeatStart_2}/bn/gamma"
    type: "Parameter"
    repeat_id: "RepeatStart_2"
    shape: { dim: 1 dim: 64 dim: 1 dim: 1 }
    initializer {
      type: "Constant"
      multiplier: 1
    }
  }
  variable {
    name: "BatchNormalization_10{RepeatStart_2}/bn/mean"
    type: "Parameter"
    repeat_id: "RepeatStart_2"
    shape: { dim: 1 dim: 64 dim: 1 dim: 1 }
    initializer {
      type: "Constant"
      multiplier: 0
    }
  }
  variable {
    name: "BatchNormalization_10{RepeatStart_2}/bn/var"
    type: "Parameter"
    repeat_id: "RepeatStart_2"
    shape: { dim: 1 dim: 64 dim: 1 dim: 1 }
    initializer {
      type: "Constant"
      multiplier: 0
    }
  }
  variable {
    name: "Convolution_11/conv/W"
    type: "Parameter"
    shape: { dim: 64 dim: 64 dim: 1 dim: 1 }
    initializer {
      type: "NormalConvolutionGlorot"
      multiplier: 1
    }
  }
  variable {
    name: "BatchNormalization_11/bn/beta"
    type: "Parameter"
    shape: { dim: 1 dim: 64 dim: 1 dim: 1 }
    initializer {
      type: "Constant"
      multiplier: 0
    }
  }
  variable {
    name: "BatchNormalization_11/bn/gamma"
    type: "Parameter"
    shape: { dim: 1 dim: 64 dim: 1 dim: 1 }
    initializer {
      type: "Constant"
      multiplier: 1
    }
  }
  variable {
    name: "BatchNormalization_11/bn/mean"
    type: "Parameter"
    shape: { dim: 1 dim: 64 dim: 1 dim: 1 }
    initializer {
      type: "Constant"
      multiplier: 0
    }
  }
  variable {
    name: "BatchNormalization_11/bn/var"
    type: "Parameter"
    shape: { dim: 1 dim: 64 dim: 1 dim: 1 }
    initializer {
      type: "Constant"
      multiplier: 0
    }
  }
  variable {
    name: "Convolution_12/conv/W"
    type: "Parameter"
    shape: { dim: 64 dim: 64 dim: 3 dim: 3 }
    initializer {
      type: "NormalConvolutionGlorot"
      multiplier: 1
    }
  }
  variable {
    name: "BatchNormalization_12/bn/beta"
    type: "Parameter"
    shape: { dim: 1 dim: 64 dim: 1 dim: 1 }
    initializer {
      type: "Constant"
      multiplier: 0
    }
  }
  variable {
    name: "BatchNormalization_12/bn/gamma"
    type: "Parameter"
    shape: { dim: 1 dim: 64 dim: 1 dim: 1 }
    initializer {
      type: "Constant"
      multiplier: 1
    }
  }
  variable {
    name: "BatchNormalization_12/bn/mean"
    type: "Parameter"
    shape: { dim: 1 dim: 64 dim: 1 dim: 1 }
    initializer {
      type: "Constant"
      multiplier: 0
    }
  }
  variable {
    name: "BatchNormalization_12/bn/var"
    type: "Parameter"
    shape: { dim: 1 dim: 64 dim: 1 dim: 1 }
    initializer {
      type: "Constant"
      multiplier: 0
    }
  }
  variable {
    name: "Convolution_13/conv/W"
    type: "Parameter"
    shape: { dim: 64 dim: 64 dim: 1 dim: 1 }
    initializer {
      type: "NormalConvolutionGlorot"
      multiplier: 1
    }
  }
  variable {
    name: "BatchNormalization_13/bn/beta"
    type: "Parameter"
    shape: { dim: 1 dim: 64 dim: 1 dim: 1 }
    initializer {
      type: "Constant"
      multiplier: 0
    }
  }
  variable {
    name: "BatchNormalization_13/bn/gamma"
    type: "Parameter"
    shape: { dim: 1 dim: 64 dim: 1 dim: 1 }
    initializer {
      type: "Constant"
      multiplier: 1
    }
  }
  variable {
    name: "BatchNormalization_13/bn/mean"
    type: "Parameter"
    shape: { dim: 1 dim: 64 dim: 1 dim: 1 }
    initializer {
      type: "Constant"
      multiplier: 0
    }
  }
  variable {
    name: "BatchNormalization_13/bn/var"
    type: "Parameter"
    shape: { dim: 1 dim: 64 dim: 1 dim: 1 }
    initializer {
      type: "Constant"
      multiplier: 0
    }
  }
  variable {
    name: "Affine/affine/W"
    type: "Parameter"
    shape: { dim: 64 dim: 10 }
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
    name: "CategoricalCrossEntropy_T"
    type: "Buffer"
    shape: { dim:-1 dim: 1 }
  }
  variable {
    name: "Convolution"
    type: "Buffer"
    shape: { dim:-1 dim: 64 dim: 32 dim: 32 }
  }
  variable {
    name: "BatchNormalization"
    type: "Buffer"
    shape: { dim:-1 dim: 64 dim: 32 dim: 32 }
  }
  variable {
    name: "ReLU"
    type: "Buffer"
    shape: { dim:-1 dim: 64 dim: 32 dim: 32 }
  }
  variable {
    name: "Convolution_2"
    type: "Buffer"
    shape: { dim:-1 dim: 64 dim: 32 dim: 32 }
  }
  variable {
    name: "BatchNormalization_2"
    type: "Buffer"
    shape: { dim:-1 dim: 64 dim: 32 dim: 32 }
  }
  variable {
    name: "ReLU_2"
    type: "Buffer"
    shape: { dim:-1 dim: 64 dim: 32 dim: 32 }
  }
  variable {
    name: "Convolution_3"
    type: "Buffer"
    shape: { dim:-1 dim: 64 dim: 32 dim: 32 }
  }
  variable {
    name: "BatchNormalization_3"
    type: "Buffer"
    shape: { dim:-1 dim: 64 dim: 32 dim: 32 }
  }
  variable {
    name: "ReLU_3"
    type: "Buffer"
    shape: { dim:-1 dim: 64 dim: 32 dim: 32 }
  }
  variable {
    name: "Convolution_4"
    type: "Buffer"
    shape: { dim:-1 dim: 64 dim: 32 dim: 32 }
  }
  variable {
    name: "BatchNormalization_4"
    type: "Buffer"
    shape: { dim:-1 dim: 64 dim: 32 dim: 32 }
  }
  variable {
    name: "Add2"
    type: "Buffer"
    shape: { dim:-1 dim: 64 dim: 32 dim: 32 }
  }
  variable {
    name: "ReLU_4"
    type: "Buffer"
    shape: { dim:-1 dim: 64 dim: 32 dim: 32 }
  }
  variable {
    name: "MaxPooling"
    type: "Buffer"
    shape: { dim:-1 dim: 64 dim: 16 dim: 16 }
  }
  variable {
    name: "RepeatStart"
    type: "Buffer"
    repeat_id: "RepeatStart"
    shape: { dim:-1 dim: 64 dim: 16 dim: 16 }
  }
  variable {
    name: "Convolution_5"
    type: "Buffer"
    repeat_id: "RepeatStart"
    shape: { dim:-1 dim: 64 dim: 16 dim: 16 }
  }
  variable {
    name: "BatchNormalization_5"
    type: "Buffer"
    repeat_id: "RepeatStart"
    shape: { dim:-1 dim: 64 dim: 16 dim: 16 }
  }
  variable {
    name: "ReLU_5"
    type: "Buffer"
    repeat_id: "RepeatStart"
    shape: { dim:-1 dim: 64 dim: 16 dim: 16 }
  }
  variable {
    name: "Convolution_6"
    type: "Buffer"
    repeat_id: "RepeatStart"
    shape: { dim:-1 dim: 64 dim: 16 dim: 16 }
  }
  variable {
    name: "BatchNormalization_6"
    type: "Buffer"
    repeat_id: "RepeatStart"
    shape: { dim:-1 dim: 64 dim: 16 dim: 16 }
  }
  variable {
    name: "ReLU_6"
    type: "Buffer"
    repeat_id: "RepeatStart"
    shape: { dim:-1 dim: 64 dim: 16 dim: 16 }
  }
  variable {
    name: "Convolution_7"
    type: "Buffer"
    repeat_id: "RepeatStart"
    shape: { dim:-1 dim: 64 dim: 16 dim: 16 }
  }
  variable {
    name: "BatchNormalization_7"
    type: "Buffer"
    repeat_id: "RepeatStart"
    shape: { dim:-1 dim: 64 dim: 16 dim: 16 }
  }
  variable {
    name: "Add2_2"
    type: "Buffer"
    repeat_id: "RepeatStart"
    shape: { dim:-1 dim: 64 dim: 16 dim: 16 }
  }
  variable {
    name: "ReLU_7"
    type: "Buffer"
    repeat_id: "RepeatStart"
    shape: { dim:-1 dim: 64 dim: 16 dim: 16 }
  }
  variable {
    name: "RepeatEnd"
    type: "Buffer"
    shape: { dim:-1 dim: 64 dim: 16 dim: 16 }
  }
  variable {
    name: "MaxPooling_2"
    type: "Buffer"
    shape: { dim:-1 dim: 64 dim: 8 dim: 8 }
  }
  variable {
    name: "RepeatStart_2"
    type: "Buffer"
    repeat_id: "RepeatStart_2"
    shape: { dim:-1 dim: 64 dim: 8 dim: 8 }
  }
  variable {
    name: "Convolution_8"
    type: "Buffer"
    repeat_id: "RepeatStart_2"
    shape: { dim:-1 dim: 64 dim: 8 dim: 8 }
  }
  variable {
    name: "BatchNormalization_8"
    type: "Buffer"
    repeat_id: "RepeatStart_2"
    shape: { dim:-1 dim: 64 dim: 8 dim: 8 }
  }
  variable {
    name: "ReLU_8"
    type: "Buffer"
    repeat_id: "RepeatStart_2"
    shape: { dim:-1 dim: 64 dim: 8 dim: 8 }
  }
  variable {
    name: "Convolution_9"
    type: "Buffer"
    repeat_id: "RepeatStart_2"
    shape: { dim:-1 dim: 64 dim: 8 dim: 8 }
  }
  variable {
    name: "BatchNormalization_9"
    type: "Buffer"
    repeat_id: "RepeatStart_2"
    shape: { dim:-1 dim: 64 dim: 8 dim: 8 }
  }
  variable {
    name: "ReLU_9"
    type: "Buffer"
    repeat_id: "RepeatStart_2"
    shape: { dim:-1 dim: 64 dim: 8 dim: 8 }
  }
  variable {
    name: "Convolution_10"
    type: "Buffer"
    repeat_id: "RepeatStart_2"
    shape: { dim:-1 dim: 64 dim: 8 dim: 8 }
  }
  variable {
    name: "BatchNormalization_10"
    type: "Buffer"
    repeat_id: "RepeatStart_2"
    shape: { dim:-1 dim: 64 dim: 8 dim: 8 }
  }
  variable {
    name: "Add2_3"
    type: "Buffer"
    repeat_id: "RepeatStart_2"
    shape: { dim:-1 dim: 64 dim: 8 dim: 8 }
  }
  variable {
    name: "ReLU_10"
    type: "Buffer"
    repeat_id: "RepeatStart_2"
    shape: { dim:-1 dim: 64 dim: 8 dim: 8 }
  }
  variable {
    name: "RepeatEnd_2"
    type: "Buffer"
    shape: { dim:-1 dim: 64 dim: 8 dim: 8 }
  }
  variable {
    name: "MaxPooling_3"
    type: "Buffer"
    shape: { dim:-1 dim: 64 dim: 4 dim: 4 }
  }
  variable {
    name: "Convolution_11"
    type: "Buffer"
    shape: { dim:-1 dim: 64 dim: 4 dim: 4 }
  }
  variable {
    name: "BatchNormalization_11"
    type: "Buffer"
    shape: { dim:-1 dim: 64 dim: 4 dim: 4 }
  }
  variable {
    name: "ReLU_11"
    type: "Buffer"
    shape: { dim:-1 dim: 64 dim: 4 dim: 4 }
  }
  variable {
    name: "Convolution_12"
    type: "Buffer"
    shape: { dim:-1 dim: 64 dim: 4 dim: 4 }
  }
  variable {
    name: "BatchNormalization_12"
    type: "Buffer"
    shape: { dim:-1 dim: 64 dim: 4 dim: 4 }
  }
  variable {
    name: "ReLU_12"
    type: "Buffer"
    shape: { dim:-1 dim: 64 dim: 4 dim: 4 }
  }
  variable {
    name: "Convolution_13"
    type: "Buffer"
    shape: { dim:-1 dim: 64 dim: 4 dim: 4 }
  }
  variable {
    name: "BatchNormalization_13"
    type: "Buffer"
    shape: { dim:-1 dim: 64 dim: 4 dim: 4 }
  }
  variable {
    name: "Add2_4"
    type: "Buffer"
    shape: { dim:-1 dim: 64 dim: 4 dim: 4 }
  }
  variable {
    name: "ReLU_13"
    type: "Buffer"
    shape: { dim:-1 dim: 64 dim: 4 dim: 4 }
  }
  variable {
    name: "AveragePooling"
    type: "Buffer"
    shape: { dim:-1 dim: 64 dim: 1 dim: 1 }
  }
  variable {
    name: "Affine"
    type: "Buffer"
    shape: { dim:-1 dim: 10 }
  }
  variable {
    name: "Softmax"
    type: "Buffer"
    shape: { dim:-1 dim: 10 }
  }
  variable {
    name: "CategoricalCrossEntropy"
    type: "Buffer"
    shape: { dim:-1 dim: 1 }
  }
  function {
    name: "Convolution"
    type: "Convolution"
    input: "Input"
    input: "Convolution/conv/W"
    output: "Convolution"
    convolution_param {
      pad: { dim: 4 dim: 4 }
      stride: { dim: 1 dim: 1 }
      dilation: { dim: 1 dim: 1 }
      group: 1
      base_axis: 1
    }
  }
  function {
    name: "BatchNormalization"
    type: "BatchNormalization"
    input: "Convolution"
    input: "BatchNormalization/bn/beta"
    input: "BatchNormalization/bn/gamma"
    input: "BatchNormalization/bn/mean"
    input: "BatchNormalization/bn/var"
    output: "BatchNormalization"
    batch_normalization_param {
      axes: 1
      decay_rate: 0.5
      eps: 0.01
      batch_stat: False
    }
  }
  function {
    name: "ReLU"
    type: "ReLU"
    input: "BatchNormalization"
    output: "ReLU"
    relu_param {
      inplace: True
    }
  }
  function {
    name: "Convolution_2"
    type: "Convolution"
    input: "ReLU"
    input: "Convolution_2/conv/W"
    output: "Convolution_2"
    convolution_param {
      pad: { dim: 0 dim: 0 }
      stride: { dim: 1 dim: 1 }
      dilation: { dim: 1 dim: 1 }
      group: 1
      base_axis: 1
    }
  }
  function {
    name: "BatchNormalization_2"
    type: "BatchNormalization"
    input: "Convolution_2"
    input: "BatchNormalization_2/bn/beta"
    input: "BatchNormalization_2/bn/gamma"
    input: "BatchNormalization_2/bn/mean"
    input: "BatchNormalization_2/bn/var"
    output: "BatchNormalization_2"
    batch_normalization_param {
      axes: 1
      decay_rate: 0.5
      eps: 0.01
      batch_stat: False
    }
  }
  function {
    name: "ReLU_2"
    type: "ReLU"
    input: "BatchNormalization_2"
    output: "ReLU_2"
    relu_param {
      inplace: True
    }
  }
  function {
    name: "Convolution_3"
    type: "Convolution"
    input: "ReLU_2"
    input: "Convolution_3/conv/W"
    output: "Convolution_3"
    convolution_param {
      pad: { dim: 1 dim: 1 }
      stride: { dim: 1 dim: 1 }
      dilation: { dim: 1 dim: 1 }
      group: 1
      base_axis: 1
    }
  }
  function {
    name: "BatchNormalization_3"
    type: "BatchNormalization"
    input: "Convolution_3"
    input: "BatchNormalization_3/bn/beta"
    input: "BatchNormalization_3/bn/gamma"
    input: "BatchNormalization_3/bn/mean"
    input: "BatchNormalization_3/bn/var"
    output: "BatchNormalization_3"
    batch_normalization_param {
      axes: 1
      decay_rate: 0.5
      eps: 0.01
      batch_stat: False
    }
  }
  function {
    name: "ReLU_3"
    type: "ReLU"
    input: "BatchNormalization_3"
    output: "ReLU_3"
    relu_param {
      inplace: True
    }
  }
  function {
    name: "Convolution_4"
    type: "Convolution"
    input: "ReLU_3"
    input: "Convolution_4/conv/W"
    output: "Convolution_4"
    convolution_param {
      pad: { dim: 0 dim: 0 }
      stride: { dim: 1 dim: 1 }
      dilation: { dim: 1 dim: 1 }
      group: 1
      base_axis: 1
    }
  }
  function {
    name: "BatchNormalization_4"
    type: "BatchNormalization"
    input: "Convolution_4"
    input: "BatchNormalization_4/bn/beta"
    input: "BatchNormalization_4/bn/gamma"
    input: "BatchNormalization_4/bn/mean"
    input: "BatchNormalization_4/bn/var"
    output: "BatchNormalization_4"
    batch_normalization_param {
      axes: 1
      decay_rate: 0.5
      eps: 0.01
      batch_stat: False
    }
  }
  function {
    name: "Add2"
    type: "Add2"
    input: "BatchNormalization_4"
    input: "ReLU"
    output: "Add2"
    add2_param {
      inplace: False
    }
  }
  function {
    name: "ReLU_4"
    type: "ReLU"
    input: "Add2"
    output: "ReLU_4"
    relu_param {
      inplace: True
    }
  }
  function {
    name: "MaxPooling"
    type: "MaxPooling"
    input: "ReLU_4"
    output: "MaxPooling"
    max_pooling_param {
      kernel: { dim: 2 dim: 2 }
      stride: { dim: 2 dim: 2 }
      ignore_border: true
      pad: { dim: 0 dim: 0 }
    }
  }
  function {
    name: "RepeatStart"
    type: "RepeatStart"
    repeat_id: "RepeatStart"
    input: "MaxPooling"
    input: "ReLU_7"
    output: "RepeatStart"
    repeat_param {
      repeat_id: "RepeatStart"
    }
  }
  function {
    name: "Convolution_5"
    type: "Convolution"
    repeat_id: "RepeatStart"
    input: "RepeatStart"
    input: "Convolution_5{RepeatStart}/conv/W"
    output: "Convolution_5"
    convolution_param {
      pad: { dim: 0 dim: 0 }
      stride: { dim: 1 dim: 1 }
      dilation: { dim: 1 dim: 1 }
      group: 1
      base_axis: 1
    }
  }
  function {
    name: "BatchNormalization_5"
    type: "BatchNormalization"
    repeat_id: "RepeatStart"
    input: "Convolution_5"
    input: "BatchNormalization_5{RepeatStart}/bn/beta"
    input: "BatchNormalization_5{RepeatStart}/bn/gamma"
    input: "BatchNormalization_5{RepeatStart}/bn/mean"
    input: "BatchNormalization_5{RepeatStart}/bn/var"
    output: "BatchNormalization_5"
    batch_normalization_param {
      axes: 1
      decay_rate: 0.5
      eps: 0.01
      batch_stat: False
    }
  }
  function {
    name: "ReLU_5"
    type: "ReLU"
    repeat_id: "RepeatStart"
    input: "BatchNormalization_5"
    output: "ReLU_5"
    relu_param {
      inplace: True
    }
  }
  function {
    name: "Convolution_6"
    type: "Convolution"
    repeat_id: "RepeatStart"
    input: "ReLU_5"
    input: "Convolution_6{RepeatStart}/conv/W"
    output: "Convolution_6"
    convolution_param {
      pad: { dim: 1 dim: 1 }
      stride: { dim: 1 dim: 1 }
      dilation: { dim: 1 dim: 1 }
      group: 1
      base_axis: 1
    }
  }
  function {
    name: "BatchNormalization_6"
    type: "BatchNormalization"
    repeat_id: "RepeatStart"
    input: "Convolution_6"
    input: "BatchNormalization_6{RepeatStart}/bn/beta"
    input: "BatchNormalization_6{RepeatStart}/bn/gamma"
    input: "BatchNormalization_6{RepeatStart}/bn/mean"
    input: "BatchNormalization_6{RepeatStart}/bn/var"
    output: "BatchNormalization_6"
    batch_normalization_param {
      axes: 1
      decay_rate: 0.5
      eps: 0.01
      batch_stat: False
    }
  }
  function {
    name: "ReLU_6"
    type: "ReLU"
    repeat_id: "RepeatStart"
    input: "BatchNormalization_6"
    output: "ReLU_6"
    relu_param {
      inplace: True
    }
  }
  function {
    name: "Convolution_7"
    type: "Convolution"
    repeat_id: "RepeatStart"
    input: "ReLU_6"
    input: "Convolution_7{RepeatStart}/conv/W"
    output: "Convolution_7"
    convolution_param {
      pad: { dim: 0 dim: 0 }
      stride: { dim: 1 dim: 1 }
      dilation: { dim: 1 dim: 1 }
      group: 1
      base_axis: 1
    }
  }
  function {
    name: "BatchNormalization_7"
    type: "BatchNormalization"
    repeat_id: "RepeatStart"
    input: "Convolution_7"
    input: "BatchNormalization_7{RepeatStart}/bn/beta"
    input: "BatchNormalization_7{RepeatStart}/bn/gamma"
    input: "BatchNormalization_7{RepeatStart}/bn/mean"
    input: "BatchNormalization_7{RepeatStart}/bn/var"
    output: "BatchNormalization_7"
    batch_normalization_param {
      axes: 1
      decay_rate: 0.5
      eps: 0.01
      batch_stat: False
    }
  }
  function {
    name: "Add2_2"
    type: "Add2"
    repeat_id: "RepeatStart"
    input: "BatchNormalization_7"
    input: "RepeatStart"
    output: "Add2_2"
    add2_param {
      inplace: False
    }
  }
  function {
    name: "ReLU_7"
    type: "ReLU"
    repeat_id: "RepeatStart"
    input: "Add2_2"
    output: "ReLU_7"
    relu_param {
      inplace: True
    }
  }
  function {
    name: "RepeatEnd"
    type: "RepeatEnd"
    input: "ReLU_7"
    output: "RepeatEnd"
    repeat_param {
      repeat_id: "RepeatStart"
      times: 2
    }
  }
  function {
    name: "MaxPooling_2"
    type: "MaxPooling"
    input: "RepeatEnd"
    output: "MaxPooling_2"
    max_pooling_param {
      kernel: { dim: 2 dim: 2 }
      stride: { dim: 2 dim: 2 }
      ignore_border: true
      pad: { dim: 0 dim: 0 }
    }
  }
  function {
    name: "RepeatStart_2"
    type: "RepeatStart"
    repeat_id: "RepeatStart_2"
    input: "MaxPooling_2"
    input: "ReLU_10"
    output: "RepeatStart_2"
    repeat_param {
      repeat_id: "RepeatStart_2"
    }
  }
  function {
    name: "Convolution_8"
    type: "Convolution"
    repeat_id: "RepeatStart_2"
    input: "RepeatStart_2"
    input: "Convolution_8{RepeatStart_2}/conv/W"
    output: "Convolution_8"
    convolution_param {
      pad: { dim: 0 dim: 0 }
      stride: { dim: 1 dim: 1 }
      dilation: { dim: 1 dim: 1 }
      group: 1
      base_axis: 1
    }
  }
  function {
    name: "BatchNormalization_8"
    type: "BatchNormalization"
    repeat_id: "RepeatStart_2"
    input: "Convolution_8"
    input: "BatchNormalization_8{RepeatStart_2}/bn/beta"
    input: "BatchNormalization_8{RepeatStart_2}/bn/gamma"
    input: "BatchNormalization_8{RepeatStart_2}/bn/mean"
    input: "BatchNormalization_8{RepeatStart_2}/bn/var"
    output: "BatchNormalization_8"
    batch_normalization_param {
      axes: 1
      decay_rate: 0.5
      eps: 0.01
      batch_stat: False
    }
  }
  function {
    name: "ReLU_8"
    type: "ReLU"
    repeat_id: "RepeatStart_2"
    input: "BatchNormalization_8"
    output: "ReLU_8"
    relu_param {
      inplace: True
    }
  }
  function {
    name: "Convolution_9"
    type: "Convolution"
    repeat_id: "RepeatStart_2"
    input: "ReLU_8"
    input: "Convolution_9{RepeatStart_2}/conv/W"
    output: "Convolution_9"
    convolution_param {
      pad: { dim: 1 dim: 1 }
      stride: { dim: 1 dim: 1 }
      dilation: { dim: 1 dim: 1 }
      group: 1
      base_axis: 1
    }
  }
  function {
    name: "BatchNormalization_9"
    type: "BatchNormalization"
    repeat_id: "RepeatStart_2"
    input: "Convolution_9"
    input: "BatchNormalization_9{RepeatStart_2}/bn/beta"
    input: "BatchNormalization_9{RepeatStart_2}/bn/gamma"
    input: "BatchNormalization_9{RepeatStart_2}/bn/mean"
    input: "BatchNormalization_9{RepeatStart_2}/bn/var"
    output: "BatchNormalization_9"
    batch_normalization_param {
      axes: 1
      decay_rate: 0.5
      eps: 0.01
      batch_stat: False
    }
  }
  function {
    name: "ReLU_9"
    type: "ReLU"
    repeat_id: "RepeatStart_2"
    input: "BatchNormalization_9"
    output: "ReLU_9"
    relu_param {
      inplace: True
    }
  }
  function {
    name: "Convolution_10"
    type: "Convolution"
    repeat_id: "RepeatStart_2"
    input: "ReLU_9"
    input: "Convolution_10{RepeatStart_2}/conv/W"
    output: "Convolution_10"
    convolution_param {
      pad: { dim: 0 dim: 0 }
      stride: { dim: 1 dim: 1 }
      dilation: { dim: 1 dim: 1 }
      group: 1
      base_axis: 1
    }
  }
  function {
    name: "BatchNormalization_10"
    type: "BatchNormalization"
    repeat_id: "RepeatStart_2"
    input: "Convolution_10"
    input: "BatchNormalization_10{RepeatStart_2}/bn/beta"
    input: "BatchNormalization_10{RepeatStart_2}/bn/gamma"
    input: "BatchNormalization_10{RepeatStart_2}/bn/mean"
    input: "BatchNormalization_10{RepeatStart_2}/bn/var"
    output: "BatchNormalization_10"
    batch_normalization_param {
      axes: 1
      decay_rate: 0.5
      eps: 0.01
      batch_stat: False
    }
  }
  function {
    name: "Add2_3"
    type: "Add2"
    repeat_id: "RepeatStart_2"
    input: "BatchNormalization_10"
    input: "RepeatStart_2"
    output: "Add2_3"
    add2_param {
      inplace: False
    }
  }
  function {
    name: "ReLU_10"
    type: "ReLU"
    repeat_id: "RepeatStart_2"
    input: "Add2_3"
    output: "ReLU_10"
    relu_param {
      inplace: True
    }
  }
  function {
    name: "RepeatEnd_2"
    type: "RepeatEnd"
    input: "ReLU_10"
    output: "RepeatEnd_2"
    repeat_param {
      repeat_id: "RepeatStart_2"
      times: 2
    }
  }
  function {
    name: "MaxPooling_3"
    type: "MaxPooling"
    input: "RepeatEnd_2"
    output: "MaxPooling_3"
    max_pooling_param {
      kernel: { dim: 2 dim: 2 }
      stride: { dim: 2 dim: 2 }
      ignore_border: true
      pad: { dim: 0 dim: 0 }
    }
  }
  function {
    name: "Convolution_11"
    type: "Convolution"
    input: "MaxPooling_3"
    input: "Convolution_11/conv/W"
    output: "Convolution_11"
    convolution_param {
      pad: { dim: 0 dim: 0 }
      stride: { dim: 1 dim: 1 }
      dilation: { dim: 1 dim: 1 }
      group: 1
      base_axis: 1
    }
  }
  function {
    name: "BatchNormalization_11"
    type: "BatchNormalization"
    input: "Convolution_11"
    input: "BatchNormalization_11/bn/beta"
    input: "BatchNormalization_11/bn/gamma"
    input: "BatchNormalization_11/bn/mean"
    input: "BatchNormalization_11/bn/var"
    output: "BatchNormalization_11"
    batch_normalization_param {
      axes: 1
      decay_rate: 0.5
      eps: 0.01
      batch_stat: False
    }
  }
  function {
    name: "ReLU_11"
    type: "ReLU"
    input: "BatchNormalization_11"
    output: "ReLU_11"
    relu_param {
      inplace: True
    }
  }
  function {
    name: "Convolution_12"
    type: "Convolution"
    input: "ReLU_11"
    input: "Convolution_12/conv/W"
    output: "Convolution_12"
    convolution_param {
      pad: { dim: 1 dim: 1 }
      stride: { dim: 1 dim: 1 }
      dilation: { dim: 1 dim: 1 }
      group: 1
      base_axis: 1
    }
  }
  function {
    name: "BatchNormalization_12"
    type: "BatchNormalization"
    input: "Convolution_12"
    input: "BatchNormalization_12/bn/beta"
    input: "BatchNormalization_12/bn/gamma"
    input: "BatchNormalization_12/bn/mean"
    input: "BatchNormalization_12/bn/var"
    output: "BatchNormalization_12"
    batch_normalization_param {
      axes: 1
      decay_rate: 0.5
      eps: 0.01
      batch_stat: False
    }
  }
  function {
    name: "ReLU_12"
    type: "ReLU"
    input: "BatchNormalization_12"
    output: "ReLU_12"
    relu_param {
      inplace: True
    }
  }
  function {
    name: "Convolution_13"
    type: "Convolution"
    input: "ReLU_12"
    input: "Convolution_13/conv/W"
    output: "Convolution_13"
    convolution_param {
      pad: { dim: 0 dim: 0 }
      stride: { dim: 1 dim: 1 }
      dilation: { dim: 1 dim: 1 }
      group: 1
      base_axis: 1
    }
  }
  function {
    name: "BatchNormalization_13"
    type: "BatchNormalization"
    input: "Convolution_13"
    input: "BatchNormalization_13/bn/beta"
    input: "BatchNormalization_13/bn/gamma"
    input: "BatchNormalization_13/bn/mean"
    input: "BatchNormalization_13/bn/var"
    output: "BatchNormalization_13"
    batch_normalization_param {
      axes: 1
      decay_rate: 0.5
      eps: 0.01
      batch_stat: False
    }
  }
  function {
    name: "Add2_4"
    type: "Add2"
    input: "BatchNormalization_13"
    input: "MaxPooling_3"
    output: "Add2_4"
    add2_param {
      inplace: False
    }
  }
  function {
    name: "ReLU_13"
    type: "ReLU"
    input: "Add2_4"
    output: "ReLU_13"
    relu_param {
      inplace: True
    }
  }
  function {
    name: "AveragePooling"
    type: "AveragePooling"
    input: "ReLU_13"
    output: "AveragePooling"
    average_pooling_param {
      kernel: { dim: 4 dim: 4 }
      stride: { dim: 4 dim: 4 }
      ignore_border: true
      pad: { dim: 0 dim: 0 }
      including_pad: true
    }
  }
  function {
    name: "Affine"
    type: "Affine"
    input: "AveragePooling"
    input: "Affine/affine/W"
    input: "Affine/affine/b"
    output: "Affine"
    affine_param {
      base_axis: 1
    }
  }
  function {
    name: "Softmax"
    type: "Softmax"
    input: "Affine"
    output: "Softmax"
    softmax_param {
      axis: 1
    }
  }
  function {
    name: "CategoricalCrossEntropy"
    type: "CategoricalCrossEntropy"
    input: "Softmax"
    input: "CategoricalCrossEntropy_T"
    output: "CategoricalCrossEntropy"
    categorical_cross_entropy_param {
      axis: 1
    }
  }
}
network {
  name: "MainRuntime"
  batch_size: 64
  repeat_info {
    id: "RepeatStart"
    times: 2
  }
  repeat_info {
    id: "RepeatStart_2"
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
    shape: { dim: 64 dim: 1 dim: 5 dim: 5 }
    initializer {
      type: "NormalConvolutionGlorot"
      multiplier: 1
    }
  }
  variable {
    name: "BatchNormalization/bn/beta"
    type: "Parameter"
    shape: { dim: 1 dim: 64 dim: 1 dim: 1 }
    initializer {
      type: "Constant"
      multiplier: 0
    }
  }
  variable {
    name: "BatchNormalization/bn/gamma"
    type: "Parameter"
    shape: { dim: 1 dim: 64 dim: 1 dim: 1 }
    initializer {
      type: "Constant"
      multiplier: 1
    }
  }
  variable {
    name: "BatchNormalization/bn/mean"
    type: "Parameter"
    shape: { dim: 1 dim: 64 dim: 1 dim: 1 }
    initializer {
      type: "Constant"
      multiplier: 0
    }
  }
  variable {
    name: "BatchNormalization/bn/var"
    type: "Parameter"
    shape: { dim: 1 dim: 64 dim: 1 dim: 1 }
    initializer {
      type: "Constant"
      multiplier: 0
    }
  }
  variable {
    name: "Convolution_2/conv/W"
    type: "Parameter"
    shape: { dim: 64 dim: 64 dim: 1 dim: 1 }
    initializer {
      type: "NormalConvolutionGlorot"
      multiplier: 1
    }
  }
  variable {
    name: "BatchNormalization_2/bn/beta"
    type: "Parameter"
    shape: { dim: 1 dim: 64 dim: 1 dim: 1 }
    initializer {
      type: "Constant"
      multiplier: 0
    }
  }
  variable {
    name: "BatchNormalization_2/bn/gamma"
    type: "Parameter"
    shape: { dim: 1 dim: 64 dim: 1 dim: 1 }
    initializer {
      type: "Constant"
      multiplier: 1
    }
  }
  variable {
    name: "BatchNormalization_2/bn/mean"
    type: "Parameter"
    shape: { dim: 1 dim: 64 dim: 1 dim: 1 }
    initializer {
      type: "Constant"
      multiplier: 0
    }
  }
  variable {
    name: "BatchNormalization_2/bn/var"
    type: "Parameter"
    shape: { dim: 1 dim: 64 dim: 1 dim: 1 }
    initializer {
      type: "Constant"
      multiplier: 0
    }
  }
  variable {
    name: "Convolution_3/conv/W"
    type: "Parameter"
    shape: { dim: 64 dim: 64 dim: 3 dim: 3 }
    initializer {
      type: "NormalConvolutionGlorot"
      multiplier: 1
    }
  }
  variable {
    name: "BatchNormalization_3/bn/beta"
    type: "Parameter"
    shape: { dim: 1 dim: 64 dim: 1 dim: 1 }
    initializer {
      type: "Constant"
      multiplier: 0
    }
  }
  variable {
    name: "BatchNormalization_3/bn/gamma"
    type: "Parameter"
    shape: { dim: 1 dim: 64 dim: 1 dim: 1 }
    initializer {
      type: "Constant"
      multiplier: 1
    }
  }
  variable {
    name: "BatchNormalization_3/bn/mean"
    type: "Parameter"
    shape: { dim: 1 dim: 64 dim: 1 dim: 1 }
    initializer {
      type: "Constant"
      multiplier: 0
    }
  }
  variable {
    name: "BatchNormalization_3/bn/var"
    type: "Parameter"
    shape: { dim: 1 dim: 64 dim: 1 dim: 1 }
    initializer {
      type: "Constant"
      multiplier: 0
    }
  }
  variable {
    name: "Convolution_4/conv/W"
    type: "Parameter"
    shape: { dim: 64 dim: 64 dim: 1 dim: 1 }
    initializer {
      type: "NormalConvolutionGlorot"
      multiplier: 1
    }
  }
  variable {
    name: "BatchNormalization_4/bn/beta"
    type: "Parameter"
    shape: { dim: 1 dim: 64 dim: 1 dim: 1 }
    initializer {
      type: "Constant"
      multiplier: 0
    }
  }
  variable {
    name: "BatchNormalization_4/bn/gamma"
    type: "Parameter"
    shape: { dim: 1 dim: 64 dim: 1 dim: 1 }
    initializer {
      type: "Constant"
      multiplier: 1
    }
  }
  variable {
    name: "BatchNormalization_4/bn/mean"
    type: "Parameter"
    shape: { dim: 1 dim: 64 dim: 1 dim: 1 }
    initializer {
      type: "Constant"
      multiplier: 0
    }
  }
  variable {
    name: "BatchNormalization_4/bn/var"
    type: "Parameter"
    shape: { dim: 1 dim: 64 dim: 1 dim: 1 }
    initializer {
      type: "Constant"
      multiplier: 0
    }
  }
  variable {
    name: "Convolution_5{RepeatStart}/conv/W"
    type: "Parameter"
    repeat_id: "RepeatStart"
    shape: { dim: 64 dim: 64 dim: 1 dim: 1 }
    initializer {
      type: "NormalConvolutionGlorot"
      multiplier: 1
    }
  }
  variable {
    name: "BatchNormalization_5{RepeatStart}/bn/beta"
    type: "Parameter"
    repeat_id: "RepeatStart"
    shape: { dim: 1 dim: 64 dim: 1 dim: 1 }
    initializer {
      type: "Constant"
      multiplier: 0
    }
  }
  variable {
    name: "BatchNormalization_5{RepeatStart}/bn/gamma"
    type: "Parameter"
    repeat_id: "RepeatStart"
    shape: { dim: 1 dim: 64 dim: 1 dim: 1 }
    initializer {
      type: "Constant"
      multiplier: 1
    }
  }
  variable {
    name: "BatchNormalization_5{RepeatStart}/bn/mean"
    type: "Parameter"
    repeat_id: "RepeatStart"
    shape: { dim: 1 dim: 64 dim: 1 dim: 1 }
    initializer {
      type: "Constant"
      multiplier: 0
    }
  }
  variable {
    name: "BatchNormalization_5{RepeatStart}/bn/var"
    type: "Parameter"
    repeat_id: "RepeatStart"
    shape: { dim: 1 dim: 64 dim: 1 dim: 1 }
    initializer {
      type: "Constant"
      multiplier: 0
    }
  }
  variable {
    name: "Convolution_6{RepeatStart}/conv/W"
    type: "Parameter"
    repeat_id: "RepeatStart"
    shape: { dim: 64 dim: 64 dim: 3 dim: 3 }
    initializer {
      type: "NormalConvolutionGlorot"
      multiplier: 1
    }
  }
  variable {
    name: "BatchNormalization_6{RepeatStart}/bn/beta"
    type: "Parameter"
    repeat_id: "RepeatStart"
    shape: { dim: 1 dim: 64 dim: 1 dim: 1 }
    initializer {
      type: "Constant"
      multiplier: 0
    }
  }
  variable {
    name: "BatchNormalization_6{RepeatStart}/bn/gamma"
    type: "Parameter"
    repeat_id: "RepeatStart"
    shape: { dim: 1 dim: 64 dim: 1 dim: 1 }
    initializer {
      type: "Constant"
      multiplier: 1
    }
  }
  variable {
    name: "BatchNormalization_6{RepeatStart}/bn/mean"
    type: "Parameter"
    repeat_id: "RepeatStart"
    shape: { dim: 1 dim: 64 dim: 1 dim: 1 }
    initializer {
      type: "Constant"
      multiplier: 0
    }
  }
  variable {
    name: "BatchNormalization_6{RepeatStart}/bn/var"
    type: "Parameter"
    repeat_id: "RepeatStart"
    shape: { dim: 1 dim: 64 dim: 1 dim: 1 }
    initializer {
      type: "Constant"
      multiplier: 0
    }
  }
  variable {
    name: "Convolution_7{RepeatStart}/conv/W"
    type: "Parameter"
    repeat_id: "RepeatStart"
    shape: { dim: 64 dim: 64 dim: 1 dim: 1 }
    initializer {
      type: "NormalConvolutionGlorot"
      multiplier: 1
    }
  }
  variable {
    name: "BatchNormalization_7{RepeatStart}/bn/beta"
    type: "Parameter"
    repeat_id: "RepeatStart"
    shape: { dim: 1 dim: 64 dim: 1 dim: 1 }
    initializer {
      type: "Constant"
      multiplier: 0
    }
  }
  variable {
    name: "BatchNormalization_7{RepeatStart}/bn/gamma"
    type: "Parameter"
    repeat_id: "RepeatStart"
    shape: { dim: 1 dim: 64 dim: 1 dim: 1 }
    initializer {
      type: "Constant"
      multiplier: 1
    }
  }
  variable {
    name: "BatchNormalization_7{RepeatStart}/bn/mean"
    type: "Parameter"
    repeat_id: "RepeatStart"
    shape: { dim: 1 dim: 64 dim: 1 dim: 1 }
    initializer {
      type: "Constant"
      multiplier: 0
    }
  }
  variable {
    name: "BatchNormalization_7{RepeatStart}/bn/var"
    type: "Parameter"
    repeat_id: "RepeatStart"
    shape: { dim: 1 dim: 64 dim: 1 dim: 1 }
    initializer {
      type: "Constant"
      multiplier: 0
    }
  }
  variable {
    name: "Convolution_8{RepeatStart_2}/conv/W"
    type: "Parameter"
    repeat_id: "RepeatStart_2"
    shape: { dim: 64 dim: 64 dim: 1 dim: 1 }
    initializer {
      type: "NormalConvolutionGlorot"
      multiplier: 1
    }
  }
  variable {
    name: "BatchNormalization_8{RepeatStart_2}/bn/beta"
    type: "Parameter"
    repeat_id: "RepeatStart_2"
    shape: { dim: 1 dim: 64 dim: 1 dim: 1 }
    initializer {
      type: "Constant"
      multiplier: 0
    }
  }
  variable {
    name: "BatchNormalization_8{RepeatStart_2}/bn/gamma"
    type: "Parameter"
    repeat_id: "RepeatStart_2"
    shape: { dim: 1 dim: 64 dim: 1 dim: 1 }
    initializer {
      type: "Constant"
      multiplier: 1
    }
  }
  variable {
    name: "BatchNormalization_8{RepeatStart_2}/bn/mean"
    type: "Parameter"
    repeat_id: "RepeatStart_2"
    shape: { dim: 1 dim: 64 dim: 1 dim: 1 }
    initializer {
      type: "Constant"
      multiplier: 0
    }
  }
  variable {
    name: "BatchNormalization_8{RepeatStart_2}/bn/var"
    type: "Parameter"
    repeat_id: "RepeatStart_2"
    shape: { dim: 1 dim: 64 dim: 1 dim: 1 }
    initializer {
      type: "Constant"
      multiplier: 0
    }
  }
  variable {
    name: "Convolution_9{RepeatStart_2}/conv/W"
    type: "Parameter"
    repeat_id: "RepeatStart_2"
    shape: { dim: 64 dim: 64 dim: 3 dim: 3 }
    initializer {
      type: "NormalConvolutionGlorot"
      multiplier: 1
    }
  }
  variable {
    name: "BatchNormalization_9{RepeatStart_2}/bn/beta"
    type: "Parameter"
    repeat_id: "RepeatStart_2"
    shape: { dim: 1 dim: 64 dim: 1 dim: 1 }
    initializer {
      type: "Constant"
      multiplier: 0
    }
  }
  variable {
    name: "BatchNormalization_9{RepeatStart_2}/bn/gamma"
    type: "Parameter"
    repeat_id: "RepeatStart_2"
    shape: { dim: 1 dim: 64 dim: 1 dim: 1 }
    initializer {
      type: "Constant"
      multiplier: 1
    }
  }
  variable {
    name: "BatchNormalization_9{RepeatStart_2}/bn/mean"
    type: "Parameter"
    repeat_id: "RepeatStart_2"
    shape: { dim: 1 dim: 64 dim: 1 dim: 1 }
    initializer {
      type: "Constant"
      multiplier: 0
    }
  }
  variable {
    name: "BatchNormalization_9{RepeatStart_2}/bn/var"
    type: "Parameter"
    repeat_id: "RepeatStart_2"
    shape: { dim: 1 dim: 64 dim: 1 dim: 1 }
    initializer {
      type: "Constant"
      multiplier: 0
    }
  }
  variable {
    name: "Convolution_10{RepeatStart_2}/conv/W"
    type: "Parameter"
    repeat_id: "RepeatStart_2"
    shape: { dim: 64 dim: 64 dim: 1 dim: 1 }
    initializer {
      type: "NormalConvolutionGlorot"
      multiplier: 1
    }
  }
  variable {
    name: "BatchNormalization_10{RepeatStart_2}/bn/beta"
    type: "Parameter"
    repeat_id: "RepeatStart_2"
    shape: { dim: 1 dim: 64 dim: 1 dim: 1 }
    initializer {
      type: "Constant"
      multiplier: 0
    }
  }
  variable {
    name: "BatchNormalization_10{RepeatStart_2}/bn/gamma"
    type: "Parameter"
    repeat_id: "RepeatStart_2"
    shape: { dim: 1 dim: 64 dim: 1 dim: 1 }
    initializer {
      type: "Constant"
      multiplier: 1
    }
  }
  variable {
    name: "BatchNormalization_10{RepeatStart_2}/bn/mean"
    type: "Parameter"
    repeat_id: "RepeatStart_2"
    shape: { dim: 1 dim: 64 dim: 1 dim: 1 }
    initializer {
      type: "Constant"
      multiplier: 0
    }
  }
  variable {
    name: "BatchNormalization_10{RepeatStart_2}/bn/var"
    type: "Parameter"
    repeat_id: "RepeatStart_2"
    shape: { dim: 1 dim: 64 dim: 1 dim: 1 }
    initializer {
      type: "Constant"
      multiplier: 0
    }
  }
  variable {
    name: "Convolution_11/conv/W"
    type: "Parameter"
    shape: { dim: 64 dim: 64 dim: 1 dim: 1 }
    initializer {
      type: "NormalConvolutionGlorot"
      multiplier: 1
    }
  }
  variable {
    name: "BatchNormalization_11/bn/beta"
    type: "Parameter"
    shape: { dim: 1 dim: 64 dim: 1 dim: 1 }
    initializer {
      type: "Constant"
      multiplier: 0
    }
  }
  variable {
    name: "BatchNormalization_11/bn/gamma"
    type: "Parameter"
    shape: { dim: 1 dim: 64 dim: 1 dim: 1 }
    initializer {
      type: "Constant"
      multiplier: 1
    }
  }
  variable {
    name: "BatchNormalization_11/bn/mean"
    type: "Parameter"
    shape: { dim: 1 dim: 64 dim: 1 dim: 1 }
    initializer {
      type: "Constant"
      multiplier: 0
    }
  }
  variable {
    name: "BatchNormalization_11/bn/var"
    type: "Parameter"
    shape: { dim: 1 dim: 64 dim: 1 dim: 1 }
    initializer {
      type: "Constant"
      multiplier: 0
    }
  }
  variable {
    name: "Convolution_12/conv/W"
    type: "Parameter"
    shape: { dim: 64 dim: 64 dim: 3 dim: 3 }
    initializer {
      type: "NormalConvolutionGlorot"
      multiplier: 1
    }
  }
  variable {
    name: "BatchNormalization_12/bn/beta"
    type: "Parameter"
    shape: { dim: 1 dim: 64 dim: 1 dim: 1 }
    initializer {
      type: "Constant"
      multiplier: 0
    }
  }
  variable {
    name: "BatchNormalization_12/bn/gamma"
    type: "Parameter"
    shape: { dim: 1 dim: 64 dim: 1 dim: 1 }
    initializer {
      type: "Constant"
      multiplier: 1
    }
  }
  variable {
    name: "BatchNormalization_12/bn/mean"
    type: "Parameter"
    shape: { dim: 1 dim: 64 dim: 1 dim: 1 }
    initializer {
      type: "Constant"
      multiplier: 0
    }
  }
  variable {
    name: "BatchNormalization_12/bn/var"
    type: "Parameter"
    shape: { dim: 1 dim: 64 dim: 1 dim: 1 }
    initializer {
      type: "Constant"
      multiplier: 0
    }
  }
  variable {
    name: "Convolution_13/conv/W"
    type: "Parameter"
    shape: { dim: 64 dim: 64 dim: 1 dim: 1 }
    initializer {
      type: "NormalConvolutionGlorot"
      multiplier: 1
    }
  }
  variable {
    name: "BatchNormalization_13/bn/beta"
    type: "Parameter"
    shape: { dim: 1 dim: 64 dim: 1 dim: 1 }
    initializer {
      type: "Constant"
      multiplier: 0
    }
  }
  variable {
    name: "BatchNormalization_13/bn/gamma"
    type: "Parameter"
    shape: { dim: 1 dim: 64 dim: 1 dim: 1 }
    initializer {
      type: "Constant"
      multiplier: 1
    }
  }
  variable {
    name: "BatchNormalization_13/bn/mean"
    type: "Parameter"
    shape: { dim: 1 dim: 64 dim: 1 dim: 1 }
    initializer {
      type: "Constant"
      multiplier: 0
    }
  }
  variable {
    name: "BatchNormalization_13/bn/var"
    type: "Parameter"
    shape: { dim: 1 dim: 64 dim: 1 dim: 1 }
    initializer {
      type: "Constant"
      multiplier: 0
    }
  }
  variable {
    name: "Affine/affine/W"
    type: "Parameter"
    shape: { dim: 64 dim: 10 }
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
    name: "Convolution"
    type: "Buffer"
    shape: { dim:-1 dim: 64 dim: 32 dim: 32 }
  }
  variable {
    name: "BatchNormalization"
    type: "Buffer"
    shape: { dim:-1 dim: 64 dim: 32 dim: 32 }
  }
  variable {
    name: "ReLU"
    type: "Buffer"
    shape: { dim:-1 dim: 64 dim: 32 dim: 32 }
  }
  variable {
    name: "Convolution_2"
    type: "Buffer"
    shape: { dim:-1 dim: 64 dim: 32 dim: 32 }
  }
  variable {
    name: "BatchNormalization_2"
    type: "Buffer"
    shape: { dim:-1 dim: 64 dim: 32 dim: 32 }
  }
  variable {
    name: "ReLU_2"
    type: "Buffer"
    shape: { dim:-1 dim: 64 dim: 32 dim: 32 }
  }
  variable {
    name: "Convolution_3"
    type: "Buffer"
    shape: { dim:-1 dim: 64 dim: 32 dim: 32 }
  }
  variable {
    name: "BatchNormalization_3"
    type: "Buffer"
    shape: { dim:-1 dim: 64 dim: 32 dim: 32 }
  }
  variable {
    name: "ReLU_3"
    type: "Buffer"
    shape: { dim:-1 dim: 64 dim: 32 dim: 32 }
  }
  variable {
    name: "Convolution_4"
    type: "Buffer"
    shape: { dim:-1 dim: 64 dim: 32 dim: 32 }
  }
  variable {
    name: "BatchNormalization_4"
    type: "Buffer"
    shape: { dim:-1 dim: 64 dim: 32 dim: 32 }
  }
  variable {
    name: "Add2"
    type: "Buffer"
    shape: { dim:-1 dim: 64 dim: 32 dim: 32 }
  }
  variable {
    name: "ReLU_4"
    type: "Buffer"
    shape: { dim:-1 dim: 64 dim: 32 dim: 32 }
  }
  variable {
    name: "MaxPooling"
    type: "Buffer"
    shape: { dim:-1 dim: 64 dim: 16 dim: 16 }
  }
  variable {
    name: "RepeatStart"
    type: "Buffer"
    repeat_id: "RepeatStart"
    shape: { dim:-1 dim: 64 dim: 16 dim: 16 }
  }
  variable {
    name: "Convolution_5"
    type: "Buffer"
    repeat_id: "RepeatStart"
    shape: { dim:-1 dim: 64 dim: 16 dim: 16 }
  }
  variable {
    name: "BatchNormalization_5"
    type: "Buffer"
    repeat_id: "RepeatStart"
    shape: { dim:-1 dim: 64 dim: 16 dim: 16 }
  }
  variable {
    name: "ReLU_5"
    type: "Buffer"
    repeat_id: "RepeatStart"
    shape: { dim:-1 dim: 64 dim: 16 dim: 16 }
  }
  variable {
    name: "Convolution_6"
    type: "Buffer"
    repeat_id: "RepeatStart"
    shape: { dim:-1 dim: 64 dim: 16 dim: 16 }
  }
  variable {
    name: "BatchNormalization_6"
    type: "Buffer"
    repeat_id: "RepeatStart"
    shape: { dim:-1 dim: 64 dim: 16 dim: 16 }
  }
  variable {
    name: "ReLU_6"
    type: "Buffer"
    repeat_id: "RepeatStart"
    shape: { dim:-1 dim: 64 dim: 16 dim: 16 }
  }
  variable {
    name: "Convolution_7"
    type: "Buffer"
    repeat_id: "RepeatStart"
    shape: { dim:-1 dim: 64 dim: 16 dim: 16 }
  }
  variable {
    name: "BatchNormalization_7"
    type: "Buffer"
    repeat_id: "RepeatStart"
    shape: { dim:-1 dim: 64 dim: 16 dim: 16 }
  }
  variable {
    name: "Add2_2"
    type: "Buffer"
    repeat_id: "RepeatStart"
    shape: { dim:-1 dim: 64 dim: 16 dim: 16 }
  }
  variable {
    name: "ReLU_7"
    type: "Buffer"
    repeat_id: "RepeatStart"
    shape: { dim:-1 dim: 64 dim: 16 dim: 16 }
  }
  variable {
    name: "RepeatEnd"
    type: "Buffer"
    shape: { dim:-1 dim: 64 dim: 16 dim: 16 }
  }
  variable {
    name: "MaxPooling_2"
    type: "Buffer"
    shape: { dim:-1 dim: 64 dim: 8 dim: 8 }
  }
  variable {
    name: "RepeatStart_2"
    type: "Buffer"
    repeat_id: "RepeatStart_2"
    shape: { dim:-1 dim: 64 dim: 8 dim: 8 }
  }
  variable {
    name: "Convolution_8"
    type: "Buffer"
    repeat_id: "RepeatStart_2"
    shape: { dim:-1 dim: 64 dim: 8 dim: 8 }
  }
  variable {
    name: "BatchNormalization_8"
    type: "Buffer"
    repeat_id: "RepeatStart_2"
    shape: { dim:-1 dim: 64 dim: 8 dim: 8 }
  }
  variable {
    name: "ReLU_8"
    type: "Buffer"
    repeat_id: "RepeatStart_2"
    shape: { dim:-1 dim: 64 dim: 8 dim: 8 }
  }
  variable {
    name: "Convolution_9"
    type: "Buffer"
    repeat_id: "RepeatStart_2"
    shape: { dim:-1 dim: 64 dim: 8 dim: 8 }
  }
  variable {
    name: "BatchNormalization_9"
    type: "Buffer"
    repeat_id: "RepeatStart_2"
    shape: { dim:-1 dim: 64 dim: 8 dim: 8 }
  }
  variable {
    name: "ReLU_9"
    type: "Buffer"
    repeat_id: "RepeatStart_2"
    shape: { dim:-1 dim: 64 dim: 8 dim: 8 }
  }
  variable {
    name: "Convolution_10"
    type: "Buffer"
    repeat_id: "RepeatStart_2"
    shape: { dim:-1 dim: 64 dim: 8 dim: 8 }
  }
  variable {
    name: "BatchNormalization_10"
    type: "Buffer"
    repeat_id: "RepeatStart_2"
    shape: { dim:-1 dim: 64 dim: 8 dim: 8 }
  }
  variable {
    name: "Add2_3"
    type: "Buffer"
    repeat_id: "RepeatStart_2"
    shape: { dim:-1 dim: 64 dim: 8 dim: 8 }
  }
  variable {
    name: "ReLU_10"
    type: "Buffer"
    repeat_id: "RepeatStart_2"
    shape: { dim:-1 dim: 64 dim: 8 dim: 8 }
  }
  variable {
    name: "RepeatEnd_2"
    type: "Buffer"
    shape: { dim:-1 dim: 64 dim: 8 dim: 8 }
  }
  variable {
    name: "MaxPooling_3"
    type: "Buffer"
    shape: { dim:-1 dim: 64 dim: 4 dim: 4 }
  }
  variable {
    name: "Convolution_11"
    type: "Buffer"
    shape: { dim:-1 dim: 64 dim: 4 dim: 4 }
  }
  variable {
    name: "BatchNormalization_11"
    type: "Buffer"
    shape: { dim:-1 dim: 64 dim: 4 dim: 4 }
  }
  variable {
    name: "ReLU_11"
    type: "Buffer"
    shape: { dim:-1 dim: 64 dim: 4 dim: 4 }
  }
  variable {
    name: "Convolution_12"
    type: "Buffer"
    shape: { dim:-1 dim: 64 dim: 4 dim: 4 }
  }
  variable {
    name: "BatchNormalization_12"
    type: "Buffer"
    shape: { dim:-1 dim: 64 dim: 4 dim: 4 }
  }
  variable {
    name: "ReLU_12"
    type: "Buffer"
    shape: { dim:-1 dim: 64 dim: 4 dim: 4 }
  }
  variable {
    name: "Convolution_13"
    type: "Buffer"
    shape: { dim:-1 dim: 64 dim: 4 dim: 4 }
  }
  variable {
    name: "BatchNormalization_13"
    type: "Buffer"
    shape: { dim:-1 dim: 64 dim: 4 dim: 4 }
  }
  variable {
    name: "Add2_4"
    type: "Buffer"
    shape: { dim:-1 dim: 64 dim: 4 dim: 4 }
  }
  variable {
    name: "ReLU_13"
    type: "Buffer"
    shape: { dim:-1 dim: 64 dim: 4 dim: 4 }
  }
  variable {
    name: "AveragePooling"
    type: "Buffer"
    shape: { dim:-1 dim: 64 dim: 1 dim: 1 }
  }
  variable {
    name: "Affine"
    type: "Buffer"
    shape: { dim:-1 dim: 10 }
  }
  variable {
    name: "Softmax"
    type: "Buffer"
    shape: { dim:-1 dim: 10 }
  }
  function {
    name: "Convolution"
    type: "Convolution"
    input: "Input"
    input: "Convolution/conv/W"
    output: "Convolution"
    convolution_param {
      pad: { dim: 4 dim: 4 }
      stride: { dim: 1 dim: 1 }
      dilation: { dim: 1 dim: 1 }
      group: 1
      base_axis: 1
    }
  }
  function {
    name: "BatchNormalization"
    type: "BatchNormalization"
    input: "Convolution"
    input: "BatchNormalization/bn/beta"
    input: "BatchNormalization/bn/gamma"
    input: "BatchNormalization/bn/mean"
    input: "BatchNormalization/bn/var"
    output: "BatchNormalization"
    batch_normalization_param {
      axes: 1
      decay_rate: 0.5
      eps: 0.01
      batch_stat: False
    }
  }
  function {
    name: "ReLU"
    type: "ReLU"
    input: "BatchNormalization"
    output: "ReLU"
    relu_param {
      inplace: True
    }
  }
  function {
    name: "Convolution_2"
    type: "Convolution"
    input: "ReLU"
    input: "Convolution_2/conv/W"
    output: "Convolution_2"
    convolution_param {
      pad: { dim: 0 dim: 0 }
      stride: { dim: 1 dim: 1 }
      dilation: { dim: 1 dim: 1 }
      group: 1
      base_axis: 1
    }
  }
  function {
    name: "BatchNormalization_2"
    type: "BatchNormalization"
    input: "Convolution_2"
    input: "BatchNormalization_2/bn/beta"
    input: "BatchNormalization_2/bn/gamma"
    input: "BatchNormalization_2/bn/mean"
    input: "BatchNormalization_2/bn/var"
    output: "BatchNormalization_2"
    batch_normalization_param {
      axes: 1
      decay_rate: 0.5
      eps: 0.01
      batch_stat: False
    }
  }
  function {
    name: "ReLU_2"
    type: "ReLU"
    input: "BatchNormalization_2"
    output: "ReLU_2"
    relu_param {
      inplace: True
    }
  }
  function {
    name: "Convolution_3"
    type: "Convolution"
    input: "ReLU_2"
    input: "Convolution_3/conv/W"
    output: "Convolution_3"
    convolution_param {
      pad: { dim: 1 dim: 1 }
      stride: { dim: 1 dim: 1 }
      dilation: { dim: 1 dim: 1 }
      group: 1
      base_axis: 1
    }
  }
  function {
    name: "BatchNormalization_3"
    type: "BatchNormalization"
    input: "Convolution_3"
    input: "BatchNormalization_3/bn/beta"
    input: "BatchNormalization_3/bn/gamma"
    input: "BatchNormalization_3/bn/mean"
    input: "BatchNormalization_3/bn/var"
    output: "BatchNormalization_3"
    batch_normalization_param {
      axes: 1
      decay_rate: 0.5
      eps: 0.01
      batch_stat: False
    }
  }
  function {
    name: "ReLU_3"
    type: "ReLU"
    input: "BatchNormalization_3"
    output: "ReLU_3"
    relu_param {
      inplace: True
    }
  }
  function {
    name: "Convolution_4"
    type: "Convolution"
    input: "ReLU_3"
    input: "Convolution_4/conv/W"
    output: "Convolution_4"
    convolution_param {
      pad: { dim: 0 dim: 0 }
      stride: { dim: 1 dim: 1 }
      dilation: { dim: 1 dim: 1 }
      group: 1
      base_axis: 1
    }
  }
  function {
    name: "BatchNormalization_4"
    type: "BatchNormalization"
    input: "Convolution_4"
    input: "BatchNormalization_4/bn/beta"
    input: "BatchNormalization_4/bn/gamma"
    input: "BatchNormalization_4/bn/mean"
    input: "BatchNormalization_4/bn/var"
    output: "BatchNormalization_4"
    batch_normalization_param {
      axes: 1
      decay_rate: 0.5
      eps: 0.01
      batch_stat: False
    }
  }
  function {
    name: "Add2"
    type: "Add2"
    input: "BatchNormalization_4"
    input: "ReLU"
    output: "Add2"
    add2_param {
      inplace: False
    }
  }
  function {
    name: "ReLU_4"
    type: "ReLU"
    input: "Add2"
    output: "ReLU_4"
    relu_param {
      inplace: True
    }
  }
  function {
    name: "MaxPooling"
    type: "MaxPooling"
    input: "ReLU_4"
    output: "MaxPooling"
    max_pooling_param {
      kernel: { dim: 2 dim: 2 }
      stride: { dim: 2 dim: 2 }
      ignore_border: true
      pad: { dim: 0 dim: 0 }
    }
  }
  function {
    name: "RepeatStart"
    type: "RepeatStart"
    repeat_id: "RepeatStart"
    input: "MaxPooling"
    input: "ReLU_7"
    output: "RepeatStart"
    repeat_param {
      repeat_id: "RepeatStart"
    }
  }
  function {
    name: "Convolution_5"
    type: "Convolution"
    repeat_id: "RepeatStart"
    input: "RepeatStart"
    input: "Convolution_5{RepeatStart}/conv/W"
    output: "Convolution_5"
    convolution_param {
      pad: { dim: 0 dim: 0 }
      stride: { dim: 1 dim: 1 }
      dilation: { dim: 1 dim: 1 }
      group: 1
      base_axis: 1
    }
  }
  function {
    name: "BatchNormalization_5"
    type: "BatchNormalization"
    repeat_id: "RepeatStart"
    input: "Convolution_5"
    input: "BatchNormalization_5{RepeatStart}/bn/beta"
    input: "BatchNormalization_5{RepeatStart}/bn/gamma"
    input: "BatchNormalization_5{RepeatStart}/bn/mean"
    input: "BatchNormalization_5{RepeatStart}/bn/var"
    output: "BatchNormalization_5"
    batch_normalization_param {
      axes: 1
      decay_rate: 0.5
      eps: 0.01
      batch_stat: False
    }
  }
  function {
    name: "ReLU_5"
    type: "ReLU"
    repeat_id: "RepeatStart"
    input: "BatchNormalization_5"
    output: "ReLU_5"
    relu_param {
      inplace: True
    }
  }
  function {
    name: "Convolution_6"
    type: "Convolution"
    repeat_id: "RepeatStart"
    input: "ReLU_5"
    input: "Convolution_6{RepeatStart}/conv/W"
    output: "Convolution_6"
    convolution_param {
      pad: { dim: 1 dim: 1 }
      stride: { dim: 1 dim: 1 }
      dilation: { dim: 1 dim: 1 }
      group: 1
      base_axis: 1
    }
  }
  function {
    name: "BatchNormalization_6"
    type: "BatchNormalization"
    repeat_id: "RepeatStart"
    input: "Convolution_6"
    input: "BatchNormalization_6{RepeatStart}/bn/beta"
    input: "BatchNormalization_6{RepeatStart}/bn/gamma"
    input: "BatchNormalization_6{RepeatStart}/bn/mean"
    input: "BatchNormalization_6{RepeatStart}/bn/var"
    output: "BatchNormalization_6"
    batch_normalization_param {
      axes: 1
      decay_rate: 0.5
      eps: 0.01
      batch_stat: False
    }
  }
  function {
    name: "ReLU_6"
    type: "ReLU"
    repeat_id: "RepeatStart"
    input: "BatchNormalization_6"
    output: "ReLU_6"
    relu_param {
      inplace: True
    }
  }
  function {
    name: "Convolution_7"
    type: "Convolution"
    repeat_id: "RepeatStart"
    input: "ReLU_6"
    input: "Convolution_7{RepeatStart}/conv/W"
    output: "Convolution_7"
    convolution_param {
      pad: { dim: 0 dim: 0 }
      stride: { dim: 1 dim: 1 }
      dilation: { dim: 1 dim: 1 }
      group: 1
      base_axis: 1
    }
  }
  function {
    name: "BatchNormalization_7"
    type: "BatchNormalization"
    repeat_id: "RepeatStart"
    input: "Convolution_7"
    input: "BatchNormalization_7{RepeatStart}/bn/beta"
    input: "BatchNormalization_7{RepeatStart}/bn/gamma"
    input: "BatchNormalization_7{RepeatStart}/bn/mean"
    input: "BatchNormalization_7{RepeatStart}/bn/var"
    output: "BatchNormalization_7"
    batch_normalization_param {
      axes: 1
      decay_rate: 0.5
      eps: 0.01
      batch_stat: False
    }
  }
  function {
    name: "Add2_2"
    type: "Add2"
    repeat_id: "RepeatStart"
    input: "BatchNormalization_7"
    input: "RepeatStart"
    output: "Add2_2"
    add2_param {
      inplace: False
    }
  }
  function {
    name: "ReLU_7"
    type: "ReLU"
    repeat_id: "RepeatStart"
    input: "Add2_2"
    output: "ReLU_7"
    relu_param {
      inplace: True
    }
  }
  function {
    name: "RepeatEnd"
    type: "RepeatEnd"
    input: "ReLU_7"
    output: "RepeatEnd"
    repeat_param {
      repeat_id: "RepeatStart"
      times: 2
    }
  }
  function {
    name: "MaxPooling_2"
    type: "MaxPooling"
    input: "RepeatEnd"
    output: "MaxPooling_2"
    max_pooling_param {
      kernel: { dim: 2 dim: 2 }
      stride: { dim: 2 dim: 2 }
      ignore_border: true
      pad: { dim: 0 dim: 0 }
    }
  }
  function {
    name: "RepeatStart_2"
    type: "RepeatStart"
    repeat_id: "RepeatStart_2"
    input: "MaxPooling_2"
    input: "ReLU_10"
    output: "RepeatStart_2"
    repeat_param {
      repeat_id: "RepeatStart_2"
    }
  }
  function {
    name: "Convolution_8"
    type: "Convolution"
    repeat_id: "RepeatStart_2"
    input: "RepeatStart_2"
    input: "Convolution_8{RepeatStart_2}/conv/W"
    output: "Convolution_8"
    convolution_param {
      pad: { dim: 0 dim: 0 }
      stride: { dim: 1 dim: 1 }
      dilation: { dim: 1 dim: 1 }
      group: 1
      base_axis: 1
    }
  }
  function {
    name: "BatchNormalization_8"
    type: "BatchNormalization"
    repeat_id: "RepeatStart_2"
    input: "Convolution_8"
    input: "BatchNormalization_8{RepeatStart_2}/bn/beta"
    input: "BatchNormalization_8{RepeatStart_2}/bn/gamma"
    input: "BatchNormalization_8{RepeatStart_2}/bn/mean"
    input: "BatchNormalization_8{RepeatStart_2}/bn/var"
    output: "BatchNormalization_8"
    batch_normalization_param {
      axes: 1
      decay_rate: 0.5
      eps: 0.01
      batch_stat: False
    }
  }
  function {
    name: "ReLU_8"
    type: "ReLU"
    repeat_id: "RepeatStart_2"
    input: "BatchNormalization_8"
    output: "ReLU_8"
    relu_param {
      inplace: True
    }
  }
  function {
    name: "Convolution_9"
    type: "Convolution"
    repeat_id: "RepeatStart_2"
    input: "ReLU_8"
    input: "Convolution_9{RepeatStart_2}/conv/W"
    output: "Convolution_9"
    convolution_param {
      pad: { dim: 1 dim: 1 }
      stride: { dim: 1 dim: 1 }
      dilation: { dim: 1 dim: 1 }
      group: 1
      base_axis: 1
    }
  }
  function {
    name: "BatchNormalization_9"
    type: "BatchNormalization"
    repeat_id: "RepeatStart_2"
    input: "Convolution_9"
    input: "BatchNormalization_9{RepeatStart_2}/bn/beta"
    input: "BatchNormalization_9{RepeatStart_2}/bn/gamma"
    input: "BatchNormalization_9{RepeatStart_2}/bn/mean"
    input: "BatchNormalization_9{RepeatStart_2}/bn/var"
    output: "BatchNormalization_9"
    batch_normalization_param {
      axes: 1
      decay_rate: 0.5
      eps: 0.01
      batch_stat: False
    }
  }
  function {
    name: "ReLU_9"
    type: "ReLU"
    repeat_id: "RepeatStart_2"
    input: "BatchNormalization_9"
    output: "ReLU_9"
    relu_param {
      inplace: True
    }
  }
  function {
    name: "Convolution_10"
    type: "Convolution"
    repeat_id: "RepeatStart_2"
    input: "ReLU_9"
    input: "Convolution_10{RepeatStart_2}/conv/W"
    output: "Convolution_10"
    convolution_param {
      pad: { dim: 0 dim: 0 }
      stride: { dim: 1 dim: 1 }
      dilation: { dim: 1 dim: 1 }
      group: 1
      base_axis: 1
    }
  }
  function {
    name: "BatchNormalization_10"
    type: "BatchNormalization"
    repeat_id: "RepeatStart_2"
    input: "Convolution_10"
    input: "BatchNormalization_10{RepeatStart_2}/bn/beta"
    input: "BatchNormalization_10{RepeatStart_2}/bn/gamma"
    input: "BatchNormalization_10{RepeatStart_2}/bn/mean"
    input: "BatchNormalization_10{RepeatStart_2}/bn/var"
    output: "BatchNormalization_10"
    batch_normalization_param {
      axes: 1
      decay_rate: 0.5
      eps: 0.01
      batch_stat: False
    }
  }
  function {
    name: "Add2_3"
    type: "Add2"
    repeat_id: "RepeatStart_2"
    input: "BatchNormalization_10"
    input: "RepeatStart_2"
    output: "Add2_3"
    add2_param {
      inplace: False
    }
  }
  function {
    name: "ReLU_10"
    type: "ReLU"
    repeat_id: "RepeatStart_2"
    input: "Add2_3"
    output: "ReLU_10"
    relu_param {
      inplace: True
    }
  }
  function {
    name: "RepeatEnd_2"
    type: "RepeatEnd"
    input: "ReLU_10"
    output: "RepeatEnd_2"
    repeat_param {
      repeat_id: "RepeatStart_2"
      times: 2
    }
  }
  function {
    name: "MaxPooling_3"
    type: "MaxPooling"
    input: "RepeatEnd_2"
    output: "MaxPooling_3"
    max_pooling_param {
      kernel: { dim: 2 dim: 2 }
      stride: { dim: 2 dim: 2 }
      ignore_border: true
      pad: { dim: 0 dim: 0 }
    }
  }
  function {
    name: "Convolution_11"
    type: "Convolution"
    input: "MaxPooling_3"
    input: "Convolution_11/conv/W"
    output: "Convolution_11"
    convolution_param {
      pad: { dim: 0 dim: 0 }
      stride: { dim: 1 dim: 1 }
      dilation: { dim: 1 dim: 1 }
      group: 1
      base_axis: 1
    }
  }
  function {
    name: "BatchNormalization_11"
    type: "BatchNormalization"
    input: "Convolution_11"
    input: "BatchNormalization_11/bn/beta"
    input: "BatchNormalization_11/bn/gamma"
    input: "BatchNormalization_11/bn/mean"
    input: "BatchNormalization_11/bn/var"
    output: "BatchNormalization_11"
    batch_normalization_param {
      axes: 1
      decay_rate: 0.5
      eps: 0.01
      batch_stat: False
    }
  }
  function {
    name: "ReLU_11"
    type: "ReLU"
    input: "BatchNormalization_11"
    output: "ReLU_11"
    relu_param {
      inplace: True
    }
  }
  function {
    name: "Convolution_12"
    type: "Convolution"
    input: "ReLU_11"
    input: "Convolution_12/conv/W"
    output: "Convolution_12"
    convolution_param {
      pad: { dim: 1 dim: 1 }
      stride: { dim: 1 dim: 1 }
      dilation: { dim: 1 dim: 1 }
      group: 1
      base_axis: 1
    }
  }
  function {
    name: "BatchNormalization_12"
    type: "BatchNormalization"
    input: "Convolution_12"
    input: "BatchNormalization_12/bn/beta"
    input: "BatchNormalization_12/bn/gamma"
    input: "BatchNormalization_12/bn/mean"
    input: "BatchNormalization_12/bn/var"
    output: "BatchNormalization_12"
    batch_normalization_param {
      axes: 1
      decay_rate: 0.5
      eps: 0.01
      batch_stat: False
    }
  }
  function {
    name: "ReLU_12"
    type: "ReLU"
    input: "BatchNormalization_12"
    output: "ReLU_12"
    relu_param {
      inplace: True
    }
  }
  function {
    name: "Convolution_13"
    type: "Convolution"
    input: "ReLU_12"
    input: "Convolution_13/conv/W"
    output: "Convolution_13"
    convolution_param {
      pad: { dim: 0 dim: 0 }
      stride: { dim: 1 dim: 1 }
      dilation: { dim: 1 dim: 1 }
      group: 1
      base_axis: 1
    }
  }
  function {
    name: "BatchNormalization_13"
    type: "BatchNormalization"
    input: "Convolution_13"
    input: "BatchNormalization_13/bn/beta"
    input: "BatchNormalization_13/bn/gamma"
    input: "BatchNormalization_13/bn/mean"
    input: "BatchNormalization_13/bn/var"
    output: "BatchNormalization_13"
    batch_normalization_param {
      axes: 1
      decay_rate: 0.5
      eps: 0.01
      batch_stat: False
    }
  }
  function {
    name: "Add2_4"
    type: "Add2"
    input: "BatchNormalization_13"
    input: "MaxPooling_3"
    output: "Add2_4"
    add2_param {
      inplace: False
    }
  }
  function {
    name: "ReLU_13"
    type: "ReLU"
    input: "Add2_4"
    output: "ReLU_13"
    relu_param {
      inplace: True
    }
  }
  function {
    name: "AveragePooling"
    type: "AveragePooling"
    input: "ReLU_13"
    output: "AveragePooling"
    average_pooling_param {
      kernel: { dim: 4 dim: 4 }
      stride: { dim: 4 dim: 4 }
      ignore_border: true
      pad: { dim: 0 dim: 0 }
      including_pad: true
    }
  }
  function {
    name: "Affine"
    type: "Affine"
    input: "AveragePooling"
    input: "Affine/affine/W"
    input: "Affine/affine/b"
    output: "Affine"
    affine_param {
      base_axis: 1
    }
  }
  function {
    name: "Softmax"
    type: "Softmax"
    input: "Affine"
    output: "Softmax"
    softmax_param {
      axis: 1
    }
  }
}
dataset {
  name: "Training"
  uri: "C:\\Utils\\neural_network_console_internal_v120\\samples\\sample_dataset\\mnist\\mnist_training.csv"
  cache_dir: "C:\\Utils\\neural_network_console_internal_v120\\samples\\sample_dataset\\mnist\\mnist_training.cache"
  overwrite_cache: False
  create_cache_explicitly: True
  shuffle: true
  no_image_normalization: False
  batch_size: 64
}
dataset {
  name: "Validation"
  uri: "C:\\Utils\\neural_network_console_internal_v120\\samples\\sample_dataset\\mnist\\mnist_test.csv"
  cache_dir: "C:\\Utils\\neural_network_console_internal_v120\\samples\\sample_dataset\\mnist\\mnist_test.cache"
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
  data_variable {
    variable_name: "CategoricalCrossEntropy_T"
    data_name: "y"
  }
  loss_variable {
    variable_name: "CategoricalCrossEntropy"
  }
  parameter_variable {
    variable_name: "Convolution/conv/W"
    learning_rate_multiplier: 1
  }
  parameter_variable {
    variable_name: "BatchNormalization/bn/beta"
    learning_rate_multiplier: 1
  }
  parameter_variable {
    variable_name: "BatchNormalization/bn/gamma"
    learning_rate_multiplier: 1
  }
  parameter_variable {
    variable_name: "BatchNormalization/bn/mean"
    learning_rate_multiplier: 0
  }
  parameter_variable {
    variable_name: "BatchNormalization/bn/var"
    learning_rate_multiplier: 0
  }
  parameter_variable {
    variable_name: "Convolution_2/conv/W"
    learning_rate_multiplier: 1
  }
  parameter_variable {
    variable_name: "BatchNormalization_2/bn/beta"
    learning_rate_multiplier: 1
  }
  parameter_variable {
    variable_name: "BatchNormalization_2/bn/gamma"
    learning_rate_multiplier: 1
  }
  parameter_variable {
    variable_name: "BatchNormalization_2/bn/mean"
    learning_rate_multiplier: 0
  }
  parameter_variable {
    variable_name: "BatchNormalization_2/bn/var"
    learning_rate_multiplier: 0
  }
  parameter_variable {
    variable_name: "Convolution_3/conv/W"
    learning_rate_multiplier: 1
  }
  parameter_variable {
    variable_name: "BatchNormalization_3/bn/beta"
    learning_rate_multiplier: 1
  }
  parameter_variable {
    variable_name: "BatchNormalization_3/bn/gamma"
    learning_rate_multiplier: 1
  }
  parameter_variable {
    variable_name: "BatchNormalization_3/bn/mean"
    learning_rate_multiplier: 0
  }
  parameter_variable {
    variable_name: "BatchNormalization_3/bn/var"
    learning_rate_multiplier: 0
  }
  parameter_variable {
    variable_name: "Convolution_4/conv/W"
    learning_rate_multiplier: 1
  }
  parameter_variable {
    variable_name: "BatchNormalization_4/bn/beta"
    learning_rate_multiplier: 1
  }
  parameter_variable {
    variable_name: "BatchNormalization_4/bn/gamma"
    learning_rate_multiplier: 1
  }
  parameter_variable {
    variable_name: "BatchNormalization_4/bn/mean"
    learning_rate_multiplier: 0
  }
  parameter_variable {
    variable_name: "BatchNormalization_4/bn/var"
    learning_rate_multiplier: 0
  }
  parameter_variable {
    variable_name: "Convolution_5{RepeatStart}/conv/W"
    learning_rate_multiplier: 1
  }
  parameter_variable {
    variable_name: "BatchNormalization_5{RepeatStart}/bn/beta"
    learning_rate_multiplier: 1
  }
  parameter_variable {
    variable_name: "BatchNormalization_5{RepeatStart}/bn/gamma"
    learning_rate_multiplier: 1
  }
  parameter_variable {
    variable_name: "BatchNormalization_5{RepeatStart}/bn/mean"
    learning_rate_multiplier: 0
  }
  parameter_variable {
    variable_name: "BatchNormalization_5{RepeatStart}/bn/var"
    learning_rate_multiplier: 0
  }
  parameter_variable {
    variable_name: "Convolution_6{RepeatStart}/conv/W"
    learning_rate_multiplier: 1
  }
  parameter_variable {
    variable_name: "BatchNormalization_6{RepeatStart}/bn/beta"
    learning_rate_multiplier: 1
  }
  parameter_variable {
    variable_name: "BatchNormalization_6{RepeatStart}/bn/gamma"
    learning_rate_multiplier: 1
  }
  parameter_variable {
    variable_name: "BatchNormalization_6{RepeatStart}/bn/mean"
    learning_rate_multiplier: 0
  }
  parameter_variable {
    variable_name: "BatchNormalization_6{RepeatStart}/bn/var"
    learning_rate_multiplier: 0
  }
  parameter_variable {
    variable_name: "Convolution_7{RepeatStart}/conv/W"
    learning_rate_multiplier: 1
  }
  parameter_variable {
    variable_name: "BatchNormalization_7{RepeatStart}/bn/beta"
    learning_rate_multiplier: 1
  }
  parameter_variable {
    variable_name: "BatchNormalization_7{RepeatStart}/bn/gamma"
    learning_rate_multiplier: 1
  }
  parameter_variable {
    variable_name: "BatchNormalization_7{RepeatStart}/bn/mean"
    learning_rate_multiplier: 0
  }
  parameter_variable {
    variable_name: "BatchNormalization_7{RepeatStart}/bn/var"
    learning_rate_multiplier: 0
  }
  parameter_variable {
    variable_name: "Convolution_8{RepeatStart_2}/conv/W"
    learning_rate_multiplier: 1
  }
  parameter_variable {
    variable_name: "BatchNormalization_8{RepeatStart_2}/bn/beta"
    learning_rate_multiplier: 1
  }
  parameter_variable {
    variable_name: "BatchNormalization_8{RepeatStart_2}/bn/gamma"
    learning_rate_multiplier: 1
  }
  parameter_variable {
    variable_name: "BatchNormalization_8{RepeatStart_2}/bn/mean"
    learning_rate_multiplier: 0
  }
  parameter_variable {
    variable_name: "BatchNormalization_8{RepeatStart_2}/bn/var"
    learning_rate_multiplier: 0
  }
  parameter_variable {
    variable_name: "Convolution_9{RepeatStart_2}/conv/W"
    learning_rate_multiplier: 1
  }
  parameter_variable {
    variable_name: "BatchNormalization_9{RepeatStart_2}/bn/beta"
    learning_rate_multiplier: 1
  }
  parameter_variable {
    variable_name: "BatchNormalization_9{RepeatStart_2}/bn/gamma"
    learning_rate_multiplier: 1
  }
  parameter_variable {
    variable_name: "BatchNormalization_9{RepeatStart_2}/bn/mean"
    learning_rate_multiplier: 0
  }
  parameter_variable {
    variable_name: "BatchNormalization_9{RepeatStart_2}/bn/var"
    learning_rate_multiplier: 0
  }
  parameter_variable {
    variable_name: "Convolution_10{RepeatStart_2}/conv/W"
    learning_rate_multiplier: 1
  }
  parameter_variable {
    variable_name: "BatchNormalization_10{RepeatStart_2}/bn/beta"
    learning_rate_multiplier: 1
  }
  parameter_variable {
    variable_name: "BatchNormalization_10{RepeatStart_2}/bn/gamma"
    learning_rate_multiplier: 1
  }
  parameter_variable {
    variable_name: "BatchNormalization_10{RepeatStart_2}/bn/mean"
    learning_rate_multiplier: 0
  }
  parameter_variable {
    variable_name: "BatchNormalization_10{RepeatStart_2}/bn/var"
    learning_rate_multiplier: 0
  }
  parameter_variable {
    variable_name: "Convolution_11/conv/W"
    learning_rate_multiplier: 1
  }
  parameter_variable {
    variable_name: "BatchNormalization_11/bn/beta"
    learning_rate_multiplier: 1
  }
  parameter_variable {
    variable_name: "BatchNormalization_11/bn/gamma"
    learning_rate_multiplier: 1
  }
  parameter_variable {
    variable_name: "BatchNormalization_11/bn/mean"
    learning_rate_multiplier: 0
  }
  parameter_variable {
    variable_name: "BatchNormalization_11/bn/var"
    learning_rate_multiplier: 0
  }
  parameter_variable {
    variable_name: "Convolution_12/conv/W"
    learning_rate_multiplier: 1
  }
  parameter_variable {
    variable_name: "BatchNormalization_12/bn/beta"
    learning_rate_multiplier: 1
  }
  parameter_variable {
    variable_name: "BatchNormalization_12/bn/gamma"
    learning_rate_multiplier: 1
  }
  parameter_variable {
    variable_name: "BatchNormalization_12/bn/mean"
    learning_rate_multiplier: 0
  }
  parameter_variable {
    variable_name: "BatchNormalization_12/bn/var"
    learning_rate_multiplier: 0
  }
  parameter_variable {
    variable_name: "Convolution_13/conv/W"
    learning_rate_multiplier: 1
  }
  parameter_variable {
    variable_name: "BatchNormalization_13/bn/beta"
    learning_rate_multiplier: 1
  }
  parameter_variable {
    variable_name: "BatchNormalization_13/bn/gamma"
    learning_rate_multiplier: 1
  }
  parameter_variable {
    variable_name: "BatchNormalization_13/bn/mean"
    learning_rate_multiplier: 0
  }
  parameter_variable {
    variable_name: "BatchNormalization_13/bn/var"
    learning_rate_multiplier: 0
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
    variable_name: "CategoricalCrossEntropy_T"
    data_name: "y"
  }
  monitor_variable {
    type: "Error"
    variable_name: "CategoricalCrossEntropy"
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
    variable_name: "CategoricalCrossEntropy_T"
    data_name: "y"
  }
  monitor_variable {
    type: "Error"
    variable_name: "CategoricalCrossEntropy"
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
    variable_name: "Softmax"
    data_name: "y'"
  }
  parameter_variable {
    variable_name: "Convolution/conv/W"
  }
  parameter_variable {
    variable_name: "BatchNormalization/bn/beta"
  }
  parameter_variable {
    variable_name: "BatchNormalization/bn/gamma"
  }
  parameter_variable {
    variable_name: "BatchNormalization/bn/mean"
  }
  parameter_variable {
    variable_name: "BatchNormalization/bn/var"
  }
  parameter_variable {
    variable_name: "Convolution_2/conv/W"
  }
  parameter_variable {
    variable_name: "BatchNormalization_2/bn/beta"
  }
  parameter_variable {
    variable_name: "BatchNormalization_2/bn/gamma"
  }
  parameter_variable {
    variable_name: "BatchNormalization_2/bn/mean"
  }
  parameter_variable {
    variable_name: "BatchNormalization_2/bn/var"
  }
  parameter_variable {
    variable_name: "Convolution_3/conv/W"
  }
  parameter_variable {
    variable_name: "BatchNormalization_3/bn/beta"
  }
  parameter_variable {
    variable_name: "BatchNormalization_3/bn/gamma"
  }
  parameter_variable {
    variable_name: "BatchNormalization_3/bn/mean"
  }
  parameter_variable {
    variable_name: "BatchNormalization_3/bn/var"
  }
  parameter_variable {
    variable_name: "Convolution_4/conv/W"
  }
  parameter_variable {
    variable_name: "BatchNormalization_4/bn/beta"
  }
  parameter_variable {
    variable_name: "BatchNormalization_4/bn/gamma"
  }
  parameter_variable {
    variable_name: "BatchNormalization_4/bn/mean"
  }
  parameter_variable {
    variable_name: "BatchNormalization_4/bn/var"
  }
  parameter_variable {
    variable_name: "Convolution_5{RepeatStart}/conv/W"
  }
  parameter_variable {
    variable_name: "BatchNormalization_5{RepeatStart}/bn/beta"
  }
  parameter_variable {
    variable_name: "BatchNormalization_5{RepeatStart}/bn/gamma"
  }
  parameter_variable {
    variable_name: "BatchNormalization_5{RepeatStart}/bn/mean"
  }
  parameter_variable {
    variable_name: "BatchNormalization_5{RepeatStart}/bn/var"
  }
  parameter_variable {
    variable_name: "Convolution_6{RepeatStart}/conv/W"
  }
  parameter_variable {
    variable_name: "BatchNormalization_6{RepeatStart}/bn/beta"
  }
  parameter_variable {
    variable_name: "BatchNormalization_6{RepeatStart}/bn/gamma"
  }
  parameter_variable {
    variable_name: "BatchNormalization_6{RepeatStart}/bn/mean"
  }
  parameter_variable {
    variable_name: "BatchNormalization_6{RepeatStart}/bn/var"
  }
  parameter_variable {
    variable_name: "Convolution_7{RepeatStart}/conv/W"
  }
  parameter_variable {
    variable_name: "BatchNormalization_7{RepeatStart}/bn/beta"
  }
  parameter_variable {
    variable_name: "BatchNormalization_7{RepeatStart}/bn/gamma"
  }
  parameter_variable {
    variable_name: "BatchNormalization_7{RepeatStart}/bn/mean"
  }
  parameter_variable {
    variable_name: "BatchNormalization_7{RepeatStart}/bn/var"
  }
  parameter_variable {
    variable_name: "Convolution_8{RepeatStart_2}/conv/W"
  }
  parameter_variable {
    variable_name: "BatchNormalization_8{RepeatStart_2}/bn/beta"
  }
  parameter_variable {
    variable_name: "BatchNormalization_8{RepeatStart_2}/bn/gamma"
  }
  parameter_variable {
    variable_name: "BatchNormalization_8{RepeatStart_2}/bn/mean"
  }
  parameter_variable {
    variable_name: "BatchNormalization_8{RepeatStart_2}/bn/var"
  }
  parameter_variable {
    variable_name: "Convolution_9{RepeatStart_2}/conv/W"
  }
  parameter_variable {
    variable_name: "BatchNormalization_9{RepeatStart_2}/bn/beta"
  }
  parameter_variable {
    variable_name: "BatchNormalization_9{RepeatStart_2}/bn/gamma"
  }
  parameter_variable {
    variable_name: "BatchNormalization_9{RepeatStart_2}/bn/mean"
  }
  parameter_variable {
    variable_name: "BatchNormalization_9{RepeatStart_2}/bn/var"
  }
  parameter_variable {
    variable_name: "Convolution_10{RepeatStart_2}/conv/W"
  }
  parameter_variable {
    variable_name: "BatchNormalization_10{RepeatStart_2}/bn/beta"
  }
  parameter_variable {
    variable_name: "BatchNormalization_10{RepeatStart_2}/bn/gamma"
  }
  parameter_variable {
    variable_name: "BatchNormalization_10{RepeatStart_2}/bn/mean"
  }
  parameter_variable {
    variable_name: "BatchNormalization_10{RepeatStart_2}/bn/var"
  }
  parameter_variable {
    variable_name: "Convolution_11/conv/W"
  }
  parameter_variable {
    variable_name: "BatchNormalization_11/bn/beta"
  }
  parameter_variable {
    variable_name: "BatchNormalization_11/bn/gamma"
  }
  parameter_variable {
    variable_name: "BatchNormalization_11/bn/mean"
  }
  parameter_variable {
    variable_name: "BatchNormalization_11/bn/var"
  }
  parameter_variable {
    variable_name: "Convolution_12/conv/W"
  }
  parameter_variable {
    variable_name: "BatchNormalization_12/bn/beta"
  }
  parameter_variable {
    variable_name: "BatchNormalization_12/bn/gamma"
  }
  parameter_variable {
    variable_name: "BatchNormalization_12/bn/mean"
  }
  parameter_variable {
    variable_name: "BatchNormalization_12/bn/var"
  }
  parameter_variable {
    variable_name: "Convolution_13/conv/W"
  }
  parameter_variable {
    variable_name: "BatchNormalization_13/bn/beta"
  }
  parameter_variable {
    variable_name: "BatchNormalization_13/bn/gamma"
  }
  parameter_variable {
    variable_name: "BatchNormalization_13/bn/mean"
  }
  parameter_variable {
    variable_name: "BatchNormalization_13/bn/var"
  }
  parameter_variable {
    variable_name: "Affine/affine/W"
  }
  parameter_variable {
    variable_name: "Affine/affine/b"
  }
}
'''
# The nntxt is generated from /home/woody/aid/nnabla-sample-data/sample_project/tutorial/basics/10_deep_mlp.files/20180720_181928/net.nntxt.
N00000021 = r'''global_config {
  default_context {
    array_class: "CudaCachedArray"
    device_id: "0"
    backends: "cudnn:float"
    backends: "cuda:float"
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
    shape: { dim: 784 dim: 256 }
    initializer {
      type: "NormalAffineGlorot"
      multiplier: 1
    }
  }
  variable {
    name: "Affine/affine/b"
    type: "Parameter"
    shape: { dim: 256 }
    initializer {
      type: "Constant"
      multiplier: 0
    }
  }
  variable {
    name: "Affine_2/affine/W"
    type: "Parameter"
    shape: { dim: 256 dim: 128 }
    initializer {
      type: "NormalAffineGlorot"
      multiplier: 1
    }
  }
  variable {
    name: "Affine_2/affine/b"
    type: "Parameter"
    shape: { dim: 128 }
    initializer {
      type: "Constant"
      multiplier: 0
    }
  }
  variable {
    name: "Affine_3/affine/W"
    type: "Parameter"
    shape: { dim: 128 dim: 64 }
    initializer {
      type: "NormalAffineGlorot"
      multiplier: 1
    }
  }
  variable {
    name: "Affine_3/affine/b"
    type: "Parameter"
    shape: { dim: 64 }
    initializer {
      type: "Constant"
      multiplier: 0
    }
  }
  variable {
    name: "Affine_4/affine/W"
    type: "Parameter"
    shape: { dim: 64 dim: 8 }
    initializer {
      type: "NormalAffineGlorot"
      multiplier: 1
    }
  }
  variable {
    name: "Affine_4/affine/b"
    type: "Parameter"
    shape: { dim: 8 }
    initializer {
      type: "Constant"
      multiplier: 0
    }
  }
  variable {
    name: "Affine_5/affine/W"
    type: "Parameter"
    shape: { dim: 8 dim: 1 }
    initializer {
      type: "NormalAffineGlorot"
      multiplier: 1
    }
  }
  variable {
    name: "Affine_5/affine/b"
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
    shape: { dim:-1 dim: 256 }
  }
  variable {
    name: "Tanh"
    type: "Buffer"
    shape: { dim:-1 dim: 256 }
  }
  variable {
    name: "Affine_2"
    type: "Buffer"
    shape: { dim:-1 dim: 128 }
  }
  variable {
    name: "Tanh_2"
    type: "Buffer"
    shape: { dim:-1 dim: 128 }
  }
  variable {
    name: "Affine_3"
    type: "Buffer"
    shape: { dim:-1 dim: 64 }
  }
  variable {
    name: "Tanh_3"
    type: "Buffer"
    shape: { dim:-1 dim: 64 }
  }
  variable {
    name: "Affine_4"
    type: "Buffer"
    shape: { dim:-1 dim: 8 }
  }
  variable {
    name: "Tanh_4"
    type: "Buffer"
    shape: { dim:-1 dim: 8 }
  }
  variable {
    name: "Affine_5"
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
    name: "Tanh"
    type: "Tanh"
    input: "Affine"
    output: "Tanh"
  }
  function {
    name: "Affine_2"
    type: "Affine"
    input: "Tanh"
    input: "Affine_2/affine/W"
    input: "Affine_2/affine/b"
    output: "Affine_2"
    affine_param {
      base_axis: 1
    }
  }
  function {
    name: "Tanh_2"
    type: "Tanh"
    input: "Affine_2"
    output: "Tanh_2"
  }
  function {
    name: "Affine_3"
    type: "Affine"
    input: "Tanh_2"
    input: "Affine_3/affine/W"
    input: "Affine_3/affine/b"
    output: "Affine_3"
    affine_param {
      base_axis: 1
    }
  }
  function {
    name: "Tanh_3"
    type: "Tanh"
    input: "Affine_3"
    output: "Tanh_3"
  }
  function {
    name: "Affine_4"
    type: "Affine"
    input: "Tanh_3"
    input: "Affine_4/affine/W"
    input: "Affine_4/affine/b"
    output: "Affine_4"
    affine_param {
      base_axis: 1
    }
  }
  function {
    name: "Tanh_4"
    type: "Tanh"
    input: "Affine_4"
    output: "Tanh_4"
  }
  function {
    name: "Affine_5"
    type: "Affine"
    input: "Tanh_4"
    input: "Affine_5/affine/W"
    input: "Affine_5/affine/b"
    output: "Affine_5"
    affine_param {
      base_axis: 1
    }
  }
  function {
    name: "Sigmoid"
    type: "Sigmoid"
    input: "Affine_5"
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
    shape: { dim: 784 dim: 256 }
    initializer {
      type: "NormalAffineGlorot"
      multiplier: 1
    }
  }
  variable {
    name: "Affine/affine/b"
    type: "Parameter"
    shape: { dim: 256 }
    initializer {
      type: "Constant"
      multiplier: 0
    }
  }
  variable {
    name: "Affine_2/affine/W"
    type: "Parameter"
    shape: { dim: 256 dim: 128 }
    initializer {
      type: "NormalAffineGlorot"
      multiplier: 1
    }
  }
  variable {
    name: "Affine_2/affine/b"
    type: "Parameter"
    shape: { dim: 128 }
    initializer {
      type: "Constant"
      multiplier: 0
    }
  }
  variable {
    name: "Affine_3/affine/W"
    type: "Parameter"
    shape: { dim: 128 dim: 64 }
    initializer {
      type: "NormalAffineGlorot"
      multiplier: 1
    }
  }
  variable {
    name: "Affine_3/affine/b"
    type: "Parameter"
    shape: { dim: 64 }
    initializer {
      type: "Constant"
      multiplier: 0
    }
  }
  variable {
    name: "Affine_4/affine/W"
    type: "Parameter"
    shape: { dim: 64 dim: 8 }
    initializer {
      type: "NormalAffineGlorot"
      multiplier: 1
    }
  }
  variable {
    name: "Affine_4/affine/b"
    type: "Parameter"
    shape: { dim: 8 }
    initializer {
      type: "Constant"
      multiplier: 0
    }
  }
  variable {
    name: "Affine_5/affine/W"
    type: "Parameter"
    shape: { dim: 8 dim: 1 }
    initializer {
      type: "NormalAffineGlorot"
      multiplier: 1
    }
  }
  variable {
    name: "Affine_5/affine/b"
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
    shape: { dim:-1 dim: 256 }
  }
  variable {
    name: "Tanh"
    type: "Buffer"
    shape: { dim:-1 dim: 256 }
  }
  variable {
    name: "Affine_2"
    type: "Buffer"
    shape: { dim:-1 dim: 128 }
  }
  variable {
    name: "Tanh_2"
    type: "Buffer"
    shape: { dim:-1 dim: 128 }
  }
  variable {
    name: "Affine_3"
    type: "Buffer"
    shape: { dim:-1 dim: 64 }
  }
  variable {
    name: "Tanh_3"
    type: "Buffer"
    shape: { dim:-1 dim: 64 }
  }
  variable {
    name: "Affine_4"
    type: "Buffer"
    shape: { dim:-1 dim: 8 }
  }
  variable {
    name: "Tanh_4"
    type: "Buffer"
    shape: { dim:-1 dim: 8 }
  }
  variable {
    name: "Affine_5"
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
    name: "Tanh"
    type: "Tanh"
    input: "Affine"
    output: "Tanh"
  }
  function {
    name: "Affine_2"
    type: "Affine"
    input: "Tanh"
    input: "Affine_2/affine/W"
    input: "Affine_2/affine/b"
    output: "Affine_2"
    affine_param {
      base_axis: 1
    }
  }
  function {
    name: "Tanh_2"
    type: "Tanh"
    input: "Affine_2"
    output: "Tanh_2"
  }
  function {
    name: "Affine_3"
    type: "Affine"
    input: "Tanh_2"
    input: "Affine_3/affine/W"
    input: "Affine_3/affine/b"
    output: "Affine_3"
    affine_param {
      base_axis: 1
    }
  }
  function {
    name: "Tanh_3"
    type: "Tanh"
    input: "Affine_3"
    output: "Tanh_3"
  }
  function {
    name: "Affine_4"
    type: "Affine"
    input: "Tanh_3"
    input: "Affine_4/affine/W"
    input: "Affine_4/affine/b"
    output: "Affine_4"
    affine_param {
      base_axis: 1
    }
  }
  function {
    name: "Tanh_4"
    type: "Tanh"
    input: "Affine_4"
    output: "Tanh_4"
  }
  function {
    name: "Affine_5"
    type: "Affine"
    input: "Tanh_4"
    input: "Affine_5/affine/W"
    input: "Affine_5/affine/b"
    output: "Affine_5"
    affine_param {
      base_axis: 1
    }
  }
  function {
    name: "Sigmoid"
    type: "Sigmoid"
    input: "Affine_5"
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
    shape: { dim: 784 dim: 256 }
    initializer {
      type: "NormalAffineGlorot"
      multiplier: 1
    }
  }
  variable {
    name: "Affine/affine/b"
    type: "Parameter"
    shape: { dim: 256 }
    initializer {
      type: "Constant"
      multiplier: 0
    }
  }
  variable {
    name: "Affine_2/affine/W"
    type: "Parameter"
    shape: { dim: 256 dim: 128 }
    initializer {
      type: "NormalAffineGlorot"
      multiplier: 1
    }
  }
  variable {
    name: "Affine_2/affine/b"
    type: "Parameter"
    shape: { dim: 128 }
    initializer {
      type: "Constant"
      multiplier: 0
    }
  }
  variable {
    name: "Affine_3/affine/W"
    type: "Parameter"
    shape: { dim: 128 dim: 64 }
    initializer {
      type: "NormalAffineGlorot"
      multiplier: 1
    }
  }
  variable {
    name: "Affine_3/affine/b"
    type: "Parameter"
    shape: { dim: 64 }
    initializer {
      type: "Constant"
      multiplier: 0
    }
  }
  variable {
    name: "Affine_4/affine/W"
    type: "Parameter"
    shape: { dim: 64 dim: 8 }
    initializer {
      type: "NormalAffineGlorot"
      multiplier: 1
    }
  }
  variable {
    name: "Affine_4/affine/b"
    type: "Parameter"
    shape: { dim: 8 }
    initializer {
      type: "Constant"
      multiplier: 0
    }
  }
  variable {
    name: "Affine_5/affine/W"
    type: "Parameter"
    shape: { dim: 8 dim: 1 }
    initializer {
      type: "NormalAffineGlorot"
      multiplier: 1
    }
  }
  variable {
    name: "Affine_5/affine/b"
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
    shape: { dim:-1 dim: 256 }
  }
  variable {
    name: "Tanh"
    type: "Buffer"
    shape: { dim:-1 dim: 256 }
  }
  variable {
    name: "Affine_2"
    type: "Buffer"
    shape: { dim:-1 dim: 128 }
  }
  variable {
    name: "Tanh_2"
    type: "Buffer"
    shape: { dim:-1 dim: 128 }
  }
  variable {
    name: "Affine_3"
    type: "Buffer"
    shape: { dim:-1 dim: 64 }
  }
  variable {
    name: "Tanh_3"
    type: "Buffer"
    shape: { dim:-1 dim: 64 }
  }
  variable {
    name: "Affine_4"
    type: "Buffer"
    shape: { dim:-1 dim: 8 }
  }
  variable {
    name: "Tanh_4"
    type: "Buffer"
    shape: { dim:-1 dim: 8 }
  }
  variable {
    name: "Affine_5"
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
    name: "Tanh"
    type: "Tanh"
    input: "Affine"
    output: "Tanh"
  }
  function {
    name: "Affine_2"
    type: "Affine"
    input: "Tanh"
    input: "Affine_2/affine/W"
    input: "Affine_2/affine/b"
    output: "Affine_2"
    affine_param {
      base_axis: 1
    }
  }
  function {
    name: "Tanh_2"
    type: "Tanh"
    input: "Affine_2"
    output: "Tanh_2"
  }
  function {
    name: "Affine_3"
    type: "Affine"
    input: "Tanh_2"
    input: "Affine_3/affine/W"
    input: "Affine_3/affine/b"
    output: "Affine_3"
    affine_param {
      base_axis: 1
    }
  }
  function {
    name: "Tanh_3"
    type: "Tanh"
    input: "Affine_3"
    output: "Tanh_3"
  }
  function {
    name: "Affine_4"
    type: "Affine"
    input: "Tanh_3"
    input: "Affine_4/affine/W"
    input: "Affine_4/affine/b"
    output: "Affine_4"
    affine_param {
      base_axis: 1
    }
  }
  function {
    name: "Tanh_4"
    type: "Tanh"
    input: "Affine_4"
    output: "Tanh_4"
  }
  function {
    name: "Affine_5"
    type: "Affine"
    input: "Tanh_4"
    input: "Affine_5/affine/W"
    input: "Affine_5/affine/b"
    output: "Affine_5"
    affine_param {
      base_axis: 1
    }
  }
  function {
    name: "Sigmoid"
    type: "Sigmoid"
    input: "Affine_5"
    output: "Sigmoid"
  }
}
dataset {
  name: "Training"
  uri: "C:\\Utils\\neural_network_console_internal_v120\\samples\\sample_dataset\\mnist\\small_mnist_4or9_training.csv"
  cache_dir: "C:\\Utils\\neural_network_console_internal_v120\\samples\\sample_dataset\\mnist\\small_mnist_4or9_training.cache"
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
  parameter_variable {
    variable_name: "Affine_2/affine/W"
    learning_rate_multiplier: 1
  }
  parameter_variable {
    variable_name: "Affine_2/affine/b"
    learning_rate_multiplier: 1
  }
  parameter_variable {
    variable_name: "Affine_3/affine/W"
    learning_rate_multiplier: 1
  }
  parameter_variable {
    variable_name: "Affine_3/affine/b"
    learning_rate_multiplier: 1
  }
  parameter_variable {
    variable_name: "Affine_4/affine/W"
    learning_rate_multiplier: 1
  }
  parameter_variable {
    variable_name: "Affine_4/affine/b"
    learning_rate_multiplier: 1
  }
  parameter_variable {
    variable_name: "Affine_5/affine/W"
    learning_rate_multiplier: 1
  }
  parameter_variable {
    variable_name: "Affine_5/affine/b"
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
  parameter_variable {
    variable_name: "Affine_2/affine/W"
  }
  parameter_variable {
    variable_name: "Affine_2/affine/b"
  }
  parameter_variable {
    variable_name: "Affine_3/affine/W"
  }
  parameter_variable {
    variable_name: "Affine_3/affine/b"
  }
  parameter_variable {
    variable_name: "Affine_4/affine/W"
  }
  parameter_variable {
    variable_name: "Affine_4/affine/b"
  }
  parameter_variable {
    variable_name: "Affine_5/affine/W"
  }
  parameter_variable {
    variable_name: "Affine_5/affine/b"
  }
}
'''
# The nntxt is generated from /home/woody/aid/nnabla-sample-data/sample_project/tutorial/basics/10_deep_mlp.files/20170804_140049/net.nntxt.
N00000022 = r'''global_config {
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
  variable {
    name: "Input"
    type: "Buffer"
    shape: { dim:-1 dim: 1 dim: 28 dim: 28 }
  }
  variable {
    name: "Affine"
    type: "Buffer"
    shape: { dim:-1 dim: 256 }
  }
  variable {
    name: "Tanh"
    type: "Buffer"
    shape: { dim:-1 dim: 256 }
  }
  variable {
    name: "Affine_2"
    type: "Buffer"
    shape: { dim:-1 dim: 128 }
  }
  variable {
    name: "Tanh_2"
    type: "Buffer"
    shape: { dim:-1 dim: 128 }
  }
  variable {
    name: "Affine_3"
    type: "Buffer"
    shape: { dim:-1 dim: 64 }
  }
  variable {
    name: "Tanh_3"
    type: "Buffer"
    shape: { dim:-1 dim: 64 }
  }
  variable {
    name: "Affine_4"
    type: "Buffer"
    shape: { dim:-1 dim: 8 }
  }
  variable {
    name: "Tanh_4"
    type: "Buffer"
    shape: { dim:-1 dim: 8 }
  }
  variable {
    name: "Affine_5"
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
  variable {
    name: "Affine/affine/W"
    type: "Parameter"
    shape: { dim: 784 dim: 256 }
    initializer {
      type: "NormalAffineGlorot"
      multiplier: 1
    }
  }
  variable {
    name: "Affine/affine/b"
    type: "Parameter"
    shape: { dim: 256 }
    initializer {
      type: "Constant"
      multiplier: 0.0
    }
  }
  variable {
    name: "Affine_2/affine/W"
    type: "Parameter"
    shape: { dim: 256 dim: 128 }
    initializer {
      type: "NormalAffineGlorot"
      multiplier: 1
    }
  }
  variable {
    name: "Affine_2/affine/b"
    type: "Parameter"
    shape: { dim: 128 }
    initializer {
      type: "Constant"
      multiplier: 0.0
    }
  }
  variable {
    name: "Affine_3/affine/W"
    type: "Parameter"
    shape: { dim: 128 dim: 64 }
    initializer {
      type: "NormalAffineGlorot"
      multiplier: 1
    }
  }
  variable {
    name: "Affine_3/affine/b"
    type: "Parameter"
    shape: { dim: 64 }
    initializer {
      type: "Constant"
      multiplier: 0.0
    }
  }
  variable {
    name: "Affine_4/affine/W"
    type: "Parameter"
    shape: { dim: 64 dim: 8 }
    initializer {
      type: "NormalAffineGlorot"
      multiplier: 1
    }
  }
  variable {
    name: "Affine_4/affine/b"
    type: "Parameter"
    shape: { dim: 8 }
    initializer {
      type: "Constant"
      multiplier: 0.0
    }
  }
  variable {
    name: "Affine_5/affine/W"
    type: "Parameter"
    shape: { dim: 8 dim: 1 }
    initializer {
      type: "NormalAffineGlorot"
      multiplier: 1
    }
  }
  variable {
    name: "Affine_5/affine/b"
    type: "Parameter"
    shape: { dim: 1 }
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
    name: "Tanh"
    type: "Tanh"
    input: "Affine"
    output: "Tanh"
  }
  function {
    name: "Affine_2"
    type: "Affine"
    input: "Tanh"
    input: "Affine_2/affine/W"
    input: "Affine_2/affine/b"
    output: "Affine_2"
    affine_param {
      base_axis: 1
    }
  }
  function {
    name: "Tanh_2"
    type: "Tanh"
    input: "Affine_2"
    output: "Tanh_2"
  }
  function {
    name: "Affine_3"
    type: "Affine"
    input: "Tanh_2"
    input: "Affine_3/affine/W"
    input: "Affine_3/affine/b"
    output: "Affine_3"
    affine_param {
      base_axis: 1
    }
  }
  function {
    name: "Tanh_3"
    type: "Tanh"
    input: "Affine_3"
    output: "Tanh_3"
  }
  function {
    name: "Affine_4"
    type: "Affine"
    input: "Tanh_3"
    input: "Affine_4/affine/W"
    input: "Affine_4/affine/b"
    output: "Affine_4"
    affine_param {
      base_axis: 1
    }
  }
  function {
    name: "Tanh_4"
    type: "Tanh"
    input: "Affine_4"
    output: "Tanh_4"
  }
  function {
    name: "Affine_5"
    type: "Affine"
    input: "Tanh_4"
    input: "Affine_5/affine/W"
    input: "Affine_5/affine/b"
    output: "Affine_5"
    affine_param {
      base_axis: 1
    }
  }
  function {
    name: "Sigmoid"
    type: "Sigmoid"
    input: "Affine_5"
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
    name: "Affine"
    type: "Buffer"
    shape: { dim:-1 dim: 256 }
  }
  variable {
    name: "Tanh"
    type: "Buffer"
    shape: { dim:-1 dim: 256 }
  }
  variable {
    name: "Affine_2"
    type: "Buffer"
    shape: { dim:-1 dim: 128 }
  }
  variable {
    name: "Tanh_2"
    type: "Buffer"
    shape: { dim:-1 dim: 128 }
  }
  variable {
    name: "Affine_3"
    type: "Buffer"
    shape: { dim:-1 dim: 64 }
  }
  variable {
    name: "Tanh_3"
    type: "Buffer"
    shape: { dim:-1 dim: 64 }
  }
  variable {
    name: "Affine_4"
    type: "Buffer"
    shape: { dim:-1 dim: 8 }
  }
  variable {
    name: "Tanh_4"
    type: "Buffer"
    shape: { dim:-1 dim: 8 }
  }
  variable {
    name: "Affine_5"
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
  variable {
    name: "Affine/affine/W"
    type: "Parameter"
    shape: { dim: 784 dim: 256 }
    initializer {
      type: "NormalAffineGlorot"
      multiplier: 1
    }
  }
  variable {
    name: "Affine/affine/b"
    type: "Parameter"
    shape: { dim: 256 }
    initializer {
      type: "Constant"
      multiplier: 0.0
    }
  }
  variable {
    name: "Affine_2/affine/W"
    type: "Parameter"
    shape: { dim: 256 dim: 128 }
    initializer {
      type: "NormalAffineGlorot"
      multiplier: 1
    }
  }
  variable {
    name: "Affine_2/affine/b"
    type: "Parameter"
    shape: { dim: 128 }
    initializer {
      type: "Constant"
      multiplier: 0.0
    }
  }
  variable {
    name: "Affine_3/affine/W"
    type: "Parameter"
    shape: { dim: 128 dim: 64 }
    initializer {
      type: "NormalAffineGlorot"
      multiplier: 1
    }
  }
  variable {
    name: "Affine_3/affine/b"
    type: "Parameter"
    shape: { dim: 64 }
    initializer {
      type: "Constant"
      multiplier: 0.0
    }
  }
  variable {
    name: "Affine_4/affine/W"
    type: "Parameter"
    shape: { dim: 64 dim: 8 }
    initializer {
      type: "NormalAffineGlorot"
      multiplier: 1
    }
  }
  variable {
    name: "Affine_4/affine/b"
    type: "Parameter"
    shape: { dim: 8 }
    initializer {
      type: "Constant"
      multiplier: 0.0
    }
  }
  variable {
    name: "Affine_5/affine/W"
    type: "Parameter"
    shape: { dim: 8 dim: 1 }
    initializer {
      type: "NormalAffineGlorot"
      multiplier: 1
    }
  }
  variable {
    name: "Affine_5/affine/b"
    type: "Parameter"
    shape: { dim: 1 }
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
    name: "Tanh"
    type: "Tanh"
    input: "Affine"
    output: "Tanh"
  }
  function {
    name: "Affine_2"
    type: "Affine"
    input: "Tanh"
    input: "Affine_2/affine/W"
    input: "Affine_2/affine/b"
    output: "Affine_2"
    affine_param {
      base_axis: 1
    }
  }
  function {
    name: "Tanh_2"
    type: "Tanh"
    input: "Affine_2"
    output: "Tanh_2"
  }
  function {
    name: "Affine_3"
    type: "Affine"
    input: "Tanh_2"
    input: "Affine_3/affine/W"
    input: "Affine_3/affine/b"
    output: "Affine_3"
    affine_param {
      base_axis: 1
    }
  }
  function {
    name: "Tanh_3"
    type: "Tanh"
    input: "Affine_3"
    output: "Tanh_3"
  }
  function {
    name: "Affine_4"
    type: "Affine"
    input: "Tanh_3"
    input: "Affine_4/affine/W"
    input: "Affine_4/affine/b"
    output: "Affine_4"
    affine_param {
      base_axis: 1
    }
  }
  function {
    name: "Tanh_4"
    type: "Tanh"
    input: "Affine_4"
    output: "Tanh_4"
  }
  function {
    name: "Affine_5"
    type: "Affine"
    input: "Tanh_4"
    input: "Affine_5/affine/W"
    input: "Affine_5/affine/b"
    output: "Affine_5"
    affine_param {
      base_axis: 1
    }
  }
  function {
    name: "Sigmoid"
    type: "Sigmoid"
    input: "Affine_5"
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
    name: "Affine"
    type: "Buffer"
    shape: { dim:-1 dim: 256 }
  }
  variable {
    name: "Tanh"
    type: "Buffer"
    shape: { dim:-1 dim: 256 }
  }
  variable {
    name: "Affine_2"
    type: "Buffer"
    shape: { dim:-1 dim: 128 }
  }
  variable {
    name: "Tanh_2"
    type: "Buffer"
    shape: { dim:-1 dim: 128 }
  }
  variable {
    name: "Affine_3"
    type: "Buffer"
    shape: { dim:-1 dim: 64 }
  }
  variable {
    name: "Tanh_3"
    type: "Buffer"
    shape: { dim:-1 dim: 64 }
  }
  variable {
    name: "Affine_4"
    type: "Buffer"
    shape: { dim:-1 dim: 8 }
  }
  variable {
    name: "Tanh_4"
    type: "Buffer"
    shape: { dim:-1 dim: 8 }
  }
  variable {
    name: "Affine_5"
    type: "Buffer"
    shape: { dim:-1 dim: 1 }
  }
  variable {
    name: "Sigmoid"
    type: "Buffer"
    shape: { dim:-1 dim: 1 }
  }
  variable {
    name: "Affine/affine/W"
    type: "Parameter"
    shape: { dim: 784 dim: 256 }
    initializer {
      type: "NormalAffineGlorot"
      multiplier: 1
    }
  }
  variable {
    name: "Affine/affine/b"
    type: "Parameter"
    shape: { dim: 256 }
    initializer {
      type: "Constant"
      multiplier: 0.0
    }
  }
  variable {
    name: "Affine_2/affine/W"
    type: "Parameter"
    shape: { dim: 256 dim: 128 }
    initializer {
      type: "NormalAffineGlorot"
      multiplier: 1
    }
  }
  variable {
    name: "Affine_2/affine/b"
    type: "Parameter"
    shape: { dim: 128 }
    initializer {
      type: "Constant"
      multiplier: 0.0
    }
  }
  variable {
    name: "Affine_3/affine/W"
    type: "Parameter"
    shape: { dim: 128 dim: 64 }
    initializer {
      type: "NormalAffineGlorot"
      multiplier: 1
    }
  }
  variable {
    name: "Affine_3/affine/b"
    type: "Parameter"
    shape: { dim: 64 }
    initializer {
      type: "Constant"
      multiplier: 0.0
    }
  }
  variable {
    name: "Affine_4/affine/W"
    type: "Parameter"
    shape: { dim: 64 dim: 8 }
    initializer {
      type: "NormalAffineGlorot"
      multiplier: 1
    }
  }
  variable {
    name: "Affine_4/affine/b"
    type: "Parameter"
    shape: { dim: 8 }
    initializer {
      type: "Constant"
      multiplier: 0.0
    }
  }
  variable {
    name: "Affine_5/affine/W"
    type: "Parameter"
    shape: { dim: 8 dim: 1 }
    initializer {
      type: "NormalAffineGlorot"
      multiplier: 1
    }
  }
  variable {
    name: "Affine_5/affine/b"
    type: "Parameter"
    shape: { dim: 1 }
    initializer {
      type: "Constant"
      multiplier: 0.0
    }
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
    name: "Tanh"
    type: "Tanh"
    input: "Affine"
    output: "Tanh"
  }
  function {
    name: "Affine_2"
    type: "Affine"
    input: "Tanh"
    input: "Affine_2/affine/W"
    input: "Affine_2/affine/b"
    output: "Affine_2"
    affine_param {
      base_axis: 1
    }
  }
  function {
    name: "Tanh_2"
    type: "Tanh"
    input: "Affine_2"
    output: "Tanh_2"
  }
  function {
    name: "Affine_3"
    type: "Affine"
    input: "Tanh_2"
    input: "Affine_3/affine/W"
    input: "Affine_3/affine/b"
    output: "Affine_3"
    affine_param {
      base_axis: 1
    }
  }
  function {
    name: "Tanh_3"
    type: "Tanh"
    input: "Affine_3"
    output: "Tanh_3"
  }
  function {
    name: "Affine_4"
    type: "Affine"
    input: "Tanh_3"
    input: "Affine_4/affine/W"
    input: "Affine_4/affine/b"
    output: "Affine_4"
    affine_param {
      base_axis: 1
    }
  }
  function {
    name: "Tanh_4"
    type: "Tanh"
    input: "Affine_4"
    output: "Tanh_4"
  }
  function {
    name: "Affine_5"
    type: "Affine"
    input: "Tanh_4"
    input: "Affine_5/affine/W"
    input: "Affine_5/affine/b"
    output: "Affine_5"
    affine_param {
      base_axis: 1
    }
  }
  function {
    name: "Sigmoid"
    type: "Sigmoid"
    input: "Affine_5"
    output: "Sigmoid"
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
  parameter_variable {
    variable_name: "Affine_3/affine/W"
    learning_rate_multiplier: 1.0
  }
  parameter_variable {
    variable_name: "Affine_3/affine/b"
    learning_rate_multiplier: 1.0
  }
  parameter_variable {
    variable_name: "Affine_4/affine/W"
    learning_rate_multiplier: 1.0
  }
  parameter_variable {
    variable_name: "Affine_4/affine/b"
    learning_rate_multiplier: 1.0
  }
  parameter_variable {
    variable_name: "Affine_5/affine/W"
    learning_rate_multiplier: 1.0
  }
  parameter_variable {
    variable_name: "Affine_5/affine/b"
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
  parameter_variable {
    variable_name: "Affine_2/affine/W"
  }
  parameter_variable {
    variable_name: "Affine_2/affine/b"
  }
  parameter_variable {
    variable_name: "Affine_3/affine/W"
  }
  parameter_variable {
    variable_name: "Affine_3/affine/b"
  }
  parameter_variable {
    variable_name: "Affine_4/affine/W"
  }
  parameter_variable {
    variable_name: "Affine_4/affine/b"
  }
  parameter_variable {
    variable_name: "Affine_5/affine/W"
  }
  parameter_variable {
    variable_name: "Affine_5/affine/b"
  }
}
'''
# The nntxt is generated from /home/woody/aid/nnabla-sample-data/sample_project/tutorial/basics/11_deconvolution.files/20170804_140111/net.nntxt.
N00000023 = r'''global_config {
  default_context {
    backend: "cpu|cuda"
    array_class: "CudaArray"
    device_id: "0"
    compute_backend: "default|cudnn"
  }
}
training_config {
  max_epoch: 500
  iter_per_epoch: 15
  save_best: true
}
network {
  name: "Main"
  batch_size: 100
  variable {
    name: "Input"
    type: "Buffer"
    shape: { dim:-1 dim: 1 dim: 28 dim: 28 }
  }
  variable {
    name: "Convolution"
    type: "Buffer"
    shape: { dim:-1 dim: 8 dim: 14 dim: 14 }
  }
  variable {
    name: "BatchNormalization"
    type: "Buffer"
    shape: { dim:-1 dim: 8 dim: 14 dim: 14 }
  }
  variable {
    name: "PRelu"
    type: "Buffer"
    shape: { dim:-1 dim: 8 dim: 14 dim: 14 }
  }
  variable {
    name: "Convolution_2"
    type: "Buffer"
    shape: { dim:-1 dim: 8 dim: 7 dim: 7 }
  }
  variable {
    name: "BatchNormalization_2"
    type: "Buffer"
    shape: { dim:-1 dim: 8 dim: 7 dim: 7 }
  }
  variable {
    name: "PRelu_2"
    type: "Buffer"
    shape: { dim:-1 dim: 8 dim: 7 dim: 7 }
  }
  variable {
    name: "Deconvolution"
    type: "Buffer"
    shape: { dim:-1 dim: 8 dim: 14 dim: 14 }
  }
  variable {
    name: "BatchNormalization_3"
    type: "Buffer"
    shape: { dim:-1 dim: 8 dim: 14 dim: 14 }
  }
  variable {
    name: "PRelu_3"
    type: "Buffer"
    shape: { dim:-1 dim: 8 dim: 14 dim: 14 }
  }
  variable {
    name: "Deconvolution_2"
    type: "Buffer"
    shape: { dim:-1 dim: 1 dim: 28 dim: 28 }
  }
  variable {
    name: "Sigmoid"
    type: "Buffer"
    shape: { dim:-1 dim: 1 dim: 28 dim: 28 }
  }
  variable {
    name: "Output"
    type: "Buffer"
    shape: { dim:-1 dim: 1 dim: 28 dim: 28 }
  }
  variable {
    name: "Convolution/conv/W"
    type: "Parameter"
    shape: { dim: 8 dim: 1 dim: 5 dim: 5 }
    initializer {
      type: "NormalConvolutionGlorot"
      multiplier: 1.0
    }
  }
  variable {
    name: "Convolution/conv/b"
    type: "Parameter"
    shape: { dim: 8 }
    initializer {
      type: "Constant"
      multiplier: 0.0
    }
  }
  variable {
    name: "BatchNormalization/bn/beta"
    type: "Parameter"
    shape: { dim: 1 dim: 8 dim: 1 dim: 1 }
    initializer {
      type: "Constant"
      multiplier: 0.0
    }
  }
  variable {
    name: "BatchNormalization/bn/gamma"
    type: "Parameter"
    shape: { dim: 1 dim: 8 dim: 1 dim: 1 }
    initializer {
      type: "Constant"
      multiplier: 1.0
    }
  }
  variable {
    name: "BatchNormalization/bn/mean"
    type: "Parameter"
    shape: { dim: 1 dim: 8 dim: 1 dim: 1 }
    initializer {
      type: "Constant"
      multiplier: 0.0
    }
  }
  variable {
    name: "BatchNormalization/bn/var"
    type: "Parameter"
    shape: { dim: 1 dim: 8 dim: 1 dim: 1 }
    initializer {
      type: "Constant"
      multiplier: 0.0
    }
  }
  variable {
    name: "PRelu/prelu/slope"
    type: "Parameter"
    shape: { dim: 8 }
    initializer {
      type: "Constant"
      multiplier: 0.25
    }
  }
  variable {
    name: "Convolution_2/conv/W"
    type: "Parameter"
    shape: { dim: 8 dim: 8 dim: 5 dim: 5 }
    initializer {
      type: "NormalConvolutionGlorot"
      multiplier: 1.0
    }
  }
  variable {
    name: "Convolution_2/conv/b"
    type: "Parameter"
    shape: { dim: 8 }
    initializer {
      type: "Constant"
      multiplier: 0.0
    }
  }
  variable {
    name: "BatchNormalization_2/bn/beta"
    type: "Parameter"
    shape: { dim: 1 dim: 8 dim: 1 dim: 1 }
    initializer {
      type: "Constant"
      multiplier: 0.0
    }
  }
  variable {
    name: "BatchNormalization_2/bn/gamma"
    type: "Parameter"
    shape: { dim: 1 dim: 8 dim: 1 dim: 1 }
    initializer {
      type: "Constant"
      multiplier: 1.0
    }
  }
  variable {
    name: "BatchNormalization_2/bn/mean"
    type: "Parameter"
    shape: { dim: 1 dim: 8 dim: 1 dim: 1 }
    initializer {
      type: "Constant"
      multiplier: 0.0
    }
  }
  variable {
    name: "BatchNormalization_2/bn/var"
    type: "Parameter"
    shape: { dim: 1 dim: 8 dim: 1 dim: 1 }
    initializer {
      type: "Constant"
      multiplier: 0.0
    }
  }
  variable {
    name: "PRelu_2/prelu/slope"
    type: "Parameter"
    shape: { dim: 8 }
    initializer {
      type: "Constant"
      multiplier: 0.25
    }
  }
  variable {
    name: "BatchNormalization_3/bn/beta"
    type: "Parameter"
    shape: { dim: 1 dim: 8 dim: 1 dim: 1 }
    initializer {
      type: "Constant"
      multiplier: 0.0
    }
  }
  variable {
    name: "BatchNormalization_3/bn/gamma"
    type: "Parameter"
    shape: { dim: 1 dim: 8 dim: 1 dim: 1 }
    initializer {
      type: "Constant"
      multiplier: 1.0
    }
  }
  variable {
    name: "BatchNormalization_3/bn/mean"
    type: "Parameter"
    shape: { dim: 1 dim: 8 dim: 1 dim: 1 }
    initializer {
      type: "Constant"
      multiplier: 0.0
    }
  }
  variable {
    name: "BatchNormalization_3/bn/var"
    type: "Parameter"
    shape: { dim: 1 dim: 8 dim: 1 dim: 1 }
    initializer {
      type: "Constant"
      multiplier: 0.0
    }
  }
  variable {
    name: "PRelu_3/prelu/slope"
    type: "Parameter"
    shape: { dim: 8 }
    initializer {
      type: "Constant"
      multiplier: 0.25
    }
  }
  variable {
    name: "Output_T"
    type: "Buffer"
    shape: { dim:-1 dim: 1 dim: 28 dim: 28 }
  }
  variable {
    name: "Deconvolution/conv/W"
    type: "Parameter"
    shape: { dim: 8 dim: 8 dim: 6 dim: 6 }
    initializer {
      type: "NormalConvolutionGlorot"
      multiplier: 1.0
    }
  }
  variable {
    name: "Deconvolution/conv/b"
    type: "Parameter"
    shape: { dim: 8 }
    initializer {
      type: "Constant"
      multiplier: 0.0
    }
  }
  variable {
    name: "Deconvolution_2/conv/W"
    type: "Parameter"
    shape: { dim: 8 dim: 1 dim: 6 dim: 6 }
    initializer {
      type: "NormalConvolutionGlorot"
      multiplier: 1.0
    }
  }
  variable {
    name: "Deconvolution_2/conv/b"
    type: "Parameter"
    shape: { dim: 1 }
    initializer {
      type: "Constant"
      multiplier: 0.0
    }
  }
  function {
    name: "Convolution"
    type: "Convolution"
    input: "Input"
    input: "Convolution/conv/W"
    input: "Convolution/conv/b"
    output: "Convolution"
    convolution_param {
      pad: { dim: 2 dim: 2 }
      stride: { dim: 2 dim: 2 }
      dilation: { dim: 1 dim: 1 }
      group: 1
      base_axis: 1
    }
  }
  function {
    name: "BatchNormalization"
    type: "BatchNormalization"
    input: "Convolution"
    input: "BatchNormalization/bn/beta"
    input: "BatchNormalization/bn/gamma"
    input: "BatchNormalization/bn/mean"
    input: "BatchNormalization/bn/var"
    output: "BatchNormalization"
    batch_normalization_param {
      axes: 1
      decay_rate: 0.5
      eps: 0.01
      batch_stat: true
    }
  }
  function {
    name: "PRelu"
    type: "PReLU"
    input: "BatchNormalization"
    input: "PRelu/prelu/slope"
    output: "PRelu"
    prelu_param {
      base_axis: 1
    }
  }
  function {
    name: "Convolution_2"
    type: "Convolution"
    input: "PRelu"
    input: "Convolution_2/conv/W"
    input: "Convolution_2/conv/b"
    output: "Convolution_2"
    convolution_param {
      pad: { dim: 2 dim: 2 }
      stride: { dim: 2 dim: 2 }
      dilation: { dim: 1 dim: 1 }
      group: 1
      base_axis: 1
    }
  }
  function {
    name: "BatchNormalization_2"
    type: "BatchNormalization"
    input: "Convolution_2"
    input: "BatchNormalization_2/bn/beta"
    input: "BatchNormalization_2/bn/gamma"
    input: "BatchNormalization_2/bn/mean"
    input: "BatchNormalization_2/bn/var"
    output: "BatchNormalization_2"
    batch_normalization_param {
      axes: 1
      decay_rate: 0.5
      eps: 0.01
      batch_stat: true
    }
  }
  function {
    name: "PRelu_2"
    type: "PReLU"
    input: "BatchNormalization_2"
    input: "PRelu_2/prelu/slope"
    output: "PRelu_2"
    prelu_param {
      base_axis: 1
    }
  }
  function {
    name: "Deconvolution"
    type: "Deconvolution"
    input: "PRelu_2"
    input: "Deconvolution/conv/W"
    input: "Deconvolution/conv/b"
    output: "Deconvolution"
    deconvolution_param {
      pad: { dim: 2 dim: 2 }
      stride: { dim: 2 dim: 2 }
      dilation: { dim: 1 dim: 1 }
      group: 1
      base_axis: 1
    }
  }
  function {
    name: "BatchNormalization_3"
    type: "BatchNormalization"
    input: "Deconvolution"
    input: "BatchNormalization_3/bn/beta"
    input: "BatchNormalization_3/bn/gamma"
    input: "BatchNormalization_3/bn/mean"
    input: "BatchNormalization_3/bn/var"
    output: "BatchNormalization_3"
    batch_normalization_param {
      axes: 1
      decay_rate: 0.5
      eps: 0.01
      batch_stat: true
    }
  }
  function {
    name: "PRelu_3"
    type: "PReLU"
    input: "BatchNormalization_3"
    input: "PRelu_3/prelu/slope"
    output: "PRelu_3"
    prelu_param {
      base_axis: 1
    }
  }
  function {
    name: "Deconvolution_2"
    type: "Deconvolution"
    input: "PRelu_3"
    input: "Deconvolution_2/conv/W"
    input: "Deconvolution_2/conv/b"
    output: "Deconvolution_2"
    deconvolution_param {
      pad: { dim: 2 dim: 2 }
      stride: { dim: 2 dim: 2 }
      dilation: { dim: 1 dim: 1 }
      group: 1
      base_axis: 1
    }
  }
  function {
    name: "Sigmoid"
    type: "Sigmoid"
    input: "Deconvolution_2"
    output: "Sigmoid"
  }
  function {
    name: "Output"
    type: "SquaredError"
    input: "Sigmoid"
    input: "Output_T"
    output: "Output"
  }
}
network {
  name: "MainValidation"
  batch_size: 100
  variable {
    name: "Input"
    type: "Buffer"
    shape: { dim:-1 dim: 1 dim: 28 dim: 28 }
  }
  variable {
    name: "Convolution"
    type: "Buffer"
    shape: { dim:-1 dim: 8 dim: 14 dim: 14 }
  }
  variable {
    name: "BatchNormalization"
    type: "Buffer"
    shape: { dim:-1 dim: 8 dim: 14 dim: 14 }
  }
  variable {
    name: "PRelu"
    type: "Buffer"
    shape: { dim:-1 dim: 8 dim: 14 dim: 14 }
  }
  variable {
    name: "Convolution_2"
    type: "Buffer"
    shape: { dim:-1 dim: 8 dim: 7 dim: 7 }
  }
  variable {
    name: "BatchNormalization_2"
    type: "Buffer"
    shape: { dim:-1 dim: 8 dim: 7 dim: 7 }
  }
  variable {
    name: "PRelu_2"
    type: "Buffer"
    shape: { dim:-1 dim: 8 dim: 7 dim: 7 }
  }
  variable {
    name: "Deconvolution"
    type: "Buffer"
    shape: { dim:-1 dim: 8 dim: 14 dim: 14 }
  }
  variable {
    name: "BatchNormalization_3"
    type: "Buffer"
    shape: { dim:-1 dim: 8 dim: 14 dim: 14 }
  }
  variable {
    name: "PRelu_3"
    type: "Buffer"
    shape: { dim:-1 dim: 8 dim: 14 dim: 14 }
  }
  variable {
    name: "Deconvolution_2"
    type: "Buffer"
    shape: { dim:-1 dim: 1 dim: 28 dim: 28 }
  }
  variable {
    name: "Sigmoid"
    type: "Buffer"
    shape: { dim:-1 dim: 1 dim: 28 dim: 28 }
  }
  variable {
    name: "Output"
    type: "Buffer"
    shape: { dim:-1 dim: 1 dim: 28 dim: 28 }
  }
  variable {
    name: "Convolution/conv/W"
    type: "Parameter"
    shape: { dim: 8 dim: 1 dim: 5 dim: 5 }
    initializer {
      type: "NormalConvolutionGlorot"
      multiplier: 1.0
    }
  }
  variable {
    name: "Convolution/conv/b"
    type: "Parameter"
    shape: { dim: 8 }
    initializer {
      type: "Constant"
      multiplier: 0.0
    }
  }
  variable {
    name: "BatchNormalization/bn/beta"
    type: "Parameter"
    shape: { dim: 1 dim: 8 dim: 1 dim: 1 }
    initializer {
      type: "Constant"
      multiplier: 0.0
    }
  }
  variable {
    name: "BatchNormalization/bn/gamma"
    type: "Parameter"
    shape: { dim: 1 dim: 8 dim: 1 dim: 1 }
    initializer {
      type: "Constant"
      multiplier: 1.0
    }
  }
  variable {
    name: "BatchNormalization/bn/mean"
    type: "Parameter"
    shape: { dim: 1 dim: 8 dim: 1 dim: 1 }
    initializer {
      type: "Constant"
      multiplier: 0.0
    }
  }
  variable {
    name: "BatchNormalization/bn/var"
    type: "Parameter"
    shape: { dim: 1 dim: 8 dim: 1 dim: 1 }
    initializer {
      type: "Constant"
      multiplier: 0.0
    }
  }
  variable {
    name: "PRelu/prelu/slope"
    type: "Parameter"
    shape: { dim: 8 }
    initializer {
      type: "Constant"
      multiplier: 0.25
    }
  }
  variable {
    name: "Convolution_2/conv/W"
    type: "Parameter"
    shape: { dim: 8 dim: 8 dim: 5 dim: 5 }
    initializer {
      type: "NormalConvolutionGlorot"
      multiplier: 1.0
    }
  }
  variable {
    name: "Convolution_2/conv/b"
    type: "Parameter"
    shape: { dim: 8 }
    initializer {
      type: "Constant"
      multiplier: 0.0
    }
  }
  variable {
    name: "BatchNormalization_2/bn/beta"
    type: "Parameter"
    shape: { dim: 1 dim: 8 dim: 1 dim: 1 }
    initializer {
      type: "Constant"
      multiplier: 0.0
    }
  }
  variable {
    name: "BatchNormalization_2/bn/gamma"
    type: "Parameter"
    shape: { dim: 1 dim: 8 dim: 1 dim: 1 }
    initializer {
      type: "Constant"
      multiplier: 1.0
    }
  }
  variable {
    name: "BatchNormalization_2/bn/mean"
    type: "Parameter"
    shape: { dim: 1 dim: 8 dim: 1 dim: 1 }
    initializer {
      type: "Constant"
      multiplier: 0.0
    }
  }
  variable {
    name: "BatchNormalization_2/bn/var"
    type: "Parameter"
    shape: { dim: 1 dim: 8 dim: 1 dim: 1 }
    initializer {
      type: "Constant"
      multiplier: 0.0
    }
  }
  variable {
    name: "PRelu_2/prelu/slope"
    type: "Parameter"
    shape: { dim: 8 }
    initializer {
      type: "Constant"
      multiplier: 0.25
    }
  }
  variable {
    name: "BatchNormalization_3/bn/beta"
    type: "Parameter"
    shape: { dim: 1 dim: 8 dim: 1 dim: 1 }
    initializer {
      type: "Constant"
      multiplier: 0.0
    }
  }
  variable {
    name: "BatchNormalization_3/bn/gamma"
    type: "Parameter"
    shape: { dim: 1 dim: 8 dim: 1 dim: 1 }
    initializer {
      type: "Constant"
      multiplier: 1.0
    }
  }
  variable {
    name: "BatchNormalization_3/bn/mean"
    type: "Parameter"
    shape: { dim: 1 dim: 8 dim: 1 dim: 1 }
    initializer {
      type: "Constant"
      multiplier: 0.0
    }
  }
  variable {
    name: "BatchNormalization_3/bn/var"
    type: "Parameter"
    shape: { dim: 1 dim: 8 dim: 1 dim: 1 }
    initializer {
      type: "Constant"
      multiplier: 0.0
    }
  }
  variable {
    name: "PRelu_3/prelu/slope"
    type: "Parameter"
    shape: { dim: 8 }
    initializer {
      type: "Constant"
      multiplier: 0.25
    }
  }
  variable {
    name: "Output_T"
    type: "Buffer"
    shape: { dim:-1 dim: 1 dim: 28 dim: 28 }
  }
  variable {
    name: "Deconvolution/conv/W"
    type: "Parameter"
    shape: { dim: 8 dim: 8 dim: 6 dim: 6 }
    initializer {
      type: "NormalConvolutionGlorot"
      multiplier: 1.0
    }
  }
  variable {
    name: "Deconvolution/conv/b"
    type: "Parameter"
    shape: { dim: 8 }
    initializer {
      type: "Constant"
      multiplier: 0.0
    }
  }
  variable {
    name: "Deconvolution_2/conv/W"
    type: "Parameter"
    shape: { dim: 8 dim: 1 dim: 6 dim: 6 }
    initializer {
      type: "NormalConvolutionGlorot"
      multiplier: 1.0
    }
  }
  variable {
    name: "Deconvolution_2/conv/b"
    type: "Parameter"
    shape: { dim: 1 }
    initializer {
      type: "Constant"
      multiplier: 0.0
    }
  }
  function {
    name: "Convolution"
    type: "Convolution"
    input: "Input"
    input: "Convolution/conv/W"
    input: "Convolution/conv/b"
    output: "Convolution"
    convolution_param {
      pad: { dim: 2 dim: 2 }
      stride: { dim: 2 dim: 2 }
      dilation: { dim: 1 dim: 1 }
      group: 1
      base_axis: 1
    }
  }
  function {
    name: "BatchNormalization"
    type: "BatchNormalization"
    input: "Convolution"
    input: "BatchNormalization/bn/beta"
    input: "BatchNormalization/bn/gamma"
    input: "BatchNormalization/bn/mean"
    input: "BatchNormalization/bn/var"
    output: "BatchNormalization"
    batch_normalization_param {
      axes: 1
      decay_rate: 0.5
      eps: 0.01
      batch_stat: false
    }
  }
  function {
    name: "PRelu"
    type: "PReLU"
    input: "BatchNormalization"
    input: "PRelu/prelu/slope"
    output: "PRelu"
    prelu_param {
      base_axis: 1
    }
  }
  function {
    name: "Convolution_2"
    type: "Convolution"
    input: "PRelu"
    input: "Convolution_2/conv/W"
    input: "Convolution_2/conv/b"
    output: "Convolution_2"
    convolution_param {
      pad: { dim: 2 dim: 2 }
      stride: { dim: 2 dim: 2 }
      dilation: { dim: 1 dim: 1 }
      group: 1
      base_axis: 1
    }
  }
  function {
    name: "BatchNormalization_2"
    type: "BatchNormalization"
    input: "Convolution_2"
    input: "BatchNormalization_2/bn/beta"
    input: "BatchNormalization_2/bn/gamma"
    input: "BatchNormalization_2/bn/mean"
    input: "BatchNormalization_2/bn/var"
    output: "BatchNormalization_2"
    batch_normalization_param {
      axes: 1
      decay_rate: 0.5
      eps: 0.01
      batch_stat: false
    }
  }
  function {
    name: "PRelu_2"
    type: "PReLU"
    input: "BatchNormalization_2"
    input: "PRelu_2/prelu/slope"
    output: "PRelu_2"
    prelu_param {
      base_axis: 1
    }
  }
  function {
    name: "Deconvolution"
    type: "Deconvolution"
    input: "PRelu_2"
    input: "Deconvolution/conv/W"
    input: "Deconvolution/conv/b"
    output: "Deconvolution"
    deconvolution_param {
      pad: { dim: 2 dim: 2 }
      stride: { dim: 2 dim: 2 }
      dilation: { dim: 1 dim: 1 }
      group: 1
      base_axis: 1
    }
  }
  function {
    name: "BatchNormalization_3"
    type: "BatchNormalization"
    input: "Deconvolution"
    input: "BatchNormalization_3/bn/beta"
    input: "BatchNormalization_3/bn/gamma"
    input: "BatchNormalization_3/bn/mean"
    input: "BatchNormalization_3/bn/var"
    output: "BatchNormalization_3"
    batch_normalization_param {
      axes: 1
      decay_rate: 0.5
      eps: 0.01
      batch_stat: false
    }
  }
  function {
    name: "PRelu_3"
    type: "PReLU"
    input: "BatchNormalization_3"
    input: "PRelu_3/prelu/slope"
    output: "PRelu_3"
    prelu_param {
      base_axis: 1
    }
  }
  function {
    name: "Deconvolution_2"
    type: "Deconvolution"
    input: "PRelu_3"
    input: "Deconvolution_2/conv/W"
    input: "Deconvolution_2/conv/b"
    output: "Deconvolution_2"
    deconvolution_param {
      pad: { dim: 2 dim: 2 }
      stride: { dim: 2 dim: 2 }
      dilation: { dim: 1 dim: 1 }
      group: 1
      base_axis: 1
    }
  }
  function {
    name: "Sigmoid"
    type: "Sigmoid"
    input: "Deconvolution_2"
    output: "Sigmoid"
  }
  function {
    name: "Output"
    type: "SquaredError"
    input: "Sigmoid"
    input: "Output_T"
    output: "Output"
  }
}
network {
  name: "MainRuntime"
  batch_size: 100
  variable {
    name: "Input"
    type: "Buffer"
    shape: { dim:-1 dim: 1 dim: 28 dim: 28 }
  }
  variable {
    name: "Convolution"
    type: "Buffer"
    shape: { dim:-1 dim: 8 dim: 14 dim: 14 }
  }
  variable {
    name: "BatchNormalization"
    type: "Buffer"
    shape: { dim:-1 dim: 8 dim: 14 dim: 14 }
  }
  variable {
    name: "PRelu"
    type: "Buffer"
    shape: { dim:-1 dim: 8 dim: 14 dim: 14 }
  }
  variable {
    name: "Convolution_2"
    type: "Buffer"
    shape: { dim:-1 dim: 8 dim: 7 dim: 7 }
  }
  variable {
    name: "BatchNormalization_2"
    type: "Buffer"
    shape: { dim:-1 dim: 8 dim: 7 dim: 7 }
  }
  variable {
    name: "PRelu_2"
    type: "Buffer"
    shape: { dim:-1 dim: 8 dim: 7 dim: 7 }
  }
  variable {
    name: "Deconvolution"
    type: "Buffer"
    shape: { dim:-1 dim: 8 dim: 14 dim: 14 }
  }
  variable {
    name: "BatchNormalization_3"
    type: "Buffer"
    shape: { dim:-1 dim: 8 dim: 14 dim: 14 }
  }
  variable {
    name: "PRelu_3"
    type: "Buffer"
    shape: { dim:-1 dim: 8 dim: 14 dim: 14 }
  }
  variable {
    name: "Deconvolution_2"
    type: "Buffer"
    shape: { dim:-1 dim: 1 dim: 28 dim: 28 }
  }
  variable {
    name: "Sigmoid"
    type: "Buffer"
    shape: { dim:-1 dim: 1 dim: 28 dim: 28 }
  }
  variable {
    name: "Convolution/conv/W"
    type: "Parameter"
    shape: { dim: 8 dim: 1 dim: 5 dim: 5 }
    initializer {
      type: "NormalConvolutionGlorot"
      multiplier: 1.0
    }
  }
  variable {
    name: "Convolution/conv/b"
    type: "Parameter"
    shape: { dim: 8 }
    initializer {
      type: "Constant"
      multiplier: 0.0
    }
  }
  variable {
    name: "BatchNormalization/bn/beta"
    type: "Parameter"
    shape: { dim: 1 dim: 8 dim: 1 dim: 1 }
    initializer {
      type: "Constant"
      multiplier: 0.0
    }
  }
  variable {
    name: "BatchNormalization/bn/gamma"
    type: "Parameter"
    shape: { dim: 1 dim: 8 dim: 1 dim: 1 }
    initializer {
      type: "Constant"
      multiplier: 1.0
    }
  }
  variable {
    name: "BatchNormalization/bn/mean"
    type: "Parameter"
    shape: { dim: 1 dim: 8 dim: 1 dim: 1 }
    initializer {
      type: "Constant"
      multiplier: 0.0
    }
  }
  variable {
    name: "BatchNormalization/bn/var"
    type: "Parameter"
    shape: { dim: 1 dim: 8 dim: 1 dim: 1 }
    initializer {
      type: "Constant"
      multiplier: 0.0
    }
  }
  variable {
    name: "PRelu/prelu/slope"
    type: "Parameter"
    shape: { dim: 8 }
    initializer {
      type: "Constant"
      multiplier: 0.25
    }
  }
  variable {
    name: "Convolution_2/conv/W"
    type: "Parameter"
    shape: { dim: 8 dim: 8 dim: 5 dim: 5 }
    initializer {
      type: "NormalConvolutionGlorot"
      multiplier: 1.0
    }
  }
  variable {
    name: "Convolution_2/conv/b"
    type: "Parameter"
    shape: { dim: 8 }
    initializer {
      type: "Constant"
      multiplier: 0.0
    }
  }
  variable {
    name: "BatchNormalization_2/bn/beta"
    type: "Parameter"
    shape: { dim: 1 dim: 8 dim: 1 dim: 1 }
    initializer {
      type: "Constant"
      multiplier: 0.0
    }
  }
  variable {
    name: "BatchNormalization_2/bn/gamma"
    type: "Parameter"
    shape: { dim: 1 dim: 8 dim: 1 dim: 1 }
    initializer {
      type: "Constant"
      multiplier: 1.0
    }
  }
  variable {
    name: "BatchNormalization_2/bn/mean"
    type: "Parameter"
    shape: { dim: 1 dim: 8 dim: 1 dim: 1 }
    initializer {
      type: "Constant"
      multiplier: 0.0
    }
  }
  variable {
    name: "BatchNormalization_2/bn/var"
    type: "Parameter"
    shape: { dim: 1 dim: 8 dim: 1 dim: 1 }
    initializer {
      type: "Constant"
      multiplier: 0.0
    }
  }
  variable {
    name: "PRelu_2/prelu/slope"
    type: "Parameter"
    shape: { dim: 8 }
    initializer {
      type: "Constant"
      multiplier: 0.25
    }
  }
  variable {
    name: "BatchNormalization_3/bn/beta"
    type: "Parameter"
    shape: { dim: 1 dim: 8 dim: 1 dim: 1 }
    initializer {
      type: "Constant"
      multiplier: 0.0
    }
  }
  variable {
    name: "BatchNormalization_3/bn/gamma"
    type: "Parameter"
    shape: { dim: 1 dim: 8 dim: 1 dim: 1 }
    initializer {
      type: "Constant"
      multiplier: 1.0
    }
  }
  variable {
    name: "BatchNormalization_3/bn/mean"
    type: "Parameter"
    shape: { dim: 1 dim: 8 dim: 1 dim: 1 }
    initializer {
      type: "Constant"
      multiplier: 0.0
    }
  }
  variable {
    name: "BatchNormalization_3/bn/var"
    type: "Parameter"
    shape: { dim: 1 dim: 8 dim: 1 dim: 1 }
    initializer {
      type: "Constant"
      multiplier: 0.0
    }
  }
  variable {
    name: "PRelu_3/prelu/slope"
    type: "Parameter"
    shape: { dim: 8 }
    initializer {
      type: "Constant"
      multiplier: 0.25
    }
  }
  variable {
    name: "Deconvolution/conv/W"
    type: "Parameter"
    shape: { dim: 8 dim: 8 dim: 6 dim: 6 }
    initializer {
      type: "NormalConvolutionGlorot"
      multiplier: 1.0
    }
  }
  variable {
    name: "Deconvolution/conv/b"
    type: "Parameter"
    shape: { dim: 8 }
    initializer {
      type: "Constant"
      multiplier: 0.0
    }
  }
  variable {
    name: "Deconvolution_2/conv/W"
    type: "Parameter"
    shape: { dim: 8 dim: 1 dim: 6 dim: 6 }
    initializer {
      type: "NormalConvolutionGlorot"
      multiplier: 1.0
    }
  }
  variable {
    name: "Deconvolution_2/conv/b"
    type: "Parameter"
    shape: { dim: 1 }
    initializer {
      type: "Constant"
      multiplier: 0.0
    }
  }
  function {
    name: "Convolution"
    type: "Convolution"
    input: "Input"
    input: "Convolution/conv/W"
    input: "Convolution/conv/b"
    output: "Convolution"
    convolution_param {
      pad: { dim: 2 dim: 2 }
      stride: { dim: 2 dim: 2 }
      dilation: { dim: 1 dim: 1 }
      group: 1
      base_axis: 1
    }
  }
  function {
    name: "BatchNormalization"
    type: "BatchNormalization"
    input: "Convolution"
    input: "BatchNormalization/bn/beta"
    input: "BatchNormalization/bn/gamma"
    input: "BatchNormalization/bn/mean"
    input: "BatchNormalization/bn/var"
    output: "BatchNormalization"
    batch_normalization_param {
      axes: 1
      decay_rate: 0.5
      eps: 0.01
      batch_stat: false
    }
  }
  function {
    name: "PRelu"
    type: "PReLU"
    input: "BatchNormalization"
    input: "PRelu/prelu/slope"
    output: "PRelu"
    prelu_param {
      base_axis: 1
    }
  }
  function {
    name: "Convolution_2"
    type: "Convolution"
    input: "PRelu"
    input: "Convolution_2/conv/W"
    input: "Convolution_2/conv/b"
    output: "Convolution_2"
    convolution_param {
      pad: { dim: 2 dim: 2 }
      stride: { dim: 2 dim: 2 }
      dilation: { dim: 1 dim: 1 }
      group: 1
      base_axis: 1
    }
  }
  function {
    name: "BatchNormalization_2"
    type: "BatchNormalization"
    input: "Convolution_2"
    input: "BatchNormalization_2/bn/beta"
    input: "BatchNormalization_2/bn/gamma"
    input: "BatchNormalization_2/bn/mean"
    input: "BatchNormalization_2/bn/var"
    output: "BatchNormalization_2"
    batch_normalization_param {
      axes: 1
      decay_rate: 0.5
      eps: 0.01
      batch_stat: false
    }
  }
  function {
    name: "PRelu_2"
    type: "PReLU"
    input: "BatchNormalization_2"
    input: "PRelu_2/prelu/slope"
    output: "PRelu_2"
    prelu_param {
      base_axis: 1
    }
  }
  function {
    name: "Deconvolution"
    type: "Deconvolution"
    input: "PRelu_2"
    input: "Deconvolution/conv/W"
    input: "Deconvolution/conv/b"
    output: "Deconvolution"
    deconvolution_param {
      pad: { dim: 2 dim: 2 }
      stride: { dim: 2 dim: 2 }
      dilation: { dim: 1 dim: 1 }
      group: 1
      base_axis: 1
    }
  }
  function {
    name: "BatchNormalization_3"
    type: "BatchNormalization"
    input: "Deconvolution"
    input: "BatchNormalization_3/bn/beta"
    input: "BatchNormalization_3/bn/gamma"
    input: "BatchNormalization_3/bn/mean"
    input: "BatchNormalization_3/bn/var"
    output: "BatchNormalization_3"
    batch_normalization_param {
      axes: 1
      decay_rate: 0.5
      eps: 0.01
      batch_stat: false
    }
  }
  function {
    name: "PRelu_3"
    type: "PReLU"
    input: "BatchNormalization_3"
    input: "PRelu_3/prelu/slope"
    output: "PRelu_3"
    prelu_param {
      base_axis: 1
    }
  }
  function {
    name: "Deconvolution_2"
    type: "Deconvolution"
    input: "PRelu_3"
    input: "Deconvolution_2/conv/W"
    input: "Deconvolution_2/conv/b"
    output: "Deconvolution_2"
    deconvolution_param {
      pad: { dim: 2 dim: 2 }
      stride: { dim: 2 dim: 2 }
      dilation: { dim: 1 dim: 1 }
      group: 1
      base_axis: 1
    }
  }
  function {
    name: "Sigmoid"
    type: "Sigmoid"
    input: "Deconvolution_2"
    output: "Sigmoid"
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
  batch_size: 100
}
dataset {
  name: "Validation"
  uri: "K:\\neural_network_console_170725\\samples\\sample_dataset\\mnist\\small_mnist_4or9_test.csv"
  cache_dir: "K:\\neural_network_console_170725\\samples\\sample_dataset\\mnist\\small_mnist_4or9_test.cache"
  overwrite_cache: False
  create_cache_explicitly: True
  shuffle: false
  no_image_normalization: False
  batch_size: 100
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
    variable_name: "Output_T"
    data_name: "x"
  }
  loss_variable {
    variable_name: "Output"
  }
  parameter_variable {
    variable_name: "Convolution/conv/W"
    learning_rate_multiplier: 1.0
  }
  parameter_variable {
    variable_name: "Convolution/conv/b"
    learning_rate_multiplier: 1.0
  }
  parameter_variable {
    variable_name: "BatchNormalization/bn/beta"
    learning_rate_multiplier: 1.0
  }
  parameter_variable {
    variable_name: "BatchNormalization/bn/gamma"
    learning_rate_multiplier: 1.0
  }
  parameter_variable {
    variable_name: "BatchNormalization/bn/mean"
    learning_rate_multiplier: 0.0
  }
  parameter_variable {
    variable_name: "BatchNormalization/bn/var"
    learning_rate_multiplier: 0.0
  }
  parameter_variable {
    variable_name: "PRelu/prelu/slope"
    learning_rate_multiplier: 1.0
  }
  parameter_variable {
    variable_name: "Convolution_2/conv/W"
    learning_rate_multiplier: 1.0
  }
  parameter_variable {
    variable_name: "Convolution_2/conv/b"
    learning_rate_multiplier: 1.0
  }
  parameter_variable {
    variable_name: "BatchNormalization_2/bn/beta"
    learning_rate_multiplier: 1.0
  }
  parameter_variable {
    variable_name: "BatchNormalization_2/bn/gamma"
    learning_rate_multiplier: 1.0
  }
  parameter_variable {
    variable_name: "BatchNormalization_2/bn/mean"
    learning_rate_multiplier: 0.0
  }
  parameter_variable {
    variable_name: "BatchNormalization_2/bn/var"
    learning_rate_multiplier: 0.0
  }
  parameter_variable {
    variable_name: "PRelu_2/prelu/slope"
    learning_rate_multiplier: 1.0
  }
  parameter_variable {
    variable_name: "BatchNormalization_3/bn/beta"
    learning_rate_multiplier: 1.0
  }
  parameter_variable {
    variable_name: "BatchNormalization_3/bn/gamma"
    learning_rate_multiplier: 1.0
  }
  parameter_variable {
    variable_name: "BatchNormalization_3/bn/mean"
    learning_rate_multiplier: 0.0
  }
  parameter_variable {
    variable_name: "BatchNormalization_3/bn/var"
    learning_rate_multiplier: 0.0
  }
  parameter_variable {
    variable_name: "PRelu_3/prelu/slope"
    learning_rate_multiplier: 1.0
  }
  parameter_variable {
    variable_name: "Deconvolution/conv/W"
    learning_rate_multiplier: 1.0
  }
  parameter_variable {
    variable_name: "Deconvolution/conv/b"
    learning_rate_multiplier: 1.0
  }
  parameter_variable {
    variable_name: "Deconvolution_2/conv/W"
    learning_rate_multiplier: 1.0
  }
  parameter_variable {
    variable_name: "Deconvolution_2/conv/b"
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
  data_variable {
    variable_name: "Output_T"
    data_name: "x"
  }
  monitor_variable {
    type: "Error"
    variable_name: "Output"
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
    variable_name: "Output_T"
    data_name: "x"
  }
  monitor_variable {
    type: "Error"
    variable_name: "Output"
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
    data_name: "x'"
  }
  parameter_variable {
    variable_name: "Convolution/conv/W"
  }
  parameter_variable {
    variable_name: "Convolution/conv/b"
  }
  parameter_variable {
    variable_name: "BatchNormalization/bn/beta"
  }
  parameter_variable {
    variable_name: "BatchNormalization/bn/gamma"
  }
  parameter_variable {
    variable_name: "BatchNormalization/bn/mean"
  }
  parameter_variable {
    variable_name: "BatchNormalization/bn/var"
  }
  parameter_variable {
    variable_name: "PRelu/prelu/slope"
  }
  parameter_variable {
    variable_name: "Convolution_2/conv/W"
  }
  parameter_variable {
    variable_name: "Convolution_2/conv/b"
  }
  parameter_variable {
    variable_name: "BatchNormalization_2/bn/beta"
  }
  parameter_variable {
    variable_name: "BatchNormalization_2/bn/gamma"
  }
  parameter_variable {
    variable_name: "BatchNormalization_2/bn/mean"
  }
  parameter_variable {
    variable_name: "BatchNormalization_2/bn/var"
  }
  parameter_variable {
    variable_name: "PRelu_2/prelu/slope"
  }
  parameter_variable {
    variable_name: "BatchNormalization_3/bn/beta"
  }
  parameter_variable {
    variable_name: "BatchNormalization_3/bn/gamma"
  }
  parameter_variable {
    variable_name: "BatchNormalization_3/bn/mean"
  }
  parameter_variable {
    variable_name: "BatchNormalization_3/bn/var"
  }
  parameter_variable {
    variable_name: "PRelu_3/prelu/slope"
  }
  parameter_variable {
    variable_name: "Deconvolution/conv/W"
  }
  parameter_variable {
    variable_name: "Deconvolution/conv/b"
  }
  parameter_variable {
    variable_name: "Deconvolution_2/conv/W"
  }
  parameter_variable {
    variable_name: "Deconvolution_2/conv/b"
  }
}
'''
# The nntxt is generated from /home/woody/aid/nnabla-sample-data/sample_project/tutorial/basics/11_deconvolution.files/20180720_182001/net.nntxt.
N00000024 = r'''global_config {
  default_context {
    array_class: "CudaCachedArray"
    device_id: "0"
    backends: "cudnn:float"
    backends: "cuda:float"
    backends: "cpu:float"
  }
}
training_config {
  max_epoch: 500
  iter_per_epoch: 15
  save_best: true
  monitor_interval: 10
}
network {
  name: "Main"
  batch_size: 100
  variable {
    name: "Input"
    type: "Buffer"
    shape: { dim:-1 dim: 1 dim: 28 dim: 28 }
  }
  variable {
    name: "Convolution/conv/W"
    type: "Parameter"
    shape: { dim: 8 dim: 1 dim: 5 dim: 5 }
    initializer {
      type: "NormalConvolutionGlorot"
      multiplier: 1
    }
  }
  variable {
    name: "Convolution/conv/b"
    type: "Parameter"
    shape: { dim: 8 }
    initializer {
      type: "Constant"
      multiplier: 0
    }
  }
  variable {
    name: "BatchNormalization/bn/beta"
    type: "Parameter"
    shape: { dim: 1 dim: 8 dim: 1 dim: 1 }
    initializer {
      type: "Constant"
      multiplier: 0
    }
  }
  variable {
    name: "BatchNormalization/bn/gamma"
    type: "Parameter"
    shape: { dim: 1 dim: 8 dim: 1 dim: 1 }
    initializer {
      type: "Constant"
      multiplier: 1
    }
  }
  variable {
    name: "BatchNormalization/bn/mean"
    type: "Parameter"
    shape: { dim: 1 dim: 8 dim: 1 dim: 1 }
    initializer {
      type: "Constant"
      multiplier: 0
    }
  }
  variable {
    name: "BatchNormalization/bn/var"
    type: "Parameter"
    shape: { dim: 1 dim: 8 dim: 1 dim: 1 }
    initializer {
      type: "Constant"
      multiplier: 0
    }
  }
  variable {
    name: "PRelu/prelu/slope"
    type: "Parameter"
    shape: { dim: 8 }
    initializer {
      type: "Constant"
      multiplier: 0.25
    }
  }
  variable {
    name: "Convolution_2/conv/W"
    type: "Parameter"
    shape: { dim: 8 dim: 8 dim: 5 dim: 5 }
    initializer {
      type: "NormalConvolutionGlorot"
      multiplier: 1
    }
  }
  variable {
    name: "Convolution_2/conv/b"
    type: "Parameter"
    shape: { dim: 8 }
    initializer {
      type: "Constant"
      multiplier: 0
    }
  }
  variable {
    name: "BatchNormalization_2/bn/beta"
    type: "Parameter"
    shape: { dim: 1 dim: 8 dim: 1 dim: 1 }
    initializer {
      type: "Constant"
      multiplier: 0
    }
  }
  variable {
    name: "BatchNormalization_2/bn/gamma"
    type: "Parameter"
    shape: { dim: 1 dim: 8 dim: 1 dim: 1 }
    initializer {
      type: "Constant"
      multiplier: 1
    }
  }
  variable {
    name: "BatchNormalization_2/bn/mean"
    type: "Parameter"
    shape: { dim: 1 dim: 8 dim: 1 dim: 1 }
    initializer {
      type: "Constant"
      multiplier: 0
    }
  }
  variable {
    name: "BatchNormalization_2/bn/var"
    type: "Parameter"
    shape: { dim: 1 dim: 8 dim: 1 dim: 1 }
    initializer {
      type: "Constant"
      multiplier: 0
    }
  }
  variable {
    name: "PRelu_2/prelu/slope"
    type: "Parameter"
    shape: { dim: 8 }
    initializer {
      type: "Constant"
      multiplier: 0.25
    }
  }
  variable {
    name: "Deconvolution/deconv/W"
    type: "Parameter"
    shape: { dim: 8 dim: 8 dim: 6 dim: 6 }
    initializer {
      type: "NormalConvolutionGlorot"
      multiplier: 1
    }
  }
  variable {
    name: "Deconvolution/deconv/b"
    type: "Parameter"
    shape: { dim: 8 }
    initializer {
      type: "Constant"
      multiplier: 0
    }
  }
  variable {
    name: "BatchNormalization_3/bn/beta"
    type: "Parameter"
    shape: { dim: 1 dim: 8 dim: 1 dim: 1 }
    initializer {
      type: "Constant"
      multiplier: 0
    }
  }
  variable {
    name: "BatchNormalization_3/bn/gamma"
    type: "Parameter"
    shape: { dim: 1 dim: 8 dim: 1 dim: 1 }
    initializer {
      type: "Constant"
      multiplier: 1
    }
  }
  variable {
    name: "BatchNormalization_3/bn/mean"
    type: "Parameter"
    shape: { dim: 1 dim: 8 dim: 1 dim: 1 }
    initializer {
      type: "Constant"
      multiplier: 0
    }
  }
  variable {
    name: "BatchNormalization_3/bn/var"
    type: "Parameter"
    shape: { dim: 1 dim: 8 dim: 1 dim: 1 }
    initializer {
      type: "Constant"
      multiplier: 0
    }
  }
  variable {
    name: "PRelu_3/prelu/slope"
    type: "Parameter"
    shape: { dim: 8 }
    initializer {
      type: "Constant"
      multiplier: 0.25
    }
  }
  variable {
    name: "Deconvolution_2/deconv/W"
    type: "Parameter"
    shape: { dim: 8 dim: 1 dim: 6 dim: 6 }
    initializer {
      type: "NormalConvolutionGlorot"
      multiplier: 1
    }
  }
  variable {
    name: "Deconvolution_2/deconv/b"
    type: "Parameter"
    shape: { dim: 1 }
    initializer {
      type: "Constant"
      multiplier: 0
    }
  }
  variable {
    name: "Output_T"
    type: "Buffer"
    shape: { dim:-1 dim: 1 dim: 28 dim: 28 }
  }
  variable {
    name: "Convolution"
    type: "Buffer"
    shape: { dim:-1 dim: 8 dim: 14 dim: 14 }
  }
  variable {
    name: "BatchNormalization"
    type: "Buffer"
    shape: { dim:-1 dim: 8 dim: 14 dim: 14 }
  }
  variable {
    name: "PRelu"
    type: "Buffer"
    shape: { dim:-1 dim: 8 dim: 14 dim: 14 }
  }
  variable {
    name: "Convolution_2"
    type: "Buffer"
    shape: { dim:-1 dim: 8 dim: 7 dim: 7 }
  }
  variable {
    name: "BatchNormalization_2"
    type: "Buffer"
    shape: { dim:-1 dim: 8 dim: 7 dim: 7 }
  }
  variable {
    name: "PRelu_2"
    type: "Buffer"
    shape: { dim:-1 dim: 8 dim: 7 dim: 7 }
  }
  variable {
    name: "Deconvolution"
    type: "Buffer"
    shape: { dim:-1 dim: 8 dim: 14 dim: 14 }
  }
  variable {
    name: "BatchNormalization_3"
    type: "Buffer"
    shape: { dim:-1 dim: 8 dim: 14 dim: 14 }
  }
  variable {
    name: "PRelu_3"
    type: "Buffer"
    shape: { dim:-1 dim: 8 dim: 14 dim: 14 }
  }
  variable {
    name: "Deconvolution_2"
    type: "Buffer"
    shape: { dim:-1 dim: 1 dim: 28 dim: 28 }
  }
  variable {
    name: "Sigmoid"
    type: "Buffer"
    shape: { dim:-1 dim: 1 dim: 28 dim: 28 }
  }
  variable {
    name: "Output"
    type: "Buffer"
    shape: { dim:-1 dim: 1 dim: 28 dim: 28 }
  }
  function {
    name: "Convolution"
    type: "Convolution"
    input: "Input"
    input: "Convolution/conv/W"
    input: "Convolution/conv/b"
    output: "Convolution"
    convolution_param {
      pad: { dim: 2 dim: 2 }
      stride: { dim: 2 dim: 2 }
      dilation: { dim: 1 dim: 1 }
      group: 1
      base_axis: 1
    }
  }
  function {
    name: "BatchNormalization"
    type: "BatchNormalization"
    input: "Convolution"
    input: "BatchNormalization/bn/beta"
    input: "BatchNormalization/bn/gamma"
    input: "BatchNormalization/bn/mean"
    input: "BatchNormalization/bn/var"
    output: "BatchNormalization"
    batch_normalization_param {
      axes: 1
      decay_rate: 0.5
      eps: 0.01
      batch_stat: True
    }
  }
  function {
    name: "PRelu"
    type: "PReLU"
    input: "BatchNormalization"
    input: "PRelu/prelu/slope"
    output: "PRelu"
    prelu_param {
      base_axis: 1
    }
  }
  function {
    name: "Convolution_2"
    type: "Convolution"
    input: "PRelu"
    input: "Convolution_2/conv/W"
    input: "Convolution_2/conv/b"
    output: "Convolution_2"
    convolution_param {
      pad: { dim: 2 dim: 2 }
      stride: { dim: 2 dim: 2 }
      dilation: { dim: 1 dim: 1 }
      group: 1
      base_axis: 1
    }
  }
  function {
    name: "BatchNormalization_2"
    type: "BatchNormalization"
    input: "Convolution_2"
    input: "BatchNormalization_2/bn/beta"
    input: "BatchNormalization_2/bn/gamma"
    input: "BatchNormalization_2/bn/mean"
    input: "BatchNormalization_2/bn/var"
    output: "BatchNormalization_2"
    batch_normalization_param {
      axes: 1
      decay_rate: 0.5
      eps: 0.01
      batch_stat: True
    }
  }
  function {
    name: "PRelu_2"
    type: "PReLU"
    input: "BatchNormalization_2"
    input: "PRelu_2/prelu/slope"
    output: "PRelu_2"
    prelu_param {
      base_axis: 1
    }
  }
  function {
    name: "Deconvolution"
    type: "Deconvolution"
    input: "PRelu_2"
    input: "Deconvolution/deconv/W"
    input: "Deconvolution/deconv/b"
    output: "Deconvolution"
    deconvolution_param {
      pad: { dim: 2 dim: 2 }
      stride: { dim: 2 dim: 2 }
      dilation: { dim: 1 dim: 1 }
      group: 1
      base_axis: 1
    }
  }
  function {
    name: "BatchNormalization_3"
    type: "BatchNormalization"
    input: "Deconvolution"
    input: "BatchNormalization_3/bn/beta"
    input: "BatchNormalization_3/bn/gamma"
    input: "BatchNormalization_3/bn/mean"
    input: "BatchNormalization_3/bn/var"
    output: "BatchNormalization_3"
    batch_normalization_param {
      axes: 1
      decay_rate: 0.5
      eps: 0.01
      batch_stat: True
    }
  }
  function {
    name: "PRelu_3"
    type: "PReLU"
    input: "BatchNormalization_3"
    input: "PRelu_3/prelu/slope"
    output: "PRelu_3"
    prelu_param {
      base_axis: 1
    }
  }
  function {
    name: "Deconvolution_2"
    type: "Deconvolution"
    input: "PRelu_3"
    input: "Deconvolution_2/deconv/W"
    input: "Deconvolution_2/deconv/b"
    output: "Deconvolution_2"
    deconvolution_param {
      pad: { dim: 2 dim: 2 }
      stride: { dim: 2 dim: 2 }
      dilation: { dim: 1 dim: 1 }
      group: 1
      base_axis: 1
    }
  }
  function {
    name: "Sigmoid"
    type: "Sigmoid"
    input: "Deconvolution_2"
    output: "Sigmoid"
  }
  function {
    name: "Output"
    type: "SquaredError"
    input: "Sigmoid"
    input: "Output_T"
    output: "Output"
  }
}
network {
  name: "MainValidation"
  batch_size: 100
  variable {
    name: "Input"
    type: "Buffer"
    shape: { dim:-1 dim: 1 dim: 28 dim: 28 }
  }
  variable {
    name: "Convolution/conv/W"
    type: "Parameter"
    shape: { dim: 8 dim: 1 dim: 5 dim: 5 }
    initializer {
      type: "NormalConvolutionGlorot"
      multiplier: 1
    }
  }
  variable {
    name: "Convolution/conv/b"
    type: "Parameter"
    shape: { dim: 8 }
    initializer {
      type: "Constant"
      multiplier: 0
    }
  }
  variable {
    name: "BatchNormalization/bn/beta"
    type: "Parameter"
    shape: { dim: 1 dim: 8 dim: 1 dim: 1 }
    initializer {
      type: "Constant"
      multiplier: 0
    }
  }
  variable {
    name: "BatchNormalization/bn/gamma"
    type: "Parameter"
    shape: { dim: 1 dim: 8 dim: 1 dim: 1 }
    initializer {
      type: "Constant"
      multiplier: 1
    }
  }
  variable {
    name: "BatchNormalization/bn/mean"
    type: "Parameter"
    shape: { dim: 1 dim: 8 dim: 1 dim: 1 }
    initializer {
      type: "Constant"
      multiplier: 0
    }
  }
  variable {
    name: "BatchNormalization/bn/var"
    type: "Parameter"
    shape: { dim: 1 dim: 8 dim: 1 dim: 1 }
    initializer {
      type: "Constant"
      multiplier: 0
    }
  }
  variable {
    name: "PRelu/prelu/slope"
    type: "Parameter"
    shape: { dim: 8 }
    initializer {
      type: "Constant"
      multiplier: 0.25
    }
  }
  variable {
    name: "Convolution_2/conv/W"
    type: "Parameter"
    shape: { dim: 8 dim: 8 dim: 5 dim: 5 }
    initializer {
      type: "NormalConvolutionGlorot"
      multiplier: 1
    }
  }
  variable {
    name: "Convolution_2/conv/b"
    type: "Parameter"
    shape: { dim: 8 }
    initializer {
      type: "Constant"
      multiplier: 0
    }
  }
  variable {
    name: "BatchNormalization_2/bn/beta"
    type: "Parameter"
    shape: { dim: 1 dim: 8 dim: 1 dim: 1 }
    initializer {
      type: "Constant"
      multiplier: 0
    }
  }
  variable {
    name: "BatchNormalization_2/bn/gamma"
    type: "Parameter"
    shape: { dim: 1 dim: 8 dim: 1 dim: 1 }
    initializer {
      type: "Constant"
      multiplier: 1
    }
  }
  variable {
    name: "BatchNormalization_2/bn/mean"
    type: "Parameter"
    shape: { dim: 1 dim: 8 dim: 1 dim: 1 }
    initializer {
      type: "Constant"
      multiplier: 0
    }
  }
  variable {
    name: "BatchNormalization_2/bn/var"
    type: "Parameter"
    shape: { dim: 1 dim: 8 dim: 1 dim: 1 }
    initializer {
      type: "Constant"
      multiplier: 0
    }
  }
  variable {
    name: "PRelu_2/prelu/slope"
    type: "Parameter"
    shape: { dim: 8 }
    initializer {
      type: "Constant"
      multiplier: 0.25
    }
  }
  variable {
    name: "Deconvolution/deconv/W"
    type: "Parameter"
    shape: { dim: 8 dim: 8 dim: 6 dim: 6 }
    initializer {
      type: "NormalConvolutionGlorot"
      multiplier: 1
    }
  }
  variable {
    name: "Deconvolution/deconv/b"
    type: "Parameter"
    shape: { dim: 8 }
    initializer {
      type: "Constant"
      multiplier: 0
    }
  }
  variable {
    name: "BatchNormalization_3/bn/beta"
    type: "Parameter"
    shape: { dim: 1 dim: 8 dim: 1 dim: 1 }
    initializer {
      type: "Constant"
      multiplier: 0
    }
  }
  variable {
    name: "BatchNormalization_3/bn/gamma"
    type: "Parameter"
    shape: { dim: 1 dim: 8 dim: 1 dim: 1 }
    initializer {
      type: "Constant"
      multiplier: 1
    }
  }
  variable {
    name: "BatchNormalization_3/bn/mean"
    type: "Parameter"
    shape: { dim: 1 dim: 8 dim: 1 dim: 1 }
    initializer {
      type: "Constant"
      multiplier: 0
    }
  }
  variable {
    name: "BatchNormalization_3/bn/var"
    type: "Parameter"
    shape: { dim: 1 dim: 8 dim: 1 dim: 1 }
    initializer {
      type: "Constant"
      multiplier: 0
    }
  }
  variable {
    name: "PRelu_3/prelu/slope"
    type: "Parameter"
    shape: { dim: 8 }
    initializer {
      type: "Constant"
      multiplier: 0.25
    }
  }
  variable {
    name: "Deconvolution_2/deconv/W"
    type: "Parameter"
    shape: { dim: 8 dim: 1 dim: 6 dim: 6 }
    initializer {
      type: "NormalConvolutionGlorot"
      multiplier: 1
    }
  }
  variable {
    name: "Deconvolution_2/deconv/b"
    type: "Parameter"
    shape: { dim: 1 }
    initializer {
      type: "Constant"
      multiplier: 0
    }
  }
  variable {
    name: "Output_T"
    type: "Buffer"
    shape: { dim:-1 dim: 1 dim: 28 dim: 28 }
  }
  variable {
    name: "Convolution"
    type: "Buffer"
    shape: { dim:-1 dim: 8 dim: 14 dim: 14 }
  }
  variable {
    name: "BatchNormalization"
    type: "Buffer"
    shape: { dim:-1 dim: 8 dim: 14 dim: 14 }
  }
  variable {
    name: "PRelu"
    type: "Buffer"
    shape: { dim:-1 dim: 8 dim: 14 dim: 14 }
  }
  variable {
    name: "Convolution_2"
    type: "Buffer"
    shape: { dim:-1 dim: 8 dim: 7 dim: 7 }
  }
  variable {
    name: "BatchNormalization_2"
    type: "Buffer"
    shape: { dim:-1 dim: 8 dim: 7 dim: 7 }
  }
  variable {
    name: "PRelu_2"
    type: "Buffer"
    shape: { dim:-1 dim: 8 dim: 7 dim: 7 }
  }
  variable {
    name: "Deconvolution"
    type: "Buffer"
    shape: { dim:-1 dim: 8 dim: 14 dim: 14 }
  }
  variable {
    name: "BatchNormalization_3"
    type: "Buffer"
    shape: { dim:-1 dim: 8 dim: 14 dim: 14 }
  }
  variable {
    name: "PRelu_3"
    type: "Buffer"
    shape: { dim:-1 dim: 8 dim: 14 dim: 14 }
  }
  variable {
    name: "Deconvolution_2"
    type: "Buffer"
    shape: { dim:-1 dim: 1 dim: 28 dim: 28 }
  }
  variable {
    name: "Sigmoid"
    type: "Buffer"
    shape: { dim:-1 dim: 1 dim: 28 dim: 28 }
  }
  variable {
    name: "Output"
    type: "Buffer"
    shape: { dim:-1 dim: 1 dim: 28 dim: 28 }
  }
  function {
    name: "Convolution"
    type: "Convolution"
    input: "Input"
    input: "Convolution/conv/W"
    input: "Convolution/conv/b"
    output: "Convolution"
    convolution_param {
      pad: { dim: 2 dim: 2 }
      stride: { dim: 2 dim: 2 }
      dilation: { dim: 1 dim: 1 }
      group: 1
      base_axis: 1
    }
  }
  function {
    name: "BatchNormalization"
    type: "BatchNormalization"
    input: "Convolution"
    input: "BatchNormalization/bn/beta"
    input: "BatchNormalization/bn/gamma"
    input: "BatchNormalization/bn/mean"
    input: "BatchNormalization/bn/var"
    output: "BatchNormalization"
    batch_normalization_param {
      axes: 1
      decay_rate: 0.5
      eps: 0.01
      batch_stat: False
    }
  }
  function {
    name: "PRelu"
    type: "PReLU"
    input: "BatchNormalization"
    input: "PRelu/prelu/slope"
    output: "PRelu"
    prelu_param {
      base_axis: 1
    }
  }
  function {
    name: "Convolution_2"
    type: "Convolution"
    input: "PRelu"
    input: "Convolution_2/conv/W"
    input: "Convolution_2/conv/b"
    output: "Convolution_2"
    convolution_param {
      pad: { dim: 2 dim: 2 }
      stride: { dim: 2 dim: 2 }
      dilation: { dim: 1 dim: 1 }
      group: 1
      base_axis: 1
    }
  }
  function {
    name: "BatchNormalization_2"
    type: "BatchNormalization"
    input: "Convolution_2"
    input: "BatchNormalization_2/bn/beta"
    input: "BatchNormalization_2/bn/gamma"
    input: "BatchNormalization_2/bn/mean"
    input: "BatchNormalization_2/bn/var"
    output: "BatchNormalization_2"
    batch_normalization_param {
      axes: 1
      decay_rate: 0.5
      eps: 0.01
      batch_stat: False
    }
  }
  function {
    name: "PRelu_2"
    type: "PReLU"
    input: "BatchNormalization_2"
    input: "PRelu_2/prelu/slope"
    output: "PRelu_2"
    prelu_param {
      base_axis: 1
    }
  }
  function {
    name: "Deconvolution"
    type: "Deconvolution"
    input: "PRelu_2"
    input: "Deconvolution/deconv/W"
    input: "Deconvolution/deconv/b"
    output: "Deconvolution"
    deconvolution_param {
      pad: { dim: 2 dim: 2 }
      stride: { dim: 2 dim: 2 }
      dilation: { dim: 1 dim: 1 }
      group: 1
      base_axis: 1
    }
  }
  function {
    name: "BatchNormalization_3"
    type: "BatchNormalization"
    input: "Deconvolution"
    input: "BatchNormalization_3/bn/beta"
    input: "BatchNormalization_3/bn/gamma"
    input: "BatchNormalization_3/bn/mean"
    input: "BatchNormalization_3/bn/var"
    output: "BatchNormalization_3"
    batch_normalization_param {
      axes: 1
      decay_rate: 0.5
      eps: 0.01
      batch_stat: False
    }
  }
  function {
    name: "PRelu_3"
    type: "PReLU"
    input: "BatchNormalization_3"
    input: "PRelu_3/prelu/slope"
    output: "PRelu_3"
    prelu_param {
      base_axis: 1
    }
  }
  function {
    name: "Deconvolution_2"
    type: "Deconvolution"
    input: "PRelu_3"
    input: "Deconvolution_2/deconv/W"
    input: "Deconvolution_2/deconv/b"
    output: "Deconvolution_2"
    deconvolution_param {
      pad: { dim: 2 dim: 2 }
      stride: { dim: 2 dim: 2 }
      dilation: { dim: 1 dim: 1 }
      group: 1
      base_axis: 1
    }
  }
  function {
    name: "Sigmoid"
    type: "Sigmoid"
    input: "Deconvolution_2"
    output: "Sigmoid"
  }
  function {
    name: "Output"
    type: "SquaredError"
    input: "Sigmoid"
    input: "Output_T"
    output: "Output"
  }
}
network {
  name: "MainRuntime"
  batch_size: 100
  variable {
    name: "Input"
    type: "Buffer"
    shape: { dim:-1 dim: 1 dim: 28 dim: 28 }
  }
  variable {
    name: "Convolution/conv/W"
    type: "Parameter"
    shape: { dim: 8 dim: 1 dim: 5 dim: 5 }
    initializer {
      type: "NormalConvolutionGlorot"
      multiplier: 1
    }
  }
  variable {
    name: "Convolution/conv/b"
    type: "Parameter"
    shape: { dim: 8 }
    initializer {
      type: "Constant"
      multiplier: 0
    }
  }
  variable {
    name: "BatchNormalization/bn/beta"
    type: "Parameter"
    shape: { dim: 1 dim: 8 dim: 1 dim: 1 }
    initializer {
      type: "Constant"
      multiplier: 0
    }
  }
  variable {
    name: "BatchNormalization/bn/gamma"
    type: "Parameter"
    shape: { dim: 1 dim: 8 dim: 1 dim: 1 }
    initializer {
      type: "Constant"
      multiplier: 1
    }
  }
  variable {
    name: "BatchNormalization/bn/mean"
    type: "Parameter"
    shape: { dim: 1 dim: 8 dim: 1 dim: 1 }
    initializer {
      type: "Constant"
      multiplier: 0
    }
  }
  variable {
    name: "BatchNormalization/bn/var"
    type: "Parameter"
    shape: { dim: 1 dim: 8 dim: 1 dim: 1 }
    initializer {
      type: "Constant"
      multiplier: 0
    }
  }
  variable {
    name: "PRelu/prelu/slope"
    type: "Parameter"
    shape: { dim: 8 }
    initializer {
      type: "Constant"
      multiplier: 0.25
    }
  }
  variable {
    name: "Convolution_2/conv/W"
    type: "Parameter"
    shape: { dim: 8 dim: 8 dim: 5 dim: 5 }
    initializer {
      type: "NormalConvolutionGlorot"
      multiplier: 1
    }
  }
  variable {
    name: "Convolution_2/conv/b"
    type: "Parameter"
    shape: { dim: 8 }
    initializer {
      type: "Constant"
      multiplier: 0
    }
  }
  variable {
    name: "BatchNormalization_2/bn/beta"
    type: "Parameter"
    shape: { dim: 1 dim: 8 dim: 1 dim: 1 }
    initializer {
      type: "Constant"
      multiplier: 0
    }
  }
  variable {
    name: "BatchNormalization_2/bn/gamma"
    type: "Parameter"
    shape: { dim: 1 dim: 8 dim: 1 dim: 1 }
    initializer {
      type: "Constant"
      multiplier: 1
    }
  }
  variable {
    name: "BatchNormalization_2/bn/mean"
    type: "Parameter"
    shape: { dim: 1 dim: 8 dim: 1 dim: 1 }
    initializer {
      type: "Constant"
      multiplier: 0
    }
  }
  variable {
    name: "BatchNormalization_2/bn/var"
    type: "Parameter"
    shape: { dim: 1 dim: 8 dim: 1 dim: 1 }
    initializer {
      type: "Constant"
      multiplier: 0
    }
  }
  variable {
    name: "PRelu_2/prelu/slope"
    type: "Parameter"
    shape: { dim: 8 }
    initializer {
      type: "Constant"
      multiplier: 0.25
    }
  }
  variable {
    name: "Deconvolution/deconv/W"
    type: "Parameter"
    shape: { dim: 8 dim: 8 dim: 6 dim: 6 }
    initializer {
      type: "NormalConvolutionGlorot"
      multiplier: 1
    }
  }
  variable {
    name: "Deconvolution/deconv/b"
    type: "Parameter"
    shape: { dim: 8 }
    initializer {
      type: "Constant"
      multiplier: 0
    }
  }
  variable {
    name: "BatchNormalization_3/bn/beta"
    type: "Parameter"
    shape: { dim: 1 dim: 8 dim: 1 dim: 1 }
    initializer {
      type: "Constant"
      multiplier: 0
    }
  }
  variable {
    name: "BatchNormalization_3/bn/gamma"
    type: "Parameter"
    shape: { dim: 1 dim: 8 dim: 1 dim: 1 }
    initializer {
      type: "Constant"
      multiplier: 1
    }
  }
  variable {
    name: "BatchNormalization_3/bn/mean"
    type: "Parameter"
    shape: { dim: 1 dim: 8 dim: 1 dim: 1 }
    initializer {
      type: "Constant"
      multiplier: 0
    }
  }
  variable {
    name: "BatchNormalization_3/bn/var"
    type: "Parameter"
    shape: { dim: 1 dim: 8 dim: 1 dim: 1 }
    initializer {
      type: "Constant"
      multiplier: 0
    }
  }
  variable {
    name: "PRelu_3/prelu/slope"
    type: "Parameter"
    shape: { dim: 8 }
    initializer {
      type: "Constant"
      multiplier: 0.25
    }
  }
  variable {
    name: "Deconvolution_2/deconv/W"
    type: "Parameter"
    shape: { dim: 8 dim: 1 dim: 6 dim: 6 }
    initializer {
      type: "NormalConvolutionGlorot"
      multiplier: 1
    }
  }
  variable {
    name: "Deconvolution_2/deconv/b"
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
    shape: { dim:-1 dim: 8 dim: 14 dim: 14 }
  }
  variable {
    name: "BatchNormalization"
    type: "Buffer"
    shape: { dim:-1 dim: 8 dim: 14 dim: 14 }
  }
  variable {
    name: "PRelu"
    type: "Buffer"
    shape: { dim:-1 dim: 8 dim: 14 dim: 14 }
  }
  variable {
    name: "Convolution_2"
    type: "Buffer"
    shape: { dim:-1 dim: 8 dim: 7 dim: 7 }
  }
  variable {
    name: "BatchNormalization_2"
    type: "Buffer"
    shape: { dim:-1 dim: 8 dim: 7 dim: 7 }
  }
  variable {
    name: "PRelu_2"
    type: "Buffer"
    shape: { dim:-1 dim: 8 dim: 7 dim: 7 }
  }
  variable {
    name: "Deconvolution"
    type: "Buffer"
    shape: { dim:-1 dim: 8 dim: 14 dim: 14 }
  }
  variable {
    name: "BatchNormalization_3"
    type: "Buffer"
    shape: { dim:-1 dim: 8 dim: 14 dim: 14 }
  }
  variable {
    name: "PRelu_3"
    type: "Buffer"
    shape: { dim:-1 dim: 8 dim: 14 dim: 14 }
  }
  variable {
    name: "Deconvolution_2"
    type: "Buffer"
    shape: { dim:-1 dim: 1 dim: 28 dim: 28 }
  }
  variable {
    name: "Sigmoid"
    type: "Buffer"
    shape: { dim:-1 dim: 1 dim: 28 dim: 28 }
  }
  function {
    name: "Convolution"
    type: "Convolution"
    input: "Input"
    input: "Convolution/conv/W"
    input: "Convolution/conv/b"
    output: "Convolution"
    convolution_param {
      pad: { dim: 2 dim: 2 }
      stride: { dim: 2 dim: 2 }
      dilation: { dim: 1 dim: 1 }
      group: 1
      base_axis: 1
    }
  }
  function {
    name: "BatchNormalization"
    type: "BatchNormalization"
    input: "Convolution"
    input: "BatchNormalization/bn/beta"
    input: "BatchNormalization/bn/gamma"
    input: "BatchNormalization/bn/mean"
    input: "BatchNormalization/bn/var"
    output: "BatchNormalization"
    batch_normalization_param {
      axes: 1
      decay_rate: 0.5
      eps: 0.01
      batch_stat: False
    }
  }
  function {
    name: "PRelu"
    type: "PReLU"
    input: "BatchNormalization"
    input: "PRelu/prelu/slope"
    output: "PRelu"
    prelu_param {
      base_axis: 1
    }
  }
  function {
    name: "Convolution_2"
    type: "Convolution"
    input: "PRelu"
    input: "Convolution_2/conv/W"
    input: "Convolution_2/conv/b"
    output: "Convolution_2"
    convolution_param {
      pad: { dim: 2 dim: 2 }
      stride: { dim: 2 dim: 2 }
      dilation: { dim: 1 dim: 1 }
      group: 1
      base_axis: 1
    }
  }
  function {
    name: "BatchNormalization_2"
    type: "BatchNormalization"
    input: "Convolution_2"
    input: "BatchNormalization_2/bn/beta"
    input: "BatchNormalization_2/bn/gamma"
    input: "BatchNormalization_2/bn/mean"
    input: "BatchNormalization_2/bn/var"
    output: "BatchNormalization_2"
    batch_normalization_param {
      axes: 1
      decay_rate: 0.5
      eps: 0.01
      batch_stat: False
    }
  }
  function {
    name: "PRelu_2"
    type: "PReLU"
    input: "BatchNormalization_2"
    input: "PRelu_2/prelu/slope"
    output: "PRelu_2"
    prelu_param {
      base_axis: 1
    }
  }
  function {
    name: "Deconvolution"
    type: "Deconvolution"
    input: "PRelu_2"
    input: "Deconvolution/deconv/W"
    input: "Deconvolution/deconv/b"
    output: "Deconvolution"
    deconvolution_param {
      pad: { dim: 2 dim: 2 }
      stride: { dim: 2 dim: 2 }
      dilation: { dim: 1 dim: 1 }
      group: 1
      base_axis: 1
    }
  }
  function {
    name: "BatchNormalization_3"
    type: "BatchNormalization"
    input: "Deconvolution"
    input: "BatchNormalization_3/bn/beta"
    input: "BatchNormalization_3/bn/gamma"
    input: "BatchNormalization_3/bn/mean"
    input: "BatchNormalization_3/bn/var"
    output: "BatchNormalization_3"
    batch_normalization_param {
      axes: 1
      decay_rate: 0.5
      eps: 0.01
      batch_stat: False
    }
  }
  function {
    name: "PRelu_3"
    type: "PReLU"
    input: "BatchNormalization_3"
    input: "PRelu_3/prelu/slope"
    output: "PRelu_3"
    prelu_param {
      base_axis: 1
    }
  }
  function {
    name: "Deconvolution_2"
    type: "Deconvolution"
    input: "PRelu_3"
    input: "Deconvolution_2/deconv/W"
    input: "Deconvolution_2/deconv/b"
    output: "Deconvolution_2"
    deconvolution_param {
      pad: { dim: 2 dim: 2 }
      stride: { dim: 2 dim: 2 }
      dilation: { dim: 1 dim: 1 }
      group: 1
      base_axis: 1
    }
  }
  function {
    name: "Sigmoid"
    type: "Sigmoid"
    input: "Deconvolution_2"
    output: "Sigmoid"
  }
}
dataset {
  name: "Training"
  uri: "C:\\Utils\\neural_network_console_internal_v120\\samples\\sample_dataset\\mnist\\small_mnist_4or9_training.csv"
  cache_dir: "C:\\Utils\\neural_network_console_internal_v120\\samples\\sample_dataset\\mnist\\small_mnist_4or9_training.cache"
  overwrite_cache: True
  create_cache_explicitly: True
  shuffle: true
  no_image_normalization: False
  batch_size: 100
}
dataset {
  name: "Validation"
  uri: "C:\\Utils\\neural_network_console_internal_v120\\samples\\sample_dataset\\mnist\\small_mnist_4or9_test.csv"
  cache_dir: "C:\\Utils\\neural_network_console_internal_v120\\samples\\sample_dataset\\mnist\\small_mnist_4or9_test.cache"
  overwrite_cache: True
  create_cache_explicitly: True
  shuffle: false
  no_image_normalization: False
  batch_size: 100
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
    variable_name: "Output_T"
    data_name: "x"
  }
  loss_variable {
    variable_name: "Output"
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
    variable_name: "BatchNormalization/bn/beta"
    learning_rate_multiplier: 1
  }
  parameter_variable {
    variable_name: "BatchNormalization/bn/gamma"
    learning_rate_multiplier: 1
  }
  parameter_variable {
    variable_name: "BatchNormalization/bn/mean"
    learning_rate_multiplier: 0
  }
  parameter_variable {
    variable_name: "BatchNormalization/bn/var"
    learning_rate_multiplier: 0
  }
  parameter_variable {
    variable_name: "PRelu/prelu/slope"
    learning_rate_multiplier: 1
  }
  parameter_variable {
    variable_name: "Convolution_2/conv/W"
    learning_rate_multiplier: 1
  }
  parameter_variable {
    variable_name: "Convolution_2/conv/b"
    learning_rate_multiplier: 1
  }
  parameter_variable {
    variable_name: "BatchNormalization_2/bn/beta"
    learning_rate_multiplier: 1
  }
  parameter_variable {
    variable_name: "BatchNormalization_2/bn/gamma"
    learning_rate_multiplier: 1
  }
  parameter_variable {
    variable_name: "BatchNormalization_2/bn/mean"
    learning_rate_multiplier: 0
  }
  parameter_variable {
    variable_name: "BatchNormalization_2/bn/var"
    learning_rate_multiplier: 0
  }
  parameter_variable {
    variable_name: "PRelu_2/prelu/slope"
    learning_rate_multiplier: 1
  }
  parameter_variable {
    variable_name: "Deconvolution/deconv/W"
    learning_rate_multiplier: 1
  }
  parameter_variable {
    variable_name: "Deconvolution/deconv/b"
    learning_rate_multiplier: 1
  }
  parameter_variable {
    variable_name: "BatchNormalization_3/bn/beta"
    learning_rate_multiplier: 1
  }
  parameter_variable {
    variable_name: "BatchNormalization_3/bn/gamma"
    learning_rate_multiplier: 1
  }
  parameter_variable {
    variable_name: "BatchNormalization_3/bn/mean"
    learning_rate_multiplier: 0
  }
  parameter_variable {
    variable_name: "BatchNormalization_3/bn/var"
    learning_rate_multiplier: 0
  }
  parameter_variable {
    variable_name: "PRelu_3/prelu/slope"
    learning_rate_multiplier: 1
  }
  parameter_variable {
    variable_name: "Deconvolution_2/deconv/W"
    learning_rate_multiplier: 1
  }
  parameter_variable {
    variable_name: "Deconvolution_2/deconv/b"
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
    variable_name: "Output_T"
    data_name: "x"
  }
  monitor_variable {
    type: "Error"
    variable_name: "Output"
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
    variable_name: "Output_T"
    data_name: "x"
  }
  monitor_variable {
    type: "Error"
    variable_name: "Output"
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
    data_name: "x'"
  }
  parameter_variable {
    variable_name: "Convolution/conv/W"
  }
  parameter_variable {
    variable_name: "Convolution/conv/b"
  }
  parameter_variable {
    variable_name: "BatchNormalization/bn/beta"
  }
  parameter_variable {
    variable_name: "BatchNormalization/bn/gamma"
  }
  parameter_variable {
    variable_name: "BatchNormalization/bn/mean"
  }
  parameter_variable {
    variable_name: "BatchNormalization/bn/var"
  }
  parameter_variable {
    variable_name: "PRelu/prelu/slope"
  }
  parameter_variable {
    variable_name: "Convolution_2/conv/W"
  }
  parameter_variable {
    variable_name: "Convolution_2/conv/b"
  }
  parameter_variable {
    variable_name: "BatchNormalization_2/bn/beta"
  }
  parameter_variable {
    variable_name: "BatchNormalization_2/bn/gamma"
  }
  parameter_variable {
    variable_name: "BatchNormalization_2/bn/mean"
  }
  parameter_variable {
    variable_name: "BatchNormalization_2/bn/var"
  }
  parameter_variable {
    variable_name: "PRelu_2/prelu/slope"
  }
  parameter_variable {
    variable_name: "Deconvolution/deconv/W"
  }
  parameter_variable {
    variable_name: "Deconvolution/deconv/b"
  }
  parameter_variable {
    variable_name: "BatchNormalization_3/bn/beta"
  }
  parameter_variable {
    variable_name: "BatchNormalization_3/bn/gamma"
  }
  parameter_variable {
    variable_name: "BatchNormalization_3/bn/mean"
  }
  parameter_variable {
    variable_name: "BatchNormalization_3/bn/var"
  }
  parameter_variable {
    variable_name: "PRelu_3/prelu/slope"
  }
  parameter_variable {
    variable_name: "Deconvolution_2/deconv/W"
  }
  parameter_variable {
    variable_name: "Deconvolution_2/deconv/b"
  }
}
'''
# The nntxt is generated from /home/woody/aid/nnabla-sample-data/sample_project/tutorial/basics/01_logistic_regression.files/20170804_135944/net.nntxt.
N00000025 = r'''global_config {
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
  variable {
    name: "Input"
    type: "Buffer"
    shape: { dim:-1 dim: 1 dim: 28 dim: 28 }
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
      multiplier: 0.0
    }
  }
  variable {
    name: "BinaryCrossEntropy_T"
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
      multiplier: 0.0
    }
  }
  variable {
    name: "BinaryCrossEntropy_T"
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
      multiplier: 0.0
    }
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
'''
# The nntxt is generated from /home/woody/aid/nnabla-sample-data/sample_project/tutorial/basics/01_logistic_regression.files/20180720_162850/net.nntxt.
N00000026 = r'''global_config {
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
  uri: "C:\\Utils\\neural_network_console_internal_v120\\samples\\sample_dataset\\mnist\\small_mnist_4or9_training.csv"
  cache_dir: "C:\\Utils\\neural_network_console_internal_v120\\samples\\sample_dataset\\mnist\\small_mnist_4or9_training.cache"
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
'''
# The nntxt is generated from /home/woody/aid/nnabla-sample-data/sample_project/tutorial/basics/06_auto_encoder.files/20170804_140025/net.nntxt.
N00000027 = r'''global_config {
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
  variable {
    name: "Input"
    type: "Buffer"
    shape: { dim:-1 dim: 1 dim: 28 dim: 28 }
  }
  variable {
    name: "Dropout"
    type: "Buffer"
    shape: { dim:-1 dim: 1 dim: 28 dim: 28 }
  }
  variable {
    name: "Affine"
    type: "Buffer"
    shape: { dim:-1 dim: 256 }
  }
  variable {
    name: "Sigmoid"
    type: "Buffer"
    shape: { dim:-1 dim: 256 }
  }
  variable {
    name: "Affine_2"
    type: "Buffer"
    shape: { dim:-1 dim: 1 dim: 28 dim: 28 }
  }
  variable {
    name: "Sigmoid_2"
    type: "Buffer"
    shape: { dim:-1 dim: 1 dim: 28 dim: 28 }
  }
  variable {
    name: "SquaredError"
    type: "Buffer"
    shape: { dim:-1 dim: 1 dim: 28 dim: 28 }
  }
  variable {
    name: "Affine/affine/W"
    type: "Parameter"
    shape: { dim: 784 dim: 256 }
    initializer {
      type: "NormalAffineGlorot"
      multiplier: 1
    }
  }
  variable {
    name: "Affine/affine/b"
    type: "Parameter"
    shape: { dim: 256 }
    initializer {
      type: "Constant"
      multiplier: 0.0
    }
  }
  variable {
    name: "Affine_2/affine/W"
    type: "Parameter"
    shape: { dim: 256 dim: 1 dim: 28 dim: 28 }
    initializer {
      type: "NormalAffineGlorot"
      multiplier: 1
    }
  }
  variable {
    name: "Affine_2/affine/b"
    type: "Parameter"
    shape: { dim: 1 dim: 28 dim: 28 }
    initializer {
      type: "Constant"
      multiplier: 0.0
    }
  }
  variable {
    name: "SquaredError_T"
    type: "Buffer"
    shape: { dim:-1 dim: 1 dim: 28 dim: 28 }
  }
  function {
    name: "Dropout"
    type: "Dropout"
    input: "Input"
    output: "Dropout"
    dropout_param {
      p: 0.5
      seed: -1
    }
  }
  function {
    name: "Affine"
    type: "Affine"
    input: "Dropout"
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
    name: "Affine_2"
    type: "Affine"
    input: "Sigmoid"
    input: "Affine_2/affine/W"
    input: "Affine_2/affine/b"
    output: "Affine_2"
    affine_param {
      base_axis: 1
    }
  }
  function {
    name: "Sigmoid_2"
    type: "Sigmoid"
    input: "Affine_2"
    output: "Sigmoid_2"
  }
  function {
    name: "SquaredError"
    type: "SquaredError"
    input: "Sigmoid_2"
    input: "SquaredError_T"
    output: "SquaredError"
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
    name: "Affine"
    type: "Buffer"
    shape: { dim:-1 dim: 256 }
  }
  variable {
    name: "Sigmoid"
    type: "Buffer"
    shape: { dim:-1 dim: 256 }
  }
  variable {
    name: "Affine_2"
    type: "Buffer"
    shape: { dim:-1 dim: 1 dim: 28 dim: 28 }
  }
  variable {
    name: "Sigmoid_2"
    type: "Buffer"
    shape: { dim:-1 dim: 1 dim: 28 dim: 28 }
  }
  variable {
    name: "SquaredError"
    type: "Buffer"
    shape: { dim:-1 dim: 1 dim: 28 dim: 28 }
  }
  variable {
    name: "Affine/affine/W"
    type: "Parameter"
    shape: { dim: 784 dim: 256 }
    initializer {
      type: "NormalAffineGlorot"
      multiplier: 1
    }
  }
  variable {
    name: "Affine/affine/b"
    type: "Parameter"
    shape: { dim: 256 }
    initializer {
      type: "Constant"
      multiplier: 0.0
    }
  }
  variable {
    name: "Affine_2/affine/W"
    type: "Parameter"
    shape: { dim: 256 dim: 1 dim: 28 dim: 28 }
    initializer {
      type: "NormalAffineGlorot"
      multiplier: 1
    }
  }
  variable {
    name: "Affine_2/affine/b"
    type: "Parameter"
    shape: { dim: 1 dim: 28 dim: 28 }
    initializer {
      type: "Constant"
      multiplier: 0.0
    }
  }
  variable {
    name: "SquaredError_T"
    type: "Buffer"
    shape: { dim:-1 dim: 1 dim: 28 dim: 28 }
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
    name: "Affine_2"
    type: "Affine"
    input: "Sigmoid"
    input: "Affine_2/affine/W"
    input: "Affine_2/affine/b"
    output: "Affine_2"
    affine_param {
      base_axis: 1
    }
  }
  function {
    name: "Sigmoid_2"
    type: "Sigmoid"
    input: "Affine_2"
    output: "Sigmoid_2"
  }
  function {
    name: "SquaredError"
    type: "SquaredError"
    input: "Sigmoid_2"
    input: "SquaredError_T"
    output: "SquaredError"
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
    name: "Affine"
    type: "Buffer"
    shape: { dim:-1 dim: 256 }
  }
  variable {
    name: "Sigmoid"
    type: "Buffer"
    shape: { dim:-1 dim: 256 }
  }
  variable {
    name: "Affine_2"
    type: "Buffer"
    shape: { dim:-1 dim: 1 dim: 28 dim: 28 }
  }
  variable {
    name: "Sigmoid_2"
    type: "Buffer"
    shape: { dim:-1 dim: 1 dim: 28 dim: 28 }
  }
  variable {
    name: "Affine/affine/W"
    type: "Parameter"
    shape: { dim: 784 dim: 256 }
    initializer {
      type: "NormalAffineGlorot"
      multiplier: 1
    }
  }
  variable {
    name: "Affine/affine/b"
    type: "Parameter"
    shape: { dim: 256 }
    initializer {
      type: "Constant"
      multiplier: 0.0
    }
  }
  variable {
    name: "Affine_2/affine/W"
    type: "Parameter"
    shape: { dim: 256 dim: 1 dim: 28 dim: 28 }
    initializer {
      type: "NormalAffineGlorot"
      multiplier: 1
    }
  }
  variable {
    name: "Affine_2/affine/b"
    type: "Parameter"
    shape: { dim: 1 dim: 28 dim: 28 }
    initializer {
      type: "Constant"
      multiplier: 0.0
    }
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
    name: "Affine_2"
    type: "Affine"
    input: "Sigmoid"
    input: "Affine_2/affine/W"
    input: "Affine_2/affine/b"
    output: "Affine_2"
    affine_param {
      base_axis: 1
    }
  }
  function {
    name: "Sigmoid_2"
    type: "Sigmoid"
    input: "Affine_2"
    output: "Sigmoid_2"
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
  data_variable {
    variable_name: "SquaredError_T"
    data_name: "x"
  }
  loss_variable {
    variable_name: "SquaredError"
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
  data_variable {
    variable_name: "SquaredError_T"
    data_name: "x"
  }
  monitor_variable {
    type: "Error"
    variable_name: "SquaredError"
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
    variable_name: "SquaredError_T"
    data_name: "x"
  }
  monitor_variable {
    type: "Error"
    variable_name: "SquaredError"
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
    variable_name: "Sigmoid_2"
    data_name: "x'"
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
'''
# The nntxt is generated from /home/woody/aid/nnabla-sample-data/sample_project/tutorial/basics/06_auto_encoder.files/20180720_181844/net.nntxt.
N00000028 = r'''global_config {
  default_context {
    array_class: "CudaCachedArray"
    device_id: "0"
    backends: "cudnn:float"
    backends: "cuda:float"
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
    shape: { dim: 784 dim: 256 }
    initializer {
      type: "NormalAffineGlorot"
      multiplier: 1
    }
  }
  variable {
    name: "Affine/affine/b"
    type: "Parameter"
    shape: { dim: 256 }
    initializer {
      type: "Constant"
      multiplier: 0
    }
  }
  variable {
    name: "Affine_2/affine/W"
    type: "Parameter"
    shape: { dim: 256 dim: 1 dim: 28 dim: 28 }
    initializer {
      type: "NormalAffineGlorot"
      multiplier: 1
    }
  }
  variable {
    name: "Affine_2/affine/b"
    type: "Parameter"
    shape: { dim: 1 dim: 28 dim: 28 }
    initializer {
      type: "Constant"
      multiplier: 0
    }
  }
  variable {
    name: "SquaredError_T"
    type: "Buffer"
    shape: { dim:-1 dim: 1 dim: 28 dim: 28 }
  }
  variable {
    name: "Dropout"
    type: "Buffer"
    shape: { dim:-1 dim: 1 dim: 28 dim: 28 }
  }
  variable {
    name: "Affine"
    type: "Buffer"
    shape: { dim:-1 dim: 256 }
  }
  variable {
    name: "Sigmoid"
    type: "Buffer"
    shape: { dim:-1 dim: 256 }
  }
  variable {
    name: "Affine_2"
    type: "Buffer"
    shape: { dim:-1 dim: 1 dim: 28 dim: 28 }
  }
  variable {
    name: "Sigmoid_2"
    type: "Buffer"
    shape: { dim:-1 dim: 1 dim: 28 dim: 28 }
  }
  variable {
    name: "SquaredError"
    type: "Buffer"
    shape: { dim:-1 dim: 1 dim: 28 dim: 28 }
  }
  function {
    name: "Dropout"
    type: "Dropout"
    input: "Input"
    output: "Dropout"
    dropout_param {
      p: 0.5
      seed: -1
    }
  }
  function {
    name: "Affine"
    type: "Affine"
    input: "Dropout"
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
    name: "Affine_2"
    type: "Affine"
    input: "Sigmoid"
    input: "Affine_2/affine/W"
    input: "Affine_2/affine/b"
    output: "Affine_2"
    affine_param {
      base_axis: 1
    }
  }
  function {
    name: "Sigmoid_2"
    type: "Sigmoid"
    input: "Affine_2"
    output: "Sigmoid_2"
  }
  function {
    name: "SquaredError"
    type: "SquaredError"
    input: "Sigmoid_2"
    input: "SquaredError_T"
    output: "SquaredError"
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
    shape: { dim: 784 dim: 256 }
    initializer {
      type: "NormalAffineGlorot"
      multiplier: 1
    }
  }
  variable {
    name: "Affine/affine/b"
    type: "Parameter"
    shape: { dim: 256 }
    initializer {
      type: "Constant"
      multiplier: 0
    }
  }
  variable {
    name: "Affine_2/affine/W"
    type: "Parameter"
    shape: { dim: 256 dim: 1 dim: 28 dim: 28 }
    initializer {
      type: "NormalAffineGlorot"
      multiplier: 1
    }
  }
  variable {
    name: "Affine_2/affine/b"
    type: "Parameter"
    shape: { dim: 1 dim: 28 dim: 28 }
    initializer {
      type: "Constant"
      multiplier: 0
    }
  }
  variable {
    name: "SquaredError_T"
    type: "Buffer"
    shape: { dim:-1 dim: 1 dim: 28 dim: 28 }
  }
  variable {
    name: "Affine"
    type: "Buffer"
    shape: { dim:-1 dim: 256 }
  }
  variable {
    name: "Sigmoid"
    type: "Buffer"
    shape: { dim:-1 dim: 256 }
  }
  variable {
    name: "Affine_2"
    type: "Buffer"
    shape: { dim:-1 dim: 1 dim: 28 dim: 28 }
  }
  variable {
    name: "Sigmoid_2"
    type: "Buffer"
    shape: { dim:-1 dim: 1 dim: 28 dim: 28 }
  }
  variable {
    name: "SquaredError"
    type: "Buffer"
    shape: { dim:-1 dim: 1 dim: 28 dim: 28 }
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
    name: "Affine_2"
    type: "Affine"
    input: "Sigmoid"
    input: "Affine_2/affine/W"
    input: "Affine_2/affine/b"
    output: "Affine_2"
    affine_param {
      base_axis: 1
    }
  }
  function {
    name: "Sigmoid_2"
    type: "Sigmoid"
    input: "Affine_2"
    output: "Sigmoid_2"
  }
  function {
    name: "SquaredError"
    type: "SquaredError"
    input: "Sigmoid_2"
    input: "SquaredError_T"
    output: "SquaredError"
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
    shape: { dim: 784 dim: 256 }
    initializer {
      type: "NormalAffineGlorot"
      multiplier: 1
    }
  }
  variable {
    name: "Affine/affine/b"
    type: "Parameter"
    shape: { dim: 256 }
    initializer {
      type: "Constant"
      multiplier: 0
    }
  }
  variable {
    name: "Affine_2/affine/W"
    type: "Parameter"
    shape: { dim: 256 dim: 1 dim: 28 dim: 28 }
    initializer {
      type: "NormalAffineGlorot"
      multiplier: 1
    }
  }
  variable {
    name: "Affine_2/affine/b"
    type: "Parameter"
    shape: { dim: 1 dim: 28 dim: 28 }
    initializer {
      type: "Constant"
      multiplier: 0
    }
  }
  variable {
    name: "Affine"
    type: "Buffer"
    shape: { dim:-1 dim: 256 }
  }
  variable {
    name: "Sigmoid"
    type: "Buffer"
    shape: { dim:-1 dim: 256 }
  }
  variable {
    name: "Affine_2"
    type: "Buffer"
    shape: { dim:-1 dim: 1 dim: 28 dim: 28 }
  }
  variable {
    name: "Sigmoid_2"
    type: "Buffer"
    shape: { dim:-1 dim: 1 dim: 28 dim: 28 }
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
    name: "Affine_2"
    type: "Affine"
    input: "Sigmoid"
    input: "Affine_2/affine/W"
    input: "Affine_2/affine/b"
    output: "Affine_2"
    affine_param {
      base_axis: 1
    }
  }
  function {
    name: "Sigmoid_2"
    type: "Sigmoid"
    input: "Affine_2"
    output: "Sigmoid_2"
  }
}
dataset {
  name: "Training"
  uri: "C:\\Utils\\neural_network_console_internal_v120\\samples\\sample_dataset\\mnist\\small_mnist_4or9_training.csv"
  cache_dir: "C:\\Utils\\neural_network_console_internal_v120\\samples\\sample_dataset\\mnist\\small_mnist_4or9_training.cache"
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
    variable_name: "SquaredError_T"
    data_name: "x"
  }
  loss_variable {
    variable_name: "SquaredError"
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
    variable_name: "SquaredError_T"
    data_name: "x"
  }
  monitor_variable {
    type: "Error"
    variable_name: "SquaredError"
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
    variable_name: "SquaredError_T"
    data_name: "x"
  }
  monitor_variable {
    type: "Error"
    variable_name: "SquaredError"
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
    variable_name: "Sigmoid_2"
    data_name: "x'"
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
'''
# The nntxt is generated from /home/woody/aid/nnabla-sample-data/sample_project/tutorial/basics/02_binary_cnn.files/20170804_140002/net.nntxt.
N00000029 = r'''global_config {
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
  variable {
    name: "Input"
    type: "Buffer"
    shape: { dim:-1 dim: 1 dim: 28 dim: 28 }
  }
  variable {
    name: "Convolution"
    type: "Buffer"
    shape: { dim:-1 dim: 16 dim: 24 dim: 24 }
  }
  variable {
    name: "MaxPooling"
    type: "Buffer"
    shape: { dim:-1 dim: 16 dim: 12 dim: 12 }
  }
  variable {
    name: "Tanh"
    type: "Buffer"
    shape: { dim:-1 dim: 16 dim: 12 dim: 12 }
  }
  variable {
    name: "Convolution_2"
    type: "Buffer"
    shape: { dim:-1 dim: 8 dim: 8 dim: 8 }
  }
  variable {
    name: "MaxPooling_2"
    type: "Buffer"
    shape: { dim:-1 dim: 8 dim: 4 dim: 4 }
  }
  variable {
    name: "Tanh_2"
    type: "Buffer"
    shape: { dim:-1 dim: 8 dim: 4 dim: 4 }
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
  variable {
    name: "Convolution/conv/W"
    type: "Parameter"
    shape: { dim: 16 dim: 1 dim: 5 dim: 5 }
    initializer {
      type: "NormalConvolutionGlorot"
      multiplier: 1
    }
  }
  variable {
    name: "Convolution/conv/b"
    type: "Parameter"
    shape: { dim: 16 }
    initializer {
      type: "Constant"
      multiplier: 0.0
    }
  }
  variable {
    name: "Convolution_2/conv/W"
    type: "Parameter"
    shape: { dim: 8 dim: 16 dim: 5 dim: 5 }
    initializer {
      type: "NormalConvolutionGlorot"
      multiplier: 1
    }
  }
  variable {
    name: "Convolution_2/conv/b"
    type: "Parameter"
    shape: { dim: 8 }
    initializer {
      type: "Constant"
      multiplier: 0.0
    }
  }
  variable {
    name: "Affine/affine/W"
    type: "Parameter"
    shape: { dim: 128 dim: 10 }
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
      multiplier: 0.0
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
      multiplier: 0.0
    }
  }
  variable {
    name: "BinaryCrossEntropy_T"
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
    name: "Convolution_2"
    type: "Convolution"
    input: "Tanh"
    input: "Convolution_2/conv/W"
    input: "Convolution_2/conv/b"
    output: "Convolution_2"
    convolution_param {
      pad: { dim: 0 dim: 0 }
      stride: { dim: 1 dim: 1 }
      dilation: { dim: 1 dim: 1 }
      group: 1
      base_axis: 1
    }
  }
  function {
    name: "MaxPooling_2"
    type: "MaxPooling"
    input: "Convolution_2"
    output: "MaxPooling_2"
    max_pooling_param {
      kernel: { dim: 2 dim: 2 }
      stride: { dim: 2 dim: 2 }
      ignore_border: true
      pad: { dim: 0 dim: 0 }
    }
  }
  function {
    name: "Tanh_2"
    type: "Tanh"
    input: "MaxPooling_2"
    output: "Tanh_2"
  }
  function {
    name: "Affine"
    type: "Affine"
    input: "Tanh_2"
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
  variable {
    name: "Input"
    type: "Buffer"
    shape: { dim:-1 dim: 1 dim: 28 dim: 28 }
  }
  variable {
    name: "Convolution"
    type: "Buffer"
    shape: { dim:-1 dim: 16 dim: 24 dim: 24 }
  }
  variable {
    name: "MaxPooling"
    type: "Buffer"
    shape: { dim:-1 dim: 16 dim: 12 dim: 12 }
  }
  variable {
    name: "Tanh"
    type: "Buffer"
    shape: { dim:-1 dim: 16 dim: 12 dim: 12 }
  }
  variable {
    name: "Convolution_2"
    type: "Buffer"
    shape: { dim:-1 dim: 8 dim: 8 dim: 8 }
  }
  variable {
    name: "MaxPooling_2"
    type: "Buffer"
    shape: { dim:-1 dim: 8 dim: 4 dim: 4 }
  }
  variable {
    name: "Tanh_2"
    type: "Buffer"
    shape: { dim:-1 dim: 8 dim: 4 dim: 4 }
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
  variable {
    name: "Convolution/conv/W"
    type: "Parameter"
    shape: { dim: 16 dim: 1 dim: 5 dim: 5 }
    initializer {
      type: "NormalConvolutionGlorot"
      multiplier: 1
    }
  }
  variable {
    name: "Convolution/conv/b"
    type: "Parameter"
    shape: { dim: 16 }
    initializer {
      type: "Constant"
      multiplier: 0.0
    }
  }
  variable {
    name: "Convolution_2/conv/W"
    type: "Parameter"
    shape: { dim: 8 dim: 16 dim: 5 dim: 5 }
    initializer {
      type: "NormalConvolutionGlorot"
      multiplier: 1
    }
  }
  variable {
    name: "Convolution_2/conv/b"
    type: "Parameter"
    shape: { dim: 8 }
    initializer {
      type: "Constant"
      multiplier: 0.0
    }
  }
  variable {
    name: "Affine/affine/W"
    type: "Parameter"
    shape: { dim: 128 dim: 10 }
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
      multiplier: 0.0
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
      multiplier: 0.0
    }
  }
  variable {
    name: "BinaryCrossEntropy_T"
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
    name: "Convolution_2"
    type: "Convolution"
    input: "Tanh"
    input: "Convolution_2/conv/W"
    input: "Convolution_2/conv/b"
    output: "Convolution_2"
    convolution_param {
      pad: { dim: 0 dim: 0 }
      stride: { dim: 1 dim: 1 }
      dilation: { dim: 1 dim: 1 }
      group: 1
      base_axis: 1
    }
  }
  function {
    name: "MaxPooling_2"
    type: "MaxPooling"
    input: "Convolution_2"
    output: "MaxPooling_2"
    max_pooling_param {
      kernel: { dim: 2 dim: 2 }
      stride: { dim: 2 dim: 2 }
      ignore_border: true
      pad: { dim: 0 dim: 0 }
    }
  }
  function {
    name: "Tanh_2"
    type: "Tanh"
    input: "MaxPooling_2"
    output: "Tanh_2"
  }
  function {
    name: "Affine"
    type: "Affine"
    input: "Tanh_2"
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
  variable {
    name: "Input"
    type: "Buffer"
    shape: { dim:-1 dim: 1 dim: 28 dim: 28 }
  }
  variable {
    name: "Convolution"
    type: "Buffer"
    shape: { dim:-1 dim: 16 dim: 24 dim: 24 }
  }
  variable {
    name: "MaxPooling"
    type: "Buffer"
    shape: { dim:-1 dim: 16 dim: 12 dim: 12 }
  }
  variable {
    name: "Tanh"
    type: "Buffer"
    shape: { dim:-1 dim: 16 dim: 12 dim: 12 }
  }
  variable {
    name: "Convolution_2"
    type: "Buffer"
    shape: { dim:-1 dim: 8 dim: 8 dim: 8 }
  }
  variable {
    name: "MaxPooling_2"
    type: "Buffer"
    shape: { dim:-1 dim: 8 dim: 4 dim: 4 }
  }
  variable {
    name: "Tanh_2"
    type: "Buffer"
    shape: { dim:-1 dim: 8 dim: 4 dim: 4 }
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
    name: "Convolution/conv/W"
    type: "Parameter"
    shape: { dim: 16 dim: 1 dim: 5 dim: 5 }
    initializer {
      type: "NormalConvolutionGlorot"
      multiplier: 1
    }
  }
  variable {
    name: "Convolution/conv/b"
    type: "Parameter"
    shape: { dim: 16 }
    initializer {
      type: "Constant"
      multiplier: 0.0
    }
  }
  variable {
    name: "Convolution_2/conv/W"
    type: "Parameter"
    shape: { dim: 8 dim: 16 dim: 5 dim: 5 }
    initializer {
      type: "NormalConvolutionGlorot"
      multiplier: 1
    }
  }
  variable {
    name: "Convolution_2/conv/b"
    type: "Parameter"
    shape: { dim: 8 }
    initializer {
      type: "Constant"
      multiplier: 0.0
    }
  }
  variable {
    name: "Affine/affine/W"
    type: "Parameter"
    shape: { dim: 128 dim: 10 }
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
      multiplier: 0.0
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
      multiplier: 0.0
    }
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
    name: "Convolution_2"
    type: "Convolution"
    input: "Tanh"
    input: "Convolution_2/conv/W"
    input: "Convolution_2/conv/b"
    output: "Convolution_2"
    convolution_param {
      pad: { dim: 0 dim: 0 }
      stride: { dim: 1 dim: 1 }
      dilation: { dim: 1 dim: 1 }
      group: 1
      base_axis: 1
    }
  }
  function {
    name: "MaxPooling_2"
    type: "MaxPooling"
    input: "Convolution_2"
    output: "MaxPooling_2"
    max_pooling_param {
      kernel: { dim: 2 dim: 2 }
      stride: { dim: 2 dim: 2 }
      ignore_border: true
      pad: { dim: 0 dim: 0 }
    }
  }
  function {
    name: "Tanh_2"
    type: "Tanh"
    input: "MaxPooling_2"
    output: "Tanh_2"
  }
  function {
    name: "Affine"
    type: "Affine"
    input: "Tanh_2"
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
  data_variable {
    variable_name: "BinaryCrossEntropy_T"
    data_name: "y"
  }
  loss_variable {
    variable_name: "BinaryCrossEntropy"
  }
  parameter_variable {
    variable_name: "Convolution/conv/W"
    learning_rate_multiplier: 1.0
  }
  parameter_variable {
    variable_name: "Convolution/conv/b"
    learning_rate_multiplier: 1.0
  }
  parameter_variable {
    variable_name: "Convolution_2/conv/W"
    learning_rate_multiplier: 1.0
  }
  parameter_variable {
    variable_name: "Convolution_2/conv/b"
    learning_rate_multiplier: 1.0
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
    variable_name: "Convolution_2/conv/W"
  }
  parameter_variable {
    variable_name: "Convolution_2/conv/b"
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
'''
# The nntxt is generated from /home/woody/aid/nnabla-sample-data/sample_project/tutorial/basics/02_binary_cnn.files/20180720_181750/net.nntxt.
N00000030 = r'''global_config {
  default_context {
    array_class: "CudaCachedArray"
    device_id: "0"
    backends: "cudnn:float"
    backends: "cuda:float"
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
    name: "Convolution/conv/W"
    type: "Parameter"
    shape: { dim: 16 dim: 1 dim: 5 dim: 5 }
    initializer {
      type: "NormalConvolutionGlorot"
      multiplier: 1
    }
  }
  variable {
    name: "Convolution/conv/b"
    type: "Parameter"
    shape: { dim: 16 }
    initializer {
      type: "Constant"
      multiplier: 0
    }
  }
  variable {
    name: "Convolution_2/conv/W"
    type: "Parameter"
    shape: { dim: 8 dim: 16 dim: 5 dim: 5 }
    initializer {
      type: "NormalConvolutionGlorot"
      multiplier: 1
    }
  }
  variable {
    name: "Convolution_2/conv/b"
    type: "Parameter"
    shape: { dim: 8 }
    initializer {
      type: "Constant"
      multiplier: 0
    }
  }
  variable {
    name: "Affine/affine/W"
    type: "Parameter"
    shape: { dim: 128 dim: 10 }
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
    shape: { dim:-1 dim: 16 dim: 24 dim: 24 }
  }
  variable {
    name: "MaxPooling"
    type: "Buffer"
    shape: { dim:-1 dim: 16 dim: 12 dim: 12 }
  }
  variable {
    name: "Tanh"
    type: "Buffer"
    shape: { dim:-1 dim: 16 dim: 12 dim: 12 }
  }
  variable {
    name: "Convolution_2"
    type: "Buffer"
    shape: { dim:-1 dim: 8 dim: 8 dim: 8 }
  }
  variable {
    name: "MaxPooling_2"
    type: "Buffer"
    shape: { dim:-1 dim: 8 dim: 4 dim: 4 }
  }
  variable {
    name: "Tanh_2"
    type: "Buffer"
    shape: { dim:-1 dim: 8 dim: 4 dim: 4 }
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
    name: "Convolution_2"
    type: "Convolution"
    input: "Tanh"
    input: "Convolution_2/conv/W"
    input: "Convolution_2/conv/b"
    output: "Convolution_2"
    convolution_param {
      pad: { dim: 0 dim: 0 }
      stride: { dim: 1 dim: 1 }
      dilation: { dim: 1 dim: 1 }
      group: 1
      base_axis: 1
    }
  }
  function {
    name: "MaxPooling_2"
    type: "MaxPooling"
    input: "Convolution_2"
    output: "MaxPooling_2"
    max_pooling_param {
      kernel: { dim: 2 dim: 2 }
      stride: { dim: 2 dim: 2 }
      ignore_border: true
      pad: { dim: 0 dim: 0 }
    }
  }
  function {
    name: "Tanh_2"
    type: "Tanh"
    input: "MaxPooling_2"
    output: "Tanh_2"
  }
  function {
    name: "Affine"
    type: "Affine"
    input: "Tanh_2"
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
  variable {
    name: "Input"
    type: "Buffer"
    shape: { dim:-1 dim: 1 dim: 28 dim: 28 }
  }
  variable {
    name: "Convolution/conv/W"
    type: "Parameter"
    shape: { dim: 16 dim: 1 dim: 5 dim: 5 }
    initializer {
      type: "NormalConvolutionGlorot"
      multiplier: 1
    }
  }
  variable {
    name: "Convolution/conv/b"
    type: "Parameter"
    shape: { dim: 16 }
    initializer {
      type: "Constant"
      multiplier: 0
    }
  }
  variable {
    name: "Convolution_2/conv/W"
    type: "Parameter"
    shape: { dim: 8 dim: 16 dim: 5 dim: 5 }
    initializer {
      type: "NormalConvolutionGlorot"
      multiplier: 1
    }
  }
  variable {
    name: "Convolution_2/conv/b"
    type: "Parameter"
    shape: { dim: 8 }
    initializer {
      type: "Constant"
      multiplier: 0
    }
  }
  variable {
    name: "Affine/affine/W"
    type: "Parameter"
    shape: { dim: 128 dim: 10 }
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
    shape: { dim:-1 dim: 16 dim: 24 dim: 24 }
  }
  variable {
    name: "MaxPooling"
    type: "Buffer"
    shape: { dim:-1 dim: 16 dim: 12 dim: 12 }
  }
  variable {
    name: "Tanh"
    type: "Buffer"
    shape: { dim:-1 dim: 16 dim: 12 dim: 12 }
  }
  variable {
    name: "Convolution_2"
    type: "Buffer"
    shape: { dim:-1 dim: 8 dim: 8 dim: 8 }
  }
  variable {
    name: "MaxPooling_2"
    type: "Buffer"
    shape: { dim:-1 dim: 8 dim: 4 dim: 4 }
  }
  variable {
    name: "Tanh_2"
    type: "Buffer"
    shape: { dim:-1 dim: 8 dim: 4 dim: 4 }
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
    name: "Convolution_2"
    type: "Convolution"
    input: "Tanh"
    input: "Convolution_2/conv/W"
    input: "Convolution_2/conv/b"
    output: "Convolution_2"
    convolution_param {
      pad: { dim: 0 dim: 0 }
      stride: { dim: 1 dim: 1 }
      dilation: { dim: 1 dim: 1 }
      group: 1
      base_axis: 1
    }
  }
  function {
    name: "MaxPooling_2"
    type: "MaxPooling"
    input: "Convolution_2"
    output: "MaxPooling_2"
    max_pooling_param {
      kernel: { dim: 2 dim: 2 }
      stride: { dim: 2 dim: 2 }
      ignore_border: true
      pad: { dim: 0 dim: 0 }
    }
  }
  function {
    name: "Tanh_2"
    type: "Tanh"
    input: "MaxPooling_2"
    output: "Tanh_2"
  }
  function {
    name: "Affine"
    type: "Affine"
    input: "Tanh_2"
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
  variable {
    name: "Input"
    type: "Buffer"
    shape: { dim:-1 dim: 1 dim: 28 dim: 28 }
  }
  variable {
    name: "Convolution/conv/W"
    type: "Parameter"
    shape: { dim: 16 dim: 1 dim: 5 dim: 5 }
    initializer {
      type: "NormalConvolutionGlorot"
      multiplier: 1
    }
  }
  variable {
    name: "Convolution/conv/b"
    type: "Parameter"
    shape: { dim: 16 }
    initializer {
      type: "Constant"
      multiplier: 0
    }
  }
  variable {
    name: "Convolution_2/conv/W"
    type: "Parameter"
    shape: { dim: 8 dim: 16 dim: 5 dim: 5 }
    initializer {
      type: "NormalConvolutionGlorot"
      multiplier: 1
    }
  }
  variable {
    name: "Convolution_2/conv/b"
    type: "Parameter"
    shape: { dim: 8 }
    initializer {
      type: "Constant"
      multiplier: 0
    }
  }
  variable {
    name: "Affine/affine/W"
    type: "Parameter"
    shape: { dim: 128 dim: 10 }
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
    shape: { dim:-1 dim: 16 dim: 24 dim: 24 }
  }
  variable {
    name: "MaxPooling"
    type: "Buffer"
    shape: { dim:-1 dim: 16 dim: 12 dim: 12 }
  }
  variable {
    name: "Tanh"
    type: "Buffer"
    shape: { dim:-1 dim: 16 dim: 12 dim: 12 }
  }
  variable {
    name: "Convolution_2"
    type: "Buffer"
    shape: { dim:-1 dim: 8 dim: 8 dim: 8 }
  }
  variable {
    name: "MaxPooling_2"
    type: "Buffer"
    shape: { dim:-1 dim: 8 dim: 4 dim: 4 }
  }
  variable {
    name: "Tanh_2"
    type: "Buffer"
    shape: { dim:-1 dim: 8 dim: 4 dim: 4 }
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
    name: "Convolution_2"
    type: "Convolution"
    input: "Tanh"
    input: "Convolution_2/conv/W"
    input: "Convolution_2/conv/b"
    output: "Convolution_2"
    convolution_param {
      pad: { dim: 0 dim: 0 }
      stride: { dim: 1 dim: 1 }
      dilation: { dim: 1 dim: 1 }
      group: 1
      base_axis: 1
    }
  }
  function {
    name: "MaxPooling_2"
    type: "MaxPooling"
    input: "Convolution_2"
    output: "MaxPooling_2"
    max_pooling_param {
      kernel: { dim: 2 dim: 2 }
      stride: { dim: 2 dim: 2 }
      ignore_border: true
      pad: { dim: 0 dim: 0 }
    }
  }
  function {
    name: "Tanh_2"
    type: "Tanh"
    input: "MaxPooling_2"
    output: "Tanh_2"
  }
  function {
    name: "Affine"
    type: "Affine"
    input: "Tanh_2"
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
  uri: "C:\\Utils\\neural_network_console_internal_v120\\samples\\sample_dataset\\mnist\\small_mnist_4or9_training.csv"
  cache_dir: "C:\\Utils\\neural_network_console_internal_v120\\samples\\sample_dataset\\mnist\\small_mnist_4or9_training.cache"
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
    variable_name: "Convolution/conv/W"
    learning_rate_multiplier: 1
  }
  parameter_variable {
    variable_name: "Convolution/conv/b"
    learning_rate_multiplier: 1
  }
  parameter_variable {
    variable_name: "Convolution_2/conv/W"
    learning_rate_multiplier: 1
  }
  parameter_variable {
    variable_name: "Convolution_2/conv/b"
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
    variable_name: "Convolution_2/conv/W"
  }
  parameter_variable {
    variable_name: "Convolution_2/conv/b"
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
'''

N00000100 = r'''
dataset {
  name: "Evaluate"
  uri: "C:\\Utils\\neural_network_console_internal_v120\\samples\\sample_dataset\\mnist\\small_mnist_4or9_training.csv"
  cache_dir: "C:\\Utils\\neural_network_console_internal_v120\\samples\\sample_dataset\\mnist\\small_mnist_4or9_training.cache"
  overwrite_cache: True
  create_cache_explicitly: True
  shuffle: true
  no_image_normalization: False
  batch_size: 32
}

network {
  name: "network1"
  batch_size: 32
  variable {
    name: "x"
    type: "Buffer"
    shape {
      dim: 32
      dim: 1
      dim: 28
      dim: 28
    }
  }
  variable {
    name: "y0"
    type: "Buffer"
    shape {
      dim: 32
      dim: 56
      dim: 14
    }
  }
  function {
    name: "Reshape"
    type: "Reshape"
    input: "x"
    output: "y0"
    reshape_param {
      shape {
        dim: 32
        dim: 56
        dim: -1
      }
    }
  }
}
executor {
  name: "inference"
  network_name: "network1"
  data_variable {
    variable_name: "x"
    data_name: "x"
  }
  output_variable {
    variable_name: "y0"
    data_name: "y0"
  }
}
'''


NNTXT_EQUIVALENCE_CASES = [N00000000, N00000001, N00000002, N00000003, N00000004, N00000005, N00000006, N00000007, N00000008, N00000009, N00000010, N00000011, N00000012, N00000013,
                           N00000014, N00000015, N00000016, N00000017, N00000018, N00000019, N00000020, N00000021, N00000022, N00000023, N00000024, N00000025, N00000026, N00000027, N00000028, N00000029, N00000030]
EXCLUDE_SLOW_CASES = {20, 19, 17, 18, 14, 10, 11, 16, 15, 27, 28}
CASE_INDEX = list(
    set(range(0, len(NNTXT_EQUIVALENCE_CASES))) - EXCLUDE_SLOW_CASES)
NNTXT_IMPROVEMENT_CASES = [N00000100]
