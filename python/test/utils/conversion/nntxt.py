# Copyright 2022 Sony Group Corporation.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# BoolFill not supported case
N0001 = r'''
network {
  name: "test"
  variable {
    name: "@test/input"
    type: "Buffer"
    shape {
      dim: 2
      dim: 2
    }
  }
  variable {
    name: "@test/mask"
    type: "Buffer"
    shape {
      dim: 2
      dim: 2
    }
  }
  variable {
    name: "@test/output"
    type: "Buffer"
    shape {
      dim: 2
      dim: 2
    }
  }
  function {
    name: "@test/BoolFill"
    type: "BoolFill"
    input: "@test/input"
    input: "@test/mask"
    output: "@test/output"
    bool_fill_param {
       value: 1
    }
  }
}

executor {
  name: "runtime"
  network_name: "test"
  data_variable {
    variable_name: "@test/input"
    data_name: "input"
  }
  data_variable {
    variable_name: "@test/mask"
    data_name: "mask"
  }
  output_variable {
    variable_name: "@test/output"
    data_name: "output"
  }
}  
'''


# SoftPlus, arguments change case, argument value equal to default value
N0002 = r'''
network {
  name: "test"
  variable {
    name: "@test/x"
    type: "Buffer"
    shape {
      dim: 2
      dim: 2
    }
  }
  variable {
    name: "@test/output"
    type: "Buffer"
    shape {
      dim: 2
      dim: 2
    }
  }
  function {
    name: "@test/SoftPlus"
    type: "SoftPlus"
    input: "@test/x"
    output: "@test/mask"
    softplus_param {
       beta: 1.0
    }
  }
}

executor {
  name: "runtime"
  network_name: "test"
  data_variable {
    variable_name: "@test/x"
    data_name: "x"
  }
  output_variable {
    variable_name: "@test/output"
    data_name: "output"
  }
}  
'''

# SoftPlus, arguments change case, argument value not equal to default value
N0003 = r'''
network {
  name: "test"
  variable {
    name: "@test/x"
    type: "Buffer"
    shape {
      dim: 2
      dim: 2
    }
  }
  variable {
    name: "@test/output"
    type: "Buffer"
    shape {
      dim: 2
      dim: 2
    }
  }
  function {
    name: "@test/SoftPlus"
    type: "SoftPlus"
    input: "@test/x"
    output: "@test/mask"
    softplus_param {
       beta: 2.0
    }
  }
}

executor {
  name: "runtime"
  network_name: "test"
  data_variable {
    variable_name: "@test/x"
    data_name: "x"
  }
  output_variable {
    variable_name: "@test/output"
    data_name: "output"
  }
}  
'''

# MulScalar, with camel to snake naming conversion
N0004 = r'''
network {
  name: "net1"
  batch_size: 1
  variable {
    name: "x"
    type: "Buffer"
    shape {
      dim: -1
      dim: 3
      dim: 224
      dim: 224
    }
  }
  variable {
    name: "@net1/MulScalar_out"
    type: "Buffer"
    shape {
      dim: -1
      dim: 3
      dim: 224
      dim: 224
    }
  }
  variable {
    name: "@net1/AddScalar_out"
    type: "Buffer"
    shape {
      dim: -1
      dim: 3
      dim: 224
      dim: 224
    }
  }
  variable {
    name: "@net1/Convolution_out"
    type: "Buffer"
    shape {
      dim: -1
      dim: 64
      dim: 112
      dim: 112
    }
  }
  variable {
    name: "@net1/BatchNormalization_out"
    type: "Buffer"
    shape {
      dim: -1
      dim: 64
      dim: 112
      dim: 112
    }
  }
  variable {
    name: "@net1/ReLU_out"
    type: "Buffer"
    shape {
      dim: -1
      dim: 64
      dim: 112
      dim: 112
    }
  }
  variable {
    name: "@net1/MaxPooling_out"
    type: "Buffer"
    shape {
      dim: -1
      dim: 64
      dim: 56
      dim: 56
    }
  }
  variable {
    name: "@net1/Convolution_out_1"
    type: "Buffer"
    shape {
      dim: -1
      dim: 64
      dim: 56
      dim: 56
    }
  }
  variable {
    name: "@net1/BatchNormalization_out_1"
    type: "Buffer"
    shape {
      dim: -1
      dim: 64
      dim: 56
      dim: 56
    }
  }
  variable {
    name: "@net1/ReLU_out_1"
    type: "Buffer"
    shape {
      dim: -1
      dim: 64
      dim: 56
      dim: 56
    }
  }
  variable {
    name: "@net1/Convolution_out_2"
    type: "Buffer"
    shape {
      dim: -1
      dim: 64
      dim: 56
      dim: 56
    }
  }
  variable {
    name: "@net1/BatchNormalization_out_2"
    type: "Buffer"
    shape {
      dim: -1
      dim: 64
      dim: 56
      dim: 56
    }
  }
  variable {
    name: "@net1/Add2_out"
    type: "Buffer"
    shape {
      dim: -1
      dim: 64
      dim: 56
      dim: 56
    }
  }
  variable {
    name: "@net1/ReLU_out_2"
    type: "Buffer"
    shape {
      dim: -1
      dim: 64
      dim: 56
      dim: 56
    }
  }
  variable {
    name: "@net1/Convolution_out_3"
    type: "Buffer"
    shape {
      dim: -1
      dim: 64
      dim: 56
      dim: 56
    }
  }
  variable {
    name: "@net1/BatchNormalization_out_3"
    type: "Buffer"
    shape {
      dim: -1
      dim: 64
      dim: 56
      dim: 56
    }
  }
  variable {
    name: "@net1/ReLU_out_3"
    type: "Buffer"
    shape {
      dim: -1
      dim: 64
      dim: 56
      dim: 56
    }
  }
  variable {
    name: "@net1/Convolution_out_4"
    type: "Buffer"
    shape {
      dim: -1
      dim: 64
      dim: 56
      dim: 56
    }
  }
  variable {
    name: "@net1/BatchNormalization_out_4"
    type: "Buffer"
    shape {
      dim: -1
      dim: 64
      dim: 56
      dim: 56
    }
  }
  variable {
    name: "@net1/Add2_out_1"
    type: "Buffer"
    shape {
      dim: -1
      dim: 64
      dim: 56
      dim: 56
    }
  }
  variable {
    name: "@net1/ReLU_out_4"
    type: "Buffer"
    shape {
      dim: -1
      dim: 64
      dim: 56
      dim: 56
    }
  }
  variable {
    name: "@net1/Convolution_out_5"
    type: "Buffer"
    shape {
      dim: -1
      dim: 128
      dim: 28
      dim: 28
    }
  }
  variable {
    name: "@net1/BatchNormalization_out_5"
    type: "Buffer"
    shape {
      dim: -1
      dim: 128
      dim: 28
      dim: 28
    }
  }
  variable {
    name: "@net1/Convolution_out_6"
    type: "Buffer"
    shape {
      dim: -1
      dim: 128
      dim: 28
      dim: 28
    }
  }
  variable {
    name: "@net1/BatchNormalization_out_6"
    type: "Buffer"
    shape {
      dim: -1
      dim: 128
      dim: 28
      dim: 28
    }
  }
  variable {
    name: "@net1/ReLU_out_5"
    type: "Buffer"
    shape {
      dim: -1
      dim: 128
      dim: 28
      dim: 28
    }
  }
  variable {
    name: "@net1/Convolution_out_7"
    type: "Buffer"
    shape {
      dim: -1
      dim: 128
      dim: 28
      dim: 28
    }
  }
  variable {
    name: "@net1/BatchNormalization_out_7"
    type: "Buffer"
    shape {
      dim: -1
      dim: 128
      dim: 28
      dim: 28
    }
  }
  variable {
    name: "@net1/Add2_out_2"
    type: "Buffer"
    shape {
      dim: -1
      dim: 128
      dim: 28
      dim: 28
    }
  }
  variable {
    name: "@net1/ReLU_out_6"
    type: "Buffer"
    shape {
      dim: -1
      dim: 128
      dim: 28
      dim: 28
    }
  }
  variable {
    name: "@net1/Convolution_out_8"
    type: "Buffer"
    shape {
      dim: -1
      dim: 128
      dim: 28
      dim: 28
    }
  }
  variable {
    name: "@net1/BatchNormalization_out_8"
    type: "Buffer"
    shape {
      dim: -1
      dim: 128
      dim: 28
      dim: 28
    }
  }
  variable {
    name: "@net1/ReLU_out_7"
    type: "Buffer"
    shape {
      dim: -1
      dim: 128
      dim: 28
      dim: 28
    }
  }
  variable {
    name: "@net1/Convolution_out_9"
    type: "Buffer"
    shape {
      dim: -1
      dim: 128
      dim: 28
      dim: 28
    }
  }
  variable {
    name: "@net1/BatchNormalization_out_9"
    type: "Buffer"
    shape {
      dim: -1
      dim: 128
      dim: 28
      dim: 28
    }
  }
  variable {
    name: "@net1/Add2_out_3"
    type: "Buffer"
    shape {
      dim: -1
      dim: 128
      dim: 28
      dim: 28
    }
  }
  variable {
    name: "@net1/ReLU_out_8"
    type: "Buffer"
    shape {
      dim: -1
      dim: 128
      dim: 28
      dim: 28
    }
  }
  variable {
    name: "@net1/Convolution_out_10"
    type: "Buffer"
    shape {
      dim: -1
      dim: 256
      dim: 14
      dim: 14
    }
  }
  variable {
    name: "@net1/BatchNormalization_out_10"
    type: "Buffer"
    shape {
      dim: -1
      dim: 256
      dim: 14
      dim: 14
    }
  }
  variable {
    name: "@net1/Convolution_out_11"
    type: "Buffer"
    shape {
      dim: -1
      dim: 256
      dim: 14
      dim: 14
    }
  }
  variable {
    name: "@net1/BatchNormalization_out_11"
    type: "Buffer"
    shape {
      dim: -1
      dim: 256
      dim: 14
      dim: 14
    }
  }
  variable {
    name: "@net1/ReLU_out_9"
    type: "Buffer"
    shape {
      dim: -1
      dim: 256
      dim: 14
      dim: 14
    }
  }
  variable {
    name: "@net1/Convolution_out_12"
    type: "Buffer"
    shape {
      dim: -1
      dim: 256
      dim: 14
      dim: 14
    }
  }
  variable {
    name: "@net1/BatchNormalization_out_12"
    type: "Buffer"
    shape {
      dim: -1
      dim: 256
      dim: 14
      dim: 14
    }
  }
  variable {
    name: "@net1/Add2_out_4"
    type: "Buffer"
    shape {
      dim: -1
      dim: 256
      dim: 14
      dim: 14
    }
  }
  variable {
    name: "@net1/ReLU_out_10"
    type: "Buffer"
    shape {
      dim: -1
      dim: 256
      dim: 14
      dim: 14
    }
  }
  variable {
    name: "@net1/Convolution_out_13"
    type: "Buffer"
    shape {
      dim: -1
      dim: 256
      dim: 14
      dim: 14
    }
  }
  variable {
    name: "@net1/BatchNormalization_out_13"
    type: "Buffer"
    shape {
      dim: -1
      dim: 256
      dim: 14
      dim: 14
    }
  }
  variable {
    name: "@net1/ReLU_out_11"
    type: "Buffer"
    shape {
      dim: -1
      dim: 256
      dim: 14
      dim: 14
    }
  }
  variable {
    name: "@net1/Convolution_out_14"
    type: "Buffer"
    shape {
      dim: -1
      dim: 256
      dim: 14
      dim: 14
    }
  }
  variable {
    name: "@net1/BatchNormalization_out_14"
    type: "Buffer"
    shape {
      dim: -1
      dim: 256
      dim: 14
      dim: 14
    }
  }
  variable {
    name: "@net1/Add2_out_5"
    type: "Buffer"
    shape {
      dim: -1
      dim: 256
      dim: 14
      dim: 14
    }
  }
  variable {
    name: "@net1/ReLU_out_12"
    type: "Buffer"
    shape {
      dim: -1
      dim: 256
      dim: 14
      dim: 14
    }
  }
  variable {
    name: "@net1/Convolution_out_15"
    type: "Buffer"
    shape {
      dim: -1
      dim: 512
      dim: 7
      dim: 7
    }
  }
  variable {
    name: "@net1/BatchNormalization_out_15"
    type: "Buffer"
    shape {
      dim: -1
      dim: 512
      dim: 7
      dim: 7
    }
  }
  variable {
    name: "@net1/Convolution_out_16"
    type: "Buffer"
    shape {
      dim: -1
      dim: 512
      dim: 7
      dim: 7
    }
  }
  variable {
    name: "@net1/BatchNormalization_out_16"
    type: "Buffer"
    shape {
      dim: -1
      dim: 512
      dim: 7
      dim: 7
    }
  }
  variable {
    name: "@net1/ReLU_out_13"
    type: "Buffer"
    shape {
      dim: -1
      dim: 512
      dim: 7
      dim: 7
    }
  }
  variable {
    name: "@net1/Convolution_out_17"
    type: "Buffer"
    shape {
      dim: -1
      dim: 512
      dim: 7
      dim: 7
    }
  }
  variable {
    name: "@net1/BatchNormalization_out_17"
    type: "Buffer"
    shape {
      dim: -1
      dim: 512
      dim: 7
      dim: 7
    }
  }
  variable {
    name: "@net1/Add2_out_6"
    type: "Buffer"
    shape {
      dim: -1
      dim: 512
      dim: 7
      dim: 7
    }
  }
  variable {
    name: "@net1/ReLU_out_14"
    type: "Buffer"
    shape {
      dim: -1
      dim: 512
      dim: 7
      dim: 7
    }
  }
  variable {
    name: "@net1/Convolution_out_18"
    type: "Buffer"
    shape {
      dim: -1
      dim: 512
      dim: 7
      dim: 7
    }
  }
  variable {
    name: "@net1/BatchNormalization_out_18"
    type: "Buffer"
    shape {
      dim: -1
      dim: 512
      dim: 7
      dim: 7
    }
  }
  variable {
    name: "@net1/ReLU_out_15"
    type: "Buffer"
    shape {
      dim: -1
      dim: 512
      dim: 7
      dim: 7
    }
  }
  variable {
    name: "@net1/Convolution_out_19"
    type: "Buffer"
    shape {
      dim: -1
      dim: 512
      dim: 7
      dim: 7
    }
  }
  variable {
    name: "@net1/BatchNormalization_out_19"
    type: "Buffer"
    shape {
      dim: -1
      dim: 512
      dim: 7
      dim: 7
    }
  }
  variable {
    name: "@net1/Add2_out_7"
    type: "Buffer"
    shape {
      dim: -1
      dim: 512
      dim: 7
      dim: 7
    }
  }
  variable {
    name: "@net1/ReLU_out_16"
    type: "Buffer"
    shape {
      dim: -1
      dim: 512
      dim: 7
      dim: 7
    }
  }
  variable {
    name: "@net1/AveragePooling_out"
    type: "Buffer"
    shape {
      dim: -1
      dim: 512
      dim: 1
      dim: 1
    }
  }
  variable {
    name: "y"
    type: "Buffer"
    shape {
      dim: -1
      dim: 1000
    }
  }
  variable {
    name: "Convolution/conv/W"
    type: "Parameter"
    shape {
      dim: 64
      dim: 3
      dim: 7
      dim: 7
    }
  }
  variable {
    name: "BatchNormalization/bn/beta"
    type: "Parameter"
    shape {
      dim: 1
      dim: 64
      dim: 1
      dim: 1
    }
  }
  variable {
    name: "BatchNormalization/bn/gamma"
    type: "Parameter"
    shape {
      dim: 1
      dim: 64
      dim: 1
      dim: 1
    }
  }
  variable {
    name: "BatchNormalization/bn/mean"
    type: "Parameter"
    shape {
      dim: 1
      dim: 64
      dim: 1
      dim: 1
    }
  }
  variable {
    name: "BatchNormalization/bn/var"
    type: "Parameter"
    shape {
      dim: 1
      dim: 64
      dim: 1
      dim: 1
    }
  }
  variable {
    name: "Convolution_3/conv/W"
    type: "Parameter"
    shape {
      dim: 64
      dim: 64
      dim: 3
      dim: 3
    }
  }
  variable {
    name: "BatchNormalization_3/bn/beta"
    type: "Parameter"
    shape {
      dim: 1
      dim: 64
      dim: 1
      dim: 1
    }
  }
  variable {
    name: "BatchNormalization_3/bn/gamma"
    type: "Parameter"
    shape {
      dim: 1
      dim: 64
      dim: 1
      dim: 1
    }
  }
  variable {
    name: "BatchNormalization_3/bn/mean"
    type: "Parameter"
    shape {
      dim: 1
      dim: 64
      dim: 1
      dim: 1
    }
  }
  variable {
    name: "BatchNormalization_3/bn/var"
    type: "Parameter"
    shape {
      dim: 1
      dim: 64
      dim: 1
      dim: 1
    }
  }
  variable {
    name: "Convolution_4/conv/W"
    type: "Parameter"
    shape {
      dim: 64
      dim: 64
      dim: 3
      dim: 3
    }
  }
  variable {
    name: "BatchNormalization_4/bn/beta"
    type: "Parameter"
    shape {
      dim: 1
      dim: 64
      dim: 1
      dim: 1
    }
  }
  variable {
    name: "BatchNormalization_4/bn/gamma"
    type: "Parameter"
    shape {
      dim: 1
      dim: 64
      dim: 1
      dim: 1
    }
  }
  variable {
    name: "BatchNormalization_4/bn/mean"
    type: "Parameter"
    shape {
      dim: 1
      dim: 64
      dim: 1
      dim: 1
    }
  }
  variable {
    name: "BatchNormalization_4/bn/var"
    type: "Parameter"
    shape {
      dim: 1
      dim: 64
      dim: 1
      dim: 1
    }
  }
  variable {
    name: "Convolution_6[0]/conv/W"
    type: "Parameter"
    shape {
      dim: 64
      dim: 64
      dim: 3
      dim: 3
    }
  }
  variable {
    name: "BatchNormalization_6[0]/bn/beta"
    type: "Parameter"
    shape {
      dim: 1
      dim: 64
      dim: 1
      dim: 1
    }
  }
  variable {
    name: "BatchNormalization_6[0]/bn/gamma"
    type: "Parameter"
    shape {
      dim: 1
      dim: 64
      dim: 1
      dim: 1
    }
  }
  variable {
    name: "BatchNormalization_6[0]/bn/mean"
    type: "Parameter"
    shape {
      dim: 1
      dim: 64
      dim: 1
      dim: 1
    }
  }
  variable {
    name: "BatchNormalization_6[0]/bn/var"
    type: "Parameter"
    shape {
      dim: 1
      dim: 64
      dim: 1
      dim: 1
    }
  }
  variable {
    name: "Convolution_7[0]/conv/W"
    type: "Parameter"
    shape {
      dim: 64
      dim: 64
      dim: 3
      dim: 3
    }
  }
  variable {
    name: "BatchNormalization_7[0]/bn/beta"
    type: "Parameter"
    shape {
      dim: 1
      dim: 64
      dim: 1
      dim: 1
    }
  }
  variable {
    name: "BatchNormalization_7[0]/bn/gamma"
    type: "Parameter"
    shape {
      dim: 1
      dim: 64
      dim: 1
      dim: 1
    }
  }
  variable {
    name: "BatchNormalization_7[0]/bn/mean"
    type: "Parameter"
    shape {
      dim: 1
      dim: 64
      dim: 1
      dim: 1
    }
  }
  variable {
    name: "BatchNormalization_7[0]/bn/var"
    type: "Parameter"
    shape {
      dim: 1
      dim: 64
      dim: 1
      dim: 1
    }
  }
  variable {
    name: "Convolution_9/conv/W"
    type: "Parameter"
    shape {
      dim: 128
      dim: 64
      dim: 1
      dim: 1
    }
  }
  variable {
    name: "BatchNormalization_9/bn/beta"
    type: "Parameter"
    shape {
      dim: 1
      dim: 128
      dim: 1
      dim: 1
    }
  }
  variable {
    name: "BatchNormalization_9/bn/gamma"
    type: "Parameter"
    shape {
      dim: 1
      dim: 128
      dim: 1
      dim: 1
    }
  }
  variable {
    name: "BatchNormalization_9/bn/mean"
    type: "Parameter"
    shape {
      dim: 1
      dim: 128
      dim: 1
      dim: 1
    }
  }
  variable {
    name: "BatchNormalization_9/bn/var"
    type: "Parameter"
    shape {
      dim: 1
      dim: 128
      dim: 1
      dim: 1
    }
  }
  variable {
    name: "Convolution_10/conv/W"
    type: "Parameter"
    shape {
      dim: 128
      dim: 64
      dim: 3
      dim: 3
    }
  }
  variable {
    name: "BatchNormalization_10/bn/beta"
    type: "Parameter"
    shape {
      dim: 1
      dim: 128
      dim: 1
      dim: 1
    }
  }
  variable {
    name: "BatchNormalization_10/bn/gamma"
    type: "Parameter"
    shape {
      dim: 1
      dim: 128
      dim: 1
      dim: 1
    }
  }
  variable {
    name: "BatchNormalization_10/bn/mean"
    type: "Parameter"
    shape {
      dim: 1
      dim: 128
      dim: 1
      dim: 1
    }
  }
  variable {
    name: "BatchNormalization_10/bn/var"
    type: "Parameter"
    shape {
      dim: 1
      dim: 128
      dim: 1
      dim: 1
    }
  }
  variable {
    name: "Convolution_11/conv/W"
    type: "Parameter"
    shape {
      dim: 128
      dim: 128
      dim: 3
      dim: 3
    }
  }
  variable {
    name: "BatchNormalization_11/bn/beta"
    type: "Parameter"
    shape {
      dim: 1
      dim: 128
      dim: 1
      dim: 1
    }
  }
  variable {
    name: "BatchNormalization_11/bn/gamma"
    type: "Parameter"
    shape {
      dim: 1
      dim: 128
      dim: 1
      dim: 1
    }
  }
  variable {
    name: "BatchNormalization_11/bn/mean"
    type: "Parameter"
    shape {
      dim: 1
      dim: 128
      dim: 1
      dim: 1
    }
  }
  variable {
    name: "BatchNormalization_11/bn/var"
    type: "Parameter"
    shape {
      dim: 1
      dim: 128
      dim: 1
      dim: 1
    }
  }
  variable {
    name: "Convolution_13[0]/conv/W"
    type: "Parameter"
    shape {
      dim: 128
      dim: 128
      dim: 3
      dim: 3
    }
  }
  variable {
    name: "BatchNormalization_13[0]/bn/beta"
    type: "Parameter"
    shape {
      dim: 1
      dim: 128
      dim: 1
      dim: 1
    }
  }
  variable {
    name: "BatchNormalization_13[0]/bn/gamma"
    type: "Parameter"
    shape {
      dim: 1
      dim: 128
      dim: 1
      dim: 1
    }
  }
  variable {
    name: "BatchNormalization_13[0]/bn/mean"
    type: "Parameter"
    shape {
      dim: 1
      dim: 128
      dim: 1
      dim: 1
    }
  }
  variable {
    name: "BatchNormalization_13[0]/bn/var"
    type: "Parameter"
    shape {
      dim: 1
      dim: 128
      dim: 1
      dim: 1
    }
  }
  variable {
    name: "Convolution_14[0]/conv/W"
    type: "Parameter"
    shape {
      dim: 128
      dim: 128
      dim: 3
      dim: 3
    }
  }
  variable {
    name: "BatchNormalization_14[0]/bn/beta"
    type: "Parameter"
    shape {
      dim: 1
      dim: 128
      dim: 1
      dim: 1
    }
  }
  variable {
    name: "BatchNormalization_14[0]/bn/gamma"
    type: "Parameter"
    shape {
      dim: 1
      dim: 128
      dim: 1
      dim: 1
    }
  }
  variable {
    name: "BatchNormalization_14[0]/bn/mean"
    type: "Parameter"
    shape {
      dim: 1
      dim: 128
      dim: 1
      dim: 1
    }
  }
  variable {
    name: "BatchNormalization_14[0]/bn/var"
    type: "Parameter"
    shape {
      dim: 1
      dim: 128
      dim: 1
      dim: 1
    }
  }
  variable {
    name: "Convolution_16/conv/W"
    type: "Parameter"
    shape {
      dim: 256
      dim: 128
      dim: 1
      dim: 1
    }
  }
  variable {
    name: "BatchNormalization_16/bn/beta"
    type: "Parameter"
    shape {
      dim: 1
      dim: 256
      dim: 1
      dim: 1
    }
  }
  variable {
    name: "BatchNormalization_16/bn/gamma"
    type: "Parameter"
    shape {
      dim: 1
      dim: 256
      dim: 1
      dim: 1
    }
  }
  variable {
    name: "BatchNormalization_16/bn/mean"
    type: "Parameter"
    shape {
      dim: 1
      dim: 256
      dim: 1
      dim: 1
    }
  }
  variable {
    name: "BatchNormalization_16/bn/var"
    type: "Parameter"
    shape {
      dim: 1
      dim: 256
      dim: 1
      dim: 1
    }
  }
  variable {
    name: "Convolution_17/conv/W"
    type: "Parameter"
    shape {
      dim: 256
      dim: 128
      dim: 3
      dim: 3
    }
  }
  variable {
    name: "BatchNormalization_17/bn/beta"
    type: "Parameter"
    shape {
      dim: 1
      dim: 256
      dim: 1
      dim: 1
    }
  }
  variable {
    name: "BatchNormalization_17/bn/gamma"
    type: "Parameter"
    shape {
      dim: 1
      dim: 256
      dim: 1
      dim: 1
    }
  }
  variable {
    name: "BatchNormalization_17/bn/mean"
    type: "Parameter"
    shape {
      dim: 1
      dim: 256
      dim: 1
      dim: 1
    }
  }
  variable {
    name: "BatchNormalization_17/bn/var"
    type: "Parameter"
    shape {
      dim: 1
      dim: 256
      dim: 1
      dim: 1
    }
  }
  variable {
    name: "Convolution_18/conv/W"
    type: "Parameter"
    shape {
      dim: 256
      dim: 256
      dim: 3
      dim: 3
    }
  }
  variable {
    name: "BatchNormalization_18/bn/beta"
    type: "Parameter"
    shape {
      dim: 1
      dim: 256
      dim: 1
      dim: 1
    }
  }
  variable {
    name: "BatchNormalization_18/bn/gamma"
    type: "Parameter"
    shape {
      dim: 1
      dim: 256
      dim: 1
      dim: 1
    }
  }
  variable {
    name: "BatchNormalization_18/bn/mean"
    type: "Parameter"
    shape {
      dim: 1
      dim: 256
      dim: 1
      dim: 1
    }
  }
  variable {
    name: "BatchNormalization_18/bn/var"
    type: "Parameter"
    shape {
      dim: 1
      dim: 256
      dim: 1
      dim: 1
    }
  }
  variable {
    name: "Convolution_20[0]/conv/W"
    type: "Parameter"
    shape {
      dim: 256
      dim: 256
      dim: 3
      dim: 3
    }
  }
  variable {
    name: "BatchNormalization_20[0]/bn/beta"
    type: "Parameter"
    shape {
      dim: 1
      dim: 256
      dim: 1
      dim: 1
    }
  }
  variable {
    name: "BatchNormalization_20[0]/bn/gamma"
    type: "Parameter"
    shape {
      dim: 1
      dim: 256
      dim: 1
      dim: 1
    }
  }
  variable {
    name: "BatchNormalization_20[0]/bn/mean"
    type: "Parameter"
    shape {
      dim: 1
      dim: 256
      dim: 1
      dim: 1
    }
  }
  variable {
    name: "BatchNormalization_20[0]/bn/var"
    type: "Parameter"
    shape {
      dim: 1
      dim: 256
      dim: 1
      dim: 1
    }
  }
  variable {
    name: "Convolution_21[0]/conv/W"
    type: "Parameter"
    shape {
      dim: 256
      dim: 256
      dim: 3
      dim: 3
    }
  }
  variable {
    name: "BatchNormalization_21[0]/bn/beta"
    type: "Parameter"
    shape {
      dim: 1
      dim: 256
      dim: 1
      dim: 1
    }
  }
  variable {
    name: "BatchNormalization_21[0]/bn/gamma"
    type: "Parameter"
    shape {
      dim: 1
      dim: 256
      dim: 1
      dim: 1
    }
  }
  variable {
    name: "BatchNormalization_21[0]/bn/mean"
    type: "Parameter"
    shape {
      dim: 1
      dim: 256
      dim: 1
      dim: 1
    }
  }
  variable {
    name: "BatchNormalization_21[0]/bn/var"
    type: "Parameter"
    shape {
      dim: 1
      dim: 256
      dim: 1
      dim: 1
    }
  }
  variable {
    name: "Convolution_23/conv/W"
    type: "Parameter"
    shape {
      dim: 512
      dim: 256
      dim: 1
      dim: 1
    }
  }
  variable {
    name: "BatchNormalization_23/bn/beta"
    type: "Parameter"
    shape {
      dim: 1
      dim: 512
      dim: 1
      dim: 1
    }
  }
  variable {
    name: "BatchNormalization_23/bn/gamma"
    type: "Parameter"
    shape {
      dim: 1
      dim: 512
      dim: 1
      dim: 1
    }
  }
  variable {
    name: "BatchNormalization_23/bn/mean"
    type: "Parameter"
    shape {
      dim: 1
      dim: 512
      dim: 1
      dim: 1
    }
  }
  variable {
    name: "BatchNormalization_23/bn/var"
    type: "Parameter"
    shape {
      dim: 1
      dim: 512
      dim: 1
      dim: 1
    }
  }
  variable {
    name: "Convolution_24/conv/W"
    type: "Parameter"
    shape {
      dim: 512
      dim: 256
      dim: 3
      dim: 3
    }
  }
  variable {
    name: "BatchNormalization_24/bn/beta"
    type: "Parameter"
    shape {
      dim: 1
      dim: 512
      dim: 1
      dim: 1
    }
  }
  variable {
    name: "BatchNormalization_24/bn/gamma"
    type: "Parameter"
    shape {
      dim: 1
      dim: 512
      dim: 1
      dim: 1
    }
  }
  variable {
    name: "BatchNormalization_24/bn/mean"
    type: "Parameter"
    shape {
      dim: 1
      dim: 512
      dim: 1
      dim: 1
    }
  }
  variable {
    name: "BatchNormalization_24/bn/var"
    type: "Parameter"
    shape {
      dim: 1
      dim: 512
      dim: 1
      dim: 1
    }
  }
  variable {
    name: "Convolution_25/conv/W"
    type: "Parameter"
    shape {
      dim: 512
      dim: 512
      dim: 3
      dim: 3
    }
  }
  variable {
    name: "BatchNormalization_25/bn/beta"
    type: "Parameter"
    shape {
      dim: 1
      dim: 512
      dim: 1
      dim: 1
    }
  }
  variable {
    name: "BatchNormalization_25/bn/gamma"
    type: "Parameter"
    shape {
      dim: 1
      dim: 512
      dim: 1
      dim: 1
    }
  }
  variable {
    name: "BatchNormalization_25/bn/mean"
    type: "Parameter"
    shape {
      dim: 1
      dim: 512
      dim: 1
      dim: 1
    }
  }
  variable {
    name: "BatchNormalization_25/bn/var"
    type: "Parameter"
    shape {
      dim: 1
      dim: 512
      dim: 1
      dim: 1
    }
  }
  variable {
    name: "Convolution_27[0]/conv/W"
    type: "Parameter"
    shape {
      dim: 512
      dim: 512
      dim: 3
      dim: 3
    }
  }
  variable {
    name: "BatchNormalization_27[0]/bn/beta"
    type: "Parameter"
    shape {
      dim: 1
      dim: 512
      dim: 1
      dim: 1
    }
  }
  variable {
    name: "BatchNormalization_27[0]/bn/gamma"
    type: "Parameter"
    shape {
      dim: 1
      dim: 512
      dim: 1
      dim: 1
    }
  }
  variable {
    name: "BatchNormalization_27[0]/bn/mean"
    type: "Parameter"
    shape {
      dim: 1
      dim: 512
      dim: 1
      dim: 1
    }
  }
  variable {
    name: "BatchNormalization_27[0]/bn/var"
    type: "Parameter"
    shape {
      dim: 1
      dim: 512
      dim: 1
      dim: 1
    }
  }
  variable {
    name: "Convolution_28[0]/conv/W"
    type: "Parameter"
    shape {
      dim: 512
      dim: 512
      dim: 3
      dim: 3
    }
  }
  variable {
    name: "BatchNormalization_28[0]/bn/beta"
    type: "Parameter"
    shape {
      dim: 1
      dim: 512
      dim: 1
      dim: 1
    }
  }
  variable {
    name: "BatchNormalization_28[0]/bn/gamma"
    type: "Parameter"
    shape {
      dim: 1
      dim: 512
      dim: 1
      dim: 1
    }
  }
  variable {
    name: "BatchNormalization_28[0]/bn/mean"
    type: "Parameter"
    shape {
      dim: 1
      dim: 512
      dim: 1
      dim: 1
    }
  }
  variable {
    name: "BatchNormalization_28[0]/bn/var"
    type: "Parameter"
    shape {
      dim: 1
      dim: 512
      dim: 1
      dim: 1
    }
  }
  variable {
    name: "Affine/affine/W"
    type: "Parameter"
    shape {
      dim: 512
      dim: 1000
    }
  }
  variable {
    name: "Affine/affine/b"
    type: "Parameter"
    shape {
      dim: 1000
    }
  }
  function {
    name: "@net1/MulScalar"
    type: "MulScalar"
    input: "x"
    output: "@net1/MulScalar_out"
    mul_scalar_param {
      val: 0.01735
    }
  }
  function {
    name: "@net1/AddScalar"
    type: "AddScalar"
    input: "@net1/MulScalar_out"
    output: "@net1/AddScalar_out"
    add_scalar_param {
      val: -1.99
    }
  }
  function {
    name: "@net1/Convolution"
    type: "Convolution"
    input: "@net1/AddScalar_out"
    input: "Convolution/conv/W"
    output: "@net1/Convolution_out"
    convolution_param {
      base_axis: 1
      pad {
        dim: 3
        dim: 3
      }
      stride {
        dim: 2
        dim: 2
      }
      dilation {
        dim: 1
        dim: 1
      }
      group: 1
    }
  }
  function {
    name: "@net1/BatchNormalization"
    type: "BatchNormalization"
    input: "@net1/Convolution_out"
    input: "BatchNormalization/bn/beta"
    input: "BatchNormalization/bn/gamma"
    input: "BatchNormalization/bn/mean"
    input: "BatchNormalization/bn/var"
    output: "@net1/BatchNormalization_out"
    batch_normalization_param {
      axes: 1
      decay_rate: 0.9
      eps: 0.0001
    }
  }
  function {
    name: "@net1/ReLU"
    type: "ReLU"
    input: "@net1/BatchNormalization_out"
    output: "@net1/ReLU_out"
    relu_param {
      inplace: true
    }
  }
  function {
    name: "@net1/MaxPooling"
    type: "MaxPooling"
    input: "@net1/ReLU_out"
    output: "@net1/MaxPooling_out"
    max_pooling_param {
      kernel {
        dim: 3
        dim: 3
      }
      stride {
        dim: 2
        dim: 2
      }
      ignore_border: true
      pad {
        dim: 1
        dim: 1
      }
    }
  }
  function {
    name: "@net1/Convolution_1"
    type: "Convolution"
    input: "@net1/MaxPooling_out"
    input: "Convolution_3/conv/W"
    output: "@net1/Convolution_out_1"
    convolution_param {
      base_axis: 1
      pad {
        dim: 1
        dim: 1
      }
      stride {
        dim: 1
        dim: 1
      }
      dilation {
        dim: 1
        dim: 1
      }
      group: 1
    }
  }
  function {
    name: "@net1/BatchNormalization_1"
    type: "BatchNormalization"
    input: "@net1/Convolution_out_1"
    input: "BatchNormalization_3/bn/beta"
    input: "BatchNormalization_3/bn/gamma"
    input: "BatchNormalization_3/bn/mean"
    input: "BatchNormalization_3/bn/var"
    output: "@net1/BatchNormalization_out_1"
    batch_normalization_param {
      axes: 1
      decay_rate: 0.9
      eps: 0.0001
    }
  }
  function {
    name: "@net1/ReLU_1"
    type: "ReLU"
    input: "@net1/BatchNormalization_out_1"
    output: "@net1/ReLU_out_1"
    relu_param {
      inplace: true
    }
  }
  function {
    name: "@net1/Convolution_2"
    type: "Convolution"
    input: "@net1/ReLU_out_1"
    input: "Convolution_4/conv/W"
    output: "@net1/Convolution_out_2"
    convolution_param {
      base_axis: 1
      pad {
        dim: 1
        dim: 1
      }
      stride {
        dim: 1
        dim: 1
      }
      dilation {
        dim: 1
        dim: 1
      }
      group: 1
    }
  }
  function {
    name: "@net1/BatchNormalization_2"
    type: "BatchNormalization"
    input: "@net1/Convolution_out_2"
    input: "BatchNormalization_4/bn/beta"
    input: "BatchNormalization_4/bn/gamma"
    input: "BatchNormalization_4/bn/mean"
    input: "BatchNormalization_4/bn/var"
    output: "@net1/BatchNormalization_out_2"
    batch_normalization_param {
      axes: 1
      decay_rate: 0.9
      eps: 0.0001
    }
  }
  function {
    name: "@net1/Add2"
    type: "Add2"
    input: "@net1/MaxPooling_out"
    input: "@net1/BatchNormalization_out_2"
    output: "@net1/Add2_out"
    add2_param {
    }
  }
  function {
    name: "@net1/ReLU_2"
    type: "ReLU"
    input: "@net1/Add2_out"
    output: "@net1/ReLU_out_2"
    relu_param {
      inplace: true
    }
  }
  function {
    name: "@net1/Convolution_3"
    type: "Convolution"
    input: "@net1/ReLU_out_2"
    input: "Convolution_6[0]/conv/W"
    output: "@net1/Convolution_out_3"
    convolution_param {
      base_axis: 1
      pad {
        dim: 1
        dim: 1
      }
      stride {
        dim: 1
        dim: 1
      }
      dilation {
        dim: 1
        dim: 1
      }
      group: 1
    }
  }
  function {
    name: "@net1/BatchNormalization_3"
    type: "BatchNormalization"
    input: "@net1/Convolution_out_3"
    input: "BatchNormalization_6[0]/bn/beta"
    input: "BatchNormalization_6[0]/bn/gamma"
    input: "BatchNormalization_6[0]/bn/mean"
    input: "BatchNormalization_6[0]/bn/var"
    output: "@net1/BatchNormalization_out_3"
    batch_normalization_param {
      axes: 1
      decay_rate: 0.9
      eps: 0.0001
    }
  }
  function {
    name: "@net1/ReLU_3"
    type: "ReLU"
    input: "@net1/BatchNormalization_out_3"
    output: "@net1/ReLU_out_3"
    relu_param {
      inplace: true
    }
  }
  function {
    name: "@net1/Convolution_4"
    type: "Convolution"
    input: "@net1/ReLU_out_3"
    input: "Convolution_7[0]/conv/W"
    output: "@net1/Convolution_out_4"
    convolution_param {
      base_axis: 1
      pad {
        dim: 1
        dim: 1
      }
      stride {
        dim: 1
        dim: 1
      }
      dilation {
        dim: 1
        dim: 1
      }
      group: 1
    }
  }
  function {
    name: "@net1/BatchNormalization_4"
    type: "BatchNormalization"
    input: "@net1/Convolution_out_4"
    input: "BatchNormalization_7[0]/bn/beta"
    input: "BatchNormalization_7[0]/bn/gamma"
    input: "BatchNormalization_7[0]/bn/mean"
    input: "BatchNormalization_7[0]/bn/var"
    output: "@net1/BatchNormalization_out_4"
    batch_normalization_param {
      axes: 1
      decay_rate: 0.9
      eps: 0.0001
    }
  }
  function {
    name: "@net1/Add2_1"
    type: "Add2"
    input: "@net1/ReLU_out_2"
    input: "@net1/BatchNormalization_out_4"
    output: "@net1/Add2_out_1"
    add2_param {
    }
  }
  function {
    name: "@net1/ReLU_4"
    type: "ReLU"
    input: "@net1/Add2_out_1"
    output: "@net1/ReLU_out_4"
    relu_param {
      inplace: true
    }
  }
  function {
    name: "@net1/Convolution_5"
    type: "Convolution"
    input: "@net1/ReLU_out_4"
    input: "Convolution_9/conv/W"
    output: "@net1/Convolution_out_5"
    convolution_param {
      base_axis: 1
      pad {
        dim: 0
        dim: 0
      }
      stride {
        dim: 2
        dim: 2
      }
      dilation {
        dim: 1
        dim: 1
      }
      group: 1
    }
  }
  function {
    name: "@net1/BatchNormalization_5"
    type: "BatchNormalization"
    input: "@net1/Convolution_out_5"
    input: "BatchNormalization_9/bn/beta"
    input: "BatchNormalization_9/bn/gamma"
    input: "BatchNormalization_9/bn/mean"
    input: "BatchNormalization_9/bn/var"
    output: "@net1/BatchNormalization_out_5"
    batch_normalization_param {
      axes: 1
      decay_rate: 0.9
      eps: 0.0001
    }
  }
  function {
    name: "@net1/Convolution_6"
    type: "Convolution"
    input: "@net1/ReLU_out_4"
    input: "Convolution_10/conv/W"
    output: "@net1/Convolution_out_6"
    convolution_param {
      base_axis: 1
      pad {
        dim: 1
        dim: 1
      }
      stride {
        dim: 2
        dim: 2
      }
      dilation {
        dim: 1
        dim: 1
      }
      group: 1
    }
  }
  function {
    name: "@net1/BatchNormalization_6"
    type: "BatchNormalization"
    input: "@net1/Convolution_out_6"
    input: "BatchNormalization_10/bn/beta"
    input: "BatchNormalization_10/bn/gamma"
    input: "BatchNormalization_10/bn/mean"
    input: "BatchNormalization_10/bn/var"
    output: "@net1/BatchNormalization_out_6"
    batch_normalization_param {
      axes: 1
      decay_rate: 0.9
      eps: 0.0001
    }
  }
  function {
    name: "@net1/ReLU_5"
    type: "ReLU"
    input: "@net1/BatchNormalization_out_6"
    output: "@net1/ReLU_out_5"
    relu_param {
      inplace: true
    }
  }
  function {
    name: "@net1/Convolution_7"
    type: "Convolution"
    input: "@net1/ReLU_out_5"
    input: "Convolution_11/conv/W"
    output: "@net1/Convolution_out_7"
    convolution_param {
      base_axis: 1
      pad {
        dim: 1
        dim: 1
      }
      stride {
        dim: 1
        dim: 1
      }
      dilation {
        dim: 1
        dim: 1
      }
      group: 1
    }
  }
  function {
    name: "@net1/BatchNormalization_7"
    type: "BatchNormalization"
    input: "@net1/Convolution_out_7"
    input: "BatchNormalization_11/bn/beta"
    input: "BatchNormalization_11/bn/gamma"
    input: "BatchNormalization_11/bn/mean"
    input: "BatchNormalization_11/bn/var"
    output: "@net1/BatchNormalization_out_7"
    batch_normalization_param {
      axes: 1
      decay_rate: 0.9
      eps: 0.0001
    }
  }
  function {
    name: "@net1/Add2_2"
    type: "Add2"
    input: "@net1/BatchNormalization_out_5"
    input: "@net1/BatchNormalization_out_7"
    output: "@net1/Add2_out_2"
    add2_param {
      inplace: true
    }
  }
  function {
    name: "@net1/ReLU_6"
    type: "ReLU"
    input: "@net1/Add2_out_2"
    output: "@net1/ReLU_out_6"
    relu_param {
      inplace: true
    }
  }
  function {
    name: "@net1/Convolution_8"
    type: "Convolution"
    input: "@net1/ReLU_out_6"
    input: "Convolution_13[0]/conv/W"
    output: "@net1/Convolution_out_8"
    convolution_param {
      base_axis: 1
      pad {
        dim: 1
        dim: 1
      }
      stride {
        dim: 1
        dim: 1
      }
      dilation {
        dim: 1
        dim: 1
      }
      group: 1
    }
  }
  function {
    name: "@net1/BatchNormalization_8"
    type: "BatchNormalization"
    input: "@net1/Convolution_out_8"
    input: "BatchNormalization_13[0]/bn/beta"
    input: "BatchNormalization_13[0]/bn/gamma"
    input: "BatchNormalization_13[0]/bn/mean"
    input: "BatchNormalization_13[0]/bn/var"
    output: "@net1/BatchNormalization_out_8"
    batch_normalization_param {
      axes: 1
      decay_rate: 0.9
      eps: 0.0001
    }
  }
  function {
    name: "@net1/ReLU_7"
    type: "ReLU"
    input: "@net1/BatchNormalization_out_8"
    output: "@net1/ReLU_out_7"
    relu_param {
      inplace: true
    }
  }
  function {
    name: "@net1/Convolution_9"
    type: "Convolution"
    input: "@net1/ReLU_out_7"
    input: "Convolution_14[0]/conv/W"
    output: "@net1/Convolution_out_9"
    convolution_param {
      base_axis: 1
      pad {
        dim: 1
        dim: 1
      }
      stride {
        dim: 1
        dim: 1
      }
      dilation {
        dim: 1
        dim: 1
      }
      group: 1
    }
  }
  function {
    name: "@net1/BatchNormalization_9"
    type: "BatchNormalization"
    input: "@net1/Convolution_out_9"
    input: "BatchNormalization_14[0]/bn/beta"
    input: "BatchNormalization_14[0]/bn/gamma"
    input: "BatchNormalization_14[0]/bn/mean"
    input: "BatchNormalization_14[0]/bn/var"
    output: "@net1/BatchNormalization_out_9"
    batch_normalization_param {
      axes: 1
      decay_rate: 0.9
      eps: 0.0001
    }
  }
  function {
    name: "@net1/Add2_3"
    type: "Add2"
    input: "@net1/ReLU_out_6"
    input: "@net1/BatchNormalization_out_9"
    output: "@net1/Add2_out_3"
    add2_param {
    }
  }
  function {
    name: "@net1/ReLU_8"
    type: "ReLU"
    input: "@net1/Add2_out_3"
    output: "@net1/ReLU_out_8"
    relu_param {
      inplace: true
    }
  }
  function {
    name: "@net1/Convolution_10"
    type: "Convolution"
    input: "@net1/ReLU_out_8"
    input: "Convolution_16/conv/W"
    output: "@net1/Convolution_out_10"
    convolution_param {
      base_axis: 1
      pad {
        dim: 0
        dim: 0
      }
      stride {
        dim: 2
        dim: 2
      }
      dilation {
        dim: 1
        dim: 1
      }
      group: 1
    }
  }
  function {
    name: "@net1/BatchNormalization_10"
    type: "BatchNormalization"
    input: "@net1/Convolution_out_10"
    input: "BatchNormalization_16/bn/beta"
    input: "BatchNormalization_16/bn/gamma"
    input: "BatchNormalization_16/bn/mean"
    input: "BatchNormalization_16/bn/var"
    output: "@net1/BatchNormalization_out_10"
    batch_normalization_param {
      axes: 1
      decay_rate: 0.9
      eps: 0.0001
    }
  }
  function {
    name: "@net1/Convolution_11"
    type: "Convolution"
    input: "@net1/ReLU_out_8"
    input: "Convolution_17/conv/W"
    output: "@net1/Convolution_out_11"
    convolution_param {
      base_axis: 1
      pad {
        dim: 1
        dim: 1
      }
      stride {
        dim: 2
        dim: 2
      }
      dilation {
        dim: 1
        dim: 1
      }
      group: 1
    }
  }
  function {
    name: "@net1/BatchNormalization_11"
    type: "BatchNormalization"
    input: "@net1/Convolution_out_11"
    input: "BatchNormalization_17/bn/beta"
    input: "BatchNormalization_17/bn/gamma"
    input: "BatchNormalization_17/bn/mean"
    input: "BatchNormalization_17/bn/var"
    output: "@net1/BatchNormalization_out_11"
    batch_normalization_param {
      axes: 1
      decay_rate: 0.9
      eps: 0.0001
    }
  }
  function {
    name: "@net1/ReLU_9"
    type: "ReLU"
    input: "@net1/BatchNormalization_out_11"
    output: "@net1/ReLU_out_9"
    relu_param {
      inplace: true
    }
  }
  function {
    name: "@net1/Convolution_12"
    type: "Convolution"
    input: "@net1/ReLU_out_9"
    input: "Convolution_18/conv/W"
    output: "@net1/Convolution_out_12"
    convolution_param {
      base_axis: 1
      pad {
        dim: 1
        dim: 1
      }
      stride {
        dim: 1
        dim: 1
      }
      dilation {
        dim: 1
        dim: 1
      }
      group: 1
    }
  }
  function {
    name: "@net1/BatchNormalization_12"
    type: "BatchNormalization"
    input: "@net1/Convolution_out_12"
    input: "BatchNormalization_18/bn/beta"
    input: "BatchNormalization_18/bn/gamma"
    input: "BatchNormalization_18/bn/mean"
    input: "BatchNormalization_18/bn/var"
    output: "@net1/BatchNormalization_out_12"
    batch_normalization_param {
      axes: 1
      decay_rate: 0.9
      eps: 0.0001
    }
  }
  function {
    name: "@net1/Add2_4"
    type: "Add2"
    input: "@net1/BatchNormalization_out_10"
    input: "@net1/BatchNormalization_out_12"
    output: "@net1/Add2_out_4"
    add2_param {
      inplace: true
    }
  }
  function {
    name: "@net1/ReLU_10"
    type: "ReLU"
    input: "@net1/Add2_out_4"
    output: "@net1/ReLU_out_10"
    relu_param {
      inplace: true
    }
  }
  function {
    name: "@net1/Convolution_13"
    type: "Convolution"
    input: "@net1/ReLU_out_10"
    input: "Convolution_20[0]/conv/W"
    output: "@net1/Convolution_out_13"
    convolution_param {
      base_axis: 1
      pad {
        dim: 1
        dim: 1
      }
      stride {
        dim: 1
        dim: 1
      }
      dilation {
        dim: 1
        dim: 1
      }
      group: 1
    }
  }
  function {
    name: "@net1/BatchNormalization_13"
    type: "BatchNormalization"
    input: "@net1/Convolution_out_13"
    input: "BatchNormalization_20[0]/bn/beta"
    input: "BatchNormalization_20[0]/bn/gamma"
    input: "BatchNormalization_20[0]/bn/mean"
    input: "BatchNormalization_20[0]/bn/var"
    output: "@net1/BatchNormalization_out_13"
    batch_normalization_param {
      axes: 1
      decay_rate: 0.9
      eps: 0.0001
    }
  }
  function {
    name: "@net1/ReLU_11"
    type: "ReLU"
    input: "@net1/BatchNormalization_out_13"
    output: "@net1/ReLU_out_11"
    relu_param {
      inplace: true
    }
  }
  function {
    name: "@net1/Convolution_14"
    type: "Convolution"
    input: "@net1/ReLU_out_11"
    input: "Convolution_21[0]/conv/W"
    output: "@net1/Convolution_out_14"
    convolution_param {
      base_axis: 1
      pad {
        dim: 1
        dim: 1
      }
      stride {
        dim: 1
        dim: 1
      }
      dilation {
        dim: 1
        dim: 1
      }
      group: 1
    }
  }
  function {
    name: "@net1/BatchNormalization_14"
    type: "BatchNormalization"
    input: "@net1/Convolution_out_14"
    input: "BatchNormalization_21[0]/bn/beta"
    input: "BatchNormalization_21[0]/bn/gamma"
    input: "BatchNormalization_21[0]/bn/mean"
    input: "BatchNormalization_21[0]/bn/var"
    output: "@net1/BatchNormalization_out_14"
    batch_normalization_param {
      axes: 1
      decay_rate: 0.9
      eps: 0.0001
    }
  }
  function {
    name: "@net1/Add2_5"
    type: "Add2"
    input: "@net1/ReLU_out_10"
    input: "@net1/BatchNormalization_out_14"
    output: "@net1/Add2_out_5"
    add2_param {
    }
  }
  function {
    name: "@net1/ReLU_12"
    type: "ReLU"
    input: "@net1/Add2_out_5"
    output: "@net1/ReLU_out_12"
    relu_param {
      inplace: true
    }
  }
  function {
    name: "@net1/Convolution_15"
    type: "Convolution"
    input: "@net1/ReLU_out_12"
    input: "Convolution_23/conv/W"
    output: "@net1/Convolution_out_15"
    convolution_param {
      base_axis: 1
      pad {
        dim: 0
        dim: 0
      }
      stride {
        dim: 2
        dim: 2
      }
      dilation {
        dim: 1
        dim: 1
      }
      group: 1
    }
  }
  function {
    name: "@net1/BatchNormalization_15"
    type: "BatchNormalization"
    input: "@net1/Convolution_out_15"
    input: "BatchNormalization_23/bn/beta"
    input: "BatchNormalization_23/bn/gamma"
    input: "BatchNormalization_23/bn/mean"
    input: "BatchNormalization_23/bn/var"
    output: "@net1/BatchNormalization_out_15"
    batch_normalization_param {
      axes: 1
      decay_rate: 0.9
      eps: 0.0001
    }
  }
  function {
    name: "@net1/Convolution_16"
    type: "Convolution"
    input: "@net1/ReLU_out_12"
    input: "Convolution_24/conv/W"
    output: "@net1/Convolution_out_16"
    convolution_param {
      base_axis: 1
      pad {
        dim: 1
        dim: 1
      }
      stride {
        dim: 2
        dim: 2
      }
      dilation {
        dim: 1
        dim: 1
      }
      group: 1
    }
  }
  function {
    name: "@net1/BatchNormalization_16"
    type: "BatchNormalization"
    input: "@net1/Convolution_out_16"
    input: "BatchNormalization_24/bn/beta"
    input: "BatchNormalization_24/bn/gamma"
    input: "BatchNormalization_24/bn/mean"
    input: "BatchNormalization_24/bn/var"
    output: "@net1/BatchNormalization_out_16"
    batch_normalization_param {
      axes: 1
      decay_rate: 0.9
      eps: 0.0001
    }
  }
  function {
    name: "@net1/ReLU_13"
    type: "ReLU"
    input: "@net1/BatchNormalization_out_16"
    output: "@net1/ReLU_out_13"
    relu_param {
      inplace: true
    }
  }
  function {
    name: "@net1/Convolution_17"
    type: "Convolution"
    input: "@net1/ReLU_out_13"
    input: "Convolution_25/conv/W"
    output: "@net1/Convolution_out_17"
    convolution_param {
      base_axis: 1
      pad {
        dim: 1
        dim: 1
      }
      stride {
        dim: 1
        dim: 1
      }
      dilation {
        dim: 1
        dim: 1
      }
      group: 1
    }
  }
  function {
    name: "@net1/BatchNormalization_17"
    type: "BatchNormalization"
    input: "@net1/Convolution_out_17"
    input: "BatchNormalization_25/bn/beta"
    input: "BatchNormalization_25/bn/gamma"
    input: "BatchNormalization_25/bn/mean"
    input: "BatchNormalization_25/bn/var"
    output: "@net1/BatchNormalization_out_17"
    batch_normalization_param {
      axes: 1
      decay_rate: 0.9
      eps: 0.0001
    }
  }
  function {
    name: "@net1/Add2_6"
    type: "Add2"
    input: "@net1/BatchNormalization_out_15"
    input: "@net1/BatchNormalization_out_17"
    output: "@net1/Add2_out_6"
    add2_param {
      inplace: true
    }
  }
  function {
    name: "@net1/ReLU_14"
    type: "ReLU"
    input: "@net1/Add2_out_6"
    output: "@net1/ReLU_out_14"
    relu_param {
      inplace: true
    }
  }
  function {
    name: "@net1/Convolution_18"
    type: "Convolution"
    input: "@net1/ReLU_out_14"
    input: "Convolution_27[0]/conv/W"
    output: "@net1/Convolution_out_18"
    convolution_param {
      base_axis: 1
      pad {
        dim: 1
        dim: 1
      }
      stride {
        dim: 1
        dim: 1
      }
      dilation {
        dim: 1
        dim: 1
      }
      group: 1
    }
  }
  function {
    name: "@net1/BatchNormalization_18"
    type: "BatchNormalization"
    input: "@net1/Convolution_out_18"
    input: "BatchNormalization_27[0]/bn/beta"
    input: "BatchNormalization_27[0]/bn/gamma"
    input: "BatchNormalization_27[0]/bn/mean"
    input: "BatchNormalization_27[0]/bn/var"
    output: "@net1/BatchNormalization_out_18"
    batch_normalization_param {
      axes: 1
      decay_rate: 0.9
      eps: 0.0001
    }
  }
  function {
    name: "@net1/ReLU_15"
    type: "ReLU"
    input: "@net1/BatchNormalization_out_18"
    output: "@net1/ReLU_out_15"
    relu_param {
      inplace: true
    }
  }
  function {
    name: "@net1/Convolution_19"
    type: "Convolution"
    input: "@net1/ReLU_out_15"
    input: "Convolution_28[0]/conv/W"
    output: "@net1/Convolution_out_19"
    convolution_param {
      base_axis: 1
      pad {
        dim: 1
        dim: 1
      }
      stride {
        dim: 1
        dim: 1
      }
      dilation {
        dim: 1
        dim: 1
      }
      group: 1
    }
  }
  function {
    name: "@net1/BatchNormalization_19"
    type: "BatchNormalization"
    input: "@net1/Convolution_out_19"
    input: "BatchNormalization_28[0]/bn/beta"
    input: "BatchNormalization_28[0]/bn/gamma"
    input: "BatchNormalization_28[0]/bn/mean"
    input: "BatchNormalization_28[0]/bn/var"
    output: "@net1/BatchNormalization_out_19"
    batch_normalization_param {
      axes: 1
      decay_rate: 0.9
      eps: 0.0001
    }
  }
  function {
    name: "@net1/Add2_7"
    type: "Add2"
    input: "@net1/ReLU_out_14"
    input: "@net1/BatchNormalization_out_19"
    output: "@net1/Add2_out_7"
    add2_param {
    }
  }
  function {
    name: "@net1/ReLU_16"
    type: "ReLU"
    input: "@net1/Add2_out_7"
    output: "@net1/ReLU_out_16"
    relu_param {
      inplace: true
    }
  }
  function {
    name: "@net1/AveragePooling"
    type: "AveragePooling"
    input: "@net1/ReLU_out_16"
    output: "@net1/AveragePooling_out"
    average_pooling_param {
      kernel {
        dim: 7
        dim: 7
      }
      stride {
        dim: 7
        dim: 7
      }
      pad {
        dim: 0
        dim: 0
      }
      including_pad: true
    }
  }
  function {
    name: "@net1/Affine"
    type: "Affine"
    input: "@net1/AveragePooling_out"
    input: "Affine/affine/W"
    input: "Affine/affine/b"
    output: "y"
    affine_param {
      base_axis: 1
    }
  }
}
executor {
  name: "runtime"
  network_name: "net1"
  data_variable {
    variable_name: "x"
    data_name: "x"
  }
  output_variable {
    variable_name: "y"
    data_name: "y"
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
    variable_name: "Convolution_6[0]/conv/W"
  }
  parameter_variable {
    variable_name: "BatchNormalization_6[0]/bn/beta"
  }
  parameter_variable {
    variable_name: "BatchNormalization_6[0]/bn/gamma"
  }
  parameter_variable {
    variable_name: "BatchNormalization_6[0]/bn/mean"
  }
  parameter_variable {
    variable_name: "BatchNormalization_6[0]/bn/var"
  }
  parameter_variable {
    variable_name: "Convolution_7[0]/conv/W"
  }
  parameter_variable {
    variable_name: "BatchNormalization_7[0]/bn/beta"
  }
  parameter_variable {
    variable_name: "BatchNormalization_7[0]/bn/gamma"
  }
  parameter_variable {
    variable_name: "BatchNormalization_7[0]/bn/mean"
  }
  parameter_variable {
    variable_name: "BatchNormalization_7[0]/bn/var"
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
    variable_name: "Convolution_13[0]/conv/W"
  }
  parameter_variable {
    variable_name: "BatchNormalization_13[0]/bn/beta"
  }
  parameter_variable {
    variable_name: "BatchNormalization_13[0]/bn/gamma"
  }
  parameter_variable {
    variable_name: "BatchNormalization_13[0]/bn/mean"
  }
  parameter_variable {
    variable_name: "BatchNormalization_13[0]/bn/var"
  }
  parameter_variable {
    variable_name: "Convolution_14[0]/conv/W"
  }
  parameter_variable {
    variable_name: "BatchNormalization_14[0]/bn/beta"
  }
  parameter_variable {
    variable_name: "BatchNormalization_14[0]/bn/gamma"
  }
  parameter_variable {
    variable_name: "BatchNormalization_14[0]/bn/mean"
  }
  parameter_variable {
    variable_name: "BatchNormalization_14[0]/bn/var"
  }
  parameter_variable {
    variable_name: "Convolution_16/conv/W"
  }
  parameter_variable {
    variable_name: "BatchNormalization_16/bn/beta"
  }
  parameter_variable {
    variable_name: "BatchNormalization_16/bn/gamma"
  }
  parameter_variable {
    variable_name: "BatchNormalization_16/bn/mean"
  }
  parameter_variable {
    variable_name: "BatchNormalization_16/bn/var"
  }
  parameter_variable {
    variable_name: "Convolution_17/conv/W"
  }
  parameter_variable {
    variable_name: "BatchNormalization_17/bn/beta"
  }
  parameter_variable {
    variable_name: "BatchNormalization_17/bn/gamma"
  }
  parameter_variable {
    variable_name: "BatchNormalization_17/bn/mean"
  }
  parameter_variable {
    variable_name: "BatchNormalization_17/bn/var"
  }
  parameter_variable {
    variable_name: "Convolution_18/conv/W"
  }
  parameter_variable {
    variable_name: "BatchNormalization_18/bn/beta"
  }
  parameter_variable {
    variable_name: "BatchNormalization_18/bn/gamma"
  }
  parameter_variable {
    variable_name: "BatchNormalization_18/bn/mean"
  }
  parameter_variable {
    variable_name: "BatchNormalization_18/bn/var"
  }
  parameter_variable {
    variable_name: "Convolution_20[0]/conv/W"
  }
  parameter_variable {
    variable_name: "BatchNormalization_20[0]/bn/beta"
  }
  parameter_variable {
    variable_name: "BatchNormalization_20[0]/bn/gamma"
  }
  parameter_variable {
    variable_name: "BatchNormalization_20[0]/bn/mean"
  }
  parameter_variable {
    variable_name: "BatchNormalization_20[0]/bn/var"
  }
  parameter_variable {
    variable_name: "Convolution_21[0]/conv/W"
  }
  parameter_variable {
    variable_name: "BatchNormalization_21[0]/bn/beta"
  }
  parameter_variable {
    variable_name: "BatchNormalization_21[0]/bn/gamma"
  }
  parameter_variable {
    variable_name: "BatchNormalization_21[0]/bn/mean"
  }
  parameter_variable {
    variable_name: "BatchNormalization_21[0]/bn/var"
  }
  parameter_variable {
    variable_name: "Convolution_23/conv/W"
  }
  parameter_variable {
    variable_name: "BatchNormalization_23/bn/beta"
  }
  parameter_variable {
    variable_name: "BatchNormalization_23/bn/gamma"
  }
  parameter_variable {
    variable_name: "BatchNormalization_23/bn/mean"
  }
  parameter_variable {
    variable_name: "BatchNormalization_23/bn/var"
  }
  parameter_variable {
    variable_name: "Convolution_24/conv/W"
  }
  parameter_variable {
    variable_name: "BatchNormalization_24/bn/beta"
  }
  parameter_variable {
    variable_name: "BatchNormalization_24/bn/gamma"
  }
  parameter_variable {
    variable_name: "BatchNormalization_24/bn/mean"
  }
  parameter_variable {
    variable_name: "BatchNormalization_24/bn/var"
  }
  parameter_variable {
    variable_name: "Convolution_25/conv/W"
  }
  parameter_variable {
    variable_name: "BatchNormalization_25/bn/beta"
  }
  parameter_variable {
    variable_name: "BatchNormalization_25/bn/gamma"
  }
  parameter_variable {
    variable_name: "BatchNormalization_25/bn/mean"
  }
  parameter_variable {
    variable_name: "BatchNormalization_25/bn/var"
  }
  parameter_variable {
    variable_name: "Convolution_27[0]/conv/W"
  }
  parameter_variable {
    variable_name: "BatchNormalization_27[0]/bn/beta"
  }
  parameter_variable {
    variable_name: "BatchNormalization_27[0]/bn/gamma"
  }
  parameter_variable {
    variable_name: "BatchNormalization_27[0]/bn/mean"
  }
  parameter_variable {
    variable_name: "BatchNormalization_27[0]/bn/var"
  }
  parameter_variable {
    variable_name: "Convolution_28[0]/conv/W"
  }
  parameter_variable {
    variable_name: "BatchNormalization_28[0]/bn/beta"
  }
  parameter_variable {
    variable_name: "BatchNormalization_28[0]/bn/gamma"
  }
  parameter_variable {
    variable_name: "BatchNormalization_28[0]/bn/mean"
  }
  parameter_variable {
    variable_name: "BatchNormalization_28[0]/bn/var"
  }
  parameter_variable {
    variable_name: "Affine/affine/W"
  }
  parameter_variable {
    variable_name: "Affine/affine/b"
  }
}
'''


N0005 = r'''
global_config {
  default_context {
    backends: "cpu:float"
    array_class: "CpuCachedArray"
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
    shape {
      dim: -1
      dim: 1
      dim: 28
      dim: 28
    }
  }
  variable {
    name: "Affine/affine/W"
    type: "Parameter"
    shape {
      dim: 784
      dim: 1
    }
    initializer {
      type: "NormalAffineGlorot"
      multiplier: 1.0
    }
  }
  variable {
    name: "Affine/affine/b"
    type: "Parameter"
    shape {
      dim: 1
    }
    initializer {
      type: "Constant"
    }
  }
  variable {
    name: "BinaryCrossEntropy_T"
    type: "Buffer"
    shape {
      dim: -1
      dim: 1
    }
  }
  variable {
    name: "Affine"
    type: "Buffer"
    shape {
      dim: -1
      dim: 1
    }
  }
  variable {
    name: "Sigmoid"
    type: "Buffer"
    shape {
      dim: -1
      dim: 1
    }
  }
  variable {
    name: "BinaryCrossEntropy"
    type: "Buffer"
    shape {
      dim: -1
      dim: 1
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
    shape {
      dim: -1
      dim: 1
      dim: 28
      dim: 28
    }
  }
  variable {
    name: "Affine/affine/W"
    type: "Parameter"
    shape {
      dim: 784
      dim: 1
    }
    initializer {
      type: "NormalAffineGlorot"
      multiplier: 1.0
    }
  }
  variable {
    name: "Affine/affine/b"
    type: "Parameter"
    shape {
      dim: 1
    }
    initializer {
      type: "Constant"
    }
  }
  variable {
    name: "BinaryCrossEntropy_T"
    type: "Buffer"
    shape {
      dim: -1
      dim: 1
    }
  }
  variable {
    name: "Affine"
    type: "Buffer"
    shape {
      dim: -1
      dim: 1
    }
  }
  variable {
    name: "Sigmoid"
    type: "Buffer"
    shape {
      dim: -1
      dim: 1
    }
  }
  variable {
    name: "BinaryCrossEntropy"
    type: "Buffer"
    shape {
      dim: -1
      dim: 1
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
    shape {
      dim: -1
      dim: 1
      dim: 28
      dim: 28
    }
  }
  variable {
    name: "Affine/affine/W"
    type: "Parameter"
    shape {
      dim: 784
      dim: 1
    }
    initializer {
      type: "NormalAffineGlorot"
      multiplier: 1.0
    }
  }
  variable {
    name: "Affine/affine/b"
    type: "Parameter"
    shape {
      dim: 1
    }
    initializer {
      type: "Constant"
    }
  }
  variable {
    name: "Affine"
    type: "Buffer"
    shape {
      dim: -1
      dim: 1
    }
  }
  variable {
    name: "Sigmoid"
    type: "Buffer"
    shape {
      dim: -1
      dim: 1
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
  uri: "C:\\Users\\woody\\Downloads\\re\\neural_network_console\\samples\\sample_dataset\\mnist\\small_mnist_4or9_training.csv"
  batch_size: 64
  cache_dir: "C:\\Users\\woody\\Downloads\\re\\neural_network_console\\samples\\sample_dataset\\mnist\\small_mnist_4or9_training.cache"
  overwrite_cache: true
  create_cache_explicitly: true
  shuffle: true
}
dataset {
  name: "Validation"
  uri: "C:\\Users\\woody\\Downloads\\re\\neural_network_console\\samples\\sample_dataset\\mnist\\small_mnist_4or9_test.csv"
  batch_size: 64
  cache_dir: "C:\\Users\\woody\\Downloads\\re\\neural_network_console\\samples\\sample_dataset\\mnist\\small_mnist_4or9_test.cache"
  overwrite_cache: true
  create_cache_explicitly: true
}
optimizer {
  name: "Optimizer"
  network_name: "Main"
  dataset_name: "Training"
  solver {
    type: "Adam"
    lr_decay: 1.0
    lr_decay_interval: 1
    adam_param {
      alpha: 0.001
      beta1: 0.9
      beta2: 0.999
      eps: 1e-08
    }
    lr_scheduler_type: "Exponential"
    exponential_scheduler_param {
      gamma: 1.0
      iter_interval: 1
    }
    lr_warmup_scheduler_type: "None"
  }
  update_interval: 1
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
    variable_name: "BinaryCrossEntropy"
    type: "Error"
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
    variable_name: "BinaryCrossEntropy"
    type: "Error"
  }
}
executor {
  name: "Executor"
  network_name: "MainRuntime"
  num_evaluations: 1
  repeat_evaluation_type: "mean"
  data_variable {
    variable_name: "Input"
    data_name: "x"
  }
  output_variable {
    variable_name: "Sigmoid"
    data_name: "y\'"
  }
  parameter_variable {
    variable_name: "Affine/affine/W"
  }
  parameter_variable {
    variable_name: "Affine/affine/b"
  }
}
'''
