# Copyright 2021 Sony Group Corporation.
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
import pytest
import nnabla as nn
from helper import create_temp_with_dir
from nnabla.utils import nnp_graph


nntxt = r'''
network {
  name: "classifier@{\"class_names\": [\"cat\", \"dog\"]}"
  batch_size: 1
  variable {
    name: "image"
    type: "Buffer"
    shape {
      dim: -1
      dim: 3
      dim: 112
      dim: 144
    }
  }
  variable {
    name: "@classifier@{\"class_names\": [\"cat\", \"dog\"]}/Convolution_out"
    type: "Buffer"
    shape {
      dim: -1
      dim: 16
      dim: 56
      dim: 72
    }
  }
  variable {
    name: "@classifier@{\"class_names\": [\"cat\", \"dog\"]}/BatchNormalization_out"
    type: "Buffer"
    shape {
      dim: -1
      dim: 16
      dim: 56
      dim: 72
    }
  }
  variable {
    name: "@classifier@{\"class_names\": [\"cat\", \"dog\"]}/LeakyReLU_out"
    type: "Buffer"
    shape {
      dim: -1
      dim: 16
      dim: 56
      dim: 72
    }
  }
  variable {
    name: "@classifier@{\"class_names\": [\"cat\", \"dog\"]}/Convolution_out_1"
    type: "Buffer"
    shape {
      dim: -1
      dim: 24
      dim: 28
      dim: 36
    }
  }
  variable {
    name: "@classifier@{\"class_names\": [\"cat\", \"dog\"]}/BatchNormalization_out_1"
    type: "Buffer"
    shape {
      dim: -1
      dim: 24
      dim: 28
      dim: 36
    }
  }
  variable {
    name: "@classifier@{\"class_names\": [\"cat\", \"dog\"]}/LeakyReLU_out_1"
    type: "Buffer"
    shape {
      dim: -1
      dim: 24
      dim: 28
      dim: 36
    }
  }
  variable {
    name: "@classifier@{\"class_names\": [\"cat\", \"dog\"]}/Convolution_out_2"
    type: "Buffer"
    shape {
      dim: -1
      dim: 24
      dim: 28
      dim: 36
    }
  }
  variable {
    name: "@classifier@{\"class_names\": [\"cat\", \"dog\"]}/BatchNormalization_out_2"
    type: "Buffer"
    shape {
      dim: -1
      dim: 24
      dim: 28
      dim: 36
    }
  }
  variable {
    name: "@classifier@{\"class_names\": [\"cat\", \"dog\"]}/LeakyReLU_out_2"
    type: "Buffer"
    shape {
      dim: -1
      dim: 24
      dim: 28
      dim: 36
    }
  }
  variable {
    name: "@classifier@{\"class_names\": [\"cat\", \"dog\"]}/Convolution_out_3"
    type: "Buffer"
    shape {
      dim: -1
      dim: 24
      dim: 28
      dim: 36
    }
  }
  variable {
    name: "@classifier@{\"class_names\": [\"cat\", \"dog\"]}/BatchNormalization_out_3"
    type: "Buffer"
    shape {
      dim: -1
      dim: 24
      dim: 28
      dim: 36
    }
  }
  variable {
    name: "@classifier@{\"class_names\": [\"cat\", \"dog\"]}/LeakyReLU_out_3"
    type: "Buffer"
    shape {
      dim: -1
      dim: 24
      dim: 28
      dim: 36
    }
  }
  variable {
    name: "@classifier@{\"class_names\": [\"cat\", \"dog\"]}/Convolution_out_4"
    type: "Buffer"
    shape {
      dim: -1
      dim: 32
      dim: 28
      dim: 36
    }
  }
  variable {
    name: "@classifier@{\"class_names\": [\"cat\", \"dog\"]}/BatchNormalization_out_4"
    type: "Buffer"
    shape {
      dim: -1
      dim: 32
      dim: 28
      dim: 36
    }
  }
  variable {
    name: "@classifier@{\"class_names\": [\"cat\", \"dog\"]}/LeakyReLU_out_4"
    type: "Buffer"
    shape {
      dim: -1
      dim: 32
      dim: 28
      dim: 36
    }
  }
  variable {
    name: "@classifier@{\"class_names\": [\"cat\", \"dog\"]}/Convolution_out_5"
    type: "Buffer"
    shape {
      dim: -1
      dim: 32
      dim: 28
      dim: 36
    }
  }
  variable {
    name: "@classifier@{\"class_names\": [\"cat\", \"dog\"]}/BatchNormalization_out_5"
    type: "Buffer"
    shape {
      dim: -1
      dim: 32
      dim: 28
      dim: 36
    }
  }
  variable {
    name: "@classifier@{\"class_names\": [\"cat\", \"dog\"]}/LeakyReLU_out_5"
    type: "Buffer"
    shape {
      dim: -1
      dim: 32
      dim: 28
      dim: 36
    }
  }
  variable {
    name: "@classifier@{\"class_names\": [\"cat\", \"dog\"]}/MaxPooling_out"
    type: "Buffer"
    shape {
      dim: -1
      dim: 32
      dim: 14
      dim: 18
    }
  }
  variable {
    name: "@classifier@{\"class_names\": [\"cat\", \"dog\"]}/Convolution_out_6"
    type: "Buffer"
    shape {
      dim: -1
      dim: 48
      dim: 14
      dim: 18
    }
  }
  variable {
    name: "@classifier@{\"class_names\": [\"cat\", \"dog\"]}/BatchNormalization_out_6"
    type: "Buffer"
    shape {
      dim: -1
      dim: 48
      dim: 14
      dim: 18
    }
  }
  variable {
    name: "@classifier@{\"class_names\": [\"cat\", \"dog\"]}/LeakyReLU_out_6"
    type: "Buffer"
    shape {
      dim: -1
      dim: 48
      dim: 14
      dim: 18
    }
  }
  variable {
    name: "@classifier@{\"class_names\": [\"cat\", \"dog\"]}/Convolution_out_7"
    type: "Buffer"
    shape {
      dim: -1
      dim: 48
      dim: 14
      dim: 18
    }
  }
  variable {
    name: "@classifier@{\"class_names\": [\"cat\", \"dog\"]}/BatchNormalization_out_7"
    type: "Buffer"
    shape {
      dim: -1
      dim: 48
      dim: 14
      dim: 18
    }
  }
  variable {
    name: "@classifier@{\"class_names\": [\"cat\", \"dog\"]}/LeakyReLU_out_7"
    type: "Buffer"
    shape {
      dim: -1
      dim: 48
      dim: 14
      dim: 18
    }
  }
  variable {
    name: "@classifier@{\"class_names\": [\"cat\", \"dog\"]}/MaxPooling_out_1"
    type: "Buffer"
    shape {
      dim: -1
      dim: 48
      dim: 7
      dim: 9
    }
  }
  variable {
    name: "@classifier@{\"class_names\": [\"cat\", \"dog\"]}/Affine_out"
    type: "Buffer"
    shape {
      dim: -1
      dim: 2
    }
  }
  variable {
    name: "pred"
    type: "Buffer"
    shape {
      dim: -1
      dim: 2
    }
  }
  variable {
    name: "blk_1_gw_conv_1/conv/W"
    type: "Parameter"
    shape {
      dim: 16
      dim: 3
      dim: 3
      dim: 3
    }
  }
  variable {
    name: "blk_1_gw_conv_1/conv/b"
    type: "Parameter"
    shape {
      dim: 16
    }
  }
  variable {
    name: "blk_1_gw_bn_1/bn/beta"
    type: "Parameter"
    shape {
      dim: 1
      dim: 16
      dim: 1
      dim: 1
    }
  }
  variable {
    name: "blk_1_gw_bn_1/bn/gamma"
    type: "Parameter"
    shape {
      dim: 1
      dim: 16
      dim: 1
      dim: 1
    }
  }
  variable {
    name: "blk_1_gw_bn_1/bn/mean"
    type: "Parameter"
    shape {
      dim: 1
      dim: 16
      dim: 1
      dim: 1
    }
  }
  variable {
    name: "blk_1_gw_bn_1/bn/var"
    type: "Parameter"
    shape {
      dim: 1
      dim: 16
      dim: 1
      dim: 1
    }
  }
  variable {
    name: "blk_2_gw_conv_1/conv/W"
    type: "Parameter"
    shape {
      dim: 24
      dim: 16
      dim: 3
      dim: 3
    }
  }
  variable {
    name: "blk_2_gw_conv_1/conv/b"
    type: "Parameter"
    shape {
      dim: 24
    }
  }
  variable {
    name: "blk_2_gw_bn_1/bn/beta"
    type: "Parameter"
    shape {
      dim: 1
      dim: 24
      dim: 1
      dim: 1
    }
  }
  variable {
    name: "blk_2_gw_bn_1/bn/gamma"
    type: "Parameter"
    shape {
      dim: 1
      dim: 24
      dim: 1
      dim: 1
    }
  }
  variable {
    name: "blk_2_gw_bn_1/bn/mean"
    type: "Parameter"
    shape {
      dim: 1
      dim: 24
      dim: 1
      dim: 1
    }
  }
  variable {
    name: "blk_2_gw_bn_1/bn/var"
    type: "Parameter"
    shape {
      dim: 1
      dim: 24
      dim: 1
      dim: 1
    }
  }
  variable {
    name: "blk_2_gw_conv_2/conv/W"
    type: "Parameter"
    shape {
      dim: 24
      dim: 24
      dim: 3
      dim: 3
    }
  }
  variable {
    name: "blk_2_gw_conv_2/conv/b"
    type: "Parameter"
    shape {
      dim: 24
    }
  }
  variable {
    name: "blk_2_gw_bn_2/bn/beta"
    type: "Parameter"
    shape {
      dim: 1
      dim: 24
      dim: 1
      dim: 1
    }
  }
  variable {
    name: "blk_2_gw_bn_2/bn/gamma"
    type: "Parameter"
    shape {
      dim: 1
      dim: 24
      dim: 1
      dim: 1
    }
  }
  variable {
    name: "blk_2_gw_bn_2/bn/mean"
    type: "Parameter"
    shape {
      dim: 1
      dim: 24
      dim: 1
      dim: 1
    }
  }
  variable {
    name: "blk_2_gw_bn_2/bn/var"
    type: "Parameter"
    shape {
      dim: 1
      dim: 24
      dim: 1
      dim: 1
    }
  }
  variable {
    name: "blk_2_gw_conv_3/conv/W"
    type: "Parameter"
    shape {
      dim: 24
      dim: 24
      dim: 3
      dim: 3
    }
  }
  variable {
    name: "blk_2_gw_conv_3/conv/b"
    type: "Parameter"
    shape {
      dim: 24
    }
  }
  variable {
    name: "blk_2_gw_bn_3/bn/beta"
    type: "Parameter"
    shape {
      dim: 1
      dim: 24
      dim: 1
      dim: 1
    }
  }
  variable {
    name: "blk_2_gw_bn_3/bn/gamma"
    type: "Parameter"
    shape {
      dim: 1
      dim: 24
      dim: 1
      dim: 1
    }
  }
  variable {
    name: "blk_2_gw_bn_3/bn/mean"
    type: "Parameter"
    shape {
      dim: 1
      dim: 24
      dim: 1
      dim: 1
    }
  }
  variable {
    name: "blk_2_gw_bn_3/bn/var"
    type: "Parameter"
    shape {
      dim: 1
      dim: 24
      dim: 1
      dim: 1
    }
  }
  variable {
    name: "blk_3_gw_conv_1/conv/W"
    type: "Parameter"
    shape {
      dim: 32
      dim: 24
      dim: 3
      dim: 3
    }
  }
  variable {
    name: "blk_3_gw_conv_1/conv/b"
    type: "Parameter"
    shape {
      dim: 32
    }
  }
  variable {
    name: "blk_3_gw_bn_1/bn/beta"
    type: "Parameter"
    shape {
      dim: 1
      dim: 32
      dim: 1
      dim: 1
    }
  }
  variable {
    name: "blk_3_gw_bn_1/bn/gamma"
    type: "Parameter"
    shape {
      dim: 1
      dim: 32
      dim: 1
      dim: 1
    }
  }
  variable {
    name: "blk_3_gw_bn_1/bn/mean"
    type: "Parameter"
    shape {
      dim: 1
      dim: 32
      dim: 1
      dim: 1
    }
  }
  variable {
    name: "blk_3_gw_bn_1/bn/var"
    type: "Parameter"
    shape {
      dim: 1
      dim: 32
      dim: 1
      dim: 1
    }
  }
  variable {
    name: "blk_3_gw_conv_2/conv/W"
    type: "Parameter"
    shape {
      dim: 32
      dim: 32
      dim: 3
      dim: 3
    }
  }
  variable {
    name: "blk_3_gw_conv_2/conv/b"
    type: "Parameter"
    shape {
      dim: 32
    }
  }
  variable {
    name: "blk_3_gw_bn_2/bn/beta"
    type: "Parameter"
    shape {
      dim: 1
      dim: 32
      dim: 1
      dim: 1
    }
  }
  variable {
    name: "blk_3_gw_bn_2/bn/gamma"
    type: "Parameter"
    shape {
      dim: 1
      dim: 32
      dim: 1
      dim: 1
    }
  }
  variable {
    name: "blk_3_gw_bn_2/bn/mean"
    type: "Parameter"
    shape {
      dim: 1
      dim: 32
      dim: 1
      dim: 1
    }
  }
  variable {
    name: "blk_3_gw_bn_2/bn/var"
    type: "Parameter"
    shape {
      dim: 1
      dim: 32
      dim: 1
      dim: 1
    }
  }
  variable {
    name: "blk_4_gw_conv_1/conv/W"
    type: "Parameter"
    shape {
      dim: 48
      dim: 32
      dim: 3
      dim: 3
    }
  }
  variable {
    name: "blk_4_gw_conv_1/conv/b"
    type: "Parameter"
    shape {
      dim: 48
    }
  }
  variable {
    name: "blk_4_gw_bn_1/bn/beta"
    type: "Parameter"
    shape {
      dim: 1
      dim: 48
      dim: 1
      dim: 1
    }
  }
  variable {
    name: "blk_4_gw_bn_1/bn/gamma"
    type: "Parameter"
    shape {
      dim: 1
      dim: 48
      dim: 1
      dim: 1
    }
  }
  variable {
    name: "blk_4_gw_bn_1/bn/mean"
    type: "Parameter"
    shape {
      dim: 1
      dim: 48
      dim: 1
      dim: 1
    }
  }
  variable {
    name: "blk_4_gw_bn_1/bn/var"
    type: "Parameter"
    shape {
      dim: 1
      dim: 48
      dim: 1
      dim: 1
    }
  }
  variable {
    name: "blk_4_gw_conv_2/conv/W"
    type: "Parameter"
    shape {
      dim: 48
      dim: 48
      dim: 3
      dim: 3
    }
  }
  variable {
    name: "blk_4_gw_conv_2/conv/b"
    type: "Parameter"
    shape {
      dim: 48
    }
  }
  variable {
    name: "blk_4_gw_bn_2/bn/beta"
    type: "Parameter"
    shape {
      dim: 1
      dim: 48
      dim: 1
      dim: 1
    }
  }
  variable {
    name: "blk_4_gw_bn_2/bn/gamma"
    type: "Parameter"
    shape {
      dim: 1
      dim: 48
      dim: 1
      dim: 1
    }
  }
  variable {
    name: "blk_4_gw_bn_2/bn/mean"
    type: "Parameter"
    shape {
      dim: 1
      dim: 48
      dim: 1
      dim: 1
    }
  }
  variable {
    name: "blk_4_gw_bn_2/bn/var"
    type: "Parameter"
    shape {
      dim: 1
      dim: 48
      dim: 1
      dim: 1
    }
  }
  variable {
    name: "affine/W"
    type: "Parameter"
    shape {
      dim: 3024
      dim: 2
    }
  }
  variable {
    name: "affine/b"
    type: "Parameter"
    shape {
      dim: 2
    }
  }
  function {
    name: "@classifier@{\"class_names\": [\"cat\", \"dog\"]}/Convolution"
    type: "Convolution"
    input: "image"
    input: "blk_1_gw_conv_1/conv/W"
    input: "blk_1_gw_conv_1/conv/b"
    output: "@classifier@{\"class_names\": [\"cat\", \"dog\"]}/Convolution_out"
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
    name: "@classifier@{\"class_names\": [\"cat\", \"dog\"]}/BatchNormalization"
    type: "BatchNormalization"
    input: "@classifier@{\"class_names\": [\"cat\", \"dog\"]}/Convolution_out"
    input: "blk_1_gw_bn_1/bn/beta"
    input: "blk_1_gw_bn_1/bn/gamma"
    input: "blk_1_gw_bn_1/bn/mean"
    input: "blk_1_gw_bn_1/bn/var"
    output: "@classifier@{\"class_names\": [\"cat\", \"dog\"]}/BatchNormalization_out"
    batch_normalization_param {
      axes: 1
      decay_rate: 0.9
      eps: 1e-05
    }
  }
  function {
    name: "@classifier@{\"class_names\": [\"cat\", \"dog\"]}/LeakyReLU"
    type: "LeakyReLU"
    input: "@classifier@{\"class_names\": [\"cat\", \"dog\"]}/BatchNormalization_out"
    output: "@classifier@{\"class_names\": [\"cat\", \"dog\"]}/LeakyReLU_out"
    leaky_relu_param {
      alpha: 0.1
    }
  }
  function {
    name: "@classifier@{\"class_names\": [\"cat\", \"dog\"]}/Convolution_1"
    type: "Convolution"
    input: "@classifier@{\"class_names\": [\"cat\", \"dog\"]}/LeakyReLU_out"
    input: "blk_2_gw_conv_1/conv/W"
    input: "blk_2_gw_conv_1/conv/b"
    output: "@classifier@{\"class_names\": [\"cat\", \"dog\"]}/Convolution_out_1"
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
    name: "@classifier@{\"class_names\": [\"cat\", \"dog\"]}/BatchNormalization_1"
    type: "BatchNormalization"
    input: "@classifier@{\"class_names\": [\"cat\", \"dog\"]}/Convolution_out_1"
    input: "blk_2_gw_bn_1/bn/beta"
    input: "blk_2_gw_bn_1/bn/gamma"
    input: "blk_2_gw_bn_1/bn/mean"
    input: "blk_2_gw_bn_1/bn/var"
    output: "@classifier@{\"class_names\": [\"cat\", \"dog\"]}/BatchNormalization_out_1"
    batch_normalization_param {
      axes: 1
      decay_rate: 0.9
      eps: 1e-05
    }
  }
  function {
    name: "@classifier@{\"class_names\": [\"cat\", \"dog\"]}/LeakyReLU_1"
    type: "LeakyReLU"
    input: "@classifier@{\"class_names\": [\"cat\", \"dog\"]}/BatchNormalization_out_1"
    output: "@classifier@{\"class_names\": [\"cat\", \"dog\"]}/LeakyReLU_out_1"
    leaky_relu_param {
      alpha: 0.1
    }
  }
  function {
    name: "@classifier@{\"class_names\": [\"cat\", \"dog\"]}/Convolution_2"
    type: "Convolution"
    input: "@classifier@{\"class_names\": [\"cat\", \"dog\"]}/LeakyReLU_out_1"
    input: "blk_2_gw_conv_2/conv/W"
    input: "blk_2_gw_conv_2/conv/b"
    output: "@classifier@{\"class_names\": [\"cat\", \"dog\"]}/Convolution_out_2"
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
    name: "@classifier@{\"class_names\": [\"cat\", \"dog\"]}/BatchNormalization_2"
    type: "BatchNormalization"
    input: "@classifier@{\"class_names\": [\"cat\", \"dog\"]}/Convolution_out_2"
    input: "blk_2_gw_bn_2/bn/beta"
    input: "blk_2_gw_bn_2/bn/gamma"
    input: "blk_2_gw_bn_2/bn/mean"
    input: "blk_2_gw_bn_2/bn/var"
    output: "@classifier@{\"class_names\": [\"cat\", \"dog\"]}/BatchNormalization_out_2"
    batch_normalization_param {
      axes: 1
      decay_rate: 0.9
      eps: 1e-05
    }
  }
  function {
    name: "@classifier@{\"class_names\": [\"cat\", \"dog\"]}/LeakyReLU_2"
    type: "LeakyReLU"
    input: "@classifier@{\"class_names\": [\"cat\", \"dog\"]}/BatchNormalization_out_2"
    output: "@classifier@{\"class_names\": [\"cat\", \"dog\"]}/LeakyReLU_out_2"
    leaky_relu_param {
      alpha: 0.1
    }
  }
  function {
    name: "@classifier@{\"class_names\": [\"cat\", \"dog\"]}/Convolution_3"
    type: "Convolution"
    input: "@classifier@{\"class_names\": [\"cat\", \"dog\"]}/LeakyReLU_out_2"
    input: "blk_2_gw_conv_3/conv/W"
    input: "blk_2_gw_conv_3/conv/b"
    output: "@classifier@{\"class_names\": [\"cat\", \"dog\"]}/Convolution_out_3"
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
    name: "@classifier@{\"class_names\": [\"cat\", \"dog\"]}/BatchNormalization_3"
    type: "BatchNormalization"
    input: "@classifier@{\"class_names\": [\"cat\", \"dog\"]}/Convolution_out_3"
    input: "blk_2_gw_bn_3/bn/beta"
    input: "blk_2_gw_bn_3/bn/gamma"
    input: "blk_2_gw_bn_3/bn/mean"
    input: "blk_2_gw_bn_3/bn/var"
    output: "@classifier@{\"class_names\": [\"cat\", \"dog\"]}/BatchNormalization_out_3"
    batch_normalization_param {
      axes: 1
      decay_rate: 0.9
      eps: 1e-05
    }
  }
  function {
    name: "@classifier@{\"class_names\": [\"cat\", \"dog\"]}/LeakyReLU_3"
    type: "LeakyReLU"
    input: "@classifier@{\"class_names\": [\"cat\", \"dog\"]}/BatchNormalization_out_3"
    output: "@classifier@{\"class_names\": [\"cat\", \"dog\"]}/LeakyReLU_out_3"
    leaky_relu_param {
      alpha: 0.1
    }
  }
  function {
    name: "@classifier@{\"class_names\": [\"cat\", \"dog\"]}/Convolution_4"
    type: "Convolution"
    input: "@classifier@{\"class_names\": [\"cat\", \"dog\"]}/LeakyReLU_out_3"
    input: "blk_3_gw_conv_1/conv/W"
    input: "blk_3_gw_conv_1/conv/b"
    output: "@classifier@{\"class_names\": [\"cat\", \"dog\"]}/Convolution_out_4"
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
    name: "@classifier@{\"class_names\": [\"cat\", \"dog\"]}/BatchNormalization_4"
    type: "BatchNormalization"
    input: "@classifier@{\"class_names\": [\"cat\", \"dog\"]}/Convolution_out_4"
    input: "blk_3_gw_bn_1/bn/beta"
    input: "blk_3_gw_bn_1/bn/gamma"
    input: "blk_3_gw_bn_1/bn/mean"
    input: "blk_3_gw_bn_1/bn/var"
    output: "@classifier@{\"class_names\": [\"cat\", \"dog\"]}/BatchNormalization_out_4"
    batch_normalization_param {
      axes: 1
      decay_rate: 0.9
      eps: 1e-05
    }
  }
  function {
    name: "@classifier@{\"class_names\": [\"cat\", \"dog\"]}/LeakyReLU_4"
    type: "LeakyReLU"
    input: "@classifier@{\"class_names\": [\"cat\", \"dog\"]}/BatchNormalization_out_4"
    output: "@classifier@{\"class_names\": [\"cat\", \"dog\"]}/LeakyReLU_out_4"
    leaky_relu_param {
      alpha: 0.1
    }
  }
  function {
    name: "@classifier@{\"class_names\": [\"cat\", \"dog\"]}/Convolution_5"
    type: "Convolution"
    input: "@classifier@{\"class_names\": [\"cat\", \"dog\"]}/LeakyReLU_out_4"
    input: "blk_3_gw_conv_2/conv/W"
    input: "blk_3_gw_conv_2/conv/b"
    output: "@classifier@{\"class_names\": [\"cat\", \"dog\"]}/Convolution_out_5"
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
    name: "@classifier@{\"class_names\": [\"cat\", \"dog\"]}/BatchNormalization_5"
    type: "BatchNormalization"
    input: "@classifier@{\"class_names\": [\"cat\", \"dog\"]}/Convolution_out_5"
    input: "blk_3_gw_bn_2/bn/beta"
    input: "blk_3_gw_bn_2/bn/gamma"
    input: "blk_3_gw_bn_2/bn/mean"
    input: "blk_3_gw_bn_2/bn/var"
    output: "@classifier@{\"class_names\": [\"cat\", \"dog\"]}/BatchNormalization_out_5"
    batch_normalization_param {
      axes: 1
      decay_rate: 0.9
      eps: 1e-05
    }
  }
  function {
    name: "@classifier@{\"class_names\": [\"cat\", \"dog\"]}/LeakyReLU_5"
    type: "LeakyReLU"
    input: "@classifier@{\"class_names\": [\"cat\", \"dog\"]}/BatchNormalization_out_5"
    output: "@classifier@{\"class_names\": [\"cat\", \"dog\"]}/LeakyReLU_out_5"
    leaky_relu_param {
      alpha: 0.1
    }
  }
  function {
    name: "@classifier@{\"class_names\": [\"cat\", \"dog\"]}/MaxPooling"
    type: "MaxPooling"
    input: "@classifier@{\"class_names\": [\"cat\", \"dog\"]}/LeakyReLU_out_5"
    output: "@classifier@{\"class_names\": [\"cat\", \"dog\"]}/MaxPooling_out"
    max_pooling_param {
      kernel {
        dim: 2
        dim: 2
      }
      stride {
        dim: 2
        dim: 2
      }
      ignore_border: true
      pad {
        dim: 0
        dim: 0
      }
    }
  }
  function {
    name: "@classifier@{\"class_names\": [\"cat\", \"dog\"]}/Convolution_6"
    type: "Convolution"
    input: "@classifier@{\"class_names\": [\"cat\", \"dog\"]}/MaxPooling_out"
    input: "blk_4_gw_conv_1/conv/W"
    input: "blk_4_gw_conv_1/conv/b"
    output: "@classifier@{\"class_names\": [\"cat\", \"dog\"]}/Convolution_out_6"
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
    name: "@classifier@{\"class_names\": [\"cat\", \"dog\"]}/BatchNormalization_6"
    type: "BatchNormalization"
    input: "@classifier@{\"class_names\": [\"cat\", \"dog\"]}/Convolution_out_6"
    input: "blk_4_gw_bn_1/bn/beta"
    input: "blk_4_gw_bn_1/bn/gamma"
    input: "blk_4_gw_bn_1/bn/mean"
    input: "blk_4_gw_bn_1/bn/var"
    output: "@classifier@{\"class_names\": [\"cat\", \"dog\"]}/BatchNormalization_out_6"
    batch_normalization_param {
      axes: 1
      decay_rate: 0.9
      eps: 1e-05
    }
  }
  function {
    name: "@classifier@{\"class_names\": [\"cat\", \"dog\"]}/LeakyReLU_6"
    type: "LeakyReLU"
    input: "@classifier@{\"class_names\": [\"cat\", \"dog\"]}/BatchNormalization_out_6"
    output: "@classifier@{\"class_names\": [\"cat\", \"dog\"]}/LeakyReLU_out_6"
    leaky_relu_param {
      alpha: 0.1
    }
  }
  function {
    name: "@classifier@{\"class_names\": [\"cat\", \"dog\"]}/Convolution_7"
    type: "Convolution"
    input: "@classifier@{\"class_names\": [\"cat\", \"dog\"]}/LeakyReLU_out_6"
    input: "blk_4_gw_conv_2/conv/W"
    input: "blk_4_gw_conv_2/conv/b"
    output: "@classifier@{\"class_names\": [\"cat\", \"dog\"]}/Convolution_out_7"
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
    name: "@classifier@{\"class_names\": [\"cat\", \"dog\"]}/BatchNormalization_7"
    type: "BatchNormalization"
    input: "@classifier@{\"class_names\": [\"cat\", \"dog\"]}/Convolution_out_7"
    input: "blk_4_gw_bn_2/bn/beta"
    input: "blk_4_gw_bn_2/bn/gamma"
    input: "blk_4_gw_bn_2/bn/mean"
    input: "blk_4_gw_bn_2/bn/var"
    output: "@classifier@{\"class_names\": [\"cat\", \"dog\"]}/BatchNormalization_out_7"
    batch_normalization_param {
      axes: 1
      decay_rate: 0.9
      eps: 1e-05
    }
  }
  function {
    name: "@classifier@{\"class_names\": [\"cat\", \"dog\"]}/LeakyReLU_7"
    type: "LeakyReLU"
    input: "@classifier@{\"class_names\": [\"cat\", \"dog\"]}/BatchNormalization_out_7"
    output: "@classifier@{\"class_names\": [\"cat\", \"dog\"]}/LeakyReLU_out_7"
    leaky_relu_param {
      alpha: 0.1
    }
  }
  function {
    name: "@classifier@{\"class_names\": [\"cat\", \"dog\"]}/MaxPooling_1"
    type: "MaxPooling"
    input: "@classifier@{\"class_names\": [\"cat\", \"dog\"]}/LeakyReLU_out_7"
    output: "@classifier@{\"class_names\": [\"cat\", \"dog\"]}/MaxPooling_out_1"
    max_pooling_param {
      kernel {
        dim: 2
        dim: 2
      }
      stride {
        dim: 2
        dim: 2
      }
      ignore_border: true
      pad {
        dim: 0
        dim: 0
      }
    }
  }
  function {
    name: "@classifier@{\"class_names\": [\"cat\", \"dog\"]}/Affine"
    type: "Affine"
    input: "@classifier@{\"class_names\": [\"cat\", \"dog\"]}/MaxPooling_out_1"
    input: "affine/W"
    input: "affine/b"
    output: "@classifier@{\"class_names\": [\"cat\", \"dog\"]}/Affine_out"
    affine_param {
      base_axis: 1
    }
  }
  function {
    name: "@classifier@{\"class_names\": [\"cat\", \"dog\"]}/Softmax"
    type: "Softmax"
    input: "@classifier@{\"class_names\": [\"cat\", \"dog\"]}/Affine_out"
    output: "pred"
    softmax_param {
      axis: 1
    }
  }
}
executor {
  name: "runtime"
  network_name: "classifier@{\"class_names\": [\"cat\", \"dog\"]}"
  data_variable {
    variable_name: "image"
    data_name: "image"
  }
  output_variable {
    variable_name: "pred"
    data_name: "pred"
  }
  parameter_variable {
    variable_name: "blk_1_gw_conv_1/conv/W"
  }
  parameter_variable {
    variable_name: "blk_1_gw_conv_1/conv/b"
  }
  parameter_variable {
    variable_name: "blk_1_gw_bn_1/bn/beta"
  }
  parameter_variable {
    variable_name: "blk_1_gw_bn_1/bn/gamma"
  }
  parameter_variable {
    variable_name: "blk_1_gw_bn_1/bn/mean"
  }
  parameter_variable {
    variable_name: "blk_1_gw_bn_1/bn/var"
  }
  parameter_variable {
    variable_name: "blk_2_gw_conv_1/conv/W"
  }
  parameter_variable {
    variable_name: "blk_2_gw_conv_1/conv/b"
  }
  parameter_variable {
    variable_name: "blk_2_gw_bn_1/bn/beta"
  }
  parameter_variable {
    variable_name: "blk_2_gw_bn_1/bn/gamma"
  }
  parameter_variable {
    variable_name: "blk_2_gw_bn_1/bn/mean"
  }
  parameter_variable {
    variable_name: "blk_2_gw_bn_1/bn/var"
  }
  parameter_variable {
    variable_name: "blk_2_gw_conv_2/conv/W"
  }
  parameter_variable {
    variable_name: "blk_2_gw_conv_2/conv/b"
  }
  parameter_variable {
    variable_name: "blk_2_gw_bn_2/bn/beta"
  }
  parameter_variable {
    variable_name: "blk_2_gw_bn_2/bn/gamma"
  }
  parameter_variable {
    variable_name: "blk_2_gw_bn_2/bn/mean"
  }
  parameter_variable {
    variable_name: "blk_2_gw_bn_2/bn/var"
  }
  parameter_variable {
    variable_name: "blk_2_gw_conv_3/conv/W"
  }
  parameter_variable {
    variable_name: "blk_2_gw_conv_3/conv/b"
  }
  parameter_variable {
    variable_name: "blk_2_gw_bn_3/bn/beta"
  }
  parameter_variable {
    variable_name: "blk_2_gw_bn_3/bn/gamma"
  }
  parameter_variable {
    variable_name: "blk_2_gw_bn_3/bn/mean"
  }
  parameter_variable {
    variable_name: "blk_2_gw_bn_3/bn/var"
  }
  parameter_variable {
    variable_name: "blk_3_gw_conv_1/conv/W"
  }
  parameter_variable {
    variable_name: "blk_3_gw_conv_1/conv/b"
  }
  parameter_variable {
    variable_name: "blk_3_gw_bn_1/bn/beta"
  }
  parameter_variable {
    variable_name: "blk_3_gw_bn_1/bn/gamma"
  }
  parameter_variable {
    variable_name: "blk_3_gw_bn_1/bn/mean"
  }
  parameter_variable {
    variable_name: "blk_3_gw_bn_1/bn/var"
  }
  parameter_variable {
    variable_name: "blk_3_gw_conv_2/conv/W"
  }
  parameter_variable {
    variable_name: "blk_3_gw_conv_2/conv/b"
  }
  parameter_variable {
    variable_name: "blk_3_gw_bn_2/bn/beta"
  }
  parameter_variable {
    variable_name: "blk_3_gw_bn_2/bn/gamma"
  }
  parameter_variable {
    variable_name: "blk_3_gw_bn_2/bn/mean"
  }
  parameter_variable {
    variable_name: "blk_3_gw_bn_2/bn/var"
  }
  parameter_variable {
    variable_name: "blk_4_gw_conv_1/conv/W"
  }
  parameter_variable {
    variable_name: "blk_4_gw_conv_1/conv/b"
  }
  parameter_variable {
    variable_name: "blk_4_gw_bn_1/bn/beta"
  }
  parameter_variable {
    variable_name: "blk_4_gw_bn_1/bn/gamma"
  }
  parameter_variable {
    variable_name: "blk_4_gw_bn_1/bn/mean"
  }
  parameter_variable {
    variable_name: "blk_4_gw_bn_1/bn/var"
  }
  parameter_variable {
    variable_name: "blk_4_gw_conv_2/conv/W"
  }
  parameter_variable {
    variable_name: "blk_4_gw_conv_2/conv/b"
  }
  parameter_variable {
    variable_name: "blk_4_gw_bn_2/bn/beta"
  }
  parameter_variable {
    variable_name: "blk_4_gw_bn_2/bn/gamma"
  }
  parameter_variable {
    variable_name: "blk_4_gw_bn_2/bn/mean"
  }
  parameter_variable {
    variable_name: "blk_4_gw_bn_2/bn/var"
  }
  parameter_variable {
    variable_name: "affine/W"
  }
  parameter_variable {
    variable_name: "affine/b"
  }
}
'''

# github issue #698


def test_load_twice():
    from nnabla.ext_utils import list_extensions, get_extension_context
    exts = list_extensions()
    if "cudnn" not in exts:
        pytest.skip("This test is only for cudnn context!")

    with create_temp_with_dir("network.nntxt") as fn:
        with open(fn, "w") as f:
            f.write(nntxt)
        ctx = get_extension_context('cudnn')
        nn.set_default_context(ctx)
        nnp = nnp_graph.NnpLoader(fn)
        for network_name in sorted(nnp.get_network_names()):
            print(network_name)
            network = nnp.get_network(
                network_name, batch_size=32)

        for network_name in sorted(nnp.get_network_names()):
            print(network_name)
            network = nnp.get_network(
                network_name, batch_size=32)


nntxt_mixup = r'''
network {
  name: "SetData"
  batch_size: 128
  variable {
    name: "InputX"
    type: "Buffer"
    shape: { dim:-1 dim: 3 dim: 32 dim: 32 }
  }
  variable {
    name: "InputY"
    type: "Buffer"
    shape: { dim:-1 dim: 1 }
  }
  variable {
    name: "Y2"
    type: "Buffer"
    shape: { dim:-1 dim: 1 }
    initializer {
      type: "Normal"
      multiplier: 1
    }
  }
  variable {
    name: "RandomShift"
    type: "Buffer"
    shape: { dim:-1 dim: 3 dim: 32 dim: 32 }
  }
  variable {
    name: "MulScalarY"
    type: "Buffer"
    shape: { dim:-1 dim: 1 }
  }
  variable {
    name: "RandomFlip"
    type: "Buffer"
    shape: { dim:-1 dim: 3 dim: 32 dim: 32 }
  }
  variable {
    name: "X2"
    type: "Buffer"
    shape: { dim:-1 dim: 3 dim: 32 dim: 32 }
    initializer {
      type: "Normal"
      multiplier: 1
    }
  }
  variable {
    name: "MulScalarX"
    type: "Buffer"
    shape: { dim:-1 dim: 3 dim: 32 dim: 32 }
  }
  function {
    name: "Y2"
    type: "Identity"
    input: "InputY"
    output: "Y2"
  }
  function {
    name: "RandomShift"
    type: "RandomShift"
    input: "InputX"
    output: "RandomShift"
    random_shift_param {
      shifts: 4
      shifts: 4
      border_mode: "nearest"
      base_axis: 1
      seed: -1
    }
  }
  function {
    name: "MulScalarY"
    type: "MulScalar"
    input: "Y2"
    output: "MulScalarY"
    mul_scalar_param {
      val: 0
    }
  }
  function {
    name: "RandomFlip"
    type: "RandomFlip"
    input: "RandomShift"
    output: "RandomFlip"
    random_flip_param {
      axes: 3
      base_axis: 1
      seed: -1
    }
  }
  function {
    name: "X2"
    type: "Identity"
    input: "RandomFlip"
    output: "X2"
  }
  function {
    name: "MulScalarX"
    type: "MulScalar"
    input: "X2"
    output: "MulScalarX"
    mul_scalar_param {
      val: 0
    }
  }
}

optimizer {
  start_iter: 0
  end_iter: 0
  name: "Optimizer"
  update_interval: 1
  network_name: "SetData"
  solver {
    type: "Nesterov"
    weight_decay: 0.0005
    nesterov_param {
      lr: 0.05
      momentum: 0.9
    }
    lr_scheduler_type: "Cosine"
    cosine_scheduler_param {
      max_iter: 117000
    }
    lr_warmup_scheduler_type: "None"
  }
  data_variable {
    variable_name: "InputX"
    data_name: "x"
  }
  data_variable {
    variable_name: "InputY"
    data_name: "y"
  }
  loss_variable {
    variable_name: "MulScalarY"
  }
  loss_variable {
    variable_name: "MulScalarX"
  }
  parameter_variable {
    variable_name: "Y2"
    learning_rate_multiplier: 0
  }
  parameter_variable {
    variable_name: "X2"
    learning_rate_multiplier: 0
  }
}
'''

# github issue #751


def test_workingmemory_layer():
    from nnabla.utils.load import load
    with create_temp_with_dir("network.nntxt") as fn:
        with open(fn, "w") as f:
            f.write(nntxt_mixup)
        load(fn)
