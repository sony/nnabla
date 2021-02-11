// Copyright (c) 2017 Sony Corporation. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <nbla/array.hpp>
#include <nbla/common.hpp>
#include <nbla/function/weight_standardization.hpp>
#include <nbla/variable.hpp>

#include <nbla/function/tensor_normalization.hpp>
#include <nbla/imperative.hpp>

namespace nbla {

NBLA_REGISTER_FUNCTION_SOURCE(WeightStandardization, int, float);

template <typename T>
void WeightStandardization<T>::setup_impl(const Variables &inputs,
                                          const Variables &outputs) {
  const auto w_shape = inputs[0]->shape();
  const int ndim = w_shape.size();

  // check chennel_axis
  NBLA_CHECK(0 <= channel_axis_ && channel_axis_ < ndim, error_code::value,
             "channel_axis must be in the range of [0, ndim). channel_axis : "
             "%d, ndim: {}.",
             channel_axis_, ndim);

  // convert channel_axis to axes for tensor_norm
  const vector<int> tn_axes = {channel_axis_};

  f_tensor_norm_ = create_TensorNormalization(
      ctx_, tn_axes, eps_, true /* no_scale */, true /* no_bias */);
  f_tensor_norm_->setup(inputs, outputs);
}

template <typename T>
void WeightStandardization<T>::forward_impl(const Variables &inputs,
                                            const Variables &outputs) {
  nbla::execute(f_tensor_norm_, inputs, outputs);
}

template <typename T>
void WeightStandardization<T>::backward_impl(const Variables &inputs,
                                             const Variables &outputs,
                                             const vector<bool> &propagate_down,
                                             const vector<bool> &accum) {
  if (!propagate_down[0]) {
    return;
  }

  nbla::backward(f_tensor_norm_, inputs, outputs, propagate_down, accum);
}
}
