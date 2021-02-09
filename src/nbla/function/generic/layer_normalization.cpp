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
#include <nbla/function/layer_normalization.hpp>
#include <nbla/variable.hpp>

#include <nbla/function/add2.hpp>
#include <nbla/function/mul2.hpp>
#include <nbla/function/sub2.hpp>
#include <nbla/function/tensor_normalization.hpp>
#include <nbla/imperative.hpp>

namespace nbla {

NBLA_REGISTER_FUNCTION_SOURCE(LayerNormalization, const vector<int> &, float,
                              bool, bool);

template <typename T>
void LayerNormalization<T>::setup_impl(const Variables &inputs,
                                       const Variables &outputs) {
  const int n_inputs = inputs.size();
  const int n_outputs = outputs.size();

  output_stat_ = n_outputs == 3;
  const auto x_shape = inputs[0]->shape();
  const int ndim = x_shape.size();
  beta_idx_ = no_bias_ ? -1 : 1;
  gamma_idx_ = no_scale_ ? -1 : no_bias_ ? 1 : 2;
  auto tn_axes = batch_axis_;
  auto tn_param_shape = x_shape;
  for (auto a : tn_axes) {
    tn_param_shape[a] = 1;
  }

  // check inputs size
  int n_inputs_expect = 1;
  if (!no_scale_)
    n_inputs_expect++;
  if (!no_bias_)
    n_inputs_expect++;
  NBLA_CHECK(n_inputs == n_inputs_expect, error_code::value,
             "Number of inputs must be 1, 2 or 3.");

  // check batch_axis
  for (size_t i = 0; i < batch_axis_.size(); i++) {
    const auto ba = batch_axis_[i];
    NBLA_CHECK(0 <= ba && ba < ndim, error_code::value,
               "each element of batch_axis must be in the range of [0, ndim). "
               "batch_axis[%d] : %d, ndim: {}.",
               i, ba, ndim);
  }

  // check param shapes
  const auto beta = no_bias_ ? nullptr : inputs[beta_idx_];
  const auto gamma = no_scale_ ? nullptr : inputs[gamma_idx_];
  if (beta) {
    const auto beta_shape = beta->shape();
    NBLA_CHECK(tn_param_shape == beta_shape, error_code::value,
               "Shape of beta(inputs[1]) does not match. "
               "beta: (%s) != expected: (%s).",
               string_join(beta_shape, string(", ")).c_str(),
               string_join(tn_param_shape, string(", ")).c_str());
  }
  if (gamma) {
    const auto gamma_shape = gamma->shape();
    NBLA_CHECK(tn_param_shape == gamma_shape, error_code::value,
               "Shape of gamma(inputs[1]) does not match. "
               "gamma: (%s) != expected: (%s).",
               string_join(gamma_shape, string(", ")).c_str(),
               string_join(tn_param_shape, string(", ")).c_str());
  }

  // calculate the shape of beta and gamma for tensor norm
  tn_shape_ = Shape_t(ndim, 1);
  for (auto a : tn_axes) {
    tn_shape_[a] = x_shape[a];
  }

  // reshape outputs
  outputs[0]->reshape(x_shape, true);
  if (output_stat_) {
    outputs[1]->reshape(tn_shape_, true); // batch mean
    outputs[2]->reshape(tn_shape_, true); // batch var
  }

  // functions
  f_tensor_norm_ = create_TensorNormalization(
      ctx_, tn_axes, eps_, true /* no_scale */, true /* no_bias */);
  f_tensor_norm_->setup({inputs[0]}, outputs);
  if (!no_scale_)
    f_mul2_ = create_Mul2(ctx_, false);
  if (!no_bias_)
    f_add2_ = create_Add2(ctx_, false);
  if (!no_bias_ && no_scale_)
    f_sub2_ = create_Sub2(ctx_, false); // needed for backward
}

template <typename T>
void LayerNormalization<T>::forward_impl(const Variables &inputs,
                                         const Variables &outputs) {
  // variables for affine
  auto bias = no_bias_ ? nullptr : inputs[beta_idx_];
  auto scale = no_scale_ ? nullptr : inputs[gamma_idx_];

  // tensor norm
  const auto tn_inputs = Variables{inputs[0]}; // no beta and gamma
  f_tensor_norm_->forward(tn_inputs, outputs);

  // apply affine
  auto y = outputs[0];
  if (scale) {
    nbla::execute(f_mul2_, Variables{y, scale}, Variables{y});
  }
  if (bias) {
    nbla::execute(f_add2_, Variables{y, bias}, Variables{y});
  }
}

template <typename T>
void LayerNormalization<T>::backward_impl(const Variables &inputs,
                                          const Variables &outputs,
                                          const vector<bool> &propagate_down,
                                          const vector<bool> &accum) {
  if (!(propagate_down[0] || (inputs.size() > 1 && propagate_down[1]) ||
        (inputs.size() > 2 && propagate_down[2]))) {
    return;
  }

  // tensor norm variables
  Variable out_tn_y_buf(outputs[0]->shape());
  Variable *out_tn_y = no_scale_ && no_bias_ ? outputs[0] : &out_tn_y_buf;
  const auto tn_inputs = Variables{inputs[0]}; // no beta and gamma
  const auto tn_outputs = output_stat_
                              ? Variables{out_tn_y, outputs[1], outputs[2]}
                              : Variables{out_tn_y};

  // variables for affine
  Variable out_mul2;
  auto bias = no_bias_ ? nullptr : inputs[beta_idx_];
  auto scale = no_scale_ ? nullptr : inputs[gamma_idx_];
  const auto mul2_inputs = Variables{out_tn_y, scale};
  const auto mul2_outputs = bias ? Variables{&out_mul2} : Variables{outputs[0]};
  const auto add2_inputs =
      scale ? Variables{&out_mul2, bias} : Variables{out_tn_y, bias};
  const auto add2_outputs = Variables{outputs[0]};

  // forward (tensor_norm -> affine)

  // We need tensor_norm forward recalculation only when scale operation is
  // enabled because Mul2 input cannot be restored from its output.
  const bool need_tn_forward = !no_scale_;
  if (need_tn_forward) {
    f_tensor_norm_->forward(tn_inputs, tn_outputs);

    if (scale) {
      nbla::execute(f_mul2_, mul2_inputs, mul2_outputs);
    }
    // add2 is skipped since its output is already in data region of outputs[0]
    // if (bias) {
    //   nbla::execute(f_add2_, add2_inputs, add2_outputs);
    // }
  } else { // no_scale == true
    if (bias) {
      // tensor_norm outputs are restored by layer_norm output - bias
      nbla::execute(f_sub2_, {outputs[0], bias}, {out_tn_y});
    }
  }

  // backward (affine -> tensor_norm)

  if (bias) {
    nbla::backward(f_add2_, add2_inputs, add2_outputs,
                   {true, propagate_down[beta_idx_]},
                   {false, accum[beta_idx_]});
  }
  if (scale) {
    nbla::backward(f_mul2_, mul2_inputs, mul2_outputs,
                   {true, propagate_down[gamma_idx_]},
                   {false, accum[gamma_idx_]});
  }

  if (propagate_down[0]) {
    f_tensor_norm_->backward(tn_inputs, tn_outputs, {true}, {accum[0]});
  }
}
}