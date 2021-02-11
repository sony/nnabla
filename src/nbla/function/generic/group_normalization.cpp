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
#include <nbla/function/group_normalization.hpp>
#include <nbla/variable.hpp>

#include <nbla/function/add2.hpp>
#include <nbla/function/instance_normalization.hpp>
#include <nbla/function/mul2.hpp>
#include <nbla/function/sub2.hpp>
#include <nbla/imperative.hpp>

namespace nbla {

NBLA_REGISTER_FUNCTION_SOURCE(GroupNormalization, int, int, const vector<int> &,
                              float, bool, bool);

template <typename T>
void GroupNormalization<T>::setup_impl(const Variables &inputs,
                                       const Variables &outputs) {
  const auto n_inputs = inputs.size();
  const auto n_outputs = outputs.size();
  beta_idx_ = no_bias_ ? -1 : 1;
  gamma_idx_ = no_scale_ ? -1 : no_bias_ ? 1 : 2;

  gn_x_shape_ = inputs[0]->shape();
  const int ndim = gn_x_shape_.size();
  output_stat_ = n_outputs == 3;

  // check inputs size
  size_t n_inputs_expect = 1;
  if (!no_scale_)
    n_inputs_expect++;
  if (!no_bias_)
    n_inputs_expect++;
  NBLA_CHECK(n_inputs == n_inputs_expect, error_code::value,
             "Number of inputs must be 1, 2 or 3.");

  // check chennel_axis
  NBLA_CHECK(0 <= channel_axis_ && channel_axis_ < ndim, error_code::value,
             "channel_axis must be in the range of [0, ndim). channel_axis : "
             "%d, ndim: {}.",
             channel_axis_, ndim);

  const auto cdim = gn_x_shape_[channel_axis_];
  NBLA_CHECK(cdim % num_groups_ == 0, error_code::value,
             "Channel dim (%d) must be integer multiple of num_groups (%d).",
             cdim, num_groups_);

  // check param shapes
  auto gn_param_shape = Shape_t(ndim, 1);
  gn_param_shape[channel_axis_] = gn_x_shape_[channel_axis_];
  const auto beta = no_bias_ ? nullptr : inputs[beta_idx_];
  const auto gamma = no_scale_ ? nullptr : inputs[gamma_idx_];
  if (beta) {
    const auto beta_shape = beta->shape();
    NBLA_CHECK(gn_param_shape == beta_shape, error_code::value,
               "Shape of beta(inputs[1]) does not match. "
               "beta: (%s) != expected: (%s).",
               string_join(beta_shape, string(", ")).c_str(),
               string_join(gn_param_shape, string(", ")).c_str());
  }
  if (gamma) {
    const auto gamma_shape = gamma->shape();
    NBLA_CHECK(gn_param_shape == gamma_shape, error_code::value,
               "Shape of gamma(inputs[1]) does not match. "
               "gamma: (%s) != expected: (%s).",
               string_join(gamma_shape, string(", ")).c_str(),
               string_join(gn_param_shape, string(", ")).c_str());
  }

  // calculate instance_norm input shape
  instn_x_shape_.clear();
  for (int i = 0; i < channel_axis_; i++) {
    instn_x_shape_.push_back(gn_x_shape_[i]);
  }
  instn_x_shape_.push_back(num_groups_);
  instn_x_shape_.push_back(cdim / num_groups_);
  if (channel_axis_ < ndim - 1) {
    for (int i = channel_axis_ + 1; i < ndim; i++) {
      instn_x_shape_.push_back(gn_x_shape_[i]);
    }
  }

  // calculate instance_norm param shape
  Shape_t adapt_shape(instn_x_shape_.size(), 1);
  for (auto a : batch_axis_) {
    adapt_shape[a] = instn_x_shape_[a];
  }
  adapt_shape[channel_axis_] = instn_x_shape_[channel_axis_];

  // reshape outputs
  outputs[0]->reshape(gn_x_shape_, true);
  if (n_outputs == 3) {
    outputs[1]->reshape(adapt_shape, true); // batch mean
    outputs[2]->reshape(adapt_shape, true); // batch var
  }

  // functions
  f_instance_norm_ =
      create_InstanceNormalization(ctx_, channel_axis_, batch_axis_, eps_,
                                   true /* no_scale */, true /* no_bias */);
  if (!no_scale_)
    f_mul2_ = create_Mul2(ctx_, false);
  if (!no_bias_)
    f_add2_ = create_Add2(ctx_, false);
  if (!no_bias_ && no_scale_)
    f_sub2_ = create_Sub2(ctx_, false); // needed for backward

  // setup instance norm

  auto x = inputs[0];
  auto y = outputs[0];

  // instance norm
  x->reshape(instn_x_shape_, false);
  y->reshape(instn_x_shape_, false);

  nbla::execute(f_instance_norm_, Variables{x}, outputs);

  // restore shape
  x->reshape(gn_x_shape_, false);
  y->reshape(gn_x_shape_, false);
}

template <typename T>
void GroupNormalization<T>::forward_impl(const Variables &inputs,
                                         const Variables &outputs) {
  auto x = inputs[0];
  auto y = outputs[0];

  // variables for affine
  auto bias = no_bias_ ? nullptr : inputs[beta_idx_];
  auto scale = no_scale_ ? nullptr : inputs[gamma_idx_];

  // instance norm
  x->reshape(instn_x_shape_, false);
  y->reshape(instn_x_shape_, false);

  f_instance_norm_->forward(Variables{x}, outputs);

  x->reshape(gn_x_shape_, false); // restore input shape
  y->reshape(gn_x_shape_, false);

  // apply affine
  if (scale) {
    nbla::execute(f_mul2_, Variables{y, scale}, Variables{y});
  }
  if (bias) {
    nbla::execute(f_add2_, Variables{y, bias}, Variables{y});
  }
}

template <typename T>
void GroupNormalization<T>::backward_impl(const Variables &inputs,
                                          const Variables &outputs,
                                          const vector<bool> &propagate_down,
                                          const vector<bool> &accum) {
  if (!(propagate_down[0] || (inputs.size() > 1 && propagate_down[1]) ||
        (inputs.size() > 2 && propagate_down[2]))) {
    return;
  }

  // group_norm variables
  auto x = inputs[0];

  // instance_norm variables
  Variable out_instn_y_buf(instn_x_shape_);
  Variable *out_instn_y = no_scale_ && no_bias_ ? outputs[0] : &out_instn_y_buf;
  const auto instn_inputs = Variables{x};
  const auto instn_outputs =
      output_stat_ ? Variables{out_instn_y, outputs[1], outputs[2]}
                   : Variables{out_instn_y};

  // variables for affine
  Variable out_mul2;
  auto bias = no_bias_ ? nullptr : inputs[beta_idx_];
  auto scale = no_scale_ ? nullptr : inputs[gamma_idx_];
  const auto mul2_inputs = Variables{out_instn_y, scale};
  const auto mul2_outputs = bias ? Variables{&out_mul2} : Variables{outputs[0]};
  const auto add2_inputs =
      scale ? Variables{&out_mul2, bias} : Variables{out_instn_y, bias};
  const auto add2_outputs = Variables{outputs[0]};

  // forward (instance_norm -> affine)

  // We need instance_norm forward recalculation only when scale operation is
  // enabled because Mul2 input cannot be restored from its output.
  const bool need_instn_forward = !no_scale_;
  x->reshape(instn_x_shape_, false);
  if (need_instn_forward) {
    f_instance_norm_->forward(instn_inputs, instn_outputs);
    out_instn_y->reshape(gn_x_shape_, false);

    if (scale) {
      nbla::execute(f_mul2_, mul2_inputs, mul2_outputs);
    }
    // add2 is skipped since its output is already in data region of outputs[0]
    // if (bias) {
    //   nbla::execute(f_add2_, add2_inputs, add2_outputs);
    // }
  } else { // no_scale == true
    if (bias) {
      // instance_norm outputs are restored by group_norm output - bias
      nbla::execute(f_sub2_, {outputs[0], bias}, {out_instn_y});
    }
  }

  // backward (affine -> instance_norm)

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
    out_instn_y->reshape(instn_x_shape_, false);
    f_instance_norm_->backward(instn_inputs, instn_outputs, {true}, {accum[0]});
    out_instn_y->reshape(gn_x_shape_, false);
  }
  x->reshape(gn_x_shape_, false); // restore input shape
}
}
