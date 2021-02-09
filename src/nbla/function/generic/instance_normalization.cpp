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
#include <nbla/function/instance_normalization.hpp>
#include <nbla/variable.hpp>

#include <nbla/function/broadcast.hpp>
#include <nbla/function/tensor_normalization.hpp>
#include <nbla/imperative.hpp>

namespace nbla {

NBLA_REGISTER_FUNCTION_SOURCE(InstanceNormalization, int, const vector<int> &,
                              float, bool, bool);

template <typename T>
void InstanceNormalization<T>::setup_impl(const Variables &inputs,
                                          const Variables &outputs) {
  const int n_inputs = inputs.size();
  const auto x_shape = inputs[0]->shape();
  const int ndim = x_shape.size();
  beta_idx_ = no_bias_ ? -1 : 1;
  gamma_idx_ = no_scale_ ? -1 : no_bias_ ? 1 : 2;

  // check inputs size
  int n_inputs_expect = 1;
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
  // check batch_axis
  for (size_t i = 0; i < batch_axis_.size(); i++) {
    const auto ba = batch_axis_[i];
    NBLA_CHECK(0 <= ba && ba < ndim, error_code::value,
               "each element of batch_axis must be in the range of [0, ndim). "
               "batch_axis[%d] : %d, ndim: {}.",
               i, ba, ndim);
  }

  // check whether broadcast is needed or not.
  // Unlike layer_norm and group_norm, only instance_norm can use bn scale bias
  // & scale adaptation by broadcasting channel axis to channel * batch axis.
  // (like [1, C, 1, 1] -> [N, C, 1, 1])
  vector<int> adapt_shape(ndim, 1);
  for (auto ba : batch_axis_) {
    adapt_shape[ba] = x_shape[ba];
  }
  adapt_shape[channel_axis_] = x_shape[channel_axis_];

  // check param shapes
  const auto beta = no_bias_ ? nullptr : inputs[beta_idx_];
  const auto gamma = no_scale_ ? nullptr : inputs[gamma_idx_];
  // beta
  need_beta_broadcast_ = false;
  if (beta) {
    const auto beta_shape = beta->shape();
    for (size_t i = 0; i < beta_shape.size(); i++) {
      if (beta_shape[i] != adapt_shape[i]) {
        need_beta_broadcast_ = true;
        NBLA_CHECK(beta_shape[channel_axis_] == adapt_shape[channel_axis_],
                   error_code::value,
                   "channel size of beta: %d != channel size of x (%d).",
                   beta_shape[channel_axis_], adapt_shape[channel_axis_]);
        break;
      }
    }
  }
  // gamma
  need_gamma_broadcast_ = false;
  if (gamma) {
    const auto gamma_shape = gamma->shape();
    for (size_t i = 0; i < gamma_shape.size(); i++) {
      if (gamma_shape[i] != adapt_shape[i]) {
        need_gamma_broadcast_ = true;
        NBLA_CHECK(gamma_shape[channel_axis_] == adapt_shape[channel_axis_],
                   error_code::value,
                   "channel size of gamma: %d != channel size of x (%d).",
                   gamma_shape[channel_axis_], adapt_shape[channel_axis_]);
        break;
      }
    }
  }

  // reshape outputs
  outputs[0]->reshape(x_shape, true);
  Shape_t out_param_shape;
  for (int i = 0; i < ndim; i++) {
    out_param_shape.push_back(adapt_shape[i]);
  }
  if (outputs.size() == 3) {
    outputs[1]->reshape(out_param_shape, true); // batch mean
    outputs[2]->reshape(out_param_shape, true); // batch var
  }

  // functions
  if (need_beta_broadcast_) {
    f_broadcast_beta_ = create_Broadcast(ctx_, adapt_shape);
  }
  if (need_gamma_broadcast_) {
    f_broadcast_gamma_ = create_Broadcast(ctx_, adapt_shape);
  }
  auto tn_axes = batch_axis_;
  tn_axes.push_back(channel_axis_);
  f_tensor_norm_ =
      create_TensorNormalization(ctx_, tn_axes, eps_, no_scale_, no_bias_);

  // setup tensor_norm
  auto x = inputs[0];

  // tensor_norm variables
  Variable beta_bc, gamma_bc;
  Variable *tn_beta = beta;
  Variable *tn_gamma = gamma;

  // broadcast
  if (beta && need_beta_broadcast_) {
    f_broadcast_beta_->setup(Variables{beta}, Variables{&beta_bc});
    tn_beta = &beta_bc;
  }
  if (gamma && need_gamma_broadcast_) {
    f_broadcast_gamma_->setup(Variables{gamma}, Variables{&gamma_bc});
    tn_gamma = &gamma_bc;
  }

  // tensor_norm
  auto tn_inputs = Variables{x};
  if (beta)
    tn_inputs.push_back(tn_beta);
  if (gamma)
    tn_inputs.push_back(tn_gamma);
  f_tensor_norm_->setup(tn_inputs, outputs);
}

template <typename T>
void InstanceNormalization<T>::forward_impl(const Variables &inputs,
                                            const Variables &outputs) {
  auto x = inputs[0];
  const auto beta = no_bias_ ? nullptr : inputs[beta_idx_];
  const auto gamma = no_scale_ ? nullptr : inputs[gamma_idx_];

  // tensor_norm variables
  Variable beta_bc, gamma_bc;
  Variable *tn_beta = beta;
  Variable *tn_gamma = gamma;

  // broadcast
  if (beta && need_beta_broadcast_) {
    nbla::execute(f_broadcast_beta_, Variables{beta}, Variables{&beta_bc});
    tn_beta = &beta_bc;
  }
  if (gamma && need_gamma_broadcast_) {
    nbla::execute(f_broadcast_gamma_, Variables{gamma}, Variables{&gamma_bc});
    tn_gamma = &gamma_bc;
  }

  // tensor_norm
  auto tn_inputs = Variables{x};
  if (beta)
    tn_inputs.push_back(tn_beta);
  if (gamma)
    tn_inputs.push_back(tn_gamma);
  f_tensor_norm_->forward(tn_inputs, outputs);
}

template <typename T>
void InstanceNormalization<T>::backward_impl(const Variables &inputs,
                                             const Variables &outputs,
                                             const vector<bool> &propagate_down,
                                             const vector<bool> &accum) {
  if (!(propagate_down[0] || (inputs.size() > 1 && propagate_down[1]) ||
        (inputs.size() > 2 && propagate_down[2]))) {
    return;
  }

  auto x = inputs[0];
  const auto beta = no_bias_ ? nullptr : inputs[beta_idx_];
  const auto gamma = no_scale_ ? nullptr : inputs[gamma_idx_];

  // tensor_norm variables
  Variable beta_bc, gamma_bc;
  Variable *tn_beta = beta;
  Variable *tn_gamma = gamma;

  // forward (broadcast -> tensor_norm)

  // broadcast
  if (beta && need_beta_broadcast_) {
    nbla::execute(f_broadcast_beta_, Variables{beta}, Variables{&beta_bc});
    tn_beta = &beta_bc;
  }
  if (gamma && need_gamma_broadcast_) {
    nbla::execute(f_broadcast_gamma_, Variables{gamma}, Variables{&gamma_bc});
    tn_gamma = &gamma_bc;
  }

  auto tn_inputs = Variables{x};
  if (beta)
    tn_inputs.push_back(tn_beta);
  if (gamma)
    tn_inputs.push_back(tn_gamma);

  // We can skip tensor_norm forward calculation since its outputs are already
  // in data region of `outputs`
  // nbla::execute(f_tensor_norm_, tn_inputs, outputs);

  // backward (tensor_norm -> broadcast)

  vector<bool> tn_accum = {accum[0]};
  if (beta)
    tn_accum.push_back(accum[beta_idx_] && !need_beta_broadcast_);
  if (gamma)
    tn_accum.push_back(accum[gamma_idx_] && !need_gamma_broadcast_);

  f_tensor_norm_->backward(tn_inputs, outputs, propagate_down, tn_accum);

  if (beta && need_beta_broadcast_ && propagate_down[beta_idx_]) {
    nbla::backward(f_broadcast_beta_, Variables{beta}, Variables{&beta_bc},
                   {true}, {accum[beta_idx_]});
  }
  if (gamma && need_gamma_broadcast_ && propagate_down[gamma_idx_]) {
    nbla::backward(f_broadcast_gamma_, Variables{gamma}, Variables{&gamma_bc},
                   {true}, {accum[gamma_idx_]});
  }
}
}
