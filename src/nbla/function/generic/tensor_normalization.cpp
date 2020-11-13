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
#include <nbla/function/tensor_normalization.hpp>
#include <nbla/variable.hpp>

namespace nbla {

NBLA_REGISTER_FUNCTION_SOURCE(TensorNormalization, const vector<int> &, float,
                              bool, bool);

template <typename T>
void TensorNormalization<T>::setup_impl(const Variables &inputs,
                                        const Variables &outputs) {
  const auto x_shape = inputs[0]->shape();
  const auto ndim = x_shape.size();
  beta_idx_ = no_bias_ ? -1 : 1;
  gamma_idx_ = no_scale_ ? -1 : no_bias_ ? 1 : 2;
  bn_param_shape_ = Shape_t(ndim, 1);
  for (auto a : axes_) {
    bn_param_shape_[a] = x_shape[a];
  }

  // check inputs size
  int n_inputs_expect = 1;
  if (!no_scale_)
    n_inputs_expect++;
  if (!no_bias_)
    n_inputs_expect++;
  NBLA_CHECK(inputs.size() == n_inputs_expect, error_code::value,
             "Number of inputs must be 1, 2 or 3.");

  // check param shapes
  const auto beta = no_bias_ ? nullptr : inputs[beta_idx_];
  const auto gamma = no_scale_ ? nullptr : inputs[gamma_idx_];
  if (beta) {
    const auto beta_shape = beta->shape();
    NBLA_CHECK(bn_param_shape_ == beta_shape, error_code::value,
               "Shape of beta(inputs[%d]) does not match. "
               "beta: (%s) != expected: (%s).",
               beta_idx_, string_join(beta_shape, string(", ")).c_str(),
               string_join(bn_param_shape_, string(", ")).c_str());
  }
  if (gamma) {
    const auto gamma_shape = gamma->shape();
    NBLA_CHECK(bn_param_shape_ == gamma_shape, error_code::value,
               "Shape of gamma(inputs[%d]) does not match. "
               "gamma: (%s) != expected: (%s).",
               gamma_idx_, string_join(gamma_shape, string(", ")).c_str(),
               string_join(bn_param_shape_, string(", ")).c_str());
  }

  output_stat_ = outputs.size() == 3;

  // reshape outputs
  outputs[0]->reshape(x_shape, true);
  if (output_stat_) {
    outputs[1]->reshape(bn_param_shape_, true); // batch mean
    outputs[2]->reshape(bn_param_shape_, true); // batch var
  }

  // batch_norm adapter
  bn_in_adapter_.reset(
      new BatchNormalizationInOutAdapter(ctx_, ndim, x_shape, axes_));
  bn_param_adapter_.reset(
      new BatchNormalizationInOutAdapter(ctx_, ndim, bn_param_shape_, axes_));

  // function
  const vector<int> bn_axes = {static_cast<int>(ndim - axes_.size())};
  f_batch_norm_ = create_BatchNormalization(ctx_, bn_axes, 0. /* decay_rate */,
                                            eps_, true /* batch_stat */);
}

template <typename T>
void TensorNormalization<T>::forward_impl(const Variables &inputs,
                                          const Variables &outputs) {
  // tensor norm variables
  auto x = inputs[0];
  auto beta = no_bias_ ? nullptr : inputs[beta_idx_];
  auto gamma = no_scale_ ? nullptr : inputs[gamma_idx_];

  // batch norm variables
  Variable bn_x, bn_beta, bn_gamma, bn_in_mean, bn_in_variance;
  Variable bn_y, bn_out_mean, bn_out_variance;
  Variable dummy_beta,
      dummy_gamma;                // dummy used when beta and gamma are optional
  Variable mean(bn_param_shape_); // dummy
  Variable variance(bn_param_shape_); // dummy
  if (!beta) {
    dummy_beta.reshape(bn_param_shape_, true);
    dummy_beta.data()->zero();
    beta = &dummy_beta;
  }
  if (!gamma) {
    dummy_gamma.reshape(bn_param_shape_, true);
    dummy_gamma.data()->fill(1.);
    gamma = &dummy_gamma;
  }

  const auto bn_inputs =
      Variables{&bn_x, &bn_beta, &bn_gamma, &bn_in_mean, &bn_in_variance};
  const auto bn_outputs = output_stat_
                              ? Variables{&bn_y, &bn_out_mean, &bn_out_variance}
                              : Variables{&bn_y};

  bn_in_adapter_->pre_op(x, &bn_x);
  bn_param_adapter_->pre_op(beta, &bn_beta);
  bn_param_adapter_->pre_op(gamma, &bn_gamma);
  bn_param_adapter_->pre_op(&mean, &bn_in_mean);
  bn_param_adapter_->pre_op(&variance, &bn_in_variance);

  nbla::execute(f_batch_norm_, bn_inputs, bn_outputs);

  bn_in_adapter_->post_op(&bn_y, outputs[0]);
  if (output_stat_) {
    bn_param_adapter_->post_op(&bn_out_mean, outputs[1]);
    bn_param_adapter_->post_op(&bn_out_variance, outputs[2]);
  }
}

template <typename T>
void TensorNormalization<T>::backward_impl(const Variables &inputs,
                                           const Variables &outputs,
                                           const vector<bool> &propagate_down,
                                           const vector<bool> &accum) {
  if (!(propagate_down[0] || (inputs.size() > 1 && propagate_down[1]) ||
        (inputs.size() > 2 && propagate_down[2]))) {
    return;
  }

  // tensor norm variables
  const auto x = inputs[0];
  auto beta = no_bias_ ? nullptr : inputs[beta_idx_];
  auto gamma = no_scale_ ? nullptr : inputs[gamma_idx_];

  // batch norm variables
  Variable bn_x, bn_beta, bn_gamma, bn_in_mean, bn_in_variance;
  Variable bn_y, bn_out_mean, bn_out_variance;
  Variable dummy_beta,
      dummy_gamma;                // dummy used when beta and gamma is optional
  Variable mean(bn_param_shape_); // dummy
  Variable variance(bn_param_shape_); // dummy
  if (!beta) {
    dummy_beta.reshape(bn_param_shape_, true);
    dummy_beta.data()->zero();
    beta = &dummy_beta;
  }
  if (!gamma) {
    dummy_gamma.reshape(bn_param_shape_, true);
    dummy_gamma.data()->fill(1.);
    gamma = &dummy_gamma;
  }

  // synonyms
  const auto bn_inputs =
      Variables{&bn_x, &bn_beta, &bn_gamma, &bn_in_mean, &bn_in_variance};
  const auto bn_outputs = output_stat_
                              ? Variables{&bn_y, &bn_out_mean, &bn_out_variance}
                              : Variables{&bn_y};
  const auto pd = propagate_down;
  const bool pd_beta = no_bias_ ? false : pd[beta_idx_];
  const bool pd_gamma = no_scale_ ? false : pd[gamma_idx_];
  const bool pd_param = pd_beta || pd_gamma;
  // propagate_down of beta and gamma must be same in batch_norm
  vector<bool> bn_pd = {pd[0], pd_param, pd_param, false, false};
  const vector<bool> bn_accum(5, false);

  // forward (in_adapter -> batch_norm -> out_adapter)

  bn_in_adapter_->pre_op(x, &bn_x);
  bn_param_adapter_->pre_op(beta, &bn_beta);
  bn_param_adapter_->pre_op(gamma, &bn_gamma);
  bn_param_adapter_->pre_op(&mean, &bn_in_mean);
  bn_param_adapter_->pre_op(&variance, &bn_in_variance);

  nbla::execute(f_batch_norm_, bn_inputs, bn_outputs);

  bn_in_adapter_->post_op(&bn_y, outputs[0]);
  if (output_stat_) {
    bn_param_adapter_->post_op(&bn_out_mean, outputs[1]);
    bn_param_adapter_->post_op(&bn_out_variance, outputs[2]);
  }

  // backward (out_adapter -> batch_norm -> in_adapter)

  bn_in_adapter_->post_op_backward(&bn_y, outputs[0], true, false);
  if (output_stat_) {
    bn_param_adapter_->post_op_backward(&bn_out_mean, outputs[1], true, false);
    bn_param_adapter_->post_op_backward(&bn_out_variance, outputs[2], true,
                                        false);
  }

  nbla::backward(f_batch_norm_, bn_inputs, bn_outputs, bn_pd, bn_accum);

  bn_in_adapter_->pre_op_backward(x, &bn_x, pd[0], accum[0]);
  if (pd_beta) {
    bn_param_adapter_->pre_op_backward(beta, &bn_beta, true, accum[beta_idx_]);
  }
  if (pd_gamma) {
    bn_param_adapter_->pre_op_backward(gamma, &bn_gamma, true,
                                       accum[gamma_idx_]);
  }
}
}
