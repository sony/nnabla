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
  size_t n_inputs_expect = 1;
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
  need_adapter_ = this->axes_.size() != 1;
  if (need_adapter_) {
    bn_in_adapter_.reset(
        new BatchNormalizationInOutAdapter(ctx_, ndim, x_shape, axes_));
    bn_param_adapter_.reset(
        new BatchNormalizationInOutAdapter(ctx_, ndim, bn_param_shape_, axes_));

    const int bn_axis = ndim - axes_.size();
    // ndim of batch_norm input x will be 1 when ndim - axes_.size() == 0. In
    // this case, batch_norm adapter add extra outer axis to batch_norm input x,
    // and the axis of batch_norm also need to be shifted.
    const vector<int> bn_axes = {bn_axis == 0 ? 1 : bn_axis};
    f_batch_norm_ =
        create_BatchNormalization(ctx_, bn_axes, 0. /* decay_rate */, eps_,
                                  true /* batch_stat */, no_scale_, no_bias_);
  } else {
    const vector<int> bn_axes = this->axes_;
    f_batch_norm_ =
        create_BatchNormalization(ctx_, bn_axes, 0. /* decay_rate */, eps_,
                                  true /* batch_stat */, no_scale_, no_bias_);
  }

  setup_batch_norm(inputs, outputs);
}

template <typename T>
void TensorNormalization<T>::setup_batch_norm(const Variables &inputs,
                                              const Variables &outputs) {
  if (need_adapter_) {
    // tensor norm variables
    auto x = inputs[0];
    auto beta = no_bias_ ? nullptr : inputs[beta_idx_];
    auto gamma = no_scale_ ? nullptr : inputs[gamma_idx_];

    // batch norm variables
    Variable bn_x, bn_beta, bn_gamma, bn_in_mean, bn_in_variance;
    Variable bn_y, bn_out_mean, bn_out_variance;
    Variable mean(bn_param_shape_);     // dummy
    Variable variance(bn_param_shape_); // dummy

    Variables bn_inputs;
    bn_inputs.push_back(&bn_x);
    if (!no_bias_)
      bn_inputs.push_back(&bn_beta);
    if (!no_scale_)
      bn_inputs.push_back(&bn_gamma);
    bn_inputs.push_back(&bn_in_mean);
    bn_inputs.push_back(&bn_in_variance);
    const auto bn_outputs =
        output_stat_ ? Variables{&bn_y, &bn_out_mean, &bn_out_variance}
                     : Variables{&bn_y};

    bn_in_adapter_->tn2bn(x, &bn_x);
    if (beta)
      bn_param_adapter_->tn2bn(beta, &bn_beta);
    if (gamma)
      bn_param_adapter_->tn2bn(gamma, &bn_gamma);
    bn_param_adapter_->tn2bn(&mean, &bn_in_mean);
    bn_param_adapter_->tn2bn(&variance, &bn_in_variance);

    f_batch_norm_->setup(bn_inputs, bn_outputs);
  } else {
    auto bn_inputs = inputs;
    const auto bn_outputs = outputs;

    // dummy variables for batch_norm
    Variable mean(bn_param_shape_);     // dummy
    Variable variance(bn_param_shape_); // dummy
    bn_inputs.push_back(&mean);
    bn_inputs.push_back(&variance);

    f_batch_norm_->setup(bn_inputs, bn_outputs);
  }
}

template <typename T>
void TensorNormalization<T>::forward_with_adapter(const Variables &inputs,
                                                  const Variables &outputs) {
  // tensor norm variables
  auto x = inputs[0];
  auto beta = no_bias_ ? nullptr : inputs[beta_idx_];
  auto gamma = no_scale_ ? nullptr : inputs[gamma_idx_];

  // batch norm variables
  Variable bn_x, bn_beta, bn_gamma, bn_in_mean, bn_in_variance;
  Variable bn_y, bn_out_mean, bn_out_variance;
  Variable mean(bn_param_shape_);     // dummy
  Variable variance(bn_param_shape_); // dummy

  Variables bn_inputs;
  bn_inputs.push_back(&bn_x);
  if (!no_bias_)
    bn_inputs.push_back(&bn_beta);
  if (!no_scale_)
    bn_inputs.push_back(&bn_gamma);
  bn_inputs.push_back(&bn_in_mean);
  bn_inputs.push_back(&bn_in_variance);
  const auto bn_outputs = output_stat_
                              ? Variables{&bn_y, &bn_out_mean, &bn_out_variance}
                              : Variables{&bn_y};

  bn_in_adapter_->tn2bn(x, &bn_x);
  if (beta)
    bn_param_adapter_->tn2bn(beta, &bn_beta);
  if (gamma)
    bn_param_adapter_->tn2bn(gamma, &bn_gamma);
  bn_param_adapter_->tn2bn(&mean, &bn_in_mean);
  bn_param_adapter_->tn2bn(&variance, &bn_in_variance);

  bn_y.reshape(bn_x.shape(), true);
  bn_out_mean.reshape(bn_in_mean.shape(), true);
  bn_out_variance.reshape(bn_in_variance.shape(), true);
  f_batch_norm_->forward(bn_inputs, bn_outputs);

  bn_in_adapter_->bn2tn(&bn_y, outputs[0]);
  if (output_stat_) {
    bn_param_adapter_->bn2tn(&bn_out_mean, outputs[1]);
    bn_param_adapter_->bn2tn(&bn_out_variance, outputs[2]);
  }
}

template <typename T>
void TensorNormalization<T>::forward_without_adapter(const Variables &inputs,
                                                     const Variables &outputs) {
  // tensor norm variables
  auto x = inputs[0];
  auto beta = no_bias_ ? nullptr : inputs[beta_idx_];
  auto gamma = no_scale_ ? nullptr : inputs[gamma_idx_];

  // batch norm variables
  Variable mean(bn_param_shape_);     // dummy
  Variable variance(bn_param_shape_); // dummy

  Variables bn_inputs;
  bn_inputs.push_back(x);
  if (!no_bias_)
    bn_inputs.push_back(beta);
  if (!no_scale_)
    bn_inputs.push_back(gamma);
  bn_inputs.push_back(&mean);
  bn_inputs.push_back(&variance);
  const auto bn_outputs = outputs;

  f_batch_norm_->forward(bn_inputs, bn_outputs);
}

template <typename T>
void TensorNormalization<T>::forward_impl(const Variables &inputs,
                                          const Variables &outputs) {
  if (need_adapter_) {
    forward_with_adapter(inputs, outputs);
  } else {
    forward_without_adapter(inputs, outputs);
  }
}

template <typename T>
void TensorNormalization<T>::backward_with_adapter(
    const Variables &inputs, const Variables &outputs,
    const vector<bool> &propagate_down, const vector<bool> &accum) {
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
  Variable mean(bn_param_shape_);     // dummy
  Variable variance(bn_param_shape_); // dummy

  // synonyms
  Variables bn_inputs;
  bn_inputs.push_back(&bn_x);
  if (!no_bias_)
    bn_inputs.push_back(&bn_beta);
  if (!no_scale_)
    bn_inputs.push_back(&bn_gamma);
  bn_inputs.push_back(&bn_in_mean);
  bn_inputs.push_back(&bn_in_variance);
  const auto bn_outputs = output_stat_
                              ? Variables{&bn_y, &bn_out_mean, &bn_out_variance}
                              : Variables{&bn_y};
  const auto pd = propagate_down;
  const bool pd_beta = no_bias_ ? false : pd[beta_idx_];
  const bool pd_gamma = no_scale_ ? false : pd[gamma_idx_];
  vector<bool> bn_pd;
  bn_pd.push_back(pd[0]); // x
  if (!no_bias_)
    bn_pd.push_back(pd_beta);
  if (!no_scale_)
    bn_pd.push_back(pd_gamma);
  bn_pd.push_back(false); // mean
  bn_pd.push_back(false); // variance
  const vector<bool> bn_accum(inputs.size() + 2 /* 2: mean, variance*/, false);

  // forward (in_adapter -> batch_norm -> out_adapter)

  bn_in_adapter_->tn2bn(x, &bn_x);
  if (beta)
    bn_param_adapter_->tn2bn(beta, &bn_beta);
  if (gamma)
    bn_param_adapter_->tn2bn(gamma, &bn_gamma);
  bn_param_adapter_->tn2bn(&mean, &bn_in_mean);
  bn_param_adapter_->tn2bn(&variance, &bn_in_variance);

  // We can skip following forward calculation since batch_norm outputs can be
  // restored from tensor_norm outputs.

  // nbla::execute(f_batch_norm_, bn_inputs, bn_outputs);

  // bn_in_adapter_->bn2tn(&bn_y, outputs[0]);
  // if (output_stat_) {
  //   bn_param_adapter_->bn2tn(&bn_out_mean, outputs[1]);
  //   bn_param_adapter_->bn2tn(&bn_out_variance, outputs[2]);
  // }

  // Restore batch_norm outputs from tensor_norm outputs
  bn_in_adapter_->tn2bn(outputs[0], &bn_y);
  if (output_stat_) {
    bn_param_adapter_->tn2bn(outputs[1], &bn_out_mean);
    bn_param_adapter_->tn2bn(outputs[2], &bn_out_variance);
  }

  // backward (out_adapter -> batch_norm -> in_adapter)

  bn_in_adapter_->bn2tn_backward(&bn_y, outputs[0], true, false);
  if (output_stat_) {
    bn_param_adapter_->bn2tn_backward(&bn_out_mean, outputs[1], true, false);
    bn_param_adapter_->bn2tn_backward(&bn_out_variance, outputs[2], true,
                                      false);
  }

  f_batch_norm_->backward(bn_inputs, bn_outputs, bn_pd, bn_accum);

  bn_in_adapter_->tn2bn_backward(x, &bn_x, pd[0], accum[0]);
  if (pd_beta) {
    bn_param_adapter_->tn2bn_backward(beta, &bn_beta, true, accum[beta_idx_]);
  }
  if (pd_gamma) {
    bn_param_adapter_->tn2bn_backward(gamma, &bn_gamma, true,
                                      accum[gamma_idx_]);
  }
}

template <typename T>
void TensorNormalization<T>::backward_without_adapter(
    const Variables &inputs, const Variables &outputs,
    const vector<bool> &propagate_down, const vector<bool> &accum) {
  if (!(propagate_down[0] || (inputs.size() > 1 && propagate_down[1]) ||
        (inputs.size() > 2 && propagate_down[2]))) {
    return;
  }

  // tensor norm variables
  const auto x = inputs[0];
  auto beta = no_bias_ ? nullptr : inputs[beta_idx_];
  auto gamma = no_scale_ ? nullptr : inputs[gamma_idx_];

  // batch norm variables
  Variable mean(bn_param_shape_);     // dummy
  Variable variance(bn_param_shape_); // dummy

  // synonyms
  Variables bn_inputs;
  bn_inputs.push_back(x);
  if (!no_bias_)
    bn_inputs.push_back(beta);
  if (!no_scale_)
    bn_inputs.push_back(gamma);
  bn_inputs.push_back(&mean);
  bn_inputs.push_back(&variance);
  const auto bn_outputs = outputs;
  const auto pd = propagate_down;
  const bool pd_beta = no_bias_ ? false : pd[beta_idx_];
  const bool pd_gamma = no_scale_ ? false : pd[gamma_idx_];
  vector<bool> bn_pd;
  bn_pd.push_back(pd[0]); // x
  if (!no_bias_)
    bn_pd.push_back(pd_beta);
  if (!no_scale_)
    bn_pd.push_back(pd_gamma);
  bn_pd.push_back(false); // mean
  bn_pd.push_back(false); // variance
  vector<bool> bn_accum = accum;
  bn_accum.push_back(false); // mean
  bn_accum.push_back(false); // variance

  f_batch_norm_->backward(bn_inputs, bn_outputs, bn_pd, bn_accum);
}

template <typename T>
void TensorNormalization<T>::backward_impl(const Variables &inputs,
                                           const Variables &outputs,
                                           const vector<bool> &propagate_down,
                                           const vector<bool> &accum) {
  if (need_adapter_) {
    backward_with_adapter(inputs, outputs, propagate_down, accum);
  } else {
    backward_without_adapter(inputs, outputs, propagate_down, accum);
  }
}
}
