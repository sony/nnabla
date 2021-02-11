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
#include <nbla/function/fused_batch_normalization.hpp>
#include <nbla/variable.hpp>

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>

namespace nbla {

NBLA_REGISTER_FUNCTION_SOURCE(FusedBatchNormalization, const vector<int> &,
                              float, float, bool, const string &);

namespace fused_batch_normalization {
// These functions are special cases for the fused batch normalization
template <typename T>
void relu_backward(int size, T *dx, const T *dy, const T *y) {
  for (int i = 0; i < size; i++) {
    if (y[i] > 0)
      dx[i] = dy[i];
    else
      dx[i] = T(0);
  }
}

template <typename T>
void add2_backward(int size, T *dx1, const T *dx, bool accum) {
  for (int i = 0; i < size; i++) {
    dx1[i] = accum ? dx1[i] + dx[i] : dx[i];
  }
}
}

template <class T>
void FusedBatchNormalization<T>::setup_impl(const Variables &inputs,
                                            const Variables &outputs) {
  NBLA_CHECK(nonlinearity_ == "relu", error_code::not_implemented,
             "Currently \"relu\" is only supported as a nonlinearity.");
  Variables inputs_bn(inputs.begin(), inputs.begin() + 5);
  bn_ = create_BatchNormalization(this->ctx_, axes_, decay_rate_, eps_,
                                  batch_stat_, false /* no_scale */,
                                  false /* no_bias */);
  bn_->setup(inputs_bn, outputs);
}

template <class T>
void FusedBatchNormalization<T>::forward_impl(const Variables &inputs,
                                              const Variables &outputs) {

  NBLA_CHECK(bn_, error_code::value, "setup is not called.");
  // Naive non-fused implementation by layer composition.
  // 1. Perform BatchNormalization
  Variables inputs_bn(inputs.begin(), inputs.begin() + 5);
  bn_->forward(inputs_bn, outputs);

  // 2. Perform Add2
  // NOTE: Output buffer are re-used by inplacing.
  if (inputs.size() == 6) {
    auto add2 = create_Add2(this->ctx_, true);
    add2->setup(Variables{outputs[0], inputs[5]}, Variables{outputs[0]});
    add2->forward(Variables{outputs[0], inputs[5]}, Variables{outputs[0]});
  }

  // 3. Perform ReLU
  // NOTE: Output buffer are re-used by inplacing.
  auto relu = create_ReLU(this->ctx_, true);
  relu->setup(Variables{outputs[0]}, Variables{outputs[0]});
  relu->forward(Variables{outputs[0]}, Variables{outputs[0]});
}

template <class T>
void FusedBatchNormalization<T>::backward_impl(
    const Variables &inputs, const Variables &outputs,
    const vector<bool> &propagate_down, const vector<bool> &accum) {
  NBLA_CHECK(bn_, error_code::value, "setup is not called.");

  // Naive non-fused implementation by layer composition.
  // 1. Perform ReLU backward
  // NOTE: Output buffer are re-used by inplacing.
  bool prop_down_add2 = (inputs.size() == 6 && propagate_down[5]);
  bool prop_down_bn =
      std::accumulate(propagate_down.begin(), propagate_down.begin() + 3, false,
                      std::logical_or<bool>());
  auto y = outputs[0]->get_data_pointer<T>(this->ctx_);
  auto dx = outputs[0]->cast_grad_and_get_pointer<T>(this->ctx_);
  auto dy = outputs[0]->get_grad_pointer<T>(this->ctx_);
  auto size = outputs[0]->size();
  if (prop_down_add2 || prop_down_bn) {
    fused_batch_normalization::relu_backward(size, dx, dy, y);
  }

  // 2. Perform Add2 backward
  // NOTE: Output buffer for the first operand of the addition are re-used by
  // inplacing,
  // nothing done for it.
  if (prop_down_add2) {
    auto dx1 = inputs[5]->cast_grad_and_get_pointer<T>(this->ctx_);
    fused_batch_normalization::add2_backward(size, dx1, dx, accum[5]);
  }
  // 3. Perform BN backward
  Variables inputs_bn(inputs.begin(), inputs.begin() + 5);
  vector<bool> prop_down_bn_inputs(propagate_down.begin(),
                                   propagate_down.begin() + 5);
  vector<bool> accum_bn_inputs(accum.begin(), accum.begin() + 5);
  bn_->backward(inputs_bn, outputs, prop_down_bn_inputs, accum_bn_inputs);
}
} // namespace nbla
