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
#include <nbla/function/norm.hpp>
#include <nbla/variable.hpp>

#include <nbla/function/abs.hpp>
#include <nbla/function/pow_scalar.hpp>
#include <nbla/function/sum.hpp>

#include <nbla/imperative.hpp>

namespace nbla {

NBLA_REGISTER_FUNCTION_SOURCE(Norm, float, const vector<int> &, bool);

template <typename T>
void Norm<T>::setup_impl(const Variables &inputs, const Variables &outputs) {
  NBLA_CHECK(p_ >= 1, error_code::value,
             "`p` must be greater than or equal to 1. (p = %f)", p_);

  // functions
  abs_ = create_Abs(ctx_);
  pow_scalar_0_ = create_PowScalar(ctx_, p_, false);
  sum_ = create_Sum(ctx_, axes_, keep_dims_);
  pow_scalar_1_ = create_PowScalar(ctx_, 1 / p_, false);

  // Set output shape using Sum::setup
  sum_->setup(inputs, outputs);
}

template <typename T>
void Norm<T>::forward_impl(const Variables &inputs, const Variables &outputs) {
  auto x = inputs[0];
  auto y = outputs[0];
  Variable out_abs(x->shape());

  // abs -> pow -> sum -> pow
  execute(abs_, Variables{x}, Variables{&out_abs});
  execute(pow_scalar_0_, Variables{&out_abs}, Variables{&out_abs}); // inplace
  execute(sum_, Variables{&out_abs}, Variables{y});
  execute(pow_scalar_1_, Variables{y}, Variables{y}); // inplace
}

template <typename T>
void Norm<T>::backward_impl(const Variables &inputs, const Variables &outputs,
                            const vector<bool> &propagate_down,
                            const vector<bool> &accum) {
  const auto prop_down = propagate_down[0];
  if (!prop_down) {
    return;
  }

  auto x = inputs[0];
  auto y = outputs[0];

  // forward
  // abs -> pow -> sum -> pow
  Variable out_abs, out_pow_scalar_0, out_sum;
  execute(abs_, Variables{x}, Variables{&out_abs});
  execute(pow_scalar_0_, Variables{&out_abs}, Variables{&out_pow_scalar_0});
  execute(sum_, Variables{&out_pow_scalar_0}, Variables{&out_sum});
  execute(pow_scalar_1_, Variables{&out_sum}, Variables{y});

  // backward
  // abs <- pow <- sum <- pow
  nbla::backward(pow_scalar_1_, Variables{&out_sum}, Variables{y}, {prop_down},
                 {false});
  nbla::backward(sum_, Variables{&out_pow_scalar_0}, Variables{&out_sum},
                 {prop_down}, {false});
  nbla::backward(pow_scalar_0_, Variables{&out_abs},
                 Variables{&out_pow_scalar_0}, {prop_down}, {false});
  nbla::backward(abs_, Variables{x}, Variables{&out_abs}, {prop_down},
                 {accum[0]});
}
}
