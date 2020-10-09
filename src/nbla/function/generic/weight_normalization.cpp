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
#include <nbla/function/weight_normalization.hpp>
#include <nbla/variable.hpp>

#include <nbla/function/add2.hpp>
#include <nbla/function/add_scalar.hpp>
#include <nbla/function/mul2.hpp>
#include <nbla/function/pow_scalar.hpp>
#include <nbla/function/sum.hpp>

#include <nbla/imperative.hpp>

namespace nbla {

NBLA_REGISTER_FUNCTION_SOURCE(WeightNormalization, int, float);

template <typename T>
void WeightNormalization<T>::setup_impl(const Variables &inputs,
                                        const Variables &outputs) {
  auto ndim = inputs[0]->ndim();
  auto wshape = inputs[0]->shape();
  auto gshape = inputs[1]->shape();
  NBLA_CHECK(gshape[0] == wshape[dim_], error_code::value,
             "g.shape[0] does not match w.shape[dim]. "
             "g.shape[0] = %d, w.shape[%d] = %d.",
             gshape[0], dim_, wshape[dim_]);
  NBLA_CHECK(gshape.size() == 1, error_code::value,
             "ndim of g must be 1 (ndim of g = %d).", gshape.size());
  NBLA_CHECK(eps_ > 0, error_code::value, "eps must be positive. (eps = %f)",
             eps_);
  NBLA_CHECK(dim_ >= 0 && dim_ < ndim, error_code::value,
             "0 <= dim < %d must be true. (dim = %d, ndim = %d)", ndim, dim_,
             ndim);

  // functions
  pow_scalar_0_ = create_PowScalar(ctx_, 2, false);
  add_scalar_ = create_AddScalar(ctx_, eps_, false);
  pow_scalar_1_ = create_PowScalar(ctx_, -0.5, false);
  mul2_0_ = create_Mul2(ctx_, false);
  mul2_1_ = create_Mul2(ctx_, false);

  vector<int> axes;
  for (auto i = 0; i < ndim; i++) {
    if (i == dim_)
      continue;
    axes.push_back(i);
  }
  sum_ = create_Sum(ctx_, axes, true);

  // output
  outputs[0]->reshape(inputs[0]->shape(), true);
}

template <typename T>
void WeightNormalization<T>::forward_impl(const Variables &inputs,
                                          const Variables &outputs) {
  auto wshape = inputs[0]->shape();
  auto rshape = Shape_t(wshape);
  rshape[dim_] = 1;

  // pow -> sum -> add -> pow -> mul -> mul
  auto w = inputs[0];
  Variable out0(wshape);
  execute(pow_scalar_0_, Variables{w}, Variables{&out0});
  auto w_wn = outputs[0];
  Variable out1(rshape);
  execute(sum_, Variables{&out0}, Variables{&out1});
  execute(add_scalar_, Variables{&out1}, Variables{&out1});   // inplace
  execute(pow_scalar_1_, Variables{&out1}, Variables{&out1}); // inplace
  execute(mul2_0_, Variables{w, &out1}, Variables{w_wn});     // inplace
  auto g = inputs[1];
  auto gshape = g->shape();
  auto ndim = inputs[0]->ndim();
  Shape_t rgshape;
  for (auto i = 0; i < ndim; i++) {
    if (i != dim_)
      rgshape.push_back(1);
    else
      rgshape.push_back(wshape[dim_]);
  }
  g->reshape(rgshape, false);

  execute(mul2_1_, Variables{w_wn, g}, Variables{w_wn}); // inplace
  g->reshape(gshape, false);
}

template <typename T>
void WeightNormalization<T>::backward_impl(const Variables &inputs,
                                           const Variables &outputs,
                                           const vector<bool> &propagate_down,
                                           const vector<bool> &accum) {
  if (!(propagate_down[0] || propagate_down[1])) {
    return;
  }
  auto wshape = inputs[0]->shape();
  auto rshape = Shape_t(wshape);
  rshape[dim_] = 1;

  // forward
  // pow -> sum -> add -> pow -> mul -> mul
  auto w = inputs[0];
  Variable out0(wshape);
  execute(pow_scalar_0_, Variables{w}, Variables{&out0});
  Variable out1(rshape);
  execute(sum_, Variables{&out0}, Variables{&out1});
  Variable out2(rshape);
  execute(add_scalar_, Variables{&out1}, Variables{&out2});
  Variable out3(rshape);
  execute(pow_scalar_1_, Variables{&out2}, Variables{&out3});
  Variable out4(wshape);
  execute(mul2_0_, Variables{w, &out3}, Variables{&out4});
  auto g = inputs[1];
  auto gshape = g->shape();
  auto ndim = inputs[0]->ndim();
  Shape_t rgshape;
  for (auto i = 0; i < ndim; i++) {
    if (i != dim_)
      rgshape.push_back(1);
    else
      rgshape.push_back(wshape[dim_]);
  }
  g->reshape(rgshape, false);
  auto w_wn = outputs[0];
  execute(mul2_1_, Variables{&out4, g}, Variables{w_wn});

  // backward
  // pow <- sum <- add <- pow <- mul <- mul
  auto w_prop_down = propagate_down[0];
  auto w_accum = accum[0];
  auto g_accum = accum[1];
  nbla::backward(mul2_1_, Variables{&out4, g}, Variables{w_wn}, propagate_down,
                 {false, g_accum});
  nbla::backward(mul2_0_, Variables{w, &out3}, Variables{&out4},
                 {w_prop_down, w_prop_down}, {w_accum, false});
  nbla::backward(pow_scalar_1_, Variables{&out2}, Variables{&out3},
                 {w_prop_down}, {false});
  nbla::backward(add_scalar_, Variables{&out1}, Variables{&out2}, {w_prop_down},
                 {false});
  nbla::backward(sum_, Variables{&out0}, Variables{&out1}, {w_prop_down},
                 {false});
  nbla::backward(pow_scalar_0_, Variables{w}, Variables{&out0}, {w_prop_down},
                 {true});
  g->reshape(gshape, false);
}
}
