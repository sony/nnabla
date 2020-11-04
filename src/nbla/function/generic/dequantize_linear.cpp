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
#include <nbla/function/dequantize_linear.hpp>
#include <nbla/variable.hpp>

namespace nbla {

NBLA_REGISTER_FUNCTION_SOURCE(DequantizeLinear);

template <typename T>
void DequantizeLinear<T>::setup_impl(const Variables &inputs,
                                     const Variables &outputs) {
  NBLA_CHECK((inputs[0]->ndim() == inputs[1]->ndim()) &&
                 (inputs[0]->ndim() == inputs[2]->ndim()),
             error_code::value,
             "Dimensions of inputs must be same (%d, %d, %d).",
             inputs[0]->ndim(), inputs[1]->ndim(), inputs[2]->ndim());
  for (int i = 0; i < inputs[0]->ndim(); ++i) {
    auto s0 = inputs[0]->shape()[i];
    auto s1 = inputs[1]->shape()[i];
    auto s2 = inputs[2]->shape()[i];
    NBLA_CHECK(s1 == 1 || s1 == s0, error_code::value,
               "Size at %d-th dimension of inputs[1] (%d) should be 1 "
               "or match the size at %d-th dimension of inputs[0] (%d).",
               i, s1, i, s0);
    NBLA_CHECK(s2 == 1 || s2 == s0, error_code::value,
               "Size at %d-th dimension of inputs[2] (%d) should be 1 "
               "or match the size at %d-th dimension of inputs[0] (%d).",
               i, s2, i, s0);
  }

  outputs[0]->reshape(inputs[0]->shape(), true);

  mul2_ = create_Mul2(this->ctx_, false);
  sub2_ = create_Sub2(this->ctx_, false);
  add2_ = create_Add2(this->ctx_, false);
  // sum_ = create_Sum(this->ctx_);
}

template <typename T>
void DequantizeLinear<T>::forward_impl(const Variables &inputs,
                                       const Variables &outputs) {
  auto x = inputs[0];
  auto scale = inputs[1];
  auto zero_point = inputs[2];
  auto y = outputs[0];

  // Compute: (x - zero_point) * scale
  execute(sub2_, Variables{x, zero_point}, Variables{y});
  execute(mul2_, Variables{y, scale}, Variables{y});
}

template <typename T>
void DequantizeLinear<T>::backward_impl(const Variables &inputs,
                                        const Variables &outputs,
                                        const vector<bool> &propagate_down,
                                        const vector<bool> &accum) {
  if (!(propagate_down[0] || propagate_down[1] || propagate_down[2])) {
    return;
  }

  auto x = inputs[0];
  auto scale = inputs[1];
  auto y = outputs[0];

  auto dx_sptr = make_shared<Variable>(x->shape());
  auto dy_sptr = make_shared<Variable>(y->shape());
  auto dx = dx_sptr.get();
  auto dy = dy_sptr.get();
  dx->set_data(x->grad());
  dy->set_data(y->grad());

  if (propagate_down[0]) {
    if (accum[0]) {
      auto dx_tmp_sptr = make_shared<Variable>(x->shape());
      auto dx_tmp = dx_tmp_sptr.get();
      add2_ = create_Add2(this->ctx_, false);
      execute(mul2_, Variables{dy, scale}, Variables{dx_tmp});
      execute(add2_, Variables{dx, dx_tmp}, Variables{dx});
    } else {
      execute(mul2_, Variables{dy, scale}, Variables{dx});
    }
  }
  if (propagate_down[1]) {
    NBLA_ERROR(error_code::not_implemented,
               "Backward w.r.t. the scale is not supported now.");
  }
  if (propagate_down[2]) {
    NBLA_ERROR(error_code::not_implemented,
               "Backward w.r.t. the zero point is not supported now.");
  }
}
}
