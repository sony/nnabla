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
#include <nbla/function/top_k_grad.hpp>
#include <nbla/utils/top_k.hpp>
#include <nbla/variable.hpp>

namespace nbla {

NBLA_REGISTER_FUNCTION_SOURCE(TopKGrad, int, bool, int);

template <typename T>
void TopKGrad<T>::setup_impl(const Variables &inputs,
                             const Variables &outputs) {
  const auto x = inputs[0];
  const auto y = outputs[0];
  const auto k = k_;
  Shape_t x_shape = x->shape();

  if (base_axis_ < 0)
    base_axis_ += x_shape.size();
  const auto base_axis = base_axis_;

  NBLA_CHECK(base_axis_ >= 0, error_code::value,
             "base_axis must not be less than zero, got %d", base_axis_);
  NBLA_CHECK(static_cast<Shape_t::size_type>(base_axis_) < x_shape.size(),
             error_code::value,
             "base_axis must be less than dimensions of x, but "
             "base_axis %d >= dimensions of x %d",
             base_axis, x_shape.size());

  NBLA_CHECK(k > 0, error_code::value,
             "k must not be less than 1, but k %d < 1", k);

  NBLA_CHECK(k <= x->size(base_axis), error_code::value,
             "k must not exceed the sample size, but k %d > sample size %d", k,
             x->size(base_axis));

  y->reshape(x_shape, true);
  top_k_idx_.reshape(Shape_t{k}, true);
}

template <typename T>
void TopKGrad<T>::forward_impl(const Variables &inputs,
                               const Variables &outputs) {
  const auto x = inputs[0];
  const auto y = outputs[0];

  auto x_data = x->data()->get(get_dtype<T>(), this->ctx_);
  auto y_data = y->data()->cast(get_dtype<T>(), this->ctx_, true);

  y_data->copy_from(x_data);
}

template <typename T>
void TopKGrad<T>::backward_impl(const Variables &inputs,
                                const Variables &outputs,
                                const vector<bool> &propagate_down,
                                const vector<bool> &accum) {
  if (!(propagate_down[0])) {
    return;
  }

  const auto x = inputs[0];
  const auto y = outputs[0];

  if (!accum[0])
    x->grad()->zero();

  auto y_grad = y->get_grad_pointer<T>(this->ctx_);
  auto x_grad = x->cast_grad_and_get_pointer<T>(this->ctx_);
  auto tk_idx = top_k_idx_.cast_data_and_get_pointer<size_t>(this->ctx_);

  std::function<void(const T *, const size_t, const size_t, size_t *)>
      top_k_func = this->abs_ ? top_k_abs<T> : top_k<T>;

  auto inner_size = y->size(this->base_axis_);
  auto outer_size = y->size() / inner_size;

  for (int s = 0; s < outer_size; s++) {
    top_k_func(y_grad, inner_size, this->k_, tk_idx);
    for (int k = 0; k < k_; k++) {
      const auto i = tk_idx[k];
      x_grad[i] += y_grad[i];
    }
    y_grad += inner_size;
    x_grad += inner_size;
  }
}
}
