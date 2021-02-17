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
#include <nbla/function/top_k_data.hpp>
#include <nbla/utils/top_k.hpp>
#include <nbla/variable.hpp>

namespace nbla {

NBLA_REGISTER_FUNCTION_SOURCE(TopKData, int, bool, bool, int);

template <typename T>
void TopKData<T>::setup_impl(const Variables &inputs,
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

  if (!reduce_) {
    y->reshape(x_shape, true);
  } else {
    Shape_t y_shape;
    y_shape.reserve(base_axis + 1);
    std::copy_n(x_shape.begin(), base_axis, std::back_inserter(y_shape));
    y_shape.push_back(k);
    y->reshape(y_shape, true);
  }

  ss_ = x->size(base_axis); // input sample size
  ns_ = x->size() / ss_;    // number of samples
  fs_ = y->size(base_axis); // output feature size
  top_k_idx_.reshape(Shape_t{ns_, k}, true);
  forward_done_ = false;
}

template <typename T>
void TopKData<T>::forward_impl(const Variables &inputs,
                               const Variables &outputs) {
  const auto x = inputs[0];
  const auto y = outputs[0];

  if (!reduce_)
    y->data()->zero();

  auto x_data = x->get_data_pointer<T>(this->ctx_);
  auto y_data = y->cast_data_and_get_pointer<T>(this->ctx_);
  auto tk_idx = top_k_idx_.cast_data_and_get_pointer<size_t>(this->ctx_);

  std::function<void(const T *, const size_t, const size_t, size_t *)>
      top_k_func = this->abs_ ? top_k_abs<T> : top_k<T>;

  for (int s = 0; s < this->ns_; s++) {
    top_k_func(x_data, this->ss_, this->k_, tk_idx);
    for (int k = 0; k < k_; k++) {
      const auto i = tk_idx[k];
      y_data[reduce_ ? k : i] = x_data[i];
    }
    x_data += ss_; // increase by input sample size
    y_data += fs_; // increase by output feature size
    tk_idx += k_;
  }
  forward_done_ = true;
}

template <typename T>
void TopKData<T>::backward_impl(const Variables &inputs,
                                const Variables &outputs,
                                const vector<bool> &propagate_down,
                                const vector<bool> &accum) {
  if (!propagate_down[0]) {
    return;
  }

  NBLA_CHECK(forward_done_, error_code::value,
             "Forward must be called before calling backward.");

  const auto x = inputs[0];
  const auto y = outputs[0];

  if (!accum[0])
    x->grad()->zero();

  auto y_grad = y->get_grad_pointer<T>(ctx_);
  auto x_grad = x->cast_grad_and_get_pointer<T>(this->ctx_);
  auto tk_idx = top_k_idx_.get_data_pointer<size_t>(ctx_);

  if (!reduce_) {
    for (Size_t i = 0; i < x->size(); i++) {
      x_grad[i] += y_grad[i];
    }
  } else {
    for (int s = 0; s < ns_; s++) {
      for (int k = 0; k < k_; k++) {
        x_grad[tk_idx[k]] += y_grad[k];
      }
      x_grad += ss_;
      y_grad += fs_;
      tk_idx += k_;
    }
  }
}
}
