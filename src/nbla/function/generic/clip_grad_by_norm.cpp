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

/** ClipGradByNorm
 */
#include <nbla/array.hpp>
#include <nbla/function/clip_grad_by_norm.hpp>
#include <nbla/variable.hpp>

#include <nbla/function/broadcast.hpp>
#include <nbla/function/pow_scalar.hpp>
#include <nbla/function/sum.hpp>

#include <algorithm>

namespace nbla {

NBLA_REGISTER_FUNCTION_SOURCE(ClipGradByNorm, float, const vector<int> &);

template <typename T>
void ClipGradByNorm<T>::setup_impl(const Variables &inputs,
                                   const Variables &outputs) {
  outputs[0]->reshape(inputs[0]->shape(), true);
  sum_ = create_Sum(this->ctx_, axes_, true);
  pow_scalar_ = create_PowScalar(this->ctx_, 2., false);

  vector<int> _shape;
  for (auto v : inputs[0]->shape()) {
    _shape.push_back(v);
  }
  broadcast_ = create_Broadcast(this->ctx_, _shape);
}

template <typename T>
void ClipGradByNorm<T>::forward_impl(const Variables &inputs,
                                     const Variables &outputs) {
  const T *x = inputs[0]->get_data_pointer<T>(this->ctx_);
  T *y = outputs[0]->cast_data_and_get_pointer<T>(this->ctx_, true);
  for (int i = 0; i < inputs[0]->size(); i++) {
    y[i] = x[i];
  }
}

template <typename T, bool accum>
void clip_grad_by_norm_backward_cpu(int size, T clip_norm_grad, T *dx,
                                    const T *dy, const T *m) {
  for (int s = 0; s < size; ++s) {
    T _dx = clip_norm_grad * dy[s] / std::sqrt(m[s]);
    accum ? dx[s] += _dx : dx[s] = _dx;
  }
}

template <typename T> void clip_grad_by_norm_copy(int size, T *m, const T *dy) {
  for (int s = 0; s < size; ++s) {
    m[s] = dy[s];
  }
}

template <typename T>
void ClipGradByNorm<T>::backward_impl(const Variables &inputs,
                                      const Variables &outputs,
                                      const vector<bool> &propagate_down,
                                      const vector<bool> &accum) {
  if (!propagate_down[0]) {
    return;
  }

  auto shape = inputs[0]->shape();
  Variable v0(shape);
  Variable v1(shape);
  Variable v2(shape);
  Variable v3(shape);
  auto intermediates0 = Variables{&v0};
  auto intermediates1 = Variables{&v1};
  auto intermediates2 = Variables{&v2};
  auto intermediates3 = Variables{&v3};

  Size_t size = inputs[0]->size();
  T *_m = intermediates0[0]->cast_data_and_get_pointer<T>(this->ctx_, true);
  const T *_dy = outputs[0]->get_grad_pointer<T>(this->ctx_);
  clip_grad_by_norm_copy(size, _m, _dy);

  // power grads by 2.
  pow_scalar_->setup(intermediates0, intermediates1);
  pow_scalar_->forward(intermediates0, intermediates1);

  // sum grads powered by 2.
  sum_->setup(intermediates1, intermediates2);
  sum_->forward(intermediates1, intermediates2);

  // broadcast
  broadcast_->setup(intermediates2, intermediates3);
  broadcast_->forward(intermediates2, intermediates3);

  T *dx = inputs[0]->cast_grad_and_get_pointer<T>(this->ctx_, !accum[0]);
  const T *dy = outputs[0]->get_grad_pointer<T>(this->ctx_);
  const T *m = intermediates3[0]->get_data_pointer<T>(this->ctx_);
  if (accum[0])
    clip_grad_by_norm_backward_cpu<T, true>(size, clip_norm_, dx, dy, m);
  else
    clip_grad_by_norm_backward_cpu<T, false>(size, clip_norm_, dx, dy, m);
}

} // namespace nbla
