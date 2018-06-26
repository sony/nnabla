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

#include <nbla/function/prod.hpp>

namespace nbla {

NBLA_REGISTER_FUNCTION_SOURCE(Prod, const vector<int> &, bool);

template <typename T>
void Prod<T>::forward_impl_reduce(const T *x, T *y, int outer_size,
                                  int reduction_size) {
  for (int o = 0; o < outer_size; ++o) {
    T &val = y[o];
    val = 1;
    for (int i = 0; i < reduction_size; ++i) {
      val *= x[o * reduction_size + i];
    }
  }
}

template <typename T>
void Prod<T>::backward_impl(const Variables &inputs, const Variables &outputs,
                            const vector<bool> &prop_down,
                            const vector<bool> &accum) {
  if (!prop_down[0])
    return;
  auto _get = [this](Variable *v) {
    return v->get_data_pointer<T>(this->ctx_);
  };
  auto _gcast = [this](Variable *v, bool wo) {
    return v->cast_grad_and_get_pointer<T>(this->ctx_, wo);
  };
  const T *dy = outputs[0]->get_grad_pointer<T>(this->ctx_);
  T *dx = this->f_transpose_ ? _gcast(this->o_transpose_.get(), true)
                             : _gcast(inputs[0], !accum[0]);
  const T *x =
      this->f_transpose_ ? _get(this->o_transpose_.get()) : _get(inputs[0]);
  const T *y = _get(outputs[0]);
  this->backward_impl_reduce_prod(
      dy, x, y, dx, inputs[0]->size() / this->reduction_size_,
      this->reduction_size_, accum[0] && (!this->f_transpose_));
  // If need un-transpose
  if (!this->f_transpose_)
    return;
  this->f_transpose_->backward(inputs, Variables{this->o_transpose_.get()},
                               prop_down, {accum[0]});
}

template <typename T>
void Prod<T>::backward_impl_reduce_prod(const T *dy, const T *x, const T *y,
                                        T *dx, int outer_size,
                                        int reduction_size, bool accum) {
  for (int o = 0; o < outer_size; ++o) {
    for (int i = 0; i < reduction_size; ++i) {
      const int j = o * reduction_size + i;
      // Note: Very unstable gradient. Compute (x1...xN)/xi
      // instead of (x1...xi-1xi+1...xN).
      if (accum) {
        dx[j] += ((x[j] == 0) ? (T)0 : (dy[o] * y[o] / x[j]));
      } else {
        dx[j] = ((x[j] == 0) ? (T)0 : (dy[o] * y[o] / x[j]));
      }
    }
  }
}

} // namespace nbla
