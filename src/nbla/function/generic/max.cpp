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

#include <nbla/function/max.hpp>

#include <cstring>

namespace nbla {

NBLA_REGISTER_FUNCTION_SOURCE(Max, const vector<int> &, bool, bool, bool);

template <typename T>
void Max<T>::setup_impl(const Variables &inputs, const Variables &outputs) {
  Sum<T>::setup_impl(inputs, outputs);
  int outer_size = inputs[0]->size() / this->reduction_size_;
  this->index_buff_ = make_shared<Variable>(Shape_t{outer_size});
  if (this->with_index_ && !this->only_index_)
    outputs[1]->reshape(outputs[0]->shape(), true);
}

template <typename T>
void Max<T>::forward_impl(const Variables &inputs, const Variables &outputs) {
  Sum<T>::forward_impl(inputs, outputs);
  if (this->with_index_ || this->only_index_) {
    Variable *idx_var = this->only_index_ ? outputs[0] : outputs[1];
    auto idx_arr = idx_var->data()->cast(get_dtype<size_t>(), this->ctx_, true);
    auto idx_buf = this->index_buff_->data()->get(get_dtype<int>(), this->ctx_);
    idx_arr->copy_from(idx_buf);
  }
}

template <typename T>
void Max<T>::forward_impl_reduce(const T *x, T *y, int outer_size,
                                 int reduction_size) {
  // Saving index is a bit inefficient if backward is not required.
  int *ind = index_buff_->cast_data_and_get_pointer<int>(this->ctx_, true);
  for (int o = 0; o < outer_size; ++o) {
    int mi = 0;
    T m = -1e+8;
    for (int i = 0; i < reduction_size; ++i) {
      const T v = x[o * reduction_size + i];
      if (v > m) {
        m = v;
        mi = i;
      }
    }
    y[o] = m;
    ind[o] = mi;
  }
}

template <typename T>
void Max<T>::backward_impl_reduce(const T *dy, T *dx, int outer_size,
                                  int reduction_size, bool accum) {
  const int *ind = index_buff_->get_data_pointer<int>(this->ctx_);
  if (!accum)
    memset((void *)dx, 0, sizeof(*dx) * outer_size * reduction_size);
  for (int o = 0; o < outer_size; ++o) {
    dx[o * reduction_size + ind[o]] += dy[o];
  }
}

} // namespace nbla
