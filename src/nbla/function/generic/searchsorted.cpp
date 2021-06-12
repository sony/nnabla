// Copyright 2021 Sony Group Corporation.
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
#include <nbla/function/searchsorted.hpp>
#include <nbla/variable.hpp>

namespace nbla {

NBLA_REGISTER_FUNCTION_SOURCE(SearchSorted, bool);

template <typename T>
void SearchSorted<T>::setup_impl(const Variables &inputs,
                                 const Variables &outputs) {

  NBLA_CHECK(inputs[0]->ndim() == inputs[1]->ndim(), error_code::value,
             "Sequence and Values must have the same ndim "
             "ndim of values: %d != ndim of values: %d.",
             inputs[0]->ndim(), inputs[1]->ndim());

  for (int i = 0; i < inputs[0]->ndim() - 1; ++i) {
    NBLA_CHECK(inputs[0]->shape()[i] == inputs[1]->shape()[i],
               error_code::value,
               "Sequence and Values must have same shape except at innermost "
               "dimension "
               "axes[%d]: Sequence shape %d != Values shape: %d.",
               i, inputs[0]->shape()[i], inputs[1]->shape()[i]);
  }

  outputs[0]->reshape(inputs[1]->shape(), true);

  ss_last_dim_ = inputs[0]->shape()[inputs[0]->shape().size() - 1];
  v_last_dim_ = inputs[1]->shape()[inputs[1]->shape().size() - 1];
  inner_size_ =
      inputs[0]->size() / ss_last_dim_; // same as values.size()/v_last_dim_
}

template <typename T>
size_t search(const T *sorted_sequence, const T value, int start, int end,
              bool right_) {

  if (value > sorted_sequence[end])
    return end + 1;

  if (right_ && value == sorted_sequence[end])
    return end + 1;

  if (value < sorted_sequence[start])
    return start;

  if (!right_ && value == sorted_sequence[start])
    return start;

  if (end - start <= 1)
    return end;

  size_t mid = (start + end + 1) / 2;

  bool check_condition =
      right_ ? (value < sorted_sequence[mid]) : (value <= sorted_sequence[mid]);
  if (check_condition)
    return search(sorted_sequence, value, start, mid, right_);
  else
    return search(sorted_sequence, value, mid, end, right_);
}

template <typename T>
void SearchSorted<T>::forward_impl(const Variables &inputs,
                                   const Variables &outputs) {

  const T *sorted_sequence = inputs[0]->get_data_pointer<T>(this->ctx_);
  const T *values = inputs[1]->get_data_pointer<T>(this->ctx_);

  T *y = outputs[0]->cast_data_and_get_pointer<T>(this->ctx_, true);
  for (size_t i = 0; i < inner_size_; i++) {

    for (size_t j = 0; j < v_last_dim_; j++) {

      size_t v_idx = i * v_last_dim_ + j;

      size_t i_idx = search(sorted_sequence, values[v_idx], i * ss_last_dim_,
                            (i + 1) * ss_last_dim_ - 1, right_) -
                     i * ss_last_dim_;
      y[v_idx] = i_idx;
    }
  }
}

template <typename T>
void SearchSorted<T>::backward_impl(const Variables &inputs,
                                    const Variables &outputs,
                                    const vector<bool> &propagate_down,
                                    const vector<bool> &accum) {
  NBLA_ERROR(error_code::not_implemented,
             "Do not call backward on SearchSorted. \n"
             "SearchSorted is a search and lookup function. It is not intended "
             "to be differentiable");
}
}
