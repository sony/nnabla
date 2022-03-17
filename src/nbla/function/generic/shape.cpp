// Copyright 2018,2019,2020,2021 Sony Corporation.
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

#include <iostream>
#include <nbla/array.hpp>
#include <nbla/common.hpp>
#include <nbla/function/shape.hpp>
#include <nbla/variable.hpp>
#include <typeinfo>

namespace nbla {

NBLA_REGISTER_FUNCTION_SOURCE(Shape, int, int);

template <typename T>
void Shape<T>::setup_impl(const Variables &inputs, const Variables &outputs) {
  auto shape = inputs[0]->shape();
  int shape_size = static_cast<int>(shape.size());
  NBLA_CHECK(shape_size > 0, error_code::value,
             "input shape is invalid! shape_size=%d", shape.size());
  int s = 0, e = shape_size;
  if (this->start_ < 0)
    s = this->start_ + e;
  else
    s = this->start_ > shape_size ? shape_size : this->start_;
  if (this->end_ < 0)
    e += this->end_;
  else if (this->end_ > 0)
    e = this->end_ > shape_size ? shape_size : this->end_;
  if (e > s)
    outputs[0]->reshape({e - s}, true);
}

template <typename T>
void Shape<T>::forward_impl(const Variables &inputs, const Variables &outputs) {
  auto shape = inputs[0]->shape();
  T *y = outputs[0]->cast_data_and_get_pointer<T>(this->ctx_, true);
  int shape_size = static_cast<int>(shape.size());
  NBLA_CHECK(shape_size > 0, error_code::value,
             "input shape is invalid! shape_size=%d", shape.size());
  int s = 0, e = shape_size;
  if (this->start_ < 0)
    s = this->start_ + e;
  else
    s = this->start_ > shape_size ? shape_size : this->start_;
  if (this->end_ < 0)
    e += this->end_;
  else if (this->end_ > 0)
    e = this->end_ > shape_size ? shape_size : this->end_;
  if (e > s) {
    for (int i = s, j = 0; i < e; ++i, ++j) {
      y[j] = shape[i];
    }
  }
}

template <typename T>
void Shape<T>::backward_impl(const Variables &inputs, const Variables &outputs,
                             const vector<bool> &propagate_down,
                             const vector<bool> &accum) {
  // Do nothing!
}
}
