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

// reshape.cpp

#include <nbla/array.hpp>
#include <nbla/function/reshape.hpp>
#include <nbla/variable.hpp>

#include <algorithm>
#include <cstring>

namespace nbla {

NBLA_REGISTER_FUNCTION_SOURCE(Reshape, const vector<int> &, bool);

template <typename T>
void Reshape<T>::setup_impl(const Variables &inputs, const Variables &outputs) {
  // A: Shape inference for an axis specified with negative size
  int tsize = inputs[0]->size();
  int rest_size = 1;
  int shape_infer_index = -1;
  for (int s = 0; static_cast<Shape_t::size_type>(s) < shape_.size(); s++) {
    if (shape_[s] < 0) {
      NBLA_CHECK(shape_infer_index < 0, error_code::value,
                 "The shape option in Reshape function can take negative size "
                 "only in one axis. Given in %d and %d",
                 shape_infer_index, s);
      shape_infer_index = s;
      continue;
    }
    rest_size *= shape_[s];
  }
  if (shape_infer_index >= 0) {
    shape_[shape_infer_index] = tsize / rest_size;
  }

  // B: Check if product of dimensions is total size of input.
  int tsize2 = 1;
  for (auto s : shape_)
    tsize2 *= s;
  NBLA_CHECK(tsize == tsize2, error_code::value,
             "Product of dimensions of inputs and outputs must be same. "
             "Inputs: %d != Outputs: %d.",
             tsize, tsize2);

  // C: Reshape output
  outputs[0]->reshape(shape_, true);

  // D: Inplace
  if (inplace_) {
    outputs[0]->data()->set_array(inputs[0]->data()->array());
  }
}

template <class T>
void Reshape<T>::forward_impl(const Variables &inputs,
                              const Variables &outputs) {
  if (inplace_) {
    return;
  }

  const T *x = inputs[0]->get_data_pointer<T>(this->ctx_);
  T *y = outputs[0]->cast_data_and_get_pointer<T>(this->ctx_, true);
  for (int s = 0; s < inputs[0]->size(); s++) {
    y[s] = x[s];
  }
}

template <typename T, bool accum>
void reshape_backward_cpu(int size, T *dx, const T *dy) {
  for (int s = 0; s < size; ++s) {
    if (accum)
      dx[s] += dy[s];
    else
      dx[s] = dy[s];
  }
}

template <class T>
void Reshape<T>::backward_impl(const Variables &inputs,
                               const Variables &outputs,
                               const vector<bool> &propagate_down,
                               const vector<bool> &accum) {
  if (!propagate_down[0]) {
    return;
  }

  T *dx = inputs[0]->cast_grad_and_get_pointer<T>(this->ctx_, !accum[0]);
  const T *dy = outputs[0]->get_grad_pointer<T>(this->ctx_);
  if (accum[0])
    reshape_backward_cpu<T, true>(inputs[0]->size(), dx, dy);
  else
    reshape_backward_cpu<T, false>(inputs[0]->size(), dx, dy);
}
}
