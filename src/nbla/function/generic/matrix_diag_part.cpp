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

/** MatrixDiagPart
 */
#include <nbla/array.hpp>
#include <nbla/function/matrix_diag_part.hpp>
#include <nbla/variable.hpp>

#include <algorithm>

namespace nbla {

NBLA_REGISTER_FUNCTION_SOURCE(MatrixDiagPart);

template <typename T>
void MatrixDiagPart<T>::setup_impl(const Variables &inputs,
                                   const Variables &outputs) {
  // Check shape
  Shape_t shape_x = inputs[0]->shape();
  NBLA_CHECK(shape_x.size() > 1, error_code::value,
             "Input dimensions must be greater than 1.");
  NBLA_CHECK(shape_x[shape_x.size() - 2] = shape_x[shape_x.size() - 1],
             error_code::value,
             "Last and second last dimensions must be the same.");
  last_ndim_ = shape_x[shape_x.size() - 1];

  // Create new shape and compute part size
  Shape_t shape_y;
  for (Shape_t::size_type i = 0; i < shape_x.size() - 1; ++i) {
    shape_y.push_back(shape_x[i]);
  }

  // Reshape output
  outputs[0]->reshape(shape_y, true);
}

template <typename T>
void MatrixDiagPart<T>::forward_impl(const Variables &inputs,
                                     const Variables &outputs) {
  const T *x = inputs[0]->get_data_pointer<T>(this->ctx_);
  T *y = outputs[0]->cast_data_and_get_pointer<T>(this->ctx_, true);
  Size_t size = outputs[0]->size();
  for (int i = 0; i < size; ++i) {
    y[i] = x[i * last_ndim_ + i % last_ndim_];
  }
}

template <typename T, bool accum>
void matrix_diag_part_backward_cpu(int size, int last_ndim, T *dx,
                                   const T *dy) {
  for (int i = 0; i < size; ++i) {
    if (accum) {
      dx[i * last_ndim + i % last_ndim] += dy[i];
    } else {
      for (int j = 0; j < last_ndim; ++j) {
        if (i % last_ndim == j) {
          dx[i * last_ndim + j] = dy[i];
        } else {
          dx[i * last_ndim + j] = (T)0.;
        }
      }
    }
  }
}

template <typename T>
void MatrixDiagPart<T>::backward_impl(const Variables &inputs,
                                      const Variables &outputs,
                                      const vector<bool> &propagate_down,
                                      const vector<bool> &accum) {
  if (!propagate_down[0]) {
    return;
  }

  T *dx = inputs[0]->cast_grad_and_get_pointer<T>(this->ctx_, !accum[0]);
  const T *dy = outputs[0]->get_grad_pointer<T>(this->ctx_);
  Size_t size = outputs[0]->size();
  if (accum[0])
    matrix_diag_part_backward_cpu<T, true>(size, last_ndim_, dx, dy);
  else
    matrix_diag_part_backward_cpu<T, false>(size, last_ndim_, dx, dy);
}

} // namespace nbla
