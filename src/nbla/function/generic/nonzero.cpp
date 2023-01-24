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

#include <nbla/array.hpp>
#include <nbla/common.hpp>
#include <nbla/function/nonzero.hpp>
#include <nbla/function/not_equal_scalar.hpp>
#include <nbla/function/sum.hpp>
#include <nbla/imperative.hpp>
#include <nbla/utils/nd_index.hpp>
#include <nbla/variable.hpp>

namespace nbla {

NBLA_REGISTER_FUNCTION_SOURCE(NonZero);

template <typename T>
void NonZero<T>::non_zero(const Variables &inputs, const Variables &outputs) {
  // Inputs
  const auto x = inputs[0];
  const auto x_data = x->get_data_pointer<T>(this->ctx_);

  // Compute the number of non-zero elements
  Size_t num_nonzeros = 0;
  for (Size_t i = 0; i < x->size(); ++i) {
    num_nonzeros += int(x_data[i] != T(0));
  }

  // Outputs
  outputs[0]->reshape({x->ndim(), num_nonzeros}, true);
  size_t *y = outputs[0]->cast_data_and_get_pointer<size_t>(this->ctx_, true);

  // Gather indexes of non-zero elements
  size_t nonzero_index = 0;
  for (Size_t i = 0; i < x->size(); ++i) {
    if (x_data[i] == T(0))
      continue;

    const auto nd_index = ndi::flat2nd(i, x->strides());
    for (Size_t dim = 0; dim < x->ndim(); ++dim) {
      y[dim * num_nonzeros + nonzero_index] = nd_index[dim];
    }
    nonzero_index++;
  }
}

template <typename T>
void NonZero<T>::setup_impl(const Variables &inputs, const Variables &outputs) {
  non_zero(inputs, outputs);
}

template <typename T>
void NonZero<T>::forward_impl(const Variables &inputs,
                              const Variables &outputs) {
  // Forward is done at setup_impl() because the output shape is calculated
  // during forward computation.
}

template <typename T>
void NonZero<T>::backward_impl(const Variables &inputs,
                               const Variables &outputs,
                               const vector<bool> &propagate_down,
                               const vector<bool> &accum) {
  if (!(propagate_down[0])) {
    return;
  }

  // Gradient of outputs
  const T *g_y = outputs[0]->get_grad_pointer<T>(this->ctx_);

  // Gradient of inputs
  T *g_x{nullptr};

  if (propagate_down[0]) {
    g_x = inputs[0]->cast_grad_and_get_pointer<T>(this->ctx_, !accum[0]);

    (void)g_x;
    (void)g_y;
    NBLA_ERROR(error_code::not_implemented,
               "NonZero backward is currently not implemented.");
  }
}
}
