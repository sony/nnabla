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
#include <nbla/function/meshgrid.hpp>
#include <nbla/variable.hpp>

#include <nbla/imperative.hpp>

namespace nbla {

NBLA_REGISTER_FUNCTION_SOURCE(Meshgrid, bool);

template <typename T>
void Meshgrid<T>::setup_impl(const Variables &inputs,
                             const Variables &outputs) {

  bc_fn_.clear();
  input_reshaped_shapes_.clear();

  int n = inputs.size();

  const Shape_t in_shape = inputs[0]->shape();

  vector<int64_t> shape_out(n, 1);

  for (int i = 0; i < n; i++) {
    NBLA_CHECK(
        inputs[i]->ndim() == 1, error_code::value,
        "Input %d is not a Rank 1 array. All inputs need be Rank 1 and ndim<2",
        i);

    vector<int64_t> shape_inp(n, 1);

    if (!ij_indexing_ && (i == 0 || i == 1) && n > 1) {
      shape_out[1 - i] = inputs[i]->shape()[0];
      shape_inp[1 - i] = inputs[i]->shape()[0];
    } else {
      shape_out[i] = inputs[i]->shape()[0];
      shape_inp[i] = inputs[i]->shape()[0];
    }

    input_reshaped_shapes_.push_back(shape_inp);
  }

  vector<int> shape_out_(shape_out.begin(), shape_out.end());

  for (int i = 0; i < n; i++) {
    outputs[i]->reshape(shape_out, true);
    bc_fn_.push_back(create_Broadcast(this->ctx_, shape_out_));
  }
}

template <typename T>
void Meshgrid<T>::forward_impl(const Variables &inputs,
                               const Variables &outputs) {

  int n = inputs.size();
  Shape_t shape_(1);

  for (int i = 0; i < n; i++) {
    inputs[i]->reshape(input_reshaped_shapes_[i], true);
    auto x = inputs[i];
    auto xx = outputs[i];

    execute(bc_fn_[i], Variables{x}, Variables{xx});

    if (!ij_indexing_ && (i == 0 || i == 1) && n > 1)
      shape_[0] = inputs[i]->shape()[1 - i];
    else
      shape_[0] = inputs[i]->shape()[i];

    inputs[i]->reshape(shape_, true);
  }
}

template <typename T>
void Meshgrid<T>::backward_impl(const Variables &inputs,
                                const Variables &outputs,
                                const vector<bool> &propagate_down,
                                const vector<bool> &accum) {

  if (!(propagate_down[0] || propagate_down[1])) {
    return;
  }

  const int n = inputs.size();
  Shape_t shape_(1);
  for (int i = 0; i < n; i++) {

    const auto prop_down = propagate_down[i];
    const auto accum_ = accum[i];

    inputs[i]->reshape(input_reshaped_shapes_[i], true);

    auto x = inputs[i];
    auto xx = outputs[i];

    nbla::backward(bc_fn_[i], Variables{x}, Variables{xx}, {prop_down},
                   {accum_});

    if (!ij_indexing_ && (i == 0 || i == 1) && n > 1)
      shape_[0] = inputs[i]->shape()[1 - i];
    else
      shape_[0] = inputs[i]->shape()[i];

    inputs[i]->reshape(shape_, true);
  }
}
}