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
#include <nbla/function/batch_logdet.hpp>
#include <nbla/variable.hpp>

#include <nbla/function/abs.hpp>
#include <nbla/function/batch_det.hpp>
#include <nbla/function/log.hpp>
#include <nbla/imperative.hpp>

namespace nbla {

NBLA_REGISTER_FUNCTION_SOURCE(BatchLogdet);

template <typename T>
void BatchLogdet<T>::setup_impl(const Variables &inputs,
                                const Variables &outputs) {
  NBLA_CHECK(inputs[0]->ndim() == 3, error_code::value,
             "Input must be 2D array");
  auto input_shape = inputs[0]->shape();
  NBLA_CHECK(input_shape[1] == input_shape[2], error_code::value,
             "Input must be square matrix");

  const auto batch_size = input_shape[0];
  outputs[0]->reshape(Shape_t{batch_size}, true);

  // functions
  f_batch_det_ = create_BatchDet(ctx_);
  f_abs_ = create_Abs(ctx_);
  f_log_ = create_Log(ctx_);
}

template <typename T>
void BatchLogdet<T>::forward_impl(const Variables &inputs,
                                  const Variables &outputs) {
  auto x = inputs[0];
  auto y = outputs[0];

  // batch_det -> abs -> log
  execute(f_batch_det_, Variables{x}, Variables{y});
  execute(f_abs_, Variables{y}, Variables{y}); // in-place
  execute(f_log_, Variables{y}, Variables{y}); // in-place
}

template <typename T>
void BatchLogdet<T>::backward_impl(const Variables &inputs,
                                   const Variables &outputs,
                                   const vector<bool> &propagate_down,
                                   const vector<bool> &accum) {
  const bool prop_down = propagate_down[0];
  if (!prop_down) {
    return;
  }

  auto x = inputs[0];
  auto y = outputs[0];

  // forward
  // batch_det -> abs -> log
  Variable out_batch_det, out_abs;
  execute(f_batch_det_, Variables{x}, Variables{&out_batch_det});
  execute(f_abs_, Variables{&out_batch_det}, Variables{&out_abs});
  // log forward is skipped since its output is already in data region of
  // outputs[0]
  // execute(f_log_, Variables{&out_abs}, Variables{y});

  // backward
  nbla::backward(f_log_, Variables{&out_abs}, Variables{y}, {prop_down},
                 {false}, true /* with_setup */);
  nbla::backward(f_abs_, Variables{&out_batch_det}, Variables{&out_abs},
                 {prop_down}, {false});
  nbla::backward(f_batch_det_, Variables{x}, Variables{&out_batch_det},
                 {prop_down}, {accum[0]});
}
}
