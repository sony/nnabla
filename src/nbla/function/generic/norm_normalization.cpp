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
#include <nbla/function/norm_normalization.hpp>
#include <nbla/variable.hpp>

#include <nbla/function/div2.hpp>
#include <nbla/function/norm.hpp>
#include <nbla/imperative.hpp>

namespace nbla {

NBLA_REGISTER_FUNCTION_SOURCE(NormNormalization, float, const vector<int> &,
                              float);

template <typename T>
void NormNormalization<T>::setup_impl(const Variables &inputs,
                                      const Variables &outputs) {
#ifdef __APPLE__
  NBLA_ERROR(error_code::target_specific,
             "NormNormalization is not supported in macOS.");
#endif

  NBLA_CHECK(p_ >= 1, error_code::value,
             "`p` must be greater than or equal to 1. (p = %f)", p_);
  // Set output shape
  const auto inshape = inputs[0]->shape();
  outputs[0]->reshape(inshape, true);

  // functions
  norm_ = create_Norm(ctx_, p_, axes_, true /* keep_dims */);
  div2_ = create_Div2(ctx_, false);
}

template <typename T>
void NormNormalization<T>::forward_impl(const Variables &inputs,
                                        const Variables &outputs) {
  auto x = inputs[0];
  auto y = outputs[0];

  // norm -> div
  Variable out_norm;
  execute(norm_, Variables{x}, Variables{&out_norm});
  execute(div2_, Variables{x, &out_norm}, Variables{y});
}

template <typename T>
void NormNormalization<T>::backward_impl(const Variables &inputs,
                                         const Variables &outputs,
                                         const vector<bool> &propagate_down,
                                         const vector<bool> &accum) {
  const auto prop_down = propagate_down[0];
  if (!prop_down) {
    return;
  }

  auto x = inputs[0];
  auto y = outputs[0];

  // forward
  // norm -> div
  Variable out_norm;
  execute(norm_, Variables{x}, Variables{&out_norm});
  execute(div2_, Variables{x, &out_norm}, Variables{y});

  // backward
  // div -> norm
  nbla::backward(div2_, Variables{x, &out_norm}, Variables{y},
                 {prop_down, prop_down}, {accum[0]});
  nbla::backward(norm_, Variables{x}, Variables{&out_norm}, {prop_down},
                 {true});
}
}
