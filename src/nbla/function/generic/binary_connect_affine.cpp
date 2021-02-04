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

// binary_connect_affine.cpp

#include <nbla/array.hpp>
#include <nbla/function/binary_connect_affine.hpp>
#include <nbla/function/sign.hpp>
#include <nbla/variable.hpp>

#include <algorithm>

namespace nbla {

NBLA_REGISTER_FUNCTION_SOURCE(BinaryConnectAffine, int, float);

template <typename T>
void BinaryConnectAffine<T>::setup_impl(const Variables &inputs,
                                        const Variables &outputs) {
  // Initialize function to binarize (we use the `sign` function with alpha =
  // 1.0)
  sign_ = create_Sign(this->ctx_, quantize_zero_to_);
  sign_->setup(Variables{inputs[1]}, Variables{inputs[2]});

  // Initialize internal `affine` function
  affine_ = create_Affine(this->ctx_, this->base_axis_);
  if (inputs.size() == 4) { // with bias
    affine_->setup(Variables{inputs[0], inputs[1], inputs[3]}, outputs);
  } else { // without bias
    affine_->setup(Variables{inputs[0], inputs[1]}, outputs);
  }

  // Check shape of binarized weights matrix
  NBLA_CHECK(inputs[1]->shape().size() == inputs[2]->shape().size(),
             error_code::value,
             "Binary and float weights must have same size. "
             "Ndim of inputs[1]: %d != ndim of inputs[2]: %d.",
             inputs[1]->shape().size(), inputs[2]->shape().size());
  for (Shape_t::size_type i = 0; i < inputs[1]->shape().size(); ++i) {
    NBLA_CHECK(inputs[1]->shape()[i] == inputs[2]->shape()[i],
               error_code::value,
               "Binary and float weights must have same size. "
               "float shape[%d]: %d != binary shape[%d]: %d.",
               i, inputs[1]->shape()[i], i, inputs[2]->shape()[i]);
  }
}

template <class T>
void BinaryConnectAffine<T>::forward_impl(const Variables &inputs,
                                          const Variables &outputs) {
  // binarize weights, put values into inputs[2]
  sign_->forward(Variables{inputs[1]}, Variables{inputs[2]});

  // calculate the forward pass using binarized weights `inputs[2]`
  if (inputs.size() == 4) { // with bias
    affine_->forward(Variables{inputs[0], inputs[2], inputs[3]}, outputs);
  } else {
    affine_->forward(Variables{inputs[0], inputs[2]}, outputs);
  }
}

template <class T>
void BinaryConnectAffine<T>::backward_impl(const Variables &inputs,
                                           const Variables &outputs,
                                           const vector<bool> &prop_down,
                                           const vector<bool> &accum) {
  // calculate the backward pass using the already binarized weights
  if (inputs.size() == 4) { // with bias
    affine_->backward(Variables{inputs[0], inputs[2], inputs[3]}, outputs,
                      {prop_down[0], prop_down[1], prop_down[3]},
                      {accum[0], false, accum[3]});
  } else { // without bias
    affine_->backward(Variables{inputs[0], inputs[2]}, outputs,
                      {prop_down[0], prop_down[1]}, {accum[0], false});
  }
  if (!prop_down[1]) {
    return;
  }
  // add the calculated gradient wrt the binary weights from inputs[2] to
  // inputs[1]
  sign_->backward(Variables{inputs[1]}, Variables{inputs[2]}, {prop_down[1]},
                  {accum[1]});
}
}
