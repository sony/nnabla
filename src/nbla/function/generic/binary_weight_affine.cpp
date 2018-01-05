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

/** BinaryWeightAffine
 */
#include <nbla/array.hpp>
#include <nbla/function/affine.hpp>
#include <nbla/function/binary_weight_affine.hpp>
#include <nbla/function/sign.hpp>
#include <nbla/variable.hpp>

#include <algorithm>
#include <cmath>

namespace nbla {

NBLA_REGISTER_FUNCTION_SOURCE(BinaryWeightAffine, int);

template <typename T>
void BinaryWeightAffine<T>::setup_impl(const Variables &inputs,
                                       const Variables &outputs) {
  // Initialize function to binarize (we use the `sign` function with alpha =
  // -1.0)
  sign_ = create_Sign(this->ctx_, -1.0);
  sign_->setup(Variables{inputs[1]}, Variables{inputs[2]});

  // Initialize internal `affine` function
  affine_ = create_Affine(this->ctx_, this->base_axis_);
  if (inputs.size() == 5) { // with bias
    affine_->setup(Variables{inputs[0], inputs[1], inputs[4]}, outputs);
  } else { // without bias
    affine_->setup(Variables{inputs[0], inputs[1]}, outputs);
  }

  // Check shape of binarized weights matrix
  NBLA_CHECK(inputs[1]->shape().size() == inputs[2]->shape().size(),
             error_code::value,
             "Binary and float weights must have same size. "
             "Ndim of inputs[1]: %d != ndim of inputs[2]: %d.",
             inputs[1]->shape().size(), inputs[2]->shape().size());
  for (int i = 0; i < inputs[1]->shape().size(); ++i) {
    NBLA_CHECK(inputs[1]->shape()[i] == inputs[2]->shape()[i],
               error_code::value,
               "Binary and float weights must have same size. "
               "float shape[%d]: %d != binary shape[%d]: %d.",
               i, inputs[1]->shape()[i], i, inputs[2]->shape()[i]);
  }

  // compute size of weights (needed for normalization in forward pass)
  Shape_t shape_weights = inputs[1]->shape();
  w_row_ = shape_weights[0];
  w_col_ = inputs[1]->size() / w_row_;

  // Check size of alpha.
  NBLA_CHECK(inputs[3]->size() == w_col_, error_code::value,
             "Size of alpha must equal to output size.");
}

template <typename T>
void BinaryWeightAffine<T>::forward_impl(const Variables &inputs,
                                         const Variables &outputs) {
  // binarize weights, put values into inputs[2]
  sign_->forward(Variables{inputs[1]}, Variables{inputs[2]});

  // binarization of the weights to +/-H where H is the L1 norm of the soft
  // weights for each output
  const T *w = inputs[1]->get_data_pointer<T>(this->ctx_);
  T *w_bin = inputs[2]->cast_data_and_get_pointer<T>(this->ctx_);
  T *alpha = inputs[3]->cast_data_and_get_pointer<T>(this->ctx_);
  T scale;

  for (int c = 0; c < w_col_; ++c) {
    scale = 0.0;
    for (int r = 0; r < w_row_; ++r) {
      scale += std::abs(w[c + r * w_col_]);
    }
    scale /= w_row_;
    for (int r = 0; r < w_row_; ++r) {
      w_bin[c + r * w_col_] *= scale;
    }
    alpha[c] = scale;
  }

  // calculate the forward pass using binarized weights `inputs[2]`
  if (inputs.size() == 5) { // with bias
    affine_->forward(Variables{inputs[0], inputs[2], inputs[4]}, outputs);
  } else {
    affine_->forward(Variables{inputs[0], inputs[2]}, outputs);
  }
}

template <typename T>
void BinaryWeightAffine<T>::backward_impl(const Variables &inputs,
                                          const Variables &outputs,
                                          const vector<bool> &propagate_down,
                                          const vector<bool> &accum) {
  if (propagate_down[1]) {
    // reset `grad` part of binary weights to zero
    // inputs[2]->grad()->zero();

    // set `need_grad` to true for binary weights
    inputs[2]->set_need_grad(true);
  } else {
    // set `need_grad` to false for binary weights
    inputs[2]->set_need_grad(false);
  }

  // calculate the backward pass using the already binarized weights
  if (inputs.size() == 5) { // with bias
    affine_->backward(Variables{inputs[0], inputs[2], inputs[4]}, outputs,
                      {accum[0], false, accum[4]});
  } else { // without bias
    affine_->backward(Variables{inputs[0], inputs[2]}, outputs,
                      {accum[0], false});
  }

  // add the calculated gradient wrt the binary weights from inputs[2] to
  // inputs[1]
  sign_->backward(Variables{inputs[1]}, Variables{inputs[2]}, {accum[1]});

  inputs[2]->set_need_grad(
      false); // reset `need_grad` to false as we do not need
              // backward for binary variables
}

} // namespace nbla
