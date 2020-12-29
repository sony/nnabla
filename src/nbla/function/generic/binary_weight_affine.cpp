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
#include <nbla/function/abs.hpp>
#include <nbla/function/affine.hpp>
#include <nbla/function/binary_weight_affine.hpp>
#include <nbla/function/mul2.hpp>
#include <nbla/function/mul_scalar.hpp>
#include <nbla/function/sign.hpp>
#include <nbla/function/sum.hpp>
#include <nbla/function/transpose.hpp>
#include <nbla/variable.hpp>

namespace nbla {

NBLA_REGISTER_FUNCTION_SOURCE(BinaryWeightAffine, int, float);

template <typename T>
void BinaryWeightAffine<T>::setup_impl(const Variables &inputs,
                                       const Variables &outputs) {
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
  for (Shape_t::size_type i = 0; i < inputs[1]->shape().size(); ++i) {
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
             "Size of alpha must be equal to output size.");

  transpose_ = create_Transpose(this->ctx_, vector<int>{1, 0});
  abs_ = create_Abs(this->ctx_);
  sum_ = create_Sum(this->ctx_, vector<int>{1}, false);
  div_ = create_MulScalar(this->ctx_, (T)1 / w_row_, false);
  bin_ = create_Sign(this->ctx_, quantize_zero_to_);
  mul_ = create_Mul2(this->ctx_, false);
  scaled_weights_.reshape(shape_weights, true);
}

template <typename T>
void BinaryWeightAffine<T>::forward_impl(const Variables &inputs,
                                         const Variables &outputs) {
  // Binarization of the weights to +/-H where H is the L1 norm of the
  // soft weights for each output

  Shape_t shape_weights = inputs[1]->shape();
  Shape_t shape_alpha = inputs[3]->shape();

  inputs[1]->reshape(Shape_t{w_row_, w_col_}, false);

  transpose_->setup(Variables{inputs[1]}, Variables{&scaled_weights_});
  transpose_->forward(Variables{inputs[1]}, Variables{&scaled_weights_});

  // compute absolute weight values
  abs_->setup(Variables{&scaled_weights_}, Variables{&scaled_weights_});
  abs_->forward(Variables{&scaled_weights_}, Variables{&scaled_weights_});

  // compute sums of absolute weights into inputs[3]
  sum_->setup(Variables{&scaled_weights_}, Variables{inputs[3]});
  sum_->forward(Variables{&scaled_weights_}, Variables{inputs[3]});

  // compute alpha in inputs[3] by multiply with 1/col_w_
  div_->setup(Variables{inputs[3]}, Variables{inputs[3]});
  div_->forward(Variables{inputs[3]}, Variables{inputs[3]});

  // binarize weights to +1/-1 into inputs[3] (binary_weights parameter)
  bin_->setup(Variables{inputs[1]}, Variables{inputs[2]});
  bin_->forward(Variables{inputs[1]}, Variables{inputs[2]});

  // multiply binarized weights with alpha using mul2 broadcasting
  inputs[3]->reshape(Shape_t{1, w_col_}, false);
  mul_->setup(Variables{inputs[2], inputs[3]}, Variables{&scaled_weights_});
  mul_->forward(Variables{inputs[2], inputs[3]}, Variables{&scaled_weights_});

  // reshape float/binary weights and alpha back to original
  scaled_weights_.reshape(shape_weights, false);
  inputs[1]->reshape(shape_weights, false);
  inputs[2]->reshape(shape_weights, false);
  inputs[3]->reshape(shape_alpha, false);

  // calculate the forward pass using the binarized scaled weights
  if (inputs.size() == 5) { // with bias
    affine_->forward(Variables{inputs[0], &scaled_weights_, inputs[4]},
                     outputs);
  } else {
    affine_->forward(Variables{inputs[0], &scaled_weights_}, outputs);
  }
}

template <typename T>
void BinaryWeightAffine<T>::backward_impl(const Variables &inputs,
                                          const Variables &outputs,
                                          const vector<bool> &prop_down,
                                          const vector<bool> &accum) {
  // calculate the backward pass using the binarized scaled weights
  if (inputs.size() == 5) { // with bias
    affine_->backward(Variables{inputs[0], &scaled_weights_, inputs[4]},
                      outputs, {prop_down[0], prop_down[1], prop_down[4]},
                      {accum[0], false, accum[4]});
  } else { // without bias
    affine_->backward(Variables{inputs[0], &scaled_weights_}, outputs,
                      {prop_down[0], prop_down[1]}, {accum[0], false});
  }
  if (!prop_down[1]) {
    return;
  }
  // add the calculated gradient w.r.t. the binary weights from
  // scaled_weights_ to inputs[1]
  bin_->setup(Variables{inputs[1]}, Variables{&scaled_weights_});
  bin_->backward(Variables{inputs[1]}, Variables{&scaled_weights_},
                 {prop_down[1]}, {accum[1]});
}

} // namespace nbla
