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

/** BinaryWeightConvolution
 */
#include <nbla/function/abs.hpp>
#include <nbla/function/binary_weight_convolution.hpp>
#include <nbla/function/convolution.hpp>
#include <nbla/function/mul2.hpp>
#include <nbla/function/mul_scalar.hpp>
#include <nbla/function/sign.hpp>
#include <nbla/function/sum.hpp>
#include <nbla/variable.hpp>

namespace nbla {

NBLA_REGISTER_FUNCTION_SOURCE(BinaryWeightConvolution, int, const vector<int> &,
                              const vector<int> &, const vector<int> &, int,
                              float);

template <typename T>
void BinaryWeightConvolution<T>::setup_impl(const Variables &inputs,
                                            const Variables &outputs) {
  // Initialize internal `convolution` function
  convolution_ =
      create_Convolution(this->ctx_, this->base_axis_, this->pad_,
                         this->stride_, this->dilation_, this->group_, false);
  if (inputs.size() == 5) { // with bias
    convolution_->setup(Variables{inputs[0], inputs[1], inputs[4]}, outputs);
  } else { // without bias
    convolution_->setup(Variables{inputs[0], inputs[1]}, outputs);
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
  channels_o_ = shape_weights[0];
  col_w_ = inputs[1]->size() / channels_o_;

  abs_ = create_Abs(this->ctx_);
  sum_ = create_Sum(this->ctx_, vector<int>{1}, false);
  div_ = create_MulScalar(this->ctx_, (T)1 / col_w_, false);
  bin_ = create_Sign(this->ctx_, quantize_zero_to_);
  mul_ = create_Mul2(this->ctx_, false);
  scaled_weights_.reshape(shape_weights, true);
}

template <typename T>
void BinaryWeightConvolution<T>::forward_impl(const Variables &inputs,
                                              const Variables &outputs) {
  // Binarization of the weights to +/-H where H is the L1 norm of the
  // soft weights for each output

  Shape_t shape_weights = inputs[1]->shape();
  Shape_t shape_alpha = inputs[3]->shape();

  inputs[1]->reshape(Shape_t{channels_o_, col_w_}, false);

  // compute absolute weight values in inputs[2]
  abs_->setup(Variables{inputs[1]}, Variables{&scaled_weights_});
  abs_->forward(Variables{inputs[1]}, Variables{&scaled_weights_});

  // compute channels_o_ sums of absolute weights in inputs[3]
  sum_->setup(Variables{&scaled_weights_}, Variables{inputs[3]});
  sum_->forward(Variables{&scaled_weights_}, Variables{inputs[3]});

  // compute alpha in inputs[3] by multiply with 1/col_w_
  div_->setup(Variables{inputs[3]}, Variables{inputs[3]});
  div_->forward(Variables{inputs[3]}, Variables{inputs[3]});

  // binarize weights to +1/-1 into inputs[3] (binary_weights parameter)
  bin_->setup(Variables{inputs[1]}, Variables{inputs[2]});
  bin_->forward(Variables{inputs[1]}, Variables{inputs[2]});

  // multiply binarized weights with alpha using mul2 broadcasting
  inputs[3]->reshape(Shape_t{channels_o_, 1}, false);
  mul_->setup(Variables{inputs[2], inputs[3]}, Variables{&scaled_weights_});
  mul_->forward(Variables{inputs[2], inputs[3]}, Variables{&scaled_weights_});

  // reshape float/binary weights and alpha back to original
  scaled_weights_.reshape(shape_weights, false);
  inputs[1]->reshape(shape_weights, false);
  inputs[2]->reshape(shape_weights, false);
  inputs[3]->reshape(shape_alpha, false);

  // calculate the forward pass using the binarized scaled weights
  if (inputs.size() == 5) { // with bias
    convolution_->forward(Variables{inputs[0], &scaled_weights_, inputs[4]},
                          outputs);
  } else {
    convolution_->forward(Variables{inputs[0], &scaled_weights_}, outputs);
  }
}

template <typename T>
void BinaryWeightConvolution<T>::backward_impl(const Variables &inputs,
                                               const Variables &outputs,
                                               const vector<bool> &prop_down,
                                               const vector<bool> &accum) {

  // calculate the backward pass using the binarized scaled weights
  if (inputs.size() == 5) { // with bias
    convolution_->backward(Variables{inputs[0], &scaled_weights_, inputs[4]},
                           outputs, {prop_down[0], prop_down[1], prop_down[4]},
                           {accum[0], false, accum[4]});
  } else { // without bias
    convolution_->backward(Variables{inputs[0], &scaled_weights_}, outputs,
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
