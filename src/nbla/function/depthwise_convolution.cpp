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

/** DepthwiseConvolution
 */

#include <nbla/array.hpp>
#include <nbla/function/depthwise_convolution.hpp>
#include <nbla/utils/eigen.hpp>
#include <nbla/utils/im2col.hpp>
#include <nbla/variable.hpp>

#include <algorithm>
#include <cstring>
#include <memory>

namespace nbla {

NBLA_REGISTER_FUNCTION_SOURCE(DepthwiseConvolution,
                              int,                 // base_axis
                              const vector<int> &, // pad
                              const vector<int> &, // stride
                              const vector<int> &, // dilation
                              int);                // multiplier

template <typename T>
void DepthwiseConvolution<T>::setup_impl(const Variables &inputs,
                                         const Variables &outputs) {
  Variable *const input = inputs[0];
  Variable *const weights = inputs[1];
  Variable *const bias = (inputs.size() == 3) ? inputs[2] : nullptr;
  Variable *const output = outputs[0];

  NBLA_CHECK(multiplier_ == 1, error_code::value,
             "multiplier > 1 is yet to be implemented.");

  // Shape check
  NBLA_CHECK(base_axis_ < input->shape().size() - 1, error_code::unclassified,
             "base_axis must be less than ndim - 1 of inputs[0]. "
             "base_axis: %d >= ndim of inputs[0] - 1: %d.",
             base_axis_, input->shape().size() - 1);

  spatial_dims_ = input->shape().size() - base_axis_ - 1;
  NBLA_CHECK(weights->shape().size() == 1 + spatial_dims_, error_code::value,
             "Weights must be at least a 2D tensor.");

  // Storing shape variables
  channels_i_ = input->shape()[base_axis_];
  inner_size_k_ = 1;

  for (int i = 0; i < spatial_dims_; ++i) {
    kernel_.push_back(weights->shape()[1 + i]);
    inner_size_k_ *= kernel_[i];
    spatial_shape_i_.push_back(input->shape()[base_axis_ + 1 + i]);
    const int k = dilation_[i] * (kernel_[i] - 1) + 1;
    const int o = (spatial_shape_i_[i] + 2 * pad_[i] - k) / stride_[i] + 1;
    NBLA_CHECK(
        o > 0, error_code::value,
        "Invalid configuration of convolution at %d-th spatial dimension.  "
        "{input:%d, kernel:%d, pad:%d, stride:%d, dilation:%d}.",
        i, spatial_shape_i_[i], kernel_[i], pad_[i], stride_[i], dilation_[i]);
    spatial_shape_o_.push_back(o);
  }

  // Reshaping output
  Shape_t output_shape;
  outer_size_ = 1;
  for (int i = 0; i < base_axis_; ++i) { // Fill shapes up to base axis
    output_shape.push_back(input->shape()[i]);
    outer_size_ *= input->shape()[i];
  }
  output_shape.push_back(channels_i_); // #output channels = #input channels
  inner_size_i_ = channels_i_;
  inner_size_o_ = channels_i_;
  for (int i = 0; i < spatial_dims_; ++i) {
    output_shape.push_back(spatial_shape_o_[i]);
    inner_size_i_ *= spatial_shape_i_[i];
    inner_size_o_ *= spatial_shape_o_[i];
  }
  output->reshape(output_shape, true);

  // Reshaping col buffer
  // Actual memory is not allocated until it is used.
  col_.reshape(
      Shape_t{inner_size_k_ * channels_i_, inner_size_o_ / channels_i_}, true);

  // Check for with bias
  if (bias != nullptr) {
    NBLA_CHECK(bias->shape().size() == 1, error_code::value,
               "Bias(inputs[2]) must be a 1d tensor.");
    NBLA_CHECK(bias->shape()[0] == channels_i_, error_code::value,
               "Shape of bias(inputs[2]) and weights(inputs[1]) mismatch. "
               "bias shape[0]: %d != weights shape[0]: %d.",
               bias->shape()[0], channels_i_);
  }

  // Set variables for convolution by matrix multiplication
  // In 2D case:
  // K: in maps, H: in height, W: in width
  // K': out maps, H': out height, W': out height
  // M: kernel height, N: kernel width
  col_w_ = inner_size_k_;                 // KMN
  row_col_ = col_w_;                      // KMN
  col_col_ = inner_size_o_ / channels_i_; // H'W'
  col_y_ = col_col_;                      // H'W'
}

template <typename T>
void DepthwiseConvolution<T>::forward_impl(const Variables &inputs,
                                           const Variables &outputs) {
  using namespace ::nbla::eigen;
  // Getting variable pointers
  const T *x = inputs[0]->get_data_pointer<T>(this->ctx_);
  const T *w = inputs[1]->get_data_pointer<T>(this->ctx_);
  T *col = col_.cast_data_and_get_pointer<T>(this->ctx_);
  T *y = outputs[0]->cast_data_and_get_pointer<T>(this->ctx_);
  const T *b;
  if (inputs.size() == 3) {
    b = inputs[2]->get_data_pointer<T>(this->ctx_);
  }
  // Sample loop
  for (int n = 0; n < outer_size_; ++n) {
    // Im2col
    if (spatial_dims_ == 2) {
      im2col<T>(x + n * inner_size_i_, channels_i_, spatial_shape_i_.data(),
                kernel_.data(), pad_.data(), stride_.data(), dilation_.data(),
                col);
    } else {
      im2col_nd<T>(x + n * inner_size_i_, channels_i_, spatial_dims_,
                   spatial_shape_i_.data(), kernel_.data(), pad_.data(),
                   stride_.data(), dilation_.data(), col);
    }
    // Convolution by matrix multiplication
    T *y_n = y + n * inner_size_o_;
    for (int g = 0; g < channels_i_; ++g) {
      MatrixMap<T> mcol(col + g * row_col_ * col_col_, row_col_, col_col_);
      ConstRowVectorMap<T> mk(w + g * col_w_, col_w_);
      RowVectorMap<T> my(y_n + g * col_y_, col_y_);
      my = mk * mcol;
    }
    // Adding bias
    if (inputs.size() == 3) {
      MatrixMap<T> my(y_n, channels_i_, col_y_);
      my.colwise() += ConstColVectorMap<T>(b, channels_i_);
    }
  }
}

template <typename T>
void DepthwiseConvolution<T>::backward_impl(const Variables &inputs,
                                            const Variables &outputs,
                                            const vector<bool> &propagate_down,
                                            const vector<bool> &accum) {
  if (!(propagate_down[0] || propagate_down[1] ||
        (inputs.size() == 3 && propagate_down[2]))) {
    return;
  }
  using namespace ::nbla::eigen;
  const T *dy = outputs[0]->get_grad_pointer<T>(this->ctx_);
  const T *x;
  const T *w;
  T *dx, *dw, *db, *col;
  std::unique_ptr<ColVectorMap<T>> mdb;
  if (propagate_down[0] || propagate_down[1]) {
    col = col_.cast_data_and_get_pointer<T>(this->ctx_);
  }
  if (propagate_down[0]) {
    w = inputs[1]->get_data_pointer<T>(this->ctx_);
    dx = inputs[0]->cast_grad_and_get_pointer<T>(this->ctx_);
  }
  if (propagate_down[1]) {
    x = inputs[0]->get_data_pointer<T>(this->ctx_);
    dw = inputs[1]->cast_grad_and_get_pointer<T>(this->ctx_);
    if (!accum[1])
      memset(dw, 0, sizeof(*dw) * inputs[1]->size());
  }
  if (inputs.size() == 3 && propagate_down[2]) {
    db = inputs[2]->cast_grad_and_get_pointer<T>(this->ctx_);
    mdb.reset(new ColVectorMap<T>(db, channels_i_));
    if (!accum[2])
      // mdb = 0; // Results in segfault
      memset(db, 0, sizeof(*db) * inputs[2]->size());
  }
  // Sample loop
  for (int n = 0; n < outer_size_; ++n) {
    const T *dy_n = dy + n * inner_size_o_;
    if (propagate_down[0]) {
      // Backprop to image
      T *dx_n = dx + n * inner_size_i_;
      for (int g = 0; g < channels_i_; ++g) {
        ConstRowVectorMap<T> mdy(dy_n + g * col_y_, col_y_);
        ConstRowVectorMap<T> mw(w + g * col_w_, col_w_);
        MatrixMap<T> mdx(col + g * row_col_ * col_col_, row_col_, col_col_);
        mdx = mw.transpose() * mdy;
      }
      // col2im
      if (spatial_dims_ == 2) {
        if (!accum[0])
          // Remove this by substituting at n=0
          memset(dx_n, 0, sizeof(*dx_n) * inner_size_i_);
        col2im(col, channels_i_, spatial_shape_i_.data(), kernel_.data(),
               pad_.data(), stride_.data(), dilation_.data(), dx_n);
      } else {
        if (!accum[0])
          memset(dx_n, 0, sizeof(*dx_n) * inner_size_i_);
        col2im_nd(col, channels_i_, spatial_dims_, spatial_shape_i_.data(),
                  kernel_.data(), pad_.data(), stride_.data(), dilation_.data(),
                  dx_n);
      }
    }
    if (propagate_down[1]) {
      // Backprop to weights
      // im2col
      if (spatial_dims_ == 2) {
        im2col<T>(x + n * inner_size_i_, channels_i_, spatial_shape_i_.data(),
                  kernel_.data(), pad_.data(), stride_.data(), dilation_.data(),
                  col);
      } else {
        im2col_nd<T>(x + n * inner_size_i_, channels_i_, spatial_dims_,
                     spatial_shape_i_.data(), kernel_.data(), pad_.data(),
                     stride_.data(), dilation_.data(), col);
      }
      // Weight convolution by matrix multiplication
      for (int g = 0; g < channels_i_; ++g) {
        ConstRowVectorMap<T> mdy(dy_n + g * col_y_, col_y_);
        ConstMatrixMap<T> mcol(col + g * row_col_ * col_col_, row_col_,
                               col_col_);
        RowVectorMap<T> mdw(dw + g * col_w_, col_w_);
        mdw += mdy * mcol.transpose();
      }
    }
    if (inputs.size() == 3 && propagate_down[2]) {
      // Backprop to bias
      ConstMatrixMap<T> mdy(dy_n, channels_i_, col_y_);
      *mdb += mdy.rowwise().sum();
    }
  }
}

// Template instanciation
template class DepthwiseConvolution<float>;
} // namespace nbla
