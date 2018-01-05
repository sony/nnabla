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

// convolution.cpp

#include <nbla/array.hpp>
#include <nbla/function/deconvolution.hpp>
#include <nbla/utils/eigen.hpp>
#include <nbla/utils/im2col.hpp>
#include <nbla/variable.hpp>

#include <algorithm>
#include <cstring>
#include <memory>

namespace nbla {

NBLA_REGISTER_FUNCTION_SOURCE(Deconvolution, int,  // base_axis
                              const vector<int> &, // pad
                              const vector<int> &, // stride
                              const vector<int> &, // dilation
                              int);                // group

template <typename T>
void Deconvolution<T>::setup_impl(const Variables &inputs,
                                  const Variables &outputs) {
  // Shape check
  Shape_t shape_out = inputs[0]->shape();
  Shape_t shape_weights = inputs[1]->shape();
  NBLA_CHECK(base_axis_ < shape_out.size() - 1, error_code::unclassified,
             "base_axis must be less than ndim - 1 of inputs[0]. "
             "base_axis: %d >= ndim of inputs[0] - 1: %d.",
             base_axis_, shape_out.size() - 1);
  spatial_dims_ = shape_out.size() - base_axis_ - 1;
  NBLA_CHECK(shape_weights.size() == 2 + spatial_dims_, error_code::value,
             "Weights must be a tensor more than 3D.");
  // Storing shape variables
  channels_i_ = shape_weights[1] * group_;
  channels_o_ = shape_weights[0];
  channels_g_ = shape_weights[1];
  inner_size_k_ = channels_g_;
  const int channels_i_mod_group = channels_i_ % group_;
  NBLA_CHECK(channels_i_mod_group == 0, error_code::value,
             "Number of input channel needs to be divisible by group. "
             "Input channel: %d, group: %d",
             channels_i_, group_);
  const int channels_o_mod_group = channels_o_ % group_;
  NBLA_CHECK(channels_o_mod_group == 0, error_code::value,
             "Number of output channel needs to be divisible by group. "
             "Output channel: %d, group: %d",
             channels_o_, group_);
  NBLA_CHECK(channels_i_ / group_ == channels_g_, error_code::value,
             "Number of grouped channel mismatch."
             "Input: %d != Weights[1]: %d",
             channels_i_ / group_, channels_g_);
  NBLA_CHECK(pad_.size() == spatial_dims_, error_code::value,
             "pad size mismatch. pad size: %d != spatial dims: %d.",
             pad_.size(), spatial_dims_);
  NBLA_CHECK(stride_.size() == spatial_dims_, error_code::value,
             "stride size mismatch. stride size: %d != spatial dims: %d.",
             stride_.size(), spatial_dims_);
  NBLA_CHECK(dilation_.size() == spatial_dims_, error_code::value,
             "dilation size mismatch. dilation size: %d != spatial dims: %d.",
             dilation_.size(), spatial_dims_);
  for (int i = 0; i < spatial_dims_; ++i) {
    kernel_.push_back(shape_weights[2 + i]);
    inner_size_k_ *= kernel_[i];
    spatial_shape_o_.push_back(shape_out[base_axis_ + 1 + i]);
    const int k = dilation_[i] * (kernel_[i] - 1) + 1;
    const int size_i = stride_[i] * (spatial_shape_o_[i] - 1) + k - 2 * pad_[i];
    NBLA_CHECK(
        size_i > 0, error_code::value,
        "Invalid configuration of deconvolution at %d-th spatial dimension. "
        "{input:%d, kernel:%d, pad:%d, stride:%d, dilation:%d}.",
        i, size_i, kernel_[i], pad_[i], stride_[i], dilation_[i]);
    spatial_shape_i_.push_back(size_i);
  }

  // Reshaping output
  Shape_t shape_data;
  outer_size_ = 1;
  for (int i = 0; i < base_axis_; ++i) { // Fill shapes up to base axis
    shape_data.push_back(shape_out[i]);
    outer_size_ *= shape_out[i];
  }
  shape_data.push_back(channels_i_); // output channels
  inner_size_i_ = channels_i_;
  inner_size_o_ = channels_o_;
  for (int i = 0; i < spatial_dims_; ++i) {
    shape_data.push_back(spatial_shape_i_[i]);
    inner_size_i_ *= spatial_shape_i_[i];
    inner_size_o_ *= spatial_shape_o_[i];
  }
  outputs[0]->reshape(shape_data, true);

  // Reshaping col buffer
  // Actual memory is not allocated until it is used.
  col_.reshape(Shape_t{inner_size_k_ * group_, inner_size_o_ / channels_o_},
               true);

  // Check for with bias
  if (inputs.size() == 3) {
    NBLA_CHECK(inputs[2]->shape().size() == 1, error_code::value,
               "Bias(inputs[2]) must be a 1d tensor.");
    NBLA_CHECK(inputs[2]->shape()[0] == channels_i_, error_code::value,
               "Shape of bias(inputs[2]) and weights(inputs[1]) mismatch. "
               "bias shape[0]: %d != weights shape[1] * group: %d.",
               inputs[2]->shape()[0], channels_i_);
  }
  // Check for with bias
  if (inputs.size() == 3) {
    NBLA_CHECK(inputs[2]->shape().size() == 1, error_code::value, "");
    NBLA_CHECK(inputs[2]->shape()[0] == channels_i_, error_code::value, "");
  }

  // Set variables for convolution by matrix multiplication
  // In 2D case:
  // K: in maps, H: in height, W: in width
  // K': out maps, H': out height, W': out height
  // M: kernel height, N: kernel width
  row_w_ = channels_o_ / group_;          // K'
  col_w_ = inner_size_k_;                 // KMN
  row_col_ = col_w_;                      // KMN
  col_col_ = inner_size_o_ / channels_o_; // H'W'
  row_y_ = channels_o_ / group_;          // K'
  col_y_ = col_col_;                      // H'W'
}

template <class T>
void Deconvolution<T>::forward_impl(const Variables &inputs,
                                    const Variables &outputs) {
  using namespace ::nbla::eigen;
  // Getting variable pointers
  const T *y = inputs[0]->get_data_pointer<T>(this->ctx_);
  const T *w = inputs[1]->get_data_pointer<T>(this->ctx_);
  T *col = col_.cast_data_and_get_pointer<T>(this->ctx_);
  T *x = outputs[0]->cast_data_and_get_pointer<T>(this->ctx_);
  const T *b;
  if (inputs.size() == 3) {
    b = inputs[2]->get_data_pointer<T>(this->ctx_);
  }

  // Sample loop
  for (int n = 0; n < outer_size_; ++n) {

    // matrix multiplication
    const T *y_n = y + n * inner_size_o_;
    for (int g = 0; g < group_; ++g) {
      ConstMatrixMap<T> mw(w + g * row_w_ * col_w_, row_w_, col_w_);
      ConstMatrixMap<T> my(y_n + g * row_y_ * col_y_, row_y_, col_y_);
      MatrixMap<T> mcol(col + g * row_col_ * col_col_, row_col_, col_col_);
      mcol = mw.transpose() * my;
    }

    // col2im for w * x
    T *x_n = x + n * inner_size_i_;
    memset(x_n, 0, sizeof(*x_n) * inner_size_i_);
    if (spatial_dims_ == 2) {
      col2im<T>(col, channels_i_, spatial_shape_i_.data(), kernel_.data(),
                pad_.data(), stride_.data(), dilation_.data(), x_n);
    } else {
      col2im_nd<T>(col, channels_i_, spatial_dims_, spatial_shape_i_.data(),
                   kernel_.data(), pad_.data(), stride_.data(),
                   dilation_.data(), x_n);
    }

    // adding bias
    if (inputs.size() == 3) {
      MatrixMap<T> mx(x_n, channels_i_, inner_size_i_ / channels_i_);
      mx.colwise() += ConstColVectorMap<T>(b, channels_i_);
    }
  }
}

template <class T>
void Deconvolution<T>::backward_impl(const Variables &inputs,
                                     const Variables &outputs,
                                     const vector<bool> &propagate_down,
                                     const vector<bool> &accum) {

  if (!(propagate_down[0] || propagate_down[1] ||
        (inputs.size() == 3 && propagate_down[2]))) {
    return;
  }

  using namespace ::nbla::eigen;
  const T *dx = outputs[0]->get_grad_pointer<T>(this->ctx_);
  const T *y;
  const T *w;
  T *dy, *dw, *db, *col;
  std::unique_ptr<ColVectorMap<T>> mdb;

  if (propagate_down[0] || propagate_down[1]) {
    col = col_.cast_data_and_get_pointer<T>(this->ctx_);
  }
  if (propagate_down[0]) {
    w = inputs[1]->get_data_pointer<T>(this->ctx_);
    dy = inputs[0]->cast_grad_and_get_pointer<T>(this->ctx_);
  }
  if (propagate_down[1]) {
    if (!accum[1])
      inputs[1]->grad()->zero();
    y = inputs[0]->get_data_pointer<T>(this->ctx_);
    dw = inputs[1]->cast_grad_and_get_pointer<T>(this->ctx_);
  }
  if (inputs.size() == 3 && propagate_down[2]) {
    if (!accum[2])
      inputs[2]->grad()->zero();
    db = inputs[2]->cast_grad_and_get_pointer<T>(this->ctx_);
    mdb.reset(new ColVectorMap<T>(db, channels_i_));
  }

  // Sample loop
  for (int n = 0; n < outer_size_; ++n) {
    const T *dx_n = dx + n * inner_size_i_;

    if (propagate_down[0] || propagate_down[1]) {
      // im2col
      if (spatial_dims_ == 2) {
        im2col<T>(dx_n, channels_i_, spatial_shape_i_.data(), kernel_.data(),
                  pad_.data(), stride_.data(), dilation_.data(), col);
      } else {
        im2col_nd<T>(dx_n, channels_i_, spatial_dims_, spatial_shape_i_.data(),
                     kernel_.data(), pad_.data(), stride_.data(),
                     dilation_.data(), col);
      }
    }

    if (propagate_down[0]) {
      // Backprop to image
      T *dy_n = dy + n * inner_size_o_;
      for (int g = 0; g < group_; ++g) {
        ConstMatrixMap<T> mcol(col + g * row_col_ * col_col_, row_col_,
                               col_col_);
        ConstMatrixMap<T> mw(w + g * row_w_ * col_w_, row_w_, col_w_);
        MatrixMap<T> mdy(dy_n + g * row_y_ * col_y_, row_y_, col_y_);
        if (accum[0])
          mdy += mw * mcol;
        else
          mdy = mw * mcol;
      }
    }

    if (propagate_down[1]) {
      // Backprop to weights
      const T *y_n = y + n * inner_size_o_;
      for (int g = 0; g < group_; ++g) {
        ConstMatrixMap<T> mcol(col + g * row_col_ * col_col_, row_col_,
                               col_col_);
        ConstMatrixMap<T> my(y_n + g * row_y_ * col_y_, row_y_, col_y_);
        MatrixMap<T> mdw(dw + g * row_w_ * col_w_, row_w_, col_w_);
        mdw += my * mcol.transpose();
      }
    }

    if (inputs.size() == 3 && propagate_down[2]) {
      // Backprop to bias
      ConstMatrixMap<T> mdx(dx_n, channels_i_, inner_size_i_ / channels_i_);
      *mdb += mdx.rowwise().sum();
    }
  }
}

}
