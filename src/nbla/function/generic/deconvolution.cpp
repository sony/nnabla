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

#include "../../utils/im2col-internal.hpp"
#include <nbla/array.hpp>
#include <nbla/function/deconvolution.hpp>
#include <nbla/utils/eigen.hpp>
#include <nbla/variable.hpp>

#include <nbla/utils/fold_from_patches.hpp>
#include <nbla/utils/unfold_to_patches.hpp>

#include <algorithm>
#include <cstring>
#include <memory>

using std::max;

namespace nbla {

NBLA_REGISTER_FUNCTION_SOURCE(Deconvolution, int,   // base_axis
                              const vector<int> &,  // pad
                              const vector<int> &,  // stride
                              const vector<int> &,  // dilation
                              int,                  // group
                              bool,                 // channel_last
                              const vector<int> &); // output_padding

template <typename T>
void Deconvolution<T>::setup_impl(const Variables &inputs,
                                  const Variables &outputs) {
  // Shape check
  Shape_t shape_out = inputs[0]->shape();
  Shape_t shape_weights = inputs[1]->shape();
  NBLA_CHECK(base_axis_ >= 0, error_code::value,
             "base_axis may not be less than zero, got %d", base_axis_);
  auto base_axis = static_cast<Shape_t::size_type>(base_axis_);
  NBLA_CHECK(base_axis + 1 < shape_out.size(), error_code::value,
             "base_axis must be less than ndim - 1 of inputs[0]. "
             "base_axis: %d >= ndim of inputs[0] - 1: %d.",
             base_axis_, shape_out.size() - 1);

  size_t spatial_dims = shape_out.size() - base_axis - 1;
  NBLA_CHECK(shape_weights.size() == 2 + spatial_dims, error_code::value,
             "Weights must be a tensor more than 3D.");
  this->spatial_dims_ = static_cast<int>(spatial_dims);

  size_t channel_axis = base_axis + (channel_last_ ? spatial_dims : 0);
  size_t first_spatial_axis = base_axis + (channel_last_ ? 0 : 1);
  size_t weight_channel_axis = 1 + (channel_last_ ? spatial_dims : 0);
  size_t weight_first_spatial_axis = channel_last_ ? 1 : 2;

  // Storing shape variables
  channels_i_ = shape_weights[weight_channel_axis] * group_;
  channels_o_ = shape_weights[0];
  channels_g_ = shape_weights[weight_channel_axis];
  inner_size_k_ = channels_g_;

  // Default value is 0 for all spatial dims. Note that this had to be added
  // here because this parameter was added later and nnabla_cli does not set
  // the correct defaults when reading an older nntxt file.
  if (output_padding_.empty()) {
    output_padding_.resize(spatial_dims_);
  }

  NBLA_CHECK(channels_i_ % group_ == 0, error_code::value,
             "Number of input channel needs to be divisible by group. "
             "Input channel: %d, group: %d",
             channels_i_, group_);
  NBLA_CHECK(channels_o_ % group_ == 0, error_code::value,
             "Number of output channel needs to be divisible by group. "
             "Output channel: %d, group: %d",
             channels_o_, group_);
  NBLA_CHECK(channels_i_ / group_ == channels_g_, error_code::value,
             "Number of grouped channel mismatch."
             "Input: %d != Weights[%d]: %d",
             channels_i_ / group_, weight_channel_axis, channels_g_);
  NBLA_CHECK(pad_.size() == spatial_dims, error_code::value,
             "pad size mismatch. pad size: %d != spatial dims: %d.",
             pad_.size(), spatial_dims_);
  NBLA_CHECK(stride_.size() == spatial_dims, error_code::value,
             "stride size mismatch. stride size: %d != spatial dims: %d.",
             stride_.size(), spatial_dims_);
  NBLA_CHECK(dilation_.size() == spatial_dims, error_code::value,
             "dilation size mismatch. dilation size: %d != spatial dims: %d.",
             dilation_.size(), spatial_dims_);
  NBLA_CHECK(output_padding_.size() == spatial_dims, error_code::value,
             "output_padding size mismatch: %d != spatial dims: %d.",
             output_padding_.size(), spatial_dims_);

  kernel_.clear();
  spatial_shape_i_.clear();
  spatial_shape_o_.clear();
  for (int i = 0; i < spatial_dims_; ++i) {
    kernel_.push_back(shape_weights[weight_first_spatial_axis + i]);
    inner_size_k_ *= kernel_[i];
    spatial_shape_o_.push_back(shape_out[first_spatial_axis + i]);
    const int k = dilation_[i] * (kernel_[i] - 1) + 1;
    const int size_i = stride_[i] * (spatial_shape_o_[i] - 1) + k -
                       2 * pad_[i] + output_padding_[i];
    NBLA_CHECK(
        size_i > 0, error_code::value,
        "Invalid configuration of deconvolution at %d-th spatial dimension. "
        "{input:%d, kernel:%d, pad:%d, stride:%d, dilation:%d}.",
        i, size_i, kernel_[i], pad_[i], stride_[i], dilation_[i]);
    NBLA_CHECK(
        output_padding_[i] < stride_[i], error_code::value,
        "output padding must be smaller than either stride or dilation, but ",
        "output padding:%d, stride:%d, dilation:%d at spatial dimension %d",
        output_padding_[i], stride_[i], dilation_[i], i);
    spatial_shape_i_.push_back(size_i);
  }

  // Reshaping output
  Shape_t shape_data(shape_out.size());
  outer_size_ = 1;
  // Fill shapes up to base axis.
  for (int i = 0; i < base_axis_; ++i) {
    shape_data.at(i) = (shape_out[i]);
    outer_size_ *= shape_out[i];
  }
  // Fill output channels.
  shape_data.at(channel_axis) = channels_i_;
  inner_size_i_ = channels_i_;
  inner_size_o_ = channels_o_;
  // Fill spatial dimensions.
  for (int i = 0; i < spatial_dims_; ++i) {
    shape_data.at(first_spatial_axis + i) = spatial_shape_i_[i];
    inner_size_i_ *= spatial_shape_i_[i];
    inner_size_o_ *= spatial_shape_o_[i];
  }
  outputs[0]->reshape(shape_data, true);

  // Check for with bias
  if (inputs.size() == 3) {
    NBLA_CHECK(inputs[2]->shape().size() == 1, error_code::value,
               "Bias(inputs[2]) must be a 1d tensor.");
    NBLA_CHECK(inputs[2]->shape()[0] == channels_i_, error_code::value,
               "Shape of bias(inputs[2]) and weights(inputs[1]) mismatch. "
               "bias shape[0]: %d != weights shape[1] * group: %d.",
               inputs[2]->shape()[0], channels_i_);
  }

  // TODO: The following logic until the end of this function should be moved to
  // forward and backward function. Also we should not keep a Variable buffer as
  // a class member but use CachedArray instead for an internal buffer array.

  // Reshaping col buffer
  // Actual memory is not allocated until it is used.
  col_.reshape(Shape_t{inner_size_k_ * group_, inner_size_o_ / channels_o_},
               true);

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
  NBLA_CHECK(!channel_last_, error_code::not_implemented,
             "The passed argument channel_last_=true is not supported in CPU "
             "Deconvolution.");

  using namespace ::nbla::eigen;
  // Getting variable pointers
  const T *y = inputs[0]->get_data_pointer<T>(this->ctx_);
  const T *w = inputs[1]->get_data_pointer<T>(this->ctx_);
  T *col = col_.cast_data_and_get_pointer<T>(this->ctx_, true);
  T *x = outputs[0]->cast_data_and_get_pointer<T>(this->ctx_, true);
  const T *b = nullptr;
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
    memset((void *)x_n, 0, sizeof(*x_n) * inner_size_i_);
    fold_from_patches<T>(col, x_n, channels_i_, spatial_shape_i_, kernel_, pad_,
                         stride_, dilation_);

    // adding bias
    if (inputs.size() == 3) {
      MatrixMap<T> mx(x_n, channels_i_, inner_size_i_ / channels_i_);
      mx.colwise() += ConstColVectorMap<T>(b, channels_i_);
    }
  }
  col_.data()->array()->clear();
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

  NBLA_CHECK(!channel_last_, error_code::not_implemented,
             "The passed argument channel_last_=true is not supported in CPU "
             "Deconvolution.");

  using namespace ::nbla::eigen;
  const T *dx = outputs[0]->get_grad_pointer<T>(this->ctx_);
  const T *y = nullptr;
  const T *w = nullptr;
  T *dy = nullptr;
  T *dw = nullptr;
  T *db = nullptr;
  T *col = nullptr;
  std::unique_ptr<ColVectorMap<T>> mdb;

  if (propagate_down[0] || propagate_down[1]) {
    col = col_.cast_data_and_get_pointer<T>(this->ctx_, true);
  }
  if (propagate_down[0]) {
    w = inputs[1]->get_data_pointer<T>(this->ctx_);
    dy = inputs[0]->cast_grad_and_get_pointer<T>(this->ctx_, !accum[0]);
  }
  if (propagate_down[1]) {
    if (!accum[1])
      inputs[1]->grad()->zero();
    y = inputs[0]->get_data_pointer<T>(this->ctx_);
    dw = inputs[1]->cast_grad_and_get_pointer<T>(this->ctx_, false);
  }
  if (inputs.size() == 3 && propagate_down[2]) {
    if (!accum[2])
      inputs[2]->grad()->zero();
    db = inputs[2]->cast_grad_and_get_pointer<T>(this->ctx_, false);
    mdb.reset(new ColVectorMap<T>(db, channels_i_));
  }

  // Sample loop
  for (int n = 0; n < outer_size_; ++n) {
    const T *dx_n = dx + n * inner_size_i_;

    if (propagate_down[0] || propagate_down[1]) {
      // im2col
      unfold_to_patches<T>(dx_n, col, channels_i_, spatial_shape_i_, kernel_,
                           pad_, stride_, dilation_);
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
  col_.data()->array()->clear();
}
}
