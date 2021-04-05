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
#include <nbla/function/deformable_convolution.hpp>
#include <nbla/utils/deformable-im2col-internal.hpp>
#include <nbla/utils/eigen.hpp>
#include <nbla/variable.hpp>

#include <algorithm>
#include <cstring>
#include <memory>

namespace nbla {

NBLA_REGISTER_FUNCTION_SOURCE(DeformableConvolution, int, // base_axis
                              const vector<int> &,        // pad
                              const vector<int> &,        // stride
                              const vector<int> &,        // dilation
                              int,                        // group
                              int,
                              bool); // channel_last

template <typename T>
void DeformableConvolution<T>::setup_impl(const Variables &inputs,
                                          const Variables &outputs) {
#if defined(__arm__) || defined(__arm64__)
  NBLA_ERROR(error_code::target_specific,
             "DeformableConvolution is not supported in arm.");
#endif

  // Shape check
  Shape_t shape_data = inputs[0]->shape();
  Shape_t shape_weights = inputs[1]->shape();
  Shape_t shape_offset = inputs[2]->shape();
  NBLA_CHECK(base_axis_ < (int)(shape_data.size() - 1),
             error_code::unclassified,
             "base_axis must be less than ndim - 1 of inputs[0]. "
             "base_axis: %d >= ndim of inputs[0] - 1: %d.",
             base_axis_, shape_data.size() - 1);
  spatial_dims_ = shape_data.size() - base_axis_ - 1;
  NBLA_CHECK((int)shape_weights.size() == 2 + spatial_dims_, error_code::value,
             "Weights must be a tensor more than 3D.");
  NBLA_CHECK(shape_offset[0] == shape_data[0] &&
                 (channel_last_ ? shape_offset[1] == shape_data[1]
                                : shape_offset[2] == shape_data[2]) &&
                 (channel_last_ ? shape_offset[2] == shape_data[2]
                                : shape_offset[3] == shape_data[3]),
             error_code::value,
             "Offset tensor must have the shape as input feature map");
  NBLA_CHECK(
      channel_last_
          ? shape_offset[3] ==
                2 * shape_weights[1] * shape_weights[2] * deformable_group_
          : shape_offset[1] ==
                2 * shape_weights[2] * shape_weights[3] * deformable_group_,
      error_code::value,
      "Offset channels must have size 2*kernel*kernel*deformable_group");

  // Check mask shape and fill inner mask size
  if (inputs.size() == 5 || (inputs.size() == 4 && inputs[3]->ndim() != 1)) {
    with_mask_ = true;
    Shape_t shape_mask = inputs[3]->shape();
    NBLA_CHECK(shape_mask[0] == shape_data[0] &&
                   (channel_last_ ? shape_mask[1] == shape_data[1]
                                  : shape_mask[2] == shape_data[2]) &&
                   (channel_last_ ? shape_mask[2] == shape_data[2]
                                  : shape_mask[3] == shape_data[3]),
               error_code::value,
               "Mask tensor must have the shape as input feature map");
    NBLA_CHECK(channel_last_
                   ? shape_mask[3] ==
                         shape_weights[1] * shape_weights[2] * deformable_group_
                   : shape_mask[1] ==
                         shape_weights[2] * shape_weights[3] *
                             deformable_group_,
               error_code::value,
               "Mask channels must have size kernel*kernel*deformable_group");
    NBLA_CHECK(channel_last_
                   ? (pad_[0] < (shape_weights[1] + (int64_t)1) / 2) ||
                         (pad_[1] < (shape_weights[2] + (int64_t)1) / 2)
                   : (pad_[0] < (shape_weights[2] + (int64_t)1) / 2) ||
                         (pad_[1] < (shape_weights[3] + (int64_t)1) / 2),
               error_code::value,
               "Center of kernel offset in the padding area. "
               "Make offset bigger or pad smaller.");
    mask_size_i_ = shape_mask[1] * shape_mask[2] * shape_mask[3];
  }

  // Storing shape variables
  size_t channel_axis = base_axis_ + (channel_last_ ? spatial_dims_ : 0);
  size_t first_spatial_axis = base_axis_ + (channel_last_ ? 0 : 1);
  size_t weight_channel_axis = 1 + (channel_last_ ? spatial_dims_ : 0);
  size_t weight_first_spatial_axis = channel_last_ ? 1 : 2;
  channels_i_ = shape_data[channel_axis];
  channels_o_ = shape_weights[0];
  channels_g_ = shape_weights[weight_channel_axis];
  inner_size_k_ = channels_g_;
  const int channels_i_mod_group = channels_i_ % group_;
  NBLA_CHECK(channels_i_mod_group == 0, error_code::value,
             "Number of input channel needs to be divisible by group. "
             "Input channel: %d, group: %d.",
             channels_i_, group_);
  const int channels_o_mod_group = channels_o_ % group_;
  NBLA_CHECK(channels_o_mod_group == 0, error_code::value,
             "Number of output channel needs to be divisible by group. "
             "Output channel: %d, group: %d.",
             channels_o_, group_);
  NBLA_CHECK(channels_i_ / group_ == channels_g_, error_code::value,
             "Number of grouped channel mismatch. "
             "Input: %d != Weights[%zu]: %d.",
             channels_i_ / group_, weight_channel_axis, channels_g_);
  const int channels_i_mod_deformable_group = channels_i_ % deformable_group_;
  NBLA_CHECK(
      channels_i_mod_deformable_group == 0, error_code::value,
      "Number of input channel needs to be divisible by deformable_group. "
      "Input channel: %d, deformable_group: %d.",
      channels_i_, deformable_group_);
  const int channels_o_mod_deformable_group = channels_o_ % deformable_group_;
  NBLA_CHECK(
      channels_o_mod_deformable_group == 0, error_code::value,
      "Number of output channel needs to be divisible by deformable_group. "
      "Output channel: %d, deformable_group: %d.",
      channels_o_, deformable_group_);
  NBLA_CHECK((int)pad_.size() == spatial_dims_, error_code::value,
             "pad size mismatch. pad size: %d != spatial dims: %d.",
             pad_.size(), spatial_dims_);
  NBLA_CHECK((int)stride_.size() == spatial_dims_, error_code::value,
             "stride size mismatch. stride size: %d != spatial dims: %d.",
             stride_.size(), spatial_dims_);
  NBLA_CHECK((int)dilation_.size() == spatial_dims_, error_code::value,
             "dilation size mismatch. dilation size: %d != spatial dims: %d.",
             dilation_.size(), spatial_dims_);
  for (int i = 0; i < spatial_dims_; ++i) {
    kernel_.push_back(shape_weights[weight_first_spatial_axis + i]);
    inner_size_k_ *= kernel_[i];
    spatial_shape_i_.push_back(shape_data[first_spatial_axis + i]);
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
  Shape_t shape_out(shape_data.size());
  outer_size_ = 1;
  // Fill shapes up to base axis
  for (int i = 0; i < base_axis_; ++i) {
    shape_out[i] = shape_data[i];
    outer_size_ *= shape_data[i];
  }

  // Fill output channels.
  shape_out[channel_axis] = channels_o_;
  inner_size_i_ = channels_i_;
  inner_size_o_ = channels_o_;
  // Fill spatial dims.
  for (int i = 0; i < spatial_dims_; ++i) {
    shape_out[first_spatial_axis + i] = spatial_shape_o_[i];
    inner_size_i_ *= spatial_shape_i_[i];
    inner_size_o_ *= spatial_shape_o_[i];
  }
  outputs[0]->reshape(shape_out, true);
  // Fill inner offset size
  offset_size_i_ = shape_offset[1] * shape_offset[2] * shape_offset[3];

  // Check for with bias
  if (inputs.size() == 5 || (inputs.size() == 4 && inputs[3]->ndim() == 1)) {
    with_bias_ = true;
    Shape_t bias_shape;
    if (inputs.size() == 5) {
      bias_shape = inputs[4]->shape();
    } else {
      bias_shape = inputs[3]->shape();
    }
    NBLA_CHECK(bias_shape.size() == 1, error_code::value,
               "Bias(inputs[4]) must be a 1d tensor.");
    NBLA_CHECK(bias_shape[0] == channels_o_, error_code::value,
               "Shape of bias(inputs[2]) and weights(inputs[1]) mismatch. "
               "bias shape[0]: %d != weights shape[0]: %d.",
               bias_shape[0], channels_o_);
  }

  // TODO: The following logic until the end of this function should be moved to
  // forward and backward function.
  // Also We should not keep a Variable buffer as a class member. Use
  // *CachedArray instead for an internal buffer array.

  // Reshaping col buffer
  // Actual memory is not allocated until it is used.
  // col shape (c_in*K*K,c_out*H_out*W_out)
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
void DeformableConvolution<T>::forward_impl(const Variables &inputs,
                                            const Variables &outputs) {
  NBLA_CHECK(!channel_last_, error_code::not_implemented,
             "The passed argument channel_last_=true is not supported in CPU "
             "Convolution.");

  using namespace ::nbla::eigen;
  // Getting variable pointers
  const T *x = inputs[0]->get_data_pointer<T>(this->ctx_);
  const T *w = inputs[1]->get_data_pointer<T>(this->ctx_);
  const T *offset = inputs[2]->get_data_pointer<T>(this->ctx_);
  const T *mask = nullptr;
  const T *b = nullptr;
  if (inputs.size() == 4) {
    if (inputs[3]->ndim() == 1) {
      b = inputs[3]->get_data_pointer<T>(this->ctx_);
    } else {
      mask = inputs[3]->get_data_pointer<T>(this->ctx_);
    }
  }
  if (inputs.size() == 5) {
    mask = inputs[3]->get_data_pointer<T>(this->ctx_);
    b = inputs[4]->get_data_pointer<T>(this->ctx_);
  }
  T *col = col_.cast_data_and_get_pointer<T>(this->ctx_, true);
  T *y = outputs[0]->cast_data_and_get_pointer<T>(this->ctx_, true);

  // Sample loop
  // outer_size_ = batch size
  for (int n = 0; n < outer_size_; ++n) {
    // Im2col
    if (this->spatial_dims_ == 2) {
      if (inputs.size() == 5 ||
          (inputs.size() == 4 && inputs[3]->ndim() != 1)) {
        modulated_deformable_im2col_cpu<T, true>(
            x + n * inner_size_i_, offset + n * offset_size_i_,
            mask + n * mask_size_i_, channels_i_, spatial_shape_i_, kernel_,
            pad_, stride_, dilation_, deformable_group_, col);
      } else {
        modulated_deformable_im2col_cpu<T, false>(
            x + n * inner_size_i_, offset + n * offset_size_i_, nullptr,
            channels_i_, spatial_shape_i_, kernel_, pad_, stride_, dilation_,
            deformable_group_, col);
      }

    } else {
      NBLA_ERROR(error_code::not_implemented,
                 "Only 2D-deformable convolution supported");
    }
    // Convolution by matrix multiplication
    T *y_n = y + n * inner_size_o_;
    for (int g = 0; g < group_; ++g) {
      MatrixMap<T> mcol(col + g * row_col_ * col_col_, row_col_, col_col_);
      ConstMatrixMap<T> mk(w + g * row_w_ * col_w_, row_w_, col_w_);
      MatrixMap<T> my(y_n + g * row_y_ * col_y_, row_y_, col_y_);
      my = mk * mcol;
    }
    // Adding bias
    if (inputs.size() == 5 || (inputs.size() == 4 && inputs[3]->ndim() == 1)) {
      MatrixMap<T> my(y_n, channels_o_, col_y_);
      my.colwise() += ConstColVectorMap<T>(b, channels_o_);
    }
  }
  col_.data()->array()->clear();
}

template <class T>
void DeformableConvolution<T>::backward_impl(const Variables &inputs,
                                             const Variables &outputs,
                                             const vector<bool> &propagate_down,
                                             const vector<bool> &accum) {
  if (!(propagate_down[0] || propagate_down[1] || propagate_down[2] ||
        (inputs.size() >= 4 && propagate_down[3]) ||
        (inputs.size() == 5 && propagate_down[4]))) {
    return;
  }
  NBLA_CHECK(!channel_last_, error_code::not_implemented,
             "The passed argument channel_last_=true is not supported in CPU "
             "Convolution.");

  using namespace ::nbla::eigen;
  const T *dy = outputs[0]->get_grad_pointer<T>(this->ctx_);
  const T *x = nullptr;
  const T *w = nullptr;
  const T *offset = nullptr;
  const T *mask = nullptr;
  T *dx = nullptr;
  T *dw = nullptr;
  T *db = nullptr;
  T *doff = nullptr;
  T *dmask = nullptr;
  T *col = nullptr;

  std::unique_ptr<ColVectorMap<T>> mdb;

  if (propagate_down[0] || propagate_down[1] || propagate_down[2]) {
    col = col_.cast_data_and_get_pointer<T>(this->ctx_, true);
    offset = inputs[2]->get_data_pointer<T>(this->ctx_);
    if (inputs.size() == 5 || (inputs.size() == 4 && inputs[3]->ndim() != 1) ||
        propagate_down[3]) {
      mask = inputs[3]->get_data_pointer<T>(this->ctx_);
    }
  }
  if (propagate_down[0]) {
    if (!accum[0])
      inputs[0]->grad()->zero();
    w = inputs[1]->get_data_pointer<T>(this->ctx_);
    dx = inputs[0]->cast_grad_and_get_pointer<T>(this->ctx_, false);
  }
  if (propagate_down[1]) {
    if (!accum[1])
      inputs[1]->grad()->zero();
    x = inputs[0]->get_data_pointer<T>(this->ctx_);
    dw = inputs[1]->cast_grad_and_get_pointer<T>(this->ctx_, false);
  }
  if (propagate_down[2]) {
    if (!accum[2])
      inputs[2]->grad()->zero();
    doff = inputs[2]->cast_grad_and_get_pointer<T>(this->ctx_, false);
  }

  if ((inputs.size() == 5 || (inputs.size() == 4 && inputs[3]->ndim() != 1)) &&
      propagate_down[3]) {
    if (!accum[3])
      inputs[3]->grad()->zero();
    dmask = inputs[3]->cast_grad_and_get_pointer<T>(this->ctx_, false);
  }
  if (inputs.size() == 5 && propagate_down[4]) {
    if (!accum[4])
      inputs[4]->grad()->zero();
    db = inputs[4]->cast_grad_and_get_pointer<T>(this->ctx_, false);
    mdb.reset(new ColVectorMap<T>(db, channels_o_));
  } else if ((inputs.size() == 4 && inputs[3]->ndim() == 1) &&
             propagate_down[3]) {
    if (!accum[3])
      inputs[3]->grad()->zero();
    db = inputs[3]->cast_grad_and_get_pointer<T>(this->ctx_, false);
    mdb.reset(new ColVectorMap<T>(db, channels_o_));
  }
  // Sample loop
  for (int n = 0; n < outer_size_; ++n) {
    const T *dy_n = dy + n * inner_size_o_;
    if (propagate_down[1]) {
      // Backprop to weights
      // im2col
      if (inputs.size() == 5 ||
          (inputs.size() == 4 && inputs[3]->ndim() != 1)) {
        modulated_deformable_im2col_cpu<T, true>(
            x + n * inner_size_i_, offset + n * offset_size_i_,
            mask + n * mask_size_i_, channels_i_, spatial_shape_i_, kernel_,
            pad_, stride_, dilation_, deformable_group_, col);
      } else {
        modulated_deformable_im2col_cpu<T, false>(
            x + n * inner_size_i_, offset + n * offset_size_i_, nullptr,
            channels_i_, spatial_shape_i_, kernel_, pad_, stride_, dilation_,
            deformable_group_, col);
      }
      // Weight convolution by matrix multiplication
      for (int g = 0; g < group_; ++g) {
        ConstMatrixMap<T> mdy(dy_n + g * row_y_ * col_y_, row_y_, col_y_);
        ConstMatrixMap<T> mcol(col + g * row_col_ * col_col_, row_col_,
                               col_col_);
        MatrixMap<T> mdw(dw + g * row_w_ * col_w_, row_w_, col_w_);
        mdw += mdy * mcol.transpose();
      }
    }
    if (propagate_down[0]) {
      // Backprop to image
      T *dx_n = dx + n * inner_size_i_;
      for (int g = 0; g < group_; ++g) {
        ConstMatrixMap<T> mdy(dy_n + g * row_y_ * col_y_, row_y_, col_y_);
        ConstMatrixMap<T> mw(w + g * row_w_ * col_w_, row_w_, col_w_);
        MatrixMap<T> mdx(col + g * row_col_ * col_col_, row_col_, col_col_);
        mdx = mw.transpose() * mdy;
      }
      // Backprop to offset and mask
      // offset and mask are calculated at the same time
      T *doff_n = doff + n * offset_size_i_;

      if (inputs.size() == 5 ||
          (inputs.size() == 4 && inputs[3]->ndim() != 1)) {
        T *dmask_n = dmask + n * mask_size_i_;
        modulated_deformable_col2im_coord_cpu<T, true>(
            col, x + n * inner_size_i_, offset + n * offset_size_i_,
            mask + n * mask_size_i_, channels_i_, spatial_shape_i_, kernel_,
            pad_, stride_, dilation_, deformable_group_, doff_n, dmask_n);
        // col2im
        modulated_deformable_col2im_cpu<T, true>(
            col, offset + n * offset_size_i_, mask + n * mask_size_i_,
            channels_i_, spatial_shape_i_, kernel_, pad_, stride_, dilation_,
            deformable_group_, dx_n);
      } else {
        modulated_deformable_col2im_coord_cpu<T, false>(
            col, x + n * inner_size_i_, offset + n * offset_size_i_, nullptr,
            channels_i_, spatial_shape_i_, kernel_, pad_, stride_, dilation_,
            deformable_group_, doff_n, nullptr);
        // col2im
        modulated_deformable_col2im_cpu<T, false>(
            col, offset + n * offset_size_i_, nullptr, channels_i_,
            spatial_shape_i_, kernel_, pad_, stride_, dilation_,
            deformable_group_, dx_n);
      }
    }

    if ((inputs.size() == 5 && propagate_down[4]) ||
        (inputs.size() == 4 && inputs[3]->ndim() == 1 && propagate_down[3])) {
      // Backprop to bias
      ConstMatrixMap<T> mdy(dy_n, channels_o_, col_y_);
      *mdb += mdy.rowwise().sum();
    }
  }
  col_.data()->array()->clear();
}
}
