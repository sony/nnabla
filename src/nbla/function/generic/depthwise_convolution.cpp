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

#include <nbla/function/depthwise_convolution.hpp>
#include <nbla/utils/eigen.hpp>
#include <nbla/utils/fold_from_patches.hpp>
#include <nbla/utils/unfold_to_patches.hpp>

#include <nbla/variable.hpp>

#include <algorithm>
#include <cstring>
#include <numeric>

namespace nbla {

NBLA_REGISTER_FUNCTION_SOURCE(DepthwiseConvolution,
                              int,                 // base_axis
                              const vector<int> &, // padding
                              const vector<int> &, // stride
                              const vector<int> &, // dilation
                              int);                // multiplier

namespace depthwise_convolution {

inline int multiply_dimensions(const vector<int> &v) {
  return std::accumulate(v.begin(), v.end(), 1, std::multiplies<int>());
}

inline long int multiply_dimensions(const Shape_t &s) {
  return std::accumulate(s.begin(), s.end(), 1, std::multiplies<long int>());
}

} // namespace nbla::depthwise_convolution

using namespace depthwise_convolution;

template <typename T>
void DepthwiseConvolution<T>::setup_impl(const Variables &inputs,
                                         const Variables &outputs) {
  Variable *const input = inputs[0];
  Variable *const weights = inputs[1];
  Variable *const bias = (inputs.size() == 3) ? inputs[2] : nullptr;
  Variable *const output = outputs[0];

  Shape_t input_shape = input->shape();
  Shape_t weight_shape = weights->shape();

  NBLA_CHECK(base_axis_ >= 0, error_code::value,
             "base_axis may not be less than zero, got %d", base_axis_);
  auto base_axis = static_cast<Shape_t::size_type>(base_axis_);
  NBLA_CHECK(base_axis < input_shape.size() - 1, error_code::value,
             "base_axis must be less than ndim - 1 of inputs[0]. "
             "base_axis: %d >= ndim of inputs[0] - 1: %d.",
             base_axis_, input_shape.size() - 1);

  auto kernel_dims = input_shape.size() - base_axis - 1;

  NBLA_CHECK(kernel_dims <= 2, error_code::unclassified,
             "Depthwise convolution requires 1D or 2D image shape.");

  NBLA_CHECK(weight_shape.size() == 1 + kernel_dims, error_code::value,
             "Weights must be a %dD tensor to match a %dD kernel.",
             kernel_dims + 1, kernel_dims);

  NBLA_CHECK(padding_.size() == kernel_dims, error_code::value,
             "Pad size mismatch. padding dims: %d != kernel dims: %d.",
             padding_.size(), kernel_dims);

  NBLA_CHECK(stride_.size() == kernel_dims, error_code::value,
             "Stride size mismatch. stride dims: %d != kernel dims: %d.",
             stride_.size(), kernel_dims);

  NBLA_CHECK(dilation_.size() == kernel_dims, error_code::value,
             "Dilation size mismatch. dilation dims: %d != kernel dims: %d.",
             dilation_.size(), kernel_dims);

  sample_channels_ = input_shape[base_axis_];
  outmap_channels_ = sample_channels_ * multiplier_;

  NBLA_CHECK(weight_shape[0] == outmap_channels_, error_code::value,
             "Weight size must match input channels times multiplier. "
             "weight_shape[0] %d != input_shape[%d] * multiplier %d: %d.",
             weight_shape[0], base_axis_, multiplier_, outmap_channels_);

  if (bias) {
    auto bias_shape = bias->shape();

    NBLA_CHECK(bias_shape.size() == 1, error_code::value,
               "Bias(inputs[2]) must be a 1D tensor.");

    NBLA_CHECK(bias_shape[0] == outmap_channels_, error_code::value,
               "Bias(inputs[2]) must match the number of output channels. "
               "bias_shape[0]: %d != input_shape[%d] * multiplier %d: %d.",
               bias_shape[0], base_axis_, multiplier_, outmap_channels_);
  }

  auto copy_dims_to = [](vector<int> &dst, const Shape_t &src, size_t start) {
    dst.resize(start < src.size() ? src.size() - start : 0);
    std::copy(src.begin() + start, src.end(), dst.begin());
  };

  copy_dims_to(kernel_shape_, weight_shape, 1);
  kernel_size_ = multiply_dimensions(kernel_shape_);

  copy_dims_to(sample_shape_, input_shape, base_axis_ + 1);
  sample_size_ = multiply_dimensions(sample_shape_);

  outmap_shape_.clear();
  outmap_shape_.reserve(kernel_shape_.size());
  for (vector<int>::size_type i = 0; i < kernel_shape_.size(); i++) {
    auto kernel = dilation_[i] * (kernel_shape_[i] - 1) + 1; // dilated kernel
    auto padded = sample_shape_[i] + 2 * padding_[i];        // padded sample
    outmap_shape_.push_back(1 + (padded - kernel) / stride_[i]);
    NBLA_CHECK(
        outmap_shape_[i] > 0, error_code::value,
        "Invalid configuration of depthwise convolution at %d-th dimension. "
        "{input:%d, kernel:%d, pad:%d, stride:%d, dilation:%d}.",
        i, sample_shape_[i], kernel_shape_[i], padding_[i], stride_[i],
        dilation_[i]);
  }
  outmap_size_ = multiply_dimensions(outmap_shape_);

  //
  // Construct output shape as input shape dimensions up to base axis, then
  // outmap channels and finally the outmap shape dimensions. Also compute the
  // batch size from input dimensions up to base axis.
  //
  Shape_t output_shape;
  output_shape.reserve(input_shape.size());
  auto output_shape_end = std::back_inserter(output_shape);
  std::copy_n(input_shape.begin(), base_axis_, output_shape_end);
  batch_size_ = multiply_dimensions(output_shape);
  output_shape.push_back(outmap_channels_);
  std::copy(outmap_shape_.begin(), outmap_shape_.end(), output_shape_end);
  output->reshape(output_shape, true); // resize output

  // Resize the buffer used for im2col/col2im.
  col_.reshape(Shape_t{outmap_channels_ * kernel_size_, outmap_size_}, true);
}

template <typename T>
void DepthwiseConvolution<T>::forward_impl(const Variables &inputs,
                                           const Variables &outputs) {
  using namespace ::nbla::eigen;

  Variable *const input = inputs[0];
  Variable *const output = outputs[0];
  Variable *const weights = inputs[1];
  Variable *const bias = (inputs.size() == 3) ? inputs[2] : nullptr;

  auto sample_data = input->get_data_pointer<T>(this->ctx_);
  auto outmap_data = output->cast_data_and_get_pointer<T>(this->ctx_, true);
  auto kernel_data = weights->get_data_pointer<T>(this->ctx_);
  auto bias_data = bias ? bias->get_data_pointer<T>(this->ctx_) : nullptr;
  auto col = col_.cast_data_and_get_pointer<T>(this->ctx_, true);

  for (int samp = 0; samp < batch_size_; samp++) {
    unfold_to_patches<T>(sample_data, col, sample_channels_, sample_shape_,
                         kernel_shape_, padding_, stride_, dilation_);
    {
      auto kernel_data_ptr = kernel_data;
      auto outmap_data_ptr = outmap_data;
      auto col_ptr = col;

      for (int chan = 0; chan < sample_channels_; chan++) {
        ConstMatrixMap<T> mcol(col_ptr, kernel_size_, outmap_size_);
        for (int i = 0; i < multiplier_; i++) {
          ConstRowVectorMap<T> kernel(kernel_data_ptr, kernel_size_);
          RowVectorMap<T> outmap(outmap_data_ptr, outmap_size_);
          outmap = kernel * mcol;
          kernel_data_ptr += kernel_size_;
          outmap_data_ptr += outmap_size_;
        }
        col_ptr += kernel_size_ * outmap_size_;
      }
    }
    if (bias_data) {
      MatrixMap<T> outmap(outmap_data, outmap_channels_, outmap_size_);
      outmap.colwise() += ConstColVectorMap<T>(bias_data, outmap_channels_);
    }
    sample_data += sample_channels_ * sample_size_;
    outmap_data += outmap_channels_ * outmap_size_;
  }
}

template <typename T>
void DepthwiseConvolution<T>::backward_impl(const Variables &inputs,
                                            const Variables &outputs,
                                            const vector<bool> &propagate_down,
                                            const vector<bool> &accum) {
  using namespace ::nbla::eigen;

  if (!(propagate_down[0] || propagate_down[1] ||
        (inputs.size() == 3 && propagate_down[2]))) {
    return;
  }

  Variable *const input = inputs[0];
  Variable *const output = outputs[0];
  Variable *const weights = inputs[1];
  Variable *const bias = (inputs.size() == 3) ? inputs[2] : nullptr;

  const T *outmap_grad = output->get_grad_pointer<T>(this->ctx_);
  const T *sample_data = nullptr;
  const T *weight_data = nullptr;
  T *sample_grad = nullptr;
  T *weight_grad = nullptr;
  T *bias_grad = nullptr;
  T *col = nullptr;

  if (propagate_down[0] || propagate_down[1]) {
    col = col_.cast_data_and_get_pointer<T>(this->ctx_, true);
  }
  if (propagate_down[0]) {
    if (!accum[0])
      input->grad()->zero();
    sample_grad = input->cast_grad_and_get_pointer<T>(this->ctx_, false);
    weight_data = weights->get_data_pointer<T>(this->ctx_);
  }
  if (propagate_down[1]) {
    if (!accum[1])
      weights->grad()->zero();
    weight_grad = weights->cast_grad_and_get_pointer<T>(this->ctx_, false);
    sample_data = input->get_data_pointer<T>(this->ctx_);
  }
  if (bias && propagate_down[2]) {
    if (!accum[2])
      bias->grad()->zero();
    bias_grad = bias->cast_grad_and_get_pointer<T>(this->ctx_, false);
  }

  for (int samp = 0; samp < batch_size_; samp++) {
    if (propagate_down[0]) { // backprop to input gradient
      memset((void *)col, 0, col_.size() * sizeof(T));

      auto weight_data_ptr = weight_data;
      auto outmap_grad_ptr = outmap_grad;
      auto col_ptr = col;

      for (int chan = 0; chan < sample_channels_; chan++) {
        MatrixMap<T> mcol(col_ptr, kernel_size_, outmap_size_);
        for (int i = 0; i < multiplier_; i++) {
          ConstRowVectorMap<T> outmap(outmap_grad_ptr, outmap_size_);
          ConstColVectorMap<T> kernel(weight_data_ptr, kernel_size_);
          mcol += kernel * outmap;
          weight_data_ptr += kernel_size_;
          outmap_grad_ptr += outmap_size_;
        }
        col_ptr += kernel_size_ * outmap_size_;
      }
      fold_from_patches<T>(col, sample_grad, sample_channels_, sample_shape_,
                           kernel_shape_, padding_, stride_, dilation_);
      sample_grad += sample_channels_ * sample_size_;
    }

    if (propagate_down[1]) { // backprop to weight gradient
      unfold_to_patches<T>(sample_data, col, sample_channels_, sample_shape_,
                           kernel_shape_, padding_, stride_, dilation_);

      auto outmap_grad_ptr = outmap_grad;
      auto weight_grad_ptr = weight_grad;
      auto col_ptr = col;

      for (int chan = 0; chan < sample_channels_; chan++) {
        ConstMatrixMap<T> mcol(col_ptr, kernel_size_, outmap_size_);
        for (int i = 0; i < multiplier_; i++) {
          ConstRowVectorMap<T> outmap(outmap_grad_ptr, outmap_size_);
          RowVectorMap<T> kernel(weight_grad_ptr, kernel_size_);
          kernel += outmap * mcol.transpose();
          weight_grad_ptr += kernel_size_;
          outmap_grad_ptr += outmap_size_;
        }
        col_ptr += kernel_size_ * outmap_size_;
      }
      sample_data += sample_channels_ * sample_size_;
    }

    if (bias && propagate_down[2]) { // backprop to bias gradient
      ConstMatrixMap<T> outmap(outmap_grad, outmap_channels_, outmap_size_);
      ColVectorMap<T>(bias_grad, outmap_channels_) += outmap.rowwise().sum();
    }
    outmap_grad += outmap_channels_ * outmap_size_;
  }
}

} // namespace nbla
