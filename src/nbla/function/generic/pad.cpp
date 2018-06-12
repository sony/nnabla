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
#include <nbla/function/pad.hpp>
#include <nbla/variable.hpp>

namespace nbla {

NBLA_REGISTER_FUNCTION_SOURCE(Pad, const vector<int> &, const string &, float);

template <typename T>
int get_shape_from_last(const vector<T> &shape, int index,
                        int default_value = 1) {
  int axis = shape.size() - 1 + index;
  if (axis < 0) {
    return default_value;
  }
  return shape[axis];
}

template <typename T>
void Pad<T>::setup_impl(const Variables &inputs, const Variables &outputs) {
  Shape_t shape_x = inputs[0]->shape();

  NBLA_CHECK(pad_width_.size() % 2 == 0, error_code::value,
             "pad_with should be even number.");
  NBLA_CHECK(pad_width_.size() / 2 <= shape_x.size(), error_code::value,
             "pad_with %d dimensions does not match with input %d dimensions.",
             pad_width_.size() / 2, shape_x.size());
  NBLA_CHECK(mode_.compare("constant") == 0, error_code::value,
             "Only constant padding is supported currently.");
  NBLA_CHECK(pad_width_.size() / 2 <= 3, error_code::value,
             "pad_with of size %d is not supported currently.",
             pad_width_.size());

  Shape_t shape_y = shape_x;

  // Calculate output shape
  int j = pad_width_.size() - 1, pad_pair_size = pad_width_.size() / 2;
  int modifiable_output_size = shape_x.size() - pad_pair_size;
  for (int i = shape_x.size() - 1; i >= modifiable_output_size; i--) {
    if (j >= 0) {
      shape_y[i] = pad_width_[j] + shape_x[i] + pad_width_[j - 1];
    }
    j = j - 2;
  }
  outputs[0]->reshape(shape_y, true);
}

// For ND input, 1D, 2D and 3D constant padding.
template <typename T>
void constant_pad_forward_impl_cpu(int out_size, int in_size, T *y, const T *x,
                                   float value, int p_front, int p_top,
                                   int p_left, int i_channel, int i_height,
                                   int i_width, int o_channel, int o_height,
                                   int o_width) {

  for (int i = 0; i < out_size; i++) {
    int b = i / (o_channel * o_height * o_width);
    int c = (i / (o_height * o_width)) % o_channel;
    int h = (i / o_width) % o_height;
    int w = i % o_width;

    // Calculate input index
    int ib = b;
    int ic = c - p_front;
    int ih = h - p_top;
    int iw = w - p_left;
    int j = ((ib * i_channel + ic) * i_height + ih) * i_width + iw;

    if ((c >= p_front && c < (p_front + i_channel)) &&
        (w >= p_left && w < (p_left + i_width)) &&
        (h >= p_top && h < (p_top + i_height))) {
      NBLA_CHECK(j <= in_size, error_code::value,
                 " Internal error: Input array index out of bound exception.");
      y[i] = x[j];
    } else {
      y[i] = value;
    }
  }
}

template <typename T>
void Pad<T>::forward_impl(const Variables &inputs, const Variables &outputs) {

  // Inputs
  const T *x = inputs[0]->get_data_pointer<T>(this->ctx_);
  T *y = outputs[0]->cast_data_and_get_pointer<T>(this->ctx_, true);

  // If output NDarray is not the size of 4D, then convert to 4D by adding dummy
  // dimensions (i.e. 1) or
  // if dimension is more than 4D then squeeze first dimensions by
  // multiplication except last 3 dimensions.
  int o_channel = get_shape_from_last(outputs[0]->shape(), -2);
  int o_height = get_shape_from_last(outputs[0]->shape(), -1);
  int o_width = get_shape_from_last(outputs[0]->shape(), 0);
  int o_batch = outputs[0]->size() / (o_channel * o_height * o_width);

  int i_channel = get_shape_from_last(inputs[0]->shape(), -2);
  int i_height = get_shape_from_last(inputs[0]->shape(), -1);
  int i_width = get_shape_from_last(inputs[0]->shape(), 0);
  int i_batch = inputs[0]->size() / (i_channel * i_height * i_width);

  NBLA_CHECK(
      i_batch == o_batch, error_code::value,
      " Internal error: Input array and output array batch size not same.");

  // If pad_width_ not of size 3D, then convert to 3D by adding dummy padding
  // (i.e. 0).
  int p_front = get_shape_from_last(pad_width_, -5, 0);
  int p_top = get_shape_from_last(pad_width_, -3, 0);
  int p_left = get_shape_from_last(pad_width_, -1, 0);

  switch (pad_mode_[mode_]) {
  case p_constant:
    constant_pad_forward_impl_cpu<T>(outputs[0]->size(), inputs[0]->size(), y,
                                     x, constant_value_, p_front, p_top, p_left,
                                     i_channel, i_height, i_width, o_channel,
                                     o_height, o_width);
    break;
  case p_replicate: // TODO
  case p_reflect:   // TODO
  default:
    NBLA_CHECK(false, error_code::value,
               " Internal error: pad mode is not supported.");
    break;
  }
}

// For ND input, 1D, 2D and 3D constant padding.
template <typename T, bool accum>
void constant_pad_backward_impl_cpu(int out_size, int in_size, const T *dy,
                                    T *dx, int p_front, int p_top, int p_left,
                                    int i_channel, int i_height, int i_width,
                                    int o_channel, int o_height, int o_width) {

  for (int i = 0; i < out_size; i++) {
    int b = i / (o_channel * o_height * o_width);
    int c = (i / (o_height * o_width)) % o_channel;
    int h = (i / o_width) % o_height;
    int w = i % o_width;

    // Calculate input index
    int ib = b;
    int ic = c - p_front;
    int ih = h - p_top;
    int iw = w - p_left;
    int j = ((ib * i_channel + ic) * i_height + ih) * i_width + iw;

    if ((c >= p_front && c < (p_front + i_channel)) &&
        (w >= p_left && w < (p_left + i_width)) &&
        (h >= p_top && h < (p_top + i_height))) {
      NBLA_CHECK(j <= in_size, error_code::value,
                 " Internal error: Input array index out of bound exception.");
      if (accum) {
        dx[j] += dy[i];
      } else {
        dx[j] = dy[i];
      }
    }
  }
}

template <typename T>
void Pad<T>::backward_impl(const Variables &inputs, const Variables &outputs,
                           const vector<bool> &propagate_down,
                           const vector<bool> &accum) {
  if (!(propagate_down[0])) {
    return;
  }

  // Gradient of outputs
  const T *dy = outputs[0]->get_grad_pointer<T>(this->ctx_);
  T *dx = inputs[0]->cast_grad_and_get_pointer<T>(this->ctx_, !accum[0]);

  // If output NDarray is not the size of 4D, then convert to 4D by adding dummy
  // dimensions (i.e. 1) or
  // if dimension is more than 4D then squeeze first dimensions by
  // multiplication except last 3 dimensions.
  int o_channel = get_shape_from_last(outputs[0]->shape(), -2);
  int o_height = get_shape_from_last(outputs[0]->shape(), -1);
  int o_width = get_shape_from_last(outputs[0]->shape(), 0);
  int o_batch = outputs[0]->size() / (o_channel * o_height * o_width);

  int i_channel = get_shape_from_last(inputs[0]->shape(), -2);
  int i_height = get_shape_from_last(inputs[0]->shape(), -1);
  int i_width = get_shape_from_last(inputs[0]->shape(), 0);
  int i_batch = inputs[0]->size() / (i_channel * i_height * i_width);

  NBLA_CHECK(
      i_batch == o_batch, error_code::value,
      " Internal error: Input array and output array batch size not same.");

  // If pad_width_ not of size 3D, then convert to 3D by adding dummy padding
  // (i.e. 0).
  int p_front = get_shape_from_last(pad_width_, -5, 0);
  int p_top = get_shape_from_last(pad_width_, -3, 0);
  int p_left = get_shape_from_last(pad_width_, -1, 0);

  switch (pad_mode_[mode_]) {
  case p_constant:
    if (accum[0])
      constant_pad_backward_impl_cpu<T, true>(
          outputs[0]->size(), inputs[0]->size(), dy, dx, p_front, p_top, p_left,
          i_channel, i_height, i_width, o_channel, o_height, o_width);
    else
      constant_pad_backward_impl_cpu<T, false>(
          outputs[0]->size(), inputs[0]->size(), dy, dx, p_front, p_top, p_left,
          i_channel, i_height, i_width, o_channel, o_height, o_width);
    break;
  case p_replicate: // TODO
  case p_reflect:   // TODO
  default:
    NBLA_CHECK(false, error_code::value,
               " Internal error: pad mode is not supported.");
    break;
  }
}
}
