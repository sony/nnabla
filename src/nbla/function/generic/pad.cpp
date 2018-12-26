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

#include <nbla/common.hpp>
#include <nbla/function/pad.hpp>
#include <nbla/utils/nd_index.hpp>
#include <nbla/variable.hpp>

#include <iostream>

namespace nbla {

NBLA_REGISTER_FUNCTION_SOURCE(Pad, const vector<int> &, const string &, float);

inline void init_index_map(const Shape_t &dst_ndi, Size_t *idx_map,
                           const Shape_t &src_stride, const Shape_t &dst_stride,
                           const Shape_t &dst_shape, const PadList &padding) {
  const auto dst_idx = ndi::nd2flat(dst_ndi, dst_stride);
  Shape_t::value_type src_idx = 0;
  for (int axis = 0; axis < dst_shape.size(); axis++) {
    if ((dst_ndi[axis] < padding[axis].first) ||
        (dst_ndi[axis] >= dst_shape[axis] - padding[axis].second)) {
      return;
    }
    src_idx += (dst_ndi[axis] - padding[axis].first) * src_stride[axis];
  }
  idx_map[dst_idx] = src_idx;
}

namespace pad_constant_impl {

template <typename T>
inline void pad_forward(const Shape_t &dst_ndi, const T *src, T *dst,
                        const Shape_t &src_stride, const Shape_t &dst_stride,
                        const Shape_t &dst_shape, const PadList &padding,
                        const T constant_value) {
  const auto dst_idx = ndi::nd2flat(dst_ndi, dst_stride);
  Shape_t::value_type src_idx = 0;
  for (int axis = 0; axis < dst_shape.size(); axis++) {
    if ((dst_ndi[axis] < padding[axis].first) ||
        (dst_ndi[axis] >= dst_shape[axis] - padding[axis].second)) {
      dst[dst_idx] = constant_value;
      return;
    }
    src_idx += (dst_ndi[axis] - padding[axis].first) * src_stride[axis];
  }
  dst[dst_idx] = src[src_idx];
}

template <typename T, bool ACCUMULATE>
inline void pad_backward(const Shape_t &src_ndi, const T *src, T *dst,
                         const Shape_t &dst_stride, const Shape_t &src_stride,
                         const Shape_t &src_shape, const PadList &padding) {
  const auto src_idx = ndi::nd2flat(src_ndi, src_stride);
  Shape_t::value_type dst_idx = 0;
  for (int axis = 0; axis < src_shape.size(); axis++) {
    if ((src_ndi[axis] < padding[axis].first) ||
        (src_ndi[axis] >= src_shape[axis] - padding[axis].second)) {
      return;
    }
    dst_idx += (src_ndi[axis] - padding[axis].first) * dst_stride[axis];
  }
  dst[dst_idx] = ACCUMULATE ? dst[dst_idx] + src[src_idx] : src[src_idx];
}
} // namespace pad_constant_impl

namespace pad_reflect_impl {

template <typename INT> inline INT reflect_index(INT idx, INT len) {
  // Returns an index that is within the inclusive interval [0, len]
  // and reverses direction at 0 and len. The direction is determined
  // by the last bit of the idx divided by len. Example:
  // len=3 idx=[0, 1, 2, 3, 4, 5, 6, 7] => [0, 1, 2, 3, 2, 1, 0, 1]
  return len > 0 ? std::abs(((idx / len) & 1) * len - (idx % len)) : 0;
}

inline void pad_index_map(const Shape_t &dst_ndi, const Shape_t &dst_stride,
                          const Shape_t &dst_shape, const int axis,
                          const PadList &padding, Size_t *idx_map) {
  const auto dst_idx = ndi::nd2flat(dst_ndi, dst_stride);
  const auto pad_sum = padding.at(axis).first + padding.at(axis).second;
  const auto src_len = dst_shape.at(axis) - pad_sum;

  if (dst_ndi[axis] < padding[axis].first) {
    const auto p = padding[axis].first;
    const auto r = reflect_index(p - dst_ndi[axis], src_len - 1);
    auto src_idx = ndi::nd2flat(dst_ndi, dst_stride, axis);
    src_idx += (p + r) * dst_stride[axis];
    src_idx += ndi::nd2flat(dst_ndi, dst_stride, {axis + 1, dst_shape.size()});
    idx_map[dst_idx] = idx_map[src_idx];
    return;
  }

  if (dst_ndi[axis] >= dst_shape[axis] - padding[axis].second) {
    const auto p = padding[axis].first + src_len;
    const auto r = reflect_index(dst_ndi[axis] - p + 1, src_len - 1);
    auto src_idx = ndi::nd2flat(dst_ndi, dst_stride, axis);
    src_idx += (p - r - 1) * dst_stride[axis];
    src_idx += ndi::nd2flat(dst_ndi, dst_stride, {axis + 1, dst_shape.size()});
    idx_map[dst_idx] = idx_map[src_idx];
    return;
  }
}

} // namespace pad_reflect_impl

template <typename T>
void Pad<T>::setup_impl(const Variables &inputs, const Variables &outputs) {
  Variable &x = *inputs[0];
  Variable &y = *outputs[0];

  if (this->mode_string_ == "constant") {
    this->pad_mode_ = PAD_CONSTANT;
  } else if (this->mode_string_ == "reflect") {
    this->pad_mode_ = PAD_REFLECT;
  } else {
    NBLA_ERROR(error_code::value, "Unsupported pad mode '%s'.",
               mode_string_.c_str());
  }

  NBLA_CHECK(pad_width_.size() % 2 == 0, error_code::value,
             "pad_width must hold an even number of elements.");

  NBLA_CHECK(pad_width_.size() <= 2 * x.shape().size(), error_code::value,
             "pad_width has more values than allowed by input dimensions.");

  auto greater_zero = [](int v) { return v >= 0; };
  NBLA_CHECK(std::all_of(pad_width_.begin(), pad_width_.end(), greater_zero),
             error_code::value, "All pad_width values must be positive.");

  PadList padding(x.ndim());
  { // Copy pairs of pad_width values starting at last dimension.
    // Additional outer dimensions are initialized to zero.
    auto it = padding.rbegin();
    for (int i = this->pad_width_.size() - 2; i >= 0; i -= 2) {
      *it++ = {this->pad_width_.at(i), this->pad_width_.at(i + 1)};
    }
  }

  Shape_t y_shape;
  y_shape.reserve(x.ndim());

  for (int axis = 0; axis < x.ndim(); axis++) {
    auto size = x.shape().at(axis);
    size += padding.at(axis).first;
    size += padding.at(axis).second;
    y_shape.push_back(size);
  }

  y.reshape(y_shape, true);
  this->index_map_.reshape(y_shape, true);

  const auto ndim_pad = this->pad_width_.size() / 2 + 1;
  const auto ndim_out = y_shape.size();
  auto x_stride = x.strides();
  auto y_stride = y.strides();

  if (ndim_out > ndim_pad) {
    padding.erase(padding.begin(), padding.end() - ndim_pad);
    x_stride.erase(x_stride.begin(), x_stride.begin() + ndim_out - ndim_pad);
    y_stride.erase(y_stride.begin(), y_stride.begin() + ndim_out - ndim_pad);
    y_shape.erase(y_shape.begin(), y_shape.begin() + ndim_out - ndim_pad);
    x_stride.front() = ndi::inner_size(x.shape(), ndim_out - ndim_pad + 1);
    y_stride.front() = ndi::inner_size(y.shape(), ndim_out - ndim_pad + 1);
    y_shape.front() = ndi::outer_size(y.shape(), ndim_out - ndim_pad + 1);
  }

  this->padding_ = padding;
  this->x_stride_ = x_stride;
  this->y_stride_ = y_stride;
  this->y_shape_ = y_shape;
}

template <typename T>
void Pad<T>::forward_impl(const Variables &inputs, const Variables &outputs) {
  Variable &x_var = *inputs[0];
  Variable &y_var = *outputs[0];

  const auto &x_stride = this->x_stride_;
  const auto &y_stride = this->y_stride_;
  const auto &y_shape = this->y_shape_;
  const auto &padding = this->padding_;

  auto y_ndi = ndi::flat2nd(Shape_t::value_type(0), y_stride);
  auto x = x_var.get_data_pointer<T>(this->ctx_);
  auto y = y_var.cast_data_and_get_pointer<T>(this->ctx_, true);

  if (this->pad_mode_ == this->PAD_CONSTANT) {
    using namespace pad_constant_impl;
    const auto val = this->constant_value_;
    do {
      pad_forward<T>(y_ndi, x, y, x_stride, y_stride, y_shape, padding, val);
    } while (ndi::increment(y_ndi, y_shape));
  }

  else if (this->pad_mode_ == this->PAD_REFLECT) {
    using namespace pad_reflect_impl;
    Variable &index_map = this->index_map_;
    auto idx = index_map.cast_data_and_get_pointer<Size_t>(this->ctx_, false);
    do {
      init_index_map(y_ndi, idx, x_stride, y_stride, y_shape, padding);
    } while (ndi::increment(y_ndi, y_shape));
    for (int axis = y_ndi.size() - 1; axis >= 0; --axis) {
      do {
        pad_index_map(y_ndi, y_stride, y_shape, axis, padding, idx);
      } while (ndi::increment(y_ndi, y_shape));
    }
    for (Size_t i = 0; i < y_var.size(); i++) {
      y[i] = x[idx[i]];
    }
  }
}

template <typename T>
void Pad<T>::backward_impl(const Variables &inputs, const Variables &outputs,
                           const vector<bool> &propagate_down,
                           const vector<bool> &accum) {
  if (propagate_down[0]) {
    Variable &x_var = *inputs[0];
    Variable &y_var = *outputs[0];

    const auto &x_stride = this->x_stride_;
    const auto &y_stride = this->y_stride_;
    const auto &y_shape = this->y_shape_;
    const auto &padding = this->padding_;

    auto y_ndi = ndi::flat2nd(Shape_t::value_type(0), y_stride);
    auto dy = y_var.get_grad_pointer<T>(this->ctx_);

    if (this->pad_mode_ == this->PAD_CONSTANT) {
      using namespace pad_constant_impl;
      auto dx = x_var.cast_grad_and_get_pointer<T>(this->ctx_, !accum[0]);
      auto backward = accum[0] ? pad_backward<T, true> : pad_backward<T, false>;
      do {
        backward(y_ndi, dy, dx, x_stride, y_stride, y_shape, padding);
      } while (ndi::increment(y_ndi, y_shape));
    }

    else if (this->pad_mode_ == this->PAD_REFLECT) {
      using namespace pad_reflect_impl;
      if (!accum[0]) {
        x_var.grad()->zero();
      }
      Variable &index_map = this->index_map_;
      auto idx = index_map.get_data_pointer<Size_t>(this->ctx_);
      auto dx = x_var.cast_grad_and_get_pointer<T>(this->ctx_, false);
      for (Size_t i = 0; i < y_var.size(); i++) {
        dx[idx[i]] += dy[i];
      }
    }
  }
}

} // namespace nbla
