// Copyright (c) 2019 Sony Corporation. All Rights Reserved.
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

#include <nbla/utils/deformable-im2col-internal.hpp>

#include <nbla/exception.hpp>

#include <nbla/half.hpp>

#include <cstring>

#include <vector>

#include <iostream>
namespace nbla {

template <typename T>

inline T im2col_bilinear_cpu(const T *img, const int data_width,
                             const int height, const int width, T h, T w) {

  int h_low = std::floor(h);
  int w_low = std::floor(w);
  int h_high = h_low + 1;
  int w_high = w_low + 1;

  T lh = h - h_low;
  T lw = w - w_low;
  T hh = 1 - lh, hw = 1 - lw;

  T v1 = 0;
  if (h_low >= 0 && w_low >= 0)
    v1 = img[h_low * data_width + w_low];
  T v2 = 0;
  if (h_low >= 0 && w_high <= width - 1)
    v2 = img[h_low * data_width + w_high];
  T v3 = 0;
  if (h_high <= height - 1 && w_low >= 0)
    v3 = img[h_high * data_width + w_low];
  T v4 = 0;
  if (h_high <= height - 1 && w_high <= width - 1)
    v4 = img[h_high * data_width + w_high];
  T w1 = hh * hw, w2 = hh * lw, w3 = lh * hw, w4 = lh * lw;
  T val = (w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4);
  return val;
}

template <typename T>
inline T get_gradient_weight_cpu(T argmax_h, T argmax_w, const int h,
                                 const int w, const int height,
                                 const int width) {
  if (argmax_h <= -1 || argmax_h >= height || argmax_w <= -1 ||
      argmax_w >= width) {
    // empty
    return 0;
  }

  int argmax_h_low = std::floor(argmax_h);
  int argmax_w_low = std::floor(argmax_w);
  int argmax_h_high = argmax_h_low + 1;
  int argmax_w_high = argmax_w_low + 1;

  T weight = 0;
  if (h == argmax_h_low && w == argmax_w_low)
    weight = (h + 1 - argmax_h) * (w + 1 - argmax_w);
  if (h == argmax_h_low && w == argmax_w_high)
    weight = (h + 1 - argmax_h) * (argmax_w + 1 - w);
  if (h == argmax_h_high && w == argmax_w_low)
    weight = (argmax_h + 1 - h) * (w + 1 - argmax_w);
  if (h == argmax_h_high && w == argmax_w_high)
    weight = (argmax_h + 1 - h) * (argmax_w + 1 - w);
  return weight;
}

template <typename T>
inline T get_coordinate_weight_cpu(T argmax_h, T argmax_w, const int height,
                                   const int width, const T *im_data,
                                   const int data_width, const int bp_dir) {
  if (argmax_h <= -1 || argmax_h >= height || argmax_w <= -1 ||
      argmax_w >= width) {
    // empty
    return 0;
  }

  int argmax_h_low = std::floor(argmax_h);
  int argmax_w_low = std::floor(argmax_w);
  int argmax_h_high = argmax_h_low + 1;
  int argmax_w_high = argmax_w_low + 1;

  T weight = 0;

  if (bp_dir == 0) {
    if (argmax_h_low >= 0 && argmax_w_low >= 0)
      weight += -1 * (argmax_w_low + 1 - argmax_w) *
                im_data[argmax_h_low * data_width + argmax_w_low];
    if (argmax_h_low >= 0 && argmax_w_high <= width - 1)
      weight += -1 * (argmax_w - argmax_w_low) *
                im_data[argmax_h_low * data_width + argmax_w_high];
    if (argmax_h_high <= height - 1 && argmax_w_low >= 0)
      weight += (argmax_w_low + 1 - argmax_w) *
                im_data[argmax_h_high * data_width + argmax_w_low];
    if (argmax_h_high <= height - 1 && argmax_w_high <= width - 1)
      weight += (argmax_w - argmax_w_low) *
                im_data[argmax_h_high * data_width + argmax_w_high];
  } else if (bp_dir == 1) {
    if (argmax_h_low >= 0 && argmax_w_low >= 0)
      weight += -1 * (argmax_h_low + 1 - argmax_h) *
                im_data[argmax_h_low * data_width + argmax_w_low];
    if (argmax_h_low >= 0 && argmax_w_high <= width - 1)
      weight += (argmax_h_low + 1 - argmax_h) *
                im_data[argmax_h_low * data_width + argmax_w_high];
    if (argmax_h_high <= height - 1 && argmax_w_low >= 0)
      weight += -1 * (argmax_h - argmax_h_low) *
                im_data[argmax_h_high * data_width + argmax_w_low];
    if (argmax_h_high <= height - 1 && argmax_w_high <= width - 1)
      weight += (argmax_h - argmax_h_low) *
                im_data[argmax_h_high * data_width + argmax_w_high];
  }

  return weight;
}

template <typename T, bool MODULATED>
void modulated_deformable_im2col_cpu_kernel(
    const int n, const T *data_im, const T *data_offset, const T *data_mask,
    const int height, const int width, const int kernel_h, const int kernel_w,
    const int pad_h, const int pad_w, const int stride_h, const int stride_w,
    const int dilation_h, const int dilation_w,
    const int channel_per_deformable_group, const int height_col,
    const int width_col, const int num_channels, const int deformable_group,
    T *data_col) {
  // n = num_channels*h_o * w_o
  for (int ind = 0; ind < n; ind++) {
    // coordinates at output column
    const int w_col = ind % width_col;
    const int h_col = (ind / width_col) % height_col;
    const int c_im = (ind / width_col) / height_col;
    const int c_col = c_im * kernel_h * kernel_w;
    T *data_col_ptr = data_col;
    data_col_ptr += (c_col * height_col + h_col) * width_col + w_col;

    // compute deformable group index
    const int deformable_group_index = c_im / channel_per_deformable_group;
    // input coordinates with stride
    const int h_in = h_col * stride_h;
    const int w_in = w_col * stride_w;
    // set pointer to start of input image data
    const T *data_im_ptr = data_im;
    // move input image pointer to current coordinate
    data_im_ptr += (c_im * height) * width; // 0
    // set offset pointer to start of input offset data
    const T *data_offset_ptr = data_offset;
    // move input offset pointer current coordinate
    // with channels = 2*deformable_group_index * kernel_h * kernel_w)
    data_offset_ptr +=
        (deformable_group_index * 2 * kernel_h * kernel_w) * height * width;
    // same for mask
    const T *data_mask_ptr = data_mask;
    if (MODULATED) {
      data_mask_ptr +=
          (deformable_group_index * kernel_h * kernel_w) * height * width;
    }
    // unroll matrix to kernel shape
    for (int i = 0; i < kernel_h; ++i) {
      for (int j = 0; j < kernel_w; ++j) {
        // set fetch offset coordinates based on kernel coordinates i and j
        const int data_offset_h_ptr =
            ((2 * (i * kernel_w + j)) * height + h_in) * width + w_in;
        const int data_offset_w_ptr =
            ((2 * (i * kernel_w + j) + 1) * height + h_in) * width + w_in;

        // get offset value from coordinates above
        const T offset_h = data_offset_ptr[data_offset_h_ptr];
        const T offset_w = data_offset_ptr[data_offset_w_ptr];
        T val = static_cast<T>(0);
        // get coordinates in the image adjusted by the offset and padding
        // applied
        const T h_im = h_in + i * dilation_h + offset_h - pad_h;
        const T w_im = w_in + j * dilation_w + offset_w - pad_w;
        // out of boundaries check
        if (h_im > -1 && w_im > -1 && h_im < height && w_im < width) {
          val = im2col_bilinear_cpu(data_im_ptr, width, height, width, h_im,
                                    w_im);
        }

        if (MODULATED) {
          // apply mask
          const int data_mask_hw_ptr =
              ((i * kernel_w + j) * height + h_in) * width + w_in;
          const T mask = data_mask_ptr[data_mask_hw_ptr];
          *data_col_ptr = val * mask;
        } else {
          *data_col_ptr = val;
        }
        // move pointer forward
        data_col_ptr += height_col * width_col;
      }
    }
  }
}
template <typename T, bool MODULATED>
void modulated_deformable_col2im_cpu_kernel(
    const int n, const T *data_col, const T *data_offset, const T *data_mask,
    const int channels, const int height, const int width, const int kernel_h,
    const int kernel_w, const int pad_h, const int pad_w, const int stride_h,
    const int stride_w, const int dilation_h, const int dilation_w,
    const int channel_per_deformable_group, const int deformable_group,
    const int height_col, const int width_col, T *grad_im) {
  // n = channels * kernel_h * kernel_w * heigh_col * width_col
  for (int ind = 0; ind < n; ind++) {
    // coordinates at input column
    int w_out = ind % width_col;
    int h_out = (ind / width_col) % height_col;
    int w_in = w_out * stride_w;
    int h_in = h_out * stride_h;

    // compute the start and end of the output
    // coordinates at output image
    const int i = (ind / width_col / height_col / kernel_w) % kernel_h;
    const int j = (ind / width_col / height_col) % kernel_w;
    const int c = ind / width_col / height_col / kernel_w / kernel_h;

    // compute deformable group index
    const int deformable_group_index = c / channel_per_deformable_group;

    // Set offset pointer to base coordinate based on n
    const T *data_offset_ptr =
        data_offset +
        deformable_group_index * 2 * kernel_h * kernel_w * height * width;
    // Adjust offsets pointer to correct value
    const int data_offset_h_ptr =
        ((2 * (i * kernel_w + j)) * height + h_in) * width + w_in;
    const int data_offset_w_ptr =
        ((2 * (i * kernel_w + j) + 1) * height + h_in) * width + w_in;

    // fetch offset from pointer above
    const T offset_h = data_offset_ptr[data_offset_h_ptr];
    const T offset_w = data_offset_ptr[data_offset_w_ptr];

    const T cur_inv_h_data = h_in + i * dilation_h + offset_h - pad_h;
    const T cur_inv_w_data = w_in + j * dilation_w + offset_w - pad_w;

    // get gradient with mask applied
    T cur_top_grad;
    const int cur_h = (int)cur_inv_h_data;
    const int cur_w = (int)cur_inv_w_data;

    if (MODULATED) {
      // Same as offset
      const T *data_mask_ptr =
          data_mask +
          deformable_group_index * kernel_h * kernel_w * height * width;
      const int data_mask_hw_ptr =
          ((i * kernel_w + j) * height + h_in) * width + w_in;
      const T mask = data_mask_ptr[data_mask_hw_ptr];
      cur_top_grad = data_col[ind] * mask;
    } else {
      cur_top_grad = data_col[ind];
    }
    for (int dy = -2; dy <= 2; dy++) {
      for (int dx = -2; dx <= 2; dx++) {
        if (cur_h + dy >= 0 && cur_h + dy < height && cur_w + dx >= 0 &&
            cur_w + dx < width && std::abs(cur_inv_h_data - (cur_h + dy)) < 1 &&
            std::abs(cur_inv_w_data - (cur_w + dx)) < 1) {
          int cur_bottom_grad_pos =
              (c * height + cur_h + dy) * width + cur_w + dx;
          T weight =
              get_gradient_weight_cpu(cur_inv_h_data, cur_inv_w_data,
                                      cur_h + dy, cur_w + dx, height, width);
          // atomicAdd(grad_im + cur_bottom_grad_pos, weight * cur_top_grad);
          *(grad_im + cur_bottom_grad_pos) += weight * cur_top_grad;
        }
      }
    }
  }
}

template <typename T, bool MODULATED>
void modulated_deformable_col2im_coord_cpu_kernel(
    const int n, const T *data_col, const T *data_im, const T *data_offset,
    const T *data_mask, const int channels, const int height, const int width,
    const int kernel_h, const int kernel_w, const int pad_h, const int pad_w,
    const int stride_h, const int stride_w, const int dilation_h,
    const int dilation_w, const int channel_per_deformable_group,
    const int deformable_group, const int height_col, const int width_col,
    T *grad_offset, T *grad_mask) {
  for (int ind = 0; ind < n; ind++) {
    int w = ind % width_col;
    int h = (ind / width_col) % height_col;
    int c = ind / width_col / height_col;
    // compute the start and end of the output

    const int deformable_group_index = c / (2 * kernel_h * kernel_w);
    const int col_step = kernel_h * kernel_w;
    int cnt = 0;
    const T *data_col_ptr = data_col +
                            deformable_group_index *
                                channel_per_deformable_group * width_col *
                                height_col;
    const T *data_im_ptr = data_im +
                           deformable_group_index *
                               channel_per_deformable_group / kernel_h /
                               kernel_w * height * width;
    const T *data_offset_ptr =
        data_offset +
        deformable_group_index * 2 * kernel_h * kernel_w * height * width;
    T *grad_offset_ptr =
        grad_offset +
        deformable_group_index * 2 * kernel_h * kernel_w * width * height;
    const T *data_mask_ptr = data_mask;
    T *grad_mask_ptr = grad_mask;
    if (MODULATED) {
      data_mask_ptr +=
          deformable_group_index * kernel_h * kernel_w * height * width;
      grad_mask_ptr +=
          deformable_group_index * kernel_h * kernel_w * width * height;
    }
    const int offset_c = c - deformable_group_index * 2 * kernel_h * kernel_w;

    for (int col_c = (offset_c / 2); col_c < channel_per_deformable_group;
         col_c += col_step) {
      const int col_pos = ((col_c * height_col) + h) * width_col + w;
      const int bp_dir = offset_c % 2;

      int j = (col_pos / width_col / height_col) % kernel_w;
      int i = (col_pos / width_col / height_col / kernel_w) % kernel_h;
      int w_out = col_pos % width_col;
      int h_out = (col_pos / width_col) % height_col;
      int w_in = w_out * stride_w;
      int h_in = h_out * stride_h;
      const int data_offset_h_ptr =
          (((2 * (i * kernel_w + j)) * height + h_in) * width + w_in);
      const int data_offset_w_ptr =
          (((2 * (i * kernel_w + j) + 1) * height + h_in) * width + w_in);
      const T offset_h = data_offset_ptr[data_offset_h_ptr];
      const T offset_w = data_offset_ptr[data_offset_w_ptr];
      T inv_h = h_in + i * dilation_h + offset_h - pad_h;
      T inv_w = w_in + j * dilation_w + offset_w - pad_w;
      if (inv_h <= -1 || inv_w <= -1 || inv_h >= height || inv_w >= width) {
        inv_h = inv_w = -2;
      }
      const T weight = get_coordinate_weight_cpu(
          inv_h, inv_w, height, width, data_im_ptr + cnt * height * width,
          width, bp_dir);
      const int grad_offset_idx =
          bp_dir == 0 ? data_offset_h_ptr : data_offset_w_ptr;
      if (MODULATED) {
        const int data_mask_hw_ptr =
            (((i * kernel_w + j) * height + h_in) * width + w_in);
        const T mask = data_mask_ptr[data_mask_hw_ptr];
        if (offset_c % 2 == 0) {
          if (inv_h > -1 && inv_w > -1 && inv_h < height && inv_w < width) {
            grad_mask_ptr[data_mask_hw_ptr] +=
                data_col_ptr[col_pos] *
                im2col_bilinear_cpu(data_im_ptr + cnt * height * width, width,
                                    height, width, inv_h, inv_w);
          }
        }
        grad_offset_ptr[grad_offset_idx] +=
            weight * mask * data_col_ptr[col_pos];
      } else {
        grad_offset_ptr[grad_offset_idx] += weight * data_col_ptr[col_pos];
      }

      cnt += 1;
    }
  }
}

template <typename T, bool MODULATED>
void modulated_deformable_im2col_cpu(const T *im, const T *offset,
                                     const T *mask, const int c_i,
                                     const vector<int> &shape,
                                     const vector<int> &k, const vector<int> &p,
                                     const vector<int> &s, const vector<int> &d,
                                     const int deformable_group, T *col) {
  // num_axes should be smaller than block size
  const int channel_per_deformable_group = c_i / deformable_group;
  const int h_o = (shape[0] + 2 * p[0] - (d[0] * (k[0] - 1) + 1)) / s[0] + 1;
  const int w_o = (shape[1] + 2 * p[1] - (d[1] * (k[1] - 1) + 1)) / s[1] + 1;
  const int num_kernels = c_i * h_o * w_o;

  modulated_deformable_im2col_cpu_kernel<T, MODULATED>(
      num_kernels, im, offset, mask, shape[0], shape[1], k[0], k[1], p[0], p[1],
      s[0], s[1], d[0], d[1], channel_per_deformable_group, h_o, w_o, c_i,
      deformable_group, col);
}

template <typename T, bool MODULATED>
void modulated_deformable_col2im_cpu(const T *col, const T *offset,
                                     const T *mask, const int c_i,
                                     const vector<int> &shape,
                                     const vector<int> &k, const vector<int> &p,
                                     const vector<int> &s, const vector<int> &d,
                                     const int deformable_group, T *grad_im) {

  const int channel_per_deformable_group = c_i / deformable_group;
  const int h_o = (shape[0] + 2 * p[0] - (d[0] * (k[0] - 1) + 1)) / s[0] + 1;
  const int w_o = (shape[1] + 2 * p[1] - (d[1] * (k[1] - 1) + 1)) / s[1] + 1;

  const int num_kernels = c_i * k[0] * k[1] * h_o * w_o;
  modulated_deformable_col2im_cpu_kernel<T, MODULATED>(
      num_kernels, col, offset, mask, c_i, shape[0], shape[1], k[0], k[1], p[0],
      p[1], s[0], s[1], d[0], d[1], channel_per_deformable_group,
      deformable_group, h_o, w_o, grad_im);
}

template <typename T, bool MODULATED>
void modulated_deformable_col2im_coord_cpu(
    const T *col, const T *im, const T *offset, const T *mask, const int c_i,
    const vector<int> &shape, const vector<int> &k, const vector<int> &p,
    const vector<int> &s, const vector<int> &d, const int deformable_group,
    T *grad_offset, T *grad_mask) {
  const int h_o = (shape[0] + 2 * p[0] - (d[0] * (k[0] - 1) + 1)) / s[0] + 1;
  const int w_o = (shape[1] + 2 * p[1] - (d[1] * (k[1] - 1) + 1)) / s[1] + 1;
  const int num_kernels = h_o * w_o * 2 * k[0] * k[1] * deformable_group;
  const int channel_per_deformable_group = c_i * k[0] * k[1] / deformable_group;
  modulated_deformable_col2im_coord_cpu_kernel<T, MODULATED>(
      num_kernels, col, im, offset, mask, c_i, shape[0], shape[1], k[0], k[1],
      p[0], p[1], s[0], s[1], d[0], d[1], channel_per_deformable_group,
      deformable_group, h_o, w_o, grad_offset, grad_mask);
}

#define NBLA_SPEC_MODULATED_IM2COL_CPU(TYPE, MODULATED)                        \
  template void modulated_deformable_im2col_cpu<TYPE, MODULATED>(              \
      const TYPE *im, const TYPE *offset, const TYPE *mask, const int c_i,     \
      const vector<int> &shape, const vector<int> &k, const vector<int> &p,    \
      const vector<int> &s, const vector<int> &d, const int deformable_group,  \
      TYPE *col)

#define NBLA_SPEC_MODULATED_COL2IM_CPU(TYPE, MODULATED)                        \
  template void modulated_deformable_col2im_cpu<TYPE, MODULATED>(              \
      const TYPE *col, const TYPE *offset, const TYPE *mask, const int c_i,    \
      const vector<int> &shape, const vector<int> &k, const vector<int> &p,    \
      const vector<int> &s, const vector<int> &d, const int deformable_group,  \
      TYPE *grad_im)

#define NBLA_SPEC_MODULATED_COL2IM_COORD_CPU(TYPE, MODULATED)                  \
  template void modulated_deformable_col2im_coord_cpu<TYPE, MODULATED>(        \
      const TYPE *col, const TYPE *im, const TYPE *offset, const TYPE *mask,   \
      const int c_i, const vector<int> &shape, const vector<int> &k,           \
      const vector<int> &p, const vector<int> &s, const vector<int> &d,        \
      const int deformable_group, TYPE *grad_offset, TYPE *grad_mask)

NBLA_SPEC_MODULATED_IM2COL_CPU(float, true);
NBLA_SPEC_MODULATED_IM2COL_CPU(Half, true);
NBLA_SPEC_MODULATED_COL2IM_CPU(float, true);
NBLA_SPEC_MODULATED_COL2IM_CPU(Half, true);
NBLA_SPEC_MODULATED_COL2IM_COORD_CPU(float, true);
NBLA_SPEC_MODULATED_COL2IM_COORD_CPU(Half, true);
NBLA_SPEC_MODULATED_IM2COL_CPU(float, false);
NBLA_SPEC_MODULATED_IM2COL_CPU(Half, false);
NBLA_SPEC_MODULATED_COL2IM_CPU(float, false);
NBLA_SPEC_MODULATED_COL2IM_CPU(Half, false);
NBLA_SPEC_MODULATED_COL2IM_COORD_CPU(float, false);
NBLA_SPEC_MODULATED_COL2IM_COORD_CPU(Half, false);
}
