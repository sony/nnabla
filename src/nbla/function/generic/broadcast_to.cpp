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

#include <algorithm>
#include <nbla/array.hpp>
#include <nbla/common.hpp>
#include <nbla/function/broadcast_to.hpp>
#include <nbla/variable.hpp>

namespace nbla {

NBLA_REGISTER_FUNCTION_SOURCE(BroadcastTo, int);

template <typename T>
void BroadcastTo<T>::setup_impl(const Variables &inputs,
                                const Variables &outputs) {
  const Shape_t xs = inputs[0]->shape();
  const Shape_t ys = inputs[1]->shape();
  const Size_t xdim = xs.size();
  const Size_t ydim = ys.size();
  NBLA_CHECK(xdim >= ydim, error_code::value,
             "BroadcastTo expects Y (variable to be broadcasted) to be smaller "
             "than or equal to X (target variable we want to fit to): %d vs %d",
             ydim, xdim);
  if (axis_ < 0) {
    // No axis was specified.
    // Check if y shape can fit x shape from the tail dimension
    const Size_t xofs = xdim - ydim;
    for (Size_t i = ydim - 1; i >= 0; i--) {
      Size_t xds = xs[xofs + i];
      Size_t yds = ys[i];
      NBLA_CHECK(xds == yds, error_code::value,
                 "Dimension %d's size of X and Y do not match: %d vs %d",
                 xofs + i, xds, yds);
    }
  } else {
    NBLA_CHECK(axis_ < xdim, error_code::value, "Specified axis index %d must "
                                                "be within the size of the "
                                                "actual input dimension %d",
               axis_, xdim);
    // Check if y shape can fit x shape from the axis index
    for (Size_t i = 0; i < ydim; i++) {
      Size_t xds = xs[i + axis_];
      Size_t yds = ys[i];
      NBLA_CHECK(xds == yds, error_code::value,
                 "Dimension %d's size of X and Y do not match: %d vs %d",
                 i + axis_, xds, yds);
    }
  }
  // All check passed.
  // Reshape output to fit X.
  outputs[0]->reshape(xs, true);
}

// Copy Y block to Z's tail
template <typename T>
static void copy_block_to_tail(T *z, const T *y, const Shape_t &xs, Size_t xdim,
                               Size_t ydim, Size_t ysize) {
  const Size_t diff = xdim - ydim;
  Size_t loop_num = 1;
  for (Size_t i = 0; i < diff; i++) {
    loop_num *= xs[i];
  }
  for (Size_t i = 0; i < loop_num; i++) {
    std::copy(y, y + ysize, z + i * ysize);
  }
}

// Copy each Y value to each block
template <typename T>
static void copy_value_to_block(T *z, const T *y, Size_t y_num,
                                Size_t block_size) {
  for (Size_t i = 0; i < y_num; i++) {
    T val = y[i];
    T *zblock = &z[i * block_size];
    std::fill(zblock, zblock + block_size, val);
  }
}

template <typename T>
static void copy_value_vertically_to_block(T *z, const T *y, Size_t block_num,
                                           Size_t y_num, Size_t block_width,
                                           Size_t block_size) {
  for (Size_t b = 0; b < block_num; b++) {
    T *zblock = &z[b * block_size];
    for (Size_t v = 0; v < y_num; v++) {
      T val = y[v];
      T *zrow = &zblock[v * block_width];
      std::fill(zrow, zrow + block_width, val);
    }
  }
}

template <typename T>
static void copy_buf_vertically_to_block(T *z, const T *y, Size_t block_num,
                                         Size_t y_buf_width,
                                         Size_t z_block_height,
                                         Size_t z_block_width) {
  const Size_t z_block_size = z_block_height * z_block_width;
  for (Size_t b = 0; b < block_num; b++) {
    const T *yrow = &y[b * y_buf_width];
    T *zblock = &z[b * z_block_size];
    for (Size_t v = 0; v < z_block_height; v++) {
      T val = yrow[v];
      T *zrow = &zblock[v * z_block_width];
      std::fill(zrow, zrow + z_block_width, val);
    }
  }
}

template <typename T>
void BroadcastTo<T>::forward_impl(const Variables &inputs,
                                  const Variables &outputs) {
  const T *y = inputs[1]->get_data_pointer<T>(this->ctx_);
  T *z = outputs[0]->cast_data_and_get_pointer<T>(this->ctx_);
  const Shape_t xs = inputs[0]->shape();
  const Shape_t ys = inputs[1]->shape();
  const Size_t ysize = inputs[1]->size();
  const Size_t xdim = xs.size();
  const Size_t ydim = ys.size();
  if (xdim == ydim) {
    // X and Y have exactly the same shape
    // Copy Y to Z
    std::copy(y, y + ysize, z);
    return;
  }
  if (axis_ < 0) {
    // copy the whole Y block to Z per stride
    copy_block_to_tail(z, y, xs, xdim, ydim, ysize);
    return;
  }
  // copy Y depending on the axis position
  NBLA_CHECK(xdim >= 2, error_code::value,
             "X's dimension size should be greater than 1");
  switch (xdim) {
  case 2:
    // Y dimension size is always 1
    switch (axis_) {
    case 0:
      // X: (2,3) Y: (2) axis=0
      copy_value_to_block(z, y, xs[0], xs[1]);
      break;
    case 1:
      // X: (2,3) Y: (3) axis=1
      copy_block_to_tail(z, y, xs, xdim, ydim, ysize);
      break;
    default:
      NBLA_ERROR(error_code::value, "Unexpected axis value");
    }
    break;
  case 3:
    // Y dimension size maybe 1 or 2
    switch (ydim) {
    case 1:
      switch (axis_) {
      case 0:
        // X: (2,3,4) Y: (2) axis=0
        copy_value_to_block(z, y, xs[0], xs[1] * xs[2]);
        break;
      case 1:
        // X: (2,3,4) Y: (3) axis=1
        copy_value_vertically_to_block(z, y, xs[0], xs[1], xs[2],
                                       xs[1] * xs[2]);
        break;
      case 2:
        // X: (2,3,4) Y: (4) axis=2
        copy_block_to_tail(z, y, xs, xdim, ydim, ysize);
        break;
      default:
        NBLA_ERROR(error_code::value, "Unexpected axis value");
      }
      break;
    case 2:
      switch (axis_) {
      case 0:
        // X: (2,3,4) Y: (2,3) axis=0
        copy_buf_vertically_to_block(z, y, xs[0], ys[1], xs[1], xs[2]);
        break;
      case 1:
        // X: (2,3,4) Y: (3,4) axis=1
        copy_block_to_tail(z, y, xs, xdim, ydim, ysize);
        break;
      default:
        NBLA_ERROR(error_code::value, "Unexpected axis value");
      }
      break;
    default:
      NBLA_ERROR(error_code::value, "Unexpected Y dimension size");
    }
    break;
  case 4:
    // Y dimension size maybe 1, 2, or 3
    switch (ydim) {
    case 1:
      switch (axis_) {
      case 0:
        // X: (2,3,4,5) Y: (2) axis=0
        copy_value_to_block(z, y, xs[0], xs[1] * xs[2] * xs[3]);
        break;
      case 1: {
        // X: (2,3,4,5) Y: (3) axis=1
        const Size_t block_size = xs[2] * xs[3];
        const Size_t z_slice_size = xs[1] * block_size;
        for (Size_t i = 0; i < xs[0]; i++) {
          T *z_slice = &z[i * z_slice_size];
          copy_value_to_block(z_slice, y, xs[1], block_size);
        }
      } break;
      case 2: {
        // X: (2,3,4,5) Y: (4) axis=2
        const Size_t block_size = xs[2] * xs[3];
        const Size_t slice_size = xs[1] * block_size;
        for (Size_t i = 0; i < xs[0]; i++) {
          copy_value_vertically_to_block(&z[i * slice_size], y, xs[1], xs[2],
                                         xs[3], block_size);
        }
      } break;
      case 3:
        // X: (2,3,4,5) Y: (5) axis=3
        copy_block_to_tail(z, y, xs, xdim, ydim, ysize);
        break;
      default:
        NBLA_ERROR(error_code::value, "Unexpected axis value");
      }
      break;
    case 2:
      switch (axis_) {
      case 0:
        // X: (2,3,4,5) Y: (2,3) axis=0
        copy_buf_vertically_to_block(z, y, xs[0], ys[1], xs[1], xs[2] * xs[3]);
        break;
      case 1: {
        // X: (2,3,4,5) Y: (3,4) axis=1
        const Size_t block_size = xs[2] * xs[3];
        const Size_t slice_size = xs[1] * block_size;
        for (Size_t i = 0; i < xs[0]; i++) {
          copy_buf_vertically_to_block(&z[i * slice_size], y, xs[1], ys[1],
                                       xs[2], xs[3]);
        }
      } break;
      case 2:
        // X: (2,3,4,5) Y: (4,5) axis=2
        copy_block_to_tail(z, y, xs, xdim, ydim, ysize);
        break;
      default:
        NBLA_ERROR(error_code::value, "Unexpected axis value");
      }
      break;
    case 3:
      switch (axis_) {
      case 0: {
        // X: (2,3,4,5) Y: (2,3,4) axis=0
        Size_t y_buf_width = ys[2];
        Size_t y_slice_size = ys[1] * y_buf_width;
        Size_t z_row_width = xs[3];
        Size_t z_slice_size = xs[2] * xs[3];
        Size_t z_chan_size = xs[1] * z_slice_size;
        for (Size_t n = 0; n < xs[0]; n++) {
          const T *ychan = &y[n * y_slice_size];
          T *zslice = &z[n * z_chan_size];
          for (Size_t c = 0; c < xs[1]; c++) {
            const T *yrow = &ychan[c * y_buf_width];
            T *zblock = &zslice[c * z_slice_size];
            for (Size_t h = 0; h < xs[2]; h++) {
              T val = yrow[h];
              T *zrow = &zblock[h * z_row_width];
              std::fill(zrow, zrow + z_row_width, val);
            }
          }
        }
      } break;
      case 1:
        // X: (2,3,4,5) Y: (3,4,5) axis=1
        copy_block_to_tail(z, y, xs, xdim, ydim, ysize);
        break;
      default:
        NBLA_ERROR(error_code::value, "Unexpected axis value");
      }
      break;
    default:
      NBLA_ERROR(error_code::value, "Unexpected Y dimension size");
    }
    break;
  default:
    NBLA_ERROR(error_code::value, "Unexpected X dimension size");
  }
}

template <typename T>
void BroadcastTo<T>::backward_impl(const Variables &inputs,
                                   const Variables &outputs,
                                   const vector<bool> &propagate_down,
                                   const vector<bool> &accum) {
  NBLA_ERROR(error_code::not_implemented,
             "BroadcastTo backward function is not implemented");
}
}
