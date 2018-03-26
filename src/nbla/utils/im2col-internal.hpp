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

#include <nbla/exception.hpp>

#include <cstring>

namespace nbla {
template <typename T>
void im2col(const T *img, const int c_i, const int *shape, const int *k,
            const int *p, const int *s, const int *d, T *col) {
  // Possible optimization: https://github.com/BVLC/caffe/pull/3536/files
  const int h_o = (shape[0] + 2 * p[0] - (d[0] * (k[0] - 1) + 1)) / s[0] + 1;
  const int w_o = (shape[1] + 2 * p[1] - (d[1] * (k[1] - 1) + 1)) / s[1] + 1;
  const int ckhkw = c_i * k[0] * k[1];
  // In-kernel loop (rows of col matrix)
  for (int cc = 0; cc < ckhkw; ++cc) {
    const int x_k = cc % k[1];
    const int y_k = (cc / k[1]) % k[0];
    const int z_k = cc / k[0] / k[1];
    // Output image loop (columns of col matrix)
    for (int y_o = 0; y_o < h_o; ++y_o) {
      const int col_offset = (cc * h_o + y_o) * w_o;
      const int y_i = y_o * s[0] - p[0] + y_k * d[0]; // y-pos in image
      // A technique for checking border with one comparison.
      if (static_cast<unsigned>(y_i) < static_cast<unsigned>(shape[0])) {
        const int im_offset = (z_k * shape[0] + y_i) * shape[1];
        for (int x_o = 0; x_o < w_o; ++x_o) {
          const int x_i = x_o * s[1] - p[1] + x_k * d[1]; // x-pos in image
          col[col_offset + x_o] =
              (static_cast<unsigned>(x_i) < static_cast<unsigned>(shape[1]))
                  ? img[im_offset + x_i]
                  : (T)0;
        }
      } else {
        std::memset(col + col_offset, 0, sizeof(T) * w_o);
      }
    }
  }
}

template <typename T>
void im2col_nd(const T *img, const int c, const int spatial_dims,
               const int *spatial_shape, const int *kernel, const int *pad,
               const int *stride, const int *dilation, T *col) {
  NBLA_ERROR(error_code::not_implemented, "Im2Col_ND is not implemented.");
}

template <typename T>
void col2im(const T *col, const int c_i, const int *shape, const int *k,
            const int *p, const int *s, const int *d, T *img) {

  const int h_o = (shape[0] + 2 * p[0] - (d[0] * (k[0] - 1) + 1)) / s[0] + 1;
  const int w_o = (shape[1] + 2 * p[1] - (d[1] * (k[1] - 1) + 1)) / s[1] + 1;
  const int ckhkw = c_i * k[0] * k[1];
  // In-kernel loop (rows of col matrix)
  for (int cc = 0; cc < ckhkw; ++cc) {
    const int x_k = cc % k[1];
    const int y_k = (cc / k[1]) % k[0];
    const int z_k = cc / k[0] / k[1];
    // Output image loop (columns of col matrix)
    for (int y_o = 0; y_o < h_o; ++y_o) {
      const int col_offset = (cc * h_o + y_o) * w_o;
      const int y_i = y_o * s[0] - p[0] + y_k * d[0]; // y-pos in image
      // A technique for checking border with one comparison.
      if (static_cast<unsigned>(y_i) < static_cast<unsigned>(shape[0])) {
        const int im_offset = (z_k * shape[0] + y_i) * shape[1];
        for (int x_o = 0; x_o < w_o; ++x_o) {
          const int x_i = x_o * s[1] - p[1] + x_k * d[1]; // x-pos in image
          if (static_cast<unsigned>(x_i) < static_cast<unsigned>(shape[1])) {
            img[im_offset + x_i] += col[col_offset + x_o];
          }
        }
      }
    }
  }
}

template <typename T>
void col2im_nd(const T *col, const int c, const int spatial_dims,
               const int *spatial_shape, const int *kernel, const int *pad,
               const int *stride, const int *dilation, T *img) {
  NBLA_ERROR(error_code::not_implemented, "Col2Im_ND is not implemented.");
}
}
