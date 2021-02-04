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
#include <nbla/function/interpolate.hpp>
#include <nbla/variable.hpp>

namespace nbla {

NBLA_REGISTER_FUNCTION_SOURCE(Interpolate, const vector<int> &, const string &,
                              bool, bool, bool, bool);

inline float compute_scale(int isize, int osize, bool align_corners) {
  return (osize <= 1) ? 0.0f : (align_corners ? float(isize - 1) / (osize - 1)
                                              : float(isize) / osize);
}

inline float compute_scale_for_nn(int isize, int osize, bool align_corners,
                                  bool half_pixel_for_nn) {
  return half_pixel_for_nn ? isize / static_cast<float>(osize)
                           : compute_scale(isize, osize, align_corners);
}

inline float get_src_index(float scale, int dst_index, bool half_pixel) {
  return half_pixel ? std::max(0.0f, scale * (float(dst_index) + 0.5f) - 0.5f)
                    : scale * dst_index;
}

inline float get_src_index_for_nn(float scale, int dst_index, bool half_pixel,
                                  bool half_pixel_for_nn) {
  return half_pixel_for_nn ? scale * (dst_index + 0.5f)
                           : get_src_index(scale, dst_index, half_pixel);
}

template <typename T>
void linear_interpolate_1d(const T *src, T *dst, const int iw, const int ow,
                           const float sx, const bool half_pixel) {
  for (int ox = 0; ox < ow; ox++) {
    const auto fx = get_src_index(sx, ox, half_pixel);
    const auto x1 = static_cast<int>(fx);
    const auto x2 = std::min(x1 + 1, iw - 1);
    const auto lx1 = static_cast<T>(fx - x1);
    const auto lx0 = static_cast<T>(1) - lx1;

    const T val0 = lx0 * src[x1];
    const T val1 = lx1 * src[x2];
    dst[ox] = val0 + val1;
  }
}

template <typename T>
void linear_interpolate_2d(const T *src, T *dst, const int iw, const int ih,
                           const int ow, const int oh, const float sx,
                           const float sy, const bool half_pixel) {
  for (int oy = 0; oy < oh; oy++) {
    const auto fy = get_src_index(sy, oy, half_pixel);
    const auto y1 = static_cast<int>(fy);
    const auto y2 = std::min(y1 + 1, ih - 1);
    const auto ly1 = static_cast<T>(fy - y1);
    const auto ly0 = static_cast<T>(1) - ly1;

    for (int ox = 0; ox < ow; ox++) {
      const auto fx = get_src_index(sx, ox, half_pixel);
      const auto x1 = static_cast<int>(fx);
      const auto x2 = std::min(x1 + 1, iw - 1);
      const auto lx1 = static_cast<T>(fx - x1);
      const auto lx0 = static_cast<T>(1) - lx1;

#define _I(y, x) ((y)*iw + (x))
      const T val0 = lx0 * src[_I(y1, x1)];
      const T val1 = lx1 * src[_I(y1, x2)];
      const T val2 = lx0 * src[_I(y2, x1)];
      const T val3 = lx1 * src[_I(y2, x2)];
#undef _I
      dst[oy * ow + ox] = ly0 * (val0 + val1) + ly1 * (val2 + val3);
    }
  }
}

template <typename T>
void linear_interpolate_3d(const T *src, T *dst, const int iw, const int ih,
                           const int id, const int ow, const int oh,
                           const int od, const float sx, const float sy,
                           const float sz, const bool half_pixel) {
  for (int oz = 0; oz < od; oz++) {
    const auto fz = get_src_index(sz, oz, half_pixel);
    const auto z1 = static_cast<int>(fz);
    const auto z2 = std::min(z1 + 1, id - 1);
    const auto lz1 = static_cast<T>(fz - z1);
    const auto lz0 = static_cast<T>(1) - lz1;

    for (int oy = 0; oy < oh; oy++) {
      const auto fy = get_src_index(sy, oy, half_pixel);
      const auto y1 = static_cast<int>(fy);
      const auto y2 = std::min(y1 + 1, ih - 1);
      const auto ly1 = static_cast<T>(fy - y1);
      const auto ly0 = static_cast<T>(1) - ly1;

      for (int ox = 0; ox < ow; ox++) {
        const auto fx = get_src_index(sx, ox, half_pixel);
        const auto x1 = static_cast<int>(fx);
        const auto x2 = std::min(x1 + 1, iw - 1);
        const auto lx1 = static_cast<T>(fx - x1);
        const auto lx0 = static_cast<T>(1) - lx1;

#define _I(z, y, x) ((z)*ih * iw + (y)*iw + (x))
        const T val0 = lx0 * src[_I(z1, y1, x1)];
        const T val1 = lx1 * src[_I(z1, y1, x2)];
        const T val2 = lx0 * src[_I(z1, y2, x1)];
        const T val3 = lx1 * src[_I(z1, y2, x2)];
        const T val4 = lx0 * src[_I(z2, y1, x1)];
        const T val5 = lx1 * src[_I(z2, y1, x2)];
        const T val6 = lx0 * src[_I(z2, y2, x1)];
        const T val7 = lx1 * src[_I(z2, y2, x2)];
        const T val8 = ly0 * (val0 + val1) + ly1 * (val2 + val3);
        const T val9 = ly0 * (val4 + val5) + ly1 * (val6 + val7);
#undef _I
        dst[oz * oh * ow + oy * ow + ox] = lz0 * val8 + lz1 * val9;
      }
    }
  }
}

template <typename T>
void linear_interpolate_1d_backward(T *dst, const T *src, const int iw,
                                    const int ow, const float sx,
                                    const bool half_pixel) {
  for (int ox = 0; ox < ow; ox++) {
    const auto fx = get_src_index(sx, ox, half_pixel);
    const auto x1 = static_cast<int>(fx);
    const auto x2 = (x1 < iw - 1) ? (x1 + 1) : x1;
    const auto lx1 = static_cast<T>(fx - x1);
    const auto lx0 = static_cast<T>(1) - lx1;
    const T g = src[ox];
    dst[x1] += lx0 * g;
    dst[x2] += lx1 * g;
  }
}

template <typename T>
void linear_interpolate_2d_backward(T *dst, const T *src, const int iw,
                                    const int ih, const int ow, const int oh,
                                    const float sx, const float sy,
                                    const bool half_pixel) {
  for (int oy = 0; oy < oh; oy++) {
    const auto fy = get_src_index(sy, oy, half_pixel);
    const auto y1 = static_cast<int>(fy);
    const auto y2 = (y1 < ih - 1) ? (y1 + 1) : y1;
    const auto ly1 = static_cast<T>(fy - y1);
    const auto ly0 = static_cast<T>(1) - ly1;

    for (int ox = 0; ox < ow; ox++) {
      const auto fx = get_src_index(sx, ox, half_pixel);
      const auto x1 = static_cast<int>(fx);
      const auto x2 = (x1 < iw - 1) ? (x1 + 1) : x1;
      const auto lx1 = static_cast<T>(fx - x1);
      const auto lx0 = static_cast<T>(1) - lx1;
      const T g = src[oy * ow + ox];
#define _I(y, x) ((y)*iw + (x))
      dst[_I(y1, x1)] += ly0 * lx0 * g;
      dst[_I(y1, x2)] += ly0 * lx1 * g;
      dst[_I(y2, x1)] += ly1 * lx0 * g;
      dst[_I(y2, x2)] += ly1 * lx1 * g;
#undef _I
    }
  }
}

template <typename T>
void linear_interpolate_3d_backward(T *dst, const T *src, const int iw,
                                    const int ih, const int id, const int ow,
                                    const int oh, const int od, const float sx,
                                    const float sy, const float sz,
                                    const bool half_pixel) {
  for (int oz = 0; oz < od; oz++) {
    const auto fz = get_src_index(sz, oz, half_pixel);
    const auto z1 = static_cast<int>(fz);
    const auto z2 = (z1 < id - 1) ? (z1 + 1) : z1;
    const auto lz1 = static_cast<T>(fz - z1);
    const auto lz0 = static_cast<T>(1) - lz1;

    for (int oy = 0; oy < oh; oy++) {
      const auto fy = get_src_index(sy, oy, half_pixel);
      const auto y1 = static_cast<int>(fy);
      const auto y2 = (y1 < ih - 1) ? (y1 + 1) : y1;
      const auto ly1 = static_cast<T>(fy - y1);
      const auto ly0 = static_cast<T>(1) - ly1;

      for (int ox = 0; ox < ow; ox++) {
        const auto fx = get_src_index(sx, ox, half_pixel);
        const auto x1 = static_cast<int>(fx);
        const auto x2 = (x1 < iw - 1) ? (x1 + 1) : x1;
        const auto lx1 = static_cast<T>(fx - x1);
        const auto lx0 = static_cast<T>(1) - lx1;

        const T g = src[oz * oh * ow + oy * ow + ox];
#define _I(z, y, x) ((z)*ih * iw + (y)*iw + (x))
        dst[_I(z1, y1, x1)] += lz0 * ly0 * lx0 * g;
        dst[_I(z1, y1, x2)] += lz0 * ly0 * lx1 * g;
        dst[_I(z1, y2, x1)] += lz0 * ly1 * lx0 * g;
        dst[_I(z1, y2, x2)] += lz0 * ly1 * lx1 * g;
        dst[_I(z2, y1, x1)] += lz1 * ly0 * lx0 * g;
        dst[_I(z2, y1, x2)] += lz1 * ly0 * lx1 * g;
        dst[_I(z2, y2, x1)] += lz1 * ly1 * lx0 * g;
        dst[_I(z2, y2, x2)] += lz1 * ly1 * lx1 * g;
#undef _I
      }
    }
  }
}

template <typename T>
void nearest_interpolate_1d(const T *src, T *dst, const int iw, const int ow,
                            const float sx, const bool half_pixel,
                            const bool half_pixel_for_nn) {
  for (int ox = 0; ox < ow; ox++) {
    const auto fx = get_src_index_for_nn(sx, ox, half_pixel, half_pixel_for_nn);
    const auto ix = std::min(static_cast<int>(fx), iw - 1);
    dst[ox] = src[ix];
  }
}

template <typename T>
void nearest_interpolate_2d(const T *src, T *dst, const int iw, const int ih,
                            const int ow, const int oh, const float sx,
                            const float sy, const bool half_pixel,
                            const bool half_pixel_for_nn) {
  for (int oy = 0; oy < oh; oy++) {
    const auto fy = get_src_index_for_nn(sy, oy, half_pixel, half_pixel_for_nn);
    const auto iy = std::min(static_cast<int>(fy), ih - 1);
    for (int ox = 0; ox < ow; ox++) {
      const auto fx =
          get_src_index_for_nn(sx, ox, half_pixel, half_pixel_for_nn);
      const auto ix = std::min(static_cast<int>(fx), iw - 1);
      dst[oy * ow + ox] = src[iy * iw + ix];
    }
  }
}

template <typename T>
void nearest_interpolate_3d(const T *src, T *dst, const int iw, const int ih,
                            const int id, const int ow, const int oh,
                            const int od, const float sx, const float sy,
                            const float sz, const bool half_pixel,
                            const bool half_pixel_for_nn) {
  for (int oz = 0; oz < od; oz++) {
    const auto fz = get_src_index_for_nn(sz, oz, half_pixel, half_pixel_for_nn);
    const auto iz = std::min(static_cast<int>(fz), id - 1);
    for (int oy = 0; oy < oh; oy++) {
      const auto fy =
          get_src_index_for_nn(sy, oy, half_pixel, half_pixel_for_nn);
      const auto iy = std::min(static_cast<int>(fy), ih - 1);
      for (int ox = 0; ox < ow; ox++) {
        const auto fx =
            get_src_index_for_nn(sx, ox, half_pixel, half_pixel_for_nn);
        const auto ix = std::min(static_cast<int>(fx), iw - 1);
        dst[oz * oh * ow + oy * ow + ox] = src[iz * ih * iw + iy * iw + ix];
      }
    }
  }
}

template <typename T>
void nearest_interpolate_1d_backward(T *dst, const T *src, const int iw,
                                     const int ow, const float sx,
                                     const bool half_pixel,
                                     const bool half_pixel_for_nn) {
  for (int ox = 0; ox < ow; ox++) {
    const auto fx = get_src_index_for_nn(sx, ox, half_pixel, half_pixel_for_nn);
    const auto ix = std::min(static_cast<int>(fx), iw - 1);
    dst[ix] += src[ox];
  }
}

template <typename T>
void nearest_interpolate_2d_backward(T *dst, const T *src, const int iw,
                                     const int ih, const int ow, const int oh,
                                     const float sx, const float sy,
                                     const bool half_pixel,
                                     const bool half_pixel_for_nn) {
  for (int oy = 0; oy < oh; oy++) {
    const auto fy = get_src_index_for_nn(sy, oy, half_pixel, half_pixel_for_nn);
    const auto iy = std::min(static_cast<int>(fy), ih - 1);
    for (int ox = 0; ox < ow; ox++) {
      const auto fx =
          get_src_index_for_nn(sx, ox, half_pixel, half_pixel_for_nn);
      const auto ix = std::min(static_cast<int>(fx), iw - 1);
      dst[iy * iw + ix] += src[oy * ow + ox];
    }
  }
}

template <typename T>
void nearest_interpolate_3d_backward(T *dst, const T *src, const int iw,
                                     const int ih, const int id, const int ow,
                                     const int oh, const int od, const float sx,
                                     const float sy, const float sz,
                                     const bool half_pixel,
                                     const bool half_pixel_for_nn) {
  for (int oz = 0; oz < od; oz++) {
    const auto fz = get_src_index_for_nn(sz, oz, half_pixel, half_pixel_for_nn);
    const auto iz = std::min(static_cast<int>(fz), id - 1);
    for (int oy = 0; oy < oh; oy++) {
      const auto fy =
          get_src_index_for_nn(sy, oy, half_pixel, half_pixel_for_nn);
      const auto iy = std::min(static_cast<int>(fy), ih - 1);
      for (int ox = 0; ox < ow; ox++) {
        const auto fx =
            get_src_index_for_nn(sx, ox, half_pixel, half_pixel_for_nn);
        const auto ix = std::min(static_cast<int>(fx), iw - 1);
        dst[iz * ih * iw + iy * iw + ix] += src[oz * oh * ow + oy * ow + ox];
      }
    }
  }
}

template <typename T>
void Interpolate<T>::setup_impl(const Variables &inputs,
                                const Variables &outputs) {
  NBLA_CHECK((output_size_.size() >= 1) && (output_size_.size() <= 3),
             error_code::not_implemented,
             "Only 1-D, 2-D and 3-D interpolation are implemented.");
  NBLA_CHECK((mode_ == "linear") || (mode_ == "nearest"),
             error_code::not_implemented,
             "Only 'linear' and 'nearest' interpolation are implemented.");
  NBLA_CHECK(
      (align_corners_ == false) || (half_pixel_ == false), error_code::value,
      "(align_corners == true) and (half_pixel == true) is not supported.");

  Shape_t out_shape(inputs[0]->shape());
  auto offset = channel_last_ ? out_shape.size() - output_size_.size() - 1
                              : out_shape.size() - output_size_.size();
  for (Shape_t::size_type d = 0; d < output_size_.size(); d++) {
    out_shape[d + offset] = output_size_[d];
  }
  outputs[0]->reshape(out_shape, true);
}

template <typename T>
void Interpolate<T>::forward_impl(const Variables &inputs,
                                  const Variables &outputs) {
  NBLA_CHECK(!channel_last_, error_code::not_implemented,
             "Interpolation with channel_last is not supported in CPU.");

  auto src = inputs[0]->get_data_pointer<T>(this->ctx_);
  auto dst = outputs[0]->cast_data_and_get_pointer<T>(this->ctx_, true);

  const int ndim = inputs[0]->ndim();

  if (output_size_.size() == 1) {
    const int iw = inputs[0]->shape()[ndim - 1];
    const int ow = outputs[0]->shape()[ndim - 1];
    const int outer_dim = inputs[0]->size() / iw;
    const int src_inner_size = iw;
    const int dst_inner_size = ow;
    if (mode_ == "linear") {
      const float sx = compute_scale(iw, ow, align_corners_);
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
      for (int n = 0; n < outer_dim; n++) {
        auto src_ptr = src + n * src_inner_size;
        auto dst_ptr = dst + n * dst_inner_size;
        linear_interpolate_1d(src_ptr, dst_ptr, iw, ow, sx, half_pixel_);
      }
    } else if (mode_ == "nearest") {
      const float sx =
          compute_scale_for_nn(iw, ow, align_corners_, half_pixel_for_nn_);
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
      for (int n = 0; n < outer_dim; n++) {
        auto src_ptr = src + n * src_inner_size;
        auto dst_ptr = dst + n * dst_inner_size;
        nearest_interpolate_1d(src_ptr, dst_ptr, iw, ow, sx, half_pixel_,
                               half_pixel_for_nn_);
      }
    }
  }

  else if (output_size_.size() == 2) {
    const int iw = inputs[0]->shape()[ndim - 1];
    const int ih = inputs[0]->shape()[ndim - 2];
    const int ow = outputs[0]->shape()[ndim - 1];
    const int oh = outputs[0]->shape()[ndim - 2];
    const int outer_dim = inputs[0]->size() / (iw * ih);
    const int src_inner_size = iw * ih;
    const int dst_inner_size = ow * oh;
    if (mode_ == "linear") {
      const float sx = compute_scale(iw, ow, align_corners_);
      const float sy = compute_scale(ih, oh, align_corners_);
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
      for (int n = 0; n < outer_dim; n++) {
        auto src_ptr = src + n * src_inner_size;
        auto dst_ptr = dst + n * dst_inner_size;
        linear_interpolate_2d(src_ptr, dst_ptr, iw, ih, ow, oh, sx, sy,
                              half_pixel_);
      }
    } else if (mode_ == "nearest") {
      const float sx =
          compute_scale_for_nn(iw, ow, align_corners_, half_pixel_for_nn_);
      const float sy =
          compute_scale_for_nn(ih, oh, align_corners_, half_pixel_for_nn_);
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
      for (int n = 0; n < outer_dim; n++) {
        auto src_ptr = src + n * src_inner_size;
        auto dst_ptr = dst + n * dst_inner_size;
        nearest_interpolate_2d(src_ptr, dst_ptr, iw, ih, ow, oh, sx, sy,
                               half_pixel_, half_pixel_for_nn_);
      }
    }
  }

  else if (output_size_.size() == 3) {
    const int iw = inputs[0]->shape()[ndim - 1];
    const int ih = inputs[0]->shape()[ndim - 2];
    const int id = inputs[0]->shape()[ndim - 3];
    const int ow = outputs[0]->shape()[ndim - 1];
    const int oh = outputs[0]->shape()[ndim - 2];
    const int od = outputs[0]->shape()[ndim - 3];
    const int outer_dim = inputs[0]->size() / (id * iw * ih);
    const int src_inner_size = iw * ih * id;
    const int dst_inner_size = ow * oh * od;
    if (mode_ == "linear") {
      const float sx = compute_scale(iw, ow, align_corners_);
      const float sy = compute_scale(ih, oh, align_corners_);
      const float sz = compute_scale(id, od, align_corners_);
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
      for (int n = 0; n < outer_dim; n++) {
        auto src_ptr = src + n * src_inner_size;
        auto dst_ptr = dst + n * dst_inner_size;
        linear_interpolate_3d(src_ptr, dst_ptr, iw, ih, id, ow, oh, od, sx, sy,
                              sz, half_pixel_);
      }
    } else if (mode_ == "nearest") {
      const float sx =
          compute_scale_for_nn(iw, ow, align_corners_, half_pixel_for_nn_);
      const float sy =
          compute_scale_for_nn(ih, oh, align_corners_, half_pixel_for_nn_);
      const float sz =
          compute_scale_for_nn(id, od, align_corners_, half_pixel_for_nn_);
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
      for (int n = 0; n < outer_dim; n++) {
        auto src_ptr = src + n * src_inner_size;
        auto dst_ptr = dst + n * dst_inner_size;
        nearest_interpolate_3d(src_ptr, dst_ptr, iw, ih, id, ow, oh, od, sx, sy,
                               sz, half_pixel_, half_pixel_for_nn_);
      }
    }
  }
}

template <typename T>
void Interpolate<T>::backward_impl(const Variables &inputs,
                                   const Variables &outputs,
                                   const vector<bool> &propagate_down,
                                   const vector<bool> &accum) {
  NBLA_CHECK(!channel_last_, error_code::not_implemented,
             "Interpolation with channel_last is not supported in CPU.");

  if (!(propagate_down[0])) {
    return;
  }

  auto g_y = outputs[0]->get_grad_pointer<T>(this->ctx_);
  auto g_x = inputs[0]->cast_grad_and_get_pointer<T>(this->ctx_, false);

  const int ndim = inputs[0]->ndim();

  if (output_size_.size() == 1) {
    const int iw = inputs[0]->shape()[ndim - 1];
    const int ow = outputs[0]->shape()[ndim - 1];
    const int g_x_inner_size = iw;
    const int g_y_inner_size = ow;
    const int outer_dim = inputs[0]->size() / g_x_inner_size;
    if (mode_ == "linear") {
      const float sx = compute_scale(iw, ow, align_corners_);
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
      for (int n = 0; n < outer_dim; n++) {
        auto dst = g_x + n * g_x_inner_size;
        auto src = g_y + n * g_y_inner_size;
        linear_interpolate_1d_backward(dst, src, iw, ow, sx, half_pixel_);
      }
    } else if (mode_ == "nearest") {
      const float sx =
          compute_scale_for_nn(iw, ow, align_corners_, half_pixel_for_nn_);
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
      for (int n = 0; n < outer_dim; n++) {
        auto dst = g_x + n * g_x_inner_size;
        auto src = g_y + n * g_y_inner_size;
        nearest_interpolate_1d_backward(dst, src, iw, ow, sx, half_pixel_,
                                        half_pixel_for_nn_);
      }
    }
  }

  else if (output_size_.size() == 2) {
    const int iw = inputs[0]->shape()[ndim - 1];
    const int ih = inputs[0]->shape()[ndim - 2];
    const int ow = outputs[0]->shape()[ndim - 1];
    const int oh = outputs[0]->shape()[ndim - 2];
    const int g_x_inner_size = iw * ih;
    const int g_y_inner_size = ow * oh;
    const int outer_dim = inputs[0]->size() / g_x_inner_size;
    if (mode_ == "linear") {
      const float sx = compute_scale(iw, ow, align_corners_);
      const float sy = compute_scale(ih, oh, align_corners_);
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
      for (int n = 0; n < outer_dim; n++) {
        auto dst = g_x + n * g_x_inner_size;
        auto src = g_y + n * g_y_inner_size;
        linear_interpolate_2d_backward(dst, src, iw, ih, ow, oh, sx, sy,
                                       half_pixel_);
      }
    } else if (mode_ == "nearest") {
      const float sx =
          compute_scale_for_nn(iw, ow, align_corners_, half_pixel_for_nn_);
      const float sy =
          compute_scale_for_nn(ih, oh, align_corners_, half_pixel_for_nn_);
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
      for (int n = 0; n < outer_dim; n++) {
        auto dst = g_x + n * g_x_inner_size;
        auto src = g_y + n * g_y_inner_size;
        nearest_interpolate_2d_backward(dst, src, iw, ih, ow, oh, sx, sy,
                                        half_pixel_, half_pixel_for_nn_);
      }
    }
  }

  else if (output_size_.size() == 3) {
    const int iw = inputs[0]->shape()[ndim - 1];
    const int ih = inputs[0]->shape()[ndim - 2];
    const int id = inputs[0]->shape()[ndim - 3];
    const int ow = outputs[0]->shape()[ndim - 1];
    const int oh = outputs[0]->shape()[ndim - 2];
    const int od = outputs[0]->shape()[ndim - 3];
    const int g_x_inner_size = iw * ih * id;
    const int g_y_inner_size = ow * oh * od;
    const int outer_dim = inputs[0]->size() / g_x_inner_size;
    if (mode_ == "linear") {
      const float sx = compute_scale(iw, ow, align_corners_);
      const float sy = compute_scale(ih, oh, align_corners_);
      const float sz = compute_scale(id, od, align_corners_);
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
      for (int n = 0; n < outer_dim; n++) {
        auto dst = g_x + n * g_x_inner_size;
        auto src = g_y + n * g_y_inner_size;
        linear_interpolate_3d_backward(dst, src, iw, ih, id, ow, oh, od, sx, sy,
                                       sz, half_pixel_);
      }
    } else if (mode_ == "nearest") {
      const float sx =
          compute_scale_for_nn(iw, ow, align_corners_, half_pixel_for_nn_);
      const float sy =
          compute_scale_for_nn(ih, oh, align_corners_, half_pixel_for_nn_);
      const float sz =
          compute_scale_for_nn(id, od, align_corners_, half_pixel_for_nn_);
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
      for (int n = 0; n < outer_dim; n++) {
        auto dst = g_x + n * g_x_inner_size;
        auto src = g_y + n * g_y_inner_size;
        nearest_interpolate_3d_backward(dst, src, iw, ih, id, ow, oh, od, sx,
                                        sy, sz, half_pixel_,
                                        half_pixel_for_nn_);
      }
    }
  }
}
}
