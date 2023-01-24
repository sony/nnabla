// Copyright 2018,2019,2020,2021 Sony Corporation.
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
#include <nbla/function/onnx_resize.hpp>
#include <nbla/variable.hpp>

#include <array>

namespace nbla {

NBLA_REGISTER_FUNCTION_SOURCE(ONNXResize, const vector<float> &,
                              const vector<float> &, const vector<int> &,
                              const string &, const string &, float, int, float,
                              const string &);

template <typename T> const T &clamp(const T &v, const T &low, const T &high) {
  return std::min(std::max(v, low), high);
}

struct LinearInterpolation {
  static constexpr int KERNEL = 2;

  // Compute coefficients for linear interpolation
  std::array<float, KERNEL> compute_coeffs(float fx, int iw,
                                           const ResizeOption &opt) const {
    const auto x_int = static_cast<int>(std::floor(fx));
    const auto x0 = clamp(x_int + 0, 0, iw - 1);
    const auto lx1 = fx - float(x0);
    const auto lx0 = float(1) - lx1;
    return {lx0, lx1};
  }
};

struct CubicInterpolation {
  static constexpr int KERNEL = 4;

  // Compute coefficients for cubic interpolation
  std::array<float, KERNEL> compute_coeffs(float fx, int iw,
                                           const ResizeOption &opt) const {
    const auto a = opt.cubic_coeff_a;

    // 0 <= x <= 1
    const auto coeff_0_1 = [](float a, float x) {
      const auto x3 = x * x * x;
      const auto x2 = x * x;
      return (a + 2.f) * x3 - (a + 3.f) * x2 + 1.f;
    };
    // 1 <= x <= 2
    const auto coeff_1_2 = [](float a, float x) {
      const auto x3 = x * x * x;
      const auto x2 = x * x;
      return a * x3 - 5.f * a * x2 + 8.f * a * x - 4.f * a;
    };

    const auto lx = fx - std::floor(fx);
    auto c0 = coeff_1_2(a, 1.f + lx);
    auto c1 = coeff_0_1(a, 0.f + lx);
    auto c2 = coeff_0_1(a, 1.f - lx);
    auto c3 = coeff_1_2(a, 2.f - lx);

    if (opt.exclude_outside) {
      const auto x_int = static_cast<int>(std::floor(fx));

      // (unsigned(x) >= y) means (x < 0 || x >= y).
      c0 = (unsigned(x_int - 1) >= unsigned(iw)) ? 0.0 : c0;
      c1 = (unsigned(x_int + 0) >= unsigned(iw)) ? 0.0 : c1;
      c2 = (unsigned(x_int + 1) >= unsigned(iw)) ? 0.0 : c2;
      c3 = (unsigned(x_int + 2) >= unsigned(iw)) ? 0.0 : c3;

      // Normalize coeffs so that their sum is 1.0
      float coeff_sum = c0 + c1 + c2 + c3;
      c0 /= coeff_sum;
      c1 /= coeff_sum;
      c2 /= coeff_sum;
      c3 /= coeff_sum;
    }

    return {c0, c1, c2, c3};
  }
};

static float get_src_index(float scale, int dst_index, int dst_size,
                           int src_size, const ResizeOption &opt, int axis) {
  const auto x_resized = float(dst_index);
  const auto length_resized = float(dst_size);
  const auto length_orig = float(src_size);

  const auto handle_scale_1 = [&](float x_orig) {
    return (scale == 1.f) ? x_resized : x_orig;
  };

  // (x_resized / scale) can be implemented as (x_resized * (1 / scale)),
  // but it causes floating-point errors and the errors sometimes make
  // a big difference between the interpolation results especially
  // when the mode is "nearest".
  // This code employs the former approach to match the result with ONNX
  // Runtime.
  switch (opt.coord_mode) {
  case ResizeCoordTransformMode::HALF_PIXEL:
    return handle_scale_1((x_resized + 0.5f) / scale - 0.5f);
  case ResizeCoordTransformMode::PYTORCH_HALF_PIXEL:
    return handle_scale_1(
        length_resized > 1.f ? (x_resized + 0.5f) / scale - 0.5f : 0.f);
  case ResizeCoordTransformMode::ALIGN_CORNERS:
    return handle_scale_1(length_resized == 1.f
                              ? 0.f
                              : x_resized * (length_orig - 1.f) /
                                    (length_resized - 1.f));
  case ResizeCoordTransformMode::ASYMMETRIC:
    return handle_scale_1(x_resized / scale);
  case ResizeCoordTransformMode::TF_HALF_PIXEL_FOR_NN:
    return handle_scale_1((x_resized + 0.5f) / scale);
  case ResizeCoordTransformMode::TF_CROP_AND_RESIZE:
    const auto ndim = opt.num_dims;
    const auto i = opt.num_outer_dims + axis;
    const auto start_x = float(opt.roi[0 * ndim + i]);
    const auto end_x = float(opt.roi[1 * ndim + i]);
    return handle_scale_1(length_resized > 1.f
                              ? start_x * (length_orig - 1.f) +
                                    x_resized * (end_x - start_x) *
                                        (length_orig - 1.f) /
                                        (length_resized - 1.f)
                              : 0.5f * (start_x + end_x) * (length_orig - 1.f));
  }
  NBLA_ERROR(error_code::not_implemented,
             "ResizeCoordTransformMode %d is not implemented.",
             int(opt.coord_mode));
}

// Compute neighbor indices for linear interpolation (KERNEL = 2)
template <int KERNEL>
typename std::enable_if<KERNEL == 2, std::array<int, KERNEL>>::type
compute_neighbors(float fx, int iw) {
  // x0 <= fx < x1
  const auto x_int = static_cast<int>(std::floor(fx));
  const auto x0 = clamp(x_int + 0, 0, iw - 1);
  const auto x1 = clamp(x_int + 1, 0, iw - 1);
  return {x0, x1};
}

// Compute neighbor indices for cubic interpolation (KERNEL = 4)
template <int KERNEL>
typename std::enable_if<KERNEL == 4, std::array<int, KERNEL>>::type
compute_neighbors(float fx, int iw) {
  // x0 < x1 <= fx < x2 < x3
  const auto x_int = static_cast<int>(std::floor(fx));
  const auto x0 = clamp(x_int - 1, 0, iw - 1);
  const auto x1 = clamp(x_int + 0, 0, iw - 1);
  const auto x2 = clamp(x_int + 1, 0, iw - 1);
  const auto x3 = clamp(x_int + 2, 0, iw - 1);
  return {x0, x1, x2, x3};
}

template <typename T, typename Interpolation>
void generic_interpolate_1d(const T *src, T *dst, const int iw, const int ow,
                            const float sx, const ResizeOption &opt,
                            const Interpolation &interp) {
  constexpr auto KERNEL = Interpolation::KERNEL;
  for (int ox = 0; ox < ow; ox++) {
    const auto fx = get_src_index(sx, ox, ow, iw, opt, 0);

    const auto dst_index = ox;
    if (opt.coord_mode == ResizeCoordTransformMode::TF_CROP_AND_RESIZE &&
        (fx < 0 || fx > float(iw - 1))) {
      dst[dst_index] = T(opt.extrapolation_value);
      continue;
    }

    const auto cx = interp.compute_coeffs(fx, iw, opt);
    const auto ix = compute_neighbors<KERNEL>(fx, iw);

    float xval = 0.f;
    for (int i = 0; i < KERNEL; ++i) {
      xval += cx[i] * src[ix[i]];
    }
    dst[dst_index] = xval;
  }
}

template <typename T, typename Interpolation>
void generic_interpolate_2d(const T *src, T *dst, const int iw, const int ih,
                            const int ow, const int oh, const float sx,
                            const float sy, const ResizeOption &opt,
                            const Interpolation &interp) {
  constexpr auto KERNEL = Interpolation::KERNEL;
  for (int oy = 0; oy < oh; oy++) {
    const auto fy = get_src_index(sy, oy, oh, ih, opt, 0);
    const auto cy = interp.compute_coeffs(fy, ih, opt);
    const auto iy = compute_neighbors<KERNEL>(fy, ih);

    for (int ox = 0; ox < ow; ox++) {
      const auto fx = get_src_index(sx, ox, ow, iw, opt, 1);

      const auto dst_index = oy * ow + ox;
      if (opt.coord_mode == ResizeCoordTransformMode::TF_CROP_AND_RESIZE &&
          (fy < 0 || fy > float(ih - 1) || //
           fx < 0 || fx > float(iw - 1))) {
        dst[dst_index] = T(opt.extrapolation_value);
        continue;
      }

      const auto cx = interp.compute_coeffs(fx, iw, opt);
      const auto ix = compute_neighbors<KERNEL>(fx, iw);

      float yval = 0.f;
      for (int i = 0; i < KERNEL; ++i) {
        float xval = 0.f;
        for (int j = 0; j < KERNEL; ++j) {
          xval += cx[j] * src[iy[i] * iw + ix[j]];
        }
        yval += cy[i] * xval;
      }
      dst[dst_index] = yval;
    }
  }
}

template <typename T, typename Interpolation>
void generic_interpolate_3d(const T *src, T *dst, const int iw, const int ih,
                            const int id, const int ow, const int oh,
                            const int od, const float sx, const float sy,
                            const float sz, const ResizeOption &opt,
                            const Interpolation &interp) {
  constexpr auto KERNEL = Interpolation::KERNEL;
  for (int oz = 0; oz < od; oz++) {
    const auto fz = get_src_index(sz, oz, od, id, opt, 0);
    const auto cz = interp.compute_coeffs(fz, id, opt);
    const auto iz = compute_neighbors<KERNEL>(fz, id);

    for (int oy = 0; oy < oh; oy++) {
      const auto fy = get_src_index(sy, oy, oh, ih, opt, 1);
      const auto cy = interp.compute_coeffs(fy, ih, opt);
      const auto iy = compute_neighbors<KERNEL>(fy, ih);

      for (int ox = 0; ox < ow; ox++) {
        const auto fx = get_src_index(sx, ox, ow, iw, opt, 2);

        const auto dst_index = (oz * oh + oy) * ow + ox;
        if (opt.coord_mode == ResizeCoordTransformMode::TF_CROP_AND_RESIZE &&
            (fz < 0 || fz > float(id - 1) || //
             fy < 0 || fy > float(ih - 1) || //
             fx < 0 || fx > float(iw - 1))) {
          dst[dst_index] = T(opt.extrapolation_value);
          continue;
        }

        const auto cx = interp.compute_coeffs(fx, iw, opt);
        const auto ix = compute_neighbors<KERNEL>(fx, iw);

        float zval = 0.f;
        for (int i = 0; i < KERNEL; ++i) {
          float yval = 0.f;
          for (int j = 0; j < KERNEL; ++j) {
            float xval = 0.f;
            for (int k = 0; k < KERNEL; ++k) {
              xval += cx[k] * src[(iz[i] * ih + iy[j]) * iw + ix[k]];
            }
            yval += cy[j] * xval;
          }
          zval += cz[i] * yval;
        }
        dst[dst_index] = zval;
      }
    }
  }
}

template <typename T>
void linear_interpolate_1d(const T *src, T *dst, const int iw, const int ow,
                           const float sx, const ResizeOption &opt) {
  generic_interpolate_1d(src, dst, iw, ow, sx, opt, LinearInterpolation());
}

template <typename T>
void linear_interpolate_2d(const T *src, T *dst, const int iw, const int ih,
                           const int ow, const int oh, const float sx,
                           const float sy, const ResizeOption &opt) {
  generic_interpolate_2d(src, dst, iw, ih, ow, oh, sx, sy, opt,
                         LinearInterpolation());
}

template <typename T>
void linear_interpolate_3d(const T *src, T *dst, const int iw, const int ih,
                           const int id, const int ow, const int oh,
                           const int od, const float sx, const float sy,
                           const float sz, const ResizeOption &opt) {
  generic_interpolate_3d(src, dst, iw, ih, id, ow, oh, od, sx, sy, sz, opt,
                         LinearInterpolation());
}

template <typename T>
void cubic_interpolate_1d(const T *src, T *dst, const int iw, const int ow,
                          const float sx, const ResizeOption &opt) {
  generic_interpolate_1d(src, dst, iw, ow, sx, opt, CubicInterpolation());
}

template <typename T>
void cubic_interpolate_2d(const T *src, T *dst, const int iw, const int ih,
                          const int ow, const int oh, const float sx,
                          const float sy, const ResizeOption &opt) {
  generic_interpolate_2d(src, dst, iw, ih, ow, oh, sx, sy, opt,
                         CubicInterpolation());
}

template <typename T>
void cubic_interpolate_3d(const T *src, T *dst, const int iw, const int ih,
                          const int id, const int ow, const int oh,
                          const int od, const float sx, const float sy,
                          const float sz, const ResizeOption &opt) {
  generic_interpolate_3d(src, dst, iw, ih, id, ow, oh, od, sx, sy, sz, opt,
                         CubicInterpolation());
}

static int get_nearest_index(float fx, const ResizeOption &opt) {
  switch (opt.nearest_mode) {
  case ResizeNearestMode::ROUND_PREFER_CEIL:
    return static_cast<int>(std::round(fx));
  case ResizeNearestMode::FLOOR:
    return static_cast<int>(std::floor(fx));
  case ResizeNearestMode::CEIL:
    return static_cast<int>(std::ceil(fx));
  case ResizeNearestMode::ROUND_PREFER_FLOOR:
  // fall through
  default:
    if (fx == std::floor(fx) + 0.5) {
      return static_cast<int>(std::floor(fx));
    } else {
      return static_cast<int>(std::round(fx));
    }
  }
}

template <typename T>
void nearest_interpolate_1d(const T *src, T *dst, const int iw, const int ow,
                            const float sx, const ResizeOption &opt) {
  for (int ox = 0; ox < ow; ox++) {
    const auto fx = get_src_index(sx, ox, ow, iw, opt, 0);
    const auto ix = clamp(get_nearest_index(fx, opt), 0, iw - 1);

    const auto dst_index = ox;
    if (opt.coord_mode == ResizeCoordTransformMode::TF_CROP_AND_RESIZE &&
        (fx < 0 || fx > float(iw - 1))) {
      dst[dst_index] = T(opt.extrapolation_value);
      continue;
    }

    dst[dst_index] = src[ix];
  }
}

template <typename T>
void nearest_interpolate_2d(const T *src, T *dst, const int iw, const int ih,
                            const int ow, const int oh, const float sx,
                            const float sy, const ResizeOption &opt) {
  for (int oy = 0; oy < oh; oy++) {
    const auto fy = get_src_index(sy, oy, oh, ih, opt, 0);
    const auto iy = clamp(get_nearest_index(fy, opt), 0, ih - 1);
    for (int ox = 0; ox < ow; ox++) {
      const auto fx = get_src_index(sx, ox, ow, iw, opt, 1);
      const auto ix = clamp(get_nearest_index(fx, opt), 0, iw - 1);

      const auto dst_index = oy * ow + ox;
      if (opt.coord_mode == ResizeCoordTransformMode::TF_CROP_AND_RESIZE &&
          (fy < 0 || fy > float(ih - 1) || //
           fx < 0 || fx > float(iw - 1))) {
        dst[dst_index] = T(opt.extrapolation_value);
        continue;
      }

      dst[dst_index] = src[iy * iw + ix];
    }
  }
}

template <typename T>
void nearest_interpolate_3d(const T *src, T *dst, const int iw, const int ih,
                            const int id, const int ow, const int oh,
                            const int od, const float sx, const float sy,
                            const float sz, const ResizeOption &opt) {
  for (int oz = 0; oz < od; oz++) {
    const auto fz = get_src_index(sz, oz, od, id, opt, 0);
    const auto iz = clamp(get_nearest_index(fz, opt), 0, id - 1);
    for (int oy = 0; oy < oh; oy++) {
      const auto fy = get_src_index(sy, oy, oh, ih, opt, 1);
      const auto iy = clamp(get_nearest_index(fy, opt), 0, ih - 1);
      for (int ox = 0; ox < ow; ox++) {
        const auto fx = get_src_index(sx, ox, ow, iw, opt, 2);
        const auto ix = clamp(get_nearest_index(fx, opt), 0, iw - 1);

        const auto dst_index = (oz * oh + oy) * ow + ox;
        if (opt.coord_mode == ResizeCoordTransformMode::TF_CROP_AND_RESIZE &&
            (fz < 0 || fz > float(id - 1) || //
             fy < 0 || fy > float(ih - 1) || //
             fx < 0 || fx > float(iw - 1))) {
          dst[dst_index] = T(opt.extrapolation_value);
          continue;
        }

        dst[dst_index] = src[iz * ih * iw + iy * iw + ix];
      }
    }
  }
}

static ResizeCoordTransformMode string_to_coord_mode(const string &s) {
  if (s == "half_pixel")
    return ResizeCoordTransformMode::HALF_PIXEL;
  if (s == "pytorch_half_pixel")
    return ResizeCoordTransformMode::PYTORCH_HALF_PIXEL;
  if (s == "align_corners")
    return ResizeCoordTransformMode::ALIGN_CORNERS;
  if (s == "asymmetric")
    return ResizeCoordTransformMode::ASYMMETRIC;
  if (s == "tf_half_pixel_for_nn")
    return ResizeCoordTransformMode::TF_HALF_PIXEL_FOR_NN;
  if (s == "tf_crop_and_resize")
    return ResizeCoordTransformMode::TF_CROP_AND_RESIZE;
  NBLA_ERROR(error_code::value,
             "coordinate_transformation_mode '%s' is not supported.",
             s.c_str());
}

static ResizeNearestMode string_to_nearest_mode(const string &s) {
  if (s == "round_prefer_floor")
    return ResizeNearestMode::ROUND_PREFER_FLOOR;
  if (s == "round_prefer_ceil")
    return ResizeNearestMode::ROUND_PREFER_CEIL;
  if (s == "floor")
    return ResizeNearestMode::FLOOR;
  if (s == "ceil")
    return ResizeNearestMode::CEIL;
  NBLA_ERROR(error_code::value, "nearest_mode '%s' is not supported.",
             s.c_str());
}

template <typename T>
void ONNXResize<T>::setup_impl(const Variables &inputs,
                               const Variables &outputs) {
  const auto inshape = inputs[0]->shape();
  const auto ndim = inshape.size();

  // Check coordinate_transformation_mode
  NBLA_CHECK((coordinate_transformation_mode_ == "half_pixel") ||
                 (coordinate_transformation_mode_ == "pytorch_half_pixel") ||
                 (coordinate_transformation_mode_ == "align_corners") ||
                 (coordinate_transformation_mode_ == "asymmetric") ||
                 (coordinate_transformation_mode_ == "tf_half_pixel_for_nn") ||
                 (coordinate_transformation_mode_ == "tf_crop_and_resize"),
             error_code::value,
             "coordinate_transformation_mode '%s' is not supported.",
             coordinate_transformation_mode_.c_str());
  const auto enum_coord_mode =
      string_to_coord_mode(coordinate_transformation_mode_);
  const auto enum_nearest_mode = string_to_nearest_mode(nearest_mode_);

  // Check if either sizes or scales is specified
  const auto sizes_specified = (sizes_.size() >= 1 && scales_.size() == 0);
  const auto scales_specified = (scales_.size() >= 1 && sizes_.size() == 0);
  NBLA_CHECK(sizes_specified || scales_specified, error_code::value,
             "One of sizes and scales must be specified.");

  // Convert sizes or scales to outshape and actual_scales
  Shape_t outshape(ndim);
  actual_scales_.clear();
  if (sizes_specified) {
    NBLA_CHECK(sizes_.size() == ndim, error_code::value,
               "The number of sizes dimensions must be same as input.");
    std::copy(sizes_.begin(), sizes_.end(), outshape.begin());
    for (size_t i = 0; i < sizes_.size(); ++i) {
      const auto output_size = sizes_[i];
      const auto input_size = inshape[i];
      const auto scale = float(output_size) / float(input_size);
      actual_scales_.emplace_back(scale);
    }
  } else if (scales_specified) {
    NBLA_CHECK(scales_.size() == ndim, error_code::value,
               "The number of scales dimensions must be same as input.");
    for (size_t i = 0; i < outshape.size(); ++i) {
      outshape[i] = Size_t(std::floor(float(inshape[i]) * scales_[i]));
    }
    // Use argument scales. This value is not always same as
    // (float(output_size) / float(input_size)).
    actual_scales_ = scales_;
  }

  // Check the number of resize dimensions
  Size_t num_outer_dims = 0;
  for (size_t i = 0; i < actual_scales_.size(); ++i) {
    if (actual_scales_[i] != 1.f)
      break;
    num_outer_dims += 1;
  }
  const Size_t num_resize_dims = actual_scales_.size() - num_outer_dims;
  NBLA_CHECK(1 <= num_resize_dims && num_resize_dims <= 3,
             error_code::not_implemented,
             "Only 1-D, 2-D and 3-D interpolation are implemented.");

  // Check mode
  NBLA_CHECK((mode_ == "linear") || (mode_ == "cubic") || (mode_ == "nearest"),
             error_code::value, "mode '%s' are not supported.", mode_.c_str());

  // Check nearest_mode
  if (mode_ == "nearest") {
    NBLA_CHECK((nearest_mode_ == "round_prefer_floor") ||
                   (nearest_mode_ == "round_prefer_ceil") ||
                   (nearest_mode_ == "floor") || (nearest_mode_ == "ceil"),
               error_code::value, "nearest_mode '%s' is not supported.",
               nearest_mode_.c_str());
  }

  // Check roi
  if (enum_coord_mode == ResizeCoordTransformMode::TF_CROP_AND_RESIZE) {
    NBLA_CHECK(roi_.size() == 2 * ndim, error_code::value,
               "The size of roi is invalid: %zu (expect: %zu).", roi_.size(),
               2 * ndim);
    for (Size_t i = 0; i < num_outer_dims; ++i) {
      NBLA_CHECK(
          roi_[0 * ndim + i] == 0.f && roi_[1 * ndim + i] == 1.f,
          error_code::not_implemented,
          "The RoI values of non-resized dimensions must be (0.0, 1.0).");
    }
  }

  option_ =
      ResizeOption(enum_coord_mode, cubic_coeff_a_, bool(exclude_outside_),
                   extrapolation_value_, enum_nearest_mode, roi_,
                   num_outer_dims, num_resize_dims);

  outputs[0]->reshape(outshape, true);
}

template <typename T>
void ONNXResize<T>::forward_impl(const Variables &inputs,
                                 const Variables &outputs) {
  const auto src = inputs[0]->get_data_pointer<T>(this->ctx_);
  const auto dst = outputs[0]->cast_data_and_get_pointer<T>(this->ctx_, true);

  const int ndim = inputs[0]->ndim();
  const int num_outer_dims = option_.num_outer_dims;
  const int num_resize_dims = option_.num_resize_dims;

  if (num_resize_dims == 1) {
    const int iw = inputs[0]->shape()[ndim - 1];
    const int ow = outputs[0]->shape()[ndim - 1];
    const int outer_dim = inputs[0]->size() / iw;
    const int src_inner_size = iw;
    const int dst_inner_size = ow;
    const float sx = actual_scales_[num_outer_dims + 0];
    if (mode_ == "linear") {
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
      for (int n = 0; n < outer_dim; n++) {
        auto src_ptr = src + n * src_inner_size;
        auto dst_ptr = dst + n * dst_inner_size;
        linear_interpolate_1d(src_ptr, dst_ptr, iw, ow, sx, option_);
      }
    } else if (mode_ == "cubic") {
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
      for (int n = 0; n < outer_dim; n++) {
        auto src_ptr = src + n * src_inner_size;
        auto dst_ptr = dst + n * dst_inner_size;
        cubic_interpolate_1d(src_ptr, dst_ptr, iw, ow, sx, option_);
      }
    } else if (mode_ == "nearest") {
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
      for (int n = 0; n < outer_dim; n++) {
        auto src_ptr = src + n * src_inner_size;
        auto dst_ptr = dst + n * dst_inner_size;
        nearest_interpolate_1d(src_ptr, dst_ptr, iw, ow, sx, option_);
      }
    }
  }

  else if (num_resize_dims == 2) {
    const int iw = inputs[0]->shape()[ndim - 1];
    const int ih = inputs[0]->shape()[ndim - 2];
    const int ow = outputs[0]->shape()[ndim - 1];
    const int oh = outputs[0]->shape()[ndim - 2];
    const int outer_dim = inputs[0]->size() / (iw * ih);
    const int src_inner_size = iw * ih;
    const int dst_inner_size = ow * oh;
    const float sy = actual_scales_[num_outer_dims + 0];
    const float sx = actual_scales_[num_outer_dims + 1];
    if (mode_ == "linear") {
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
      for (int n = 0; n < outer_dim; n++) {
        auto src_ptr = src + n * src_inner_size;
        auto dst_ptr = dst + n * dst_inner_size;
        linear_interpolate_2d(src_ptr, dst_ptr, iw, ih, ow, oh, sx, sy,
                              option_);
      }
    } else if (mode_ == "cubic") {
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
      for (int n = 0; n < outer_dim; n++) {
        auto src_ptr = src + n * src_inner_size;
        auto dst_ptr = dst + n * dst_inner_size;
        cubic_interpolate_2d(src_ptr, dst_ptr, iw, ih, ow, oh, sx, sy, option_);
      }
    } else if (mode_ == "nearest") {
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
      for (int n = 0; n < outer_dim; n++) {
        auto src_ptr = src + n * src_inner_size;
        auto dst_ptr = dst + n * dst_inner_size;
        nearest_interpolate_2d(src_ptr, dst_ptr, iw, ih, ow, oh, sx, sy,
                               option_);
      }
    }
  }

  else if (num_resize_dims == 3) {
    const int iw = inputs[0]->shape()[ndim - 1];
    const int ih = inputs[0]->shape()[ndim - 2];
    const int id = inputs[0]->shape()[ndim - 3];
    const int ow = outputs[0]->shape()[ndim - 1];
    const int oh = outputs[0]->shape()[ndim - 2];
    const int od = outputs[0]->shape()[ndim - 3];
    const int outer_dim = inputs[0]->size() / (id * iw * ih);
    const int src_inner_size = iw * ih * id;
    const int dst_inner_size = ow * oh * od;
    const float sz = actual_scales_[num_outer_dims + 0];
    const float sy = actual_scales_[num_outer_dims + 1];
    const float sx = actual_scales_[num_outer_dims + 2];
    if (mode_ == "linear") {
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
      for (int n = 0; n < outer_dim; n++) {
        auto src_ptr = src + n * src_inner_size;
        auto dst_ptr = dst + n * dst_inner_size;
        linear_interpolate_3d(src_ptr, dst_ptr, iw, ih, id, ow, oh, od, sx, sy,
                              sz, option_);
      }
    } else if (mode_ == "cubic") {
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
      for (int n = 0; n < outer_dim; n++) {
        auto src_ptr = src + n * src_inner_size;
        auto dst_ptr = dst + n * dst_inner_size;
        cubic_interpolate_3d(src_ptr, dst_ptr, iw, ih, id, ow, oh, od, sx, sy,
                             sz, option_);
      }
    } else if (mode_ == "nearest") {
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
      for (int n = 0; n < outer_dim; n++) {
        auto src_ptr = src + n * src_inner_size;
        auto dst_ptr = dst + n * dst_inner_size;
        nearest_interpolate_3d(src_ptr, dst_ptr, iw, ih, id, ow, oh, od, sx, sy,
                               sz, option_);
      }
    }
  }
}

template <typename T>
void ONNXResize<T>::backward_impl(const Variables &inputs,
                                  const Variables &outputs,
                                  const vector<bool> &propagate_down,
                                  const vector<bool> &accum) {
  if (!(propagate_down[0])) {
    return;
  }

  NBLA_ERROR(error_code::not_implemented,
             "ONNXResize<T>::backward is currently not implemented.");
}
}
