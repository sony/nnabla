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
#include <nbla/function/warp_by_grid.hpp>
#include <nbla/utils/nd_index.hpp>
#include <nbla/variable.hpp>

namespace nbla {

NBLA_REGISTER_FUNCTION_SOURCE(WarpByGrid, const string &, const string &, bool,
                              bool);

template <typename T, bool align_corners = false>
inline T unnormalize_grid_with(T x, const int S) {
  if (align_corners) {
    // [-1, 1] <--> [0, S - 1]
    return (x + T(1)) * (S - T(1)) / T(2);
  } else {
    // [-1, 1] <--> [-0.5, S - 0.5] = [0 - 0.5, S - 1 + 0.5]
    return ((x + T(1)) * S - T(1)) / T(2);
  }
}

template <typename T>
inline T get_src_findex_with_zero_pad(const T s, const int S) {
  return s;
}

template <typename T>
inline T get_src_findex_with_repeat_pad(const T s, const int S) {
  if (s < 0) {
    return 0;
  } else if (s > S - 1) {
    return S - 1;
  } else {
    return s;
  }
}

template <typename T> inline T reflect(const T s, const int L, const int U) {
  auto len = (U - L);
  if (s < L) {
    auto d = L - s;
    auto nf = d / len;
    auto n = static_cast<int>(nf);
    auto r = d - n * len;
    if (n % 2 == 0) {
      return L + r;
    } else {
      return U - r;
    }
  } else if (s > U) {
    auto d = s - U;
    auto nf = d / len;
    auto n = static_cast<int>(nf);
    auto r = d - n * len;
    if (n % 2 == 0) {
      return U - r;
    } else {
      return L + r;
    }
  } else {
    return s;
  }
}

template <typename T, bool align_corners = false>
inline T get_src_findex_with_reflect_pad(T s, const int S) {
  if (align_corners) {
    return reflect(s, T(0), T(S - 1));
  } else {
    // address the borders {-0.5, S - 0.5} condition by two multiplication
    auto sf = reflect(T(2) * s, T(-1), T(2) * T(S) - T(1));
    sf = sf * T(0.5);
    sf = get_src_findex_with_repeat_pad(sf, S);
    return sf;
  }
}

template <typename T>
inline T get_grad_coef_with_zero_pad(const T s, const int S) {
  return T(1);
}

template <typename T>
inline T get_grad_coef_with_repeat_pad(const T s, const int S) {
  if (s < 0) {
    return 0;
  } else if (s > S - 1) {
    return 0;
  } else {
    return 1;
  }
}

template <typename T>
inline T reflect_coef(const T s, const int L, const int U) {
  auto len = (U - L);
  if (s < L) {
    auto d = L - s;
    auto nf = d / len;
    auto n = static_cast<int>(nf);
    if (n % 2 == 0) {
      return T(-1);
    } else {
      return T(1);
    }
  } else if (s > U) {
    auto d = s - U;
    auto nf = d / len;
    auto n = static_cast<int>(nf);
    if (n % 2 == 0) {
      return T(-1);
    } else {
      return T(1);
    }
  } else {
    return T(1);
  }
}

template <typename T, bool align_corners = false>
inline T get_grad_coef_with_reflect_pad(T s, const int S) {
  if (align_corners) {
    return reflect_coef(s, T(0), T(S - 1));
  } else {
    // address the borders {-0.5, S - 0.5} condition by two multiplication
    auto coef = reflect_coef(T(2) * s, T(-1), T(2) * T(S) - T(1));
    auto sf = reflect(T(2) * s, T(-1), T(2) * T(S) - T(1));
    sf = sf * T(0.5);
    coef *= get_grad_coef_with_repeat_pad(sf, S);
    return coef;
  }
}

template <typename T>
inline T get_pixel_value_2d(const T *input, int b, int c, int h, int w,
                            const int H, const int W, const Shape_t istride) {
  if ((h >= 0 && h < H) && (w >= 0 && w < W)) {
    auto idx = ndi::nd2flat(Shape_t{b, c, h, w}, istride);
    return input[idx];
  } else {
    return T(0);
  }
}

template <typename T>
inline T get_pixel_value_3d(const T *input, int b, int c, int d, int h, int w,
                            const int D, const int H, const int W,
                            const Shape_t istride) {
  if ((d >= 0 && d < D) && (h >= 0 && h < H) && (w >= 0 && w < W)) {
    auto idx = ndi::nd2flat(Shape_t{b, c, d, h, w}, istride);
    return input[idx];
  } else {
    return T(0);
  }
}

template <typename T>
inline void backward_data_2d(T *igrad, const T ograd, const T p, const T q,
                             int b, int c, int h, int w, const int H,
                             const int W, const Shape_t istride) {
  if ((h >= 0 && h < H) && (w >= 0 && w < W)) {
    auto idx = ndi::nd2flat(Shape_t{b, c, h, w}, istride);
    igrad[idx] += ograd * p * q;
  }
}

template <typename T>
inline void backward_data_3d(T *igrad, const T ograd, const T p, const T q,
                             const T r, int b, int c, int d, int h, int w,
                             const int D, const int H, const int W,
                             const Shape_t istride) {
  if ((d >= 0 && d < D) && (h >= 0 && h < H) && (w >= 0 && w < W)) {
    auto idx = ndi::nd2flat(Shape_t{b, c, d, h, w}, istride);
    igrad[idx] += ograd * p * q * r;
  }
}

/*
  Forward implementations
 */

template <typename T, warp_by_grid::PADDING_MODE padding_mode,
          bool align_corners = false>
void warp_linear_forward_2d(T *output, const T *input, const T *grid,
                            const Shape_t ishape, const Shape_t oshape,
                            const Shape_t istride, const Shape_t gstride,
                            const Shape_t ostride) {
  auto B = oshape[0];
  auto C = oshape[1];
  auto Ho = oshape[2];
  auto Wo = oshape[3];
  auto Hi = ishape[2];
  auto Wi = ishape[3];
  auto unnormalize_grid = align_corners ? unnormalize_grid_with<T, true>
                                        : unnormalize_grid_with<T, false>;
  auto get_src_findex_with_pad = [&](const T s, const int S) {
    if (padding_mode == warp_by_grid::PADDING_MODE::zero) {
      return get_src_findex_with_zero_pad(s, S);
    } else if (padding_mode == warp_by_grid::PADDING_MODE::reflect) {
      return align_corners ? get_src_findex_with_reflect_pad<T, true>(s, S)
                           : get_src_findex_with_reflect_pad<T, false>(s, S);
    } else if (padding_mode == warp_by_grid::PADDING_MODE::repeat) {
      return get_src_findex_with_repeat_pad(s, S);
    } else {
      return T(-1);
    }
  };

  auto oidx = 0;
  for (auto b = 0; b < B; ++b) {
    for (auto c = 0; c < C; ++c) {
      for (auto h = 0; h < Ho; ++h) {
        for (auto w = 0; w < Wo; ++w) {
          auto gidx = ndi::nd2flat(Shape_t{b, h, w, 0}, gstride);
          auto xn = grid[gidx + 0];
          auto yn = grid[gidx + 1];
          auto xf0 = unnormalize_grid(xn, Wi);
          auto yf0 = unnormalize_grid(yn, Hi);
          auto xf = get_src_findex_with_pad(xf0, Wi);
          auto yf = get_src_findex_with_pad(yf0, Hi);
          auto xi0 = static_cast<int>(std::floor(xf));
          auto yi0 = static_cast<int>(std::floor(yf));
          auto xi1 = xi0 + 1;
          auto yi1 = yi0 + 1;
          auto px0 = xf - xi0;
          auto py0 = yf - yi0;
          auto px1 = T(1) - px0;
          auto py1 = T(1) - py0;

          auto v_y0x0 =
              get_pixel_value_2d(input, b, c, yi0, xi0, Hi, Wi, istride);
          auto v_y0x1 =
              get_pixel_value_2d(input, b, c, yi0, xi1, Hi, Wi, istride);
          auto v_y1x0 =
              get_pixel_value_2d(input, b, c, yi1, xi0, Hi, Wi, istride);
          auto v_y1x1 =
              get_pixel_value_2d(input, b, c, yi1, xi1, Hi, Wi, istride);

          auto val = (v_y0x0 * py1 * px1) + (v_y0x1 * py1 * px0) +
                     (v_y1x0 * py0 * px1) + (v_y1x1 * py0 * px0);
          output[oidx++] = val;
        }
      }
    }
  }
}

template <typename T, warp_by_grid::PADDING_MODE padding_mode,
          bool align_corners = false>
void warp_linear_forward_3d(T *output, const T *input, const T *grid,
                            const Shape_t ishape, const Shape_t oshape,
                            const Shape_t istride, const Shape_t gstride,
                            const Shape_t ostride) {
  auto B = oshape[0];
  auto C = oshape[1];
  auto Do = oshape[2];
  auto Ho = oshape[3];
  auto Wo = oshape[4];
  auto Di = ishape[2];
  auto Hi = ishape[3];
  auto Wi = ishape[4];
  auto unnormalize_grid = align_corners ? unnormalize_grid_with<T, true>
                                        : unnormalize_grid_with<T, false>;
  auto get_src_findex_with_pad = [&](const T s, const int S) {
    if (padding_mode == warp_by_grid::PADDING_MODE::zero) {
      return get_src_findex_with_zero_pad(s, S);
    } else if (padding_mode == warp_by_grid::PADDING_MODE::reflect) {
      return align_corners ? get_src_findex_with_reflect_pad<T, true>(s, S)
                           : get_src_findex_with_reflect_pad<T, false>(s, S);
    } else if (padding_mode == warp_by_grid::PADDING_MODE::repeat) {
      return get_src_findex_with_repeat_pad(s, S);
    } else {
      return T(-1);
    }
  };

  auto oidx = 0;
  for (auto b = 0; b < B; ++b) {
    for (auto c = 0; c < C; ++c) {
      for (auto d = 0; d < Do; ++d) {
        for (auto h = 0; h < Ho; ++h) {
          for (auto w = 0; w < Wo; ++w) {
            auto gidx = ndi::nd2flat(Shape_t{b, d, h, w, 0}, gstride);
            auto xn = grid[gidx + 0];
            auto yn = grid[gidx + 1];
            auto zn = grid[gidx + 2];
            auto xf0 = unnormalize_grid(xn, Wi);
            auto yf0 = unnormalize_grid(yn, Hi);
            auto zf0 = unnormalize_grid(zn, Di);
            auto xf = get_src_findex_with_pad(xf0, Wi);
            auto yf = get_src_findex_with_pad(yf0, Hi);
            auto zf = get_src_findex_with_pad(zf0, Di);
            auto xi0 = static_cast<int>(std::floor(xf));
            auto yi0 = static_cast<int>(std::floor(yf));
            auto zi0 = static_cast<int>(std::floor(zf));
            auto xi1 = xi0 + 1;
            auto yi1 = yi0 + 1;
            auto zi1 = zi0 + 1;
            auto px0 = xf - xi0;
            auto py0 = yf - yi0;
            auto pz0 = zf - zi0;
            auto px1 = T(1) - px0;
            auto py1 = T(1) - py0;
            auto pz1 = T(1) - pz0;

            auto v_z0y0x0 = get_pixel_value_3d(input, b, c, zi0, yi0, xi0, Di,
                                               Hi, Wi, istride);
            auto v_z0y0x1 = get_pixel_value_3d(input, b, c, zi0, yi0, xi1, Di,
                                               Hi, Wi, istride);
            auto v_z0y1x0 = get_pixel_value_3d(input, b, c, zi0, yi1, xi0, Di,
                                               Hi, Wi, istride);
            auto v_z0y1x1 = get_pixel_value_3d(input, b, c, zi0, yi1, xi1, Di,
                                               Hi, Wi, istride);
            auto v_z1y0x0 = get_pixel_value_3d(input, b, c, zi1, yi0, xi0, Di,
                                               Hi, Wi, istride);
            auto v_z1y0x1 = get_pixel_value_3d(input, b, c, zi1, yi0, xi1, Di,
                                               Hi, Wi, istride);
            auto v_z1y1x0 = get_pixel_value_3d(input, b, c, zi1, yi1, xi0, Di,
                                               Hi, Wi, istride);
            auto v_z1y1x1 = get_pixel_value_3d(input, b, c, zi1, yi1, xi1, Di,
                                               Hi, Wi, istride);

            auto val = v_z0y0x0 * pz1 * py1 * px1 + v_z0y0x1 * pz1 * py1 * px0 +
                       v_z0y1x0 * pz1 * py0 * px1 + v_z0y1x1 * pz1 * py0 * px0 +
                       v_z1y0x0 * pz0 * py1 * px1 + v_z1y0x1 * pz0 * py1 * px0 +
                       v_z1y1x0 * pz0 * py0 * px1 + v_z1y1x1 * pz0 * py0 * px0;

            output[oidx++] = val;
          }
        }
      }
    }
  }
}

template <typename T, warp_by_grid::PADDING_MODE padding_mode,
          bool align_corners = false>
void warp_nearest_forward_2d(T *output, const T *input, const T *grid,
                             const Shape_t ishape, const Shape_t oshape,
                             const Shape_t istride, const Shape_t gstride,
                             const Shape_t ostride) {
  auto B = oshape[0];
  auto C = oshape[1];
  auto Ho = oshape[2];
  auto Wo = oshape[3];
  auto Hi = ishape[2];
  auto Wi = ishape[3];
  auto unnormalize_grid = align_corners ? unnormalize_grid_with<T, true>
                                        : unnormalize_grid_with<T, false>;
  auto get_src_findex_with_pad = [&](const T s, const int S) {
    if (padding_mode == warp_by_grid::PADDING_MODE::zero) {
      return get_src_findex_with_zero_pad(s, S);
    } else if (padding_mode == warp_by_grid::PADDING_MODE::reflect) {
      return align_corners ? get_src_findex_with_reflect_pad<T, true>(s, S)
                           : get_src_findex_with_reflect_pad<T, false>(s, S);
    } else if (padding_mode == warp_by_grid::PADDING_MODE::repeat) {
      return get_src_findex_with_repeat_pad(s, S);
    } else {
      return T(-1);
    }
  };

  auto oidx = 0;
  for (auto b = 0; b < B; ++b) {
    for (auto c = 0; c < C; ++c) {
      for (auto h = 0; h < Ho; ++h) {
        for (auto w = 0; w < Wo; ++w) {
          auto gidx = ndi::nd2flat(Shape_t{b, h, w, 0}, gstride);
          auto xn = grid[gidx + 0];
          auto yn = grid[gidx + 1];
          auto xf0 = unnormalize_grid(xn, Wi);
          auto yf0 = unnormalize_grid(yn, Hi);
          auto xf = get_src_findex_with_pad(xf0, Wi);
          auto yf = get_src_findex_with_pad(yf0, Hi);
          auto xi = static_cast<int>(std::round(xf));
          auto yi = static_cast<int>(std::round(yf));

          auto vidx = get_pixel_value_2d(input, b, c, yi, xi, Hi, Wi, istride);
          output[oidx++] = vidx;
        }
      }
    }
  }
}

template <typename T, warp_by_grid::PADDING_MODE padding_mode,
          bool align_corners = false>
void warp_nearest_forward_3d(T *output, const T *input, const T *grid,
                             const Shape_t ishape, const Shape_t oshape,
                             const Shape_t istride, const Shape_t gstride,
                             const Shape_t ostride) {
  auto B = oshape[0];
  auto C = oshape[1];
  auto Do = oshape[2];
  auto Ho = oshape[3];
  auto Wo = oshape[4];
  auto Di = ishape[2];
  auto Hi = ishape[3];
  auto Wi = ishape[4];
  auto unnormalize_grid = align_corners ? unnormalize_grid_with<T, true>
                                        : unnormalize_grid_with<T, false>;
  auto get_src_findex_with_pad = [&](const T s, const int S) {
    if (padding_mode == warp_by_grid::PADDING_MODE::zero) {
      return get_src_findex_with_zero_pad(s, S);
    } else if (padding_mode == warp_by_grid::PADDING_MODE::reflect) {
      return align_corners ? get_src_findex_with_reflect_pad<T, true>(s, S)
                           : get_src_findex_with_reflect_pad<T, false>(s, S);
    } else if (padding_mode == warp_by_grid::PADDING_MODE::repeat) {
      return get_src_findex_with_repeat_pad(s, S);
    } else {
      return T(-1);
    }
  };

  auto oidx = 0;
  for (auto b = 0; b < B; ++b) {
    for (auto c = 0; c < C; ++c) {
      for (auto d = 0; d < Do; ++d) {
        for (auto h = 0; h < Ho; ++h) {
          for (auto w = 0; w < Wo; ++w) {
            auto gidx = ndi::nd2flat(Shape_t{b, d, h, w, 0}, gstride);
            auto xn = grid[gidx + 0];
            auto yn = grid[gidx + 1];
            auto zn = grid[gidx + 2];
            auto xf0 = unnormalize_grid(xn, Wi);
            auto yf0 = unnormalize_grid(yn, Hi);
            auto zf0 = unnormalize_grid(zn, Di);
            auto xf = get_src_findex_with_pad(xf0, Wi);
            auto yf = get_src_findex_with_pad(yf0, Hi);
            auto zf = get_src_findex_with_pad(zf0, Di);
            auto xi = static_cast<int>(std::round(xf));
            auto yi = static_cast<int>(std::round(yf));
            auto zi = static_cast<int>(std::round(zf));

            auto vidx = get_pixel_value_3d(input, b, c, zi, yi, xi, Di, Hi, Wi,
                                           istride);
            output[oidx++] = vidx;
          }
        }
      }
    }
  }
}

/*
  Backward implementations wrt the data.
 */

template <typename T, warp_by_grid::PADDING_MODE padding_mode,
          bool align_corners = false>
void warp_linear_backward_data_2d(T *igrad, const T *ograd, const T *grid,
                                  const Shape_t ishape, const Shape_t oshape,
                                  const Shape_t istride, const Shape_t gstride,
                                  const Shape_t ostride) {
  auto B = oshape[0];
  auto C = oshape[1];
  auto Ho = oshape[2];
  auto Wo = oshape[3];
  auto Hi = ishape[2];
  auto Wi = ishape[3];
  auto unnormalize_grid = align_corners ? unnormalize_grid_with<T, true>
                                        : unnormalize_grid_with<T, false>;
  auto get_src_findex_with_pad = [&](const T s, const int S) {
    if (padding_mode == warp_by_grid::PADDING_MODE::zero) {
      return get_src_findex_with_zero_pad(s, S);
    } else if (padding_mode == warp_by_grid::PADDING_MODE::reflect) {
      return align_corners ? get_src_findex_with_reflect_pad<T, true>(s, S)
                           : get_src_findex_with_reflect_pad<T, false>(s, S);
    } else if (padding_mode == warp_by_grid::PADDING_MODE::repeat) {
      return get_src_findex_with_repeat_pad(s, S);
    } else {
      return T(-1);
    }
  };

  auto oidx = 0;
  for (auto b = 0; b < B; ++b) {
    for (auto c = 0; c < C; ++c) {
      for (auto h = 0; h < Ho; ++h) {
        for (auto w = 0; w < Wo; ++w) {
          auto gidx = ndi::nd2flat(Shape_t{b, h, w, 0}, gstride);
          auto xn = grid[gidx + 0];
          auto yn = grid[gidx + 1];
          auto xf0 = unnormalize_grid(xn, Wi);
          auto yf0 = unnormalize_grid(yn, Hi);
          auto xf = get_src_findex_with_pad(xf0, Wi);
          auto yf = get_src_findex_with_pad(yf0, Hi);
          auto xi0 = static_cast<int>(std::floor(xf));
          auto yi0 = static_cast<int>(std::floor(yf));
          auto xi1 = xi0 + 1;
          auto yi1 = yi0 + 1;
          auto px0 = xf - xi0;
          auto py0 = yf - yi0;
          auto px1 = T(1) - px0;
          auto py1 = T(1) - py0;

          auto grad = ograd[oidx++];
          backward_data_2d(igrad, grad, py1, px1, b, c, yi0, xi0, Hi, Wi,
                           istride);
          backward_data_2d(igrad, grad, py1, px0, b, c, yi0, xi1, Hi, Wi,
                           istride);
          backward_data_2d(igrad, grad, py0, px1, b, c, yi1, xi0, Hi, Wi,
                           istride);
          backward_data_2d(igrad, grad, py0, px0, b, c, yi1, xi1, Hi, Wi,
                           istride);
        }
      }
    }
  }
}

template <typename T, warp_by_grid::PADDING_MODE padding_mode,
          bool align_corners = false>
void warp_nearest_backward_data_2d(T *igrad, const T *ograd, const T *grid,
                                   const Shape_t ishape, const Shape_t oshape,
                                   const Shape_t istride, const Shape_t gstride,
                                   const Shape_t ostride) {
  auto B = oshape[0];
  auto C = oshape[1];
  auto Ho = oshape[2];
  auto Wo = oshape[3];
  auto Hi = ishape[2];
  auto Wi = ishape[3];
  auto unnormalize_grid = align_corners ? unnormalize_grid_with<T, true>
                                        : unnormalize_grid_with<T, false>;
  auto get_src_findex_with_pad = [&](const T s, const int S) {
    if (padding_mode == warp_by_grid::PADDING_MODE::zero) {
      return get_src_findex_with_zero_pad(s, S);
    } else if (padding_mode == warp_by_grid::PADDING_MODE::reflect) {
      return align_corners ? get_src_findex_with_reflect_pad<T, true>(s, S)
                           : get_src_findex_with_reflect_pad<T, false>(s, S);
    } else if (padding_mode == warp_by_grid::PADDING_MODE::repeat) {
      return get_src_findex_with_repeat_pad(s, S);
    } else {
      return T(-1);
    }
  };

  auto oidx = 0;
  for (auto b = 0; b < B; ++b) {
    for (auto c = 0; c < C; ++c) {
      for (auto h = 0; h < Ho; ++h) {
        for (auto w = 0; w < Wo; ++w) {
          auto gidx = ndi::nd2flat(Shape_t{b, h, w, 0}, gstride);
          auto xn = grid[gidx + 0];
          auto yn = grid[gidx + 1];
          auto xf0 = unnormalize_grid(xn, Wi);
          auto yf0 = unnormalize_grid(yn, Hi);
          auto xf = get_src_findex_with_pad(xf0, Wi);
          auto yf = get_src_findex_with_pad(yf0, Hi);
          auto xi = static_cast<int>(std::round(xf));
          auto yi = static_cast<int>(std::round(yf));
          auto grad = ograd[oidx++];
          backward_data_2d(igrad, grad, T(1), T(1), b, c, yi, xi, Hi, Wi,
                           istride);
        }
      }
    }
  }
}

template <typename T, warp_by_grid::PADDING_MODE padding_mode,
          bool align_corners = false>
void warp_linear_backward_data_3d(T *igrad, const T *ograd, const T *grid,
                                  const Shape_t ishape, const Shape_t oshape,
                                  const Shape_t istride, const Shape_t gstride,
                                  const Shape_t ostride) {
  auto B = oshape[0];
  auto C = oshape[1];
  auto Do = oshape[2];
  auto Ho = oshape[3];
  auto Wo = oshape[4];
  auto Di = ishape[2];
  auto Hi = ishape[3];
  auto Wi = ishape[4];
  auto unnormalize_grid = align_corners ? unnormalize_grid_with<T, true>
                                        : unnormalize_grid_with<T, false>;
  auto get_src_findex_with_pad = [&](const T s, const int S) {
    if (padding_mode == warp_by_grid::PADDING_MODE::zero) {
      return get_src_findex_with_zero_pad(s, S);
    } else if (padding_mode == warp_by_grid::PADDING_MODE::reflect) {
      return align_corners ? get_src_findex_with_reflect_pad<T, true>(s, S)
                           : get_src_findex_with_reflect_pad<T, false>(s, S);
    } else if (padding_mode == warp_by_grid::PADDING_MODE::repeat) {
      return get_src_findex_with_repeat_pad(s, S);
    } else {
      return T(-1);
    }
  };

  auto oidx = 0;
  for (auto b = 0; b < B; ++b) {
    for (auto c = 0; c < C; ++c) {
      for (auto d = 0; d < Do; ++d) {
        for (auto h = 0; h < Ho; ++h) {
          for (auto w = 0; w < Wo; ++w) {
            auto gidx = ndi::nd2flat(Shape_t{b, d, h, w, 0}, gstride);
            auto xn = grid[gidx + 0];
            auto yn = grid[gidx + 1];
            auto zn = grid[gidx + 2];
            auto xf0 = unnormalize_grid(xn, Wi);
            auto yf0 = unnormalize_grid(yn, Hi);
            auto zf0 = unnormalize_grid(zn, Di);
            auto xf = get_src_findex_with_pad(xf0, Wi);
            auto yf = get_src_findex_with_pad(yf0, Hi);
            auto zf = get_src_findex_with_pad(zf0, Di);
            auto xi0 = static_cast<int>(std::floor(xf));
            auto yi0 = static_cast<int>(std::floor(yf));
            auto zi0 = static_cast<int>(std::floor(zf));
            auto xi1 = xi0 + 1;
            auto yi1 = yi0 + 1;
            auto zi1 = zi0 + 1;
            auto px0 = xf - xi0;
            auto py0 = yf - yi0;
            auto pz0 = zf - zi0;
            auto px1 = T(1) - px0;
            auto py1 = T(1) - py0;
            auto pz1 = T(1) - pz0;

            auto grad = ograd[oidx++];
            backward_data_3d(igrad, grad, pz1, py1, px1, b, c, zi0, yi0, xi0,
                             Di, Hi, Wi, istride);
            backward_data_3d(igrad, grad, pz1, py1, px0, b, c, zi0, yi0, xi1,
                             Di, Hi, Wi, istride);
            backward_data_3d(igrad, grad, pz1, py0, px1, b, c, zi0, yi1, xi0,
                             Di, Hi, Wi, istride);
            backward_data_3d(igrad, grad, pz1, py0, px0, b, c, zi0, yi1, xi1,
                             Di, Hi, Wi, istride);
            backward_data_3d(igrad, grad, pz0, py1, px1, b, c, zi1, yi0, xi0,
                             Di, Hi, Wi, istride);
            backward_data_3d(igrad, grad, pz0, py1, px0, b, c, zi1, yi0, xi1,
                             Di, Hi, Wi, istride);
            backward_data_3d(igrad, grad, pz0, py0, px1, b, c, zi1, yi1, xi0,
                             Di, Hi, Wi, istride);
            backward_data_3d(igrad, grad, pz0, py0, px0, b, c, zi1, yi1, xi1,
                             Di, Hi, Wi, istride);
          }
        }
      }
    }
  }
}

template <typename T, warp_by_grid::PADDING_MODE padding_mode,
          bool align_corners = false>
void warp_nearest_backward_data_3d(T *igrad, const T *ograd, const T *grid,
                                   const Shape_t ishape, const Shape_t oshape,
                                   const Shape_t istride, const Shape_t gstride,
                                   const Shape_t ostride) {
  auto B = oshape[0];
  auto C = oshape[1];
  auto Do = oshape[2];
  auto Ho = oshape[3];
  auto Wo = oshape[4];
  auto Di = ishape[2];
  auto Hi = ishape[3];
  auto Wi = ishape[4];
  auto unnormalize_grid = align_corners ? unnormalize_grid_with<T, true>
                                        : unnormalize_grid_with<T, false>;
  auto get_src_findex_with_pad = [&](const T s, const int S) {
    if (padding_mode == warp_by_grid::PADDING_MODE::zero) {
      return get_src_findex_with_zero_pad(s, S);
    } else if (padding_mode == warp_by_grid::PADDING_MODE::reflect) {
      return align_corners ? get_src_findex_with_reflect_pad<T, true>(s, S)
                           : get_src_findex_with_reflect_pad<T, false>(s, S);
    } else if (padding_mode == warp_by_grid::PADDING_MODE::repeat) {
      return get_src_findex_with_repeat_pad(s, S);
    } else {
      return T(-1);
    }
  };

  auto oidx = 0;
  for (auto b = 0; b < B; ++b) {
    for (auto c = 0; c < C; ++c) {
      for (auto d = 0; d < Do; ++d) {
        for (auto h = 0; h < Ho; ++h) {
          for (auto w = 0; w < Wo; ++w) {
            auto gidx = ndi::nd2flat(Shape_t{b, d, h, w, 0}, gstride);
            auto xn = grid[gidx + 0];
            auto yn = grid[gidx + 1];
            auto zn = grid[gidx + 2];
            auto xf0 = unnormalize_grid(xn, Wi);
            auto yf0 = unnormalize_grid(yn, Hi);
            auto zf0 = unnormalize_grid(zn, Di);
            auto xf = get_src_findex_with_pad(xf0, Wi);
            auto yf = get_src_findex_with_pad(yf0, Hi);
            auto zf = get_src_findex_with_pad(zf0, Di);
            auto xi = static_cast<int>(std::round(xf));
            auto yi = static_cast<int>(std::round(yf));
            auto zi = static_cast<int>(std::round(zf));
            auto grad = ograd[oidx++];
            backward_data_3d(igrad, grad, T(1), T(1), T(1), b, c, zi, yi, xi,
                             Di, Hi, Wi, istride);
          }
        }
      }
    }
  }
}

/*
  Backward implementations wrt the grid.
 */

template <typename T, warp_by_grid::PADDING_MODE padding_mode,
          bool align_corners = false>
void warp_linear_backward_grid_2d(T *ggrad, const T *ograd, const T *input,
                                  const T *grid, const Shape_t ishape,
                                  const Shape_t oshape, const Shape_t istride,
                                  const Shape_t gstride,
                                  const Shape_t ostride) {
  auto B = oshape[0];
  auto C = oshape[1];
  auto Ho = oshape[2];
  auto Wo = oshape[3];
  auto Hi = ishape[2];
  auto Wi = ishape[3];
  auto unnormalize_grid = align_corners ? unnormalize_grid_with<T, true>
                                        : unnormalize_grid_with<T, false>;
  auto get_src_findex_with_pad = [&](const T s, const int S) {
    if (padding_mode == warp_by_grid::PADDING_MODE::zero) {
      return get_src_findex_with_zero_pad(s, S);
    } else if (padding_mode == warp_by_grid::PADDING_MODE::reflect) {
      return align_corners ? get_src_findex_with_reflect_pad<T, true>(s, S)
                           : get_src_findex_with_reflect_pad<T, false>(s, S);
    } else if (padding_mode == warp_by_grid::PADDING_MODE::repeat) {
      return get_src_findex_with_repeat_pad(s, S);
    } else {
      return T(-1);
    }
  };
  auto get_grad_coef_with_pad = [&](const T s, const int S) {
    T coef;
    if (padding_mode == warp_by_grid::PADDING_MODE::zero) {
      coef = get_grad_coef_with_zero_pad(s, S);
    } else if (padding_mode == warp_by_grid::PADDING_MODE::reflect) {
      coef = align_corners ? get_grad_coef_with_reflect_pad<T, true>(s, S)
                           : get_grad_coef_with_reflect_pad<T, false>(s, S);
    } else if (padding_mode == warp_by_grid::PADDING_MODE::repeat) {
      coef = get_grad_coef_with_repeat_pad(s, S);
    } else {
      return T(-1);
    }
    return align_corners ? coef * T(S - 1) / T(2) : coef * T(S) / T(2);
  };

  auto oidx = 0;
  for (auto b = 0; b < B; ++b) {
    for (auto c = 0; c < C; ++c) {
      for (auto h = 0; h < Ho; ++h) {
        for (auto w = 0; w < Wo; ++w) {
          auto gidx = ndi::nd2flat(Shape_t{b, h, w, 0}, gstride);
          auto xn = grid[gidx + 0];
          auto yn = grid[gidx + 1];
          auto xf0 = unnormalize_grid(xn, Wi);
          auto yf0 = unnormalize_grid(yn, Hi);
          auto xf = get_src_findex_with_pad(xf0, Wi);
          auto yf = get_src_findex_with_pad(yf0, Hi);
          auto xi0 = static_cast<int>(std::floor(xf));
          auto yi0 = static_cast<int>(std::floor(yf));
          auto xi1 = xi0 + 1;
          auto yi1 = yi0 + 1;
          auto px0 = xf - xi0;
          auto py0 = yf - yi0;
          auto px1 = T(1) - px0;
          auto py1 = T(1) - py0;

          auto v_y0x0 =
              get_pixel_value_2d(input, b, c, yi0, xi0, Hi, Wi, istride);
          auto v_y0x1 =
              get_pixel_value_2d(input, b, c, yi0, xi1, Hi, Wi, istride);
          auto v_y1x0 =
              get_pixel_value_2d(input, b, c, yi1, xi0, Hi, Wi, istride);
          auto v_y1x1 =
              get_pixel_value_2d(input, b, c, yi1, xi1, Hi, Wi, istride);
          auto grad = ograd[oidx++];

          // d_grid = d_output * local_grad{output/pad(x)} *
          // local_grad{pad(x)/x} * unnormalized_coef
          auto grad_x =
              grad * ((v_y0x1 - v_y0x0) * py1 + (v_y1x1 - v_y1x0) * py0);
          auto grad_y =
              grad * ((v_y1x0 - v_y0x0) * px1 + (v_y1x1 - v_y0x1) * px0);
          auto coef_x = get_grad_coef_with_pad(xf0, Wi);
          auto coef_y = get_grad_coef_with_pad(yf0, Hi);
          ggrad[gidx + 0] += grad_x * coef_x;
          ggrad[gidx + 1] += grad_y * coef_y;
        }
      }
    }
  }
}

template <typename T, warp_by_grid::PADDING_MODE padding_mode,
          bool align_corners = false>
void warp_linear_backward_grid_3d(T *ggrad, const T *ograd, const T *input,
                                  const T *grid, const Shape_t ishape,
                                  const Shape_t oshape, const Shape_t istride,
                                  const Shape_t gstride,
                                  const Shape_t ostride) {
  auto B = oshape[0];
  auto C = oshape[1];
  auto Do = oshape[2];
  auto Ho = oshape[3];
  auto Wo = oshape[4];
  auto Di = ishape[2];
  auto Hi = ishape[3];
  auto Wi = ishape[4];
  auto unnormalize_grid = align_corners ? unnormalize_grid_with<T, true>
                                        : unnormalize_grid_with<T, false>;
  auto get_src_findex_with_pad = [&](const T s, const int S) {
    if (padding_mode == warp_by_grid::PADDING_MODE::zero) {
      return get_src_findex_with_zero_pad(s, S);
    } else if (padding_mode == warp_by_grid::PADDING_MODE::reflect) {
      return align_corners ? get_src_findex_with_reflect_pad<T, true>(s, S)
                           : get_src_findex_with_reflect_pad<T, false>(s, S);
    } else if (padding_mode == warp_by_grid::PADDING_MODE::repeat) {
      return get_src_findex_with_repeat_pad(s, S);
    } else {
      return T(-1);
    }
  };
  auto get_grad_coef_with_pad = [&](const T s, const int S) {
    T coef;
    if (padding_mode == warp_by_grid::PADDING_MODE::zero) {
      coef = get_grad_coef_with_zero_pad(s, S);
    } else if (padding_mode == warp_by_grid::PADDING_MODE::reflect) {
      coef = align_corners ? get_grad_coef_with_reflect_pad<T, true>(s, S)
                           : get_grad_coef_with_reflect_pad<T, false>(s, S);
    } else if (padding_mode == warp_by_grid::PADDING_MODE::repeat) {
      coef = get_grad_coef_with_repeat_pad(s, S);
    } else {
      return T(-1);
    }
    return align_corners ? coef * T(S - 1) / T(2) : coef * T(S) / T(2);
  };

  auto oidx = 0;
  for (auto b = 0; b < B; ++b) {
    for (auto c = 0; c < C; ++c) {
      for (auto d = 0; d < Do; ++d) {
        for (auto h = 0; h < Ho; ++h) {
          for (auto w = 0; w < Wo; ++w) {
            auto gidx = ndi::nd2flat(Shape_t{b, d, h, w, 0}, gstride);
            auto xn = grid[gidx + 0];
            auto yn = grid[gidx + 1];
            auto zn = grid[gidx + 2];
            auto xf0 = unnormalize_grid(xn, Wi);
            auto yf0 = unnormalize_grid(yn, Hi);
            auto zf0 = unnormalize_grid(zn, Di);
            auto xf = get_src_findex_with_pad(xf0, Wi);
            auto yf = get_src_findex_with_pad(yf0, Hi);
            auto zf = get_src_findex_with_pad(zf0, Di);
            auto xi0 = static_cast<int>(std::floor(xf));
            auto yi0 = static_cast<int>(std::floor(yf));
            auto zi0 = static_cast<int>(std::floor(zf));
            auto xi1 = xi0 + 1;
            auto yi1 = yi0 + 1;
            auto zi1 = zi0 + 1;
            auto px0 = xf - xi0;
            auto py0 = yf - yi0;
            auto pz0 = zf - zi0;
            auto px1 = T(1) - px0;
            auto py1 = T(1) - py0;
            auto pz1 = T(1) - pz0;

            auto v_z0y0x0 = get_pixel_value_3d(input, b, c, zi0, yi0, xi0, Di,
                                               Hi, Wi, istride);
            auto v_z0y0x1 = get_pixel_value_3d(input, b, c, zi0, yi0, xi1, Di,
                                               Hi, Wi, istride);
            auto v_z0y1x0 = get_pixel_value_3d(input, b, c, zi0, yi1, xi0, Di,
                                               Hi, Wi, istride);
            auto v_z0y1x1 = get_pixel_value_3d(input, b, c, zi0, yi1, xi1, Di,
                                               Hi, Wi, istride);
            auto v_z1y0x0 = get_pixel_value_3d(input, b, c, zi1, yi0, xi0, Di,
                                               Hi, Wi, istride);
            auto v_z1y0x1 = get_pixel_value_3d(input, b, c, zi1, yi0, xi1, Di,
                                               Hi, Wi, istride);
            auto v_z1y1x0 = get_pixel_value_3d(input, b, c, zi1, yi1, xi0, Di,
                                               Hi, Wi, istride);
            auto v_z1y1x1 = get_pixel_value_3d(input, b, c, zi1, yi1, xi1, Di,
                                               Hi, Wi, istride);

            auto grad = ograd[oidx++];

            // d_grid = d_output * local_grad{output/pad(x)} *
            // local_grad{pad(x)/x} * unnormalized_coef
            auto grad_x = grad * ((v_z0y0x1 - v_z0y0x0) * pz1 * py1 +
                                  (v_z0y1x1 - v_z0y1x0) * pz1 * py0 +
                                  (v_z1y0x1 - v_z1y0x0) * pz0 * py1 +
                                  (v_z1y1x1 - v_z1y1x0) * pz0 * py0);
            auto grad_y = grad * ((v_z0y1x0 - v_z0y0x0) * pz1 * px1 +
                                  (v_z0y1x1 - v_z0y0x1) * pz1 * px0 +
                                  (v_z1y1x0 - v_z1y0x0) * pz0 * px1 +
                                  (v_z1y1x1 - v_z1y0x1) * pz0 * px0);
            auto grad_z = grad * ((v_z1y0x0 - v_z0y0x0) * py1 * px1 +
                                  (v_z1y0x1 - v_z0y0x1) * py1 * px0 +
                                  (v_z1y1x0 - v_z0y1x0) * py0 * px1 +
                                  (v_z1y1x1 - v_z0y1x1) * py0 * px0);
            auto coef_x = get_grad_coef_with_pad(xf0, Wi);
            auto coef_y = get_grad_coef_with_pad(yf0, Hi);
            auto coef_z = get_grad_coef_with_pad(zf0, Di);
            ggrad[gidx + 0] += grad_x * coef_x;
            ggrad[gidx + 1] += grad_y * coef_y;
            ggrad[gidx + 2] += grad_z * coef_z;
          }
        }
      }
    }
  }
}

template <typename T>
void WarpByGrid<T>::setup_impl(const Variables &inputs,
                               const Variables &outputs) {
  auto ishape = inputs[0]->shape();
  auto gshape = inputs[1]->shape();
  auto ndims = gshape.size();
  auto B = ishape[0];

  NBLA_CHECK(mode_ == "linear" || mode_ == "nearest",
             error_code::not_implemented, "%s is not implemented.",
             mode_.c_str());
  NBLA_CHECK(ishape[0] == gshape[0], error_code::value,
             "Input and grid batch size differs (%d != %d).", ishape[0],
             gshape[0]);
  NBLA_CHECK(gshape[ndims - 1] == 2 || gshape[ndims - 1] == 3,
             error_code::not_implemented,
             "Last dimension of the grid must be in {2, 3} ({} not in {2, 3}).",
             gshape[ndims - 1]);
  if (padding_mode_ == "zero") {
    padding_mode_t_ = warp_by_grid::PADDING_MODE::zero;
  } else if (padding_mode_ == "repeat") {
    padding_mode_t_ = warp_by_grid::PADDING_MODE::repeat;
  } else if (padding_mode_ == "reflect") {
    padding_mode_t_ = warp_by_grid::PADDING_MODE::reflect;
  } else {
    NBLA_ERROR(error_code::not_implemented, "%s is not implemented.",
               padding_mode_.c_str());
  }

  Shape_t oshape;
  if (channel_last_) {
    auto C = ishape[ndims - 1];
    if (ndims == 4) {
      oshape = Shape_t{B, gshape[1], gshape[2], C};
    } else if (ndims == 5) {
      oshape = Shape_t{B, gshape[1], gshape[2], gshape[3], C};
    }
  } else {
    auto C = ishape[1];
    if (ndims == 4) {
      oshape = Shape_t{B, C, gshape[1], gshape[2]};
    } else if (ndims == 5) {
      oshape = Shape_t{B, C, gshape[1], gshape[2], gshape[3]};
    }
  }
  outputs[0]->reshape(oshape, true);
}

template <typename T>
void WarpByGrid<T>::forward_impl(const Variables &inputs,
                                 const Variables &outputs) {
  NBLA_CHECK(!channel_last_, error_code::not_implemented,
             "WarpByGrid w/ the channel_last is not supported.");
  using PADDING_MODE = warp_by_grid::PADDING_MODE;
  auto zero = PADDING_MODE::zero;
  auto repeat = PADDING_MODE::repeat;
  auto reflect = PADDING_MODE::reflect;

  auto ishape = inputs[0]->shape();
  auto gshape = inputs[1]->shape();
  auto oshape = outputs[0]->shape();

  auto istride = inputs[0]->strides();
  auto gstride = inputs[1]->strides();
  auto ostride = outputs[0]->strides();

  auto ndims = gshape.size();
  auto input = inputs[0]->get_data_pointer<T>(ctx_);
  auto grid = inputs[1]->get_data_pointer<T>(ctx_);
  auto output = outputs[0]->cast_data_and_get_pointer<T>(ctx_);

  if (mode_ == "linear") {
    if (ndims == 4) {
      if (padding_mode_t_ == zero && align_corners_) {
        auto kernel = warp_linear_forward_2d<T, PADDING_MODE::zero, true>;
        kernel(output, input, grid, ishape, oshape, istride, gstride, ostride);
      } else if (padding_mode_t_ == repeat && align_corners_) {
        auto kernel = warp_linear_forward_2d<T, PADDING_MODE::repeat, true>;
        kernel(output, input, grid, ishape, oshape, istride, gstride, ostride);
      } else if (padding_mode_t_ == reflect && align_corners_) {
        auto kernel = warp_linear_forward_2d<T, PADDING_MODE::reflect, true>;
        kernel(output, input, grid, ishape, oshape, istride, gstride, ostride);
      } else if (padding_mode_t_ == zero && !align_corners_) {
        auto kernel = warp_linear_forward_2d<T, PADDING_MODE::zero, false>;
        kernel(output, input, grid, ishape, oshape, istride, gstride, ostride);
      } else if (padding_mode_t_ == repeat && !align_corners_) {
        auto kernel = warp_linear_forward_2d<T, PADDING_MODE::repeat, false>;
        kernel(output, input, grid, ishape, oshape, istride, gstride, ostride);
      } else if (padding_mode_t_ == reflect && !align_corners_) {
        auto kernel = warp_linear_forward_2d<T, PADDING_MODE::reflect, false>;
        kernel(output, input, grid, ishape, oshape, istride, gstride, ostride);
      }
    } else if (ndims == 5) {
      if (padding_mode_t_ == zero && align_corners_) {
        auto kernel = warp_linear_forward_3d<T, PADDING_MODE::zero, true>;
        kernel(output, input, grid, ishape, oshape, istride, gstride, ostride);
      } else if (padding_mode_t_ == repeat && align_corners_) {
        auto kernel = warp_linear_forward_3d<T, PADDING_MODE::repeat, true>;
        kernel(output, input, grid, ishape, oshape, istride, gstride, ostride);
      } else if (padding_mode_t_ == reflect && align_corners_) {
        auto kernel = warp_linear_forward_3d<T, PADDING_MODE::reflect, true>;
        kernel(output, input, grid, ishape, oshape, istride, gstride, ostride);
      } else if (padding_mode_t_ == zero && !align_corners_) {
        auto kernel = warp_linear_forward_3d<T, PADDING_MODE::zero, false>;
        kernel(output, input, grid, ishape, oshape, istride, gstride, ostride);
      } else if (padding_mode_t_ == repeat && !align_corners_) {
        auto kernel = warp_linear_forward_3d<T, PADDING_MODE::repeat, false>;
        kernel(output, input, grid, ishape, oshape, istride, gstride, ostride);
      } else if (padding_mode_t_ == reflect && !align_corners_) {
        auto kernel = warp_linear_forward_3d<T, PADDING_MODE::reflect, false>;
        kernel(output, input, grid, ishape, oshape, istride, gstride, ostride);
      }
    }
  } else if (mode_ == "nearest") {
    if (ndims == 4) {
      if (padding_mode_t_ == zero && align_corners_) {
        auto kernel = warp_nearest_forward_2d<T, PADDING_MODE::zero, true>;
        kernel(output, input, grid, ishape, oshape, istride, gstride, ostride);
      } else if (padding_mode_t_ == repeat && align_corners_) {
        auto kernel = warp_nearest_forward_2d<T, PADDING_MODE::repeat, true>;
        kernel(output, input, grid, ishape, oshape, istride, gstride, ostride);
      } else if (padding_mode_t_ == reflect && align_corners_) {
        auto kernel = warp_nearest_forward_2d<T, PADDING_MODE::reflect, true>;
        kernel(output, input, grid, ishape, oshape, istride, gstride, ostride);
      } else if (padding_mode_t_ == zero && !align_corners_) {
        auto kernel = warp_nearest_forward_2d<T, PADDING_MODE::zero, false>;
        kernel(output, input, grid, ishape, oshape, istride, gstride, ostride);
      } else if (padding_mode_t_ == repeat && !align_corners_) {
        auto kernel = warp_nearest_forward_2d<T, PADDING_MODE::repeat, false>;
        kernel(output, input, grid, ishape, oshape, istride, gstride, ostride);
      } else if (padding_mode_t_ == reflect && !align_corners_) {
        auto kernel = warp_nearest_forward_2d<T, PADDING_MODE::reflect, false>;
        kernel(output, input, grid, ishape, oshape, istride, gstride, ostride);
      }
    } else if (ndims == 5) {
      if (padding_mode_t_ == zero && align_corners_) {
        auto kernel = warp_nearest_forward_3d<T, PADDING_MODE::zero, true>;
        kernel(output, input, grid, ishape, oshape, istride, gstride, ostride);
      } else if (padding_mode_t_ == repeat && align_corners_) {
        auto kernel = warp_nearest_forward_3d<T, PADDING_MODE::repeat, true>;
        kernel(output, input, grid, ishape, oshape, istride, gstride, ostride);
      } else if (padding_mode_t_ == reflect && align_corners_) {
        auto kernel = warp_nearest_forward_3d<T, PADDING_MODE::reflect, true>;
        kernel(output, input, grid, ishape, oshape, istride, gstride, ostride);
      } else if (padding_mode_t_ == zero && !align_corners_) {
        auto kernel = warp_nearest_forward_3d<T, PADDING_MODE::zero, false>;
        kernel(output, input, grid, ishape, oshape, istride, gstride, ostride);
      } else if (padding_mode_t_ == repeat && !align_corners_) {
        auto kernel = warp_nearest_forward_3d<T, PADDING_MODE::repeat, false>;
        kernel(output, input, grid, ishape, oshape, istride, gstride, ostride);
      } else if (padding_mode_t_ == reflect && !align_corners_) {
        auto kernel = warp_nearest_forward_3d<T, PADDING_MODE::reflect, false>;
        kernel(output, input, grid, ishape, oshape, istride, gstride, ostride);
      }
    }
  }
}

template <typename T>
void WarpByGrid<T>::backward_impl(const Variables &inputs,
                                  const Variables &outputs,
                                  const vector<bool> &propagate_down,
                                  const vector<bool> &accum) {
  NBLA_CHECK(!channel_last_, error_code::not_implemented,
             "WarpByGrid w/ the channel_last is not supported.");
  if (!(propagate_down[0] || propagate_down[1])) {
    return;
  }
  using PADDING_MODE = warp_by_grid::PADDING_MODE;
  auto zero = PADDING_MODE::zero;
  auto repeat = PADDING_MODE::repeat;
  auto reflect = PADDING_MODE::reflect;

  auto ishape = inputs[0]->shape();
  auto gshape = inputs[1]->shape();
  auto oshape = outputs[0]->shape();

  auto istride = inputs[0]->strides();
  auto gstride = inputs[1]->strides();
  auto ostride = outputs[0]->strides();

  auto ndims = gshape.size();
  auto input = inputs[0]->get_data_pointer<T>(ctx_);
  auto igrad = inputs[0]->cast_grad_and_get_pointer<T>(ctx_, false);
  auto ggrad = inputs[1]->cast_grad_and_get_pointer<T>(ctx_, false);
  auto grid = inputs[1]->get_data_pointer<T>(ctx_);
  auto ograd = outputs[0]->get_grad_pointer<T>(ctx_);

  // w.r.t. data
  if (propagate_down[0]) {
    if (ndims == 4) {
      if (mode_ == "linear") {
        if (padding_mode_t_ == zero && align_corners_) {
          auto kernel =
              warp_linear_backward_data_2d<T, PADDING_MODE::zero, true>;
          kernel(igrad, ograd, grid, ishape, oshape, istride, gstride, ostride);
        } else if (padding_mode_t_ == repeat && align_corners_) {
          auto kernel =
              warp_linear_backward_data_2d<T, PADDING_MODE::repeat, true>;
          kernel(igrad, ograd, grid, ishape, oshape, istride, gstride, ostride);
        } else if (padding_mode_t_ == reflect && align_corners_) {
          auto kernel =
              warp_linear_backward_data_2d<T, PADDING_MODE::reflect, true>;
          kernel(igrad, ograd, grid, ishape, oshape, istride, gstride, ostride);
        } else if (padding_mode_t_ == zero && !align_corners_) {
          auto kernel =
              warp_linear_backward_data_2d<T, PADDING_MODE::zero, false>;
          kernel(igrad, ograd, grid, ishape, oshape, istride, gstride, ostride);
        } else if (padding_mode_t_ == repeat && !align_corners_) {
          auto kernel =
              warp_linear_backward_data_2d<T, PADDING_MODE::repeat, false>;
          kernel(igrad, ograd, grid, ishape, oshape, istride, gstride, ostride);
        } else if (padding_mode_t_ == reflect && !align_corners_) {
          auto kernel =
              warp_linear_backward_data_2d<T, PADDING_MODE::reflect, false>;
          kernel(igrad, ograd, grid, ishape, oshape, istride, gstride, ostride);
        }
      } else if (mode_ == "nearest") {
        if (padding_mode_t_ == zero && align_corners_) {
          auto kernel =
              warp_nearest_backward_data_2d<T, PADDING_MODE::zero, true>;
          kernel(igrad, ograd, grid, ishape, oshape, istride, gstride, ostride);
        } else if (padding_mode_t_ == repeat && align_corners_) {
          auto kernel =
              warp_nearest_backward_data_2d<T, PADDING_MODE::repeat, true>;
          kernel(igrad, ograd, grid, ishape, oshape, istride, gstride, ostride);
        } else if (padding_mode_t_ == reflect && align_corners_) {
          auto kernel =
              warp_nearest_backward_data_2d<T, PADDING_MODE::reflect, true>;
          kernel(igrad, ograd, grid, ishape, oshape, istride, gstride, ostride);
        } else if (padding_mode_t_ == zero && !align_corners_) {
          auto kernel =
              warp_nearest_backward_data_2d<T, PADDING_MODE::zero, false>;
          kernel(igrad, ograd, grid, ishape, oshape, istride, gstride, ostride);
        } else if (padding_mode_t_ == repeat && !align_corners_) {
          auto kernel =
              warp_nearest_backward_data_2d<T, PADDING_MODE::repeat, false>;
          kernel(igrad, ograd, grid, ishape, oshape, istride, gstride, ostride);
        } else if (padding_mode_t_ == reflect && !align_corners_) {
          auto kernel =
              warp_nearest_backward_data_2d<T, PADDING_MODE::reflect, false>;
          kernel(igrad, ograd, grid, ishape, oshape, istride, gstride, ostride);
        }
      }
    } else if (ndims == 5) {
      if (mode_ == "linear") {
        if (padding_mode_t_ == zero && align_corners_) {
          auto kernel =
              warp_linear_backward_data_3d<T, PADDING_MODE::zero, true>;
          kernel(igrad, ograd, grid, ishape, oshape, istride, gstride, ostride);
        } else if (padding_mode_t_ == repeat && align_corners_) {
          auto kernel =
              warp_linear_backward_data_3d<T, PADDING_MODE::repeat, true>;
          kernel(igrad, ograd, grid, ishape, oshape, istride, gstride, ostride);
        } else if (padding_mode_t_ == reflect && align_corners_) {
          auto kernel =
              warp_linear_backward_data_3d<T, PADDING_MODE::reflect, true>;
          kernel(igrad, ograd, grid, ishape, oshape, istride, gstride, ostride);
        } else if (padding_mode_t_ == zero && !align_corners_) {
          auto kernel =
              warp_linear_backward_data_3d<T, PADDING_MODE::zero, false>;
          kernel(igrad, ograd, grid, ishape, oshape, istride, gstride, ostride);
        } else if (padding_mode_t_ == repeat && !align_corners_) {
          auto kernel =
              warp_linear_backward_data_3d<T, PADDING_MODE::repeat, false>;
          kernel(igrad, ograd, grid, ishape, oshape, istride, gstride, ostride);
        } else if (padding_mode_t_ == reflect && !align_corners_) {
          auto kernel =
              warp_linear_backward_data_3d<T, PADDING_MODE::reflect, false>;
          kernel(igrad, ograd, grid, ishape, oshape, istride, gstride, ostride);
        }
      } else if (mode_ == "nearest") {
        if (padding_mode_t_ == zero && align_corners_) {
          auto kernel =
              warp_nearest_backward_data_3d<T, PADDING_MODE::zero, true>;
          kernel(igrad, ograd, grid, ishape, oshape, istride, gstride, ostride);
        } else if (padding_mode_t_ == repeat && align_corners_) {
          auto kernel =
              warp_nearest_backward_data_3d<T, PADDING_MODE::repeat, true>;
          kernel(igrad, ograd, grid, ishape, oshape, istride, gstride, ostride);
        } else if (padding_mode_t_ == reflect && align_corners_) {
          auto kernel =
              warp_nearest_backward_data_3d<T, PADDING_MODE::reflect, true>;
          kernel(igrad, ograd, grid, ishape, oshape, istride, gstride, ostride);
        } else if (padding_mode_t_ == zero && !align_corners_) {
          auto kernel =
              warp_nearest_backward_data_3d<T, PADDING_MODE::zero, false>;
          kernel(igrad, ograd, grid, ishape, oshape, istride, gstride, ostride);
        } else if (padding_mode_t_ == repeat && !align_corners_) {
          auto kernel =
              warp_nearest_backward_data_3d<T, PADDING_MODE::repeat, false>;
          kernel(igrad, ograd, grid, ishape, oshape, istride, gstride, ostride);
        } else if (padding_mode_t_ == reflect && !align_corners_) {
          auto kernel =
              warp_nearest_backward_data_3d<T, PADDING_MODE::reflect, false>;
          kernel(igrad, ograd, grid, ishape, oshape, istride, gstride, ostride);
        }
      }
    }
  }

  // w.r.t. grid
  if (propagate_down[1]) {
    if (ndims == 4) {
      if (mode_ == "linear") {
        if (padding_mode_t_ == zero && align_corners_) {
          auto kernel =
              warp_linear_backward_grid_2d<T, PADDING_MODE::zero, true>;
          kernel(ggrad, ograd, input, grid, ishape, oshape, istride, gstride,
                 ostride);
        } else if (padding_mode_t_ == repeat && align_corners_) {
          auto kernel =
              warp_linear_backward_grid_2d<T, PADDING_MODE::repeat, true>;
          kernel(ggrad, ograd, input, grid, ishape, oshape, istride, gstride,
                 ostride);
        } else if (padding_mode_t_ == reflect && align_corners_) {
          auto kernel =
              warp_linear_backward_grid_2d<T, PADDING_MODE::reflect, true>;
          kernel(ggrad, ograd, input, grid, ishape, oshape, istride, gstride,
                 ostride);
        } else if (padding_mode_t_ == zero && !align_corners_) {
          auto kernel =
              warp_linear_backward_grid_2d<T, PADDING_MODE::zero, false>;
          kernel(ggrad, ograd, input, grid, ishape, oshape, istride, gstride,
                 ostride);
        } else if (padding_mode_t_ == repeat && !align_corners_) {
          auto kernel =
              warp_linear_backward_grid_2d<T, PADDING_MODE::repeat, false>;
          kernel(ggrad, ograd, input, grid, ishape, oshape, istride, gstride,
                 ostride);
        } else if (padding_mode_t_ == reflect && !align_corners_) {
          auto kernel =
              warp_linear_backward_grid_2d<T, PADDING_MODE::reflect, false>;
          kernel(ggrad, ograd, input, grid, ishape, oshape, istride, gstride,
                 ostride);
        }
      } else if (mode_ == "nearest") {
        NBLA_ERROR(
            error_code::not_implemented,
            "Backward wrt the grid is not supported in the nearest mode. "
            "Use the `linear` mode.");
      }
    } else if (ndims == 5) {
      if (mode_ == "linear") {
        if (padding_mode_t_ == zero && align_corners_) {
          auto kernel =
              warp_linear_backward_grid_3d<T, PADDING_MODE::zero, true>;
          kernel(ggrad, ograd, input, grid, ishape, oshape, istride, gstride,
                 ostride);
        } else if (padding_mode_t_ == repeat && align_corners_) {
          auto kernel =
              warp_linear_backward_grid_3d<T, PADDING_MODE::repeat, true>;
          kernel(ggrad, ograd, input, grid, ishape, oshape, istride, gstride,
                 ostride);
        } else if (padding_mode_t_ == reflect && align_corners_) {
          auto kernel =
              warp_linear_backward_grid_3d<T, PADDING_MODE::reflect, true>;
          kernel(ggrad, ograd, input, grid, ishape, oshape, istride, gstride,
                 ostride);
        } else if (padding_mode_t_ == zero && !align_corners_) {
          auto kernel =
              warp_linear_backward_grid_3d<T, PADDING_MODE::zero, false>;
          kernel(ggrad, ograd, input, grid, ishape, oshape, istride, gstride,
                 ostride);
        } else if (padding_mode_t_ == repeat && !align_corners_) {
          auto kernel =
              warp_linear_backward_grid_3d<T, PADDING_MODE::repeat, false>;
          kernel(ggrad, ograd, input, grid, ishape, oshape, istride, gstride,
                 ostride);
        } else if (padding_mode_t_ == reflect && !align_corners_) {
          auto kernel =
              warp_linear_backward_grid_3d<T, PADDING_MODE::reflect, false>;
          kernel(ggrad, ograd, input, grid, ishape, oshape, istride, gstride,
                 ostride);
        }
      } else if (mode_ == "nearest") {
        NBLA_ERROR(
            error_code::not_implemented,
            "Backward wrt the grid is not supported in the nearest mode. "
            "Use the `linear` mode.");
      }
    }
  }
}
}
