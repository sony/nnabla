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
                              bool);

template <typename T>
void Interpolate<T>::setup_impl(const Variables &inputs,
                                const Variables &outputs) {
  NBLA_CHECK(output_size_.size() == 2, error_code::not_implemented,
             "Only 2-dimensional interpolation is implemented.");
  NBLA_CHECK(mode_ == "linear", error_code::not_implemented,
             "Only 'linear' interpolation is implemented.");

  Shape_t out_shape(inputs[0]->shape());
  for (int d = 0; d < output_size_.size(); d++) {
    out_shape[d + out_shape.size() - output_size_.size()] = output_size_[d];
  }
  outputs[0]->reshape(out_shape, true);
}

template <typename T>
T get_interpolate_scale_from_dest(int src, int dst, bool align_corners) {
  if (dst == 1)
    return 0;
  return align_corners ? ((T)(src - 1) / (dst - 1)) : ((T)(src) / dst);
}

template <typename T>
T get_interpolate_source_index(T scale, int dst, bool align_corners) {
  return align_corners ? (scale * dst)
                       : (std::max((T)0, scale * (dst + (T)0.5) - (T)0.5));
}

template <typename T>
void bilinear_interpolate_2d(const T *in, T *out, int outer_dim, int iw, int ih,
                             int ow, int oh, bool align_corners) {
  // 2D implementation
  const T sx = get_interpolate_scale_from_dest<T>(iw, ow, align_corners);
  const T sy = get_interpolate_scale_from_dest<T>(ih, oh, align_corners);
  const int inner_dim = iw * ih;
  const int inner_dim_o = ow * oh;
  for (int o = 0; o < outer_dim; o++) {
    for (int oy = 0; oy < oh; oy++) {
      const T fy = get_interpolate_source_index(sy, oy, align_corners);
      const int y = fy;
      const int yp1 = std::min(y + 1, ih - 1);
      const T ly1 = fy - y;
      const T ly0 = (T)1 - ly1;

      for (int ox = 0; ox < ow; ox++) {
        const T fx = get_interpolate_source_index(sx, ox, align_corners);
        const int x = fx;
        const int xp1 = std::min(x + 1, iw - 1);
        const T lx1 = fx - x;
        const T lx0 = (T)1 - lx1;
#define _I(o, y, x) ((o)*inner_dim + (y)*iw + (x))
        const T val0 = ly0 * (lx0 * in[_I(o, y, x)] + lx1 * in[_I(o, y, xp1)]);
        const T val1 =
            ly1 * (lx0 * in[_I(o, yp1, x)] + lx1 * in[_I(o, yp1, xp1)]);
#undef _I
        out[o * inner_dim_o + oy * ow + ox] = val0 + val1;
      }
    }
  }
}

template <typename T>
void bilinear_interpolate_2d_backward(T *gin, const T *gout, int outer_dim,
                                      int iw, int ih, int ow, int oh,
                                      bool align_corners) {
  // 2D implementation
  const T sx = get_interpolate_scale_from_dest<T>(iw, ow, align_corners);
  const T sy = get_interpolate_scale_from_dest<T>(ih, oh, align_corners);
  const int inner_dim = iw * ih;
  const int inner_dim_o = ow * oh;
  for (int o = 0; o < outer_dim; o++) {
    for (int oy = 0; oy < oh; oy++) {
      const T fy = get_interpolate_source_index(sy, oy, align_corners);
      const int y = fy;
      const int yp1 = (y < ih - 1) ? (y + 1) : y;
      const T ly1 = fy - y;
      const T ly0 = (T)1 - ly1;

      for (int ox = 0; ox < ow; ox++) {
        const T fx = get_interpolate_source_index(sx, ox, align_corners);
        const int x = fx;
        const int xp1 = (x < iw - 1) ? (x + 1) : x;
        const T lx1 = fx - x;
        const T lx0 = (T)1 - lx1;
        const T g = gout[o * inner_dim_o + oy * ow + ox];
#define _I(o, y, x) ((o)*inner_dim + (y)*iw + (x))
        gin[_I(o, y, x)] += ly0 * lx0 * g;
        gin[_I(o, y, xp1)] += ly0 * lx1 * g;
        gin[_I(o, yp1, x)] += ly1 * lx0 * g;
        gin[_I(o, yp1, xp1)] += ly1 * lx1 * g;
#undef _I
      }
    }
  }
}

template <typename T>
void Interpolate<T>::forward_impl(const Variables &inputs,
                                  const Variables &outputs) {
  // Inputs
  const T *x = inputs[0]->get_data_pointer<T>(this->ctx_);

  // Outputs
  T *y = outputs[0]->cast_data_and_get_pointer<T>(this->ctx_, true);

  // 2D bilinear
  const int ndim = inputs[0]->ndim();
  const int iw = inputs[0]->shape()[ndim - 1];
  const int ih = inputs[0]->shape()[ndim - 2];
  const int ow = outputs[0]->shape()[ndim - 1];
  const int oh = outputs[0]->shape()[ndim - 2];
  const int outer_dim = inputs[0]->size() / (iw * ih);
  bilinear_interpolate_2d(x, y, outer_dim, iw, ih, ow, oh, align_corners_);
}

template <typename T>
void Interpolate<T>::backward_impl(const Variables &inputs,
                                   const Variables &outputs,
                                   const vector<bool> &propagate_down,
                                   const vector<bool> &accum) {
  if (!(propagate_down[0])) {
    return;
  }

  // Gradient of outputs
  const T *g_y = outputs[0]->get_grad_pointer<T>(this->ctx_);

  // Gradient of inputs
  // NOTE: Not using write_only flag, because the following accumulates
  // gradient.
  T *g_x = inputs[0]->cast_grad_and_get_pointer<T>(this->ctx_);

  // 2D bilinear
  const int ndim = inputs[0]->ndim();
  const int iw = inputs[0]->shape()[ndim - 1];
  const int ih = inputs[0]->shape()[ndim - 2];
  const int ow = outputs[0]->shape()[ndim - 1];
  const int oh = outputs[0]->shape()[ndim - 2];
  const int outer_dim = inputs[0]->size() / (iw * ih);
  bilinear_interpolate_2d_backward(g_x, g_y, outer_dim, iw, ih, ow, oh,
                                   align_corners_);
}
}
