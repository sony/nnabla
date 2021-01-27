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

/** ImageAugmentation
 */
#include <nbla/array.hpp>
#include <nbla/function/image_augmentation.hpp>
#include <nbla/random_manager.hpp>
#include <nbla/variable.hpp>

#include <algorithm>
#include <cstring>

namespace nbla {

NBLA_REGISTER_FUNCTION_SOURCE(ImageAugmentation, const vector<int> &,
                              const vector<int> &, float, float, float, float,
                              float, bool, bool, float, bool, float, float,
                              bool, float, int);

template <typename T>
void ImageAugmentation<T>::setup_impl(const Variables &inputs,
                                      const Variables &outputs) {
  NBLA_CHECK(shape_.size() >= 2, error_code::value,
             "Shape must be larger than 2D (height and width).");
  NBLA_CHECK(pad_.size() == 2, error_code::value,
             "Pad must be 2D (height and width).");
  NBLA_CHECK(inputs[0]->shape().size() >= 2, error_code::value,
             "Input shape must be larger than 2D (height and width).");

  std::random_device rdev_;
  rgen_ = std::mt19937((seed_ == -1 ? rdev_() : seed_));

  Shape_t shape_y = inputs[0]->shape();
  int dim_offset = shape_y.size() - shape_.size();
  for (Shape_t::size_type i = 0; i < shape_.size(); i++) {
    shape_y[i + dim_offset] = shape_[i];
  }

  outputs[0]->reshape(shape_y, true);
}

template <typename T>
void ImageAugmentation<T>::forward_impl(const Variables &inputs,
                                        const Variables &outputs) {
  /*
  std::cout <<
    "min_scale=" << min_scale_ <<
    ", max_scale=" << max_scale_ <<
    ", angle=" << angle_ <<
    ", flip_lr=" << flip_lr_ <<
    ", flip_ud=" << flip_ud_ <<
    ", brightness=" << brightness_ <<
    ", brightness_each=" << brightness_each_ <<
    ", contrast=" << contrast_ <<
    ", contrast_each=" << contrast_each_ <<
    "\n";
  //*/

  Shape_t shape_in = inputs[0]->shape();
  const int w_in = shape_in[shape_in.size() - 1];
  const int h_in = shape_in[shape_in.size() - 2];
  const int w_in_pad = w_in + pad_[1] * 2;
  const int h_in_pad = h_in + pad_[0] * 2;
  const int num_ch = shape_in.size() >= 3 ? shape_in[shape_in.size() - 3] : 1;
  const int num_image = inputs[0]->size() / (w_in * h_in * num_ch);
  // std::cout << "shape_in : w=" << w_in << ", h=" << h_in << ", ch=" << num_ch
  // << ", num=" << num_image << "\n";

  Shape_t shape_out = outputs[0]->shape();
  const int w_out = shape_out[shape_out.size() - 1];
  const int h_out = shape_out[shape_out.size() - 2];
  // std::cout << "shape_out : w=" << w_out << ", h=" << h_out << "\n";

  const T *x = inputs[0]->get_data_pointer<T>(this->ctx_);
  T *y = outputs[0]->cast_data_and_get_pointer<T>(this->ctx_, true);

  const int ch_size_in = h_in * w_in;
  const int ch_size_out = h_out * w_out;

  std::mt19937 &rgen =
      seed_ == -1 ? SingletonManager::get<RandomManager>()->get_rand_generator()
                  : rgen_;
  std::normal_distribution<> norm(0.0, 1.0);

  const float w_out_half = w_out * 0.5f;
  const float h_out_half = h_out * 0.5f;
  const float i_w_out_half = 1.0f / w_out_half;
  const float i_h_out_half = 1.0f / h_out_half;

  T *channel_brightness_buf = new T[num_ch * num_image];
  T *channel_contrast_buf = new T[num_ch * num_image];
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
  for (int iim = 0; iim < num_image; ++iim) {
    // Define augmentation settings
    // std::cout << "* image " << iim << "\n";

    const int im_offset_in = iim * w_in * h_in * num_ch;
    const T *x_im = x + im_offset_in;
    int im_offset_out = iim * w_out * h_out * num_ch;
    T *y_im = y + im_offset_out;
    // std::cout << "offset : in=" << im_offset_in << ", out=" << im_offset_out
    // << "\n";

    const float scale =
        min_scale_ *
        std::exp((rgen() % 1001) * 0.001f *
                 std::log(max_scale_ / min_scale_)); // [min_scale_, max_scale_]
    const float scale_x =
        std::exp(-std::log(this->aspect_ratio_) * 0.5 +
                 (rgen() % 1001) * 0.001f * std::log(this->aspect_ratio_));
    const float scale_y = 1.0 / scale_x;
    const float i_scale_x = 1.0f / (scale * scale_x);
    const float i_scale_y = 1.0f / (scale * scale_y);
    // std::cout << "scale : min=" << min_scale_ << ", max=" << max_scale_ << ",
    // v=" << scale << ", inv=" << i_scale << "\n";

    const float angle =
        -angle_ + ((rgen() % 1001) * 0.001f) * angle_ * 2; // [-angle_, angle_]
    // std::cout << "angle : " << angle << "\n";

    // Preparation
    const float w_scaled = w_in_pad * scale * scale_x;
    const float h_scaled = h_in_pad * scale * scale_y;
    // std::cout << "shape_scaled : w=" << w_scaled << ", h=" << h_scaled <<
    // "\n";

    const float cx = (w_out - 1) * 0.5f;
    const float cy = (h_out - 1) * 0.5f;
    // std::cout << "center : x=" << cx << ", y=" << cy << "\n";

    const float cx_scaled =
        ((rgen() % 1001) * 0.001f) * (w_scaled - w_out) + cx;
    const float cy_scaled =
        ((rgen() % 1001) * 0.001f) * (h_scaled - h_out) + cy;
    // std::cout << "center_scaled : x=" << cx_scaled << ", y=" << cy_scaled <<
    // "\n";

    const bool flip_lr = flip_lr_ & (rgen() % 2);
    const bool flip_ud = flip_ud_ & (rgen() % 2);
    const float global_brightness =
        ((rgen() % 1001) * 0.001f * brightness_ * 2.0f) - brightness_;
    // std::cout << "global_brightness : " << global_brightness << "\n";
    const float global_contrast =
        std::exp((rgen() % 1001) * 0.001f * std::log(contrast_) * 2.0f) /
        contrast_;
    // std::cout << "global_contrast : " << global_contrast << "\n";

    T *channel_brightness = channel_brightness_buf + iim * num_ch;
    T *channel_contrast = channel_contrast_buf + iim * num_ch;
    for (int ic = 0; ic < num_ch; ++ic) {
      const float ch_brightness =
          brightness_each_
              ? ((rgen() % 1001) * 0.001f * brightness_ * 2.0f) - brightness_
              : global_brightness;
      channel_brightness[ic] = ch_brightness - contrast_center_;
      // std::cout << "channel_brightness - contrast_center_ : " <<
      // channel_brightness[ic] <<
      // "\n";

      const float ch_contrast = contrast_each_
                                    ? std::exp((rgen() % 1001) * 0.001f *
                                               std::log(contrast_) * 2.0f) /
                                          contrast_
                                    : global_contrast;
      channel_contrast[ic] = ch_contrast;
      // std::cout << "channel_contrast : " << channel_contrast[ic] << "\n";
    }

    const float distortion =
        std::exp(((rgen() % 1001) * 0.001f * 2.0f * distortion_) -
                 distortion_) -
        1.0f;
    // std::cout << "distortion : " << distortion << "\n";
    const float noise = (rgen() % 1001) * 0.001f * noise_;

    // Pixel loop
    const float cos_theta = std::cos(angle);
    const float sin_theta = std::sin(angle);
    const float x_ax = (flip_lr ? -cos_theta : cos_theta) * i_scale_x;
    const float y_ax = (flip_lr ? sin_theta : -sin_theta) * i_scale_y;
    const float x_ay = (flip_ud ? -sin_theta : sin_theta) * i_scale_x;
    const float y_ay = (flip_ud ? -cos_theta : cos_theta) * i_scale_y;
    int pixel_offset[4];
    T pixel_gain[4];
    float x0_in = (cx_scaled * i_scale_x) - (x_ax * cx + y_ax * cy) - pad_[1];
    float y0_in = (cy_scaled * i_scale_y) - (x_ay * cx + y_ay * cy) - pad_[0];
    for (int iy = 0; iy < h_out; ++iy) {
      float xl_in = x0_in;
      float yl_in = y0_in;

      for (int ix = 0; ix < w_out; ++ix) {
        // Clip border
        float x_in = xl_in;
        float y_in = yl_in;
        if (distortion != 0.0f) {
          float dist_x = (ix - w_out_half) * i_w_out_half;
          float dist_y = (iy - h_out_half) * i_h_out_half;
          const float r = sqrt(dist_x * dist_x + dist_y * dist_y);
          const float r2 = r * r;
          const float dist_scale = 1.0f / (1.0f + distortion);
          dist_x =
              (dist_x + dist_x * distortion * r2) * w_out_half * dist_scale +
              w_out_half;
          dist_y =
              (dist_y + dist_y * distortion * r2) * h_out_half * dist_scale +
              h_out_half;

          x_in = x0_in + dist_x * x_ax + dist_y * y_ax;
          y_in = y0_in + dist_x * x_ay + dist_y * y_ay;
        }

        if (x_in < 0) {
          x_in = 0.0f;
        } else if (x_in > w_in - 1) {
          x_in = w_in - 1;
        }
        if (y_in < 0) {
          y_in = 0.0f;
        } else if (y_in > h_in - 1) {
          y_in = h_in - 1;
        }

        // Prepare linear interpolation
        const int intx = (int)x_in;
        const int inty = (int)y_in;
        const float fmodx = x_in - intx;
        const float fmody = y_in - inty;
        const int intx_plus1 = intx < w_in - 1 ? intx + 1 : intx;
        const int inty_plus1 = inty < h_in - 1 ? inty + 1 : inty;
        // Top left
        pixel_offset[0] = intx + inty * w_in;
        pixel_gain[0] = (1 - fmodx) * (1 - fmody);
        // Top right
        pixel_offset[1] = intx_plus1 + inty * w_in;
        pixel_gain[1] = fmodx * (1 - fmody);
        // Bottom left
        pixel_offset[2] = intx + inty_plus1 * w_in;
        pixel_gain[2] = (1 - fmodx) * fmody;
        // Bottom right
        pixel_offset[3] = intx_plus1 + inty_plus1 * w_in;
        pixel_gain[3] = fmodx * fmody;

        // Channel loop
        for (int ic = 0; ic < num_ch; ++ic) {
          const T *xr = x_im + ic * ch_size_in;

          // Linear interpolation
          T result = (T)0;
          for (int i = 0; i < 4; ++i) {
            result += xr[pixel_offset[i]] * pixel_gain[i];
          }
          result = (result + channel_brightness[ic]) * channel_contrast[ic] +
                   contrast_center_;
          if (noise > 0.0f) {
            result += norm(rgen) * noise;
          }
          y_im[ic * ch_size_out] = result;
        }
        xl_in += x_ax;
        yl_in += x_ay;
        y_im++;
      }
      if (distortion != 0.0f) {
      } else {
        x0_in += y_ax;
        y0_in += y_ay;
      }
    }
  }
  delete[] channel_brightness_buf;
  delete[] channel_contrast_buf;
}

template <typename T>
void ImageAugmentation<T>::backward_impl(const Variables &inputs,
                                         const Variables &outputs,
                                         const vector<bool> &propagate_down,
                                         const vector<bool> &accum) {
  // Not supported
}

} // namespace nbla
