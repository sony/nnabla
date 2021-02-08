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
#include <nbla/function/random_erase.hpp>
#include <nbla/random_manager.hpp>
#include <nbla/variable.hpp>

namespace nbla {

NBLA_REGISTER_FUNCTION_SOURCE(RandomErase, float, const vector<float> &,
                              const vector<float> &, const vector<float> &, int,
                              bool, bool, int, int, bool, bool);

template <typename T>
void generate_random_coords(float *random_coords, const size_t N,
                            const size_t B, const size_t C, const size_t H,
                            const size_t W, const vector<float> &area_ratios,
                            const vector<float> &aspect_ratios,
                            const bool share, std::mt19937 &rgen) {
  std::uniform_real_distribution<typename force_float<T>::type> rdist_prob(
      T(0.0), T(1.0));
  std::uniform_real_distribution<typename force_float<T>::type> rdist_area(
      area_ratios[0], area_ratios[1]);
  std::uniform_real_distribution<typename force_float<T>::type> rdist_aspect(
      aspect_ratios[0], aspect_ratios[1]);
  size_t area = H * W;
  float eprob;
  float Se;
  float Re;
  size_t He;
  size_t We;
  size_t ye_start;
  size_t xe_start;
  size_t ye_end;
  size_t xe_end;

  auto generate_coords_and_next = [&](float *random_coords) {
    Se = rdist_area(rgen) * area;
    Re = rdist_aspect(rgen);
    He = std::sqrt(Se * Re);
    We = std::sqrt(Se / Re);
    He = std::min(He, H);
    We = std::min(We, W);
    std::uniform_int_distribution<int> rdist0(0, H - He);
    std::uniform_int_distribution<int> rdist1(0, W - We);
    eprob = rdist_prob(rgen);
    ye_start = rdist0(rgen);
    xe_start = rdist1(rgen);
    ye_end = ye_start + He;
    xe_end = xe_start + We;
    // coordinates
    random_coords[0] = eprob;
    random_coords[1] = ye_start;
    random_coords[2] = xe_start;
    random_coords[3] = ye_end;
    random_coords[4] = xe_end;
    return random_coords + 5;
  };

  if (share) {
    for (int n = 0; static_cast<size_t>(n) < N; n++) {
      for (int b = 0; static_cast<size_t>(b) < B; b++) {
        random_coords = generate_coords_and_next(random_coords);
      }
    }
  } else {
    for (int n = 0; static_cast<size_t>(n) < N; n++) {
      for (int b = 0; static_cast<size_t>(b) < B; b++) {
        for (size_t c = 0; static_cast<size_t>(c) < C; c++) {
          random_coords = generate_coords_and_next(random_coords);
        }
      }
    }
  }
}

template <typename T>
void erase_2d(T *out, const float *random_coords, const size_t C,
              const size_t H, const size_t W, const float prob,
              const vector<float> &replacements, const bool share,
              std::mt19937 &rgen) {
  std::uniform_real_distribution<typename force_float<T>::type>
      rdist_replacement(replacements[0], replacements[1]);
  float eprob;
  size_t ye_start;
  size_t xe_start;
  size_t ye_end;
  size_t xe_end;
  if (share) {
    eprob = random_coords[0];
    ye_start = random_coords[1];
    xe_start = random_coords[2];
    ye_end = random_coords[3];
    xe_end = random_coords[4];
    for (size_t c = 0; c < C; c++) {
      for (size_t h = 0; h < H; h++) {
        for (size_t w = 0; w < W; w++) {
          if ((eprob <= prob) && (ye_start <= h && h <= ye_end) &&
              (xe_start <= w && w <= xe_end)) {
            *out = rdist_replacement(rgen);
          }
          out++;
        }
      }
    }
  } else {
    for (size_t c = 0; c < C; c++) {
      eprob = random_coords[0];
      ye_start = random_coords[1];
      xe_start = random_coords[2];
      ye_end = random_coords[3];
      xe_end = random_coords[4];
      for (size_t h = 0; h < H; h++) {
        for (size_t w = 0; w < W; w++) {
          if ((eprob <= prob) && (ye_start <= h && h <= ye_end) &&
              (xe_start <= w && w <= xe_end)) {
            *out = rdist_replacement(rgen);
          }
          out++;
        }
      }
      random_coords = random_coords + 5;
    }
  }
}

template <typename T, bool accum = false>
void random_erase_2d_backward(T *gx, const T *gy, const float *random_coords,
                              const size_t C, const size_t H, const size_t W,
                              const float prob, const bool share) {
  float eprob;
  size_t ye_start;
  size_t xe_start;
  size_t ye_end;
  size_t xe_end;
  if (share) {
    eprob = random_coords[0];
    ye_start = random_coords[1];
    xe_start = random_coords[2];
    ye_end = random_coords[3];
    xe_end = random_coords[4];
    for (size_t c = 0; c < C; c++) {
      for (size_t h = 0; h < H; h++) {
        for (size_t w = 0; w < W; w++) {
          if ((eprob <= prob) && (ye_start <= h && h <= ye_end) &&
              (xe_start <= w && w <= xe_end)) {
            *gx = accum ? *gx + T(0) : T(0);
          } else {
            *gx = accum ? *gx + *gy : *gy;
          }
          gx++;
          gy++;
        }
      }
    }
  } else {
    for (size_t c = 0; c < C; c++) {
      eprob = random_coords[0];
      ye_start = random_coords[1];
      xe_start = random_coords[2];
      ye_end = random_coords[3];
      xe_end = random_coords[4];
      for (size_t h = 0; h < H; h++) {
        for (size_t w = 0; w < W; w++) {
          if ((eprob <= prob) && (ye_start <= h && h <= ye_end) &&
              (xe_start <= w && w <= xe_end)) {
            *gx = accum ? *gx + T(0) : T(0);
          } else {
            *gx = accum ? *gx + *gy : *gy;
          }
          gx++;
          gy++;
        }
      }
      random_coords = random_coords + 5;
    }
  }
}

template <typename T>
void RandomErase<T>::setup_impl(const Variables &inputs,
                                const Variables &outputs) {
  NBLA_CHECK(prob_ >= 0.0 && prob_ <= 1.0, error_code::value,
             "prob must be in [0.0, 1.0]. prob = %f.", prob_);
  NBLA_CHECK(area_ratios_.size() == 2, error_code::value,
             "Length of area_ratios must be 2.");
  NBLA_CHECK(aspect_ratios_.size() == 2, error_code::value,
             "Length of aspect_ratios must be 2.");
  NBLA_CHECK(n_ > 0, error_code::value, "n must be positive. n = %d.", n_);
  NBLA_CHECK(replacements_.size() == 2, error_code::value,
             "Length of replacements must be 2.");
  NBLA_CHECK((size_t)base_axis_ < inputs[0]->shape().size(), error_code::value,
             "base_axis must be less than ndim of inputs[0]. "
             "base_axis: %d >= ndim of inputs[0]: %d.",
             base_axis_, inputs[0]->shape().size());
  NBLA_CHECK(
      inputs[0]->shape().size() - base_axis_ == 3, error_code::value,
      "Image (the number of the spatial dimensions is 2) is only supported.");

  outputs[0]->reshape(inputs[0]->shape(), true);
  if (inplace_) {
    outputs[0]->data()->set_array(inputs[0]->data()->array());
  }

  rgen_ = std::mt19937((seed_ == -1 ? std::random_device()() : seed_));
}

template <typename T>
void RandomErase<T>::forward_impl(const Variables &inputs,
                                  const Variables &outputs) {
  NBLA_CHECK(!channel_last_, error_code::value,
             "Channel Last is not supported in CPU.");

  auto shape = outputs[0]->shape();
  auto N = n_;
  auto B = std::accumulate(shape.begin(), std::next(shape.begin(), base_axis_),
                           1, std::multiplies<size_t>());
  auto C = shape[base_axis_];
  auto H = shape[base_axis_ + 1];
  auto W = shape[base_axis_ + 2];
  auto Bs = C * H * W;
  std::mt19937 &rgen =
      seed_ == -1 ? SingletonManager::get<RandomManager>()->get_rand_generator()
                  : rgen_;

  const T *x = inputs[0]->get_data_pointer<T>(this->ctx_);
  T *y = outputs[0]->cast_data_and_get_pointer<T>(this->ctx_, !inplace_);

  // Generate N x B (x C) x 5, 5 is {prob, Se, re, xe, ye}
  this->random_coordinates_ =
      this->share_ ? std::make_shared<NdArray>(Shape_t{N, B, 5})
                   : std::make_shared<NdArray>(Shape_t{N, B, C, 5});
  float *random_coords =
      this->random_coordinates_->cast(get_dtype<float>(), this->ctx_)
          ->template pointer<float>();
  generate_random_coords<T>(random_coords, N, B, C, H, W, area_ratios_,
                            aspect_ratios_, share_, rgen);

  // Copy once
  auto size = inputs[0]->size();
  for (int i = 0; i < size; ++i) {
    y[i] = x[i];
  }

  // Erase randomly
  for (decltype(n_) n = 0; n < n_; n++) {
    for (decltype(B) b = 0; b < B; b++) {
      auto out = y + (b * Bs);
      erase_2d(out, random_coords, C, H, W, prob_, replacements_, share_, rgen);
      if (share_) {
        random_coords = random_coords + 5;
      } else {
        random_coords = random_coords + (C * 5);
      }
    }
  }
  // Release memory
  if (!ste_fine_grained_) {
    this->random_coordinates_ = nullptr;
  }
}

template <typename T>
void RandomErase<T>::backward_impl(const Variables &inputs,
                                   const Variables &outputs,
                                   const vector<bool> &propagate_down,
                                   const vector<bool> &accum) {
  if (!(propagate_down[0])) {
    return;
  }

  auto size = inputs[0]->size();
  auto shape = outputs[0]->shape();
  auto N = n_;
  auto B = std::accumulate(shape.begin(), std::next(shape.begin(), base_axis_),
                           1, std::multiplies<size_t>());
  auto C = shape[base_axis_];
  auto H = shape[base_axis_ + 1];
  auto W = shape[base_axis_ + 2];

  T *g_x = inputs[0]->cast_grad_and_get_pointer<T>(this->ctx_, !accum[0]);
  const T *g_y = outputs[0]->get_grad_pointer<T>(this->ctx_);

  // STE
  if (!ste_fine_grained_) {
    for (int s = 0; s < size; ++s) {
      g_x[s] = accum[0] ? (g_x[s] + g_y[s]) : g_y[s];
    }
    return;
  }

  // Correct backward
  const float *random_coords =
      this->random_coordinates_->cast(get_dtype<float>(), this->ctx_)
          ->template pointer<float>();
  float eprob;
  size_t ye_start;
  size_t xe_start;
  size_t ye_end;
  size_t xe_end;
  auto stride_n = share_ ? (B * 5) : (B * C * 5);
  auto stride_b = share_ ? (5) : (C * 5);
  auto stride_c = share_ ? (0) : (5);
  for (decltype(B) b = 0; b < B; b++) {
    for (decltype(C) c = 0; c < C; c++) {
      for (decltype(H) h = 0; h < H; h++) {
        for (decltype(W) w = 0; w < W; w++) {
          // No accumulation for the overlapping pixel of patches other than the
          // original value, g_x.
          auto fall = false;
          for (decltype(N) n = 0; n < N; n++) {
            auto idx = n * stride_n + b * stride_b + c * stride_c;
            eprob = random_coords[idx + 0];
            ye_start = random_coords[idx + 1];
            xe_start = random_coords[idx + 2];
            ye_end = random_coords[idx + 3];
            xe_end = random_coords[idx + 4];
            if ((eprob <= prob_) && (ye_start <= static_cast<size_t>(h) &&
                                     static_cast<size_t>(h) <= ye_end) &&
                (xe_start <= static_cast<size_t>(w) &&
                 static_cast<size_t>(w) <= xe_end)) {
              fall = true;
              break;
            }
          }
          if (fall) {
            *g_x = accum[0] ? (*g_x + T(0)) : T(0);
          } else {
            *g_x = accum[0] ? (*g_x + *g_y) : *g_y;
          }
          g_x++;
          g_y++;
        }
      }
    }
  }

  // Release memory
  this->random_coordinates_ = nullptr;
}
}
