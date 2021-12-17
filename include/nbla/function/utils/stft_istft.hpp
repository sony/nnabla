// Copyright 2021 Sony Corporation.
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

#ifndef NBLA_FUNCTION_STFT_ISTFT_HPP
#define NBLA_FUNCTION_STFT_ISTFT_HPP

#include <nbla/common.hpp>
#include <nbla/variable.hpp>

namespace nbla {
template <typename T>
void create_window(Variable *window, const string &window_type,
                   const int window_size, const int fft_size,
                   const Context &ctx) {
  window->data()->zero(); // For 0 padding
  auto window_data = window->cast_data_and_get_pointer<T>(ctx);
  const double pi = std::acos(-1);

  const int left_pad = (fft_size - window_size) / 2;
  if (window_type == "hanning") {
    for (int i = 0; i < window_size; i++) {
      window_data[left_pad + i] =
          0.5 - 0.5 * std::cos(2.0 * pi * i / (window_size));
    }
  } else if (window_type == "hamming") {
    for (int i = 0; i < window_size; i++) {
      window_data[left_pad + i] =
          0.54 - 0.46 * std::cos(2.0 * pi * i / (window_size));
    }
  } else { // window_type == "rectangular"
    // fill 1
    for (int i = 0; i < window_size; i++) {
      window_data[left_pad + i] = 1.;
    }
  }
}
}

#endif
