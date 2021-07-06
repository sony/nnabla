// Copyright 2020,2021 Sony Corporation.
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

#ifndef NBLA_FUNCTION_RNN_HPP
#define NBLA_FUNCTION_RNN_HPP

#include <nbla/cpu.hpp>
#include <nbla/function.hpp>
#include <nbla/function_registry.hpp>

namespace nbla {
namespace function {
namespace utils {
namespace rnn {

inline void compute_batch_sizes(const int *lengths, int lsize,
                                int *batch_sizes) {
  vector<int> tmp_lengths(lengths, lengths + lsize);
  auto i = 0;
  while (true) {
    // Count #{e > 0}
    auto c = std::count_if(tmp_lengths.begin(), tmp_lengths.end(),
                           [](int l) { return l > 0; });
    if (c == 0) {
      break;
    }
    batch_sizes[i] = c;
    i++;
    // Decrement
    std::transform(tmp_lengths.begin(), tmp_lengths.end(), tmp_lengths.begin(),
                   [](int l) { return l - 1; });
  }
}

template <typename U, bool accum = false>
inline void pack(const U *padded_sequence, const int *batch_sizes,
                 U *packed_sequence, int T, int B, int D) {
  auto stride_t = B * D;
  for (int t = 0; t < T; t++) {
    auto batch_size = batch_sizes[t];
    auto padded_sequence_t = padded_sequence + t * stride_t;
    for (int b = 0; b < batch_size; b++) {
      for (int d = 0; d < D; d++) {
        *packed_sequence =
            accum ? *packed_sequence + *padded_sequence_t : *padded_sequence_t;
        packed_sequence++;
        padded_sequence_t++;
      }
    }
  }
}

template <typename U, bool accum = false>
inline void pack_batch_first(const U *padded_sequence, const int *batch_sizes,
                             U *packed_sequence, int T, int B, int D) {
  auto stride_b = T * D;
  auto stride_t = D;
  for (int t = 0; t < T; t++) {
    auto batch_size = batch_sizes[t];
    for (int b = 0; b < batch_size; b++) {
      auto padded_sequence_bt = padded_sequence + b * stride_b + t * stride_t;
      for (int d = 0; d < D; d++) {
        *packed_sequence = accum ? *packed_sequence + *padded_sequence_bt
                                 : *padded_sequence_bt;
        packed_sequence++;
        padded_sequence_bt++;
      }
    }
  }
}

template <typename U, bool accum = false>
inline void pack_batch_first(const U *padded_sequence, const int *batch_sizes,
                             U *packed_sequence, int T, int B, int D, int TL) {
  auto stride_b = TL * D;
  auto stride_t = D;
  for (int t = 0; t < T; t++) {
    auto batch_size = batch_sizes[t];
    for (int b = 0; b < batch_size; b++) {
      auto padded_sequence_bt = padded_sequence + b * stride_b + t * stride_t;
      for (int d = 0; d < D; d++) {
        *packed_sequence = accum ? *packed_sequence + *padded_sequence_bt
                                 : *padded_sequence_bt;
        packed_sequence++;
        padded_sequence_bt++;
      }
    }
  }
}

inline void compute_lengths(const int *batch_sizes, int bsize, int *lengths) {
  vector<int> tmp_batch_sizes;
  for (int b = 0; b < bsize; b++) {
    tmp_batch_sizes.push_back(batch_sizes[b]);
  }
  auto i = 0;
  while (true) {
    // Count #{e > 0}
    auto c = std::count_if(tmp_batch_sizes.begin(), tmp_batch_sizes.end(),
                           [](int b) { return b > 0; });
    if (c == 0) {
      break;
    }
    lengths[i] = c;
    i++;
    // Decrement
    std::transform(tmp_batch_sizes.begin(), tmp_batch_sizes.end(),
                   tmp_batch_sizes.begin(), [](int b) { return b - 1; });
  }
}

template <typename U, bool accum = false>
inline void unpack(const U *packed_sequence, const int *batch_sizes,
                   U *padded_sequence, int T, int B, int D) {
  for (int t = 0; t < T; t++) {
    auto batch_size = batch_sizes[t];
    auto padded_sequence_t = padded_sequence + t * (B * D);
    for (int b = 0; b < B; b++) {
      for (int d = 0; d < D; d++) {
        if (b < batch_size) {
          *padded_sequence_t =
              accum ? *padded_sequence_t + *packed_sequence : *packed_sequence;
          packed_sequence++;
        } else {
          if (!accum)
            *padded_sequence_t = U(0);
        }
        padded_sequence_t++;
      }
    }
  }
}

template <typename U, bool accum = false>
inline void unpack(const U *packed_sequence, const int *batch_sizes,
                   U *padded_sequence, int T, int B, int D, int TL) {
  auto stride_t = B * D;
  for (int t = 0; t < TL; t++) {
    auto padded_sequence_t = padded_sequence + t * stride_t;
    if (t >= T) {
      for (int b = 0; b < B; b++) {
        for (int d = 0; d < D; d++) {
          if (!accum) {
            *padded_sequence_t++ = U(0);
          }
        }
      }
      continue;
    }
    auto batch_size = batch_sizes[t];
    for (int b = 0; b < B; b++) {
      for (int d = 0; d < D; d++) {
        if (b < batch_size) {
          *padded_sequence_t =
              accum ? *padded_sequence_t + *packed_sequence : *packed_sequence;
          packed_sequence++;
        } else {
          if (!accum)
            *padded_sequence_t = U(0);
        }
        padded_sequence_t++;
      }
    }
  }
}

template <typename U, bool accum = false>
inline void unpack_batch_first(const U *packed_sequence, const int *batch_sizes,
                               U *padded_sequence, int T, int B, int D) {
  auto stride_b = T * D;
  auto stride_t = D;
  for (int t = 0; t < T; t++) {
    auto batch_size = batch_sizes[t];
    for (int b = 0; b < B; b++) {
      auto padded_sequence_bt = padded_sequence + b * stride_b + t * stride_t;
      for (int d = 0; d < D; d++) {
        if (b < batch_size) {
          *padded_sequence_bt =
              accum ? *padded_sequence_bt + *packed_sequence : *packed_sequence;
          packed_sequence++;
        } else {
          if (!accum)
            *padded_sequence_bt = U(0);
        }
        padded_sequence_bt++;
      }
    }
  }
}

template <typename U, bool accum = false>
inline void unpack_batch_first(const U *packed_sequence, const int *batch_sizes,
                               U *padded_sequence, int T, int B, int D,
                               int TL) {
  auto stride_b = TL * D;
  auto stride_t = D;
  for (int t = 0; t < TL; t++) {
    if (t >= T) {
      for (int b = 0; b < B; b++) {
        auto padded_sequence_bt = padded_sequence + b * stride_b + t * stride_t;
        for (int d = 0; d < D; d++) {
          if (!accum) {
            *padded_sequence_bt++ = U(0);
          }
        }
      }
      continue;
    }
    auto batch_size = batch_sizes[t];
    for (int b = 0; b < B; b++) {
      auto padded_sequence_bt = padded_sequence + b * stride_b + t * stride_t;
      for (int d = 0; d < D; d++) {
        if (b < batch_size) {
          *padded_sequence_bt =
              accum ? *padded_sequence_bt + *packed_sequence : *packed_sequence;
          packed_sequence++;
        } else {
          if (!accum)
            *padded_sequence_bt = U(0);
        }
        padded_sequence_bt++;
      }
    }
  }
}

} // rnn
} // utils
} // function
} // nbla

#endif
