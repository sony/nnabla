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

#ifndef __NBLA_FUNCTION_UTILS_BOOL_INDEXING_HPP__
#define __NBLA_FUNCTION_UTILS_BOOL_INDEXING_HPP__

#include <algorithm>
#include <utility>
#include <vector>

namespace nbla {

template <typename T, bool accum = false>
inline void kernel_bool_gather(int D, int B, int nnz, T *sdata, const T *gdata,
                               const T *mask) {
  for (int d = 0; d < D; ++d) {
    auto idx_nnz = 0;
    for (int b = 0; b < B && idx_nnz < nnz; ++b) {
      auto mask_b = int(mask[b] != T(0));
      auto masked_gdata = mask_b * gdata[b * D + d];
      // After written, increment idx_nnz, thus no overwrite to the previously
      // written value.
      sdata[idx_nnz * D + d] =
          accum ? sdata[idx_nnz * D + d] + masked_gdata : masked_gdata;
      idx_nnz += mask_b;
    }
  }
}

template <typename T, bool accum = false, bool inplace = false>
inline void kernel_bool_scatter(int D, int B, int nnz, T *gdata, const T *sdata,
                                const T *mask) {
  for (int d = 0; d < D; ++d) {
    auto idx_nnz = 0;
    for (int b = 0; b < B; ++b) {
      auto mask_b = int(mask[b] != T(0));
      auto sdata_i = sdata[idx_nnz * D + d];

      auto masked_sdata_i = mask_b * sdata_i;
      if (inplace) {
        // (1 - mask_b)-multiply trick does not work if gdata is in {-inf, inf,
        // nan}
        masked_sdata_i = mask_b ? masked_sdata_i : gdata[b * D + d];
      }

      if (accum)
        gdata[b * D + d] += masked_sdata_i;
      else
        gdata[b * D + d] = masked_sdata_i;
      idx_nnz += mask_b;
      idx_nnz = std::min(idx_nnz, nnz - 1); // prevent illegal memory access
    }
  }
}
}
#endif
