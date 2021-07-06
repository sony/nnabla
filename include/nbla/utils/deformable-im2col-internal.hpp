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

#ifndef __NBLA_UTILS_DEFORMABLE_IM2COL_INTERNAL_HPP__
#define __NBLA_UTILS_DEFORMABLE_IM2COL_INTERNAL_HPP__

#include <vector>

using std::vector;

namespace nbla {

// Deformable Convolution functions
template <typename T, bool MODULATED>
void modulated_deformable_im2col_cpu(const T *im, const T *offset,
                                     const T *mask, const int c_i,
                                     const vector<int> &shape,
                                     const vector<int> &k, const vector<int> &p,
                                     const vector<int> &s, const vector<int> &d,
                                     const int deformable_group, T *col);

template <typename T, bool MODULATED>
void modulated_deformable_col2im_cpu(const T *col, const T *offset,
                                     const T *mask, const int c_i,
                                     const vector<int> &shape,
                                     const vector<int> &k, const vector<int> &p,
                                     const vector<int> &s, const vector<int> &d,
                                     const int deformable_group, T *grad_im);

template <typename T, bool MODULATED>
void modulated_deformable_col2im_coord_cpu(
    const T *col, const T *im, const T *offset, const T *mask, const int c_i,
    const vector<int> &shape, const vector<int> &k, const vector<int> &p,
    const vector<int> &s, const vector<int> &d, const int deformable_group,
    T *grad_offset, T *grad_mask);
}
#endif
