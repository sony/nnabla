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

#ifndef __NBLA_UTILS_IM2COL_HPP__
#define __NBLA_UTILS_IM2COL_HPP__
namespace nbla {

template <typename T>
void im2col(const T *img, const int c_i, const int *shape, const int *k,
            const int *p, const int *s, const int *d, T *col);

template <typename T>
void im2col_nd(const T *img, const int c, const int spatial_dims,
               const int *spatial_shape, const int *kernel, const int *pad,
               const int *stride, const int *dilation, T *col);

template <typename T>
void col2im(const T *col, const int c_i, const int *shape, const int *k,
            const int *p, const int *s, const int *d, T *img);

template <typename T>
void col2im_nd(const T *col, const int c, const int spatial_dims,
               const int *spatial_shape, const int *kernel, const int *pad,
               const int *stride, const int *dilation, T *img);
}
#endif
