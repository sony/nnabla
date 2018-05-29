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

#include <nbla/utils/fold_from_patches.hpp>

#include <nbla/half.hpp>

namespace nbla {

namespace ns_fold_from_patches {

inline bool index_in_shape(int index, int shape) {
  return static_cast<unsigned>(index) < static_cast<unsigned>(shape);
}

template <typename T>
inline void kernel_1d(const T *column_data, const int *outmap_shape,
                      const int *sample_shape, const int *sample_shift,
                      const int *stride, T *outmap_data) {
  int sample_index = *sample_shift;

  for (int outmap_index = 0; outmap_index < *outmap_shape; outmap_index++) {
    if (index_in_shape(sample_index, *sample_shape)) {
      outmap_data[sample_index] += column_data[outmap_index];
    }
    sample_index += *stride;
  }
}

template <typename T>
inline void kernel_2d(const T *column_data, const int *outmap_shape,
                      const int *outmap_isize, const int *sample_shape,
                      const int *sample_isize, const int *sample_shift,
                      const int *stride, T *outmap_data) {
  int sample_index = *sample_shift;

  for (int outmap_index = 0; outmap_index < *outmap_shape; outmap_index++) {
    if (index_in_shape(sample_index, *sample_shape)) {
      auto outmap_data_ptr = outmap_data + *sample_isize * sample_index;
      kernel_1d<T>(column_data, outmap_shape + 1, sample_shape + 1,
                   sample_shift + 1, stride + 1, outmap_data_ptr);
    }
    column_data += *outmap_isize;
    sample_index += *stride;
  }
}

template <typename T>
void kernel_nd(const int dimensions, const T *column_data,
               const int *outmap_shape, const int *outmap_isize,
               const int *sample_shape, const int *sample_isize,
               const int *sample_shift, const int *stride, T *outmap_data) {
  int sample_index = *sample_shift;

  for (int outmap_index = 0; outmap_index < *outmap_shape; outmap_index++) {
    if (index_in_shape(sample_index, *sample_shape)) {
      auto outmap_data_ptr = outmap_data + *sample_isize * sample_index;
      if (dimensions == 2) {
        kernel_1d<T>(column_data, outmap_shape + 1, sample_shape + 1,
                     sample_shift + 1, stride + 1, outmap_data_ptr);
      } else {
        kernel_nd<T>(dimensions - 1, column_data, outmap_shape + 1,
                     outmap_isize + 1, sample_shape + 1, sample_isize + 1,
                     sample_shift + 1, stride + 1, outmap_data_ptr);
      }
    }
    column_data += *outmap_isize;
    sample_index += *stride;
  }
}

} // namespace ns_fold_from_patches

using namespace ns_fold_from_patches;

template <typename T>
void fold_from_patches(const T *column_data, T *outmap_data, const int channels,
                       const vector<int> &shape, const vector<int> &kernel,
                       const vector<int> &padding, const vector<int> &stride,
                       const vector<int> &dilation) {

  const vector<int> &sample_shape = shape;
  const int ndim = kernel.size();

  int outmap_outer_size = 1;
  vector<int> outmap_shape(ndim);
  vector<int> outmap_isize(ndim);
  for (int i = ndim - 1; i >= 0; --i) {
    auto k = kernel[i], p = padding[i], d = dilation[i], s = stride[i];
    outmap_shape[i] = (sample_shape[i] + 2 * p - (d * (k - 1) + 1)) / s + 1;
    outmap_isize[i] = outmap_outer_size;
    outmap_outer_size *= outmap_shape[i];
  }

  int sample_outer_size = 1;
  vector<int> sample_isize(ndim);
  for (int i = ndim - 1; i >= 0; --i) {
    sample_isize[i] = sample_outer_size;
    sample_outer_size *= sample_shape[i];
  }

  vector<int> kernel_shape(ndim + 1);
  vector<int> kernel_isize(ndim + 1);
  int kernel_outer_size = 1;
  for (int i = ndim - 1; i >= 0; --i) {
    kernel_shape[i + 1] = kernel[i];
    kernel_isize[i + 1] = kernel_outer_size;
    kernel_outer_size *= kernel[i];
  }
  kernel_shape[0] = channels;
  kernel_isize[0] = kernel_outer_size;
  kernel_outer_size *= channels;

  vector<int> sample_shift(ndim);
  vector<int> kernel_index(ndim + 1);

  for (int k = 0; k < kernel_outer_size; k++) {

    for (int i = 0; i < ndim + 1; i++)
      kernel_index[i] = (k / kernel_isize[i]) % kernel_shape[i];

    for (int i = 0; i < ndim; i++)
      sample_shift[i] = kernel_index[i + 1] * dilation[i] - padding[i];

    T *outmap_data_ptr = outmap_data + kernel_index[0] * sample_outer_size;

    switch (ndim) {
    case 1:
      kernel_1d<T>(column_data, outmap_shape.data(), sample_shape.data(),
                   sample_shift.data(), stride.data(), outmap_data_ptr);
      break;
    case 2:
      kernel_2d<T>(column_data, outmap_shape.data(), outmap_isize.data(),
                   sample_shape.data(), sample_isize.data(),
                   sample_shift.data(), stride.data(), outmap_data_ptr);
      break;
    default:
      kernel_nd<T>(ndim, column_data, outmap_shape.data(), outmap_isize.data(),
                   sample_shape.data(), sample_isize.data(),
                   sample_shift.data(), stride.data(), outmap_data_ptr);
    }
    column_data += outmap_outer_size;
  }
}

// Template specialization
#define NBLA_SPEC_FOLD_FROM_PATCHS(TYPE)                                       \
  template void fold_from_patches<TYPE>(                                       \
      const TYPE *column_data, TYPE *outmap_data, const int channels,          \
      const vector<int> &shape, const vector<int> &kernel,                     \
      const vector<int> &padding, const vector<int> &stride,                   \
      const vector<int> &dilation)

NBLA_SPEC_FOLD_FROM_PATCHS(float);
NBLA_SPEC_FOLD_FROM_PATCHS(Half);
}
