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

#include <nbla/utils/unfold_to_patches.hpp>

#include <nbla/half.hpp>

#include <cstring>
#include <thread>

using std::memset;
using std::thread;

namespace nbla {

namespace ns_unfold_to_patches {

inline bool index_in_shape(int index, int shape) {
  return static_cast<unsigned>(index) < static_cast<unsigned>(shape);
}

template <typename T>
inline void kernel_1d(const T *sample_data, const int *const outmap_shape,
                      const int *const sample_shape,
                      const int *const sample_shift, const int *const stride,
                      T *column_data) {
  int sample_index = *sample_shift;

  for (int i = 0; i < *outmap_shape; i++) {
    *column_data++ = index_in_shape(sample_index, *sample_shape)
                         ? sample_data[sample_index]
                         : (T)0;
    sample_index += *stride;
  }
}

template <typename T>
inline void
kernel_2d(const T *sample_data, const int *const outmap_shape,
          const int *const outmap_isize, const int *const sample_shape,
          const int *const sample_isize, const int *const sample_shift,
          const int *const stride, T *column_data) {
  int sample_index = *sample_shift;
  sample_data += *sample_isize * sample_index;
  const int sample_stride = *sample_isize * *stride;

  for (int i = 0; i < *outmap_shape; i++) {
    if (index_in_shape(sample_index, *sample_shape)) {
      kernel_1d<T>(sample_data, outmap_shape + 1, sample_shape + 1,
                   sample_shift + 1, stride + 1, column_data);
    } else {
      memset((void *)column_data, 0, *outmap_isize * sizeof(T));
    }
    sample_index += *stride;
    sample_data += sample_stride;
    column_data += *outmap_isize;
  }
}

template <typename T>
void kernel_nd(const int dimensions, const T *sample_data,
               const int *const outmap_shape, const int *const outmap_isize,
               const int *const sample_shape, const int *const sample_isize,
               const int *const sample_shift, const int *const stride,
               T *column_data) {
  int sample_index = *sample_shift;
  sample_data += *sample_isize * sample_index;
  const int sample_stride = *sample_isize * *stride;

  for (int i = 0; i < *outmap_shape; i++) {
    if (!index_in_shape(sample_index, *sample_shape)) {
      memset((void *)column_data, 0, *outmap_isize * sizeof(T));
    } else if (dimensions > 2) {
      kernel_nd<T>(dimensions - 1, sample_data, outmap_shape + 1,
                   outmap_isize + 1, sample_shape + 1, sample_isize + 1,
                   sample_shift + 1, stride + 1, column_data);
    } else {
      kernel_1d<T>(sample_data, outmap_shape + 1, sample_shape + 1,
                   sample_shift + 1, stride + 1, column_data);
    }
    sample_index += *stride;
    sample_data += sample_stride;
    column_data += *outmap_isize;
  }
}

} // namespace ns_unfold_to_patches

using namespace ns_unfold_to_patches;

template <typename T>
void unfold_to_patches(const T *sample_data, T *column_data, const int channels,
                       const vector<int> &shape, const vector<int> &kernel,
                       const vector<int> &padding, const vector<int> &stride,
                       const vector<int> &dilation) {

  // Convert an ND-Tensor *sample_data* into a 2D-Tensor *column_data*
  // where each column contains the sample data values seen by a
  // filter kernel while iterating each possible position in the
  // sample tensor constrained by the *kernel*, *padding*, *stride*
  // and *dilation* ND-Vectors.
  //
  // Sample dimensions are (channels, shape[0] ... shape[ND-1]).
  // Filter dimensions are (channels, kernel[0] ... kernel[ND-1]).
  //
  // The column data tensor has one row per filter value and one
  // column per possible position of the filter within the sample data
  // shape, such that multiplication of a filter value vector with the
  // column_data matrix produces all values of a convolutional output
  // feature map, referred to as outmap.

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

#ifdef _OPENMP
#pragma omp parallel
  {
#endif
    vector<int> sample_shift(ndim);
    vector<int> kernel_index(ndim + 1);
#ifdef _OPENMP
#pragma omp for
#endif
    for (int k = 0; k < kernel_outer_size; k++) {
      auto column_data_p = column_data + k * outmap_outer_size;
      for (int i = 0; i < ndim + 1; i++)
        kernel_index[i] = (k / kernel_isize[i]) % kernel_shape[i];

      for (int i = 0; i < ndim; i++)
        sample_shift[i] = kernel_index[i + 1] * dilation[i] - padding[i];

      auto sample_data_ptr = sample_data + kernel_index[0] * sample_outer_size;

      switch (ndim) {
      case 1:
        kernel_1d<T>(sample_data_ptr, outmap_shape.data(), sample_shape.data(),
                     sample_shift.data(), stride.data(), column_data_p);
        break;
      case 2:
        kernel_2d<T>(sample_data_ptr, outmap_shape.data(), outmap_isize.data(),
                     sample_shape.data(), sample_isize.data(),
                     sample_shift.data(), stride.data(), column_data_p);
        break;
      default:
        kernel_nd<T>(ndim, sample_data_ptr, outmap_shape.data(),
                     outmap_isize.data(), sample_shape.data(),
                     sample_isize.data(), sample_shift.data(), stride.data(),
                     column_data_p);
      }
    }
#ifdef _OPENMP
  }
#endif
}

// Template specialization
#define NBLA_SPEC_UNFOLD_TO_PATCHS(TYPE)                                       \
  template void unfold_to_patches<TYPE>(                                       \
      const TYPE *sample_data, TYPE *column_data, const int channels,          \
      const vector<int> &shape, const vector<int> &kernel,                     \
      const vector<int> &padding, const vector<int> &stride,                   \
      const vector<int> &dilation)
NBLA_SPEC_UNFOLD_TO_PATCHS(float);
NBLA_SPEC_UNFOLD_TO_PATCHS(Half);
}
