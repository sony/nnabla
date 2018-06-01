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

/** NmsDetection2d
 */
#include <iostream>
#include <nbla/array.hpp>
#include <nbla/array/cpu_array.hpp>
#include <nbla/function/nms_detection2d.hpp>
#include <nbla/variable.hpp>

namespace nbla {

NBLA_REGISTER_FUNCTION_SOURCE(NmsDetection2d, float, float, bool);

template <typename T>
void NmsDetection2d<T>::setup_impl(const Variables &inputs,
                                   const Variables &outputs) {
  // Check shape
  auto shape = inputs[0]->shape();
  int ndim = inputs[0]->ndim();
  NBLA_CHECK(ndim == 3, error_code::value,
             "Number of input dimension must be 3. Given %d.", ndim);
  NBLA_CHECK(shape[2] > 5, error_code::value,
             "Illegal input shape: The 2nd element (starting from 0) of the "
             "input's shape must be greater than 5. Given %d.",
             shape[2]);
  // Reshape output
  outputs[0]->reshape(shape, true);
}

template <typename T> T calculate_overlap(T x1, T w1, T x2, T w2) {
  return std::min(x1 + w1 / 2, x2 + w2 / 2) -
         std::max(x1 - w1 / 2, x2 - w2 / 2);
}

template <typename T> T calculate_iou(T *a, T *b) {
  // intersection
  T x1 = a[0];
  T y1 = a[1];
  T w1 = a[2];
  T h1 = a[3];
  T x2 = b[0];
  T y2 = b[1];
  T w2 = b[2];
  T h2 = b[3];

  // Intersection
  T w = calculate_overlap(x1, w1, x2, w2);
  T h = calculate_overlap(y1, h1, y2, h2);
  if (w <= 0 || h <= 0)
    return 0;

  T intersection = w * h;

  // Union
  T union_ = w1 * h1 + w2 * h2 - intersection;

  // IoU
  return intersection / union_;
}

template <typename T>
void index_sort_ascend(int *indexes, T *scores, int size, int stride_score) {
  std::sort(indexes, indexes + size, [&scores, stride_score](int i, int j) {
    return scores[i * stride_score] > scores[j * stride_score];
  });
}

template <typename T> T suppress_under_thresh(T value, T thresh) {
  return value < thresh ? (T)0 : value;
}

template <typename T>
void NmsDetection2d<T>::forward_impl_per_class(const Variables &inputs,
                                               const Variables &outputs) {

  const T *x = inputs[0]->get_data_pointer<T>(this->ctx_);
  T *y = outputs[0]->cast_data_and_get_pointer<T>(this->ctx_, true);

  auto sh = inputs[0]->shape();
  int num_b = sh[0];
  int num_nhw = sh[1];
  int num_bnhw = num_b * num_nhw;
  int num_c = sh[2];
  int num_classes = sh[2] - 5;

#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
  // Fill outputs first
  for (int i = 0; i < num_bnhw; ++i) {
    const T *xx = x + i * num_c;
    T *yy = y + i * num_c;
    yy[0] = xx[0]; // x
    yy[1] = xx[1]; // y
    yy[2] = xx[2]; // w
    yy[3] = xx[3]; // h
    T objectness = suppress_under_thresh(xx[4], thresh_);
    yy[4] = objectness;
    // Class probabilities, objectness * p.
    for (int k = 0; k < num_classes; k++) {
      yy[5 + k] = suppress_under_thresh(objectness * xx[5 + k], thresh_);
    }
  }

  CpuCachedArray indexes_arr(num_nhw * num_classes, get_dtype<int>(),
                             this->ctx_);
  int *indexes = indexes_arr.pointer<int>();

#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
  // NMS per class
  for (int k = 0; k < num_classes; k++) {
    for (int b = 0; b < num_b; b++) {
      // Initialize buffer for sort
      int *kindexes = indexes + k * num_nhw;
      for (int i = 0; i < num_nhw; ++i) {
        kindexes[i] = i;
      }
      // Sort indexes
      index_sort_ascend(kindexes, y + b * num_nhw * num_c + 5 + k, num_nhw,
                        num_c);

      // NMS
      for (int i = 0; i < num_nhw; ++i) {
        int offset = (b * num_nhw + kindexes[i]) * num_c;
        T prob = y[offset + 5 + k];
        if (prob == 0) {
          continue;
        }
        T *box = y + offset;
        for (int j = i + 1; j < num_nhw; ++j) {
          int offset2 = (b * num_nhw + kindexes[j]) * num_c;
          T &prob2 = y[offset2 + 5 + k]; // NOTE: as a reference.
          if (prob2 == 0) {
            continue;
          }
          T *box2 = y + offset2;
          T iou = calculate_iou(box, box2);
          if (iou > nms_) {
            prob2 = 0;
          }
        }
      }
    }
  }
}

template <typename T>
void NmsDetection2d<T>::forward_impl(const Variables &inputs,
                                     const Variables &outputs) {
  if (nms_per_class_) {
    forward_impl_per_class(inputs, outputs);
    return;
  }
  const T *x = inputs[0]->get_data_pointer<T>(this->ctx_);
  T *y = outputs[0]->cast_data_and_get_pointer<T>(this->ctx_, true);

  auto sh = inputs[0]->shape();
  int num_b = sh[0];
  int num_nhw = sh[1];
  int num_bnhw = num_b * num_nhw;
  int num_c = sh[2];
  int num_classes = sh[2] - 5;

#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
  // Fill outputs first
  for (int i = 0; i < num_bnhw; ++i) {
    const T *xx = x + i * num_c;
    T *yy = y + i * num_c;
    yy[0] = xx[0]; // x
    yy[1] = xx[1]; // y
    yy[2] = xx[2]; // w
    yy[3] = xx[3]; // h
    T objectness = suppress_under_thresh(xx[4], thresh_);
    yy[4] = objectness;
    // Class probabilities, objectness * p.
    for (int k = 0; k < num_classes; k++) {
      yy[5 + k] = suppress_under_thresh(objectness * xx[5 + k], thresh_);
    }
  }

  CpuCachedArray indexes_arr(num_nhw, get_dtype<int>(), this->ctx_);
  int *indexes = indexes_arr.pointer<int>();

  // Non-Maximum Suppression (Naive implementation)
  for (int b = 0; b < num_b; b++) {
    // Initialize buffer for sort
    for (int i = 0; i < num_nhw; ++i) {
      indexes[i] = i;
    }
    // Sort indexes
    index_sort_ascend(indexes, y + b * num_nhw * num_c + 4, num_nhw, num_c);
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (int i = num_nhw - 1; i >= 0; --i) {
      int offset = num_c * (b * num_nhw + indexes[i]);
      T &objectness = y[offset + 4]; // NOTE: as a reference.
      if (objectness == 0) {
        continue;
      }
      T *box = y + offset;
      for (int j = i - 1; j >= 0; --j) {
        T *box2 = y + num_c * (b * num_nhw + indexes[j]);
        T iou = calculate_iou(box, box2);
        if (iou > nms_) {
          objectness = 0;
          for (int k = 0; k < num_classes; ++k) {
            y[offset + 5 + k] = 0;
          }
        }
      }
    }
  }
}

template <typename T>
void NmsDetection2d<T>::backward_impl(const Variables &inputs,
                                      const Variables &outputs,
                                      const vector<bool> &propagate_down,
                                      const vector<bool> &accum) {}

} // namespace nbla
