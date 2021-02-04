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

#include <cassert>
#include <nbla/array.hpp>
#include <nbla/common.hpp>
#include <nbla/function/gather_nd.hpp>
#include <nbla/utils/nd_index.hpp>
#include <nbla/variable.hpp>

namespace nbla {

NBLA_REGISTER_FUNCTION_SOURCE(GatherNd);

template <typename T>
void GatherNd<T>::setup_impl(const Variables &inputs,
                             const Variables &outputs) {
  auto srcdata = inputs.at(0);
  auto indices = inputs.at(1);
  auto outdata = outputs.at(0);

  auto srcdata_shape = srcdata->shape();
  auto indices_shape = indices->shape();

  NBLA_CHECK(indices_shape.size() >= 2, error_code::value,
             "gather_nd requires the map to have at least 2 dimensions");

  assert(indices_shape.at(0) >= 0);
  auto indices_at_zero = static_cast<Shape_t::size_type>(indices_shape.at(0));
  NBLA_CHECK(indices_at_zero <= srcdata_shape.size(), error_code::value,
             "Number of indices exceeds data dimension");

  Shape_t outdata_shape(indices_shape.size() - 1 + srcdata_shape.size() -
                        indices_shape[0]);

  std::copy(indices_shape.begin() + 1, indices_shape.end(),
            outdata_shape.begin());
  std::copy_n(srcdata_shape.begin() + indices_shape[0],
              srcdata_shape.size() - indices_shape[0],
              outdata_shape.begin() + indices_shape.size() - 1);

  outdata->reshape(outdata_shape, true);
}

template <typename T>
void GatherNd<T>::forward_impl(const Variables &inputs,
                               const Variables &outputs) {
  auto src = inputs[0]->get_data_pointer<T>(this->ctx_);
  auto dst = outputs[0]->cast_data_and_get_pointer<T>(this->ctx_, true);
  auto idx = inputs[1]->get_data_pointer<int>(this->ctx_);

  auto src_shape = inputs[0]->shape();
  auto idx_shape = inputs[1]->shape();

  auto src_ndi = ndi::make_index<Size_t>(src_shape.size());
  auto src_strides = ndi::strides(src_shape);

  auto idx_rows = idx_shape[0];
  auto idx_cols = ndi::inner_size(idx_shape, 1);

  for (int i = 0; i < idx_cols; i++) {
    for (int m = 0; m < idx_rows; m++) {
      auto index = idx[m * idx_cols + i];
      NBLA_CHECK(index < src_shape[m], error_code::value,
                 "index %d for axis %d overflows shape %d", index, m,
                 src_shape[m]);
      NBLA_CHECK(index >= -src_shape[m], error_code::value,
                 "index %d for axis %d underflows shape %d", index, m,
                 src_shape[m]);
      src_ndi[m] = (index < 0) ? src_shape[m] + index : index;
    }
    auto slice_length = src_strides.at(idx_rows - 1);
    auto slice_offset = ndi::nd2flat(src_ndi, src_strides);
    for (int k = 0; k < slice_length; k++) {
      dst[i * slice_length + k] = src[slice_offset + k];
    }
  }
}

template <typename T>
void GatherNd<T>::backward_impl(const Variables &inputs,
                                const Variables &outputs,
                                const vector<bool> &propagate_down,
                                const vector<bool> &accum) {
  if (!propagate_down[0]) {
    return;
  }

  if (!accum[0]) {
    inputs[0]->grad()->zero();
  }

  auto g_y = outputs[0]->get_grad_pointer<T>(this->ctx_);
  auto g_x = inputs[0]->cast_grad_and_get_pointer<T>(this->ctx_, false);
  auto idx = inputs[1]->get_data_pointer<int>(this->ctx_);

  auto src_shape = inputs[0]->shape();
  auto idx_shape = inputs[1]->shape();

  auto src_ndi = ndi::make_index<Size_t>(src_shape.size());
  auto src_strides = ndi::strides(src_shape);

  auto idx_rows = idx_shape[0];
  auto idx_cols = ndi::inner_size(idx_shape, 1);

  for (int i = 0; i < idx_cols; i++) {
    for (int m = 0; m < idx_rows; m++) {
      auto index = idx[m * idx_cols + i];
      NBLA_CHECK(index < src_shape[m], error_code::value,
                 "index %d for axis %d overflows shape %d", index, m,
                 src_shape[m]);
      NBLA_CHECK(index >= -src_shape[m], error_code::value,
                 "index %d for axis %d underflows shape %d", index, m,
                 src_shape[m]);
      src_ndi[m] = (index < 0) ? src_shape[m] + index : index;
    }
    auto slice_length = src_strides.at(idx_rows - 1);
    auto slice_offset = ndi::nd2flat(src_ndi, src_strides);
    for (int k = 0; k < slice_length; k++) {
      g_x[slice_offset + k] += g_y[i * slice_length + k];
    }
  }
}
}
