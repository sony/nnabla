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
#include <nbla/function/scatter_nd.hpp>
#include <nbla/utils/nd_index.hpp>
#include <nbla/variable.hpp>

namespace nbla {

NBLA_REGISTER_FUNCTION_SOURCE(ScatterNd, const vector<int> &);

template <typename T>
void ScatterNd<T>::setup_impl(const Variables &inputs,
                              const Variables &outputs) {
  auto data = inputs.at(0);
  auto indices = inputs.at(1);

  NBLA_CHECK(indices->ndim() >= 2, error_code::value,
             "scatter_nd requires indices to have at least 2 dimensions");

  NBLA_CHECK(static_cast<Shape_t::size_type>(indices->shape().at(0)) <=
                 shape_.size(),
             error_code::value, "Number of indices exceeds output dimension");

  auto N = data->ndim();
  auto M = indices->shape().at(0);
  auto K = indices->ndim() - 1;

  NBLA_CHECK(shape_.size() == static_cast<Shape_t::size_type>(N + M - K),
             error_code::value,
             "Output shape size does not match input data and indices.");

  for (int i = 0; i < K; i++) {
    NBLA_CHECK(data->shape().at(i) == indices->shape().at(i + 1),
               error_code::value,
               "Shape error: data shape[%d] %d != indices shape[%d] %d", i,
               data->shape()[i], i + 1, indices->shape()[i + 1]);
  }

  for (int i = 0; static_cast<Shape_t::size_type>(i) < shape_.size() - M; i++) {
    NBLA_CHECK(data->shape().at(K + i) == shape_.at(M + i), error_code::value,
               "Shape error: data shape[%d] %d != output shape[%d] %d", K + i,
               data->shape()[K + i], M + i, indices->shape()[M + i]);
  }

  Shape_t output_shape(N + M - K);
  std::copy(shape_.begin(), shape_.end(), output_shape.begin());
  outputs.at(0)->reshape(output_shape, true);

  if (inputs.size() > 2) {
    outputs[0]->data()->set_array(inputs[2]->data()->array());
  }
}

template <typename T>
void ScatterNd<T>::forward_impl(const Variables &inputs,
                                const Variables &outputs) {
  if (inputs.size() < 3)
    outputs[0]->data()->zero();

  auto src = inputs[0]->get_data_pointer<T>(this->ctx_);
  auto idx = inputs[1]->get_data_pointer<int>(this->ctx_);
  auto dst = outputs[0]->cast_data_and_get_pointer<T>(this->ctx_, false);

  auto idx_shape = inputs[1]->shape();
  auto dst_shape = outputs[0]->shape();

  auto dst_ndi = ndi::make_index<Size_t>(dst_shape.size());
  auto dst_strides = ndi::strides(dst_shape);

  auto idx_rows = idx_shape[0];
  auto idx_cols = ndi::inner_size(idx_shape, 1);

  for (int i = 0; i < idx_cols; i++) {
    for (int m = 0; m < idx_rows; m++) {
      auto index = idx[m * idx_cols + i];
      NBLA_CHECK(index < dst_shape[m], error_code::value,
                 "index %d for axis %d overflows shape %d", index, m,
                 dst_shape[m]);
      NBLA_CHECK(index >= -dst_shape[m], error_code::value,
                 "index %d for axis %d underflows shape %d", index, m,
                 dst_shape[m]);
      dst_ndi[m] = (index < 0) ? dst_shape[m] + index : index;
    }
    auto slice_length = dst_strides.at(idx_rows - 1);
    auto slice_offset = ndi::nd2flat(dst_ndi, dst_strides);
    for (int k = 0; k < slice_length; k++) {
      dst[slice_offset + k] = src[i * slice_length + k];
    }
  }
}

template <typename T>
void ScatterNd<T>::backward_impl(const Variables &inputs,
                                 const Variables &outputs,
                                 const vector<bool> &propagate_down,
                                 const vector<bool> &accum) {
  if (!propagate_down[0]) {
    return;
  }

  auto g_x = inputs[0]->cast_grad_and_get_pointer<T>(this->ctx_, !accum[0]);
  auto idx = inputs[1]->get_data_pointer<int>(this->ctx_);

  auto idx_shape = inputs[1]->shape();
  auto src_shape = outputs[0]->shape();

  auto src_ndi = ndi::make_index<Size_t>(src_shape.size());
  auto src_strides = ndi::strides(src_shape);

  auto idx_rows = idx_shape[0];
  auto idx_cols = ndi::inner_size(idx_shape, 1);

  if (inputs.size() < 3) {
    // Because input[0] data is scattered into a new output variable during
    // forward, output[0] gradient values from scatter indices are propagated
    // back to input[0] gradient.
    auto g_y = outputs[0]->get_grad_pointer<T>(this->ctx_);
    for (int i = 0; i < idx_cols; i++) {
      for (int m = 0; m < idx_rows; m++) {
        auto index = idx[m * idx_cols + i];
        src_ndi[m] = (index < 0) ? src_shape[m] + index : index;
      }
      auto slice_length = src_strides.at(idx_rows - 1);
      auto slice_offset = ndi::nd2flat(src_ndi, src_strides);
      for (int k = 0; k < slice_length; k++) {
        g_x[i * slice_length + k] =
            !accum[0] ? g_y[slice_offset + k]
                      : g_x[i * slice_length + k] + g_y[slice_offset + k];
      }
    }
  } else {
    // Because input[0] data is scattered into the data of input[2] (the input
    // parameter named `out`) inplaced with output[0], the gradient values of
    // output[0] that belong to the scatter indices are propagated back to the
    // input[0] gradient and set to 0 (masked) in the grad array of output[0].
    auto g_y = outputs[0]->cast_grad_and_get_pointer<T>(this->ctx_);
    for (int i = 0; i < idx_cols; i++) {
      for (int m = 0; m < idx_rows; m++) {
        auto index = idx[m * idx_cols + i];
        src_ndi[m] = (index < 0) ? src_shape[m] + index : index;
      }
      auto slice_length = src_strides.at(idx_rows - 1);
      auto slice_offset = ndi::nd2flat(src_ndi, src_strides);
      for (int k = 0; k < slice_length; k++) {
        g_x[i * slice_length + k] =
            !accum[0] ? g_y[slice_offset + k]
                      : g_x[i * slice_length + k] + g_y[slice_offset + k];
        g_y[slice_offset + k] = T(0);
      }
    }
  }
}
}
