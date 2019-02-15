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
#include <nbla/function/inverse.hpp>
#include <nbla/utils/eigen.hpp>
#include <nbla/variable.hpp>



namespace nbla {

NBLA_REGISTER_FUNCTION_SOURCE(Inverse);


template <typename T>
void Inverse<T>::setup_impl(const Variables &inputs,
                               const Variables &outputs) {
  NBLA_CHECK(inputs[0]->ndim() == 3, error_code::value,
             "Input must be 2D array");
  auto input_shape = inputs[0]->shape();
  NBLA_CHECK(input_shape[1] == input_shape[2], error_code::value,
             "Input must be square matrix");
  outputs[0]->reshape(input_shape, true);
  dim_ = input_shape[1];
  offset_ = dim_ * dim_;
}

template <typename T>
void Inverse<T>::forward_impl(const Variables &inputs,
                               const Variables &outputs) {
  using namespace ::nbla::eigen;
  const T* x = inputs[0]->get_data_pointer<T>(this->ctx_);
  T* y = outputs[0]->cast_data_and_get_pointer<T>(this->ctx_, true);
  for (int i = 0; i < inputs[0]->shape()[0]; ++i) {
    ConstMatrixMap<T> mx(x + i * offset_, dim_, dim_);
    MatrixMap<T> my(y + i * offset_, dim_, dim_);
    auto inv_x = mx.inverse();
    my = inv_x;
  }
}

template <typename T>
void Inverse<T>::backward_impl(const Variables &inputs,
                                const Variables &outputs,
                                const vector<bool> &propagate_down,
                                const vector<bool> &accum) {
  if (!propagate_down[0]) {
    return;
  }
  using namespace ::nbla::eigen;
  const T* dy = outputs[0]->get_grad_pointer<T>(this->ctx_);
  const T* x = inputs[0]->get_data_pointer<T>(this->ctx_);
  T* dx = inputs[0]->cast_grad_and_get_pointer<T>(this->ctx_, !accum[0]);
  for (int i = 0; i < inputs[0]->shape()[0]; ++i) {
    ConstMatrixMap<T> mx(x + i * offset_, dim_, dim_);
    ConstMatrixMap<T> mdy(dy + i * offset_, dim_, dim_);
    auto inv_mx = mx.inverse();
    auto inv_mxT = inv_mx.transpose();
    MatrixMap<T> mdx(dx + i * offset_, dim_, dim_);
    auto g = -inv_mxT * mdy * inv_mxT;
    if (accum[0]) {
      mdx += g;
    } else {
      mdx = g;
    }
  }
}
}
