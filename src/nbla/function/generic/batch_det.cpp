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
#include <nbla/function/add2.hpp>
#include <nbla/function/batch_det.hpp>
#include <nbla/function/batch_inv.hpp>
#include <nbla/function/mul2.hpp>
#include <nbla/function/transpose.hpp>
#include <nbla/utils/eigen.hpp>
#include <nbla/variable.hpp>

namespace nbla {

NBLA_REGISTER_FUNCTION_SOURCE(BatchDet);

template <typename T>
void BatchDet<T>::setup_impl(const Variables &inputs,
                             const Variables &outputs) {
  NBLA_CHECK(inputs[0]->ndim() == 3, error_code::value,
             "Input must be 2D array");
  auto input_shape = inputs[0]->shape();
  NBLA_CHECK(input_shape[1] == input_shape[2], error_code::value,
             "Input must be square matrix");

  batch_size_ = input_shape[0];
  dim_ = input_shape[1];
  offset_ = dim_ * dim_;

  outputs[0]->reshape(Shape_t{batch_size_}, true);
}

template <typename T>
void BatchDet<T>::forward_impl(const Variables &inputs,
                               const Variables &outputs) {
  using namespace ::nbla::eigen;
  const T *x = inputs[0]->get_data_pointer<T>(this->ctx_);
  T *y = outputs[0]->cast_data_and_get_pointer<T>(this->ctx_, true);
  for (int i = 0; i < batch_size_; ++i) {
    ConstMatrixMap<T> mx(x + i * offset_, dim_, dim_);
    y[i] = mx.determinant();
  }
}

template <typename T>
void BatchDet<T>::backward_impl(const Variables &inputs,
                                const Variables &outputs,
                                const vector<bool> &propagate_down,
                                const vector<bool> &accum) {
  if (!propagate_down[0]) {
    return;
  }

  Variable gx(inputs[0]->grad());
  Variable reshaped_gy(Shape_t{batch_size_, 1, 1});
  Variable inv_x(inputs[0]->shape());
  Variable transposed_inv_x(inv_x.data()->shape());
  Variable reshaped_det_x(Shape_t{batch_size_, 1, 1});
  Variable mul1_out(reshaped_det_x.data()->shape());
  Variable mul2_out(mul1_out.data()->shape());

  reshaped_gy.data()->set_array(outputs[0]->grad()->array());
  reshaped_det_x.data()->set_array(outputs[0]->data()->array());

  // gx += gy * det_x * inv_x^T (element-wise multiplication)

  // batch_inv = input^-1
  auto f_batch_inv = create_BatchInv(this->ctx_);
  f_batch_inv->setup(inputs, Variables{&inv_x});
  f_batch_inv->forward(inputs, Variables{&inv_x});

  // mul1_out = gy * det_x = gy * output
  auto f_mul1 = create_Mul2(this->ctx_, false);
  f_mul1->setup(Variables{&reshaped_gy, &reshaped_det_x}, Variables{&mul1_out});
  f_mul1->forward(Variables{&reshaped_gy, &reshaped_det_x},
                  Variables{&mul1_out});

  // inv_x^T
  auto f_transpose = create_Transpose(this->ctx_, vector<int>{0, 2, 1});
  f_transpose->setup(Variables{&inv_x}, Variables{&transposed_inv_x});
  f_transpose->forward(Variables{&inv_x}, Variables{&transposed_inv_x});

  // mul2_out = mul1_out * inv_x^T
  auto f_mul2 = create_Mul2(this->ctx_, false);
  f_mul2->setup(Variables{&mul1_out, &transposed_inv_x}, Variables{&mul2_out});
  f_mul2->forward(Variables{&mul1_out, &transposed_inv_x},
                  Variables{&mul2_out});

  if (!accum[0]) {
    // gx = mul2_out
    const Array *mul2_ptr = mul2_out.data()->get(get_dtype<T>(), this->ctx_);
    Array *gx_ptr = gx.data()->cast(get_dtype<T>(), this->ctx_, true);
    gx_ptr->copy_from(mul2_ptr);
  } else {
    // gx = gx + mul2_out
    auto f_add = create_Add2(this->ctx_, true);
    f_add->setup(Variables{&gx, &mul2_out}, Variables{&gx});
    f_add->forward(Variables{&gx, &mul2_out}, Variables{&gx});
  }
}
}
