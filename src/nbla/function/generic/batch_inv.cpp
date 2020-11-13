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
#include <nbla/function/batch_inv.hpp>
#include <nbla/function/batch_matmul.hpp>
#include <nbla/function/mul_scalar.hpp>
#include <nbla/utils/eigen.hpp>
#include <nbla/variable.hpp>

namespace nbla {

NBLA_REGISTER_FUNCTION_SOURCE(BatchInv);

template <typename T>
void BatchInv<T>::setup_impl(const Variables &inputs,
                             const Variables &outputs) {
  NBLA_CHECK(inputs[0]->ndim() == 3, error_code::value,
             "Input must be 2D array");
  auto input_shape = inputs[0]->shape();
  NBLA_CHECK(input_shape[1] == input_shape[2], error_code::value,
             "Input must be square matrix");
  outputs[0]->reshape(input_shape, true);
  batch_size_ = input_shape[0];
  dim_ = input_shape[1];
  offset_ = dim_ * dim_;
}

template <typename T>
void BatchInv<T>::forward_impl(const Variables &inputs,
                               const Variables &outputs) {
  using namespace ::nbla::eigen;
  const T *x = inputs[0]->get_data_pointer<T>(this->ctx_);
  T *y = outputs[0]->cast_data_and_get_pointer<T>(this->ctx_, true);
  for (int i = 0; i < batch_size_; ++i) {
    ConstMatrixMap<T> mx(x + i * offset_, dim_, dim_);
    MatrixMap<T> my(y + i * offset_, dim_, dim_);
    auto inv_x = mx.inverse();
    my = inv_x;
  }
}

template <typename T>
void BatchInv<T>::backward_impl(const Variables &inputs,
                                const Variables &outputs,
                                const vector<bool> &propagate_down,
                                const vector<bool> &accum) {
  if (!propagate_down[0]) {
    return;
  }

  Variable gx(inputs[0]->grad());
  Variable gy(outputs[0]->grad());
  Variable inv_x(outputs[0]->data());
  Variable neg_inv_x(inv_x.data()->shape());
  Variable matmul1_out(inv_x.data()->shape());
  Variable matmul2_out(inv_x.data()->shape());

  // gx = -inv_x^T * gy * inv_x^T

  // neg_inv_x = -inv_x = inv_x * -1
  auto f_mul_scalar = create_MulScalar(this->ctx_, -1.0, false);
  f_mul_scalar->setup(Variables{&inv_x}, Variables{&neg_inv_x});
  f_mul_scalar->forward(Variables{&inv_x}, Variables{&neg_inv_x});

  // matmul1_out = neg_inv_x^T * gy
  auto f_batch_matmul1 = create_BatchMatmul(this->ctx_, true, false);
  f_batch_matmul1->setup(Variables{&neg_inv_x, &gy}, Variables{&matmul1_out});
  f_batch_matmul1->forward(Variables{&neg_inv_x, &gy}, Variables{&matmul1_out});

  // matmul2_out = matmul1_out * inv_x^T
  auto f_batch_matmul2 = create_BatchMatmul(this->ctx_, false, true);
  f_batch_matmul2->setup(Variables{&matmul1_out, &inv_x},
                         Variables{&matmul2_out});
  f_batch_matmul2->forward(Variables{&matmul1_out, &inv_x},
                           Variables{&matmul2_out});

  if (!accum[0]) {
    // gx = matmul2_out
    const Array *matmul2_ptr =
        matmul2_out.data()->get(get_dtype<T>(), this->ctx_);
    Array *gx_ptr = gx.data()->cast(get_dtype<T>(), this->ctx_, true);
    gx_ptr->copy_from(matmul2_ptr);
  } else {
    // gx = gx + matmul_2_out
    auto f_add = create_Add2(this->ctx_, true);
    f_add->setup(Variables{&gx, &matmul2_out}, Variables{&gx});
    f_add->forward(Variables{&gx, &matmul2_out}, Variables{&gx});
  }
}
}
