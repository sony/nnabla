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
#include <nbla/function/batch_inv.hpp>
#include <nbla/function/batch_matmul.hpp>
#include <nbla/function/add2.hpp>
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
  const T* x = inputs[0]->get_data_pointer<T>(this->ctx_);
  T* y = outputs[0]->cast_data_and_get_pointer<T>(this->ctx_, true);
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

  auto inv_x = make_shared<Variable>(outputs[0]->data());
  auto neg_inv_x = make_shared<Variable>();
  auto f_mul_scalar = create_MulScalar(this->ctx_, -1.0);
  f_mul_scalar->setup(Variables{inv_x.get()}, Variables{neg_inv_x.get()});

  auto matmul1_out = make_shared<Variable>();
  auto gy = make_shared<Variable>(outputs[0]->grad());
  auto f_batch_matmul1 = create_BatchMatmul(this->ctx_, true, false);
  f_batch_matmul1->setup(Variables{neg_inv_x.get(), gy.get()},
                          Variables{matmul1_out.get()});

  auto matmul2_out = make_shared<Variable>();
  auto f_batch_matmul2 = create_BatchMatmul(this->ctx_, false, true);
  f_batch_matmul2->setup(Variables{matmul1_out.get(), inv_x.get()},
                          Variables{matmul2_out.get()});

  auto gx = make_shared<Variable>(inputs[0]->grad());
  auto f_add = create_Add2(this->ctx_, true);
  f_add->setup(Variables{gx.get(), matmul2_out.get()},
                Variables{gx.get()});

  if (!accum[0])
    gx->data()->zero();

  // gx += -inv_x^T * gy * inv_x^T
  f_mul_scalar->forward(Variables{inv_x.get()}, Variables{neg_inv_x.get()});
  f_batch_matmul1->forward(Variables{neg_inv_x.get(), gy.get()},
                            Variables{matmul1_out.get()});
  f_batch_matmul2->forward(Variables{matmul1_out.get(), inv_x.get()},
                            Variables{matmul2_out.get()});
  f_add->forward(Variables{gx.get(), matmul2_out.get()},
                  Variables{gx.get()});
}
}
