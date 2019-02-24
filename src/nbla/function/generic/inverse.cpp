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
#include <nbla/function/batch_matmul.hpp>
#include <nbla/function/add2.hpp>
#include <nbla/function/mul_scalar.hpp>
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
  batch_size_ = input_shape[0];
  dim_ = input_shape[1];
  offset_ = dim_ * dim_;

  // for backward
  inv_x_ = make_shared<Variable>(outputs[0]->data());
  neg_inv_x_ = make_shared<Variable>();
  matmul1_out_ = make_shared<Variable>();
  matmul2_out_ = make_shared<Variable>();
  gy_ = make_shared<Variable>(outputs[0]->grad());
  gx_ = make_shared<Variable>(inputs[0]->grad());
  gx_accum_ = make_shared<Variable>();

  f_mul_scalar = create_MulScalar(this->ctx_, -1.0);
  f_mul_scalar->setup(Variables{inv_x_.get()}, Variables{neg_inv_x_.get()});

  f_batch_matmul1_ = create_BatchMatmul(this->ctx_, true, false);
  f_batch_matmul1_->setup(Variables{neg_inv_x_.get(), gy_.get()},
                          Variables{matmul1_out_.get()});

  f_batch_matmul2_ = create_BatchMatmul(this->ctx_, false, true);
  f_batch_matmul2_->setup(Variables{matmul1_out_.get(), inv_x_.get()},
                          Variables{matmul2_out_.get()});

  f_add_ = create_Add2(this->ctx_, false);
  f_add_->setup(Variables{gx_.get(), matmul2_out_.get()},
                Variables{gx_accum_.get()});
}

template <typename T>
void Inverse<T>::forward_impl(const Variables &inputs,
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
void Inverse<T>::backward_impl(const Variables &inputs,
                                const Variables &outputs,
                                const vector<bool> &propagate_down,
                                const vector<bool> &accum) {
  if (!propagate_down[0]) {
    return;
  }
  if (!accum[0]) {
    gx_->data()->zero();
  }
  // gx_ += -inv_x^T * gy * inv_x^T
  f_mul_scalar->forward(Variables{inv_x_.get()}, Variables{neg_inv_x_.get()});
  f_batch_matmul1_->forward(Variables{neg_inv_x_.get(), gy_.get()},
                            Variables{matmul1_out_.get()});
  f_batch_matmul2_->forward(Variables{matmul1_out_.get(), inv_x_.get()},
                            Variables{matmul2_out_.get()});
  f_add_->forward(Variables{gx_.get(), matmul2_out_.get()},
                  Variables{gx_accum_.get()});
  // TODO: Remove this line
  inputs[0]->grad()->cast(get_dtype<T>(), this->ctx_, true)->copy_from(gx_accum_->data()->get(get_dtype<T>(), this->ctx_));
}
}
