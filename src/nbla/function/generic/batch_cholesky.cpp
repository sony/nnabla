// Copyright 2022 Sony Group Corporation.
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
#include <nbla/function/batch_cholesky.hpp>
#include <nbla/function/batch_inv.hpp>
#include <nbla/function/batch_matmul.hpp>
#include <nbla/function/mul_scalar.hpp>
#include <nbla/function/transpose.hpp>
#include <nbla/utils/eigen.hpp>
#include <nbla/variable.hpp>

namespace nbla {

NBLA_REGISTER_FUNCTION_SOURCE(BatchCholesky, bool);

template <typename T>
void BatchCholesky<T>::setup_impl(const Variables &inputs,
                                  const Variables &outputs) {
  auto x_shape = inputs.at(0)->shape();
  NBLA_CHECK(x_shape[1] == x_shape[2], error_code::value,
             "cholesky decomposition can only be applied to square matrix");
  outputs.at(0)->reshape(x_shape, true);

  batch_size_ = x_shape[0];
  dim_ = x_shape[1];
  offset_ = dim_ * dim_;
}

template <typename T>
void BatchCholesky<T>::forward_impl(const Variables &inputs,
                                    const Variables &outputs) {
  using namespace ::nbla::eigen;
  const T *x = inputs[0]->get_data_pointer<T>(this->ctx_);
  T *y = outputs[0]->cast_data_and_get_pointer<T>(this->ctx_, true);
  for (int i = 0; i < batch_size_; ++i) {
    ConstMatrixMap<T> mx(x + i * offset_, dim_, dim_);
    MatrixMap<T> my(y + i * offset_, dim_, dim_);
    if (upper_) {
      my = mx.llt().matrixU();
    } else {
      my = mx.llt().matrixL();
    }
  }
}

template <typename T>
void BatchCholesky<T>::backward_impl(const Variables &inputs,
                                     const Variables &outputs,
                                     const vector<bool> &propagate_down,
                                     const vector<bool> &accum) {
  if (!(propagate_down[0])) {
    return;
  }
  // For derivation see: https://arxiv.org/abs/1602.07527

  // Gradient of outputs
  Variable gy(outputs[0]->grad());
  Variable L(outputs[0]->data());

  // batch_inv = output^-1
  Variable L_inv(outputs[0]->shape());
  auto batch_inv = create_BatchInv(this->ctx_);
  batch_inv->setup(Variables{&L}, Variables{&L_inv});
  batch_inv->forward(Variables{&L}, Variables{&L_inv});

  Variable P(outputs[0]->shape());
  auto f_matmul1 = create_BatchMatmul(this->ctx_, !upper_, upper_);
  f_matmul1->setup(Variables{&L, &gy}, Variables{&P});
  f_matmul1->forward(Variables{&L, &gy}, Variables{&P});
  this->phi(P);

  Variable L_inv_T_P(outputs[0]->shape());
  auto f_matmul2 = create_BatchMatmul(this->ctx_, !upper_, false);
  f_matmul2->setup(Variables{&L_inv, &P}, Variables{&L_inv_T_P});
  f_matmul2->forward(Variables{&L_inv, &P}, Variables{&L_inv_T_P});

  Variable S(outputs[0]->shape());
  auto f_matmul3 = create_BatchMatmul(this->ctx_, false, upper_);
  f_matmul3->setup(Variables{&L_inv_T_P, &L_inv}, Variables{&S});
  f_matmul3->forward(Variables{&L_inv_T_P, &L_inv}, Variables{&S});

  Variable S_T(S.shape());
  auto f_transpose = create_Transpose(this->ctx_, vector<int>{0, 2, 1});
  f_transpose->setup(Variables{&S}, Variables{&S_T});
  f_transpose->forward(Variables{&S}, Variables{&S_T});

  Variable add_out(S.shape());
  auto f_add1 = create_Add2(this->ctx_, true);
  f_add1->setup(Variables{&S, &S_T}, Variables{&add_out});
  f_add1->forward(Variables{&S, &S_T}, Variables{&add_out});

  Variable mul_out(S.shape());
  auto f_mul_scalar = create_MulScalar(this->ctx_, 0.5, false);
  f_mul_scalar->setup(Variables{&add_out}, Variables{&mul_out});
  f_mul_scalar->forward(Variables{&add_out}, Variables{&mul_out});

  // Gradient of inputs
  Variable gx(inputs[0]->grad());

  if (!accum[0]) {
    // gx = add_out
    const Array *mul_ptr = mul_out.data()->get(get_dtype<T>(), this->ctx_);
    Array *gx_ptr = gx.data()->cast(get_dtype<T>(), this->ctx_, true);
    gx_ptr->copy_from(mul_ptr);
  } else {
    // gx = gx + add_out
    auto f_add2 = create_Add2(this->ctx_, true);
    f_add2->setup(Variables{&gx, &mul_out}, Variables{&gx});
    f_add2->forward(Variables{&gx, &mul_out}, Variables{&gx});
  }
}

template <typename T> void BatchCholesky<T>::phi(Variable &var) {
  auto var_shape = var.shape();
  auto batch_size = var_shape[0];
  auto rows = var_shape[1];
  auto cols = var_shape[2];
  auto matrix_size = rows * cols;

  // Output
  T *out = var.cast_data_and_get_pointer<T>(this->ctx_, true);
  for (int64_t batch = 0; batch < batch_size; ++batch) {
    T *out_matrix_ptr = out + batch * matrix_size;
    for (int64_t row = 0; row < rows; ++row) {
      for (int64_t col = 0; col < cols; ++col) {
        if (row == col) {
          out_matrix_ptr[row * rows + col] *= 0.5;
        } else if (row < col) {
          out_matrix_ptr[row * rows + col] = 0.0;
        } else {
          // Do nothing
        }
      }
    }
  }
}
}
