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

/** BatchMatmul
 */
#include <nbla/array.hpp>
#include <nbla/function/batch_matmul.hpp>
#include <nbla/utils/eigen.hpp>
#include <nbla/variable.hpp>

namespace nbla {

NBLA_REGISTER_FUNCTION_SOURCE(BatchMatmul, bool, bool);

template <typename T>
void BatchMatmul<T>::setup_impl(const Variables &inputs,
                                const Variables &outputs) {
  auto a = inputs[0];
  auto b = inputs[1];
  auto shape_a = a->shape();
  auto shape_b = b->shape();
  NBLA_CHECK(a->ndim() >= 2, error_code::value,
             "Input must be >=2 dimensions. (Given %ld at input 0)", a->ndim());
  NBLA_CHECK(b->ndim() >= 2, error_code::value,
             "Input must be >=2 dimensions. (Given %ld at input 1)", b->ndim());
  row_a_ = shape_a[a->ndim() - 2];
  col_a_ = shape_a[a->ndim() - 1];
  row_b_ = shape_b[b->ndim() - 2];
  col_b_ = shape_b[b->ndim() - 1];
  row_y_ = transpose_a_ ? col_a_ : row_a_;
  col_y_ = transpose_b_ ? row_b_ : col_b_;
  offset_a_ = row_a_ * col_a_;
  offset_b_ = row_b_ * col_b_;
  offset_y_ = row_y_ * col_y_;
  samples_ = 1;
  for (int d = 0; d < a->ndim() - 2; ++d) {
    samples_ *= shape_a[d];
  }
  int samples_b = 1;
  for (int d = 0; d < b->ndim() - 2; ++d) {
    samples_b *= shape_b[d];
  }
  NBLA_CHECK(samples_ == samples_b, error_code::value,
             "Inconsistent batch samples (%d != %d)", samples_, samples_b);
  int reduction_dim_a = a->ndim() - (transpose_a_ ? 2 : 1);
  int reduction_dim_b = b->ndim() - (transpose_b_ ? 1 : 2);
  int reduction_size_a = shape_a[reduction_dim_a];
  int reduction_size_b = shape_b[reduction_dim_b];
  NBLA_CHECK(reduction_size_a == reduction_size_b, error_code::value,
             "Reduction sizes mismatch (%d !=%d).", reduction_size_a,
             reduction_size_b);
  Shape_t shape_o(shape_a.begin(), shape_a.end() - 2);
  shape_o.push_back(row_y_);
  shape_o.push_back(col_y_);
  outputs[0]->reshape(shape_o, true);
}

template <typename T>
void BatchMatmul<T>::forward_impl(const Variables &inputs,
                                  const Variables &outputs) {
  using namespace ::nbla::eigen;
  const T *a = inputs[0]->get_data_pointer<T>(this->ctx_);
  const T *b = inputs[1]->get_data_pointer<T>(this->ctx_);
  T *y = outputs[0]->cast_data_and_get_pointer<T>(this->ctx_);
  for (int s = 0; s < samples_; ++s) {
    ConstMatrixMap<T> ma(a + s * offset_a_, row_a_, col_a_);
    ConstMatrixMap<T> mb(b + s * offset_b_, row_b_, col_b_);
    MatrixMap<T> my(y + s * offset_y_, row_y_, col_y_);
    NBLA_EIGEN_MATMUL_T(my, false, ma, transpose_a_, mb, transpose_b_, =);
  }
}

template <typename T>
void BatchMatmul<T>::backward_impl(const Variables &inputs,
                                   const Variables &outputs,
                                   const vector<bool> &propagate_down,
                                   const vector<bool> &accum) {
  if (!(propagate_down[0] || propagate_down[1])) {
    return;
  }
  using namespace ::nbla::eigen;
  const T *dy = outputs[0]->get_grad_pointer<T>(this->ctx_);
  if (propagate_down[0]) {
    const T *b = inputs[1]->get_data_pointer<T>(this->ctx_);
    T *da = inputs[0]->cast_grad_and_get_pointer<T>(this->ctx_);
    for (int s = 0; s < samples_; ++s) {
      MatrixMap<T> mda(da + s * offset_a_, row_a_, col_a_);
      ConstMatrixMap<T> mb(b + s * offset_b_, row_b_, col_b_);
      ConstMatrixMap<T> mdy(dy + s * offset_y_, row_y_, col_y_);
      if (accum[0]) {
        NBLA_EIGEN_MATMUL_T(mda, transpose_a_, mdy, false, mb, !transpose_b_,
                            +=);
      } else {
        NBLA_EIGEN_MATMUL_T(mda, transpose_a_, mdy, false, mb, !transpose_b_,
                            =);
      }
    }
  }
  if (propagate_down[1]) {
    const T *a = inputs[0]->get_data_pointer<T>(this->ctx_);
    T *db = inputs[1]->cast_grad_and_get_pointer<T>(this->ctx_);
    for (int s = 0; s < samples_; ++s) {
      ConstMatrixMap<T> ma(a + s * offset_a_, row_a_, col_a_);
      MatrixMap<T> mdb(db + s * offset_b_, row_b_, col_b_);
      ConstMatrixMap<T> mdy(dy + s * offset_y_, row_y_, col_y_);
      if (accum[1]) {
        NBLA_EIGEN_MATMUL_T(mdb, transpose_b_, ma, !transpose_a_, mdy, false,
                            +=);
      } else {
        NBLA_EIGEN_MATMUL_T(mdb, transpose_b_, ma, !transpose_a_, mdy, false,
                            =);
      }
    }
  }
}

} // namespace nbla
