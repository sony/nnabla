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
#include <nbla/imperative.hpp>
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
  NBLA_CHECK(a->ndim() >= 3, error_code::value,
             "Input must be >=3 dimensions. (Given %ld at input 0)", a->ndim());
  NBLA_CHECK(b->ndim() >= 3, error_code::value,
             "Input must be >=3 dimensions. (Given %ld at input 1)", b->ndim());
  row_a_ = shape_a[a->ndim() - 2];
  col_a_ = shape_a[a->ndim() - 1];
  row_b_ = shape_b[b->ndim() - 2];
  col_b_ = shape_b[b->ndim() - 1];
  row_y_ = transpose_a_ ? col_a_ : row_a_;
  col_y_ = transpose_b_ ? row_b_ : col_b_;
  offset_a_ = row_a_ * col_a_;
  offset_b_ = row_b_ * col_b_;
  offset_y_ = row_y_ * col_y_;

  int reduction_dim_a = a->ndim() - (transpose_a_ ? 2 : 1);
  int reduction_dim_b = b->ndim() - (transpose_b_ ? 1 : 2);
  int reduction_size_a = shape_a[reduction_dim_a];
  int reduction_size_b = shape_b[reduction_dim_b];
  NBLA_CHECK(reduction_size_a == reduction_size_b, error_code::value,
             "Reduction sizes mismatch (%d !=%d).", reduction_size_a,
             reduction_size_b);
  NBLA_CHECK(a->ndim() == b->ndim(), error_code::value,
             "ndim of inputs[0] (%d) must be ndim of inputs[1] (%d).",
             a->ndim(), b->ndim());
  Shape_t shape_o;
  vector<int> shape_a_broadcast;
  vector<int> shape_b_broadcast;
  bool broadcast_a = false;
  bool broadcast_b = false;
  for (int d = 0; d < a->ndim() - 2; d++) {
    NBLA_CHECK(
        shape_a[d] == shape_b[d] || shape_a[d] == 1 || shape_b[d] == 1,
        error_code::value,
        "The following condition must hold\n"
        "shape[%d] of inputs[0] (%d) == shape[%d] of inputs[1] (%d) or \n"
        "shape[%d] of inputs[0] (%d) == 1 or \n"
        "shape[%d] of inputs[1] (%d) == 1.",
        d, shape_a[d], d, shape_b[d], d, shape_a[d], d, shape_b[d]);
    if (shape_a[d] != 1 && shape_b[d] == 1) {
      broadcast_b = true;
      shape_a_broadcast.push_back(shape_a[d]);
      shape_b_broadcast.push_back(shape_a[d]);
      shape_o.push_back(shape_a[d]);
    } else if (shape_a[d] == 1 && shape_b[d] != 1) {
      broadcast_a = true;
      shape_a_broadcast.push_back(shape_b[d]);
      shape_b_broadcast.push_back(shape_b[d]);
      shape_o.push_back(shape_b[d]);
    } else {
      shape_a_broadcast.push_back(shape_a[d]);
      shape_b_broadcast.push_back(shape_b[d]);
      shape_o.push_back(shape_a[d]);
    }
  }
  shape_a_broadcast.push_back(shape_a[a->ndim() - 2]);
  shape_a_broadcast.push_back(shape_a[a->ndim() - 1]);
  shape_b_broadcast.push_back(shape_b[b->ndim() - 2]);
  shape_b_broadcast.push_back(shape_b[b->ndim() - 1]);
  f_broadcast_a_ =
      broadcast_a ? create_Broadcast(ctx_, shape_a_broadcast) : nullptr;
  f_broadcast_b_ =
      broadcast_b ? create_Broadcast(ctx_, shape_b_broadcast) : nullptr;
  samples_ = std::accumulate(shape_o.begin(), shape_o.end(), 1,
                             std::multiplies<int>());
  shape_o.push_back(row_y_);
  shape_o.push_back(col_y_);
  outputs[0]->reshape(shape_o, true);
}

template <typename T>
void BatchMatmul<T>::forward_impl(const Variables &inputs,
                                  const Variables &outputs) {
  using namespace ::nbla::eigen;
  auto _get_data = [this](Variable *v) {
    return v->get_data_pointer<T>(this->ctx_);
  };
  // Broadcast
  Variable a_broadcast;
  Variable b_broadcast;
  if (f_broadcast_a_)
    execute(f_broadcast_a_, {inputs[0]}, {&a_broadcast});
  if (f_broadcast_b_)
    execute(f_broadcast_b_, {inputs[1]}, {&b_broadcast});
  // BMM
  const T *a = f_broadcast_a_ ? _get_data(&a_broadcast) : _get_data(inputs[0]);
  const T *b = f_broadcast_b_ ? _get_data(&b_broadcast) : _get_data(inputs[1]);
  T *y = outputs[0]->cast_data_and_get_pointer<T>(this->ctx_, true);
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
  auto _get_data = [this](Variable *v) {
    return v->get_data_pointer<T>(this->ctx_);
  };
  auto _cast_grad = [this](Variable *v, bool wo = true) {
    return v->cast_grad_and_get_pointer<T>(this->ctx_, wo);
  };

  const T *dy = outputs[0]->get_grad_pointer<T>(this->ctx_);
  if (propagate_down[0]) {
    // Broadcast
    Variable a_broadcast;
    Variable b_broadcast;
    if (f_broadcast_a_)
      execute(f_broadcast_a_, {inputs[0]}, {&a_broadcast});
    if (f_broadcast_b_)
      execute(f_broadcast_b_, {inputs[1]}, {&b_broadcast});
    // BMM backward
    const T *b =
        f_broadcast_b_ ? _get_data(&b_broadcast) : _get_data(inputs[1]);
    T *da = f_broadcast_a_ ? _cast_grad(&a_broadcast)
                           : _cast_grad(inputs[0], !accum[0]);
    for (int s = 0; s < samples_; ++s) {
      MatrixMap<T> mda(da + s * offset_a_, row_a_, col_a_);
      ConstMatrixMap<T> mb(b + s * offset_b_, row_b_, col_b_);
      ConstMatrixMap<T> mdy(dy + s * offset_y_, row_y_, col_y_);
      if (accum[0] && !f_broadcast_a_) {
        NBLA_EIGEN_MATMUL_T(mda, transpose_a_, mdy, false, mb, !transpose_b_,
                            +=);
      } else {
        NBLA_EIGEN_MATMUL_T(mda, transpose_a_, mdy, false, mb, !transpose_b_,
                            =);
      }
    }
    // Broadcast backward
    if (f_broadcast_a_)
      nbla::backward(f_broadcast_a_, {inputs[0]}, {&a_broadcast}, {true},
                     {accum[0]});
  }
  if (propagate_down[1]) {
    // Broadcast
    Variable a_broadcast;
    Variable b_broadcast;
    if (f_broadcast_a_)
      execute(f_broadcast_a_, {inputs[0]}, {&a_broadcast});
    if (f_broadcast_b_)
      execute(f_broadcast_b_, {inputs[1]}, {&b_broadcast});
    // BMM backward
    const T *a =
        f_broadcast_a_ ? _get_data(&a_broadcast) : _get_data(inputs[0]);
    T *db = (f_broadcast_b_) ? _cast_grad(&b_broadcast)
                             : _cast_grad(inputs[1], !accum[1]);
    for (int s = 0; s < samples_; ++s) {
      ConstMatrixMap<T> ma(a + s * offset_a_, row_a_, col_a_);
      MatrixMap<T> mdb(db + s * offset_b_, row_b_, col_b_);
      ConstMatrixMap<T> mdy(dy + s * offset_y_, row_y_, col_y_);
      if (accum[1] && !f_broadcast_b_) {
        NBLA_EIGEN_MATMUL_T(mdb, transpose_b_, ma, !transpose_a_, mdy, false,
                            +=);
      } else {
        NBLA_EIGEN_MATMUL_T(mdb, transpose_b_, ma, !transpose_a_, mdy, false,
                            =);
      }
    }
    // Broadcast backward
    if (f_broadcast_b_)
      nbla::backward(f_broadcast_b_, {inputs[1]}, {&b_broadcast}, {true},
                     {accum[1]});
  }
}

} // namespace nbla
