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

// -*- coding:utf-8 -*-
/*
 * Copyright (C) 2016 Sony Corporation
 * This is UNPUBLISHED PROPRIETARY SOURCE CODE of Sony Corporation;
 * the contents of this file is not to be disclosed to third parties, copied
 * or duplicated in any form, in whole or in part, without the prior written
 * permission of Sony Corporation.
 */

/** Sum
 */
#include <nbla/array.hpp>
#include <nbla/function/sum.hpp>
#include <nbla/function/transpose.hpp>
#include <nbla/singleton_manager.hpp>
#include <nbla/utils/eigen.hpp>
#include <nbla/variable.hpp>

#include <algorithm>
#include <numeric> // iota

namespace nbla {

using namespace ::nbla::eigen;

NBLA_REGISTER_FUNCTION_SOURCE(Sum, const vector<int> &, bool);

template <typename T>
void Sum<T>::setup_impl(const Variables &inputs, const Variables &outputs) {
  int ndim = inputs[0]->ndim();
  auto inshape = inputs[0]->shape();
  // =========
  // Transpose
  // =========
  // get transpose axes.
  // Create transpose axes and reduction size
  vector<int> transpose_axes;
  int prev_a = -1;
  reduction_size_ = 1;
  Shape_t outshape;
  for (int a : axes_) {
    NBLA_CHECK(a < ndim && a >= 0, error_code::value,
               "Axes out of range. 0 <= %d < %d", a, ndim);
    for (int b = prev_a + 1; b < a; ++b) {
      transpose_axes.push_back(b);
      outshape.push_back(inshape[b]);
    }
    if (keep_dims_) {
      outshape.push_back(1);
    }
    reduction_size_ *= inshape[a];
    prev_a = a;
  }
  for (int a = prev_a + 1; a < ndim; ++a) {
    transpose_axes.push_back(a);
    outshape.push_back(inshape[a]);
  }

  std::copy(axes_.begin(), axes_.end(), std::back_inserter(transpose_axes));
  // Sequence of numbers [0, ndim)
  vector<int> seq(ndim);
  std::iota(seq.begin(), seq.end(), 0);
  if (transpose_axes != seq) {
    // Need transpose
    f_transpose_ = create_Transpose(this->ctx_, transpose_axes);
    o_transpose_ = make_shared<Variable>(Shape_t{}, inputs[0]->need_grad());
    f_transpose_->setup(inputs, Variables{o_transpose_.get()});
  }
  outputs[0]->reshape(outshape, true);
}

template <typename T>
void Sum<T>::forward_impl(const Variables &inputs, const Variables &outputs) {
  if (f_transpose_) {
    f_transpose_->forward(inputs, Variables{o_transpose_.get()});
  }
  auto _get = [this](Variable *v) {
    return v->get_data_pointer<T>(this->ctx_);
  };
  const T *x = f_transpose_ ? _get(o_transpose_.get()) : _get(inputs[0]);
  T *y = outputs[0]->cast_data_and_get_pointer<T>(this->ctx_);
  const int outer_size = inputs[0]->size() / reduction_size_;
  this->forward_impl_reduce(x, y, outer_size, reduction_size_);
}

template <typename T>
void Sum<T>::backward_impl(const Variables &inputs, const Variables &outputs,
                           const vector<bool> &propagate_down,
                           const vector<bool> &accum) {
  if (!propagate_down[0])
    return;
  auto _gcast = [this](Variable *v) {
    return v->cast_grad_and_get_pointer<T>(this->ctx_);
  };
  const T *dy = outputs[0]->get_grad_pointer<T>(this->ctx_);
  T *dx = f_transpose_ ? _gcast(o_transpose_.get()) : _gcast(inputs[0]);
  this->backward_impl_reduce(dy, dx, inputs[0]->size() / reduction_size_,
                             reduction_size_, (!f_transpose_) && accum[0]);

  // If need un-transpose
  if (!f_transpose_)
    return;
  f_transpose_->backward(inputs, Variables{o_transpose_.get()}, {accum[0]});
}

template <typename T>
void Sum<T>::forward_impl_reduce(const T *x, T *y, int outer_size,
                                 int reduction_size) {
  using namespace ::nbla::eigen;
  ConstMatrixMap<T> mx(x, outer_size, reduction_size);
  ColVectorMap<T> my(y, outer_size);
  my = mx.rowwise().sum();
}

template <typename T>
void Sum<T>::backward_impl_reduce(const T *dy, T *dx, int outer_size,
                                  int reduction_size, bool accum) {
  using namespace ::nbla::eigen;
  ConstColVectorMap<T> mdy(dy, outer_size);
  MatrixMap<T> mdx(dx, outer_size, reduction_size);
  if (accum)
    mdx.colwise() += mdy;
  else
    mdx.colwise() = mdy;
}

// Template instanciation
template class Sum<float>;
} // namespace nbla
