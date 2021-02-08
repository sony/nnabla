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

#include <functional>
#include <nbla/array.hpp>
#include <nbla/common.hpp>
#include <nbla/function/gather.hpp>
#include <nbla/utils/nd_index.hpp>
#include <nbla/variable.hpp>
#include <numeric>

namespace nbla {

NBLA_REGISTER_FUNCTION_SOURCE(Gather, int, int);

template <typename T>
void Gather<T>::setup_impl(const Variables &inputs, const Variables &outputs) {
  // For example:
  /*
  batch_dims = 2
  axis = 3
  Inp: (B0, B1, D0, D1, D2)        -->  (B0 B1 D0, D1   , D2)
  Idx: (B0, B1, I0, I1)            -->  (B0 B1   , I0 I1)
  Out: (B0, B1, D0, I0, I1, D2)    -->  (B0 B1 D0, I0 I1, D2)

  Out[b0, b1, d0, i0, i1, d2] = Inp[b0, b1, d0, Idx[b0, b1, i0, i1], d1]
  */
  auto x = inputs[0];
  auto indices = inputs[1];
  auto xshape = x->shape();
  auto ishape = indices->shape();
  if (axis_ < 0)
    axis_ += xshape.size();

  NBLA_CHECK(axis_ >= batch_dims_, error_code::value,
             "axis (%d) must be greater than or equal to batch_dims_ (%d).",
             axis_, batch_dims_);
  NBLA_CHECK(
      x->ndim() >= axis_ + 1, error_code::value,
      "ndim (%d) of inputs[0] must be greater than or equal to axis (%d) + 1.",
      x->ndim(), axis_);
  NBLA_CHECK(x->ndim() >= batch_dims_, error_code::value,
             "ndim (%d) of inputs[0] must be greater than or equal to "
             "batch_dims (%d).",
             x->ndim(), batch_dims_);
  NBLA_CHECK(indices->ndim() >= batch_dims_, error_code::value,
             "ndim (%d) of inputs[1] must be greater than or equal to "
             "batch_dims (%d).",
             indices->ndim(), batch_dims_);
  for (auto b = 0; b < batch_dims_; b++) {
    NBLA_CHECK(
        xshape[b] == ishape[b], error_code::value,
        "inputs[0].shape[%d] (%d) must equal to inputs[1].shape[%d] (%d).", b,
        xshape[b], b, ishape[b]);
  }

  // y.shape = x.shape[:axis] + indices.shape[batch_dims_:] + x.shape[axis+1: ]
  Shape_t yshape;
  for (auto d = 0; d < axis_; d++) {
    yshape.push_back(xshape[d]);
  }
  for (auto i = batch_dims_; i < indices->ndim(); i++) {
    yshape.push_back(ishape[i]);
  }
  for (auto d = axis_ + 1; d < x->ndim(); d++) {
    yshape.push_back(xshape[d]);
  }
  auto y = outputs[0];
  y->reshape(yshape, true);
}

template <typename T>
void Gather<T>::forward_impl(const Variables &inputs,
                             const Variables &outputs) {

  auto x = inputs[0];
  auto indices = inputs[1];
  auto y = outputs[0];
  auto xshape = x->shape();
  auto ishape = indices->shape();
  auto yshape = y->shape();

  auto prod = [](Shape_t &shape, int b, int e) {
    return std::accumulate(shape.begin() + b, shape.begin() + e, 1,
                           std::multiplies<int64_t>());
  };

  // Fold xshape: prod(xshape[:axis]), xshape[axis], prod(xshape[axis+1:])
  auto xsize0 = prod(xshape, 0, axis_);
  auto xsize1 = xshape[axis_];
  auto xsize2 = prod(xshape, axis_ + 1, xshape.size());
  auto xshape_f = Shape_t{xsize0, xsize1, xsize2};
  auto xstrides_f = ndi::strides(xshape_f);
  // Fold ishape: prod(ishape[:batch_dims]), prod(ishape[batch_dims:])
  auto isize0 = prod(ishape, 0, batch_dims_);
  auto isize1 = prod(ishape, batch_dims_, ishape.size());
  auto ishape_f = Shape_t{isize0, isize1};
  auto istrides_f = ndi::strides(ishape_f);
  // Fold yshape: prod(yshape[:axis]), prod(ishape[batch_dims:]),
  // prod(xshape[axis+1:])
  auto ysize0 = prod(yshape, 0, axis_);
  auto ysize1 = isize1;
  auto ysize2 = xsize2;
  auto yshape_f = Shape_t{ysize0, ysize1, ysize2};
  auto ystrides_f = ndi::strides(yshape_f);
  // Fold leading dimensions of y
  auto bsize = isize0;
  auto leading_dsize = ysize0 / bsize;
  auto lshape = Shape_t{bsize, leading_dsize};
  auto lstrides = ndi::strides(lshape);

  // Loop over y
  // Inp[i, g, k]
  // Idx[b, j]
  // Out[i, j, k]
  // i: dimensions up to axis of Out
  // j: dimensions of indices of Out
  // k: dimensions of the remains of Out
  // b: dimensions of batches
  // g: gather index (actual value of Idx)
  auto x_data = x->get_data_pointer<T>(ctx_);
  auto i_data = indices->get_data_pointer<int64_t>(ctx_);
  auto y_data = y->cast_data_and_get_pointer<T>(ctx_);
  for (int64_t i = 0; i < yshape_f[0]; ++i) {
    auto nd_lidx = ndi::flat2nd(i, lstrides);
    auto b = nd_lidx[0];
    for (int64_t j = 0; j < yshape_f[1]; ++j) {
      auto iidx = ndi::nd2flat(Shape_t{b, j}, istrides_f);
      auto g = i_data[iidx];
      NBLA_CHECK(0 <= g && g < xsize1, error_code::value,
                 "Out-of-bounds index: 0 <= %d < %d", g, xsize1);
      for (int64_t k = 0; k < yshape_f[2]; ++k) {
        auto xidx = ndi::nd2flat(Shape_t{i, g, k}, xstrides_f);
        auto yidx = ndi::nd2flat(Shape_t{i, j, k}, ystrides_f);
        y_data[yidx] = x_data[xidx];
      }
    }
  }
}

template <typename T>
void Gather<T>::backward_impl(const Variables &inputs, const Variables &outputs,
                              const vector<bool> &propagate_down,
                              const vector<bool> &accum) {
  if (!propagate_down[0]) {
    return;
  }
  auto x = inputs[0];
  auto indices = inputs[1];
  auto y = outputs[0];
  auto xshape = x->shape();
  auto ishape = indices->shape();
  auto yshape = y->shape();

  auto prod = [](Shape_t &shape, int b, int e) {
    return std::accumulate(shape.begin() + b, shape.begin() + e, 1,
                           std::multiplies<int64_t>());
  };

  // Fold xshape: prod(xshape[:axis]), xshape[axis], prod(xshape[axis+1:])
  auto xsize0 = prod(xshape, 0, axis_);
  auto xsize1 = xshape[axis_];
  auto xsize2 = prod(xshape, axis_ + 1, xshape.size());
  auto xshape_f = Shape_t{xsize0, xsize1, xsize2};
  auto xstrides_f = ndi::strides(xshape_f);
  // Fold ishape: prod(ishape[:batch_dims]), prod(ishape[batch_dims:])
  auto isize0 = prod(ishape, 0, batch_dims_);
  auto isize1 = prod(ishape, batch_dims_, ishape.size());
  auto ishape_f = Shape_t{isize0, isize1};
  auto istrides_f = ndi::strides(ishape_f);
  // Fold yshape: prod(yshape[:axis]), prod(ishape[batch_dims:]),
  // prod(xshape[axis+1:])
  auto ysize0 = prod(yshape, 0, axis_);
  auto ysize1 = isize1;
  auto ysize2 = xsize2;
  auto yshape_f = Shape_t{ysize0, ysize1, ysize2};
  auto ystrides_f = ndi::strides(yshape_f);
  // Fold leading dimensions of y
  auto bsize = isize0;
  auto leading_dsize = ysize0 / bsize;
  auto lshape_f = Shape_t{bsize, leading_dsize};
  auto lstrides_f = ndi::strides(lshape_f);

  // Loop over y
  // Inp[i, g, k]
  // Idx[b, j]
  // Out[i, j, k]
  // i: dimensions up to axis of Out
  // j: dimensions of indices of Out
  // k: dimensions of the remains of Out
  // b: dimensinos of batches
  // g: gather index (actual value of Idx)
  auto x_grad = x->cast_grad_and_get_pointer<T>(ctx_);
  auto i_data = indices->get_data_pointer<int64_t>(ctx_);
  auto y_grad = y->get_grad_pointer<T>(ctx_);
  for (int64_t i = 0; i < yshape_f[0]; ++i) {
    auto nd_lidx = ndi::flat2nd(i, lstrides_f);
    auto b = nd_lidx[0];
    for (int64_t j = 0; j < yshape_f[1]; ++j) {
      auto iidx = ndi::nd2flat(Shape_t{b, j}, istrides_f);
      auto g = i_data[iidx];
      NBLA_CHECK(0 <= g && g < xsize1, error_code::value,
                 "Out-of-bounds index: 0 <= %d < %d", g, xsize1);
      for (int64_t k = 0; k < yshape_f[2]; ++k) {
        auto xidx = ndi::nd2flat(Shape_t{i, g, k}, xstrides_f);
        auto yidx = ndi::nd2flat(Shape_t{i, j, k}, ystrides_f);
        // Gather output may take a same input, thus accumulate grads.
        x_grad[xidx] += y_grad[yidx];
      }
    }
  }
}
}
