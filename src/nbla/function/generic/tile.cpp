// Copyright (c) 2019 Sony Corporation. All Rights Reserved.
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
#include <nbla/function/tile.hpp>
#include <nbla/utils/nd_index.hpp>
#include <nbla/variable.hpp>

#include <cstring>

namespace nbla {

NBLA_REGISTER_FUNCTION_SOURCE(Tile, const vector<int> &);

template <typename T>
void Tile<T>::setup_impl(const Variables &inputs, const Variables &outputs) {
  // For run time efficiency we create an index mapping `y[i] = x[idxmap[i]]`
  // for `0 <= i < y.size()` during setup and just perform lookup and copy/add
  // operations during forward and backward.

  auto less_than_one = std::bind(std::less<int>(), std::placeholders::_1, 1);
  NBLA_CHECK(std::none_of(reps_.begin(), reps_.end(), less_than_one),
             error_code::value, "No repetition may be less than 1.");

  auto oshape = ndi::multiply(inputs[0]->shape(), this->reps_);
  outputs[0]->reshape(oshape, true);

  auto reps = ndi::expand(this->reps_, oshape.size() + 1, 1);
  auto map_shape = ndi::expand(oshape, reps.size(), (Size_t)1);
  auto src_shape = ndi::expand(inputs[0]->shape(), reps.size(), (Size_t)1);
  auto src_index = ndi::make_index<Size_t>(src_shape.size());

  idxmap_.reshape(map_shape, true);
  Context cpu_ctx{{"cpu:float"}, "CpuCachedArray", "0"};
  auto arr = idxmap_.cast(get_dtype<int>(), cpu_ctx);
  auto idx = arr->template pointer<int>();

  int *src, *dst;
  int iota = 0;

  // Create running index and tiles for the innermost dimension (the outer
  // while loop increments src_index for axes 0 to ndim-2). The first for loop
  // writes incrementing numbers for the last dimension of src_shape. The
  // second for loop creates the requested number of copies (tiles) for the
  // last dimension.
  do {
    src = dst = idx + ndi::nd2flat(src_index, idxmap_.strides());
    for (int i = 0; i < src_shape.back(); i++) {
      *dst++ = iota++;
    }
    for (int r = 1; r < reps.back(); r++) {
      std::memcpy(dst, src, src_shape.back() * sizeof(int));
      dst += src_shape.back();
    }
  } while (ndi::increment(src_index, src_shape, -2));

  // For each dimension starting with the second last axis then upwards create
  // the requested number of copies (tiles) for that dimension (this is a bit
  // like unfolding a cross folded piece of paper).
  for (int axis = reps.size() - 2; axis > 0; axis--) {
    do {
      auto src = dst = idx + ndi::nd2flat(src_index, idxmap_.strides());
      auto cnt = ndi::inner_size(map_shape, axis) / reps.at(axis);
      for (int rep = 1; rep < reps.at(axis); rep++) {
        std::memcpy(dst + rep * cnt, src, cnt * sizeof(int));
      }
    } while (ndi::increment(src_index, src_shape, axis - 1));
  }
}

template <typename T>
void Tile<T>::forward_impl(const Variables &inputs, const Variables &outputs) {
  auto src = inputs[0]->get_data_pointer<T>(ctx_);
  auto dst = outputs[0]->cast_data_and_get_pointer<T>(ctx_, true);
  auto arr = idxmap_.get(get_dtype<int>(), ctx_);
  auto idx = arr->template const_pointer<int>();

  for (int i = 0; i < idxmap_.size(); i++) {
    dst[i] = src[idx[i]];
  }
}

template <typename T>
void Tile<T>::backward_impl(const Variables &inputs, const Variables &outputs,
                            const vector<bool> &propagate_down,
                            const vector<bool> &accum) {
  if (!propagate_down[0]) {
    return;
  }

  if (!accum[0]) {
    inputs[0]->grad()->zero();
  }

  auto g_y = outputs[0]->get_grad_pointer<T>(ctx_);
  auto g_x = inputs[0]->cast_grad_and_get_pointer<T>(ctx_, false);
  auto arr = idxmap_.get(get_dtype<int>(), ctx_);
  auto idx = arr->template const_pointer<int>();

  for (int i = 0; i < idxmap_.size(); i++) {
    g_x[idx[i]] += g_y[i];
  }
}
}
