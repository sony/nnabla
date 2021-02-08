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

/** RandomFlip
 */
#include <nbla/array.hpp>
#include <nbla/function/random_flip.hpp>
#include <nbla/random_manager.hpp>
#include <nbla/variable.hpp>

#include <algorithm>
#include <cstring>

namespace nbla {

NBLA_REGISTER_FUNCTION_SOURCE(RandomFlip, const vector<int> &, int, int);

template <typename T>
void RandomFlip<T>::setup_impl(const Variables &inputs,
                               const Variables &outputs) {
  std::random_device rdev_;
  rgen_ = std::mt19937((seed_ == -1 ? rdev_() : seed_));
  size_ = inputs[0]->size() / inputs[0]->size(base_axis_);
  outputs[0]->reshape(inputs[0]->shape(), true);
}

template <typename T>
void RandomFlip<T>::flip_recursive(const Variable *inp, const T *x, T *y,
                                   bool add, int x_offset, int y_offset,
                                   int dim, int &flip_index) {
  int current_x_offset = x_offset, current_y_offset = y_offset;
  const int y_stride = inp->strides()[dim];
  int x_stride = y_stride;
  const int size = inp->shape()[dim];
  if (flip_[flip_index][dim]) {
    current_x_offset += x_stride * (size - 1);
    x_stride = -x_stride;
  }
  if (static_cast<Shape_t::size_type>(dim) == inp->shape().size() - 1) {
    const T *current_x = x + current_x_offset;
    const T *end_x = current_x + size * x_stride;
    T *current_y = y + current_y_offset;
    if (add) {
      while (current_x != end_x) {
        *current_y += *current_x;
        current_x += x_stride;
        current_y += y_stride;
      }
    } else {
      if (x_stride == 1) {
        memcpy((void *)current_y, current_x, sizeof(T) * size);
      } else
        while (current_x != end_x) {
          *current_y = *current_x;
          current_x += x_stride;
          current_y += y_stride;
        }
    }
  } else {
    for (int i = 0; i < size; i++) {
      flip_recursive(inp, x, y, add, current_x_offset, current_y_offset,
                     dim + 1, flip_index);
      current_x_offset += x_stride;
      current_y_offset += y_stride;
      if (dim < base_axis_) {
        flip_index = (flip_index + 1) % flip_.size();
      }
    }
  }
}

template <typename T>
void RandomFlip<T>::forward_impl(const Variables &inputs,
                                 const Variables &outputs) {
  std::mt19937 &rgen =
      seed_ == -1 ? SingletonManager::get<RandomManager>()->get_rand_generator()
                  : rgen_;
  flip_.resize(size_);
  auto input0_shape_size = inputs[0]->shape().size();
  for (int i = 0; i < size_; i++) {
    flip_[i].resize(input0_shape_size);
    for (int id = 0; static_cast<size_t>(id) < input0_shape_size; id++) {
      auto itr = std::find(axes_.begin(), axes_.end(), id);
      flip_[i][id] = (rgen() % 2) && (itr != axes_.end());
    }
  }

  const T *x = inputs[0]->get_data_pointer<T>(this->ctx_);
  T *y = outputs[0]->cast_data_and_get_pointer<T>(this->ctx_, true);
  int flip_index = 0;
  flip_recursive(inputs[0], x, y, false, 0, 0, 0, flip_index);
}

template <typename T>
void RandomFlip<T>::backward_impl(const Variables &inputs,
                                  const Variables &outputs,
                                  const vector<bool> &propagate_down,
                                  const vector<bool> &accum) {
  if (!propagate_down[0]) {
    return;
  }
  const T *dy = outputs[0]->get_grad_pointer<T>(this->ctx_);
  T *dx = inputs[0]->cast_grad_and_get_pointer<T>(this->ctx_, !accum[0]);
  int flip_index = 0;
  flip_recursive(outputs[0], dy, dx, accum[0], 0, 0, 0, flip_index);
}

} // namespace nbla
