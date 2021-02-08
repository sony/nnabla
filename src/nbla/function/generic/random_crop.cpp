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

/** RandomCrop
 */
#include <nbla/array.hpp>
#include <nbla/function/random_crop.hpp>
#include <nbla/random_manager.hpp>
#include <nbla/variable.hpp>

#include <algorithm>
#include <cstring>

namespace nbla {

NBLA_REGISTER_FUNCTION_SOURCE(RandomCrop, const vector<int> &, int, int);

template <typename T>
void RandomCrop<T>::setup_impl(const Variables &inputs,
                               const Variables &outputs) {
  NBLA_CHECK(base_axis_ >= 0, error_code::value,
             "base_axis may not be less than zero, got %d", base_axis_);
  auto base_axis = static_cast<Shape_t::size_type>(base_axis_);
  NBLA_CHECK(base_axis < inputs[0]->shape().size(), error_code::value,
             "base_axis must be less than ndim of inputs[0]. "
             "base_axis: %d >= ndim of inputs[0]: %d.",
             base_axis_, inputs[0]->shape().size());

  std::random_device rdev_;
  rgen_ = std::mt19937((seed_ == -1 ? rdev_() : seed_));
  size_ = inputs[0]->size() / inputs[0]->size(base_axis);

  Shape_t shape_y = inputs[0]->shape();
  dim_offset_ = shape_y.size() - shape_.size();
  for (Shape_t::size_type i = 0; i < shape_.size(); i++) {
    NBLA_CHECK(shape_[i] <= shape_y[i + dim_offset_], error_code::value,
               "Shape must be smaller than input shape. "
               "Shape[%id]: %d > Input shape[%d]: %d",
               i, shape_[i], i, shape_y[i + dim_offset_]);
    shape_y[i + dim_offset_] = shape_[i];
  }

  outputs[0]->reshape(shape_y, true);
}

template <typename T>
void RandomCrop<T>::slice_forward_recursive(const Variable *inp, Variable *outp,
                                            const T *x, T *y, int x_offset,
                                            int y_offset, int dim,
                                            int &slice_index) {
  int current_x_offset = x_offset, current_y_offset = y_offset;
  const int x_stride = inp->strides()[dim] * step_[slice_index][dim];
  const int y_stride = outp->strides()[dim];
  current_x_offset += inp->strides()[dim] * start_[slice_index][dim];
  const int size = outp->shape()[dim];

  if (static_cast<Shape_t::size_type>(dim) == inp->shape().size() - 1) {
    const T *current_x = x + current_x_offset;
    T *current_y = y + current_y_offset;
    if (x_stride == 1) {
      memcpy((void *)current_y, current_x, sizeof(T) * size);
    } else {
      const T *end_x = current_x + size * x_stride;
      while (current_x != end_x) {
        *current_y = *current_x;
        current_x += x_stride;
        current_y += y_stride;
      }
    }
  } else {
    for (int i = 0; i < size; i++) {
      slice_forward_recursive(inp, outp, x, y, current_x_offset,
                              current_y_offset, dim + 1, slice_index);
      current_x_offset += x_stride;
      current_y_offset += y_stride;
      if (dim < base_axis_) {
        slice_index = (slice_index + 1) % start_.size();
      }
    }
  }
}

template <typename T>
void RandomCrop<T>::slice_backward_recursive(Variable *outp,
                                             const Variable *inp, T *dx,
                                             const T *dy, int x_offset,
                                             int y_offset, int dim,
                                             int &slice_index) {
  int current_x_offset = x_offset, current_y_offset = y_offset;
  const int x_stride = outp->strides()[dim] * step_[slice_index][dim];
  const int y_stride = inp->strides()[dim];
  current_x_offset += outp->strides()[dim] * start_[slice_index][dim];
  const int size = inp->shape()[dim];

  if (dim == static_cast<int>(outp->shape().size()) - 1) {
    T *current_dx = dx + current_x_offset;
    const T *current_dy = dy + current_y_offset;
    T *end_dx = current_dx + size * x_stride;
    while (current_dx != end_dx) {
      *current_dx += (*current_dy);
      current_dx += x_stride;
      current_dy += y_stride;
    }
  } else {
    for (int i = 0; i < size; i++) {
      slice_backward_recursive(outp, inp, dx, dy, current_x_offset,
                               current_y_offset, dim + 1, slice_index);
      current_x_offset += x_stride;
      current_y_offset += y_stride;
      if (dim < base_axis_) {
        slice_index = (slice_index + 1) % start_.size();
      }
    }
  }
}

template <typename T>
void RandomCrop<T>::forward_impl(const Variables &inputs,
                                 const Variables &outputs) {
  std::mt19937 &rgen =
      seed_ == -1 ? SingletonManager::get<RandomManager>()->get_rand_generator()
                  : rgen_;
  start_.resize(size_);
  stop_.resize(size_);
  step_.resize(size_);
  for (int id = 0; id < size_; id++) {
    start_[id].clear();
    stop_[id].clear();
    step_[id].clear();
    for (int i = 0; i < static_cast<int>(inputs[0]->shape().size()); i++) {
      const int left =
          i >= dim_offset_
              ? rgen() % (inputs[0]->shape()[i] - shape_[i - dim_offset_] + 1)
              : 0;

      start_[id].push_back(left);
      stop_[id].push_back(left + (i >= dim_offset_ ? shape_[i - dim_offset_]
                                                   : inputs[0]->shape()[i]));
      step_[id].push_back(1);
    }
  }

  const T *x = inputs[0]->get_data_pointer<T>(this->ctx_);
  T *y = outputs[0]->cast_data_and_get_pointer<T>(this->ctx_, true);

  int slice_index = 0;
  slice_forward_recursive(inputs[0], outputs[0], x, y, 0, 0, 0, slice_index);
}

template <class T>
void RandomCrop<T>::backward_impl(const Variables &inputs,
                                  const Variables &outputs,
                                  const vector<bool> &propagate_down,
                                  const vector<bool> &accum) {
  if (!propagate_down[0]) {
    return;
  }
  if (!accum[0]) {
    inputs[0]->grad()->zero();
  }
  const T *dy = outputs[0]->get_grad_pointer<T>(this->ctx_);
  T *dx = inputs[0]->cast_grad_and_get_pointer<T>(this->ctx_, false);

  int slice_index = 0;
  slice_backward_recursive(inputs[0], outputs[0], dx, dy, 0, 0, 0, slice_index);
}

} // namespace nbla
