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

/** OneHot
 */
#include <nbla/array.hpp>
#include <nbla/function/one_hot.hpp>
#include <nbla/variable.hpp>

#include <algorithm>
#include <cassert>

namespace nbla {

NBLA_REGISTER_FUNCTION_SOURCE(OneHot, const vector<int> &);

template <typename T, typename T1>
void OneHot<T, T1>::setup_impl(const Variables &inputs,
                               const Variables &outputs) {
  Shape_t shape_x = inputs[0]->shape();
  assert(shape_x.size() >= 1);
  dim_ = shape_x[shape_x.size() - 1];
  NBLA_CHECK(shape_.size() == static_cast<size_t>(dim_), error_code::value,
             "Shape size does not match last dimension of inputs[0]."
             "shape size: %d != input dim: %d.",
             shape_.size(), dim_);
  num_ = inputs[0]->size() / dim_;
  Shape_t shape_y = shape_x;
  shape_y.erase(shape_y.begin() + shape_y.size() - 1);
  size_ = 1;
  for (Shape_t::size_type i = 0; i < shape_.size(); ++i) {
    size_ *= shape_[i];
    shape_y.push_back(shape_[i]);
  }
  outputs[0]->reshape(shape_y, true);
  NBLA_CHECK(outputs[0]->size() == num_ * size_, error_code::unclassified,
             "An error occurred during setup OneHot function.");
}

template <typename T, typename T1>
void OneHot<T, T1>::forward_impl(const Variables &inputs,
                                 const Variables &outputs) {
  const T *x = inputs[0]->get_data_pointer<T>(this->ctx_);
  outputs[0]->data()->zero();
  T1 *y = outputs[0]->cast_data_and_get_pointer<T1>(this->ctx_, false);
  for (int i = 0; i < num_; ++i) {
    int addr = 0;
    Size_t size = 1;
    for (int i2 = dim_ - 1; i2 >= 0; --i2) {
      addr += x[i * dim_ + i2] * size;
      size *= shape_[i2];
    }
    y[i * size_ + addr] = (T1)1;
  }
}

template <typename T, typename T1>
void OneHot<T, T1>::backward_impl(const Variables &inputs,
                                  const Variables &outputs,
                                  const vector<bool> &propagate_down,
                                  const vector<bool> &accum) {
  NBLA_CHECK(!propagate_down[0], error_code::value,
             "Index array can not be propagated down.");
}

} // namespace nbla
