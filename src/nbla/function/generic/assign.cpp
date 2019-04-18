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
#include <nbla/function/assign.hpp>
#include <nbla/function/add2.hpp>
#include <nbla/variable.hpp>

namespace nbla {

NBLA_REGISTER_FUNCTION_SOURCE(Assign);

template <typename T>
void Assign<T>::setup_impl(const Variables &inputs,
                               const Variables &outputs) {
  NBLA_CHECK(inputs[0]->shape() == inputs[1]->shape(), error_code::value,
             "Dimensions of inputs must match. "
             "inputs[0]: %s != inputs[1]: %s.",
             string_join(inputs[0]->shape(), string(", ")).c_str(),
             string_join(inputs[1]->shape(), string(", ")).c_str());
  outputs[0]->reshape(inputs[0]->shape(), true);

  gy_ = make_shared<Variable>(outputs[0]->grad());
  gx_ = make_shared<Variable>(inputs[0]->grad());

  f_add_ = create_Add2(this->ctx_, true);
  f_add_->setup(Variables{gx_.get(), gy_.get()}, Variables{gx_.get()});
}

template <typename T>
void Assign<T>::forward_impl(const Variables &inputs,
                               const Variables &outputs) {
  Array *dst = inputs[0]->data()->cast(get_dtype<T>(), this->ctx_, true);
  const Array *src = inputs[1]->data()->get(get_dtype<T>(), this->ctx_);
  Array *y = outputs[0]->data()->cast(get_dtype<T>(), this->ctx_, true);
  dst->copy_from(src);
  y->copy_from(src);
}


template <typename T>
void Assign<T>::backward_impl(const Variables &inputs,
                                const Variables &outputs,
                                const vector<bool> &propagate_down,
                                const vector<bool> &accum) {
  if (!propagate_down[0])
    return;

  if (!accum[0])
    inputs[0]->grad()->zero();

  gx_->data()->set_array(inputs[0]->grad()->array());
  f_add_->forward(Variables{gx_.get(), gy_.get()}, Variables{gx_.get()});
}
}
