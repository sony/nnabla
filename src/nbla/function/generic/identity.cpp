// Copyright 2018,2019,2020,2021 Sony Corporation.
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

// identity.cpp

#include <nbla/array.hpp>
#include <nbla/function/identity.hpp>
#include <nbla/variable.hpp>

#include <algorithm>
#include <cstring>

namespace nbla {

NBLA_REGISTER_FUNCTION_SOURCE(Identity);

template <typename T>
void Identity<T>::setup_impl(const Variables &inputs,
                             const Variables &outputs) {
  outputs[0]->reshape(inputs[0]->shape(), true);
}

template <class T>
void Identity<T>::forward_impl(const Variables &inputs,
                               const Variables &outputs) {
  dtypes dtype = get_dtype<T>();
#if 1
  auto arr = inputs[0]->data()->array();
  if (arr->has_head_array()) {
    dtype = arr->dtype();
  }
#endif
  const Array *x = inputs[0]->data()->get(dtype, this->ctx_);
  Array *y = outputs[0]->data()->cast(dtype, this->ctx_, true);
  y->copy_from(x);
}

template <class T>
void Identity<T>::backward_impl(const Variables &inputs,
                                const Variables &outputs,
                                const vector<bool> &propagate_down,
                                const vector<bool> &accum) {
  if (!propagate_down[0]) {
    return;
  }

  T *dx = inputs[0]->cast_grad_and_get_pointer<T>(this->ctx_, !accum[0]);
  const T *dy = outputs[0]->get_grad_pointer<T>(this->ctx_);
  if (dx != dy) {
    for (int i = 0; i < inputs[0]->size(); ++i) {
      if (accum[0])
        dx[i] += dy[i];
      else
        dx[i] = dy[i];
    }
  }
}
}
