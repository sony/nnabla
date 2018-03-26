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

// reshape.cpp

#include <nbla/array.hpp>
#include <nbla/function/reshape.hpp>
#include <nbla/variable.hpp>

#include <algorithm>
#include <cstring>

namespace nbla {

NBLA_REGISTER_FUNCTION_SOURCE(Reshape, const vector<int> &);

template <typename T>
void Reshape<T>::setup_impl(const Variables &inputs, const Variables &outputs) {
  int tsize = inputs[0]->size();
  int tsize2 = 1;
  for (auto s : shape_)
    tsize2 *= s;
  NBLA_CHECK(tsize == tsize2, error_code::value,
             "Product of dimensions of inputs and outputs must be same. "
             "Inputs: %d != Outputs: %d.",
             tsize, tsize2);
  outputs[0]->reshape(shape_, true);
  // Reshape function is always in-place.
  outputs[0]->data()->set_array(inputs[0]->data()->array());
  outputs[0]->grad()->set_array(inputs[0]->grad()->array());
}

template <class T>
void Reshape<T>::forward_impl(const Variables &inputs,
                              const Variables &outputs) {}

template <class T>
void Reshape<T>::backward_impl(const Variables &inputs,
                               const Variables &outputs,
                               const vector<bool> &propagate_down,
                               const vector<bool> &accum) {}
}
