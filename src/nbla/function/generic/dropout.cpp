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

// dropout.cpp

#include <nbla/array.hpp>
#include <nbla/function/dropout.hpp>
#include <nbla/random_manager.hpp>
#include <nbla/variable.hpp>

#include <algorithm>

namespace nbla {

NBLA_REGISTER_FUNCTION_SOURCE(Dropout, double, int);

template <typename T>
void Dropout<T>::setup_impl(const Variables &inputs, const Variables &outputs) {
  NBLA_CHECK(p_ > 0. && p_ < 1., error_code::value,
             "p must be between 0.0 and 1.0. p: %f.", p_);
  outputs[0]->reshape(inputs[0]->shape(), true);
  mask_.reshape(inputs[0]->shape(), true);
  std::random_device rdev_;
  rgen_ = std::mt19937((seed_ == -1 ? rdev_() : seed_));
  rdist_ = std::bernoulli_distribution(1 - p_);
  scale_ = 1. / (1. - p_);
}

template <class T>
void Dropout<T>::forward_impl(const Variables &inputs,
                              const Variables &outputs) {
  std::mt19937 &rgen =
      seed_ == -1 ? SingletonManager::get<RandomManager>()->get_rand_generator()
                  : rgen_;
  const T *x = inputs[0]->get_data_pointer<T>(this->ctx_);
  T *y = outputs[0]->cast_data_and_get_pointer<T>(this->ctx_, true);
  T *m = mask_.cast_data_and_get_pointer<T>(this->ctx_, true);
  for (int s = 0; s < inputs[0]->size(); s++) {
    m[s] = rdist_(rgen);
    y[s] = x[s] * m[s] * scale_;
  }
}

template <class T>
void Dropout<T>::backward_impl(const Variables &inputs,
                               const Variables &outputs,
                               const vector<bool> &propagate_down,
                               const vector<bool> &accum) {
  if (!propagate_down[0]) {
    return;
  }
  T *dx = inputs[0]->cast_grad_and_get_pointer<T>(this->ctx_, !accum[0]);
  const T *dy = outputs[0]->get_grad_pointer<T>(this->ctx_);
  const T *m = mask_.get_data_pointer<T>(this->ctx_);
  for (int s = 0; s < inputs[0]->size(); ++s) {
    dx[s] = (accum[0] ? dx[s] : (T)0) + dy[s] * m[s] * scale_;
  }
}
}
