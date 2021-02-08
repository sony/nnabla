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

/** RandBeta
*/
#include <nbla/array.hpp>
#include <nbla/common.hpp>
#include <nbla/function/rand_beta.hpp>
#include <nbla/random_manager.hpp>
#include <nbla/variable.hpp>

#include <random>

namespace nbla {

NBLA_REGISTER_FUNCTION_SOURCE(RandBeta, float, float, const vector<int> &, int);

template <typename T>
void RandBeta<T>::setup_impl(const Variables &inputs,
                             const Variables &outputs) {
  outputs[0]->reshape(Shape_t(shape_.cbegin(), shape_.cend()), true);
  rgen_ = std::mt19937((seed_ == -1 ? std::random_device()() : seed_));
}

template <typename T>
void RandBeta<T>::forward_impl(const Variables &inputs,
                               const Variables &outputs) {
  std::uniform_real_distribution<typename force_float<T>::type> rdist(0.0, 1.0);
  std::gamma_distribution<typename force_float<T>::type> gdist1(alpha_, 1);
  std::gamma_distribution<typename force_float<T>::type> gdist2(beta_, 1);
  std::mt19937 &rgen =
      seed_ == -1 ? SingletonManager::get<RandomManager>()->get_rand_generator()
                  : rgen_;

  T *y = outputs[0]->cast_data_and_get_pointer<T>(this->ctx_, true);
  if ((alpha_ <= 1.0) && (beta_ <= 1.0)) {
    int s = 0;
    while (s < outputs[0]->size()) {
      float U, V, X, Y, XpY;
      U = (T)rdist(rgen);
      V = (T)rdist(rgen);
      X = std::pow(U, 1 / alpha_);
      Y = std::pow(V, 1 / beta_);
      XpY = X + Y;
      if ((XpY <= 1.0) && (XpY > 0.0)) {
        if (X + Y > 0) {
          y[s] = X / XpY;
          s++;
        } else {
          float logX, logY, logM;
          logX = std::log(U) / alpha_;
          logY = std::log(V) / beta_;
          logM = logX > logY ? logX : logY;
          logX -= logM;
          logY -= logM;
          y[s] = std::exp(logX - log(std::exp(logX) + std::exp(logY)));
          s++;
        }
      }
    }
  } else {
    for (int s = 0; s < outputs[0]->size(); s++) {
      float Ga, Gb;
      Ga = (T)gdist1(rgen);
      Gb = (T)gdist2(rgen);
      y[s] = Ga / (Ga + Gb);
    }
  }
}

template <typename T>
void RandBeta<T>::backward_impl(const Variables &inputs,
                                const Variables &outputs,
                                const vector<bool> &propagate_down,
                                const vector<bool> &accum) {
  // Pass
}

} // namespace nbla
