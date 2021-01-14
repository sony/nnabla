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

/** INQConvolution
 */
#include <nbla/array.hpp>
#include <nbla/function/inq_convolution.hpp>
#include <nbla/random_manager.hpp>
#include <nbla/variable.hpp>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <numeric>

namespace nbla {

NBLA_REGISTER_FUNCTION_SOURCE(INQConvolution, int, const vector<int> &,
                              const vector<int> &, const vector<int> &, int,
                              int, const vector<int> &, const string &, int);

template <typename T, typename T1>
void INQConvolution<T, T1>::setup_impl(const Variables &inputs,
                                       const Variables &outputs) {
  // A: Check shape of indicator matrix
  NBLA_CHECK(inputs[1]->shape().size() == inputs[2]->shape().size(),
             error_code::value,
             "Indicators and weights must have same size. "
             "Ndim of weights: %d != ndim of indicators: %d.",
             inputs[1]->shape().size(), inputs[2]->shape().size());
  for (Shape_t::size_type i = 0; i < inputs[1]->shape().size(); ++i) {
    NBLA_CHECK(inputs[1]->shape()[i] == inputs[2]->shape()[i],
               error_code::value,
               "Indicators and weights must have same size. "
               "weight shape[%d]: %d != indicator shape[%d]: %d.",
               i, inputs[1]->shape()[i], i, inputs[2]->shape()[i]);
  }

  // B: Check that chosen algorithm is either "largest_abs" or "random"
  NBLA_CHECK(
      selection_algorithm_ == "largest_abs" || selection_algorithm_ == "random",
      error_code::value, "Provided value for selection algorithm not valid: %s."
                         "Valid values are \"largest_abs\" and \"random\".",
      selection_algorithm_.c_str());

  // C: Initialize internal `convolution` function
  convolution_ =
      create_Convolution(this->ctx_, this->base_axis_, this->pad_,
                         this->stride_, this->dilation_, this->group_, false);
  if (inputs.size() == 4) { // with bias
    convolution_->setup(Variables{inputs[0], inputs[1], inputs[3]}, outputs);
  } else { // without bias
    convolution_->setup(Variables{inputs[0], inputs[1]}, outputs);
  }

  // D: Initialize random number generator (required for randomly selecting the
  // indices to fix)
  if (selection_algorithm_ == "random") {
    std::random_device rdev_;
    rgen_ = std::mt19937((seed_ == -1 ? rdev_() : seed_));
    rdist_ = std::bernoulli_distribution(0.5);
  }

  // F: Initialize minibatch counter and internal copies of weights/indicators
  minibatch_counter_ = 0;
  old_weights_.reshape(inputs[1]->shape(), true);
  old_indicators_.reshape(inputs[1]->shape(), true);
  old_indicators_.data()->zero();
}

template <typename T, typename T1>
void INQConvolution<T, T1>::forward_impl(const Variables &inputs,
                                         const Variables &outputs) {
  T *weights = inputs[1]->cast_data_and_get_pointer<T>(this->ctx_);
  T *old_weights = old_weights_.cast_data_and_get_pointer<T>(this->ctx_);

  T1 *indicators = inputs[2]->cast_data_and_get_pointer<T1>(this->ctx_);
  T1 *old_indicators =
      old_indicators_.cast_data_and_get_pointer<T1>(this->ctx_);

  // A: Go through each element and copy old value for weights if weight was
  // fixed before.
  //    This is done to make sure that we do not update fixed weights.
  for (int i = 0; i < inputs[1]->size(); ++i) {
    if (old_indicators[i] == 1) {
      weights[i] = old_weights[i];
    }
  }

  // B: Check whether we need to fix 50% of the learnable weights
  if (std::find(inq_iterations_.begin(), inq_iterations_.end(),
                minibatch_counter_) != inq_iterations_.end()) {
    // determine weights that we will fix
    if (inq_iterations_.back() == minibatch_counter_) {
      // if we have reached the last element in `inq_iterations` then we fix all
      // weights
      for (int i = 0; i < inputs[1]->size(); ++i) {
        indicators[i] = 1;
      }
    } else {
      // not last element in `inq_iterations`, hence fix 50% of the learnable
      // weights
      if (selection_algorithm_ == "largest_abs") {
        // fix weights with largest absolute value
        std::vector<size_t> indices(inputs[1]->size());
        std::iota(begin(indices), end(indices), 0);
        std::sort(begin(indices), end(indices), [&](size_t a, size_t b) {
          return std::abs(weights[a]) > std::abs(weights[b]);
        });

        int num_learnable = 0;
        for (int i = 0; i < inputs[1]->size(); ++i) {
          if (indicators[i] == 0) {
            num_learnable++;
          }
        }

        int num_fixed = 0;
        for (int i = 0; i < inputs[1]->size(); ++i) {
          if (indicators[indices[i]] == 0) {
            indicators[indices[i]] = 1;
            num_fixed++;
          }
          if (num_fixed >= num_learnable / 2) {
            break;
          }
        }
      } else {
        // random selection
        std::mt19937 &rgen =
            seed_ == -1
                ? SingletonManager::get<RandomManager>()->get_rand_generator()
                : rgen_;
        for (int i = 0; i < inputs[1]->size(); ++i) {
          if (indicators[i] == 0) {
            indicators[i] = rdist_(rgen);
          }
        }
      }
    }
  }

  // C: convert all fixed weights to power-of-two values
  T max_absval = 0.0f;
  for (int i = 0; i < inputs[1]->size(); ++i) {
    if (std::abs(weights[i]) > max_absval) {
      max_absval = std::abs(weights[i]);
    }
  }

  if (max_absval == 0.0f) {
    max_absval = 1.0f;
  }
  int n1 = (int)(std::floor(std::log2(max_absval)) +
                 (std::log2(max_absval) - std::floor(std::log2(max_absval)) >=
                  std::log2(1.5)));
  int n2 = n1 + 1 - (int)std::pow(2, num_bits_ - 2);
  T pruning_threshold = std::pow(2, n2 - 1);

  for (int i = 0; i < inputs[1]->size(); ++i) {
    if (indicators[i] == 1) {
      if (std::abs(weights[i]) < pruning_threshold) {
        weights[i] = 0.0f;
      } else {
        T s = (weights[i] < 0.0) ? -1.0 : 1.0;
        T b = std::log2(std::abs(weights[i]));
        T d = 0.58496250072115619; // quantization threshold log2(1.5)
        // (d = 0.5: use geometric mean; d = log2(1.5): use arithmetic mean)
        int e = (int)(std::floor(b) + (b - std::floor(b) >= d));

        if (e > n1) {
          e = n1;
        }
        if (e < n2) {
          e = n2;
        }
        weights[i] = std::ldexp(s, e);
      }
    }
  }

  // D: Calculate the forward pass
  if (inputs.size() == 4) { // with bias
    convolution_->forward(Variables{inputs[0], inputs[1], inputs[3]}, outputs);
  } else {
    convolution_->forward(Variables{inputs[0], inputs[1]}, outputs);
  }

  // E: Increase minibatch counter
  minibatch_counter_++;

  // F: Store weights/indicators
  memcpy((void *)old_weights, weights, inputs[1]->size() * sizeof(T));
  memcpy((void *)old_indicators, indicators, inputs[1]->size() * sizeof(T1));
}

template <typename T, typename T1>
void INQConvolution<T, T1>::backward_impl(const Variables &inputs,
                                          const Variables &outputs,
                                          const vector<bool> &prop_down,
                                          const vector<bool> &accum) {
  // Calculate the backward pass
  if (inputs.size() == 4) { // with bias
    convolution_->backward(Variables{inputs[0], inputs[1], inputs[3]}, outputs,
                           {prop_down[0], prop_down[1], prop_down[3]},
                           {accum[0], accum[1], accum[3]});
  } else { // without bias
    convolution_->backward(Variables{inputs[0], inputs[1]}, outputs,
                           {prop_down[0], prop_down[1]}, {accum[0], accum[1]});
  }
}

} // namespace nbla
