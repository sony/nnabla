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
#include <nbla/function/random_choice.hpp>
#include <nbla/random_manager.hpp>
#include <nbla/utils/nd_index.hpp>
#include <nbla/variable.hpp>
#include <numeric>

namespace nbla {

NBLA_REGISTER_FUNCTION_SOURCE(RandomChoice, const vector<int> &, bool, int);

template <typename T>
void RandomChoice<T>::setup_impl(const Variables &inputs,
                                 const Variables &outputs) {
  NBLA_CHECK(inputs[0]->shape() == inputs[1]->shape(), error_code::value,
             "Dimensions of inputs must match. "
             "inputs[0]: %s != inputs[1]: %s.",
             string_join(inputs[0]->shape(), string(", ")).c_str(),
             string_join(inputs[1]->shape(), string(", ")).c_str());

  Shape_t ishape(inputs[0]->shape());
  Shape_t oshape(ishape.begin(), ishape.end() - 1);

  if (shape_.size() > 0)
    oshape.insert(oshape.end(), shape_.begin(), shape_.end());
  else
    oshape.push_back(1);

  outer_loop_ = ndi::outer_size(oshape, ishape.size() - 1);
  inner_loop_ = ndi::inner_size(oshape, ishape.size() - 1);

  if (replace_ == false) {
    NBLA_CHECK(inner_loop_ <= ishape.back(), error_code::value,
               "Can not sample more values than population without replacement."
               " product of shape %d > last dim of inputs %d",
               inner_loop_, ishape.back());
  }

  idxbuf_.reshape(oshape, true);
  outputs[0]->reshape(oshape, true);

  rgen_ = std::mt19937((seed_ == -1 ? std::random_device()() : seed_));
}

template <typename T>
void RandomChoice<T>::forward_impl(const Variables &inputs,
                                   const Variables &outputs) {
  using std::uniform_real_distribution;
  using std::partial_sum;
  using std::count_if;
  using std::vector;

  auto x_data = inputs[0]->get_data_pointer<T>(this->ctx_);
  auto w_data = inputs[1]->get_data_pointer<T>(this->ctx_);
  auto y_data = outputs[0]->cast_data_and_get_pointer<T>(this->ctx_, true);
  auto idxbuf = idxbuf_.cast_data_and_get_pointer<int>(this->ctx_, true);
  auto w_size = inputs[0]->shape().back(); // size of each weight vector
  auto less_0 = std::bind(std::less<T>(), std::placeholders::_1, (T)0);
  std::mt19937 &rgen =
      seed_ == -1 ? SingletonManager::get<RandomManager>()->get_rand_generator()
                  : rgen_;

  if (replace_ == true) {
    vector<T> w_sum(w_size);
    for (int b = 0; b < this->outer_loop_; b++) {
      NBLA_CHECK(std::none_of(w_data, w_data + w_size, less_0),
                 error_code::value, "Negative weights are not allowed.");
      partial_sum(w_data, w_data + w_size, w_sum.begin());
      NBLA_CHECK(w_sum.back() > (T)0, error_code::value,
                 "At least one weight must be greater zero.")
      uniform_real_distribution<> uniform(0, w_sum.back());
      for (int i = 0; i < this->inner_loop_; i++) {
        T u = uniform(rgen);
        auto index = w_size - 1;
        for (int i = 0; i < w_size; i++) {
          if (u < w_sum[i]) {
            index = i;
            break;
          }
        }
        *y_data++ = x_data[index];
        *idxbuf++ = index;
      }
      w_data += w_size;
      x_data += w_size;
    }
  } else {
    vector<T> w_vec(w_size), w_sum(w_size);
    for (int b = 0; b < this->outer_loop_; b++) {
      auto greater_zero = [](T v) { return v > (T)0; };
      auto positive_weights = count_if(w_data, w_data + w_size, greater_zero);
      NBLA_CHECK(positive_weights >= this->inner_loop_, error_code::value,
                 "insufficient positive weights for sampling w/o replacement");
      w_vec.assign(w_data, w_data + w_size);
      int have = 0, need = this->inner_loop_;
      while (have < this->inner_loop_) {
        partial_sum(w_vec.begin(), w_vec.end(), w_sum.begin());
        uniform_real_distribution<> uniform(0, w_sum.back());
        while (need--) {
          T u = uniform(rgen);
          for (int i = 0; i < w_size; i++) {
            if (u < w_sum[i]) {
              if (w_vec[i] > 0) {
                *y_data++ = x_data[i];
                *idxbuf++ = i;
                w_vec[i] = 0;
                have++;
              }
              break;
            }
          }
        }
        need = this->inner_loop_ - have;
      }
      w_data += w_size;
      x_data += w_size;
    }
  }
}

template <typename T>
void RandomChoice<T>::backward_impl(const Variables &inputs,
                                    const Variables &outputs,
                                    const vector<bool> &propagate_down,
                                    const vector<bool> &accum) {
  if (!(propagate_down[0] || propagate_down[1])) {
    return;
  }

  if ((propagate_down[0]) && (!accum[0]))
    inputs[0]->grad()->zero();

  if ((propagate_down[1]) && (!accum[1]))
    inputs[1]->grad()->zero();

  auto w_size = inputs[0]->shape().back();

  if (propagate_down[0]) {
    auto x_grad = inputs[0]->cast_grad_and_get_pointer<T>(this->ctx_, false);
    auto y_grad = outputs[0]->get_grad_pointer<T>(this->ctx_);
    auto idxbuf = idxbuf_.get_data_pointer<int>(this->ctx_);
    for (int b = 0; b < this->outer_loop_; b++) {
      for (int i = 0; i < this->inner_loop_; i++) {
        x_grad[*idxbuf++] += *y_grad++;
      }
      x_grad += w_size;
    }
  }

  if (propagate_down[1]) {
    auto w_grad = inputs[1]->cast_grad_and_get_pointer<T>(this->ctx_, false);
    auto y_grad = outputs[0]->get_grad_pointer<T>(this->ctx_);
    auto idxbuf = idxbuf_.get_data_pointer<int>(this->ctx_);
    for (int b = 0; b < this->outer_loop_; b++) {
      for (int i = 0; i < this->inner_loop_; i++) {
        w_grad[*idxbuf++] += *y_grad++;
      }
      w_grad += w_size;
    }
  }
}
}
