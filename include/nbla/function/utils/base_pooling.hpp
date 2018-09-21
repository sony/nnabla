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

/** Pooling
*/
#ifndef __NBLA_FUNCTION_POOLING_HPP__
#define __NBLA_FUNCTION_POOLING_HPP__

#include <nbla/common.hpp>
#include <nbla/cpu.hpp>
#include <nbla/function.hpp>
#include <nbla/function_registry.hpp>

#include <algorithm>
#include <cmath>
#include <tuple>

namespace nbla {

using std::ceil;

/** Base class for pooling functions AveragePooling, SumPooling and MaxPooling.

Inputs:
- N-D array that has more than two dimensions.

Outputs:
- N-D array.

@note Only 2D pooling supported so far.
@tparam T Data type for computation.
@param kernel Shapes of kernel.
@param stride Subsampling factors of pooling.
@param ignore_border If false, a kernel overlapping border is also considered
                     for output unlike convolution.
@param pad Border padding values of dimensions. Padding will be added both sides
           of the dimension.
*/

template <typename T, class... Args>
class BasePooling : public BaseFunction<Args...> {
protected:
  vector<int> kernel_;
  vector<int> stride_;
  bool ignore_border_;
  vector<int> pad_;

public:
  // First arguments are
  // const vector<int> &kernel,
  // const vector<int> &stride, bool ignore_border, const vector<int> &pad
  BasePooling(const Context &ctx, Args... args)
      : BaseFunction<Args...>(ctx, args...) {
    auto t = std::forward_as_tuple(args...);
    kernel_ = std::get<0>(t);
    stride_ = std::get<1>(t);
    ignore_border_ = std::get<2>(t);
    pad_ = std::get<3>(t);
  }

  virtual ~BasePooling() {}
  virtual shared_ptr<Function> copy() const { return nullptr; }
  virtual vector<dtypes> in_types() { return vector<dtypes>{get_dtype<T>()}; }
  virtual vector<dtypes> out_types() { return vector<dtypes>{get_dtype<T>()}; }
  virtual int min_inputs() { return 1; }
  virtual int min_outputs() { return 1; }
  virtual string name() { return "BasePooling"; }
  virtual vector<string> allowed_array_classes() {
    return SingletonManager::get<Cpu>()->array_classes();
  }

protected:
  virtual void setup_impl(const Variables &inputs, const Variables &outputs) {

    // compute out shape
    const Shape_t inshape = inputs[0]->shape();
    const int s = inshape.size() - kernel_.size();

    if (stride_.size() == 0) {
      std::copy(stride_.begin(), stride_.end(), kernel_.begin());
    } else {
      NBLA_CHECK(kernel_.size() == stride_.size(), error_code::value,
                 "Length of kernel and stride must be same. "
                 "kernel: %d != stride: %d.",
                 kernel_.size(), stride_.size());
      NBLA_CHECK(kernel_.size() <= inshape.size(), error_code::value,
                 "Length of kernel must be less than length of inshape. "
                 "kernel: %d != inshape: %d.",
                 kernel_.size(), inshape.size());
    }
    NBLA_CHECK(kernel_.size() == 2, error_code::not_implemented,
               "2D Pooling is only supported so far.");

    vector<int> shape(kernel_.size());
    for (int i = 0; i < kernel_.size(); i++)
      shape[i] = inshape[i + s];
    NBLA_CHECK(kernel_.size() == pad_.size(), error_code::value,
               "Size of kernel and pad must be same. "
               "kernel: %d != pad: %d).",
               kernel_.size(), pad_.size());
    for (int i = 0; i < shape.size(); i++) {
      shape[i] += 2 * pad_[i];
      if (ignore_border_) {
        shape[i] = int((shape[i] - kernel_[i]) / stride_[i]) + 1;
      } else {
        shape[i] = ceil(shape[i] * 1.0 / stride_[i]);
      }
    }

    Shape_t outshape(inshape.size());
    for (int i = 0; i < inshape.size(); i++) {
      if (i < s)
        outshape[i] = inshape[i];
      else
        outshape[i] = shape[i - s];
    }

    outputs[0]->reshape(outshape, true);
  }
};
}
#endif
