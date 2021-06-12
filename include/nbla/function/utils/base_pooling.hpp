// Copyright 2017,2018,2019,2020,2021 Sony Corporation.
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

using std::vector;

/**
   Get output shape of pooling from input configuration.
 */
struct PoolingConfiguration {
  vector<int> inshape;
  vector<int> kernel;
  vector<int> stride;
  vector<int> pad;
  bool ignore_border;
  bool channel_last;
  vector<int> outshape;
  int base_axis;
  NBLA_API PoolingConfiguration(const vector<int> &inshape,
                                const vector<int> &kernel,
                                const vector<int> &stride,
                                const vector<int> &pad, bool ignore_border,
                                bool channel_last);
};

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
  bool channel_last_;

public:
  typedef T data_type;
  typedef BasePooling<T, Args...> base_pooling_type;
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
    channel_last_ = std::get<4>(t);
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
    PoolingConfiguration cfg(vector<int>(inshape.begin(), inshape.end()),
                             kernel_, stride_, pad_, ignore_border_,
                             channel_last_);
    stride_ = cfg.stride;
    Shape_t outshape(cfg.outshape.cbegin(), cfg.outshape.cend()); // cast
    outputs[0]->reshape(outshape, true);
  }
};
}
#endif
