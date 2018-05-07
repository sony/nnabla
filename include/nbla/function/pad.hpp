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


#ifndef NBLA_FUNCTION_PAD_HPP
#define NBLA_FUNCTION_PAD_HPP

#include <nbla/cpu.hpp>
#include <nbla/function.hpp>
#include <nbla/function_registry.hpp>

namespace nbla {

NBLA_REGISTER_FUNCTION_HEADER(Pad, const vector<int> &, const string &, float);

/** Pads given tensor with constant padding.
len(pad_width)/2 represents the padding dimension.
The dimensions that get padded begins with the last dimension and moves forward.

Inputs:
- x: N-D array.
- pad_width: n-elem tuple, where n/2 ≤ input dimensions and n is even.
- mode – ‘constant’, ‘reflect’ 'edge’. Default: ‘constant’
- constant_value – fill value for ‘constant’ padding. Default: 0

Outputs:
- N-D array (B, C, H, W) where dimension depends on pad_width.
for 1D padding:
    3D input tensor with padding of the form (padLeft, padRight)
    output tensor dimension (B, C, H, padLeft+W+padRight)
for 2D:
    4D input tensor with padding of the form (padTop, padBottom, padLeft, padRight).
    output tensor dimension (B, C, padTop+H+padBottom, padLeft+W+padRight)

@tparam T Data type for computation.
\ingroup FunctionImplGrp
 */

template <typename T> class Pad : public BaseFunction<const vector<int> &, const string &, float> {
protected:
  const vector<int> pad_width_;
  const string mode_;
  float constant_value_;
public:
  Pad(const Context &ctx, const vector<int> & pad_width, const string & mode, float constant_value) : BaseFunction(ctx, pad_width, mode, constant_value)
  , pad_width_(pad_width)
  , mode_(mode)
  , constant_value_(constant_value)
    {}
  virtual ~Pad() {}
  virtual shared_ptr<Function> copy() const {
    return create_Pad(ctx_, pad_width_, mode_, constant_value_);
  }
  virtual int min_inputs() { return 1; }
  virtual int min_outputs() { return 1; }
  virtual vector<dtypes> in_types() {
    return vector<dtypes>{get_dtype<T>()};
  }
  virtual vector<dtypes> out_types() {
    return vector<dtypes>{get_dtype<T>()};
  }
  virtual vector<string> allowed_array_classes() {
    return SingletonManager::get<Cpu>()->array_classes();
  }
  virtual string name() { return "Pad"; }

protected:
  NBLA_API virtual void setup_impl(const Variables &inputs, const Variables &outputs);
  NBLA_API virtual void forward_impl(const Variables &inputs, const Variables &outputs);
  NBLA_API virtual void backward_impl(const Variables &inputs, const Variables &outputs,
				      const vector<bool> &propagate_down,
				      const vector<bool> &accum);
};
}
#endif
