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

/** Pads given N-D array with specified sizes of dimensions.
Padding begins at the last dimension of x and continues for the specified
padding dimension.

Inputs:
- x: N-D array.
- pad_width: n-elem tuple, where n/2 <= input dimensions and n is even.
  len(pad_width)/2 represents the padding dimension. (e.g. 1D, 2D, 3D etc.)
  (Currently padding upto 3D is supported)
- mode - Padding mode is one of the following. Default: constant.
          1)constant : Elements in pad region are filled with constant_value.
          2)replicate : Padded elements are filled with the values in nearest
edges.
          3)reflect : Padded with the reflection of the vector mirrored on the
first and last values of the vector along each axis.
          (Currently only constant mode is supported)
- constant_value - Constant values filled in padded regions if mode is constant.
Default: 0

Outputs:
- Padded N-D array (e.g. (B, C, H, W) shape) where dimension depends on
pad_width.
ndim() of output N-D array will be same as ndim() of input N-D array.
for 1D padding:
    N-D input array with padding of the form (padLeft, padRight)
    output N-D array dimension (B, C, H, padLeft + W + padRight)
for 2D padding:
    N-D input array with padding of the form (padTop, padBottom, padLeft,
padRight).
    output N-D array dimension (B, C, padTop + H + padBottom, padLeft + W +
padRight)
for 3D padding:
    N-D input array with padding of the form (pasFront, padBack, padTop,
padBottom, padLeft, padRight).
    output N-D array dimension (B, padFront + C + padBack, padTop + H +
padBottom, padLeft + W + padRight)

@tparam T Data type for computation.
\ingroup FunctionImplGrp
 */

template <typename T>
class Pad : public BaseFunction<const vector<int> &, const string &, float> {
protected:
  const vector<int> pad_width_;
  const string mode_;
  float constant_value_;

public:
  Pad(const Context &ctx, const vector<int> &pad_width, const string &mode,
      float constant_value)
      : BaseFunction(ctx, pad_width, mode, constant_value),
        pad_width_(pad_width), mode_(mode), constant_value_(constant_value) {
    pad_mode_["constant"] = p_constant;
    pad_mode_["replicate"] = p_replicate;
    pad_mode_["reflect"] = p_reflect;
  }
  virtual ~Pad() {}
  virtual shared_ptr<Function> copy() const {
    return create_Pad(ctx_, pad_width_, mode_, constant_value_);
  }
  virtual int min_inputs() { return 1; }
  virtual int min_outputs() { return 1; }
  virtual vector<dtypes> in_types() { return vector<dtypes>{get_dtype<T>()}; }
  virtual vector<dtypes> out_types() { return vector<dtypes>{get_dtype<T>()}; }
  virtual vector<string> allowed_array_classes() {
    return SingletonManager::get<Cpu>()->array_classes();
  }
  virtual string name() { return "Pad"; }

  typedef enum pad_mode { p_constant, p_replicate, p_reflect } pad_mode;
  std::map<std::string, pad_mode> pad_mode_;

protected:
  NBLA_API virtual void setup_impl(const Variables &inputs,
                                   const Variables &outputs);
  NBLA_API virtual void forward_impl(const Variables &inputs,
                                     const Variables &outputs);
  NBLA_API virtual void backward_impl(const Variables &inputs,
                                      const Variables &outputs,
                                      const vector<bool> &propagate_down,
                                      const vector<bool> &accum);
};
}
#endif
