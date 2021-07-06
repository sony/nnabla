// Copyright 2019,2020,2021 Sony Corporation.
// Copyright 2021 Sony Group Corporation.
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

#ifndef NBLA_FUNCTION_WARP_BY_FLOW_HPP
#define NBLA_FUNCTION_WARP_BY_FLOW_HPP

#include <nbla/cpu.hpp>
#include <nbla/function.hpp>
#include <nbla/function_registry.hpp>

namespace nbla {

NBLA_REGISTER_FUNCTION_HEADER(WarpByFlow);

/**
Transform the image(s) *data* by *flow* field(s) of offset vectors such
that each output pixel corresponds to the input image pixel at the offset
location given by horizontal and vertical flow values. Both *data* and
*flow* are 4-D variables (in "NCHW" layout) with identical shape except
the *flow* channel dimension (which is always 2).

.. math::
    output_{n,c,y,x} = data_{n,c,y',x'} \text{ with }
    y' = y - flow_{n,1,y,x} \text{ and } x' = x - flow_{n,0,y,x}

The output pixel values at :math:`y'` and :math:`x'` locations are
obtained by bilinear interpolating between the 4 closest pixels of the
input image. Pixel values outside of the input image are implicitly
padded with the value of the closest boundary pixel.

Inputs:
- Input image data with shape `(N, Channels, Height, Width)`.
- Flow field vectors with shape `(N, 2, Height, Width)`.

Outputs:
- Transformed image data with shape `(N, Channels, Height, Width)`.

\ingroup FunctionImplGrp
 */
template <typename T> class WarpByFlow : public BaseFunction<> {
protected:
public:
  WarpByFlow(const Context &ctx) : BaseFunction(ctx) {}
  virtual ~WarpByFlow() {}
  virtual shared_ptr<Function> copy() const { return create_WarpByFlow(ctx_); }
  virtual int min_inputs() { return 2; }
  virtual int min_outputs() { return 1; }
  virtual vector<dtypes> in_types() {
    return vector<dtypes>{get_dtype<T>(), get_dtype<T>()};
  }
  virtual vector<dtypes> out_types() { return vector<dtypes>{get_dtype<T>()}; }
  virtual vector<string> allowed_array_classes() {
    return SingletonManager::get<Cpu>()->array_classes();
  }
  virtual string name() { return "WarpByFlow"; }
  virtual bool grad_depends_output_data(int i, int o) const { return false; }

protected:
  NBLA_API virtual void setup_impl(const Variables &inputs,
                                   const Variables &outputs);
  NBLA_API virtual void forward_impl(const Variables &inputs,
                                     const Variables &outputs);
  NBLA_API virtual void backward_impl(const Variables &inputs,
                                      const Variables &outputs,
                                      const vector<bool> &propagate_down,
                                      const vector<bool> &accum);
  virtual bool grad_depends_input_data_impl(int i, int j) const {
    if (i == 0) {
      if (j == 1)
        return true;
    }
    if (i == 1) {
      if (j == 0 || j == 1)
        return true;
    }
    return false;
  }
};
}
#endif
