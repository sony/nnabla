// Copyright 2020,2021 Sony Corporation.
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

#ifndef NBLA_FUNCTION_WARP_BY_GRID_HPP
#define NBLA_FUNCTION_WARP_BY_GRID_HPP

#include <nbla/cpu.hpp>
#include <nbla/function.hpp>
#include <nbla/function_registry.hpp>

namespace nbla {

NBLA_REGISTER_FUNCTION_HEADER(WarpByGrid, const string &, const string &, bool,
                              bool);

namespace warp_by_grid {
enum PADDING_MODE {
  zero = 0,
  repeat,
  reflect,
};
}

/**

Inputs:
- x: Input data to be warped with the shape (\f$B \times C \times H_{in} \times
W_{in}\f$) for 2D and (\f$B \times C \times D_{in} \times H_{in} \times
W_{in}\f$) for 3D.
- grid: Grid warping the input data with the shape (\f$B \times H_{out} \times
W_{out} \times 2\f$) for 2D and (\f$B \times D_{out} \times H_{out} \times
W_{out} \times 3\f$) for 3D. The last dimension of 2 is for (x, y) or (x, y, z).

Outputs:
- y: Output data warped by the grid.

@tparam T Data type for computation.
@param mode Interpolation mode, linear or nearest.
@param padding_mode Padding mode when the grid value is outside [-1, 1]. If this
is "zero", 0 is used for padding. "reflect" uses the values reflected at the
ends of the original input data like the mirror. "repeat" used the values at the
ends of the original input data.
@param align_corners The target grid normalized in [-1, 1] is scaled by `size -
1 / size` according to the respective spatial size (e.g., \f$H\f$ and \f$W\f$)
before the transformation if this is `False`. If this is `True`, the top-left
and bottom-right pixcels correspond to (-1, -1) and (1, 1) respectively.
@param channel_last If True, the last dimension is considered as channel
dimension, a.k.a NHWC order.

\ingroup FunctionImplGrp
 */
template <typename T>
class WarpByGrid
    : public BaseFunction<const string &, const string &, bool, bool> {
protected:
  const string mode_;
  const string padding_mode_;
  warp_by_grid::PADDING_MODE padding_mode_t_;
  bool align_corners_;
  bool channel_last_;

public:
  WarpByGrid(const Context &ctx, const string &mode, const string &padding_mode,
             bool align_corners, bool channel_last)
      : BaseFunction(ctx, mode, padding_mode, align_corners, channel_last),
        mode_(mode), padding_mode_(padding_mode), align_corners_(align_corners),
        channel_last_(channel_last) {}
  virtual ~WarpByGrid() {}
  virtual shared_ptr<Function> copy() const {
    return create_WarpByGrid(ctx_, mode_, padding_mode_, align_corners_,
                             channel_last_);
  }
  virtual int min_inputs() { return 2; }
  virtual int min_outputs() { return 1; }
  virtual vector<dtypes> in_types() {
    return vector<dtypes>{get_dtype<T>(), get_dtype<T>()};
  }
  virtual vector<dtypes> out_types() { return vector<dtypes>{get_dtype<T>()}; }
  virtual vector<string> allowed_array_classes() {
    return SingletonManager::get<Cpu>()->array_classes();
  }
  virtual string name() { return "WarpByGrid"; }
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
  virtual bool grad_depends_input_data_impl(int i, int j) const { return true; }
};
}
#endif
