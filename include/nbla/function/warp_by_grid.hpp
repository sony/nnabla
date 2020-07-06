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
    @todo Write doc.

Inputs:

Outputs:

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
