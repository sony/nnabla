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

#ifndef NBLA_FUNCTION_INTERPOLATE_HPP
#define NBLA_FUNCTION_INTERPOLATE_HPP

#include <nbla/cpu.hpp>
#include <nbla/function.hpp>
#include <nbla/function_registry.hpp>

namespace nbla {

NBLA_REGISTER_FUNCTION_HEADER(Interpolate, const vector<int> &, const string &,
                              bool);

/**
Resize an ND array with interpolation.

The last output_size.size() dimensions of the input are considered as the
spatial dimensions.

Inputs:
- x: N-D array with an arbitrary number of dimensions.

Outputs:
- y: N-D array. The shape are the same as the input except that the last
dimensions are replaced with the output_size.

@tparam T Data type for computation.
@param output_size The Spatial sizes resized to.
@param mode Interpolation mode chosen from linear or nearest.
@param align_corners If true, the corner pixels are aligned to preserve the
values of the input corner pixels.

\ingroup FunctionImplGrp
 */
template <typename T>
class Interpolate
    : public BaseFunction<const vector<int> &, const string &, bool> {
protected:
  const vector<int> output_size_;
  const string mode_;
  bool align_corners_;

public:
  Interpolate(const Context &ctx, const vector<int> &output_size,
              const string &mode, bool align_corners)
      : BaseFunction(ctx, output_size, mode, align_corners),
        output_size_(output_size), mode_(mode), align_corners_(align_corners) {}
  virtual ~Interpolate() {}
  virtual shared_ptr<Function> copy() const {
    return create_Interpolate(ctx_, output_size_, mode_, align_corners_);
  }
  virtual int min_inputs() { return 1; }
  virtual int min_outputs() { return 1; }
  virtual vector<dtypes> in_types() { return vector<dtypes>{get_dtype<T>()}; }
  virtual vector<dtypes> out_types() { return vector<dtypes>{get_dtype<T>()}; }
  virtual vector<string> allowed_array_classes() {
    return SingletonManager::get<Cpu>()->array_classes();
  }
  virtual string name() { return "Interpolate"; }

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
