// Copyright 2018,2019,2020,2021 Sony Corporation.
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

#ifndef NBLA_FUNCTION_INTERPOLATE_HPP
#define NBLA_FUNCTION_INTERPOLATE_HPP

#include <nbla/cpu.hpp>
#include <nbla/function.hpp>
#include <nbla/function_registry.hpp>

namespace nbla {

NBLA_REGISTER_FUNCTION_HEADER(Interpolate, const vector<int> &, const string &,
                              bool, bool, bool, bool);

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
@param half_pixel If true, in the coordinate transformation, 0.5 is added to the
output coordinate
and 0.5 is subtracted from the input coordinate after scaling. Default is
`False`.
This option is only applicable to the `mode` being linear.
@param half_pixel_for_nn This is a special argument to support the
backward-compatibility of the nearest neighbor interpolation. Default is
`False`.


\ingroup FunctionImplGrp
 */
template <typename T>
class Interpolate : public BaseFunction<const vector<int> &, const string &,
                                        bool, bool, bool, bool> {
protected:
  const vector<int> output_size_;
  const string mode_;
  bool align_corners_;
  bool half_pixel_;
  bool half_pixel_for_nn_;
  bool channel_last_;

public:
  Interpolate(const Context &ctx, const vector<int> &output_size,
              const string &mode, bool align_corners, bool half_pixel,
              bool half_pixel_for_nn, bool channel_last)
      : BaseFunction(ctx, output_size, mode, align_corners, half_pixel,
                     half_pixel_for_nn, channel_last),
        output_size_(output_size), mode_(mode), align_corners_(align_corners),
        half_pixel_(half_pixel), half_pixel_for_nn_(half_pixel_for_nn),
        channel_last_(channel_last) {}
  virtual ~Interpolate() {}
  virtual shared_ptr<Function> copy() const {
    return create_Interpolate(ctx_, output_size_, mode_, align_corners_,
                              half_pixel_, half_pixel_for_nn_, channel_last_);
  }
  virtual int min_inputs() { return 1; }
  virtual int min_outputs() { return 1; }
  virtual vector<dtypes> in_types() { return vector<dtypes>{get_dtype<T>()}; }
  virtual vector<dtypes> out_types() { return vector<dtypes>{get_dtype<T>()}; }
  virtual vector<string> allowed_array_classes() {
    return SingletonManager::get<Cpu>()->array_classes();
  }
  virtual string name() { return "Interpolate"; }
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
    return false;
  }
};
}
#endif
