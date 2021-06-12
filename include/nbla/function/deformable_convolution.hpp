// Copyright 2021 Sony Corporation.
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

#ifndef NBLA_FUNCTION_DEFORMABLE_CONVOLUTION_HPP
#define NBLA_FUNCTION_DEFORMABLE_CONVOLUTION_HPP

#include <nbla/cpu.hpp>
#include <nbla/function.hpp>
#include <nbla/function_registry.hpp>

#include <memory>
#include <string>

namespace nbla {

using std::string;
using std::vector;

NBLA_REGISTER_FUNCTION_HEADER(DeformableConvolution, int, // base_axis
                              const vector<int> &,        // pad
                              const vector<int> &,        // stride
                              const vector<int> &,        // dilation
                              int,                        // group
                              int,                        // deformable_group
                              bool);                      // channel_last

/**
    @todo Write doc.

Inputs:

Outputs:

\ingroup FunctionImplGrp
 */
template <typename T>
class DeformableConvolution
    : public BaseFunction<int, const vector<int> &, const vector<int> &,
                          const vector<int> &, int, int, bool> {
protected:
  int base_axis_;
  const vector<int> pad_;
  const vector<int> stride_;
  const vector<int> dilation_;
  int group_;
  int deformable_group_;
  bool channel_last_;
  vector<int> kernel_;
  int channels_i_, channels_o_, channels_g_;
  vector<int> spatial_shape_i_;
  vector<int> spatial_shape_o_;
  int spatial_dims_;
  int outer_size_;
  int inner_size_i_;
  int inner_size_o_;
  int inner_size_k_;
  Variable col_;
  // Variables from deformable convolution
  int offset_size_i_;
  int mask_size_i_;

  // Variables for convolution by matrix multiplication
  int row_w_;
  int col_w_;
  int row_col_;
  int col_col_;
  int row_y_;
  int col_y_;

  bool with_mask_{false};
  bool with_bias_{false};

public:
  DeformableConvolution(const Context &ctx, int base_axis,
                        const vector<int> &pad, const vector<int> &stride,
                        const vector<int> &dilation, int group,
                        int deformable_group, bool channel_last)
      : BaseFunction(ctx, base_axis, pad, stride, dilation, group,
                     deformable_group, channel_last),
        base_axis_(base_axis), pad_(pad), stride_(stride), dilation_(dilation),
        group_(group), deformable_group_(deformable_group),
        channel_last_(channel_last) {}
  virtual ~DeformableConvolution() {}
  virtual shared_ptr<Function> copy() const {
    return create_DeformableConvolution(ctx_, base_axis_, pad_, stride_,
                                        dilation_, group_, deformable_group_,
                                        channel_last_);
  }
  virtual int min_inputs() { return 3; }
  virtual int min_outputs() { return 1; }
  virtual vector<dtypes> in_types() {
    return vector<dtypes>{get_dtype<T>(), get_dtype<T>(), get_dtype<T>(),
                          get_dtype<T>(), get_dtype<T>()};
  }
  virtual vector<dtypes> out_types() { return vector<dtypes>{get_dtype<T>()}; }
  virtual vector<string> allowed_array_classes() {
    return SingletonManager::get<Cpu>()->array_classes();
  }
  virtual string name() { return "DeformableConvolution"; }
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
    // all input gradient needs x, weight, mask
    if (j == 0 || j == 1 || j == 2) {
      return true;
    }
    if (j == 3 && with_mask_) {
      return true;
    }
    return false;
  }
};
}
#endif
