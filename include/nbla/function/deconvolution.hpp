// Copyright 2017,2018,2019,2020,2021 Sony Corporation.
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

// deconvolution.hpp

#ifndef __NBLA_FUNCTION_DECONVOLUTION_HPP__
#define __NBLA_FUNCTION_DECONVOLUTION_HPP__

#include <nbla/cpu.hpp>
#include <nbla/function.hpp>
#include <nbla/function_registry.hpp>

#include <memory>
#include <string>

namespace nbla {

using std::string;
using std::vector;

NBLA_REGISTER_FUNCTION_HEADER(Deconvolution, int,   // base_axis
                              const vector<int> &,  // pad
                              const vector<int> &,  // stride
                              const vector<int> &,  // dilation
                              int,                  // group
                              bool,                 // channel_last
                              const vector<int> &); // output_padding

/** N-D Deconvolution with bias operates backward convolution (derivative of
output wrt input) plus channel-wise learned bias. The weights must be given with
the same format as in forward convolution, hence the number of input channels
(can be seen as output channels of forward convolution) comes to the first
dimension, and the second dimension has number of the output channels divided by
group.

Inputs (\f$B\f$ is base_axis):
- Input \f$(B + 1 + N)\f$-D array
  (\f$M_1 \times ... \times M_B \times C \times L_1 \times ... \times L_N\f$).
- Weight \f$(2 + N)\f$-D array
  (\f$C' \times C \times K_1 \times ... \times K_N\f$).
- (optional) Bias vector (\f$C'\f$).

@sa Convolution for documentation of parameters.
@sa Deconvolution is introduced in Shelhamer et al., Fully Convolutional
Networks for Semantic Segmentation. https://arxiv.org/abs/1605.06211

\ingroup FunctionImplGrp
 */
template <typename T>
class Deconvolution
    : public BaseFunction<int, const vector<int> &, const vector<int> &,
                          const vector<int> &, int, bool, const vector<int> &> {
protected:
  // Note that member variables below are actually copied from Convolution. But
  // the meanings of input and output are back and forth. We consider the output
  // of this function as the input of forward Convolution.
  int base_axis_;
  vector<int> pad_;
  vector<int> stride_;
  vector<int> dilation_;
  int group_;
  bool channel_last_;
  vector<int> output_padding_;
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

  // Variables for convolution by matrix multiplication
  int row_w_;
  int col_w_;
  int row_col_;
  int col_col_;
  int row_y_;
  int col_y_;

public:
  Deconvolution(const Context &ctx, int base_axis, const vector<int> &pad,
                const vector<int> &stride, const vector<int> &dilation,
                int group, bool channel_last, const vector<int> &output_padding)
      : BaseFunction(ctx, base_axis, pad, stride, dilation, group, channel_last,
                     output_padding),
        base_axis_(base_axis), pad_(pad), stride_(stride), dilation_(dilation),
        group_(group), channel_last_(channel_last),
        output_padding_(output_padding) {}

  virtual shared_ptr<Function> copy() const {
    return create_Deconvolution(ctx_, base_axis_, pad_, stride_, dilation_,
                                group_, channel_last_, output_padding_);
  }

  virtual vector<dtypes> in_types() {
    return vector<dtypes>{get_dtype<T>(), get_dtype<T>(), get_dtype<T>()};
  }
  virtual vector<dtypes> out_types() { return vector<dtypes>{get_dtype<T>()}; }
  virtual int min_inputs() { return 2; }
  virtual int min_outputs() { return 1; }
  virtual string name() { return "Deconvolution"; }
  virtual vector<string> allowed_array_classes() {
    return SingletonManager::get<Cpu>()->array_classes();
  }
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
    if (i == 0 && j == 1) {
      return true;
    }
    if (i == 1 && j == 0) {
      return true;
    }
    return false;
  }
};
}
#endif
