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

// convolution.hpp

#ifndef __NBLA_FUNCTION_CONVOLUTION_HPP__
#define __NBLA_FUNCTION_CONVOLUTION_HPP__

#include <nbla/cpu.hpp>
#include <nbla/function.hpp>
#include <nbla/function_registry.hpp>

#include <memory>
#include <string>

namespace nbla {

using std::string;
using std::vector;

NBLA_REGISTER_FUNCTION_HEADER(Convolution, int,    // base_axis
                              const vector<int> &, // pad
                              const vector<int> &, // stride
                              const vector<int> &, // dilation
                              int,                 // group
                              bool);               // channel_last

/** N-D Convolution with bias.

Inputs (\f$B\f$ is base_axis):
- Input \f$(B + 1 + N)\f$-D array
  (\f$M_1 \times ... \times M_B \times C \times L_1 \times ... \times L_N\f$).
- Weight \f$(2 + N)\f$-D array
  (\f$C' \times C \times K_1 \times ... \times K_N\f$).
- (optional) Bias vector (\f$C'\f$).

Outputs:
- \f$(B + 1 + N)\f$-D array
  (\f$ M_1 \times ... \times M_B \times C' \times L'_1 \times ... \times L'_N
\f$).

@tparam T Data type for computation.
@param base_axis Base axis of Convolution operation. Dimensions up to base_axis
is treated as sample dimension.
@param pad Padding sizes for dimensions.
@param stride Stride sizes for dimensions.
@param dilation Dilation sizes for dimensions.
@param group Number of groups of channels. This makes connections across
channels sparser by grouping connections along map direction.
@param channel_last If True, the last dimension is considered as channel
dimension, a.k.a NHWC order.

@sa For Dilated Convolution (a.k.a a trous), refer to:
- Chen et al., DeepLab: Semantic Image Segmentation with Deep Convolutional
Nets, Atrous Convolution, and Fully Connected CRFs.
https://arxiv.org/abs/1606.00915
- Yu et al., Multi-Scale Context Aggregation by Dilated Convolutions.
https://arxiv.org/abs/1511.07122

\ingroup FunctionImplGrp
 */
template <typename T>
class Convolution
    : public BaseFunction<int, const vector<int> &, const vector<int> &,
                          const vector<int> &, int, bool> {
protected:
  int base_axis_;
  vector<int> pad_;
  vector<int> stride_;
  vector<int> dilation_;
  int group_;
  bool channel_last_;
  vector<int> kernel_;
  Size_t channels_i_, channels_o_, channels_g_;
  vector<int> spatial_shape_i_;
  vector<int> spatial_shape_o_;
  int spatial_dims_;
  Size_t outer_size_;
  Size_t inner_size_i_;
  Size_t inner_size_o_;
  Size_t inner_size_k_;
  Variable col_;

  // Variables for convolution by matrix multiplication
  Size_t row_w_;
  Size_t col_w_;
  Size_t row_col_;
  Size_t col_col_;
  Size_t row_y_;
  Size_t col_y_;

public:
  Convolution(const Context &ctx, int base_axis, const vector<int> &pad,
              const vector<int> &stride, const vector<int> &dilation, int group,
              bool channel_last)
      : BaseFunction(ctx, base_axis, pad, stride, dilation, group,
                     channel_last),
        base_axis_(base_axis), pad_(pad), stride_(stride), dilation_(dilation),
        group_(group), channel_last_(channel_last) {}

  virtual shared_ptr<Function> copy() const {
    return create_Convolution(ctx_, base_axis_, pad_, stride_, dilation_,
                              group_, channel_last_);
  }

  virtual vector<dtypes> in_types() {
    return vector<dtypes>{get_dtype<T>(), get_dtype<T>(), get_dtype<T>()};
  }
  virtual vector<dtypes> out_types() { return vector<dtypes>{get_dtype<T>()}; }
  virtual int min_inputs() { return 2; }
  virtual int min_outputs() { return 1; }
  virtual string name() { return "Convolution"; }
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
