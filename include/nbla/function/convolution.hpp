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
                              int);                // group

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
                          const vector<int> &, int> {
protected:
  int base_axis_;
  vector<int> pad_;
  vector<int> stride_;
  vector<int> dilation_;
  int group_;
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
  Convolution(const Context &ctx, int base_axis, const vector<int> &pad,
              const vector<int> &stride, const vector<int> &dilation, int group)
      : BaseFunction(ctx, base_axis, pad, stride, dilation, group),
        base_axis_(base_axis), pad_(pad), stride_(stride), dilation_(dilation),
        group_(group) {}

  virtual shared_ptr<Function> copy() const {
    return create_Convolution(ctx_, base_axis_, pad_, stride_, dilation_,
                              group_);
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
