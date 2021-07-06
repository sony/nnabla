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

/** DepthwiseConvolution
 */
#ifndef __NBLA_FUNCTION_DEPTHWISECONVOLUTION_HPP__
#define __NBLA_FUNCTION_DEPTHWISECONVOLUTION_HPP__

#include <nbla/cpu.hpp>
#include <nbla/function.hpp>
#include <nbla/function_registry.hpp>

#include <memory>
#include <string>

namespace nbla {

NBLA_REGISTER_FUNCTION_HEADER(DepthwiseConvolution, int, const vector<int> &,
                              const vector<int> &, const vector<int> &, int);

/** N-D Depthwise Convolution with bias.

Inputs (\f$B\f$ is base_axis):
- Input \f$(B + 1 + N)\f$-D array
  (\f$M_1 \times ... \times M_B \times C \times L_1 \times ... \times L_N\f$).
- Weight \f$(1 + N)\f$-D array
  (\f$C \times K_1 \times ... \times K_N\f$).
- (optional) Bias vector (\f$C\f$).

Outputs:
- \f$(B + 1 + N)\f$-D array
  (\f$ M_1 \times ... \times M_B \times C \times L'_1 \times ... \times L'_N
\f$).

@tparam T Data type for computation.
@param base_axis Base axis of Convolution operation. Dimensions up to base_axis
is treated as sample dimension.
@param pad Padding sizes for dimensions.
@param stride Stride sizes for dimensions.
@param dilation Dilation sizes for dimensions.
@param multiplier Number of output feature maps per input feature map.

@sa Reference:
- F. Chollet: Chollet, Francois. "Xception: Deep Learning with Depthwise
Separable Convolutions.
<https://arxiv.org/abs/1610.02357>

 */
template <typename T>
class DepthwiseConvolution
    : public BaseFunction<int, const vector<int> &, const vector<int> &,
                          const vector<int> &, int> {
protected:
  int base_axis_;
  const vector<int> padding_;
  const vector<int> stride_;
  const vector<int> dilation_;
  int multiplier_;

  vector<int> sample_shape_;
  vector<int> outmap_shape_;
  vector<int> kernel_shape_;
  int sample_channels_;
  int outmap_channels_;
  int sample_size_;
  int outmap_size_;
  int kernel_size_;
  int batch_size_;

  Variable col_;

public:
  DepthwiseConvolution(const Context &ctx, int base_axis,
                       const vector<int> &pad, const vector<int> &stride,
                       const vector<int> &dilation, int multiplier)
      : BaseFunction<int, const vector<int> &, const vector<int> &,
                     const vector<int> &, int>(ctx, base_axis, pad, stride,
                                               dilation, multiplier),
        base_axis_(base_axis), padding_(pad), stride_(stride),
        dilation_(dilation), multiplier_(multiplier) {}
  virtual ~DepthwiseConvolution() {}
  virtual shared_ptr<Function> copy() const {
    return create_DepthwiseConvolution(ctx_, base_axis_, padding_, stride_,
                                       dilation_, multiplier_);
  }
  virtual vector<dtypes> in_types() {
    return vector<dtypes>{get_dtype<T>(), get_dtype<T>(), get_dtype<T>()};
  }
  virtual vector<dtypes> out_types() { return vector<dtypes>{get_dtype<T>()}; }
  virtual int min_inputs() { return 2; }
  virtual int min_outputs() { return 1; }
  virtual string name() { return "DepthwiseConvolution"; }
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
