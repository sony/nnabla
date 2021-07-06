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

// binary_connect_convolution.hpp

#ifndef __NBLA_FUNCTION_BINARY_CONNECT_CONVOLUTION_HPP__
#define __NBLA_FUNCTION_BINARY_CONNECT_CONVOLUTION_HPP__

#include <nbla/cpu.hpp>
#include <nbla/function.hpp>
#include <nbla/function/convolution.hpp>
#include <nbla/function_registry.hpp>

#include <memory>
#include <string>

namespace nbla {

using std::string;
using std::vector;

NBLA_REGISTER_FUNCTION_HEADER(BinaryConnectConvolution,
                              int,                 // base_axis
                              const vector<int> &, // pad
                              const vector<int> &, // stride
                              const vector<int> &, // dilation
                              int,                 // group
                              float);

/** N-D BinaryConnect Convolution with bias.

    Reference:
    M. Courbariaux, Y. Bengio, and J.-P. David. "BinaryConnect:
    Training Deep Neural Networks with binary weights during propagations."
    Advances in Neural Information Processing Systems. 2015.

    NOTES:

    1) if you would like to share weights between some layers, please
    make sure to share the standard, floating value weights (input parameter #2)
    and not the binarized weights (input parameter #3)

    2) Only after a call to forward() the weights and the binary weights are in
    sync, not after a call to backward(). If wanting to store the parameters of
    the network, remember to call forward() once before doing so, otherwise the
    weights and the binary weights will not be in sync.

Inputs (\f$B\f$ is base_axis):
- Input \f$(B + 1 + N)\f$-D array
  (\f$M_1 \times ... \times M_B \times C \times L_1 \times ... \times L_N\f$).
- Weight \f$(2 + N)\f$-D array
  (\f$C' \times C \times K_1 \times ... \times K_N\f$).
- Binary Weight \f$(2 + N)\f$-D array
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
@param quantize_zero_to Input value at zero is quantized to this value.

@sa For Dilated Convolution (a.k.a a trous), refer to:
- Chen et al., DeepLab: Semantic Image Segmentation with Deep Convolutional
Nets, Atrous Convolution, and Fully Connected CRFs.
https://arxiv.org/abs/1606.00915
- Yu et al., Multi-Scale Context Aggregation by Dilated Convolutions.
https://arxiv.org/abs/1511.07122

\ingroup FunctionImplGrp
 */
template <typename T>
class BinaryConnectConvolution
    : public BaseFunction<int, const vector<int> &, const vector<int> &,
                          const vector<int> &, int, float> {
protected:
  shared_ptr<Function> sign_;
  shared_ptr<Function> convolution_;

  int base_axis_;
  vector<int> pad_;
  vector<int> stride_;
  vector<int> dilation_;
  int group_;

  float quantize_zero_to_;

public:
  BinaryConnectConvolution(const Context &ctx, int base_axis,
                           const vector<int> &pad, const vector<int> &stride,
                           const vector<int> &dilation, int group,
                           float quantize_zero_to)
      : BaseFunction(ctx, base_axis, pad, stride, dilation, group,
                     quantize_zero_to),
        base_axis_(base_axis), pad_(pad), stride_(stride), dilation_(dilation),
        group_(group), quantize_zero_to_(quantize_zero_to) {}

  virtual shared_ptr<Function> copy() const {
    return create_BinaryConnectConvolution(
        ctx_, base_axis_, pad_, stride_, dilation_, group_, quantize_zero_to_);
  }

  virtual vector<dtypes> in_types() {
    return vector<dtypes>{get_dtype<T>(), get_dtype<T>(), get_dtype<T>(),
                          get_dtype<T>()};
  }
  virtual vector<dtypes> out_types() { return vector<dtypes>{get_dtype<T>()}; }
  virtual int min_inputs() { return 3; }
  virtual int min_outputs() { return 1; }
  virtual string name() { return "BinaryConnectConvolution"; }
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
    if (i == 0 && j == 2) {
      return true;
    }
    if (i == 2 && j == 0) {
      return true;
    }
    return false;
  }
};
}
#endif
