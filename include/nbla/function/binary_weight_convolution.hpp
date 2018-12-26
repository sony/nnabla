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

/** BinaryWeightConvolution
 */
#ifndef __NBLA_FUNCTION_BINARYWEIGHTCONVOLUTION_HPP__
#define __NBLA_FUNCTION_BINARYWEIGHTCONVOLUTION_HPP__

#include <nbla/cpu.hpp>
#include <nbla/function.hpp>
#include <nbla/function_registry.hpp>

namespace nbla {

NBLA_REGISTER_FUNCTION_HEADER(BinaryWeightConvolution, int, const vector<int> &,
                              const vector<int> &, const vector<int> &, int,
                              float);

/** N-D Binary Weight Convolution with bias.

    Reference:
        Rastegari, Mohammad, et al. "XNOR-Net: ImageNet Classification Using
        Binary Convolutional Neural Networks." arXiv preprint
        arXiv:1603.05279 (2016).

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
- Alpha \f$1\f$-D array
  (\f$C'\f$).
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
class BinaryWeightConvolution
    : public BaseFunction<int, const vector<int> &, const vector<int> &,
                          const vector<int> &, int, float> {
protected:
  shared_ptr<Function> convolution_;
  shared_ptr<Function> abs_;
  shared_ptr<Function> sum_;
  shared_ptr<Function> div_;
  shared_ptr<Function> bin_;
  shared_ptr<Function> mul_;
  Variable scaled_weights_;

  int base_axis_;
  const vector<int> pad_;
  const vector<int> stride_;
  const vector<int> dilation_;
  int group_;

  float quantize_zero_to_;

  int channels_o_, col_w_;

public:
  BinaryWeightConvolution(const Context &ctx, int base_axis,
                          const vector<int> &pad, const vector<int> &stride,
                          const vector<int> &dilation, int group,
                          float quantize_zero_to)
      : BaseFunction(ctx, base_axis, pad, stride, dilation, group,
                     quantize_zero_to),
        base_axis_(base_axis), pad_(pad), stride_(stride), dilation_(dilation),
        group_(group), quantize_zero_to_(quantize_zero_to) {}
  virtual ~BinaryWeightConvolution() {}
  virtual shared_ptr<Function> copy() const {
    return create_BinaryWeightConvolution(ctx_, base_axis_, pad_, stride_,
                                          dilation_, group_, quantize_zero_to_);
  }
  virtual vector<dtypes> in_types() {
    return vector<dtypes>{get_dtype<T>(), get_dtype<T>(), get_dtype<T>(),
                          get_dtype<T>()};
  }
  virtual vector<dtypes> out_types() { return vector<dtypes>{get_dtype<T>()}; }
  virtual int min_inputs() { return 4; }
  virtual int min_outputs() { return 1; }
  virtual string name() { return "BinaryWeightConvolution"; }
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
