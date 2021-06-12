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

/** VATNoise
 */
#ifndef __NBLA_FUNCTION_VATNOISE_HPP__
#define __NBLA_FUNCTION_VATNOISE_HPP__

#include <nbla/function.hpp>
#include <nbla/function_registry.hpp>

namespace nbla {

NBLA_REGISTER_FUNCTION_HEADER(VATNoise, int, float);

/** VAT Noise
   This layer is specialized for getting noise for virtual adversarial training.
   In the backward process, the input noise will be replaced with the output
   gradient.

   Inputs:
   - noise n

   Outputs:
   - eps_scaled noise eps_n

   @sa Miyato et.al, Distributional Smoothing with Virtual Adversarial Training
   https://arxiv.org/abs/1507.00677

   @tparam T Data type for computation.
   \ingroup FunctionImplGrp
 */
template <typename T> class VATNoise : public BaseFunction<int, float> {
protected:
  int base_axis_;
  float eps_;

public:
  VATNoise(const Context &ctx, int base_axis, float eps)
      : BaseFunction(ctx, base_axis, eps), base_axis_(base_axis), eps_(eps) {}
  virtual ~VATNoise() {}
  virtual shared_ptr<Function> copy() const {
    return create_VATNoise(ctx_, base_axis_, eps_);
  }
  virtual vector<dtypes> in_types() { return vector<dtypes>{get_dtype<T>()}; }
  virtual vector<dtypes> out_types() { return vector<dtypes>{get_dtype<T>()}; }
  virtual int min_inputs() { return 2; }
  virtual int min_outputs() { return 1; }
  virtual string name() { return "VATNoise"; }
  virtual vector<string> allowed_array_classes() {
    return vector<string>{"CpuArray"};
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
    return false;
  }
};
}
#endif
