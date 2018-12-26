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

/** PReLU
 */
#ifndef __NBLA_FUNCTION_PRELU_HPP__
#define __NBLA_FUNCTION_PRELU_HPP__

#include <nbla/cpu.hpp>
#include <nbla/function.hpp>
#include <nbla/function_registry.hpp>

namespace nbla {

NBLA_REGISTER_FUNCTION_HEADER(PReLU, int);

/** Parametrized Rectified Linear Unit function defined as
@f[
y_i = \max(0, x_i) + \alpha \min(0, x_i)
@f]
where negative slope @f$ \alpha @f$ is learned and can vary across channels (an
axis specified with base_axis).

Inputs:
- N-D array
- Scalar or a vector with size of channel axis (base_axis).

Outputs:
- N-D array.

@tparam T Data type for computation.
@param base_axis Valid if negative slope is a vector. The negative slope vector
size and input shape of base_axis must match.

@sa He et.al, Delving Deep into Rectifiers: Surpassing Human-Level Performance
on ImageNet Classification. https://arxiv.org/abs/1502.01852

\ingroup FunctionImplGrp
 */
template <typename T> class PReLU : public BaseFunction<int> {
protected:
  int base_axis_;
  int base_stride_;
  int base_shape_;

public:
  PReLU(const Context &ctx, int base_axis)
      : BaseFunction(ctx, base_axis), base_axis_(base_axis) {}
  virtual ~PReLU() {}
  virtual shared_ptr<Function> copy() const {
    return create_PReLU(ctx_, base_axis_);
  }
  virtual vector<dtypes> in_types() {
    return vector<dtypes>{get_dtype<T>(), get_dtype<T>()};
  }
  virtual vector<dtypes> out_types() { return vector<dtypes>{get_dtype<T>()}; }
  virtual int min_inputs() { return 2; }
  virtual int min_outputs() { return 1; }
  virtual string name() { return "PReLU"; }
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
