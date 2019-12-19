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

/** SoftmaxCrossEntropy
 */
#ifndef __NBLA_FUNCTION_SOFTMAX_XENT_HPP__
#define __NBLA_FUNCTION_SOFTMAX_XENT_HPP__

#include <nbla/cpu.hpp>
#include <nbla/function.hpp>
#include <nbla/function_registry.hpp>

#include <memory>
#include <string>

namespace nbla {

using std::string;
using std::make_shared;

NBLA_REGISTER_FUNCTION_HEADER(SoftmaxCrossEntropy, int);

/** SoftmaxCrossEntropy calculate the element-wise cross entropy between the
variables and the variables of a label given by a category index with Softmax
normalization.
@f[
y_{j} = -\ln \left(\frac{\exp(x_{t_j,j})}{\sum_{i'} exp(x_{i'j})}\right)
@f]
along dimension specified by axis.

SoftmaxCrossEntropy is equivalent to Softmax+CategoricalCrossEntropy, but
computing them at once has the effect of reducing computational error.

Inputs (i is axis normalization taken):
- Scores N-D array. (\f$D_1 \times ... \times D_i \times ... \times D_N\f$)
- Labels N-D array. (\f$D_1 \times ... \times 1 \times ... \times D_N\f$)

Outputs:
- Element-wise losses N-D array. (\f$D_1 \times ... \times 1 \times ... \times
D_N\f$)

@tparam T Data type for computation and score variable.
@tparam Tl Data type of label variable.
@param axis Axis normalization is taken.
\ingroup FunctionImplGrp
 */
template <typename T, typename Tl = int>
class SoftmaxCrossEntropy : public BaseFunction<int> {
protected:
  int axis_;
  int size0_, size1_, size2_;
  shared_ptr<Function> log_softmax_;
  Variable log_softmax_output_;

public:
  SoftmaxCrossEntropy(const Context &ctx, int axis)
      : BaseFunction(ctx, axis), axis_(axis) {}
  virtual ~SoftmaxCrossEntropy() {}
  virtual shared_ptr<Function> copy() const {
    return create_SoftmaxCrossEntropy(ctx_, axis_);
  }
  virtual vector<dtypes> in_types() {
    return vector<dtypes>{get_dtype<T>(), get_dtype<Tl>()};
  }
  virtual vector<dtypes> out_types() { return vector<dtypes>{get_dtype<T>()}; }
  virtual int min_inputs() { return 2; }
  virtual int min_outputs() { return 1; }
  virtual string name() { return "SoftmaxCrossEntropy"; }
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
