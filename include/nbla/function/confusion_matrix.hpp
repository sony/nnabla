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

/** ConfusionMatrix
 */
#ifndef __NBLA_FUNCTION_CONFUSIONMATRIX_HPP__
#define __NBLA_FUNCTION_CONFUSIONMATRIX_HPP__

#include <nbla/cpu.hpp>
#include <nbla/function.hpp>
#include <nbla/function_registry.hpp>

namespace nbla {

NBLA_REGISTER_FUNCTION_HEADER(ConfusionMatrix, int);

/** Confusion matrix.

The return value is already summed over samples.

Inputs (i is the axis on which the confusion matrix is calculated):
- Probabilities N-D array. (\f$D_1 \times ... \times D_i \times ... \times
D_N\f$)
- Labels N-D array. (\f$D_1 \times ... \times 1 \times ... \times D_N\f$)

Outputs:
- Confusion matrix 2-D array. Col index is estimated class. Row index is label
class.

@tparam T Data type for computation and score variable.
@tparam Tl Data type of label variable.
@param axis Axis on which the confusion matrix is calculated.
\ingroup FunctionImplGrp
 */
template <typename T, typename T1> class ConfusionMatrix : public Function {
protected:
  int axis_;
  Size_t size0_, size1_, size2_;

public:
  ConfusionMatrix(const Context &ctx, int axis) : Function(ctx), axis_(axis) {}
  virtual ~ConfusionMatrix() {}
  virtual shared_ptr<Function> copy() const {
    return create_ConfusionMatrix(ctx_, axis_);
  }
  virtual vector<dtypes> in_types() {
    return vector<dtypes>{get_dtype<T>(), get_dtype<T1>()};
  }
  virtual vector<dtypes> out_types() { return vector<dtypes>{get_dtype<T>()}; }
  virtual int min_inputs() { return 2; }
  virtual int min_outputs() { return 1; }
  virtual string name() { return "ConfusionMatrix"; }
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
    return false;
  }
};
}
#endif
