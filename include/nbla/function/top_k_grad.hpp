// Copyright 2018,2019,2020,2021 Sony Corporation.
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

#ifndef NBLA_FUNCTION_TOP_K_GRAD_HPP
#define NBLA_FUNCTION_TOP_K_GRAD_HPP

#include <nbla/cpu.hpp>
#include <nbla/function.hpp>
#include <nbla/function_registry.hpp>

namespace nbla {

NBLA_REGISTER_FUNCTION_HEADER(TopKGrad, int, bool, int);

/** Select the `k` largest gradients for each sample in `x` to
back-propagate unmodified and set all other gradients to 0. If
`abs` is True, the `k` largest gradients are selected by magnitude.
Dimensions before `base_axis` are treated as number of sample
dimensions and `k` gradients get selected from all gradients
of a sample (dimensions from `base_axis`) regardless of shape.

Inputs:

- N-D array

Outputs:

- N-D array with same shape and data as `x`.

@tparam T Data type for computation.

@param k Number of largest gradients to back-propagate.

@param abs Determine largest gradients by magnitude.

@param base_axis First dimension of the sample shape.

\ingroup FunctionImplGrp
 */
template <typename T> class TopKGrad : public BaseFunction<int, bool, int> {
protected:
  int k_;
  bool abs_;
  int base_axis_;
  Variable top_k_idx_;

public:
  TopKGrad(const Context &ctx, int k, bool abs, int base_axis)
      : BaseFunction(ctx, k, abs, base_axis), k_(k), abs_(abs),
        base_axis_(base_axis) {}
  virtual ~TopKGrad() {}
  virtual shared_ptr<Function> copy() const {
    return create_TopKGrad(ctx_, k_, abs_, base_axis_);
  }
  virtual int min_inputs() { return 1; }
  virtual int min_outputs() { return 1; }
  virtual vector<dtypes> in_types() { return vector<dtypes>{get_dtype<T>()}; }
  virtual vector<dtypes> out_types() { return vector<dtypes>{get_dtype<T>()}; }
  virtual vector<string> allowed_array_classes() {
    return SingletonManager::get<Cpu>()->array_classes();
  }
  virtual string name() { return "TopKGrad"; }
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
