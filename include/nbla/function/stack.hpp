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

/** Stack
 */
#ifndef __NBLA_FUNCTION_STACK_HPP__
#define __NBLA_FUNCTION_STACK_HPP__

#include <nbla/cpu.hpp>
#include <nbla/function.hpp>
#include <nbla/function_registry.hpp>

namespace nbla {

NBLA_REGISTER_FUNCTION_HEADER(Stack, int);

/** Stack joins two or more arrays on a new axis. The sizes of all the arrays to
be stacked must be the same. Unlike Concatenate, which joins arrays on an
existing axis, Stack joins arrays on a new axis.

Inputs:
- list of N-D arrays.

Outputs:
- N-D array.

@tparam T Data type for computation.
@param axis The axis on which to concatenate arrays. Axis indexes take on values
0, 1, 2, and so on from the left. For example, to stack four (3,28,28) inputs on
the second axis, specify 1. In this case, the output size will be (3,4,28,28).
\ingroup FunctionImplGrp
 */
template <typename T> class Stack : public BaseFunction<int> {
protected:
  int axis_;
  int num_inputs_;
  int inner_size_, outer_size_;

public:
  Stack(const Context &ctx, int axis) : BaseFunction(ctx, axis), axis_(axis) {}
  virtual ~Stack() {}
  virtual shared_ptr<Function> copy() const {
    return create_Stack(ctx_, axis_);
  }
  virtual vector<dtypes> in_types() { return vector<dtypes>{get_dtype<T>()}; }
  virtual vector<dtypes> out_types() { return vector<dtypes>{get_dtype<T>()}; }
  virtual int min_inputs() { return 1; }
  virtual int min_outputs() { return 1; }
  virtual string name() { return "Stack"; }
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
