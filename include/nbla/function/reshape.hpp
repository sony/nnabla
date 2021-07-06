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

/** Reshape
 */
#ifndef __NBLA_FUNCTION_RESHAPE_HPP__
#define __NBLA_FUNCTION_RESHAPE_HPP__

#include <nbla/cpu.hpp>
#include <nbla/function.hpp>
#include <nbla/function_registry.hpp>

#include <algorithm>

namespace nbla {

NBLA_REGISTER_FUNCTION_HEADER(Reshape, const vector<int> &, bool);

/** Reshaping copy of an input variable.

Inputs:
- N-D array.

Outputs:
- N-D array.

@tparam T Data type for computation.
@param shape Target shape.
@param inplace The output array is will be shared with the input array if true.
\ingroup FunctionImplGrp
 */

template <typename T>
class Reshape : public BaseFunction<const vector<int> &, bool> {
protected:
  Shape_t shape_;
  bool inplace_;

public:
  Reshape(const Context &ctx, const vector<int> &shape, bool inplace)
      : BaseFunction(ctx, shape, inplace), shape_(shape.size()),
        inplace_(inplace) {
    std::copy(shape.begin(), shape.end(), shape_.begin());
  }
  virtual ~Reshape() {}
  virtual shared_ptr<Function> copy() const {
    vector<int> shape(shape_.size());
    std::copy(shape_.begin(), shape_.end(), shape.begin());
    return create_Reshape(ctx_, shape, inplace_);
  }
  virtual vector<dtypes> in_types() { return vector<dtypes>{get_dtype<T>()}; }
  virtual vector<dtypes> out_types() { return vector<dtypes>{get_dtype<T>()}; }
  virtual int min_inputs() { return 1; }
  virtual int min_outputs() { return 1; }
  virtual string name() { return "Reshape"; }
  virtual vector<string> allowed_array_classes() {
    return SingletonManager::get<Cpu>()->array_classes();
  }
  virtual bool grad_depends_output_data(int i, int o) const { return inplace_; }
  virtual int inplace_data(int i) const {
    return inplace_ ? Function::INPLACE_NOT_MODIFY : Function::NOT_INPLACE;
  }
  virtual int inplace_data_with(int i) const { return 0; }

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
