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

#ifndef NBLA_FUNCTION_BOOL_FILL_HPP
#define NBLA_FUNCTION_BOOL_FILL_HPP

#include <nbla/cpu.hpp>
#include <nbla/function.hpp>
#include <nbla/function/broadcast.hpp>
#include <nbla/function_registry.hpp>

#include <algorithm>

namespace nbla {

NBLA_REGISTER_FUNCTION_HEADER(BoolFill, float);

/**
    @todo Write doc.

Inputs:

Outputs:

\ingroup FunctionImplGrp
 */
template <typename T> class BoolFill : public BaseFunction<float> {
protected:
  float value_;
  FunctionPtr broadcast_func_ = nullptr;

public:
  BoolFill(const Context &ctx, float value)
      : BaseFunction(ctx, value), value_(value) {}
  virtual ~BoolFill() {}
  virtual shared_ptr<Function> copy() const {
    return create_BoolFill(ctx_, value_);
  }
  virtual int min_inputs() { return 2; }
  virtual int min_outputs() { return 1; }
  virtual vector<dtypes> in_types() {
    return vector<dtypes>{get_dtype<T>(), get_dtype<T>()};
  }
  virtual vector<dtypes> out_types() { return vector<dtypes>{get_dtype<T>()}; }
  virtual vector<string> allowed_array_classes() {
    return SingletonManager::get<Cpu>()->array_classes();
  }
  virtual string name() { return "BoolFill"; }

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
    if (i == 0 && j == 1)
      return true;
    if (i == 1 && j == 0)
      return true;
    return false;
  }

  virtual bool grad_depends_output_data(int i, int o) const { return false; }
};
}
#endif
