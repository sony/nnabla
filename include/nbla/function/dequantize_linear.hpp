// Copyright 2020,2021 Sony Corporation.
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

#ifndef NBLA_FUNCTION_DEQUANTIZE_LINEAR_HPP
#define NBLA_FUNCTION_DEQUANTIZE_LINEAR_HPP

#include <nbla/cpu.hpp>
#include <nbla/function.hpp>
#include <nbla/function_registry.hpp>

#include <nbla/function/add2.hpp>
#include <nbla/function/mul2.hpp>
#include <nbla/function/sub2.hpp>
#include <nbla/function/sum.hpp>

#include <nbla/imperative.hpp>

namespace nbla {

NBLA_REGISTER_FUNCTION_HEADER(DequantizeLinear);

/**
    @todo Write doc.

Inputs:

Outputs:

\ingroup FunctionImplGrp
 */
template <typename T> class DequantizeLinear : public BaseFunction<> {
protected:
  shared_ptr<Function> mul2_;
  shared_ptr<Function> add2_;
  shared_ptr<Function> sub2_;
  shared_ptr<Function> sum_;

public:
  DequantizeLinear(const Context &ctx) : BaseFunction(ctx) {}
  virtual ~DequantizeLinear() {}
  virtual shared_ptr<Function> copy() const {
    return create_DequantizeLinear(ctx_);
  }
  virtual int min_inputs() { return 3; }
  virtual int min_outputs() { return 1; }
  virtual vector<dtypes> in_types() {
    return vector<dtypes>{get_dtype<T>(), get_dtype<T>(), get_dtype<T>()};
  }
  virtual vector<dtypes> out_types() { return vector<dtypes>{get_dtype<T>()}; }
  virtual vector<string> allowed_array_classes() {
    return SingletonManager::get<Cpu>()->array_classes();
  }
  virtual string name() { return "DequantizeLinear"; }
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
    if (i == 0 && j == 1)
      return true;
    return false;
  }
};
}
#endif
