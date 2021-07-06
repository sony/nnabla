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

#ifndef NBLA_FUNCTION_QUANTIZE_LINEAR_HPP
#define NBLA_FUNCTION_QUANTIZE_LINEAR_HPP

#include <nbla/cpu.hpp>
#include <nbla/dtypes.hpp>
#include <nbla/function.hpp>
#include <nbla/function/add2.hpp>
#include <nbla/function/div2.hpp>
#include <nbla/function/sum.hpp>
#include <nbla/function_registry.hpp>

#include <nbla/imperative.hpp>

namespace nbla {

NBLA_REGISTER_FUNCTION_HEADER(QuantizeLinear, const string &, bool, int);

/**
    @todo Write doc.

Inputs:

Outputs:

\ingroup FunctionImplGrp
 */
template <typename T>
class QuantizeLinear : public BaseFunction<const string &, bool, int> {
protected:
  string round_mode_;
  bool narrow_range_;
  int dtype_;

  shared_ptr<Function> div2_;
  shared_ptr<Function> add2_;
  shared_ptr<Function> sum_;

  int min_range_;
  int max_range_;

public:
  QuantizeLinear(const Context &ctx, const string &round_mode,
                 bool narrow_range, int dtype)
      : BaseFunction(ctx, round_mode, narrow_range, dtype),
        round_mode_(round_mode), narrow_range_(narrow_range), dtype_(dtype) {}
  virtual ~QuantizeLinear() {}
  virtual shared_ptr<Function> copy() const {
    return create_QuantizeLinear(ctx_, round_mode_, narrow_range_, dtype_);
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
  virtual string name() { return "QuantizeLinear"; }
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
    // Return true since backward for inputs[1,2] may be implemented in the
    // future.
    return true;
  }
  NBLA_API virtual void round(Variable *inp, std::string round_mode);
  NBLA_API virtual void saturate(Variable *inp, int min_range, int max_range);
};
}
#endif
