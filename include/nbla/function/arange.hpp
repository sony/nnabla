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

#ifndef NBLA_FUNCTION_ARANGE_HPP
#define NBLA_FUNCTION_ARANGE_HPP

#include <nbla/cpu.hpp>
#include <nbla/function.hpp>
#include <nbla/function_registry.hpp>

namespace nbla {

NBLA_REGISTER_FUNCTION_HEADER(Arange, float, float, float);

/** Generate a range of values within the half-open interval ``[start,
stop)`` (the interval including start but excluding stop) with `step`
increments.

Inputs:
- none

Outputs:
- 1-D array.

@tparam T Data type for computation.

@param start Start value.

@param stop End value.

@param step Step value.

\ingroup FunctionImplGrp
 */
template <typename T> class Arange : public BaseFunction<float, float, float> {
protected:
  float start_;
  float stop_;
  float step_;

public:
  Arange(const Context &ctx, float start, float stop, float step)
      : BaseFunction(ctx, start, stop, step), start_(start), stop_(stop),
        step_(step) {}
  virtual ~Arange() {}
  virtual shared_ptr<Function> copy() const {
    return create_Arange(ctx_, start_, stop_, step_);
  }
  virtual int min_inputs() { return 0; }
  virtual int min_outputs() { return 1; }
  virtual vector<dtypes> in_types() { return vector<dtypes>{}; }
  virtual vector<dtypes> out_types() { return vector<dtypes>{get_dtype<T>()}; }
  virtual vector<string> allowed_array_classes() {
    return SingletonManager::get<Cpu>()->array_classes();
  }
  virtual string name() { return "Arange"; }
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
