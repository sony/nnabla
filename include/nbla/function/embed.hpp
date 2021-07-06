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

/** Embed
 */
#ifndef __NBLA_FUNCTION_EMBED_HPP__
#define __NBLA_FUNCTION_EMBED_HPP__

#include <nbla/cpu.hpp>
#include <nbla/function.hpp>
#include <nbla/function_registry.hpp>

namespace nbla {

NBLA_REGISTER_FUNCTION_HEADER(Embed);

/** Embed slices a matrix/tensor with indexing array/tensor.

Inputs:
- Indexes with shape @f$(I_0, ..., I_N)@f$
- Weights with shape @f$(W_0, ..., W_M)@f$

Outputs:
- Output with shape @f$(I_0, ..., I_N, W_1, ..., W_M)@f$

@tparam T Index type (integer)
@tparam T1 Value type (usually float)
@ingroup FunctionImplGrp
 */
template <typename T, typename T1> class Embed : public BaseFunction<> {
protected:
public:
  Embed(const Context &ctx) : BaseFunction<>(ctx) {}
  virtual ~Embed() {}
  virtual shared_ptr<Function> copy() const { return create_Embed(ctx_); }
  virtual vector<dtypes> in_types() {
    return vector<dtypes>{get_dtype<T>(), get_dtype<T1>()};
  }
  virtual vector<dtypes> out_types() { return vector<dtypes>{get_dtype<T1>()}; }
  virtual int min_inputs() { return 2; }
  virtual int min_outputs() { return 1; }
  virtual string name() { return "Embed"; }
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
    if (i == 1 && j == 0) {
      return true;
    }
    return false;
  }
};
}
#endif
