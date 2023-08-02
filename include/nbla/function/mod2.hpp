// Copyright 2023 Sony Group Corporation.
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

#ifndef NBLA_FUNCTION_MOD2_HPP
#define NBLA_FUNCTION_MOD2_HPP

#include <nbla/cpu.hpp>
#include <nbla/function.hpp>
#include <nbla/function/utils/base_transform_binary.hpp>
#include <nbla/function_registry.hpp>

namespace nbla {

NBLA_REGISTER_FUNCTION_HEADER(Mod2, bool);

/**
Element-wise remainder function.
If fmod is True, this operator behaves like numpy.fmod, otherwise it behaves
like numpy.mod.

.. math::

    y_i = \text{mod} (x_i)

Inputs:
- x0: A N-D array.
- x1: A N-D array.

Outputs:
- y: A N-D array.

@param fmod If True, this operator behaves like numpy.fmod, otherwise it behaves
like numpy.mod.

\ingroup FunctionImplGrp
 */
template <typename T> class Mod2 : public BaseTransformBinary<> {
protected:
  bool fmod_;

public:
  Mod2(const Context &ctx, bool fmod)
      : BaseTransformBinary(ctx, false), fmod_(fmod) {}
  virtual ~Mod2() {}
  virtual shared_ptr<Function> copy() const { return create_Mod2(ctx_, fmod_); }
  virtual int min_inputs() { return 2; }
  virtual int min_outputs() { return 1; }
  virtual vector<dtypes> in_types() {
    NBLA_ERROR(error_code::value, "Mod2 has multiple input dtypes.");
  }
  virtual vector<dtypes> out_types() {
    NBLA_ERROR(error_code::value, "Mod2 has multiple output dtypes.");
  }
  virtual vector<string> allowed_array_classes() {
    return SingletonManager::get<Cpu>()->array_classes();
  }
  virtual string name() { return "Mod2"; }

protected:
  NBLA_API virtual void forward_impl(const Variables &inputs,
                                     const Variables &outputs);
  NBLA_API virtual void backward_impl(const Variables &inputs,
                                      const Variables &outputs,
                                      const vector<bool> &propagate_down,
                                      const vector<bool> &accum);
};
} // namespace nbla
#endif
