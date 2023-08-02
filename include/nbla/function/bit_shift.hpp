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

#ifndef NBLA_FUNCTION_BIT_SHIFT_HPP
#define NBLA_FUNCTION_BIT_SHIFT_HPP

#include <nbla/cpu.hpp>
#include <nbla/function.hpp>
#include <nbla/function/utils/base_transform_binary.hpp>
#include <nbla/function_registry.hpp>

namespace nbla {

NBLA_REGISTER_FUNCTION_HEADER(BitShift, const string &);

/**
Element-wise bit shift function.

Inputs:
- x: A N-D array.
- shift: A N-D array.

Outputs:
- y: A N-D array.

@param direction Direction of bit shift.

\ingroup FunctionImplGrp
 */
template <typename T> class BitShift : public BaseTransformBinary<> {
protected:
  const string direction_;
  bool shift_left_;

public:
  BitShift(const Context &ctx, const string &direction)
      : BaseTransformBinary(ctx, false), direction_(direction),
        shift_left_(true) {}
  virtual ~BitShift() {}
  virtual shared_ptr<Function> copy() const {
    return create_BitShift(ctx_, direction_);
  }
  virtual int min_inputs() { return 2; }
  virtual int min_outputs() { return 1; }
  virtual vector<dtypes> in_types() {
    NBLA_ERROR(error_code::value, "BitShift has multiple input dtypes.");
  }
  virtual vector<dtypes> out_types() {
    NBLA_ERROR(error_code::value, "BitShift has multiple output dtypes.");
  }
  virtual vector<string> allowed_array_classes() {
    return SingletonManager::get<Cpu>()->array_classes();
  }
  virtual string name() { return "BitShift"; }

protected:
  NBLA_API virtual void setup_impl(const Variables &inputs,
                                   const Variables &outputs);
  NBLA_API virtual void forward_impl(const Variables &inputs,
                                     const Variables &outputs);
  NBLA_API virtual void backward_impl(const Variables &inputs,
                                      const Variables &outputs,
                                      const vector<bool> &propagate_down,
                                      const vector<bool> &accum);
};
} // namespace nbla
#endif
