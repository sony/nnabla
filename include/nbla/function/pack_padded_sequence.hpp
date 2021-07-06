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

#ifndef NBLA_FUNCTION_PACK_PADDED_SEQUENCE_HPP
#define NBLA_FUNCTION_PACK_PADDED_SEQUENCE_HPP

#include <nbla/cpu.hpp>
#include <nbla/function.hpp>
#include <nbla/function_registry.hpp>

namespace nbla {

NBLA_REGISTER_FUNCTION_HEADER(PackPaddedSequence, bool);

/**
Pack a padded variable-length sequences.

This method packs a padded variable-length sequences.

Inputs:
- Padded sequence of (\f$T \times B \times *\f$) or (\f$B \times T \times *\f$)
shape.
- Sequence length for each batch and always resides in CPU.

Outputs:
- Packed sequence of (\f$N\f$, \f$*\f$) shape.
- Batch size for each time and always resides in CPU.

@tparam T Data type for computation.
@param batch_first Packed sequence is of (\f$T\f$, \f$B\f$, \f$*\f$) shape if
False, otherwise (\f$B\f$, \f$T\f$, \f$*\f$).

\ingroup FunctionImplGrp
 */
template <typename T> class PackPaddedSequence : public BaseFunction<bool> {
protected:
  bool batch_first_;

public:
  PackPaddedSequence(const Context &ctx, bool batch_first)
      : BaseFunction(ctx, batch_first), batch_first_(batch_first) {}
  virtual ~PackPaddedSequence() {}
  virtual shared_ptr<Function> copy() const {
    return create_PackPaddedSequence(ctx_, batch_first_);
  }
  virtual int min_inputs() { return 2; }
  virtual int min_outputs() { return 2; }
  virtual vector<dtypes> in_types() { return vector<dtypes>{get_dtype<T>()}; }
  virtual vector<dtypes> out_types() {
    return vector<dtypes>{get_dtype<T>(), get_dtype<T>()};
  }
  virtual vector<string> allowed_array_classes() {
    return SingletonManager::get<Cpu>()->array_classes();
  }
  virtual string name() { return "PackPaddedSequence"; }
  virtual bool grad_depends_output_data(int i, int o) const { return o > 0; }

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
