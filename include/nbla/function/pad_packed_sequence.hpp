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

#ifndef NBLA_FUNCTION_PAD_PACKED_SEQUENCE_HPP
#define NBLA_FUNCTION_PAD_PACKED_SEQUENCE_HPP

#include <nbla/cpu.hpp>
#include <nbla/function.hpp>
#include <nbla/function_registry.hpp>

namespace nbla {

NBLA_REGISTER_FUNCTION_HEADER(PadPackedSequence, bool, float, int);

/**
Pad packed sequence.

This method unpacks the packed sequqnce and pad it, the inverse operation of
:func:`pack_padded_sequence`.

Inputs:
- Packed sequence of (\f$N\f$, \f$*\f$) shape.
- Batch size for each time and always resides in CPU.

Outputs:
- Padded sequence of (\f$T \times B \times *\f$) or (\f$B \times T \times *\f$)
shape.
- Sequence length for each batch and always resides in CPU.

@tparam T Data type for computation.
@param batch_first Padded sequence is of (\f$T\f$, \f$B\f$, \f$*\f$) shape if
False, otherwise (\f$B\f$, \f$T\f$, \f$*\f$).
@param padding_value Padding value.
@param total_length If not -1, the outputs are padded up to the `total_length`.
If the `total_length` is less than the max length in the `sequences`, the error
is thrown.

\ingroup FunctionImplGrp
 */
template <typename T>
class PadPackedSequence : public BaseFunction<bool, float, int> {
protected:
  bool batch_first_;
  float padding_value_;
  int total_length_;

public:
  PadPackedSequence(const Context &ctx, bool batch_first, float padding_value,
                    int total_length)
      : BaseFunction(ctx, batch_first, padding_value, total_length),
        batch_first_(batch_first), padding_value_(padding_value),
        total_length_(total_length) {}
  virtual ~PadPackedSequence() {}
  virtual shared_ptr<Function> copy() const {
    return create_PadPackedSequence(ctx_, batch_first_, padding_value_,
                                    total_length_);
  }
  virtual int min_inputs() { return 2; }
  virtual int min_outputs() { return 2; }
  virtual vector<dtypes> in_types() {
    return vector<dtypes>{get_dtype<T>(), get_dtype<T>()};
  }
  virtual vector<dtypes> out_types() {
    return vector<dtypes>{get_dtype<T>(), get_dtype<T>()};
  }
  virtual vector<string> allowed_array_classes() {
    return SingletonManager::get<Cpu>()->array_classes();
  }
  virtual string name() { return "PadPackedSequence"; }
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
    if (i == 0 && j == 1) {
      return true;
    }
    return false;
  }
};
}
#endif
