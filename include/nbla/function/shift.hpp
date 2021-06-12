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

/** Shift
 */
#ifndef __NBLA_FUNCTION_SHIFT_HPP__
#define __NBLA_FUNCTION_SHIFT_HPP__

#include <nbla/cpu.hpp>
#include <nbla/function.hpp>
#include <nbla/function_registry.hpp>

namespace nbla {

NBLA_REGISTER_FUNCTION_HEADER(Shift, const vector<int> &, const string &);

/** Shift shifts the array elements by the specified amount.

Inputs:
- N-D array.

Outputs:
- N-D array.

@tparam T Data type for computation.
@param shift The amount to shift elements. For example, to shift image data to
the right by 2 pixels and up 3 pixels, specify (-3,2).
@param border_mode Specify how to process the ends of arrays whose values will
be undetermined as a result of shifting. nearest: The data at the ends of the
original array is copied and used. reflect: Original data reflected at the ends
of the original array is used.
\ingroup FunctionImplGrp
*/
template <typename T>
class Shift : public BaseFunction<const vector<int> &, const string &> {
protected:
  vector<int> shifts_;
  string border_mode_;

public:
  Shift(const Context &ctx, const vector<int> &shifts,
        const string &border_mode)
      : BaseFunction(ctx, shifts, border_mode), shifts_(shifts),
        border_mode_(border_mode) {}
  virtual ~Shift() {}
  virtual shared_ptr<Function> copy() const {
    return create_Shift(ctx_, shifts_, border_mode_);
  }
  virtual vector<dtypes> in_types() { return vector<dtypes>{get_dtype<T>()}; }
  virtual vector<dtypes> out_types() { return vector<dtypes>{get_dtype<T>()}; }
  virtual int min_inputs() { return 1; }
  virtual int min_outputs() { return 1; }
  virtual string name() { return "Shift"; }
  virtual vector<string> allowed_array_classes() {
    return SingletonManager::get<Cpu>()->array_classes();
  }

  std::vector<int> &shifts() { return shifts_; }
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

private:
  template <bool is_backward>
  void shift_recursive(Variable *inp, const T *src, T *dst, int x_offset,
                       int y_offset, int dim);
};
}
#endif
