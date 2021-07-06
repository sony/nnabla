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

/** RandomShift
 */
#ifndef __NBLA_FUNCTION_RANDOM_SHIFT_HPP__
#define __NBLA_FUNCTION_RANDOM_SHIFT_HPP__

#include <nbla/cpu.hpp>
#include <nbla/function.hpp>
#include <nbla/function_registry.hpp>

#include <random>

namespace nbla {

NBLA_REGISTER_FUNCTION_HEADER(RandomShift, const vector<int> &, const string &,
                              float, int, int);

/** RandomShift randomly shifts the array elements within the specified range.

Inputs:
- N-D array.

Outputs:
- N-D array.

@tparam T Data type for computation.
@param shift Max absolute amount to shift elements. For example, to shift image
data horizontally by +-2 pixels and vertically by +-3 pixels, specify (3,2).
@param border_mode Specify how to process the ends of arrays whose values will
be undetermined as a result of shifting. nearest: The data at the ends of the
original array is copied and used. reflect: Original data reflected at the ends
of the original array is used.
\ingroup FunctionImplGrp
*/

template <typename T>
class RandomShift : public BaseFunction<const vector<int> &, const string &,
                                        float, int, int> {
protected:
  vector<int> shifts_;
  string border_mode_;
  int base_axis_;

  int size_;

  const T constant_value_;
  const int CVAL_INDEX = -1;

  // This variable is an array to realize different shift amount for each data.
  vector<vector<vector<int>>> addr_table_;

  int seed_;
  bool save_rng_ = false;
  std::mt19937 rgen_, rgen_for_recompute_;

public:
  RandomShift(const Context &ctx, const vector<int> &shifts,
              const string &border_mode, float constant_value, int base_axis,
              int seed)
      : BaseFunction(ctx, shifts, border_mode, constant_value, base_axis, seed),
        shifts_(shifts), border_mode_(border_mode), base_axis_(base_axis),
        constant_value_(constant_value), seed_(seed) {}
  virtual ~RandomShift() {}
  virtual shared_ptr<Function> copy() const {
    return create_RandomShift(ctx_, shifts_, border_mode_, constant_value_,
                              base_axis_, seed_);
  }
  virtual vector<dtypes> in_types() { return vector<dtypes>{get_dtype<T>()}; }
  virtual vector<dtypes> out_types() { return vector<dtypes>{get_dtype<T>()}; }
  virtual int min_inputs() { return 1; }
  virtual int min_outputs() { return 1; }
  virtual string name() { return "RandomShift"; }
  virtual vector<string> allowed_array_classes() {
    return SingletonManager::get<Cpu>()->array_classes();
  }
  virtual bool need_setup_recompute(int o) const { return true; }
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
  NBLA_API virtual void setup_recompute_impl(const Variables &inputs,
                                             const Variables &outputs);
  NBLA_API virtual void recompute_impl(const Variables &inputs,
                                       const Variables &outputs);
  virtual bool grad_depends_input_data_impl(int i, int j) const {
    return false;
  }

  vector<vector<int>> prepare_addr_table(const Variables &inputs,
                                         const vector<int> &shifts);

private:
  void shift_recursive(const Variable *inp, const T *x, T *y, int x_offset,
                       int y_offset, int dim, int &shift_index);
  void shift_backward_recursive(const Variable *inp, const T *dy, T *dx,
                                int x_offset, int y_offset, int dim,
                                int &shift_index);
  void random_shift(const Variables &inputs, const Variables &outputs,
                    std::mt19937 &rgen);
};
}
#endif
