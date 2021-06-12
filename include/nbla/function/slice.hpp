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

/** Slice
 */
#ifndef __NBLA_FUNCTION_SLICE_HPP__
#define __NBLA_FUNCTION_SLICE_HPP__

#include <nbla/cpu.hpp>
#include <nbla/function.hpp>
#include <nbla/function_registry.hpp>

namespace nbla {

NBLA_REGISTER_FUNCTION_HEADER(Slice, const vector<int> &, // start
                              const vector<int> &,        // stop
                              const vector<int> &);       // step

/** Slice arrays along specified axis.

Inputs:
- N-D array.

Outputs:
- M-D array.

Slice input tensor.
y = x[start[0]:stop[0]:step[0], start[1]:stop[1]:step[1], ...]

@tparam T Data type for computation.
\ingroup FunctionImplGrp
 */
template <typename T>
class Slice : public BaseFunction<const vector<int> &, const vector<int> &,
                                  const vector<int> &> {
protected:
  // These settings are array to realize different slice amount for each data.
  vector<vector<int>> start_;
  vector<vector<int>> stop_;
  vector<vector<int>> step_;

  int base_axis_;

  // SPECIAL condition
  enum { SLICE_NONE = 0x7fffffff };

public:
  Slice(const Context &ctx, const vector<int> &start, const vector<int> &stop,
        const vector<int> &step)
      : BaseFunction(ctx, start, stop, step), start_(1), stop_(1), step_(1),
        base_axis_(0) {
    start_[0] = start;
    stop_[0] = stop;
    step_[0] = step;
  }

  virtual ~Slice() {}
  virtual shared_ptr<Function> copy() const {
    return create_Slice(ctx_, start_[0], stop_[0], step_[0]);
  }
  virtual vector<dtypes> in_types() {
    return vector<dtypes>{get_dtype<T>(), get_dtype<T>()};
  }
  virtual vector<dtypes> out_types() { return vector<dtypes>{get_dtype<T>()}; }
  virtual int min_inputs() { return 1; }
  virtual int min_outputs() { return 1; }
  virtual string name() { return "Slice"; }
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
    return false;
  }
  NBLA_API bool skip_check(const Variables &outputs);

private:
  NBLA_API void slice_forward_recursive(const Variable *inp, Variable *outp,
                                        const T *x, T *y, int x_offset,
                                        int y_offset, int dim,
                                        int &slice_index);
  NBLA_API void slice_backward_recursive(Variable *outp, const Variable *inp,
                                         T *dx, const T *dy, int x_offset,
                                         int y_offset, int dim,
                                         int &slice_index);
};
}
#endif
