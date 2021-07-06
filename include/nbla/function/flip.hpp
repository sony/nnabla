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

/** Flip
 */
#ifndef __NBLA_FUNCTION_FLIP_HPP__
#define __NBLA_FUNCTION_FLIP_HPP__

#include <nbla/cpu.hpp>
#include <nbla/function.hpp>
#include <nbla/function_registry.hpp>

namespace nbla {

NBLA_REGISTER_FUNCTION_HEADER(Flip, const vector<int> &);

/** Flip reverses the order of elements of the specified dimension of an array.

Inputs:
- N-D array.

Outputs:
- N-D array.

@tparam T Data type for computation.
@param axes The index of the dimension to reverse the order of the elements.
Axis indexes take on values 0, 1, 2, and so on from the left. For example, to
flip a 32 (W) by 24 (H) 100 RGB image (100,3,24,32) vertically and horizontally,
specify (2,3).
\ingroup FunctionImplGrp
*/
template <typename T> class Flip : public BaseFunction<const vector<int> &> {
protected:
  vector<int> axes_;
  vector<bool> flip_;

public:
  Flip(const Context &ctx, const vector<int> &axes)
      : BaseFunction(ctx, axes), axes_(axes.size()) {
    std::copy(axes.begin(), axes.end(), axes_.begin());
  }
  virtual ~Flip() {}
  virtual shared_ptr<Function> copy() const {
    vector<int> axes(axes_.size());
    std::copy(axes_.begin(), axes_.end(), axes.begin());
    return create_Flip(ctx_, axes);
  }
  virtual vector<dtypes> in_types() { return vector<dtypes>{get_dtype<T>()}; }
  virtual vector<dtypes> out_types() { return vector<dtypes>{get_dtype<T>()}; }
  virtual int min_inputs() { return 1; }
  virtual int min_outputs() { return 1; }
  virtual string name() { return "Flip"; }
  virtual vector<string> allowed_array_classes() {
    return SingletonManager::get<Cpu>()->array_classes();
  }

  std::vector<int> &axes() { return axes_; }
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
  void flip_recursive(Variable *inp, const T *x, T *y,
                      const std::vector<bool> &flip, bool add, int x_offset,
                      int y_offset, int dim);
};
}
#endif
