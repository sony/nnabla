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

#ifndef NBLA_FUNCTION_SORT_HPP
#define NBLA_FUNCTION_SORT_HPP

#include <nbla/cpu.hpp>
#include <nbla/function.hpp>
#include <nbla/function_registry.hpp>

namespace nbla {

NBLA_REGISTER_FUNCTION_HEADER(Sort, int, bool, bool, bool);

/**Sorts the elements of `x` along a given `axis` in ascending order
   by value. A negative `axis` counts from the last dimension of `x`,
   so the default of -1 sorts along the last dimension. If `reverse`
   is True, then the elements are soreted in descending order.

   If `with_index` is True, result is a tuple ``(sorted, indices)`` or
   only ``indices`` if `only_index` is True. Setting `only_index` to
   True implies that `with_index` is also True.

Inputs:

- N-D array

Outputs:

- one or two N-D arrays

@tparam T Data type for computation.

@param axis Axis along which to sort.

@param reverse Sort in descending order.

@param with_index Return sorted values and index.

@param only_index Return only the sort index.

\ingroup FunctionImplGrp
 */
template <typename T> class Sort : public BaseFunction<int, bool, bool, bool> {
protected:
  int axis;
  bool reverse;
  bool with_index;
  bool only_index;
  size_t inner_size;
  size_t outer_size;
  size_t total_size;
  Variable sort_index;
  Variable temp_index;

public:
  Sort(const Context &ctx, int axis, bool reverse, bool with_index,
       bool only_index)
      : BaseFunction(ctx, axis, reverse, with_index, only_index), axis(axis),
        reverse(reverse), with_index(with_index), only_index(only_index) {}
  virtual ~Sort() {}
  virtual shared_ptr<Function> copy() const {
    return create_Sort(ctx_, axis, reverse, with_index, only_index);
  }
  virtual int min_inputs() { return 1; }
  virtual int min_outputs() { return 1; }
  virtual vector<dtypes> in_types() { return vector<dtypes>{get_dtype<T>()}; }
  virtual vector<dtypes> out_types() { return vector<dtypes>{get_dtype<T>()}; }
  virtual vector<string> allowed_array_classes() {
    return SingletonManager::get<Cpu>()->array_classes();
  }
  virtual string name() { return "Sort"; }
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
