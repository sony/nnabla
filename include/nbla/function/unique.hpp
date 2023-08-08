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

#ifndef NBLA_FUNCTION_UNIQUE_HPP
#define NBLA_FUNCTION_UNIQUE_HPP

#include <nbla/cpu.hpp>
#include <nbla/function.hpp>
#include <nbla/function_registry.hpp>

namespace nbla {

NBLA_REGISTER_FUNCTION_HEADER(Unique, bool, int, bool, bool, bool, bool);

/**
Find the unique elements of input array.

Inputs:
- x: A N-D array.

Outputs:
- y: A N-D array.
- indices: A 1-D array. It's indices of `y` elements first occurance in `x`. If
`flatten` is True, it contains indices to flattend input array `x`. If `flatten`
is False and `axis` is specified, it contains indices to input array `x` on
`axis`.
- inverse_indices: A 1-D array. It's indices of `x` elements corresponding to
`y`. If `flatten` is True, it contains indices to output array `y`. If `flatten`
is False and `axis` is specified, it contains indices to output array `y` on
`axis`.
- counts: A 1-D array. It's the count of each element of 'y' in input array `x`.

@param flatten If True, unique values of the flatten input array are returned.
@param axis If flatten is True and axis is specified, unique slices along axis
are returned.
@param sorted If True, unique values/slices sorted in ascending order are
returned.
@param with_index If True, `indices` is returned.
@param with_inverse If True, `inverse_indices` is returned.
@param with_counts `counts` is returned.
\ingroup FunctionImplGrp
 */
template <typename T>
class Unique : public BaseFunction<bool, int, bool, bool, bool, bool> {
protected:
  bool flatten_;
  int axis_;
  bool sorted_;
  bool with_index_;
  bool with_inverse_;
  bool with_counts_;

public:
  Unique(const Context &ctx, bool flatten, int axis, bool sorted,
         bool with_index, bool with_inverse, bool with_counts)
      : BaseFunction(ctx, flatten, axis, sorted, with_index, with_inverse,
                     with_counts),
        flatten_(flatten), axis_(axis), sorted_(sorted),
        with_index_(with_index), with_inverse_(with_inverse),
        with_counts_(with_counts) {}
  virtual ~Unique() {}
  virtual shared_ptr<Function> copy() const {
    return create_Unique(ctx_, flatten_, axis_, sorted_, with_index_,
                         with_inverse_, with_counts_);
  }
  virtual int min_inputs() { return 1; }
  virtual int min_outputs() {
    return 1 + int(with_index_) + int(with_inverse_) + int(with_counts_);
  }
  virtual vector<dtypes> in_types() { return vector<dtypes>{get_dtype<T>()}; }
  virtual vector<dtypes> out_types() {
    return vector<dtypes>{get_dtype<T>(), get_dtype<int64_t>(),
                          get_dtype<int64_t>(), get_dtype<int64_t>()};
  }
  virtual vector<string> allowed_array_classes() {
    return SingletonManager::get<Cpu>()->array_classes();
  }
  virtual string name() { return "Unique"; }

protected:
  NBLA_API virtual void setup_impl(const Variables &inputs,
                                   const Variables &outputs);
  NBLA_API virtual void forward_impl(const Variables &inputs,
                                     const Variables &outputs);
  virtual void unique(const Variables &inputs, const Variables &outputs,
                      VariablePtr reshaped_x_ptr, bool is_recompute = false);
  NBLA_API virtual void recompute_impl(const Variables &inputs,
                                       const Variables &outputs);
  NBLA_API virtual void backward_impl(const Variables &inputs,
                                      const Variables &outputs,
                                      const vector<bool> &propagate_down,
                                      const vector<bool> &accum);
};
} // namespace nbla
#endif
