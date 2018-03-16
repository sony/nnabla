// Copyright (c) 2017 Sony Corporation. All Rights Reserved.
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

/** TopKSelect
 */
#ifndef __NBLA_FUNCTION_TOPKSELECT_HPP__
#define __NBLA_FUNCTION_TOPKSELECT_HPP__

#include <nbla/function.hpp>
#include <nbla/function_registry.hpp>

namespace nbla {

NBLA_REGISTER_FUNCTION_HEADER(TopKSelect, int, int, int);

/** Set all except the `k` largest data values or `k_grad` largest
gradient values of `x` to zero. A negative value `k` (or `k_grad`) is
taken as `abs(k)` and `abs(value)` then used to determine top-k
elements. The default values of `k` and `k_grad` let all data and
gradient values propagate unmodified.

Dimensions from zero to `base_axis` are treated as sample dimension.
Dimensions from `base_axis` to `ndim` are treated as sample vectors.

Inputs:

- N-D array

Outputs:

- N-D array with the same shape as `x`.

@tparam T Data type for computation.

@param base_axis Dimensions up to `base_axis` is treated as sample
 dimension.

@param k Number of largest data values to keep.

@param k_grad Number of largest gradient values to keep.

\ingroup FunctionImplGrp
*/

template <typename T> class TopKSelect : public BaseFunction<int, int, int> {
protected:
  int k_data_;
  int k_grad_;
  int base_axis_;

public:
  TopKSelect(const Context &ctx, int k, int k_grad, int base_axis)
      : BaseFunction(ctx, k, k_grad, base_axis), k_data_(k), k_grad_(k_grad),
        base_axis_(base_axis) {}
  virtual ~TopKSelect() {}
  virtual shared_ptr<Function> copy() const {
    return create_TopKSelect(ctx_, k_data_, k_grad_, base_axis_);
  }
  virtual vector<dtypes> in_types() { return vector<dtypes>{get_dtype<T>()}; }
  virtual vector<dtypes> out_types() { return vector<dtypes>{get_dtype<T>()}; }
  virtual int min_inputs() { return 1; }
  virtual int min_outputs() { return 1; }
  virtual string name() { return "TopKSelect"; }
  virtual vector<string> allowed_array_classes() {
    return vector<string>{"CpuArray"};
  }

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
}
#endif
