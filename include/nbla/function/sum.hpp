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

/** Sum
 */
#ifndef __NBLA_FUNCTION_SUM_HPP__
#define __NBLA_FUNCTION_SUM_HPP__

#include <nbla/cpu.hpp>
#include <nbla/function.hpp>
#include <nbla/function_registry.hpp>

#include <algorithm>

namespace nbla {

NBLA_REGISTER_FUNCTION_HEADER(Sum, const vector<int> &, bool);

/** Reduction along axes with sum operation.

@tparam T Data type for computation.
@param axes A list of axes to be reduced.
@param keep_dims Flag whether the reduced axes are kept.

\ingroup FunctionImplGrp
 */
template <typename T>
class Sum : public BaseFunction<const vector<int> &, bool> {
protected:
  vector<int> axes_;
  bool keep_dims_;
  int reduction_size_;
  shared_ptr<Function> f_transpose_{nullptr};

public:
  Sum(const Context &ctx, const vector<int> &axes, bool keep_dims)
      : BaseFunction(ctx, axes, keep_dims), axes_(axes), keep_dims_(keep_dims) {
    if (axes.size() <= 1) {
      return;
    }
    // Sort axes vector;
    std::sort(axes_.begin(), axes_.end());
  }
  virtual ~Sum() {}
  virtual shared_ptr<Function> copy() const {
    return create_Sum(ctx_, axes_, keep_dims_);
  }
  virtual vector<dtypes> in_types() { return vector<dtypes>{get_dtype<T>()}; }
  virtual vector<dtypes> out_types() { return vector<dtypes>{get_dtype<T>()}; }
  virtual int min_inputs() { return 1; }
  virtual int min_outputs() { return 1; }
  virtual string name() { return "Sum"; }
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

  NBLA_API virtual void forward_impl_reduce(const T *x, T *y, int outer_size,
                                            int reduction_size);
  NBLA_API virtual void backward_impl_reduce(const T *dy, T *dx, int outer_size,
                                             int reduction_size, bool accum);

  virtual bool grad_depends_input_data_impl(int i, int j) const {
    return false;
  }
};
}
#endif
