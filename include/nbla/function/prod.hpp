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

/** Prod
 */
#ifndef __NBLA_FUNCTION_PROD_HPP__
#define __NBLA_FUNCTION_PROD_HPP__

#include <nbla/cpu.hpp>
#include <nbla/function.hpp>
#include <nbla/function/sum.hpp>
#include <nbla/function_registry.hpp>

namespace nbla {

NBLA_REGISTER_FUNCTION_HEADER(Prod, const vector<int> &, bool);

/** Reduction along axes with product operation.

@tparam T Data type for computation.
@param axes A list of axes to be reduced.
@param keep_dims Flag whether the reduced axes are kept.

@note Backward computation is not accurate in a zero value input.
\ingroup FunctionImplGrp
 */
template <typename T> class Prod : public Sum<T> {
public:
  Prod(const Context &ctx, const vector<int> &axes, bool keep_dims)
      : Sum<T>(ctx, axes, keep_dims) {}
  virtual ~Prod() {}
  virtual shared_ptr<Function> copy() const {
    return create_Prod(this->ctx_, this->axes_, this->keep_dims_);
  }
  virtual string name() { return "Prod"; }
  virtual bool grad_depends_output_data(int i, int o) const { return true; }

protected:
  NBLA_API virtual void backward_impl(const Variables &inputs,
                                      const Variables &outputs,
                                      const vector<bool> &propagate_down,
                                      const vector<bool> &accum);
  NBLA_API virtual void forward_impl_reduce(const T *x, T *y, int outer_size,
                                            int reduction_size);
  NBLA_API virtual void
  backward_impl_reduce_prod(const T *dy, const T *x, const T *y, T *dx,
                            int outer_size, int reduction_size, bool accum);
  virtual bool grad_depends_input_data_impl(int i, int j) const { return true; }
};
}
#endif
