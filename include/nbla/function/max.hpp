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

/** Max
 */
#ifndef __NBLA_FUNCTION_MAX_HPP__
#define __NBLA_FUNCTION_MAX_HPP__

#include <nbla/cpu.hpp>
#include <nbla/function.hpp>
#include <nbla/function/sum.hpp>
#include <nbla/function_registry.hpp>

namespace nbla {

NBLA_REGISTER_FUNCTION_HEADER(Max, const vector<int> &, bool, bool, bool);

/** Reduction along axes with max operation.

@tparam T Data type for computation.
@param axes A list of axes to be reduced.
@param keep_dims Flag whether the reduced axes are kept.

\ingroup FunctionImplGrp
 */
template <typename T> class Max : public Sum<T> {
protected:
  bool with_index_;
  bool only_index_;
  shared_ptr<Variable> index_buff_;

public:
  Max(const Context &ctx, const vector<int> &axes, bool keep_dims,
      bool with_index, bool only_index)
      : Sum<T>(ctx, axes, keep_dims), with_index_(with_index),
        only_index_(only_index) {}
  virtual ~Max() {}
  virtual shared_ptr<Function> copy() const {
    return create_Max(this->ctx_, this->axes_, this->keep_dims_,
                      this->with_index_, this->only_index_);
  }
  virtual string name() { return "Max"; }
  virtual bool grad_depends_output_data(int i, int o) const { return false; }

protected:
  NBLA_API virtual void setup_impl(const Variables &inputs,
                                   const Variables &outputs);
  NBLA_API virtual void forward_impl(const Variables &inputs,
                                     const Variables &outputs);
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
