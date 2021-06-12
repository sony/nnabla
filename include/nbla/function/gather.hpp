// Copyright 2020,2021 Sony Corporation.
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

#ifndef NBLA_FUNCTION_GATHER_HPP
#define NBLA_FUNCTION_GATHER_HPP

#include <nbla/cpu.hpp>
#include <nbla/function.hpp>
#include <nbla/function_registry.hpp>

namespace nbla {

NBLA_REGISTER_FUNCTION_HEADER(Gather, int, int);

/**
Gather from the input data according to the index.

Given the input data \f$X\f$ of \f$(D_{0}, \ldots, D_{N-1})\f$ shape and
the indices \f$IDX\f$ of \f$(I_{0}, \ldots, I_{M-1})\f$ shape, in case of
`batch_dims = 0`,
the gather outputs

\f{eqnarray*}{
  && Y[d_{0}, \ldots, d_{axis - 1}, i_{0}, \ldots, i_{M-1}, d_{axis + 1},
\ldots, d_{N-1}] = \\
  && X[d_{0}, \ldots, d_{axis - 1}, IDX[i_{0}, \ldots, i_{M-1}], d_{axis + 1},
\ldots, d_{N-1}].
\f}

Generally, the gather ouptuts

\f{eqnarray*}{
  && Y[d_{0}, \ldots, d_{axis - 1}, i_{B}, \ldots, i_{M-1}, d_{axis + 1},
\ldots, d_{N-1}] = \\
  && X[d_{0}, \ldots, d_{axis - 1}, IDX[i_{0}, \ldots, i_{B - 1}, i_{B} \ldots,
i_{M-1}], d_{axis + 1}, \ldots d_{N-1}].
\f}

where \f$B\f$ = `batch_dims`.

`x.shape[:batch_dims]` must be equal to `indices.shape[:batch_dims]`.

Output shape is `x.shape[:axis] + indices.shape[batch_dims:] + x.shape[axis +
1]`.

Inputs:
- Data from which to gather.
- Index with which to gather.

Outputs:
- Gathered output.

@tparam T Data type for computation.
@param axis Axis in the data to gather from. `axis` must be greater than or
equal to `batch_dims`.
@param batch_dims The number of batch dimensions.
\ingroup FunctionImplGrp
 */
template <typename T> class Gather : public BaseFunction<int, int> {
protected:
  int axis_;
  int batch_dims_;

public:
  Gather(const Context &ctx, int axis, int batch_dims)
      : BaseFunction(ctx, axis, batch_dims), axis_(axis),
        batch_dims_(batch_dims) {}
  virtual ~Gather() {}
  virtual shared_ptr<Function> copy() const {
    return create_Gather(ctx_, axis_, batch_dims_);
  }
  virtual int min_inputs() { return 2; }
  virtual int min_outputs() { return 1; }
  virtual vector<dtypes> in_types() {
    return vector<dtypes>{get_dtype<T>(), get_dtype<T>()};
  }
  virtual vector<dtypes> out_types() { return vector<dtypes>{get_dtype<T>()}; }
  virtual vector<string> allowed_array_classes() {
    return SingletonManager::get<Cpu>()->array_classes();
  }
  virtual string name() { return "Gather"; }
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
    if (i == 0 && j == 1) {
      return true;
    }
    return false;
  }
};
}
#endif
