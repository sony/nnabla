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

#ifndef NBLA_FUNCTION_SCATTER_ADD_HPP
#define NBLA_FUNCTION_SCATTER_ADD_HPP

#include <nbla/cpu.hpp>
#include <nbla/function.hpp>
#include <nbla/function_registry.hpp>

namespace nbla {

NBLA_REGISTER_FUNCTION_HEADER(ScatterAdd, int);

/**
  Add all values from `x1` into the `x0` according to index specified by
`indices`.
  This function adds `x1` into the copy of `x0` and outputs the copy.
  The original `x0` will not be changed.
  `x0`, `indices` and `x1` must have same number of dimensions.

  The forward of :func:`~nnabla.functions.scatter_add` is equivalent to:

  @code{.py}
  def scatter_add(x0, indices, x1, axis):
      # Assuming each input is 3 dimensional
      import numpy as np
      output = np.copy(x0)
      for i in range(indices.shape[0]):
          for j in range(indices.shape[1]):
              for k in range(indices.shape[2]):
                  if axis == 0:
                      output[indices[i][j][k]][j][k] += x1[i][j][k]
                  elif axis == 1:
                      output[i][indices[i][j][k]][k] += x1[i][j][k]
                  elif axis == 2:
                      output[i][j][indices[i][j][k]] += x1[i][j][k]
      return output

  inputs:
    x0:
      doc: N-D array which the data is added to its copy.
    indices:
      doc: N-D array scatter indices. The size of each dimension must be equal
          or smaller than that of x0 except for the specified axis. The value
          of indices must be smaller than the size of specified axis' dimension
          of x0. The size of each dimension must be equal or smaller than that
          of x1. Indices must not be negative.
    x1:
      doc: N-D array which is scattered and added to x0.
  arguments:
    axis:
      doc: Axis along which to index. The axis must not exceed the inputs'
          dimension.
      type: int64
      default: 0
  outputs:
    y:
      doc: N-D array which contains the result of scatter addition. The shape is
          same as x0.
  @endcode

Inputs:
- N-D array x0
- N-D array scatter indices
- N-D array x1

Outputs:
- N-D array of shape same as x0

@param axis Axis along which to index

\ingroup FunctionImplGrp
 */
template <typename T> class ScatterAdd : public BaseFunction<int> {
protected:
  int axis_;

public:
  ScatterAdd(const Context &ctx, int axis)
      : BaseFunction(ctx, axis), axis_(axis) {}
  virtual ~ScatterAdd() {}
  virtual shared_ptr<Function> copy() const {
    return create_ScatterAdd(ctx_, axis_);
  }
  virtual int min_inputs() { return 3; }
  virtual int min_outputs() { return 1; }
  virtual vector<dtypes> in_types() {
    return vector<dtypes>{get_dtype<T>(), get_dtype<T>(), get_dtype<T>()};
  }
  virtual vector<dtypes> out_types() { return vector<dtypes>{get_dtype<T>()}; }
  virtual vector<string> allowed_array_classes() {
    return SingletonManager::get<Cpu>()->array_classes();
  }
  virtual string name() { return "ScatterAdd"; }
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
