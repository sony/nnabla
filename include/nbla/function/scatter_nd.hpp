// Copyright 2019,2020,2021 Sony Corporation.
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

#ifndef NBLA_FUNCTION_SCATTER_ND_HPP
#define NBLA_FUNCTION_SCATTER_ND_HPP

#include <nbla/cpu.hpp>
#include <nbla/function.hpp>
#include <nbla/function_registry.hpp>

namespace nbla {

NBLA_REGISTER_FUNCTION_HEADER(ScatterNd, const vector<int> &, bool);

/** Scatter `data` into a new array of given `shape` according to `indices`.
  This operation is the inverse of :func:`~nnabla.functions.gather_nd`.

  The forward of :func:`~nnabla.functions.scatter_nd` is equivalent to:

  @code{.py}
  def scatter_nd(data, indices, shape):
      import numpy as np
      if isinstance(indices, np.ndarray)
          indices = indices.tolist()
      result = np.zeros(shape, dtype=data.dtype)
      result[indices] = data
      return result

  Examples:

    inputs:
      data:
        doc: N-D array input data.
      indices:
        doc: N-D array scatter indices.
    arguments:
      shape:
        doc: Shape of output variable.
        type: repeated int64
    outputs:
      y:
        doc: N-D array of given `shape`.
  @endcode

Inputs:
- N-D array input data
- N-D array scatter indices

Outputs:
- N-D array of given `shape`

@param shape Shape of output variable.

\ingroup FunctionImplGrp
 */
template <typename T>
class ScatterNd : public BaseFunction<const vector<int> &, bool> {
protected:
  const vector<int> shape_;
  const bool add_;

public:
  ScatterNd(const Context &ctx, const vector<int> &shape, bool add)
      : BaseFunction(ctx, shape, add), shape_(shape), add_(add) {}
  virtual ~ScatterNd() {}
  virtual shared_ptr<Function> copy() const {
    return create_ScatterNd(ctx_, shape_, add_);
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
  virtual string name() { return "ScatterNd"; }
  virtual int inplace_data(int i) const {
    return i == 2 ? Function::INPLACE : Function::NOT_INPLACE;
  }
  virtual int inplace_data_with(int i) const { return 0; }
  virtual int inplace_grad(int i) const {
    return i == 2 ? Function::INPLACE : Function::NOT_INPLACE;
  }
  virtual int inplace_grad_with(int i) const { return 0; }
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
    if (i == 0 && j == 1)
      return true;
    return false;
  }
};
}
#endif
