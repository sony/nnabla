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

#ifndef NBLA_FUNCTION_GATHER_ND_HPP
#define NBLA_FUNCTION_GATHER_ND_HPP

#include <nbla/cpu.hpp>
#include <nbla/function.hpp>
#include <nbla/function_registry.hpp>

namespace nbla {

NBLA_REGISTER_FUNCTION_HEADER(GatherNd);

/** Gather elements or slices from `data` according to `indices`, which must
  be at least two-dimensional with the first dimension :math:`M` being less or
  equal to the :math:`N` dimensions of `data`. Given `data` with shape
  :math:`(X_0, X_1, ..., X_{N-1})` and indices with shape :math:`(M, Y_0, ...,
  Y_{K-1})` output has shape :math:`(Y_0, ..., Y_{K-1}, X_M, ..., X_{N-1})`.
  If :math:`M == N`, output shape is simply :math:`(Y_0, ..., Y_{K-1})`.

  The forward of :func:`~nnabla.functions.gather_nd` is equivalent to the
  following Python code:

  @code{.py}
    def gather_nd(data, index):
        import numpy as np
        tmp_index = index.reshape(index.shape[0], -1)
        tmp_index = (idx + (Ellipsis,) for idx in zip(*new_index))
        out_shape = index.shape[1:] + data.shape[index.shape[0]:]
        return np.vstack(data[idx] for idx in tmp_index).reshape(*out_shape)
  @endcode

Inputs:

- N-D array `data`
- N-D array `indices`

Outputs:

- N-D array

\ingroup FunctionImplGrp
 */
template <typename T> class GatherNd : public BaseFunction<> {
public:
  GatherNd(const Context &ctx) : BaseFunction(ctx) {}
  virtual ~GatherNd() {}
  virtual shared_ptr<Function> copy() const { return create_GatherNd(ctx_); }
  virtual int min_inputs() { return 2; }
  virtual int min_outputs() { return 1; }
  virtual vector<dtypes> in_types() {
    return vector<dtypes>{get_dtype<T>(), get_dtype<T>()};
  }
  virtual vector<dtypes> out_types() { return vector<dtypes>{get_dtype<T>()}; }
  virtual vector<string> allowed_array_classes() {
    return SingletonManager::get<Cpu>()->array_classes();
  }
  virtual string name() { return "GatherNd"; }
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
