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

#ifndef NBLA_FUNCTION_TILE_HPP
#define NBLA_FUNCTION_TILE_HPP

#include <nbla/cpu.hpp>
#include <nbla/function.hpp>
#include <nbla/function_registry.hpp>

namespace nbla {

NBLA_REGISTER_FUNCTION_HEADER(Tile, const vector<int> &);

/** Forward input `x` repeated the number of times given by `reps`. If `reps`
is a sequence, the output has dimension of ``d = max(len(reps), x.ndim)`` and
either `x` is promoted to be d-dimensional by prepending new axes or `reps`
 is promoted to x.ndim by prepending 1's.

Inputs:

- N-D array

Outputs:

- N-D array

@param reps Repetitions of input `x` along each axis.

\ingroup FunctionImplGrp
 */

template <typename T> class Tile : public BaseFunction<const vector<int> &> {
protected:
  const vector<int> reps_;
  NdArray idxmap_;

public:
  Tile(const Context &ctx, const vector<int> &reps)
      : BaseFunction(ctx, reps), reps_(reps) {}
  virtual ~Tile() {}
  virtual shared_ptr<Function> copy() const { return create_Tile(ctx_, reps_); }
  virtual int min_inputs() { return 1; }
  virtual int min_outputs() { return 1; }
  virtual vector<dtypes> in_types() { return vector<dtypes>{get_dtype<T>()}; }
  virtual vector<dtypes> out_types() { return vector<dtypes>{get_dtype<T>()}; }
  virtual vector<string> allowed_array_classes() {
    return SingletonManager::get<Cpu>()->array_classes();
  }
  virtual string name() { return "Tile"; }
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
