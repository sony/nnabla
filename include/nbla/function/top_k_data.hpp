// Copyright 2018,2019,2020,2021 Sony Corporation.
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

#ifndef NBLA_FUNCTION_TOP_K_DATA_HPP
#define NBLA_FUNCTION_TOP_K_DATA_HPP

#include <nbla/cpu.hpp>
#include <nbla/function.hpp>
#include <nbla/function_registry.hpp>

namespace nbla {

NBLA_REGISTER_FUNCTION_HEADER(TopKData, int, bool, bool, int);

/** Select the `k` largest values from each sample in `x` to propagate
unmodified and set all other values to 0. If `abs` is True, the `k`
largest values are selected by magnitude. If `reduce` is True (the
default), all feature dimensions are reduced to a single dimension of
size `k` that propagates only the `k` largest values. Otherwise, if
`reduce` is False, input and output dimensions are identical.
 Dimensions before `base_axis` are treated as number of sample
dimensions and `k` values get selected from all elements of a
sample (dimensions from `base_axis`) regardless of shape.

Inputs:

- N-D array

Outputs:

- N-D array.

@tparam T Data type for computation.

@param k Number of largest data values to propagate.

@param abs Determine largest data values by magnitude.

@param reduce Reduce feature size to one dimension of size `k`.

@param base_axis First dimension of the sample shape.

\ingroup FunctionImplGrp
 */
template <typename T>
class TopKData : public BaseFunction<int, bool, bool, int> {
protected:
  int k_;
  bool abs_;
  bool reduce_;
  int base_axis_;

  Size_t ns_;          // number of input samples
  Size_t ss_;          // input sample size
  Size_t fs_;          // output feature size
  Variable top_k_idx_; // top-k indices needed for backprop when reducing
  bool forward_done_;  // prevent execution of backward before forward

public:
  TopKData(const Context &ctx, int k, bool abs, bool reduce, int base_axis)
      : BaseFunction(ctx, k, abs, reduce, base_axis), k_(k), abs_(abs),
        reduce_(reduce), base_axis_(base_axis) {}
  virtual ~TopKData() {}
  virtual shared_ptr<Function> copy() const {
    return create_TopKData(ctx_, k_, abs_, reduce_, base_axis_);
  }
  virtual int min_inputs() { return 1; }
  virtual int min_outputs() { return 1; }
  virtual vector<dtypes> in_types() { return vector<dtypes>{get_dtype<T>()}; }
  virtual vector<dtypes> out_types() { return vector<dtypes>{get_dtype<T>()}; }
  virtual vector<string> allowed_array_classes() {
    return SingletonManager::get<Cpu>()->array_classes();
  }
  virtual string name() { return "TopKData"; }
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
