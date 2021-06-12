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

// affine.hpp

#ifndef __NBLA_FUNCTION_DOT_HPP__
#define __NBLA_FUNCTION_DOT_HPP__

#include <nbla/cpu.hpp>
#include <nbla/function.hpp>
#include <nbla/function_registry.hpp>

#include <memory>
#include <string>

namespace nbla {

using std::string;

NBLA_REGISTER_FUNCTION_HEADER(Affine, int);

/** Affine also called as fully connected layer defined as
@f[
{\mathbf y} = {\mathbf A} {\mathbf x} + {\mathbf b}.
@f]

Inputs (\f$B\f$ is base_axis):
- Input N-D array with shape
  (\f$M_0 \times ... \times M_{B-1} \times D_B \times ... \times D_N\f$).
  Dimensions before and after base_axis are flattened as if it is a matrix.
- Weight matrix with shape (\f$(D_B \times ... \times D_N) \times L\f$)
- (optional) Bias vector (\f$L\f$)

Outputs:
- \f$(B + 1)\f$-D array. (\f$ M_0 \times ... \times M_{B-1} \times L \f$)

@tparam T Data type for computation.
@param base_axis Base axis of Affine operation. Dimensions up to base_axis
is treated as sample dimension.
\ingroup FunctionImplGrp
 */
template <typename T> class Affine : public BaseFunction<int> {
protected:
  int base_axis_;
  Size_t i_row_, i_col_, w_row_, w_col_, o_row_, o_col_;

public:
  Affine(const Context &ctx, int base_axis)
      : BaseFunction(ctx, base_axis), base_axis_(base_axis) {}

  virtual shared_ptr<Function> copy() const {
    return create_Affine(ctx_, base_axis_);
  }

  virtual vector<dtypes> in_types() {
    return vector<dtypes>{get_dtype<T>(), get_dtype<T>(), get_dtype<T>()};
  }
  virtual vector<dtypes> out_types() { return vector<dtypes>{get_dtype<T>()}; }
  virtual int min_inputs() { return 2; }
  virtual int min_outputs() { return 1; }
  virtual string name() { return "Affine"; }
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
  virtual bool grad_depends_input_data_impl(int i, int j) const {
    if (i == 0 && j == 1) {
      return true;
    }
    if (i == 1 && j == 0) {
      return true;
    }
    return false;
  }
};
}
#endif
