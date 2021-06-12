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

#ifndef NBLA_FUNCTION_PATCH_CORRELATION_HPP
#define NBLA_FUNCTION_PATCH_CORRELATION_HPP

#include <nbla/cpu.hpp>
#include <nbla/function.hpp>
#include <nbla/function_registry.hpp>

namespace nbla {

NBLA_REGISTER_FUNCTION_HEADER(PatchCorrelation, const vector<int> &,
                              const vector<int> &, const vector<int> &,
                              const vector<int> &, const vector<int> &);

/** PatchConvolution

Inputs:
- Input N-D array with shape (N, H, W, C).
- Input N-D array with shape (N, H, W, C).

Outputs:
- N-D array with shape (N, C_y, C_x, Ho, Wo)

@tparam T Data type for computation.
@param patch Height and width of the correlation patch.
@param shift Maximum vertical and horizontal displacement of patches.
@param patch_step Vertical and horizontal patch increments.
@param shift_step Vertical and horizontal shift increments.
@param padding Top, bottom, left and right padding extent.

\ingroup FunctionImplGrp
 */
template <typename T>
class PatchCorrelation
    : public BaseFunction<const vector<int> &, const vector<int> &,
                          const vector<int> &, const vector<int> &,
                          const vector<int> &> {
protected:
  const vector<int> patch_;
  const vector<int> shift_;
  const vector<int> patch_step_;
  const vector<int> shift_step_;
  const vector<int> padding_;

public:
  PatchCorrelation(const Context &ctx, const vector<int> &patch,
                   const vector<int> &shift, const vector<int> &patch_step,
                   const vector<int> &shift_step, const vector<int> &padding)
      : BaseFunction(ctx, patch, shift, patch_step, shift_step, padding),
        patch_(patch), shift_(shift), patch_step_(patch_step),
        shift_step_(shift_step), padding_(padding) {}
  virtual ~PatchCorrelation() {}
  virtual shared_ptr<Function> copy() const {
    return create_PatchCorrelation(ctx_, patch_, shift_, patch_step_,
                                   shift_step_, padding_);
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
  virtual string name() { return "PatchCorrelation"; }
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
