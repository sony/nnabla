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

#ifndef __NBLA_FUNCTION_CLIPGRADBYNORM_HPP__
#define __NBLA_FUNCTION_CLIPGRADBYNORM_HPP__

#include <nbla/cpu.hpp>
#include <nbla/function.hpp>
#include <nbla/function_registry.hpp>

namespace nbla {

NBLA_REGISTER_FUNCTION_HEADER(ClipGradByNorm, float, const vector<int> &);

/** ClipGradByNorm
@f[
g_x = clip\_norm \times \frac{g_y}{\|g_y\|_2}
@f]

Inputs:
- N-D array.

Outputs:
- N-D array.

@tparam T Data type for computation.
@param clip_norm Value of the scalar to clip to the norm of input to `clip_norm`
in the forward pass.
@param axes Axes to be reduced. If empty list is given, all dimensions are
reduced to scalar. This is used in the forward pass.

\ingroup FunctionImplGrp
 */
template <typename T>
class ClipGradByNorm : public BaseFunction<float, const vector<int> &> {

protected:
  float clip_norm_;
  const vector<int> axes_;

  shared_ptr<Function> sum_;
  shared_ptr<Function> pow_scalar_;
  shared_ptr<Function> broadcast_;

public:
  ClipGradByNorm(const Context &ctx, float clip_norm, const vector<int> &axes)
      : BaseFunction<float, const vector<int> &>(ctx, clip_norm, axes),
        clip_norm_(clip_norm), axes_(axes) {}
  virtual ~ClipGradByNorm() {}
  virtual shared_ptr<Function> copy() const {
    return create_ClipGradByNorm(ctx_, clip_norm_, axes_);
  }
  virtual vector<dtypes> in_types() { return vector<dtypes>{get_dtype<T>()}; }
  virtual vector<dtypes> out_types() { return vector<dtypes>{get_dtype<T>()}; }
  virtual int min_inputs() { return 1; }
  virtual int min_outputs() { return 1; }
  virtual string name() { return "ClipGradByNorm"; }
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
    return false;
  }
};
}
#endif
