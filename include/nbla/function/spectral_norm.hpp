// Copyright 2021 Sony Corporation.
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

#ifndef NBLA_FUNCTION_SPECTRAL_NORM_HPP
#define NBLA_FUNCTION_SPECTRAL_NORM_HPP

#include <nbla/cpu.hpp>
#include <nbla/function.hpp>
#include <nbla/function_registry.hpp>

#include <nbla/computation_graph/variable.hpp>

namespace nbla {

NBLA_REGISTER_FUNCTION_HEADER(SpectralNorm, int, int, float, bool, bool);

/** Spectral Normalization.

@f[
W_{sn} = \frac{W}{\sigma(W)}.
@f]

Inputs:
- N-D array of learnable weights. This is normally network parameter.
- 1-D array of singular vector. When `test == false`, the data region will be
updated during forward calculation.

Outputs:
- Spectrally normalized \f$W_{sn}\f$ with the same shape as \f$W\f$.

@tparam T Data type for computation.

@param dim Output dimension. If the dimension is not 0, then the specified
dimension becomes the most-left dimension by transposing.
@param itr Number of power iterations.
@param eps Epsilon for the normalization. This `eps` is added before taking the
sqrt in the norm computation.
@param test When in `true`, `u` will not be updated.

@sa Takeru Miyato, Toshiki Kataoka, Masanori Koyama, Yuichi Yoshida, "Spectral
Normalization for Generative Adversarial Networks", International Conference on
Learning Representations. 2018.

\ingroup FunctionImplGrp
 */
template <typename T>
class SpectralNorm : public BaseFunction<int, int, float, bool, bool> {
protected:
  int dim_;
  int itr_;
  float eps_;
  bool test_;
  bool output_u_;

  // Variables for shape of reshaped `w` and `u`
  int d0_, d1_;

public:
  SpectralNorm(const Context &ctx, int dim, int itr, float eps, bool test,
               bool output_u)
      : BaseFunction(ctx, dim, itr, eps, test, output_u), dim_(dim), itr_(itr),
        eps_(eps), test_(test), output_u_(output_u) {}
  virtual ~SpectralNorm() {}
  virtual shared_ptr<Function> copy() const {
    return create_SpectralNorm(ctx_, dim_, itr_, eps_, test_, output_u_);
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
  virtual string name() { return "SpectralNorm"; }
  virtual bool grad_depends_output_data(int i, int o) const { return i == 0; }
  virtual bool need_setup_recompute(int o) const { return true; }

protected:
  NBLA_API virtual CgVariablePtr spectral_norm(const Variables &inputs,
                                               const Variables &outputs);
  NBLA_API virtual CgVariablePtr
  spectral_norm_outer_most_dim(const Variables &inputs,
                               const Variables &outputs);
  NBLA_API virtual void setup_impl(const Variables &inputs,
                                   const Variables &outputs);
  NBLA_API virtual void forward_impl(const Variables &inputs,
                                     const Variables &outputs);
  NBLA_API virtual void backward_impl(const Variables &inputs,
                                      const Variables &outputs,
                                      const vector<bool> &propagate_down,
                                      const vector<bool> &accum);
  NBLA_API virtual void setup_recompute_impl(const Variables &inputs,
                                             const Variables &outputs);
  NBLA_API virtual void recompute_impl(const Variables &inputs,
                                       const Variables &outputs);
  virtual bool grad_depends_input_data_impl(int i, int j) const { return true; }
  virtual bool overwrite_input_data_in_forward_impl(int i) const {
    if (i == 1) {
      return true;
    }
    return false;
  }

private:
  // Members only used in a naive implementation with composite
  NdArrayPtr u_orig_, u_;
  CgVariablePtr last_output_cg_variable_;
};
}
#endif
