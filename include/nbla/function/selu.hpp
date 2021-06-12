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

/** SELU
 */
#ifndef __NBLA_FUNCTION_SELU_HPP__
#define __NBLA_FUNCTION_SELU_HPP__

#include <nbla/cpu.hpp>
#include <nbla/function.hpp>
#include <nbla/function_registry.hpp>

namespace nbla {

NBLA_REGISTER_FUNCTION_HEADER(SELU, double, double);

/** @class SELU
@brief The Scaled Exponential Linear Unit (SELU) function by Klambauer et al.
(2017), defined as
@f[
   y_i= \lambda \left\{
    \begin{array}{ll}
      x_i & (x > 0)\\
      \alpha (\exp(x_i) - 1) & (x \leq 0)
    \end{array} \right..
@f]

The coefficients :math:`\lambda` and :math:`\alpha` default to the following
values :math:`\lambda_{01}` and :math:`\alpha_{01}`, respectively, provided by
Klambauer et al. (2017):

@f[
    \begin{array}{lll}
      \lambda_{01} &=&  \left(  1 - \operatorname{erfc}\left( \frac{1}{\sqrt{2}}
\right) \sqrt{e}  \right)
                  \sqrt{2 \pi} \\
                 && \left(
                      2 \operatorname{erfc} \left( \sqrt{2} \right) e^2
                      + \pi \operatorname{erfc}\left( \frac{1}{\sqrt{2}}
\right)^2 e
                      \right. \\
                 && \left.
                      - 2(2 + \pi) \operatorname{erfc} \left( \frac{1}{\sqrt{2}}
\right) \sqrt{e}
                      + \pi + 2
                 \right)^{-1/2}  \\
              &\approx& 1.0507 \\
      \alpha_{01} &=&  - \frac
                    {\sqrt {\frac {2}{\pi}}}
                    {\operatorname{erfc} \left( \frac{1}{\sqrt{2}} \right) \exp
\left(\frac {1} {2} \right) - 1} \\
              &\approx& 1.67326
    \end{array}
@f]

Inputs:
- N-D array.

Outputs:
- N-D array with the same shape as input.

@tparam T Data type for computation.
@param scale The coefficient @f$\lambda@f$ in the definition.
@param alpha The coefficient @f$\alpha@f$ in the definition.

@sa
Klambauer, G., Unterthiner, T., Mayr, A., & Hochreiter, S. (2017).
Self-Normalizing Neural Networks. In Advances in Neural Information
Processing Systems (NIPS).
https://arxiv.org/abs/1706.02515

\ingroup FunctionImplGrp
 */
template <typename T> class SELU : public BaseFunction<double, double> {
protected:
  float scale_;
  float alpha_;

public:
  SELU(const Context &ctx, double scale, double alpha)
      : BaseFunction<double, double>(ctx, (float)scale, (float)alpha),
        scale_((float)scale), alpha_((float)alpha) {}
  virtual ~SELU() {}
  virtual shared_ptr<Function> copy() const {
    return create_SELU(ctx_, scale_, alpha_);
  }
  virtual vector<dtypes> in_types() { return vector<dtypes>{get_dtype<T>()}; }
  virtual vector<dtypes> out_types() { return vector<dtypes>{get_dtype<T>()}; }
  virtual int min_inputs() { return 1; }
  virtual int min_outputs() { return 1; }
  virtual string name() { return "SELU"; }
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
  virtual bool grad_depends_input_data_impl(int i, int j) const { return true; }
};
}
#endif
