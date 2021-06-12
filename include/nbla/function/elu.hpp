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

/** ELU
 */
#ifndef __NBLA_FUNCTION_ELU_HPP__
#define __NBLA_FUNCTION_ELU_HPP__

#include <nbla/function/utils/base_transform_unary.hpp>

#include <cmath>

namespace nbla {

/** @class ELU
@brief Exponential Linear Unit (ELU) defined as
@f[
   y_i= \left\{
    \begin{array}{ll}
      x_i & (x > 0)\\
      \alpha (\exp(x_i) - 1) & (x \leq 0)
    \end{array} \right..
@f]

Inputs:
- N-D array.

Outputs:
- N-D array with the same shape as input.

@tparam T Data type for computation.
@param alpha Coefficient for negative outputs. @f$\alpha@f$ in definition.

@sa
Clevart et al., Fast and Accurate Deep Network Learning by Exponential Linear
Units (ELUs).
http://arxiv.org/abs/1511.07289

\ingroup FunctionImplGrp

*/
NBLA_DEFINE_TRANSFORM_UNARY_1(ELU, x >= (T)0 ? x : (T)a0 * (std::exp(x) - (T)1),
                              x >= (T)0 ? dy : dy * (T)a0 * std::exp(x), false,
                              true, double);
}
#endif
