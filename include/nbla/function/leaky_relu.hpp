// Copyright (c) 2017 Sony Corporation. All Rights Reserved.
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

/** LeakyReLU
 */
#ifndef __NBLA_FUNCTION_LEAKYRELU_HPP__
#define __NBLA_FUNCTION_LEAKYRELU_HPP__

#include <nbla/function/utils/base_transform_unary.hpp>

namespace nbla {

NBLA_REGISTER_FUNCTION_HEADER(LeakyReLU, float);

/** @class LeakyReLU
@brief Leaky Rectified Linear Unit (ReLU) defined as

@f[
   y_i= \left\{
    \begin{array}{ll}
      x_i & (x \geq 0)\\
      \alpha x_i & (x < 0)
    \end{array} \right..
@f]

Inputs:
- N-D array.

Outputs:
- N-D array with the same shape as input.

@tparam T Data type for computation.
@param alpha The slope value multiplied to negative numbers. @f$\alpha@f$ in the
definition.

\ingroup FunctionImplGrp

 */
NBLA_DEFINE_TRANSFORM_UNARY_1(LeakyReLU, x >= (T)0 ? x : (T)a0 * x,
                              x >= (T)0 ? dy : (T)a0 * dy, false, float);
}
#endif
