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

/** Huber loss
 */
#ifndef __NBLA_FUNCTION_HUBAR_LOSS_HPP__
#define __NBLA_FUNCTION_HUBAR_LOSS_HPP__

#include <nbla/function/utils/base_transform_binary.hpp>

#include <cmath>

namespace nbla {

/** @class HuberLoss
@brief Elementwise Huber loss defined as
@f[
   y_i= \left\{
    \begin{array}{ll}
      d^2 & (|d| < \delta)\\
      \delta (2 |d| - \delta) & (otherwise)
    \end{array} \right.
@f]
where @f$d = x^{(0)}_i - x^{(1)}_i@f$.

Inputs:
- N-D array.
- N-D array.

Outputs:
- N-D array.

@tparam T Data type for computation.
@param delta
\ingroup FunctionImplGrp
 */
NBLA_DEFINE_TRANSFORM_BINARY_1(
    HuberLoss,
    (std::abs(x0 - x1) < (T)a0) ? (std::abs(x0 - x1) * std::abs(x0 - x1))
                                : ((T)a0 *((T)2 * std::abs(x0 - x1) - (T)a0)),
    (T)2 * dy * (std::abs(x0 - x1) < (T)a0
                     ? (x0 - x1)
                     : ((x0 - x1) > (T)0 ? (T)1 : (T)-1) * (T)a0),
    (T)-2 * dy * (std::abs(x0 - x1) < (T)a0
                      ? (x0 - x1)
                      : ((x0 - x1) > (T)0 ? (T)1 : (T)-1) * (T)a0),
    false, false, true, true, float);
}
#endif
