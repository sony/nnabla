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

/** SoftPlus
 */
#ifndef NBLA_FUNCTION_SOFTPLUS_HPP
#define NBLA_FUNCTION_SOFTPLUS_HPP

#include <nbla/function/utils/base_transform_unary.hpp>

#include <cmath>

namespace nbla {

/** SoftPlus
@brief SoftPlus defined as
@f[
y_i = \frac{1}{\beta} * \log(\exp(\beta * x)+1)
@f]
Inputs:
- N-D array.
Outputs:
- N-D array.
@tparam T Data type for computation.
\ingroup FunctionImplGrp
 */
NBLA_DEFINE_TRANSFORM_UNARY_1(
    SoftPlus, x > (T)0 ? x + std::log(std::exp(-x * (T)a0) + (T)1) / (T)a0
                       : (std::log(std::exp(x * (T)a0) + (T)1)) / (T)a0,
    dy / ((T)1 + std::exp(-(T)a0 * x)), false, true, double);
}
#endif
