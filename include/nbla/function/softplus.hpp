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

/** SoftPlus
 */
#ifndef NBLA_FUNCTION_SOFTPLUS_HPP
#define NBLA_FUNCTION_SOFTPLUS_HPP

#include <nbla/function/utils/base_transform_unary.hpp>

#include <cmath>

namespace nbla {

NBLA_REGISTER_FUNCTION_HEADER(SoftPlus);

/** SoftPlus
@brief SoftPlus defined as
@f[
y_i = \log(\exp(x)+1)
@f]

Inputs:
- N-D array.

Outputs:
- N-D array.

@tparam T Data type for computation.
\ingroup FunctionImplGrp
 */
NBLA_DEFINE_TRANSFORM_UNARY(SoftPlus, std::log(std::exp(x) + (T)1),
                            dy / ((T)1 + std::exp(-x)), false);
}
#endif
