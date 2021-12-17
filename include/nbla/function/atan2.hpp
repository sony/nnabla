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

#ifndef NBLA_FUNCTION_ATAN2_HPP
#define NBLA_FUNCTION_ATAN2_HPP

#include <nbla/function/utils/base_transform_binary.hpp>

#include <cmath>

namespace nbla {

/** @class ATan2
@brief Arc tangent 2 (ATan2) defined as
@f[
y_i = \arctan (x_1/x_0) \text{ if } x_0 > 0,\\
y_i = \arctan (x_1/x_0) + \pi \text{ if } x_0 < 0 \text{ and } x_1 \geq 0,\\
y_i = \arctan (x_1/x_0) - \pi \text{ if } x_0 < 0 \text{ and } x_1 < 0,\\
y_i = \pi/2 \text{ if } x_0 = 0 \text{ and } x_1 > 0,\\
y_i = -\pi/2 \text{ if } x_0 = 0 \text{ and } x_1 < 0,\\
y_i = undefined \text{ if } x_0=0 \text{ and } x_1=0
@f]

Inputs:
- N-D array.
- N-D array.

Outputs:
- N-D array.

@tparam T Data type for computation.
\ingroup FunctionImplGrp
 */

NBLA_DEFINE_TRANSFORM_BINARY(ATan2, std::atan2(x0, x1),
                             dy *x1 / (x0 * x0 + x1 * x1),
                             -dy *x0 / (x0 * x0 + x1 * x1), false, false, true,
                             true);
}
#endif
