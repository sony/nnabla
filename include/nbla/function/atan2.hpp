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

#ifndef NBLA_FUNCTION_ATAN2_HPP
#define NBLA_FUNCTION_ATAN2_HPP

#include <nbla/function/utils/base_transform_binary.hpp>

#include <cmath>

namespace nbla {

/** @class ATan2
@brief Arc tangent 2 (ATan2) defined as
@f[
y_i = \atan (x1/x0) if x0 > 0,
y_i = \atan (x1/x0) + \pi if x0 < 0 and x1 \geq 0,
y_i = \atan (x1/x0) - \pi if x0 <0 and x1 < 0,
y_i = \pi/2 if x0=0 and x1 > 0,
y_i = -\pi/2 if x0=0 and x1 < 0,
y_i = undefined if x0=0 and x1=0
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
                             -dy *x0 / (x0 * x0 + x1 * x1), true, true);
}
#endif
