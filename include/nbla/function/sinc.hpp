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

/** Sinc
 */
#ifndef NBLA_FUNCTION_SINC_HPP
#define NBLA_FUNCTION_SINC_HPP

#include <nbla/function/utils/base_transform_unary.hpp>

#include <cmath>

namespace nbla {

NBLA_REGISTER_FUNCTION_HEADER(Sinc);

/** @class Sinc
@brief Sinc defined as
@f[
y_i = 1 \text{ if } x_i = 0.\\
y_i = \sin(x_i)/x_i \text{, otherwise.}
@f]

Inputs:
- N-D array.

Outputs:
- N-D array.

@tparam T Data type for computation.
\ingroup FunctionImplGrp
 */
NBLA_DEFINE_TRANSFORM_UNARY(Sinc, x == (T)0 ? (T)1 : std::sin(x) / x,
                            x == (T)0
                                ? (T)0
                                : dy * (std::cos(x) - std::sin(x) / x) / x,
                            false, true);
}
#endif
