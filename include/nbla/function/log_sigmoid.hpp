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

/** LogSigmoid
 */
#ifndef NBLA_FUNCTION_LOGSIGMOID_HPP
#define NBLA_FUNCTION_LOGSIGMOID_HPP

#include <nbla/function/utils/base_transform_unary.hpp>

#include <cmath>

namespace nbla {

NBLA_REGISTER_FUNCTION_HEADER(LogSigmoid);

/** @class LogSigmoid
@brief LogSigmoid defined as
@f[
y_i = \log(1 /(\exp(-x_i) + 1).
@f]

Inputs:
- N-D array.

Outputs:
- N-D array.

@tparam T Data type for computation.
\ingroup FunctionImplGrp
 */
NBLA_DEFINE_TRANSFORM_UNARY(LogSigmoid,
                            x > (T)0 ? -std::log(std::exp(-x) + (T)1)
                                     : x - std::log(std::exp(x) + (T)1),
                            dy / (std::exp(x) + (T)1), false, true);
}
#endif
