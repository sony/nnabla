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

#ifndef NBLA_FUNCTION_GELU_HPP
#define NBLA_FUNCTION_GELU_HPP

#include <nbla/function/utils/base_transform_unary.hpp>

#include <cmath>

#ifndef M_PI
#define M_PI (3.14159265358979323846)
#endif

namespace nbla {

NBLA_REGISTER_FUNCTION_HEADER(GELU);

/**
Inputs:
- N-D array.

Outputs:
- N-D array.

\ingroup FunctionImplGrp
 */

NBLA_DEFINE_TRANSFORM_UNARY(
    GELU, (x / 2) * (1 + std::tanh((std::sqrt((T)(2 / M_PI)) *
                                    (x + (T)0.044715 * std::pow(x, 3))))),
    dy *(0.5 * (1 + std::tanh(std::sqrt((T)(2 / M_PI)) *
                              (x + (T)0.044715 * std::pow(x, 3)))) +
         0.5 * x * (1 - std::pow(std::tanh(std::sqrt((T)(2 / M_PI)) *
                                           (x + (T)0.044715 * std::pow(x, 3))),
                                 2)) *
             std::sqrt((T)(2 / M_PI)) * (1 + 0.134145 * std::pow(x, 2))),
    true, true);
}
#endif
