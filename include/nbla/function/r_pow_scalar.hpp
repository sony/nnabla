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

/** RPow Scalar
 */
#ifndef __NBLA_FUNCTION_RPOW_SCALAR_HPP__
#define __NBLA_FUNCTION_RPOW_SCALAR_HPP__

#include <nbla/function/utils/base_transform_unary.hpp>

#include <cmath>

namespace nbla {

/** @class RPowScalar
@brief Elementwise r pow scalar defined as
@f[
y_i = v^{x_i}.
@f]

Inputs:
- N-D array.

Outputs:
- N-D array.

@tparam T Data type for computation.
@param val Value of the scalar.
\ingroup FunctionImplGrp
 */
NBLA_DEFINE_TRANSFORM_UNARY_1(RPowScalar, std::pow((T)a0, x),
                              dy *std::pow((T)a0, x) * std::log((T)a0), false,
                              true, double);
}
#endif
