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

/** Binary Sigmoid
 */
#ifndef __NBLA_FUNCTION_BINARY_SIGMOID_HPP__
#define __NBLA_FUNCTION_BINARY_SIGMOID_HPP__

#include <nbla/function/utils/base_transform_unary.hpp>

#include <cmath>

namespace nbla {

/** @class BinarySigmoid
@brief Elementwise BinarySigmoid function defined as
@f[
y_i = \left \{
    \begin{array}{ll}
      1 & (x_i > 0)\\
      0 & (x_i \leq 0)
    \end{array} \right..
@f]

Inputs:
- N-D array.

Outputs:
- N-D array.

@tparam T Data type for computation.

@sa Courbariaux, Matthieu, and Yoshua Bengio. Binarynet: Training deep
neural networks with weights and activations constrained to+ 1 or-1.
preprint arXiv:1602.02830 (2016).

\ingroup FunctionImplGrp
 */
NBLA_DEFINE_TRANSFORM_UNARY(BinarySigmoid, (x > (T)0) ? (T)1 : (T)0,
                            (std::abs(x) >= (T)1) ? (T)0 : dy *(T)0.5, false,
                            true);
}
#endif
