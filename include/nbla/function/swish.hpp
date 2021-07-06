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

/** Swish
 */
#ifndef __NBLA_FUNCTION_SWISH_HPP__
#define __NBLA_FUNCTION_SWISH_HPP__

#include <nbla/function/utils/base_transform_unary.hpp>

#include <cmath>

namespace nbla {

/** @class Swish
@brief Element-wise swish function, by Ramachandran et al. (2017).
@f[
   y_i = \frac{x_i}{1 + \exp(-x_i)}.
@f]

Inputs:
- N-D array.

Outputs:
- N-D array with the same shape as input.

@tparam T Data type for computation.

@sa
Prajit Ramachandran, Barret Zoph, and Quoc V. Le, Swish: a Self-Gated Activation
Function, arXiv:1710.05941 [cs.NE]
https://arxiv.org/abs/1710.05941

\ingroup FunctionImplGrp

 */
NBLA_DEFINE_TRANSFORM_UNARY(Swish, x / ((T)1 + std::exp(-x)),
                            dy *(y +
                                 ((T)1 / ((T)1 + std::exp(-x))) * ((T)1 - y)),
                            true, true);
}
#endif
