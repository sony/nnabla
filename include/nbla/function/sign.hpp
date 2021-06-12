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

/** Sign
 */
#ifndef __NBLA_FUNCTION_SIGN_HPP__
#define __NBLA_FUNCTION_SIGN_HPP__

#include <nbla/function/utils/base_transform_unary.hpp>

namespace nbla {

NBLA_REGISTER_FUNCTION_HEADER(Sign, float);

/** @class Sign
@brief Implementation of the elementwise sign function
@f[
y = \left\{
    \begin{array}{ll}
      1 & (x > 0)\\
      \alpha & (x = 0)\\
     -1 & (x < 0)
    \end{array} \right.
@f]
where the gradient is a `full` straight through, i.e., the gradient
is not modified by this function. By default, @f$ \alpha @f$ = 1.0 .

Inputs:
- N-D array.

Outputs:
- N-D array.

@param alpha Value for x == 0.0
*/
NBLA_DEFINE_TRANSFORM_UNARY_1(Sign,
                              (x > (T)0) ? (T)1 : ((x < (T)0) ? (T)-1 : (T)a0),
                              dy, false, false, float);
}
#endif
