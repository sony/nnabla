// Copyright 2019,2020,2021 Sony Corporation.
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

#ifndef NBLA_FUNCTION_ISINF_HPP
#define NBLA_FUNCTION_ISINF_HPP

#include <nbla/function/utils/base_transform_unary.hpp>

namespace nbla {
/** Test element-wise for Inf/-Inf.

Inputs:
- N-D array.

Outputs:
- N-D array.

@tparam T Data type for computation.
\ingroup FunctionImplGrp
 */
NBLA_DEFINE_TRANSFORM_UNARY_NO_GRAD(IsInf, std::isinf(x) ? (T)1 : (T)0);
}
#endif
