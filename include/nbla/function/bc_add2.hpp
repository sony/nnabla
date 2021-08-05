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

/** Bc_Add2
 */
#ifndef __NBLA_FUNCTION_BC_ADD2_HPP__
#define __NBLA_FUNCTION_BC_ADD2_HPP__

#include <nbla/function/utils/base_transform_binary.hpp>

namespace nbla {

/** @class BcAdd2
@brief Broadcastable Add2 operation.
@note This shouldn't be used by users. This is used in Add2. Other elementwise
binary operations are not implementing broadcastable because the original
implementations are already broadcastable. Add2 is a special case in which
in-place computation is allowed. We need this for an implementation for
broadcastable Add2. If Add2's inputs require broadcasting, it's fallback into
BcAdd2 operation. See setup_impl of add2.cpp.
 */
NBLA_DEFINE_TRANSFORM_BINARY_INPLACE(BcAdd2, x0 + x1, dy, dy, false, false,
                                     false, false, true);
}
#endif
