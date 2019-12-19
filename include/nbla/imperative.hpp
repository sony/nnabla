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

#ifndef __NBLA_IMPERATIVE_HPP__
#define __NBLA_IMPERATIVE_HPP__

#include <nbla/function.hpp>
#include <nbla/nd_array.hpp>

namespace nbla {

/** Execute a function given NdArray instances as inputs.

    @param[in] func A shared pointer of Function.
    @param[in] inputs A vector of NdArray as function inputs. NdArrays are
   converted to data regions of  Variables.
    @param[in] n_outputs Number of function outputs.
    @param[in,out] outputs, This can be empty usually. Elements which are not
   nullptr will be used as in-place outputs.
*/
NBLA_API vector<NdArrayPtr> execute(FunctionPtr func,
                                    const vector<NdArrayPtr> &inputs,
                                    int n_outputs,
                                    vector<NdArrayPtr> outputs = {});

NBLA_API void execute(FunctionPtr f, const Variables &inputs,
                      const Variables &outputs);
}

#endif
