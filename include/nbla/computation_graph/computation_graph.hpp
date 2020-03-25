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

#ifndef __NBLA_COMPUTATION_GRAPH_HPP__
#define __NBLA_COMPUTATION_GRAPH_HPP__

#include <nbla/computation_graph/function.hpp>
#include <nbla/computation_graph/variable.hpp>
#include <nbla/nd_array.hpp>

namespace nbla {

/** Create CgVariable outputs.

    Created CGVariables are held as weak_ptr by cg_f. The `need_grad`
    flags are automatically applied to created outputs.

*/
NBLA_API vector<CgVariablePtr>
create_function_outputs(CgFunctionPtr cg_f, int n_outputs = -1,
                        bool prohibit_clear_output = false);

/** Connect function to network.

    Create outputs on-demand.
 */
NBLA_API vector<CgVariablePtr> connect(CgFunctionPtr cg_f,
                                       const vector<CgVariablePtr> &inputs,
                                       int n_outputs = 1,
                                       vector<NdArrayPtr> inplace_outputs = {},
                                       bool execute = false);

/** Steal some variable properties from `from` CgVariable to `to` in order to
   rewire previously constructed graphs.

    It takes parent, need_grad flags, inplace flags, and the variable contents.

    @param[in] from Its parent function is stolen by 'to'.
    @param[in,out] to A variable 'from''s parent stolen to.
*/
NBLA_API void steal_variable_from_to(CgVariablePtr from, CgVariablePtr to);

/** Forward given variables in single inference
 * Forward all given variables with shared fclosed flags.
 */
NBLA_API void forward_all(const vector<CgVariablePtr> variables,
                          bool clear_buffer = false,
                          bool clear_no_need_grad = false,
                          function_hook_type function_pre_hook = nullptr,
                          function_hook_type function_post_hook = nullptr);
}
#endif
