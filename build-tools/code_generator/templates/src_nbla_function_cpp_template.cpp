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
// 
// *WARNING*
// THIS FILE IS AUTO-GENERATED DUMMY CODE BY CODE GENERATOR.
// PLEASE IMPLEMENT REAL CODE AND DELETE THIS MESSAGE SOON.
// If you want to change dummy code, edit following files.
// - build-tools/code_generator/function_generator/generate_src_nbla_function_cpp.py
// - build-tools/code_generator/templates/src_nbla_function_cpp_template.cpp

/** {func_name}
 */
#include <nbla/array.hpp>
#include <nbla/variable.hpp>
#include <nbla/function/{func_name_snakecase}.hpp>

#include <algorithm>

namespace nbla {{

NBLA_REGISTER_FUNCTION_SOURCE({func_arg_variable_types});

template <{template_defines}>
void {func_name}<{templates}>::setup_impl(const Variables &inputs, const Variables &outputs) {{
  // TODO TEMPLATE CODE
}}

template <{template_defines}>
void {func_name}<{templates}>::forward_impl(const Variables &inputs, const Variables &outputs) {{
  // TEMPLATE CODE
}}

template <{template_defines}>
void {func_name}<{templates}>::backward_impl(const Variables &inputs, const Variables &outputs,
					     const vector<bool> &propagate_down,
					     const vector<bool> &accum) {{
  // TEMPLATE CODE
}}

// Template instantiation
template class {func_name}<{ctypes}>;
}} // namespace nbla
