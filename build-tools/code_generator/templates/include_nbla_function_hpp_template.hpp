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

// -*- coding:utf-8 -*-
/*
 * Copyright (C) 2016 Sony Corporation
 * This is UNPUBLISHED PROPRIETARY SOURCE CODE of Sony Corporation;
 * the contents of this file is not to be disclosed to third parties, copied
 * or duplicated in any form, in whole or in part, without the prior written
 * permission of Sony Corporation.
 *
 * *WARNING*
 * THIS FILE IS AUTO-GENERATED DUMMY CODE BY CODE GENERATOR.
 * PLEASE IMPLEMENT REAL CODE AND DELETE THIS MESSAGE SOON.
 * If you want to change dummy code, edit following files.
 * - build-tools/code_generator/function_generator/generate_include_nbla_function_hpp.py
 * - build-tools/code_generator/templates/include_nbla_function_hpp_template.hpp
 */

/** {func_name}
 */
#ifndef __NBLA_FUNCTION_{func_name_upcase}_HPP__
#define __NBLA_FUNCTION_{func_name_upcase}_HPP__

#include <nbla/cpu.hpp>
#include <nbla/function.hpp>
#include <nbla/function_registry.hpp>

namespace nbla {{

NBLA_REGISTER_FUNCTION_HEADER({func_arg_variable_types});

/** 
    @todo PLACE HERE FUNCTION DOCUMENTATION.
 */
template <{template_defines}>
class {func_name} : public BaseFunction<{base_function_types}> {{
{func_arg_variable_defines}
    
public:
  {func_name}({func_args}) : BaseFunction<{base_function_types}>({base_function_args}) {{}}
  virtual ~{func_name}() {{}}
  virtual shared_ptr<Function> copy() const {{
      return create_{func_name}({func_arg_variables});
    }}
  virtual vector<dtypes> in_types() {{
    return vector<dtypes>{{ {in_types} }};
  }}
  virtual vector<dtypes> out_types() {{
    return vector<dtypes>{{ {out_types} }};
  }}
  virtual int min_inputs() {{ return {min_inputs}; }}
  virtual int min_outputs() {{ return {min_outputs}; }}
  virtual string name() {{ return "{func_name}"; }}
  virtual vector<string> allowed_array_classes() {{
    return SingletonManager::get<Cpu>()->array_classes();
  }}

protected:
  NBLA_API virtual void setup_impl(const Variables &inputs, const Variables &outputs);
  NBLA_API virtual void forward_impl(const Variables &inputs, const Variables &outputs);
  NBLA_API virtual void backward_impl(const Variables &inputs, const Variables &outputs,
                                      const vector<bool> &propagate_down);
}};
}}
#endif
