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

#ifndef __NBLA_COMPUTATION_GRAPH_FUNCTION_HPP__
#define __NBLA_COMPUTATION_GRAPH_FUNCTION_HPP__

#include <nbla/function.hpp>

namespace nbla {

// Forward declaration
class CgVariable;
typedef shared_ptr<CgVariable> CgVariablePtr;

vector<Variable *> to_variable_pointers(const vector<CgVariablePtr> &variables);

/**
 */
class CgFunction {
  int rank_{0};
  vector<CgVariablePtr> inputs_;
  FunctionPtr func_;
  vector<std::weak_ptr<CgVariable>> outputs_;
  bool need_grad_{false};
  string info_;

public:
  typedef shared_ptr<CgFunction> Ptr;

  /** Ctor.
      @param[in] func shared_ptr of Function.
  */
  NBLA_API CgFunction(FunctionPtr func);
  /** Set inputs.
      Check if any of inputs requires gradient computation and store the flag in
      self. Also, rank will be set according to inputs' ranks.

      @param[in] inputs Function inputs as CgVariables.
  */
  NBLA_API void set_inputs(const vector<CgVariablePtr> &inputs);

  /**
   */
  inline FunctionPtr function() const { return func_; }

  /**
   */
  inline bool need_grad() const { return need_grad_; }

  /**
   */
  inline int rank() const { return rank_; }

  /** Store outputs as weak references (weak_ptr).

      @param[in] outputs Function outputs.
   */
  NBLA_API void set_outputs(const vector<CgVariablePtr> &outputs);

  /**
   */
  NBLA_API vector<CgVariablePtr> inputs() { return inputs_; }

  /**
   */
  NBLA_API vector<CgVariablePtr> outputs();

  /** Get number of inputs.
   */
  inline size_t num_inputs() const { return inputs_.size(); }

  /** Get number of outputs.
   */
  inline size_t num_outputs() const { return outputs_.size(); }

  /** Update need_grad flag by seeing output variables.
   */
  NBLA_API bool update_need_grad();

  NBLA_API vector<Variable *> function_inputs();
  NBLA_API vector<VariablePtr> function_outputs_shared();

  /**
   */
  inline void set_info(const string &info) { info_ = info; }
  /**
   */
  NBLA_API string info() const { return info_; }
};

/**
 */
typedef CgFunction::Ptr CgFunctionPtr;
}
#endif
