// Copyright 2017,2018,2019,2020,2021 Sony Corporation.
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

#include <utility>

namespace nbla {

using std::pair;

// Forward declaration
class CgVariable;
typedef shared_ptr<CgVariable> CgVariablePtr;

vector<Variable *> to_variable_pointers(const vector<CgVariablePtr> &variables);

/** Computation graph function.

A Function object is held in this object, and pointers to inputs and outputs
also kept.
 */
class CgFunction {
  friend class CgVariable;
  int rank_{0};
  vector<CgVariablePtr> inputs_;
  FunctionPtr func_;

  /* Wrapper object of output CgVariable.
   */
  struct OutputWrapper {
    std::weak_ptr<CgVariable> weak_reference;
    /*
      Output variables are weakly referenced to avoid circular dependencies,
      which means output variables may be deleted before it is used.
      To recover a deleted CgVariable instance, a Variable instance originally
      held by the CgVariable is kept aside the weak reference.
      This dosesn't cause circular dependency.
    */
    VariablePtr internal_variable;

    void set(CgVariablePtr v);
    CgVariablePtr get();
  };

  vector<OutputWrapper> outputs_;
  bool need_grad_{false};
  string info_;

public:
  typedef shared_ptr<CgFunction> Ptr;

  /** Ctor.
      @param[in] func shared_ptr of Function.
  */
  NBLA_API CgFunction(FunctionPtr func);

  /** Dtor. Erase all function_reference_ of inputs.
  */
  NBLA_API ~CgFunction();

  /** Set inputs. Note user shouldn't call this directly.

      @param[in] inputs Function inputs as CgVariables.
  */
  inline void set_inputs_(const vector<CgVariablePtr> &inputs) {
    inputs_ = inputs;
  }

  /** Calling setup function of an Function object internally held.
   */
  void setup();

  /** Get a weak reference output as a shared reference by index or raise.

      @param[in] i Output index.
  */
  inline CgVariablePtr output(int i);

  /**
   */
  inline FunctionPtr function() const { return func_; }

  /**
   */
  inline bool need_grad() const { return need_grad_; }

  /**
   */
  inline void set_need_grad(bool b) { need_grad_ = b; }

  /**
   */
  inline int rank() const { return rank_; }

  /**
   */
  inline void set_rank_(int rank) { rank_ = rank; }

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

  NBLA_API vector<Variable *> function_inputs();
  NBLA_API pair<vector<CgVariablePtr>, vector<Variable *>> function_outputs();

  /**
   */
  inline void set_info(const string &info) { info_ = info; }
  /**
   */
  NBLA_API string info() const { return info_; }

  void check_data_inplace(int i, CgVariablePtr input,
                          const vector<CgVariablePtr> &outputs);
  void verify_during_forward();
};

/**
 */
typedef CgFunction::Ptr CgFunctionPtr;
}
#endif
