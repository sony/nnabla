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

#ifndef __NBLA_COMPUTATION_GRAPH_VARIABLE_HPP__
#define __NBLA_COMPUTATION_GRAPH_VARIABLE_HPP__

#include <nbla/function.hpp>
#include <nbla/variable.hpp>

#include <functional>
#include <memory>
#include <unordered_map>
#include <unordered_set>

namespace nbla {

using std::unordered_map;
using std::unordered_set;
using std::string;

// Forward declaration
class CgFunction;
typedef shared_ptr<CgFunction> CgFunctionPtr;

/** Callback functions during backward.
 */
struct CommunicatorBackwardCallback {
  virtual void on_finish_function_backward(const CgFunctionPtr &ptr) {}
  virtual void on_finish_backward() {}
};
typedef shared_ptr<CommunicatorBackwardCallback>
    CommunicatorBackwardCallbackPtr;

/** Computation graph variable.

A Variable object is held in this object as a data container. In addition,
a CGVariable object keeps information about the computation
graph it belongs to. The information if such as the pointer to the parent
function which creates this variable and some performance optimization clues.
*/
class CgVariable {
  friend class CgFunction;
  enum NeedGrad { NG_NONE, NG_FALSE, NG_TRUE };
  struct FunctionReferenceInfo {
    bool need_setup{false};
  };
  NeedGrad need_grad_{NG_NONE}; ///< Whether the variable requires gradients.
  NeedGrad need_grad_state_{
      NG_NONE};     ///< Updated during graph construction or forward
                    /// propagation.
  VariablePtr var_; /// Variable instance.
  CgFunctionPtr parent_{nullptr}; ///< Function created this variable.
  int rank_{0};                   ///< Longest path from root variable.
  ///< Holds weak function references. <https://stackoverflow.com/a/22110715>
  unordered_map<CgFunction *,
                pair<std::weak_ptr<CgFunction>, FunctionReferenceInfo>>
      function_references_;
  bool allow_modify_data_{true}; ///< Whether the data can be in-placed.
  bool persistent_{false};       ///<Persistency flag against clearing.
  string name_{""};

  void
  visit_function_recursive(CgFunctionPtr func,
                           unordered_set<CgFunctionPtr> &fclosed,
                           std::function<void(CgFunctionPtr)> forward_callback);

  void visit_function_backward(
      CgFunctionPtr func, std::function<void(CgFunctionPtr)> backward_callback,
      vector<CommunicatorBackwardCallbackPtr> communicator_callbacks);

public:
  typedef shared_ptr<CgVariable> Ptr;

  /** Create 0-shaped variable with no need_grad flag.
   */
  NBLA_API CgVariable();

  /** Create 0-shaped variable with need_grad option.

      @param[in] need_grad Whether this variable requires gradient computation
                 or not
   */
  NBLA_API CgVariable(bool need_grad);

  /** Create a variable by shape.

      @param[in] shape Shape passed to Variable object held in the created
                 instance.
   */
  NBLA_API CgVariable(Shape_t shape);

  /** Create a variable by shape with need_grad option.

      @param[in] shape Shape passed to Variable object held in the created
                 instance.
      @param[in] need_grad Whether this variable requires gradient computation
                 or not
   */
  NBLA_API CgVariable(Shape_t shape, bool need_grad);

  /** Create by a Variable instance.

      @param[in] var Reference of an existing Variable object.
   */
  NBLA_API CgVariable(VariablePtr var);

  /** Create by a Variable instance.

      @param[in] var Reference of an existing Variable object.
      @param[in] need_grad Whether this variable requires gradient computation
                 or not
   */
  NBLA_API CgVariable(VariablePtr var, bool need_grad);

  /** Get need grad flag.
   */
  inline bool need_grad() const { return need_grad_ == NG_TRUE; }

  /** Check if need grad flag is set.
   */
  inline bool need_grad_is_set() const { return need_grad_ != NG_NONE; }

  /** Set need grad flag.
   */
  inline void set_need_grad(bool b) { need_grad_ = b ? NG_TRUE : NG_FALSE; }

  /** Unset need grad flag.
   */
  inline void unset_need_grad() { need_grad_ = NG_NONE; }

  /** Get need grad state flag.
   */
  inline bool need_grad_state() const {
    return (need_grad_ != NG_NONE) ? (need_grad_ == NG_TRUE)
                                   : (need_grad_state_ == NG_TRUE);
  }
  /** Check if need grad state is set
   */
  inline bool need_grad_state_is_set() const {
    return need_grad_state_ != NG_NONE;
  }

  /** Set need grad state flag.
   */
  inline void set_need_grad_state(bool b) {
    need_grad_state_ = b ? NG_TRUE : NG_FALSE;
  }

  /** Unset need grad state flag.
   */
  inline void unset_need_grad_state() { need_grad_state_ = NG_NONE; }

  /** Set parent function.

      @param[in] func Function.

      @note Users usually don't use this directly. Used in connect function.
   */
  inline void set_parent(CgFunctionPtr func) { parent_ = func; }

  /** Get parent function which produces outputs to this variable.
   */
  inline CgFunctionPtr parent() { return parent_; }

  /** Get variable reference held in this instance.
   */
  inline VariablePtr variable() { return var_; }

  /** Set variable reference.
   */
  inline void set_variable(VariablePtr var) { var_ = var; }

  /** @copydoc rank_
   */
  inline int rank() const { return rank_; }

  /** set rank.

      @note Users shouldn't call this directly.
   */
  inline void set_rank_(int rank) { rank_ = rank; }

  /** Forward propagation from root inputs to this variable.

      The predecessor functions are executed in order of lower rank to higher
      rank until reaching this variable.

      @param[in] clear_buffer Clear SyncedArray object of a variable
                 never be used during the rest of forward propagation. This
                 option significantly saves the memory consumption. This is not
                 usually used in training phase because backward computation
                 requires data computed during forward prop.
      @param[in] clear_need_grad Clear the unreferenced variables with
                 need_grad=False during forward propagation.
                 True is usually used when calling this during training.
                 This is ignored when clear_buffer=True.

      @seealso set_persistent() to prevent a specific variable to be cleared
               during forward propagation.
   */
  NBLA_API void forward(bool clear_buffer = false,
                        bool clear_no_need_grad = false);
  /** Performs a backward propagation

      starting from this variable until the root variable(s) is/are reached
      in the computation graph.
      The propagation will stop at a variable with need_grad=false.
      Backward propagation through predecessors of this variable.

      @param[in] grad The backward error signal of this variable. if nullptr is
                 set, its gradients are set as 1.
      @param[in] clear_buffer Clears the no longer referenced variables
                 during backpropagation to save memory.
      @param     communicator_callbacks The callback functions invoked when 1)
                 backward computation of each function is finished and
                 2) all backward computation is finished.

      @seealso set_persistent() to prevent a specific variable to be cleared
               during forward propagation.
  */
  NBLA_API void
  backward(NdArrayPtr grad = nullptr, bool clear_buffer = false,
           vector<CommunicatorBackwardCallbackPtr> communicator_callbacks = {});

  /**
  */
  NBLA_API vector<CgFunctionPtr> function_references();

  /**
   */
  inline int function_reference_count() const {
    return function_references_.size();
  }

  /**
   */
  void insert_function_reference(CgFunctionPtr func);

  /**
   */
  NBLA_API
  void remove_function_reference(CgFunction *funcp);

  /** Mark need_setup flag for all function references.
   */
  void mark_need_setup();

  /** Check need_setup signal, and unmark it.
   */
  bool check_and_unmark_need_setup(CgFunctionPtr func);

  /** @copydoc allow_modify_data_
   */
  inline bool allow_modify_data() const { return allow_modify_data_; }

  /**
      @note User shouldn't call this directly.
   */
  inline void set_allow_modify_data(bool allow) { allow_modify_data_ = allow; }

  /** Set persistent flag.

      If it's true, the variable data and grad are never cleared during forward
      or backward propagation with clear options. It is useful for visualization
      and debugging purposes.

      @param[in] p Persistent flag.
   */
  inline void set_persistent(bool p) { persistent_ = p; }

  /** Get persistent flag.
   */
  inline bool persistent() const { return persistent_; }

  /** Set variable name
   */
  inline void set_name(string name) { name_ = name; }

  /** Get variable name
   */
  inline string name() const { return name_; }
};

/** shared_ptr typedef of CGVariable
 */
typedef CgVariable::Ptr CgVariablePtr;
}
#endif
