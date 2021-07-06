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

typedef std::function<void(const CgFunctionPtr &ptr)> function_hook_type;

/** Callback helper class for function callbacks during forward and backward
class.

This is used from Python frontend.
 */
class FunctionHookWithObject {
public:
  typedef std::function<void(void *)> setup_callback_type;
  typedef std::function<void(void *)> cleanup_callback_type;
  typedef std::function<void(void *, const CgFunctionPtr &f)> callback_type;

private:
  void *obj_{nullptr};
  callback_type callback_;
  setup_callback_type setup_callback_;
  cleanup_callback_type cleanup_callback_;

public:
  NBLA_API FunctionHookWithObject();
  NBLA_API FunctionHookWithObject(const FunctionHookWithObject &from);
  NBLA_API FunctionHookWithObject(void *obj, callback_type cb,
                                  setup_callback_type setup_cb,
                                  cleanup_callback_type clean_cb);
  NBLA_API ~FunctionHookWithObject();
  NBLA_API FunctionHookWithObject &operator=(const FunctionHookWithObject &rhs);

  NBLA_API void operator()(const CgFunctionPtr &f);
};

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
    size_t count{0};
  };
  NeedGrad need_grad_{NG_NONE}; ///< Whether the variable requires gradients.
  NeedGrad need_grad_state_{
      NG_NONE};           ///< Updated during graph construction or forward
                          /// propagation.
  bool recompute_{false}; ///< Whether the data is cleared during forward
                          /// propagation and recomputation is performed during
  /// backward propagation.
  VariablePtr var_;               /// Variable instance.
  CgFunctionPtr parent_{nullptr}; ///< Function created this variable.
  int rank_{0};                   ///< Longest path from root variable.
  ///< Holds weak function references. <https://stackoverflow.com/a/22110715>
  unordered_map<CgFunction *,
                pair<std::weak_ptr<CgFunction>, FunctionReferenceInfo>>
      function_references_;
  size_t function_reference_count_{0}; ///< Number of function references
  bool allow_modify_data_{true};       ///< Whether the data can be in-placed.
  bool persistent_{false};             ///<Persistency flag against clearing.
  bool prohibit_clear_data_{false};
  string name_{""};

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

  /** Get recompute flag.
   */
  inline bool recompute() const { return recompute_; }

  /** Set recompute flag.
   */
  inline void set_recompute(bool b) { recompute_ = b; }

  /** Get prohibit_clear_data_ flag.
   */
  inline bool prohibit_clear_data() { return prohibit_clear_data_; }

  /** Set prohibit_clear_data_ flag.
   */
  inline void set_prohibit_clear_data(bool b) { prohibit_clear_data_ = b; }

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
      @param[in] fclosed Set arbitrary fclosed flags to control forward
                 computation. This is used for forward_all function.

      @seealso set_persistent() to prevent a specific variable to be cleared
               during forward propagation.
   */
  NBLA_API void forward(bool clear_buffer = false,
                        bool clear_no_need_grad = false,
                        unordered_set<CgFunctionPtr> *fclosed = nullptr,
                        function_hook_type pre_callback = nullptr,
                        function_hook_type post_callback = nullptr);
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
      @param     clear_initial_grad If true, the input parameter, grad, will be
                 cleared during backward propagation. This flag is only
                 activated when grad is set.

      @seealso set_persistent() to prevent a specific variable to be cleared
               during forward propagation.
  */
  NBLA_API void
  backward(NdArrayPtr grad = nullptr, bool clear_buffer = false,
           vector<CommunicatorBackwardCallbackPtr> communicator_callbacks = {},
           function_hook_type pre_callback = nullptr,
           function_hook_type post_callback = nullptr,
           const bool clear_initial_grad = false);

  /**
  */
  NBLA_API vector<CgFunctionPtr> function_references();

  /**
   */
  size_t function_reference_count() const;

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

  /** Deepcopy method
   */
  NBLA_API
  Ptr create_deep_copy(Context ctx, bool copy_grad = true);

  /** Execute callback at functions in forward order in a graph.
   */
  void
  visit_function_recursive(CgFunctionPtr func,
                           unordered_set<CgFunctionPtr> &fclosed,
                           const bool recomputation,
                           std::function<void(CgFunctionPtr)> forward_callback);

  /** Execute callback at functions in backward order in a graph.
   */
  void visit_function_backward(
      CgFunctionPtr func, std::function<void(CgFunctionPtr)> backward_callback,
      vector<CommunicatorBackwardCallbackPtr> communicator_callbacks);
};

/** shared_ptr typedef of CGVariable
 */
typedef CgVariable::Ptr CgVariablePtr;

class SingletonManager; // Forward declaration for friend

/** ClearCalledFlagRecorder is a singleton class to record and collect
 * the SyncedArray::clear_called flags during forward propagation.
*/
class ClearCalledFlagRecorder {

  bool is_activated_{false};

  std::vector<std::vector<std::pair<bool, bool>>> recorded_input_clear_flags_;
  std::vector<std::vector<std::pair<bool, bool>>> recorded_output_clear_flags_;

public:
  ~ClearCalledFlagRecorder();

  /** Check if this recorder is activated. */
  bool is_activated();

  /** Activate recording clear flags. */
  void activate();

  /** Deactivate recording clear flags and delete recorded flags. */
  void deactivate();

  /** Record clear flags from given function. */
  void record(const CgFunctionPtr func);

  /** Get recorded clear flags. */
  std::vector<std::vector<std::pair<bool, bool>>>
  get_recorded_input_clear_flags() const;

  /** Get recorded clear flags. */
  std::vector<std::vector<std::pair<bool, bool>>>
  get_recorded_output_clear_flags() const;

private:
  friend SingletonManager; // needs forward declaration
                           // Never called by users.
  ClearCalledFlagRecorder();

  std::vector<std::pair<bool, bool>>
  get_variable_clear_called_flag(const std::vector<CgVariablePtr> &vars);

  DISABLE_COPY_AND_ASSIGN(ClearCalledFlagRecorder);
};

NBLA_API void c_activate_clear_called_flag_recorder();

NBLA_API void c_deactivate_clear_called_flag_recorder();

NBLA_API std::vector<std::vector<std::pair<bool, bool>>>
c_get_input_clear_called_flags();

NBLA_API std::vector<std::vector<std::pair<bool, bool>>>
c_get_output_clear_called_flags();
}
#endif
