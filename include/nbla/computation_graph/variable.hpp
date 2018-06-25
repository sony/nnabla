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
#include <unordered_set>

namespace nbla {

using std::unordered_set;

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

  bool need_grad_;  ///< Whether this variable needs grad. This is going to be
                    /// updated when propagating forward .
  VariablePtr var_; /// Variable instance.
  CgFunctionPtr parent_{nullptr};   ///< Function created this variable.
  int rank_{0};                     ///< Longest path from root variable.
  int function_reference_count_{0}; ///< Reference count by child functions.
  bool allow_inplace_data_{true};   ///< Whether the data can be in-placed.
  bool grad_inplaced_{false}; ///< Gradient is in-placed with any of parent
                              /// function's inputs grad.
  bool persistent_{false};    ///<Persistency flag against clearing.

  /** set rank.

      @note Users shouldn't call this directly.
   */
  inline void set_rank(int rank) { rank_ = rank; }

  void
  visit_function_recursive(CgFunctionPtr func,
                           unordered_set<CgFunctionPtr> &fclosed,
                           std::function<void(CgFunctionPtr)> forward_callback);

  void visit_function_backward(
      CgFunctionPtr func, std::function<void(CgFunctionPtr)> backward_callback,
      vector<CommunicatorBackwardCallbackPtr> communicator_callbacks);

public:
  typedef shared_ptr<CgVariable> Ptr;

  /** Ctor wth need_grad option.

      The default shape is 0-shaped array (scalar).

      @param[in] need_grad Whether this variable requires gradient computation
                 or not
   */
  NBLA_API CgVariable(bool need_grad);
  /** Ctor wth variable shape and need_grad option.

      @param[in] shape Shape passed to Variable object held in the created
                 instance.
      @param[in] need_grad Whether this variable requires gradient computation
                 or not
   */
  NBLA_API CgVariable(Shape_t shape, bool need_grad);
  /** Ctor wth Variable object.

      @param[in] var Reference of an existing Variable object.
   */
  NBLA_API CgVariable(VariablePtr var);

  /** Get need grad flag.
   */
  inline bool need_grad() const { return need_grad_; }

  /** Set need grad flag.
   */
  inline void set_need_grad(bool b) { need_grad_ = b; }

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

  /** @copydoc rank_
   */
  inline int rank() const { return rank_; }

  /** Forward propagation from root iputs to this variable.

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

  /** @copydoc function_reference_count_
   */
  inline int function_reference_count() const {
    return function_reference_count_;
  }

  /** Increment function_reference_count_.

      @note User shouldn't call this directly.
   */
  inline void increment_function_reference_count() {
    function_reference_count_++;
  }

  /** @copydoc allow_inplace_data_
   */
  inline bool allow_inplace_data() const { return allow_inplace_data_; }

  /**
      @note User shouldn't call this directly.
   */
  inline void set_allow_inplace_data(bool allow) {
    allow_inplace_data_ = allow;
  }

  /** @copydoc grad_inplaced_
   */
  inline bool grad_inplaced() const { return grad_inplaced_; }

  /**
      @note User shouldn't call this directly.
   */
  inline void set_grad_inplaced(bool inplaced) { grad_inplaced_ = inplaced; }

  /** Set persistent flag.

      If it's true, the variable data and grad are never cleared during forward
      or backward propgation with clear options. It is useful for visualization
      and debugging purposes.

      @param[in] p Persistent flag.
   */
  inline void set_persistent(bool p) { persistent_ = p; }

  /** Get persistent flag.
   */
  inline bool persistent() const { return persistent_; }
};

/** shared_ptr typedef of CGVariable
 */
typedef CgVariable::Ptr CgVariablePtr;
}
#endif
