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

#include <nbla/variable.hpp>

namespace nbla {

// Forward declaration
class CgFunction;
typedef shared_ptr<CgFunction> CgFunctionPtr;

/**
*/
class CgVariable {
  VariablePtr var_;
  CgFunctionPtr parent_{nullptr};
  int rank_{0};
  int function_reference_count_{0}; ///< Reference count by child functions.
  int consume_counter_; ///< Consumption counter for clearing variable buffer in
                        /// forward or backward computation.
  bool allow_inplace_data_{true}; ///< Whether the data can be in-placed.
  bool grad_inplaced_{false};     ///< Gradient is in-placed with any of parent
                                  /// function's inputs grad.
  /** Data in-placed with any of parents. Used to decide whether to clear
      intermediate buffers during backward computation.
   */
  bool clear_data_in_backward_{true};
  /** Grad in-placed with any of parents. Used to decide whether to clear
      intermediate buffers during backward computation.
  */
  bool clear_grad_in_backward_{true};
  bool persistent_{false}; ///<Persistency flag against clearing.

public:
  typedef shared_ptr<CgVariable> Ptr;
  NBLA_API CgVariable(bool need_grad);
  NBLA_API CgVariable(Shape_t shape, bool need_grad);
  NBLA_API CgVariable(VariablePtr var);
  inline void set_parent(CgFunctionPtr func) { parent_ = func; }
  inline CgFunctionPtr parent() { return parent_; }
  inline VariablePtr variable() { return var_; }

  /**
   */
  inline int rank() const { return rank_; }

  /**
   */
  inline void set_rank(int rank) { rank_ = rank; }

  /**
   */
  NBLA_API void forward(bool clear_buffer = false,
                        bool clear_no_need_grad = false);
  /** Backward propagation through predecessors of this variable.

      @param[in] grad The backward error signal of this variable. if nullptr is
     set, it gradients are set as 1.
     @param[in]
   */
  NBLA_API void backward(NdArrayPtr grad = nullptr, bool clear_buffer = false);

  /**
   */
  inline int function_reference_count() const {
    return function_reference_count_;
  }

  /**
   */
  inline void increment_function_reference_count() {
    function_reference_count_++;
  }

  /**
   */
  inline bool allow_inplace_data() const { return allow_inplace_data_; }

  /**
   */
  inline void set_allow_inplace_data(bool allow) {
    allow_inplace_data_ = allow;
  }

  /**
   */
  inline bool grad_inplaced() const { return grad_inplaced_; }

  /**
   */
  inline void set_grad_inplaced(bool inplaced) { grad_inplaced_ = inplaced; }

  /**
   */
  inline bool clear_data_in_backward() const { return clear_data_in_backward_; }

  /**
   */
  inline void set_clear_data_in_backward(bool clear) {
    clear_data_in_backward_ = clear;
  }
  /**
   */
  inline bool clear_grad_in_backward() const { return clear_grad_in_backward_; }

  /**
   */
  inline void set_clear_grad_in_backward(bool clear) {
    clear_grad_in_backward_ = clear;
  }

  /**
   */
  inline int consume(bool reset = false) {
    if (reset)
      consume_counter_ = 1;
    else
      consume_counter_++;
    return consume_counter_;
  }

  /**
   */
  inline void set_persistent(bool p) { persistent_ = p; }

  /**
   */
  inline bool persistent() const { return persistent_; }
};

typedef CgVariable::Ptr CgVariablePtr;
}
#endif
