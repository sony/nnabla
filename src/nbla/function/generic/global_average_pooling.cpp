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


#include <nbla/array.hpp>
#include <nbla/common.hpp>
#include <nbla/function/global_average_pooling.hpp>
#include <nbla/variable.hpp>

// TODO: remove the following headers if not used.
#include <iostream>
#include <typeinfo>



namespace nbla {

NBLA_REGISTER_FUNCTION_SOURCE(GlobalAveragePooling);

template <typename T>
void GlobalAveragePooling<T>::setup_impl(const Variables &inputs,
                               const Variables &outputs) {
  // TODO: Remove debug message
  std::cout << "GlobalAveragePooling<" << typeid(T).name()
            << ">::setup_impl called with " << this->ctx_.to_string() << "."
            << std::endl;
  // TODO: reshape outputs
  // outputs[0]->reshape({}, true);

  /* TODO: Any preparation comes here.
     Note that, although it is called only when a compuation graph is
     constructed in a static computation graph, in a dynamic computation graph,
     it's called every time. Keep the setup computation light for the performance
     (caching heavy computation, device synchronization in GPU etc.)
  */
}

template <typename T>
void GlobalAveragePooling<T>::forward_impl(const Variables &inputs,
                               const Variables &outputs) {
  // TODO: Remove debug message
  std::cout << "GlobalAveragePooling<" << typeid(T).name()
            << ">::forward_impl called with " << this->ctx_.to_string() << "."
            << std::endl;

  /* TODO: remove this help message.
    The type `Variables` is a typedef of `vector<Variable*>`.
    The `Variable` class owns storages of data (storage for forward propagation)
    and grad (for backprop) respectively.
    
    You can get a raw pointer of a scalar type of the storage using:

    - `cosnt T* Variable::get_{data|grad}_pointer<T>(ctx)` for read-only access.
    - `T* Variable::cast_{data|grad}_and_get_pointer<T>(ctx)` for r/w access.

    By this, automatic type conversion would occur if data was held in a different type.
  */
  // Inputs
  const T* x = inputs[0]->get_data_pointer<T>(this->ctx_);

  // Outputs
  T* y = outputs[0]->cast_data_and_get_pointer<T>(this->ctx_);

  // TODO: Write implementation
}


template <typename T>
void GlobalAveragePooling<T>::backward_impl(const Variables &inputs,
                                const Variables &outputs,
                                const vector<bool> &propagate_down,
                                const vector<bool> &accum) {
  // TODO: Remove debug message
  std::cout << "GlobalAveragePooling<" << typeid(T).name()
            << ">::backward_impl called with " << this->ctx_.to_string() << "."
            << std::endl;

  /* TODO: remove this help message.
     The propagate down flags are automatically set by our graph engine, which
     specifies whether each input variable of them requires gradient
     computation. 
  */
  if (!(propagate_down[0])) {
    return;
  }

  /** TODO: remove this help message.
      The backward error signals are propagated through the graph, and the
      error from decsendant functions are set in the grad region of the output variables.
   */
  // Gradient of outputs
  const T* g_y = outputs[0]->get_grad_pointer<T>(this->ctx_);

  /* TODO: remove this help message.
     The backward error signal should be propagated to the grad region of input
     variables.

     The accum flags are also set by our graph engine, which specifies whether
     each input variable of them wants the result of the gradient computation
     by substitution or accumulation.
  */
  // Gradient of inputs
  T* g_x{nullptr};

  if (propagate_down[0]) {
    g_x = inputs[0]->cast_grad_and_get_pointer<T>(this->ctx_);
    // TODO: Write gradient computation of x
  }
}
}
