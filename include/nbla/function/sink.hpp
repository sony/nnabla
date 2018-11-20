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

/** Sink
 */
#ifndef __NBLA_FUNCTION_SINK_HPP__
#define __NBLA_FUNCTION_SINK_HPP__

#include <nbla/cpu.hpp>
#include <nbla/function.hpp>
#include <nbla/function_registry.hpp>

namespace nbla {

NBLA_REGISTER_FUNCTION_HEADER(Sink, bool);

/**
    @todo PLACE HERE FUNCTION DOCUMENTATION.
 */
template <typename T> class Sink : public BaseFunction<bool> {
protected:
  bool one_input_grad_;

public:
  Sink(const Context &ctx, bool one_input_grad)
      : BaseFunction<bool>(ctx, one_input_grad),
        one_input_grad_(one_input_grad) {}
  virtual ~Sink() {}
  virtual shared_ptr<Function> copy() const {
    return create_Sink(ctx_, one_input_grad_);
  }
  virtual vector<dtypes> in_types() { return vector<dtypes>{get_dtype<T>()}; }
  virtual vector<dtypes> out_types() { return vector<dtypes>{get_dtype<T>()}; }
  virtual int min_inputs() { return 1; }
  virtual int min_outputs() { return 1; }
  virtual string name() { return "Sink"; }
  virtual vector<string> allowed_array_classes() {
    return SingletonManager::get<Cpu>()->array_classes();
  }
  // NOTE: This avoid clearing input buffers by the graph engine during
  // execution.
  virtual bool prohibit_clear_input_buffers() const { return true; }
  // This avoids zero-ing grad buffers if one_input_grad_ is false
  // (i.e., use external gradients).
  virtual bool prohibit_zero_input_grad() const { return true; }

protected:
  NBLA_API virtual void setup_impl(const Variables &inputs,
                                   const Variables &outputs);
  NBLA_API virtual void forward_impl(const Variables &inputs,
                                     const Variables &outputs);
  NBLA_API virtual void backward_impl(const Variables &inputs,
                                      const Variables &outputs,
                                      const vector<bool> &propagate_down,
                                      const vector<bool> &accum);
};
}
#endif
