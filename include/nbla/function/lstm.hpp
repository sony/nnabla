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

#ifndef NBLA_FUNCTION_LSTM_HPP
#define NBLA_FUNCTION_LSTM_HPP

#include <nbla/cpu.hpp>
#include <nbla/function.hpp>
#include <nbla/function_registry.hpp>

namespace nbla {

NBLA_REGISTER_FUNCTION_HEADER(LSTM, int, float, bool, bool);

/**
    @todo Write doc.

Inputs:

Outputs:

\ingroup FunctionImplGrp
 */
template <typename T> class LSTM : public BaseFunction<int, float, bool, bool> {
protected:
  int num_layers_;
  float dropout_;
  bool bidirectional_;
  bool training_;

public:
  LSTM(const Context &ctx, int num_layers, float dropout, bool bidirectional,
       bool training)
      : BaseFunction(ctx, num_layers, dropout, bidirectional, training),
        num_layers_(num_layers), dropout_(dropout),
        bidirectional_(bidirectional), training_(training) {}
  virtual ~LSTM() {}
  virtual shared_ptr<Function> copy() const {
    return create_LSTM(ctx_, num_layers_, dropout_, bidirectional_, training_);
  }
  virtual int min_inputs() { return 4; }
  virtual int min_outputs() { return 3; }
  virtual vector<dtypes> in_types() {
    return vector<dtypes>{get_dtype<T>(), get_dtype<T>(), get_dtype<T>(),
                          get_dtype<T>(), get_dtype<T>(), get_dtype<T>()};
  }
  virtual vector<dtypes> out_types() {
    return vector<dtypes>{get_dtype<T>(), get_dtype<T>(), get_dtype<T>()};
  }
  virtual vector<string> allowed_array_classes() {
    return SingletonManager::get<Cpu>()->array_classes();
  }
  virtual string name() { return "LSTM"; }

protected:
  NBLA_API virtual void setup_impl(const Variables &inputs,
                                   const Variables &outputs);
  NBLA_API virtual void forward_impl(const Variables &inputs,
                                     const Variables &outputs);
  NBLA_API virtual void forward_impl_training(const Variables &inputs,
                                              const Variables &outputs);
  NBLA_API virtual void forward_impl_inference(const Variables &inputs,
                                               const Variables &outputs);
  NBLA_API virtual void backward_impl(const Variables &inputs,
                                      const Variables &outputs,
                                      const vector<bool> &propagate_down,
                                      const vector<bool> &accum);
};
}
#endif
