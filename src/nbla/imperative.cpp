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

#include <nbla/imperative.hpp>
#include <nbla/variable.hpp>

#include <memory>

namespace nbla {

using std::make_shared;

vector<NdArrayPtr> execute(FunctionPtr func, const vector<NdArrayPtr> &inputs,
                           int n_outputs, vector<NdArrayPtr> outputs) {
  // Check inplace outputs size.
  NBLA_CHECK(outputs.size() <= static_cast<unsigned>(n_outputs),
             error_code::value,
             "Num of in-place arrays must not be greater than n_outputs. "
             "In-place arrays: %d <= n_outputs: %d",
             outputs.size(), n_outputs);
  if (outputs.size() != static_cast<unsigned>(n_outputs)) {
    outputs.resize(n_outputs, nullptr);
  }

  // Copy if function is already used.
  if (func->ask_if_used_and_use()) {
    func = func->copy();
  }

  // Function inputs and outputs must be Variables.
  vector<VariablePtr> vinputs(inputs.size());
  vector<VariablePtr> voutputs(outputs.size());
  for (unsigned int i = 0; i < inputs.size(); ++i) {
    vinputs[i] = make_shared<Variable>(inputs[i]);
  }
  for (int i = 0; i < n_outputs; ++i) {
    voutputs[i] = make_shared<Variable>();
  }
  auto finputs = as_pointer_array(vinputs);
  auto foutputs = as_pointer_array(voutputs);

  // Setup function.
  func->setup(finputs, foutputs);

  // Set inplace buffer to function output buffer if size matches.
  for (unsigned int i = 0; i < outputs.size(); ++i) {
    if (!outputs[i]) {
      outputs[i] = foutputs[i]->data();
    }
    NBLA_CHECK(outputs[i]->size() == foutputs[i]->size(), error_code::value,
               "In-place array size and function output size must match. "
               "outputs[%d] size: %d, function output[%d] size: %d",
               i, outputs[i]->size(), i, foutputs[i]->size());
    foutputs[i]->data()->set_array(outputs[i]->array()); // Inplace.
  }

  // Execute Forward.
  func->forward(finputs, foutputs);
  return outputs;
}

void execute(FunctionPtr f, const Variables &inputs, const Variables &outputs) {
  f->setup(inputs, outputs);
  f->forward(inputs, outputs);
}

void backward(FunctionPtr f, const Variables &inputs, const Variables &outputs,
              const vector<bool> &propagate_down, const vector<bool> &accum,
              bool with_setup) {
  if (with_setup) {
    f->setup(inputs, outputs);
  }
  f->backward(inputs, outputs, propagate_down, accum);
}
}
