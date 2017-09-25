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

#include <nbla/computation_graph/computation_graph.hpp>

#include <memory>

namespace nbla {

using std::make_shared;

// Just a helper function.
static inline const char *b2str(bool b) { return b ? "true" : "false"; }

vector<CgVariablePtr> create_function_outputs(CgFunctionPtr cg_f,
                                              int n_outputs) {
  // Check inplace outputs size and create outputs.
  if (n_outputs < 0) {
    n_outputs = cg_f->function()->min_outputs();
  }
  vector<CgVariablePtr> outputs(n_outputs);
  for (int i = 0; i < n_outputs; ++i) {
    auto v = make_shared<CgVariable>(cg_f->need_grad());
    v->set_parent(cg_f);
    outputs[i] = v;
  }
  // Return strong references.
  return outputs;
}

vector<CgVariablePtr> connect(CgFunctionPtr cg_f,
                              const vector<CgVariablePtr> &inputs,
                              int n_outputs, vector<NdArrayPtr> inplace_outputs,
                              bool execute) {
  vector<CgVariablePtr> outputs = create_function_outputs(cg_f, n_outputs);
  return connect(cg_f, inputs, outputs, inplace_outputs, execute);
}

vector<CgVariablePtr> connect(CgFunctionPtr cg_f,
                              const vector<CgVariablePtr> &inputs,
                              const vector<CgVariablePtr> &outputs,
                              vector<NdArrayPtr> inplace_outputs,
                              bool execute) {
  cg_f->set_inputs(inputs);
  // Weak references are held inside.
  cg_f->set_outputs(outputs);

  // Function inputs and outputs must be Variables.
  vector<Variable *> finputs(inputs.size());
  vector<Variable *> foutputs(outputs.size());
  for (int i = 0; i < inputs.size(); ++i) {
    finputs[i] = inputs[i]->variable().get();
  }
  for (int i = 0; i < outputs.size(); ++i) {
    foutputs[i] = outputs[i]->variable().get();
  }

  // Setup function.
  auto f = cg_f->function();
  f->setup(finputs, foutputs);

  if (cg_f->need_grad()) {
    // Set inplace capability of output variables.
    for (int i = 0; i < inputs.size(); ++i) {
      if (!inputs[i]->variable()->need_grad())
        continue;
      if (f->inplace_grad(i)) {
        NBLA_CHECK(f->inplace_grad(i) < Function::INPLACE ||
                       inputs[i]->parent(),
                   error_code::value,
                   "A grad array of a root variable in a graph cannot be "
                   "in-placed with modification (%d-th input "
                   "of '%s').",
                   i, f->name().c_str());
        outputs[f->inplace_grad_with(i)]->set_grad_inplaced(true);
      }
      for (int o = 0; o < outputs.size(); ++o) {
        // If funcition gradient computation at i-th variable depends on o-th
        // output data, inplacing o-th variable data is prohibited.
        if (f->grad_depends_output_data(i, o)) {
          outputs[o]->set_allow_inplace_data(false);
        }
      }
    }
  }
  // Check if in-place is properly used.
  for (int i = 0; i < inputs.size(); ++i) {
    if (inputs[i]->allow_inplace_data())
      continue;
    const int inplace_level = f->inplace_data(i);
    if (inplace_level == Function::NOT_INPLACE)
      continue;
    NBLA_CHECK(inplace_level < Function::INPLACE, error_code::value,
               "In-place %d-th input data of '%s' (depth=%d) is "
               "prohibited by the parent function '%s'.",
               i, f->name().c_str(), cg_f->rank(),
               inputs[i]->parent()->function()->name().c_str());
    // Since the in-placed input's data is not modified in this function,
    // the allow-inplace flag is propagated to the output variable.
    outputs[f->inplace_data_with(i)]->set_allow_inplace_data(false);
  }

  // Check if branching doesn't appear in in-placed variables.
  for (int i = 0; i < inputs.size(); ++i) {
    bool inplace = f->inplace_data(i) || f->inplace_grad(i);
    NBLA_CHECK(
        (!inplace) || inputs[i]->function_reference_count() < 2,
        error_code::value,
        "Branching a variable is prohibited if it is in-placed. %d-th input "
        "of `%s` (depth=%d) is inplaced (data: %s, grad: %s).",
        i, f->name().c_str(), cg_f->rank(), b2str(f->inplace_data(i)),
        b2str(f->inplace_grad(i)));
  }

  // Set clear buffer flags
  for (int i = 0; i < inputs.size(); ++i) {
    if (f->inplace_data(i))
      outputs[f->inplace_data_with(i)]->set_clear_data_in_backward(false);
    if (f->inplace_grad(i))
      outputs[f->inplace_grad_with(i)]->set_clear_grad_in_backward(false);
  }

  // Set array reference to function output buffer if size matches.
  for (int i = 0; i < outputs.size(); ++i) {
    if (i >= inplace_outputs.size() || !inplace_outputs[i])
      continue;
    NBLA_CHECK(inplace_outputs[i]->size() == foutputs[i]->size(),
               error_code::value,
               "In-place array size and function output size must match. "
               "inplace_outouts[%d]: %d != function_output[%d]: %d.",
               inplace_outputs[i]->size(), foutputs[i]->size());
    foutputs[i]->data()->set_array(inplace_outputs[i]->array());
  }

  if (execute) {
    // Execute Forward.
    cg_f->function()->forward(finputs, foutputs);
  }
  return outputs;
}
}
