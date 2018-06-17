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

static vector<CgVariablePtr>
connect_core(CgFunctionPtr cg_f, const vector<CgVariablePtr> &inputs,
             const vector<CgVariablePtr> &outputs,
             vector<NdArrayPtr> inplace_outputs = {}, bool execute = false);

using std::make_shared;

vector<CgVariablePtr> create_function_outputs(CgFunctionPtr cg_f,
                                              int n_outputs) {
  // Check inplace outputs size and create outputs.
  if (n_outputs < 0) {
    n_outputs = cg_f->function()->min_outputs();
  }
  vector<CgVariablePtr> outputs(n_outputs);
  for (int i = 0; i < n_outputs; ++i) {
    auto v = make_shared<CgVariable>();
    v->set_need_grad_state(cg_f->need_grad());
    v->set_parent(cg_f);
    outputs[i] = v;
  }
  // Weak references are held inside.
  cg_f->set_outputs(outputs);
  // Return strong references.
  return outputs;
}

vector<CgVariablePtr> connect(CgFunctionPtr cg_f,
                              const vector<CgVariablePtr> &inputs,
                              int n_outputs, vector<NdArrayPtr> inplace_outputs,
                              bool execute) {
  cg_f->set_inputs(inputs);
  vector<CgVariablePtr> outputs = create_function_outputs(cg_f, n_outputs);
  return connect_core(cg_f, inputs, outputs, inplace_outputs, execute);
}

vector<CgVariablePtr> connect(CgFunctionPtr cg_f,
                              const vector<CgVariablePtr> &inputs,
                              const vector<CgVariablePtr> &outputs,
                              vector<NdArrayPtr> inplace_outputs,
                              bool execute) {
  cg_f->set_inputs(inputs);
  cg_f->set_outputs(outputs);
  return connect_core(cg_f, inputs, outputs, inplace_outputs, execute);
}

vector<CgVariablePtr> connect_core(CgFunctionPtr cg_f,
                                   const vector<CgVariablePtr> &inputs,
                                   const vector<CgVariablePtr> &outputs,
                                   vector<NdArrayPtr> inplace_outputs,
                                   bool execute) {
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

  cg_f->verify_during_forward();

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
