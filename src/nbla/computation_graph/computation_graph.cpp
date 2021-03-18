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
#include <nbla/singleton_manager-internal.hpp>

#include <algorithm>
#include <memory>

namespace nbla {

using std::make_shared;

static void set_function_inputs(CgFunctionPtr func,
                                const vector<CgVariablePtr> &inputs) {
  // Check need_grad
  bool need_grad = false;
  int rank = 0;
  for (auto i : inputs) {
    need_grad |= i->need_grad_state();
    rank = std::max(rank, i->rank());
    i->insert_function_reference(func);
  }
  func->set_need_grad(need_grad);
  func->set_rank_(rank);
  func->set_inputs_(inputs);
}

vector<CgVariablePtr> create_function_outputs(CgFunctionPtr cg_f, int n_outputs,
                                              bool prohibit_clear_output) {
  // Check inplace outputs size and create outputs.
  if (n_outputs < 0) {
    n_outputs = cg_f->function()->min_outputs();
  }
  vector<CgVariablePtr> outputs(n_outputs);
  for (int i = 0; i < n_outputs; ++i) {
    auto v = make_shared<CgVariable>();
    v->set_need_grad_state(cg_f->need_grad());
    v->set_parent(cg_f);
    v->set_prohibit_clear_data(prohibit_clear_output);
    outputs[i] = v;
  }
  // Weak references are held inside.
  cg_f->set_outputs(outputs);
  // Return strong references.
  return outputs;
}

vector<CgVariablePtr> connect(CgFunctionPtr cg_f,
                              const vector<CgVariablePtr> &inputs_orig,
                              int n_outputs, vector<NdArrayPtr> inplace_outputs,
                              bool execute) {
  // Filter null inputs (since cpp functions doesn't handle optional inputs in
  // functions.cpp)
  vector<CgVariablePtr> inputs;
  for (auto &inp : inputs_orig) {
    if (inp) {
      inputs.push_back(inp);
    }
  }

  set_function_inputs(cg_f, inputs);

  // check if data can be cleared or not.
  bool persistent = false, inplace = false, prohibit_clear = false;
  for (vector<CgVariablePtr>::size_type i = 0; i < inputs.size(); ++i) {
    auto inp = inputs[i];
    persistent |= inp->rank() == 0 || inp->persistent();
    prohibit_clear |= inp->prohibit_clear_data();
    inplace |= cg_f->function()->inplace_data(i) > 0;
  }

  bool prohibit_output_clear = (prohibit_clear || persistent) && inplace;

  vector<CgVariablePtr> outputs =
      create_function_outputs(cg_f, n_outputs, prohibit_output_clear);

  // Setup function.
  cg_f->setup();

  // Verify connections.
  cg_f->verify_during_forward();

  // Function inputs and outputs must be Variables.
  vector<Variable *> finputs(inputs.size());
  vector<Variable *> foutputs(outputs.size());
  for (vector<CgVariablePtr>::size_type i = 0; i < inputs.size(); ++i) {
    finputs[i] = inputs[i]->variable().get();
  }
  for (vector<CgVariablePtr>::size_type i = 0; i < outputs.size(); ++i) {
    foutputs[i] = outputs[i]->variable().get();
  }

  // Set array reference to function output buffer if size matches.
  for (vector<CgVariablePtr>::size_type i = 0; i < outputs.size(); ++i) {
    if (i >= inplace_outputs.size() || !inplace_outputs[i])
      continue;
    NBLA_CHECK(inplace_outputs[i]->size() == foutputs[i]->size(),
               error_code::value,
               "In-place array size and function output size must match. "
               "inplace_outputs[%d]: %d != function_output[%d]: %d.",
               inplace_outputs[i]->size(), foutputs[i]->size());
    foutputs[i]->data()->set_array(inplace_outputs[i]->array());
  }

  if (execute) {
    // Execute Forward.
    cg_f->function()->forward(finputs, foutputs);
  }
  return outputs;
}

void steal_variable_from_to(CgVariablePtr from, CgVariablePtr to) {
  // A. The shape must the same
  NBLA_CHECK(
      to->variable()->shape() == from->variable()->shape(), error_code::value,
      "Variable shapes of from and to must match. from != to : (%s) != (%s).",
      string_join(from->variable()->shape(), ", ").c_str(),
      string_join(to->variable()->shape(), ", ").c_str());
  // B. Get a parent function of from
  auto parent = from->parent();
  NBLA_CHECK(parent != nullptr, error_code::value,
             "The 1st argument CgVariablePtr must have a parent function (must "
             "be an output of a function.");

  // C. Forget parent of from and rewire the parent function to to variable.
  from->set_parent(nullptr);
  to->set_parent(parent);

  // D. Replace an output variable reference of the parent function with to
  // variable.
  auto outputs = parent->outputs();
  std::replace(outputs.begin(), outputs.end(), from, to);
  parent->set_outputs(outputs);

  // E. Copy flags.
  to->set_allow_modify_data(from->allow_modify_data());
  if (from->need_grad_is_set()) {
    to->set_need_grad(from->need_grad());
  } else {
    to->unset_need_grad();
  }
  to->set_prohibit_clear_data(from->prohibit_clear_data());

  // F. Reference contents
  to->set_variable(from->variable());

  // G. Set setup flag.
  to->mark_need_setup();
}

void forward_all(const vector<CgVariablePtr> variables, bool clear_buffer,
                 bool clear_no_need_grad, function_hook_type function_pre_hook,
                 function_hook_type function_post_hook) {

  // Set persistent for all variables to keep the data.
  vector<bool> orig_persistent_flags;
  for (auto &e : variables) {
    orig_persistent_flags.push_back(e->persistent());
    e->set_persistent(true);
  }

  // Revert persistent flags after the end of forward_all
  DestructorCallback persistent_flag_restorer([&]() -> void {
    for (vector<CgVariablePtr>::size_type i = 0; i < variables.size(); ++i) {
      variables[i]->set_persistent(orig_persistent_flags[i]);
    }
  });

  unordered_set<CgFunctionPtr> fclosed;
  for (vector<CgVariablePtr>::size_type i = 0; i < variables.size(); ++i) {
    variables[i]->forward(clear_buffer, clear_no_need_grad, &fclosed,
                          function_pre_hook, function_post_hook);
  }
}

GlobalClearBufferState::GlobalClearBufferState() {}
bool GlobalClearBufferState::clear_buffer() const {
  std::lock_guard<decltype(mtx_)> lock(mtx_);
  auto tid = std::this_thread::get_id();
  return clear_buffer_[tid];
}
bool GlobalClearBufferState::clear_no_need_grad() const {
  std::lock_guard<decltype(mtx_)> lock(mtx_);
  auto tid = std::this_thread::get_id();
  return clear_no_need_grad_[tid];
}
void GlobalClearBufferState::set(bool clear_buffer, bool clear_no_need_grad) {
  std::lock_guard<std::mutex> lock(mtx_);
  auto tid = std::this_thread::get_id();
  clear_buffer_[tid] = clear_buffer;
  clear_no_need_grad_[tid] = clear_no_need_grad;
}
std::unique_ptr<GlobalClearBufferState::ScopedState>
GlobalClearBufferState::state(bool clear_buffer, bool clear_no_need_grad) {
  return std::unique_ptr<GlobalClearBufferState::ScopedState>(
      new GlobalClearBufferState::ScopedState(this, clear_buffer,
                                              clear_no_need_grad));
}
GlobalClearBufferState::ScopedState::ScopedState(GlobalClearBufferState *self,
                                                 bool clear_buffer,
                                                 bool clear_no_need_grad)
    : self_(self), clear_buffer_(self->clear_buffer()),
      clear_no_need_grad_(self->clear_no_need_grad()) {
  self->set(clear_buffer, clear_no_need_grad);
}
GlobalClearBufferState::ScopedState::~ScopedState() {
  self_->set(clear_buffer_, clear_no_need_grad_);
}

NBLA_INSTANTIATE_SINGLETON(NBLA_API, GlobalClearBufferState);
}
