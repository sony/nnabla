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
#include <nbla/computation_graph/function.hpp>
#include <nbla/computation_graph/variable.hpp>

#include <cstdint>
#include <iostream>
#include <memory>
#include <set>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <vector>

// #define NNABLA_DUMP_ON_OUTPUTS_FORWARD
// #define NNABLA_DUMP_ON_INPUTS_FORWARD
#if defined(NNABLA_DUMP_ON_OUTPUTS_FORWARD) ||                                 \
    defined(NNBLA_DUMP_INPUTS_FORWARD)
#include <iostream>
#endif

namespace nbla {

using std::make_shared;
using std::set;
using std::unordered_map;
using std::unordered_set;
using std::tuple;
using std::make_tuple;
using std::get;
using std::unique_ptr;
using std::vector;

FunctionHookWithObject::FunctionHookWithObject() {}
FunctionHookWithObject::FunctionHookWithObject(
    void *obj, FunctionHookWithObject::callback_type cb,
    FunctionHookWithObject::cleanup_callback_type cleanup_cb)
    : obj_(obj), callback_(cb), cleanup_callback_(cleanup_cb) {}

FunctionHookWithObject::~FunctionHookWithObject() { cleanup_callback_(obj_); }
void FunctionHookWithObject::operator()(const CgFunctionPtr &f) {
  callback_(obj_, f);
}

CgVariable::CgVariable() { var_ = make_shared<Variable>(Shape_t{}); }
CgVariable::CgVariable(bool need_grad) : CgVariable() {
  set_need_grad(need_grad);
}

CgVariable::CgVariable(Shape_t shape) { var_ = make_shared<Variable>(shape); }
CgVariable::CgVariable(Shape_t shape, bool need_grad) : CgVariable(shape) {
  set_need_grad(need_grad);
}

CgVariable::CgVariable(VariablePtr var) { var_ = var; }
CgVariable::CgVariable(VariablePtr var, bool need_grad) : CgVariable(var) {
  set_need_grad(need_grad);
}

class ForwardCallback {
  bool clear_buffer_{false};
  bool clear_no_need_grad_{false};
  function_hook_type function_pre_hook_;
  function_hook_type function_post_hook_;
  unordered_map<CgVariablePtr, int> vseen_;
  unordered_set<CgVariablePtr> need_grad_variable_set_;
  vector<string> history_;

public:
  ForwardCallback(bool clear_buffer, bool clear_no_need_grad,
                  function_hook_type function_pre_hook,
                  function_hook_type function_post_hook)
      : clear_buffer_(clear_buffer), clear_no_need_grad_(clear_no_need_grad),
        function_pre_hook_(function_pre_hook),
        function_post_hook_(function_post_hook) {}

  bool check_last_visit(CgVariablePtr v) {
    if (v->function_reference_count() < 2) {
      // A variable referenced by <2 is always visited last.
      return true;
    }
    // Find from previous search history.
    auto it = vseen_.find(v);
    if (it == vseen_.end()) {
      // The first visit.
      vseen_.insert({v, 1});
      return false;
    }
    // Check
    if (++(it->second) == v->function_reference_count()) {
      // For better search performance of another. (maybe not required)
      vseen_.erase(it);
      return true;
    }
    return false;
  }

  vector<bool> get_clear_flags(CgFunctionPtr func) {
    auto inputs = func->inputs();
    vector<bool> ret(inputs.size(), false);
    for (int i = 0; i < inputs.size(); ++i) {
      auto vi = inputs[i];
      // Remember variables that should not be cleared during forward
      if (func->need_grad() && !vi->need_grad()) {
        // Any variable that is used to computate gradient shouldn't be cleared.
        // TODO: Not optimal because the input may not be used in gradient
        // computation of some function.
        need_grad_variable_set_.insert(vi);
      }
      // This comes first because check_last_visit must be called in order to
      // increment the visit count of vi.
      if (!check_last_visit(vi)) {
        continue;
      }
      if (func->function()->prohibit_clear_input_buffers()) {
        continue;
      }
      if (vi->rank() == 0 || vi->persistent() ||
          func->function()->inplace_data(i) || vi->prohibit_clear_data()) {
        continue;
      }
      if (clear_buffer_) {
        ret[i] = true;
        continue;
      }
      if (clear_no_need_grad_ && !func->need_grad()) {
        // Not clear if any function requiring gradient computation uses this
        // variable.
        if (need_grad_variable_set_.find(vi) != need_grad_variable_set_.end()) {
          continue;
        }
        ret[i] = true;
        continue;
      }
    }
    return ret;
  }

  void clear_inputs(const vector<CgVariablePtr> &inputs,
                    const vector<bool> &clear_flags) {
    // std::cout << "Clear flags: " << string_join(clear_flags, ",") <<
    // std::endl;
    for (int i = 0; i < inputs.size(); i++) {
      if (clear_flags[i]) {
        inputs[i]->variable()->data()->array()->clear();
      }
    }
  }

  void on_outputs(CgFunctionPtr func) {
#if defined(NNABLA_DUMP_ON_OUTPUTS_FORWARD)
    Context ctx{{}, "CpuCachedArray", ""};
    auto outputs = func->outputs();
    std::cout << "[" << func->function()->name() << "]" << std::endl;
    for (int i = 0; i < outputs.size(); i++) {
      const float *v = outputs[i]->variable()->get_data_pointer<float>(ctx);
      std::cout << "output " << i << ":";
      for (int s = 0; s < outputs[i]->variable()->size(); s++) {
        std::cout << " " << v[s];
      }
      std::cout << std::endl;
    }
#endif
  }

  void on_inputs(CgFunctionPtr func) {
#if defined(NNABLA_DUMP_ON_INPUTS_FORWARD)
    Context ctx{{}, "CpuCachedArray", ""};
    auto inputs = func->inputs();
    std::cout << "[" << func->function()->name() << "]" << std::endl;
    for (int i = 0; i < inputs.size(); i++) {
      const float *v = inputs[i]->variable()->get_data_pointer<float>(ctx);
      std::cout << "input " << i << ":";
      for (int s = 0; s < inputs[i]->variable()->size(); s++) {
        std::cout << " " << v[s];
      }
      std::cout << std::endl;
    }
#endif
  }

  void error_trace(const string &name_on_error) {
    // TODO: Optional verbosity
    std::cerr << "Error during forward propagation:" << std::endl;
    for (auto &name : history_) {
      std::cerr << "  " << name << std::endl;
    }
    std::cerr << "  " << name_on_error << " <-- ERROR" << std::endl;
  }

  void operator()(CgFunctionPtr func) {
    // Execute forward.
    // std::cout << "Call forward of " << func->function()->name() << "."
    //           << std::endl;
    this->on_inputs(func);

    vector<CgVariablePtr> outputs; // Get shared reference of outputs.
    vector<Variable *> voutputs;
    std::tie(outputs, voutputs) = func->function_outputs();
    try {
      auto call_callback = [func](function_hook_type &h) {
        if (h) {
          h(func);
        }
      };
      call_callback(function_pre_hook_);
      func->function()->forward(func->function_inputs(), voutputs);
      call_callback(function_post_hook_);
    } catch (...) {
      error_trace(func->function()->name());
      throw;
    }
    history_.push_back(func->function()->name());

    this->on_outputs(func);

    // Clear input buffers where possible.
    auto clear_flags = get_clear_flags(func);
    clear_inputs(func->inputs(), clear_flags);
  }
};

class BackwardCallback {
  bool clear_buffer_;
  function_hook_type function_pre_hook_;
  function_hook_type function_post_hook_;
  // Visit CgVaiable list. The value is whether this is cleared during backward.
  unordered_map<CgVariablePtr, bool> vseen_;
  vector<string> history_;

  vector<bool> get_accum(const vector<CgVariablePtr> &inputs,
                         const vector<bool> &first_visit_flags) {
    vector<bool> accum(inputs.size(), false);
    for (int i = 0; i < inputs.size(); i++) {
      // No need grad.
      if (!inputs[i]->need_grad_state())
        continue;

      // If memset with 0 is reserved, accum is not used. For shared case, the
      // first is only non-accum.
      auto array = inputs[i]->variable()->grad()->array();
      if (array->zeroing()) {
        bool input_shared = false;
        for (int j = 0; j < inputs.size(); j++) {
          if (i == j) {
            continue;
          }
          if (inputs[j]->variable()->grad()->array() == array) {
            input_shared = true;
            break;
          }
        }
        if (!input_shared) {
          continue;
        }
      }
      // First visit gradients in intermediate layers are copied.
      if (inputs[i]->parent() && first_visit_flags[i]) {
        continue;
      }
      accum[i] = true;
    }
    return accum;
  }

  void force_zero_grad_if_unseen(vector<CgVariablePtr> outputs,
                                 const vector<bool> &first_visit) {
    for (int i = 0; i < outputs.size(); i++) {
      auto o = outputs[i];
      if (first_visit[i]) {
        // The output variable has not been seen during this backprop, which
        // means no one sets the gradient previously. To prevent to propagate
        // uninitialized gradient, the output gradients are filled as 0.
        // std::cout << "Zero-ing output grad of "
        //           << o->parent()->function()->name() << std::endl;
        o->variable()->grad()->zero();
      }
    }
  }

  void clear_output_buffers(CgFunctionPtr func,
                            const vector<bool> &prohibit_clear) {
    if (clear_buffer_) {
      auto f = func->function();
      auto inputs = func->inputs();
      auto outputs = func->outputs();
      vector<pair<bool, bool>> clear(outputs.size(), {true, true});
      for (int i = 0; i < inputs.size(); i++) {
        if (f->inplace_data(i)) {
          clear[f->inplace_data_with(i)].first = false;
        }
      }
      for (int o = 0; o < outputs.size(); ++o) {
        if (prohibit_clear[o] || outputs[o]->persistent()) {
          continue;
        }
        if (clear[o].first) {
          outputs[o]->variable()->data()->array()->clear();
        }
        if (clear[o].second) {
          outputs[o]->variable()->grad()->array()->clear();
        }
      }
    }
  }

  // Get first visit flags and prohibit clear flags;
  // The prohibit clear flags are set by query_input_flags function with inputs
  // of a previously called function.
  pair<vector<bool>, vector<bool>>
  query_outputs_flags(const vector<CgVariablePtr> &outputs) {
    vector<bool> first_visit(outputs.size());
    vector<bool> prohibit_clear(outputs.size());
    for (int i = 0; i < outputs.size(); i++) {
      auto v = outputs[i];
      auto it = vseen_.find(v);
      bool first = it == vseen_.end();
      if (first) { // first visit
        // Terminal variable always doesn't allow to clear buffers.
        prohibit_clear[i] = true;

        // Note that the following vseen_[v] won't be referred in the current
        // implementation because query_input_flags is called earlier than
        // query_output_flags. TODO: We may be able to call query_output_flags
        // before query_input_flags.
        vseen_[v] = true;
      } else {
        // Propagate prohibit_clear_inputs_buffers flag from the previous seen
        // inputs.
        prohibit_clear[i] = it->second;
      }
      first_visit[i] = first;
    }
    return {first_visit, prohibit_clear};
  }

  vector<bool> query_input_flags(const vector<CgVariablePtr> &inputs,
                                 CgFunctionPtr func) {
    vector<bool> ret(inputs.size());
    bool prohibit_clear = func->function()->prohibit_clear_input_buffers();
    auto outputs = func->outputs();
    for (int i = 0; i < ret.size(); i++) {
      auto v = inputs[i];
      auto it = vseen_.find(v);
      bool first_visit = it == vseen_.end();
      ret[i] = first_visit;
      if (first_visit) {
        bool dummy;
        std::tie(it, dummy) = vseen_.insert({v, prohibit_clear});
      }

      // Prohibits clearing if any of previous function prohibits clearing
      // inputs.
      it->second |= prohibit_clear;

      // Propagate prohibit property from the output variables.
      if (func->function()->inplace_data(i)) {
        auto inplaced = outputs[func->function()->inplace_data_with(i)];
        auto it2 = vseen_.find(inplaced);
        if (it2 == vseen_.end() || it2->second) {
          it->second = true;
        }
      }
    }
    return ret;
  }

public:
  BackwardCallback(CgFunctionPtr f, bool clear_buffer,
                   function_hook_type function_pre_hook,
                   function_hook_type function_post_hook)
      : clear_buffer_(clear_buffer), function_pre_hook_(function_pre_hook),
        function_post_hook_(function_post_hook) {
    // Note prohibiting clearing variable buffers where terminal.
    for (auto o : f->outputs()) {
      vseen_.insert({o, true});
    }
  }

  void error_trace(const string &name_on_error) {
    // TODO: Optional verbosity
    std::cerr << "Error during backward propagation:" << std::endl;
    for (auto &name : history_) {
      std::cerr << "  " << name << std::endl;
    }
    std::cerr << "  " << name_on_error << " <-- ERROR" << std::endl;
  }

  void operator()(CgFunctionPtr f) {
    // Check accumulation.
    const auto inputs = f->inputs();
    auto first_visit_flags = query_input_flags(inputs, f);
    auto accum = get_accum(inputs, first_visit_flags);

    // Get output variables
    vector<CgVariablePtr> outputs;
    vector<Variable *> voutputs;
    std::tie(outputs, voutputs) = f->function_outputs();

    // Query output flags according to previous trace history.
    vector<bool> output_first_visit_flags;
    vector<bool> output_prohibit_clear;
    std::tie(output_first_visit_flags, output_prohibit_clear) =
        query_outputs_flags(outputs);

    // Check if any of outputs is unseen.
    force_zero_grad_if_unseen(outputs, output_first_visit_flags);

    // Call backward function
    vector<bool> prop_down(accum.size());
    std::transform(inputs.begin(), inputs.end(), prop_down.begin(),
                   [](CgVariablePtr v) { return v->need_grad_state(); });
    // std::cout << f->function()->name() << std::endl;
    // std::cout << "  " << string_join(prop_down, ",") << std::endl;
    // std::cout << "  " << string_join(accum, ",") << std::endl;
    try {
      auto call_callback = [f](function_hook_type &h) {
        if (h) {
          h(f);
        }
      };
      call_callback(function_pre_hook_);
      f->function()->backward(f->function_inputs(), voutputs, prop_down, accum);
      call_callback(function_post_hook_);
    } catch (...) {
      error_trace(f->function()->name());
      throw;
    }
    history_.push_back(f->function()->name());

    // Clear outputs buffer
    clear_output_buffers(f, output_prohibit_clear);
  }
};

void CgVariable::visit_function_recursive(
    CgFunctionPtr func, unordered_set<CgFunctionPtr> &fclosed,
    std::function<void(CgFunctionPtr)> forward_callback) {

  // A. Push the function to the closed list.
  fclosed.insert(func);

  // B. Open all inputs of the function.
  int max_rank = 0;       // 0 if no inputs in this function.
  bool need_grad = false; // No inputs not require grad
  bool need_setup = false;
  auto inputs = func->inputs();
  for (auto input : inputs) {
    auto parent = input->parent();
    // B-1. Input with no parent doesn't require
    if (!parent) {
      // Same as B-3.
      input->set_rank_(0);
      input->unset_need_grad_state();
      // Same as B-4.
      max_rank = std::max(0, max_rank);
      need_grad |= input->need_grad_state();
      need_setup |= input->check_and_unmark_need_setup(func);
      continue;
    }

    // B-2. Visit functions recursively if parent is not closed.
    if (fclosed.find(parent) == fclosed.end()) {
      visit_function_recursive(parent, fclosed, forward_callback);
    }

    // B-3. Update rank and need_grad of this input by propagating from the
    // parent function (backward with a rewired graph requires this).
    input->set_rank_(parent->rank());
    input->set_need_grad_state(parent->need_grad());

    // B-4. Aggregate rank and need_grad from inputs for func.
    max_rank = std::max(parent->rank(), max_rank);
    need_grad |= input->need_grad_state();
    need_setup |= input->check_and_unmark_need_setup(func);
  }

  // C. Update rank and need_grad of func (backward with a rewired graph
  // requires this).
  func->set_need_grad(need_grad); // This must be done before calling callback
                                  // because the callback may use this.
  func->set_rank_(++max_rank);    // Increment rank at function.

  // D. setup if required
  if (need_setup) {
    // std::cout << "setup is called at " << func->function()->name()
    //           << " as rank " << func->rank() << std::endl;
    func->setup();
    // Mark as all outputs require setup function call.
    for (auto output : func->outputs()) {
      output->mark_need_setup();
    }
  }

  // E. Verify flags
  func->verify_during_forward();

  // F. Call callback function at this function.
  forward_callback(func);
  // std::cout << max_rank << " " << func->function()->name() << " " <<
  // func.get() << std::endl;
}

void CgVariable::visit_function_backward(
    CgFunctionPtr p, std::function<void(CgFunctionPtr)> backward_callback,
    vector<CommunicatorBackwardCallbackPtr> communicator_callbacks) {
  // Open list of next search candidate.
  unordered_map<CgFunctionPtr, uint64_t> ids;
  /* Returns the ID for each function (layer) */
  auto get_id = [&ids](const CgFunctionPtr &ptr) -> uint64_t {
    auto it = ids.find(ptr);
    if (it == ids.end()) {
      /* Assign an ID to the function */
      auto id = ids.size();
      ids.insert({ptr, id});
      return id;
    }
    /* Return the previous ID if the ID is already assigned */
    return it->second;
  };
  set<tuple<int, uint64_t, CgFunctionPtr>> open;
  open.insert(make_tuple(-p->rank(), get_id(p), p));
  while (!open.empty()) {
    auto rank_func = open.begin();
    auto f = get<2>(*rank_func);
    DestructorCallback at_scope_exit([&]() { open.erase(rank_func); });
    // std::cout << "size: " << open.size();
    // std::cout << " --> " << open.size() << std::endl;
    if (!f->need_grad())
      continue;

    // Callback
    backward_callback(f);
    // std::cout << (int)(get<1>(*rank_func)) << ": " << f->rank() << " "
    //           << f->function()->name() << " " << f.get() << " " <<
    //           open.size()
    //           << std::endl;

    //
    for (auto &com_callback : communicator_callbacks) {
      com_callback->on_finish_function_backward(f);
    }

    // Propagate down.
    auto inputs = f->inputs();
    for (int i = 0; i < f->num_inputs(); i++) {
      auto inp = inputs[i];
      if (!inp->need_grad_state())
        continue;
      auto p_i = inp->parent();
      if (!p_i)
        continue;
      open.insert(make_tuple(-p_i->rank(), get_id(p_i), p_i));
    }
  }

  for (auto &com_callback : communicator_callbacks) {
    com_callback->on_finish_backward();
  }
}

void CgVariable::forward(bool clear_buffer, bool clear_no_need_grad,
                         unordered_set<CgFunctionPtr> *fclosed,
                         function_hook_type function_pre_hook,
                         function_hook_type function_post_hook) {
  unordered_set<CgFunctionPtr> scoped_fclosed;
  auto clear_buffer_state =
      SingletonManager::get<GlobalClearBufferState>()->state(
          clear_buffer, clear_no_need_grad);
  if (fclosed == nullptr) {
    fclosed = &scoped_fclosed;
  }
  NBLA_CHECK(parent_, error_code::value, "The variable has no parent.");
  ForwardCallback forward_callback(clear_buffer, clear_no_need_grad,
                                   function_pre_hook, function_post_hook);
  visit_function_recursive(
      parent_, *fclosed,
      [&forward_callback](CgFunctionPtr f) { forward_callback(f); });
}

void CgVariable::backward(
    NdArrayPtr grad, bool clear_buffer,
    vector<CommunicatorBackwardCallbackPtr> communicator_callbacks,
    function_hook_type function_pre_hook,
    function_hook_type function_post_hook) {
  NBLA_CHECK(parent_, error_code::value, "The variable has no parent.");
  auto clear_buffer_state =
      SingletonManager::get<GlobalClearBufferState>()->state(clear_buffer,
                                                             false);

  // Scoped context.
  // Set flags used during backward of this variable to avoid clearing
  // buffer. Also, set the grad array passed as an argument.
  NdArrayPtr bak_grad = this->variable()->grad();
  DestructorCallback at_scope_exit(
      [&]() { this->variable()->set_grad(bak_grad); });
  if (grad) {
    this->variable()->set_grad(grad);
  }

  // Create callback
  BackwardCallback backward_callback(parent_, clear_buffer, function_pre_hook,
                                     function_post_hook);

  // Visit backward
  visit_function_backward(
      parent_, [&backward_callback](CgFunctionPtr f) { backward_callback(f); },
      communicator_callbacks);
}

vector<CgFunctionPtr> CgVariable::function_references() {
  vector<CgFunctionPtr> ret(this->function_reference_count(), nullptr);
  int i = 0;
  for (auto pair : function_references_) {
    if (auto shared = pair.second.first.lock())
      ret[i++] = shared;
  }

  return ret;
}

void CgVariable::insert_function_reference(CgFunctionPtr func) {
  std::weak_ptr<CgFunction> wp(func);
  function_references_.insert(
      {func.get(), {wp, CgVariable::FunctionReferenceInfo()}});
}

void CgVariable::remove_function_reference(CgFunction *funcp) {
  auto it = function_references_.find(funcp);
  if (it == function_references_.end())
    return;
  function_references_.erase(it);
}

void CgVariable::mark_need_setup() {
  for (auto it = function_references_.begin(); it != function_references_.end();
       it++) {
    auto f = it->second.first.lock();
    if (!f) {
      function_references_.erase(it);
      continue;
    }
    it->second.second.need_setup = true;
  }
}

bool CgVariable::check_and_unmark_need_setup(CgFunctionPtr func) {
  auto it = function_references_.find(func.get());
  NBLA_CHECK(it != function_references_.end(), error_code::value,
             "Fatal issue: function reference has gone.");
  auto ret = it->second.second.need_setup;
  it->second.second.need_setup = false;
  return ret;
}

CgVariablePtr CgVariable::create_deep_copy(Context ctx, bool copy_grad) {
  auto ret = std::make_shared<CgVariable>(this->variable()->shape(),
                                          this->need_grad());

  dtypes dtype = this->variable()->data()->array()->dtype();

  const Array *x = this->variable()->data()->get(dtype, ctx);
  Array *y = ret->variable()->data()->cast(dtype, ctx, true);
  y->copy_from(x);

  if (copy_grad) {
    const Array *g = this->variable()->grad()->get(dtype, ctx);
    Array *h = ret->variable()->grad()->cast(dtype, ctx, true);
    h->copy_from(g);
  }

  return ret;
}
}
