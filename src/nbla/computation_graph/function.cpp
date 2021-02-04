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

#include <nbla/computation_graph/function.hpp>
#include <nbla/computation_graph/variable.hpp>

#include <memory>

namespace nbla {

using std::make_shared;

// Just a helper function.
static inline const char *b2str(bool b) { return b ? "true" : "false"; }

void CgFunction::OutputWrapper::set(CgVariablePtr v) {
  this->weak_reference = v;
  this->internal_variable = v->variable();
}
CgVariablePtr CgFunction::OutputWrapper::get() {
  auto o = this->weak_reference.lock();
  if (o) {
    return o;
  }
  /*
    If the CgVariable instace is deleted before using, a new CgVariable
    instance is created using the Variable instance which has been held in the
    deleted CgVariable instance.
   */
  return make_shared<CgVariable>(this->internal_variable);
}

CgFunction::CgFunction(FunctionPtr func) : rank_(0), func_(func) {}

CgFunction::~CgFunction() {
  for (auto i : this->inputs()) {
    i->remove_function_reference(this);
  }
}

void CgFunction::setup() {
  // Copy if function is already used.
  if (func_->ask_if_used_and_use()) {
    func_ = func_->copy();
  }
  // Get output variables
  vector<CgVariablePtr> outputs;
  vector<Variable *> voutputs;
  std::tie(outputs, voutputs) = this->function_outputs();
  func_->setup(this->function_inputs(), voutputs);
}

void CgFunction::check_data_inplace(int i, CgVariablePtr input,
                                    const vector<CgVariablePtr> &outputs) {
  auto f = this->function();
  // Always not allow modifying data if grad depends the output data.
  if (input->need_grad_state()) {
    for (vector<CgVariablePtr>::size_type o = 0; o < outputs.size(); ++o) {
      // If function gradient computation at i-th variable depends on o-th
      // output data, inplacing o-th variable data is prohibited.
      if (f->grad_depends_output_data(i, o)) {
        outputs[o]->set_allow_modify_data(false);
      }
    }
  }
  int inplace_level = f->inplace_data(i);
  if (inplace_level == Function::INPLACE) {
    NBLA_CHECK(input->allow_modify_data(), error_code::value,
               "Modifying data is prohibited by the parent function of the "
               "%d-th input data of '%s' (depth=%d). (Parent is '%s').",
               i, f->name().c_str(), this->rank(),
               input->parent()->function()->name().c_str());
    NBLA_CHECK(input->function_reference_count() < 2, error_code::value,
               "In-placing at a branching variable is prohibited. %d-th input "
               "data of `%s` (depth=%d) is inplaced.",
               i, f->name().c_str(), this->rank());
  } else if (inplace_level == Function::INPLACE_NOT_MODIFY) {
    // A variable that branches or requires grad doesn't allow modify data at
    // the in-placed variable.
    if (input->function_reference_count() > 1 || input->need_grad_state()) {
      outputs[f->inplace_data_with(i)]->set_allow_modify_data(false);
    }
  }
}

void CgFunction::verify_during_forward() {
  for (auto o : this->outputs()) {
    o->set_allow_modify_data(true);
  }
  auto inputs = this->inputs();
  auto outputs = this->outputs();
  for (vector<CgVariablePtr>::size_type i = 0; i < inputs.size(); ++i) {
    this->check_data_inplace(i, inputs[i], outputs);
  }
}

void CgFunction::set_outputs(const vector<CgVariablePtr> &outputs) {
  outputs_.resize(outputs.size());
  for (vector<CgVariablePtr>::size_type i = 0; i < outputs.size(); ++i) {
    outputs[i]->set_rank_(rank_ + 1);
    outputs_[i].set(outputs[i]);
  }
}

vector<CgVariablePtr> CgFunction::outputs() {
  vector<CgVariablePtr> outputs(outputs_.size());
  for (vector<CgVariablePtr>::size_type i = 0; i < outputs_.size(); ++i) {
    outputs[i] = outputs_[i].get();
  }
  return outputs;
}

vector<Variable *> CgFunction::function_inputs() {
  vector<Variable *> ret(inputs_.size());
  for (vector<CgVariablePtr>::size_type i = 0; i < inputs_.size(); ++i) {
    ret[i] = inputs_[i]->variable().get();
  }
  return ret;
}

pair<vector<CgVariablePtr>, vector<Variable *>> CgFunction::function_outputs() {
  auto outputs = this->outputs();
  vector<Variable *> voutputs(outputs.size());
  std::transform(
      outputs.begin(), outputs.end(), voutputs.begin(),
      [](CgVariablePtr v) -> Variable * { return v->variable().get(); });
  return {outputs, voutputs};
}
}
