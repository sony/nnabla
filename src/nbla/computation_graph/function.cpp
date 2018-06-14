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

CgFunction::CgFunction(FunctionPtr func) : rank_(0) {
  // Copy if function is already used.
  if (func->ask_if_used_and_use()) {
    func = func->copy();
  }
  func_ = func;
}
void CgFunction::set_inputs(const vector<CgVariablePtr> &inputs) {
  // Check need_grad
  need_grad_ = false;
  for (auto i : inputs) {
    need_grad_ |= i->need_grad_state();
    rank_ = std::max(rank_, i->rank());
    i->increment_function_reference_count();
  }
  inputs_ = inputs;
}

void CgFunction::set_outputs(const vector<CgVariablePtr> &outputs) {
  outputs_.resize(outputs.size());
  for (int i = 0; i < outputs.size(); ++i) {
    outputs[i]->set_rank(rank_ + 1);
    outputs_[i] = outputs[i];
  }
}

vector<CgVariablePtr> CgFunction::outputs() {
  vector<CgVariablePtr> outputs(outputs_.size());
  for (int i = 0; i < outputs_.size(); ++i) {
    auto o = outputs_[i].lock();
    NBLA_CHECK(o, error_code::value,
               "Weak reference to outputs[%d] has gone at %s.", i,
               func_->name().c_str());
    outputs[i] = o;
  }
  return outputs;
}

vector<Variable *> CgFunction::function_inputs() {
  vector<Variable *> ret(inputs_.size());
  for (int i = 0; i < inputs_.size(); ++i) {
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
