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

#ifndef __NBLA_SOLVER_BASE_SOLVER_HPP__
#define __NBLA_SOLVER_BASE_SOLVER_HPP__

#include <nbla/cpu.hpp>
#include <nbla/solver.hpp>
#include <nbla/solver_registry.hpp>

namespace nbla {

/** Base solver implements weight decay and some common functions.
Note that this is an abstract function.
*/
template <typename T> class NBLA_API BaseSolver : public Solver {
public:
  BaseSolver(const Context &ctx);
  virtual ~BaseSolver();
  virtual string name() { return "BaseSolver"; }
  virtual vector<string> allowed_array_classes() {
    return SingletonManager::get<Cpu>()->array_classes();
  }

protected:
  virtual void set_state_impl(const string &key, VariablePtr param);
  virtual void remove_state_impl(const string &key);
  virtual void weight_decay_impl(const string &key, VariablePtr param,
                                 float decay_rate);
};
}
#endif
