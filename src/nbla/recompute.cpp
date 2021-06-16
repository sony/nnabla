// Copyright 2021 Sony Group Corporation.
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

#include <nbla/recompute.hpp>
#include <nbla/singleton_manager-internal.hpp>

namespace nbla {
Recompute::Recompute() : current_(false) {}

Recompute::~Recompute() {}

bool Recompute::get_global_recompute() const { return current_; }

void Recompute::set_global_recompute(bool recompute) { current_ = recompute; }

NBLA_INSTANTIATE_SINGLETON(NBLA_API, Recompute);

// Wrapper function for Recompute::get_global_recompute.
bool get_global_recompute() {
  return SingletonManager::get<Recompute>()->get_global_recompute();
}

// Wrapper functions for Python interface.
void c_set_global_recompute(const bool recompute) {
  SingletonManager::get<Recompute>()->set_global_recompute(recompute);
}

bool c_get_global_recompute() { return get_global_recompute(); }
}
