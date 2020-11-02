// Copyright (c) 2019 Sony Corporation. All Rights Reserved.
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

#include <nbla/global_context.hpp>
#include <nbla/singleton_manager-internal.hpp>

namespace nbla {
GlobalContext::GlobalContext()
    : current_(Context({"cpu:float"}, "CpuCachedArray", "0")) {}

GlobalContext::~GlobalContext() {}

Context GlobalContext::get_current_context() const { return current_; }

void GlobalContext::set_current_context(const Context ctx) {
  std::lock_guard<std::mutex> lock(mutex_);
  current_ = ctx;
}

NBLA_INSTANTIATE_SINGLETON(NBLA_API, GlobalContext);
}
