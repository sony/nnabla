// Copyright (c) 2020 Sony Corporation. All Rights Reserved.
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

#include <nbla/global_solver_callback.hpp>
#include <nbla/logger.hpp>
#include <nbla/singleton_manager-internal.hpp>
#include <nbla/solver.hpp>

namespace nbla {

GlobalSolverCallback::GlobalSolverCallback() {}

GlobalSolverCallback::~GlobalSolverCallback() {}

void GlobalSolverCallback::call_pre_hooks() {
  for (auto &p : pre_hooks_)
    p.second();
}

void GlobalSolverCallback::call_post_hooks() {
  for (auto &p : post_hooks_)
    p.second();
}

void GlobalSolverCallback::set_pre_hook(const string &key,
                                        const update_hook_type &cb) {
  auto it = std::find_if(
      pre_hooks_.begin(), pre_hooks_.end(),
      [&](const KeyCallbackPair &elem) { return elem.first == key; });

  if (it != pre_hooks_.end()) {
    NBLA_LOG_WARN(
        "[GlobalSolverCallback::set_pre_hook()] key %s already exists.", key);
    return;
  }

  pre_hooks_.emplace_back(key, cb);
}

void GlobalSolverCallback::set_post_hook(const string &key,
                                         const update_hook_type &cb) {
  auto it = std::find_if(
      post_hooks_.begin(), post_hooks_.end(),
      [&](const KeyCallbackPair &elem) { return elem.first == key; });

  if (it != post_hooks_.end()) {
    NBLA_LOG_WARN(
        "[GlobalSolverCallback::set_post_hook()] key %s already exists.", key);
    return;
  }

  post_hooks_.emplace_back(key, cb);
}

void GlobalSolverCallback::unset_pre_hook(const string &key) {
  auto it = std::find_if(
      pre_hooks_.begin(), pre_hooks_.end(),
      [&](const KeyCallbackPair &elem) { return elem.first == key; });

  if (it == pre_hooks_.end()) {
    NBLA_LOG_WARN(
        "[GlobalSolverCallback::unset_pre_hook()] key %s does't exists.", key);
    return;
  }

  pre_hooks_.erase(it);
}

void GlobalSolverCallback::unset_post_hook(const string &key) {
  auto it = std::find_if(
      post_hooks_.begin(), post_hooks_.end(),
      [&](const KeyCallbackPair &elem) { return elem.first == key; });

  if (it == post_hooks_.end()) {
    NBLA_LOG_WARN(
        "[GlobalSolverCallback::unset_post_hook()] key %s does't exists.", key);
    return;
  }

  post_hooks_.erase(it);
}

NBLA_INSTANTIATE_SINGLETON(NBLA_API, GlobalSolverCallback);

/**API implementation**/
void set_solver_pre_hook(const string &key, const update_hook_type &cb) {
  SingletonManager::get<GlobalSolverCallback>()->set_pre_hook(key, cb);
}

void set_solver_post_hook(const string &key, const update_hook_type &cb) {
  SingletonManager::get<GlobalSolverCallback>()->set_post_hook(key, cb);
}

void unset_solver_pre_hook(const string &key) {
  SingletonManager::get<GlobalSolverCallback>()->unset_pre_hook(key);
}

void unset_solver_post_hook(const string &key) {
  SingletonManager::get<GlobalSolverCallback>()->unset_post_hook(key);
}
}
