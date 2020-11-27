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

#include <nbla/computation_graph/function.hpp>
#include <nbla/computation_graph/variable.hpp>
#include <nbla/global_function_callback.hpp>
#include <nbla/logger.hpp>
#include <nbla/singleton_manager-internal.hpp>

namespace nbla {

GlobalFunctionCallback::GlobalFunctionCallback() {}

GlobalFunctionCallback::~GlobalFunctionCallback() {}

void GlobalFunctionCallback::call_pre_hooks(const CgFunctionPtr &f) {
  for (auto &p : pre_hooks_)
    p.second(f);
}

void GlobalFunctionCallback::call_post_hooks(const CgFunctionPtr &f) {
  for (auto &p : post_hooks_)
    p.second(f);
}

void GlobalFunctionCallback::set_pre_hook(const string &key,
                                          const function_hook_type &cb) {
  auto it = std::find_if(
      pre_hooks_.begin(), pre_hooks_.end(),
      [&](const KeyCallbackPair &elem) { return elem.first == key; });

  if (it != pre_hooks_.end()) {
    NBLA_LOG_WARN(
        "[GlobalFunctionCallback::set_pre_hook()] key %s already exists.", key);
    return;
  }

  pre_hooks_.emplace_back(key, cb);
}

void GlobalFunctionCallback::set_post_hook(const string &key,
                                           const function_hook_type &cb) {
  auto it = std::find_if(
      post_hooks_.begin(), post_hooks_.end(),
      [&](const KeyCallbackPair &elem) { return elem.first == key; });

  if (it != post_hooks_.end()) {
    NBLA_LOG_WARN(
        "[GlobalFunctionCallback::set_post_hook()] key %s already exists.",
        key);
    return;
  }

  post_hooks_.emplace_back(key, cb);
}

void GlobalFunctionCallback::unset_pre_hook(const string &key) {
  auto it = std::find_if(
      pre_hooks_.begin(), pre_hooks_.end(),
      [&](const KeyCallbackPair &elem) { return elem.first == key; });

  if (it == pre_hooks_.end()) {
    NBLA_LOG_WARN(
        "[GlobalFunctionCallback::unset_pre_hook()] key %s does't exists.",
        key);
    return;
  }

  pre_hooks_.erase(it);
}

void GlobalFunctionCallback::unset_post_hook(const string &key) {
  auto it = std::find_if(
      post_hooks_.begin(), post_hooks_.end(),
      [&](const KeyCallbackPair &elem) { return elem.first == key; });

  if (it == post_hooks_.end()) {
    NBLA_LOG_WARN(
        "[GlobalFunctionCallback::unset_post_hook()] key %s does't exists.",
        key);
    return;
  }

  post_hooks_.erase(it);
}

NBLA_INSTANTIATE_SINGLETON(NBLA_API, GlobalFunctionCallback);

/** API implementation
 */
void set_function_pre_hook(const string &key, const function_hook_type &cb) {
  SingletonManager::get<GlobalFunctionCallback>()->set_pre_hook(key, cb);
}

void set_function_post_hook(const string &key, const function_hook_type &cb) {
  SingletonManager::get<GlobalFunctionCallback>()->set_post_hook(key, cb);
}

void unset_function_pre_hook(const string &key) {
  SingletonManager::get<GlobalFunctionCallback>()->unset_pre_hook(key);
}

void unset_function_post_hook(const string &key) {
  SingletonManager::get<GlobalFunctionCallback>()->unset_post_hook(key);
}
}
