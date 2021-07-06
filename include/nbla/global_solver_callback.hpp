// Copyright 2021 Sony Corporation.
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

#ifndef __NBLA_GLOBAL_SOLVER_CALLBACK_HPP__
#define __NBLA_GLOBAL_SOLVER_CALLBACK_HPP__

#include <nbla/defs.hpp>
#include <nbla/singleton_manager.hpp>

namespace nbla {

typedef std::function<void(void)> update_hook_type;

/**
Singleton class for storing global function callbacks.
*/
class NBLA_API GlobalSolverCallback {
  // Use vector<pair> instead of map to preserve the order of insertion.
  typedef pair<string, update_hook_type> KeyCallbackPair;
  vector<KeyCallbackPair> pre_hooks_;
  vector<KeyCallbackPair> post_hooks_;

public:
  ~GlobalSolverCallback();
  /** Call pre_hooks.
   */
  void call_pre_hooks();

  /** Call pre_hooks.
   */
  void call_post_hooks();

  /** Set a pre_hook.
 */
  void set_pre_hook(const string &key, const update_hook_type &cb);

  /** Set a post_hook.
   */
  void set_post_hook(const string &key, const update_hook_type &cb);

  /** Unset a pre_hook.
   */
  void unset_pre_hook(const string &key);

  /** Unset a post_hook.
   */
  void unset_post_hook(const string &key);

private:
  friend SingletonManager;
  // Never called by users.
  GlobalSolverCallback();
  DISABLE_COPY_AND_ASSIGN(GlobalSolverCallback);
};

/**API**/
NBLA_API void set_solver_pre_hook(const string &key,
                                  const update_hook_type &cb);
NBLA_API void set_solver_post_hook(const string &key,
                                   const update_hook_type &cb);
NBLA_API void unset_solver_pre_hook(const string &key);
NBLA_API void unset_solver_post_hook(const string &key);
}
#endif
