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

#ifndef __NBLA_GLOBAL_CONTEXT_HPP__
#define __NBLA_GLOBAL_CONTEXT_HPP__
#include <nbla/defs.hpp>
#include <nbla/singleton_manager.hpp>

namespace nbla {
/**
Singleton class for storing global context.
*/
class NBLA_API GlobalContext {
  Context current_;

public:
  ~GlobalContext();
  /** Get current context.
   */
  Context get_current_context() const;

  /** Set current context.
   */
  void set_current_context(Context ctx);

private:
  friend SingletonManager;
  // Never called by users.
  GlobalContext();
  DISABLE_COPY_AND_ASSIGN(GlobalContext);
};
}
#endif
