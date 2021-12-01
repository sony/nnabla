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

#ifndef __NBLA_RECOMPUTE_HPP__
#define __NBLA_RECOMPUTE_HPP__
#include <nbla/defs.hpp>
#include <nbla/singleton_manager.hpp>

namespace nbla {
/**
Singleton class for storing global recompute flag.
*/
class NBLA_API Recompute {
  bool current_;

public:
  ~Recompute();
  /** Get current global recompute flag.
   */
  bool get_global_recompute() const;

  /** Set global recompute flag.
   */
  void set_global_recompute(bool recompute);

private:
  friend SingletonManager;
  // Never called by users.
  Recompute();
  DISABLE_COPY_AND_ASSIGN(Recompute);
};

NBLA_API bool get_global_recompute();

NBLA_API void c_set_global_recompute(const bool recompute);

NBLA_API bool c_get_global_recompute();
}
#endif
