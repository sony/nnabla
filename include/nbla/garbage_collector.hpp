// Copyright 2017,2018,2019,2020,2021 Sony Corporation.
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

#ifndef __NBLA_GARBAGE_COLLECTOR_HPP__
#define __NBLA_GARBAGE_COLLECTOR_HPP__

#include <nbla/defs.hpp>
#include <nbla/singleton_manager.hpp>

#include <functional>
#include <vector>

namespace nbla {
using std::vector;

/** Singleton for garbage collector registry.

   This singleton class is intended to be used to freeing cpu/device memory
   held by an interpreter language which manages memory by GC (e.g. Python). GC
   function as c++ callback function can be registered via `register_collector`
   function. All registered GC function is called usually in case memory
   allocation fails.
 */
class NBLA_API GarbageCollector {
public:
  typedef std::function<void()> collector_type;

  /** Register a GC function with a form `void ()`.
   */
  void register_collector(collector_type f);

  /** Call all registered GC function.
   */
  void collect();

  ~GarbageCollector();

private:
  friend SingletonManager;
  vector<collector_type> collectors_;
  GarbageCollector();
};
}
#endif
