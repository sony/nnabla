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

#ifndef __NBLA_CPU_HPP__
#define __NBLA_CPU_HPP__
#include <nbla/defs.hpp>
#include <nbla/memory/allocator.hpp>
#include <nbla/singleton_manager.hpp>

#include <memory>
#include <string>

namespace nbla {

using std::vector;
using std::string;
using std::unique_ptr;

/**
Singleton class for storing some handles or configs for CPU Computation.
*/
class NBLA_API Cpu {

public:
  ~Cpu();

  /** Available array class list used in CPU Function implementations.
   */
  vector<string> array_classes() const;

  /** Set array class list.

      @note Dangerous to call. End users shouldn't call.
   */
  void _set_array_classes(const vector<string> &a);

  /** Register array class to available list by name.
   */
  void register_array_class(const string &name);

  /** Get a caching allocator.
   */
  shared_ptr<Allocator> caching_allocator();

  /** Get a no-cache allocator.
   */
  shared_ptr<Allocator> naive_allocator();

protected:
  vector<string> array_classes_; ///< Available array classes

  /*
    NOTE: Allocators must be shared_ptr in order to be passed to a
    AllocatorMemory instance to prevent destructing allocators before
    destructing AllocatorMemory.
   */
  shared_ptr<Allocator> naive_allocator_;
  shared_ptr<Allocator> caching_allocator_;

private:
  friend SingletonManager;
  // Never called by users.
  Cpu();
  DISABLE_COPY_AND_ASSIGN(Cpu);
};
}

#endif
