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

#ifndef __NBLA_BACKEND_BASE_HPP__
#define __NBLA_BACKEND_BASE_HPP__

#include <nbla/common.hpp>
#include <nbla/memory/allocator.hpp>
#include <nbla/singleton_manager.hpp>

#include <memory>
#include <string>
#include <vector>

namespace nbla {

using std::vector;
using std::string;
using std::shared_ptr;

/**
Base class of singleton classes for storing some handles or
configs for computation on a backend.
*/
class NBLA_API BackendBase {

public:
  virtual ~BackendBase() {}

  /** Available array class list used in CPU Function implementations.
   */
  virtual vector<string> array_classes() const = 0;

  /** Set array class list.

      @note Dangerous to call. End users shouldn't call.
   */
  virtual void _set_array_classes(const vector<string> &a) = 0;

  /** Register array class to available list by name.
   */
  virtual void register_array_class(const string &name) = 0;

  /** Get a caching allocator.
   */
  virtual shared_ptr<Allocator> caching_allocator() = 0;

  /** Get a no-cache allocator.
   */
  virtual shared_ptr<Allocator> naive_allocator() = 0;

  /** Free all unused host memory caches
   */
  virtual void free_unused_host_caches() = 0;

  /** Synchronize host to device.
   */
  virtual void device_synchronize(const string &device) = 0;

  /** Synchronize host to default stream of device.
   */
  virtual void default_stream_synchronize(const string &device) = 0;

  /** Create non blockuing streams for data transfer
   */
  virtual void create_lms_streams(int device = -1) = 0;

protected:
  vector<string> array_classes_; ///< Available array classes

  /*
    NOTE: Allocators must be shared_ptr in order to be passed to a
    AllocatorMemory instance to prevent destructing allocators before
    destructing AllocatorMemory.
   */
  shared_ptr<Allocator> naive_allocator_;
  shared_ptr<Allocator> caching_allocator_;

  BackendBase() {} // Never called by users.

private:
  friend SingletonManager;
  DISABLE_COPY_AND_ASSIGN(BackendBase);
};
}

#endif
