// Copyright 2019,2020,2021 Sony Corporation.
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

#pragma once

#include <nbla/memory/memory.hpp>

namespace nbla {
class NBLA_API AllocatorCallback {

public:
  virtual void on_alloc(size_t bytes, const string &device_id) = 0;
  virtual void on_free(size_t bytes, const string &device_id) = 0;
  virtual void on_free_unused_device_caches(const string &device_id,
                                            size_t freed_bytes) = 0;
  virtual void on_allocation_failure() = 0;
};

class PrintingAllocatorCallback : public AllocatorCallback {
  const string name_;

public:
  PrintingAllocatorCallback(const string &name);
  void on_alloc(size_t bytes, const string &device_id) override;
  void on_free(size_t bytes, const string &device_id) override;
  void on_free_unused_device_caches(const string &device_id,
                                    size_t freed_bytes) override;
  void on_allocation_failure() override;
};
}
