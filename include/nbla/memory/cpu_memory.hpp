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

/** Cpu memory implementation.

    A memory block allocated by ::malloc function is managed by an instance.

    \ingroup MemoryImplGrp
 */
class NBLA_API CpuMemory : public Memory {
  CpuMemory(size_t bytes, const string &device_id, void *ptr);

public:
  CpuMemory(size_t bytes, const string &device_id);
  ~CpuMemory();

protected:
  bool alloc_impl() override;
  shared_ptr<Memory> divide_impl(size_t second_start) override;
  void merge_next_impl(Memory *from) override;
  void merge_prev_impl(Memory *from) override;
};
}
