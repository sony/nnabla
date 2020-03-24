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

#include <nbla/cpu.hpp>
#include <nbla/singleton_manager-internal.hpp>

#include <nbla/memory/caching_allocator_with_buckets.hpp>
#include <nbla/memory/cpu_memory.hpp>
#include <nbla/memory/naive_allocator.hpp>

namespace nbla {
Cpu::Cpu()
    : naive_allocator_(make_shared<NaiveAllocator<CpuMemory>>()),
      caching_allocator_(
          make_shared<CachingAllocatorWithBuckets<CpuMemory>>()) {}

Cpu::~Cpu() {}

vector<string> Cpu::array_classes() const { return array_classes_; }

void Cpu::_set_array_classes(const vector<string> &a) { array_classes_ = a; }

void Cpu::register_array_class(const string &name) {
  array_classes_.push_back(name);
}

shared_ptr<Allocator> Cpu::caching_allocator() { return caching_allocator_; }
shared_ptr<Allocator> Cpu::naive_allocator() { return naive_allocator_; }

void Cpu::free_unused_host_caches() {
  caching_allocator_->free_unused_caches();
  naive_allocator_->free_unused_caches();
}

void Cpu::device_synchronize(const string &device) {
  cpu_device_synchronize(device);
}

void Cpu::default_stream_synchronize(const string &device) {}

NBLA_INSTANTIATE_SINGLETON(NBLA_API, Cpu);
}
