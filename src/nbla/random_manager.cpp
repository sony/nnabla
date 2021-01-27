// Copyright (c) 2021 Sony Corporation. All Rights Reserved.
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

#include <nbla/random_manager.hpp>
#include <nbla/singleton_manager-internal.hpp>

namespace nbla {
RandomManager::RandomManager() : seed_(313), count_(0) {
  rgen_ = std::mt19937(seed_);
}

RandomManager::~RandomManager() {}

std::mt19937 &RandomManager::get_rand_generator() { return rgen_; }

unsigned int RandomManager::get_seed() const { return seed_; }

void RandomManager::set_seed(unsigned int seed) {
  seed_ = seed;
  rgen_ = std::mt19937(seed);
  ++count_;
}

int RandomManager::get_count() const { return count_; }

NBLA_INSTANTIATE_SINGLETON(NBLA_API, RandomManager);
}
