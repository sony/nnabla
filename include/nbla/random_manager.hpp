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

#ifndef __NBLA_RANDOM_HPP__
#define __NBLA_RANDOM_HPP__
#include <nbla/defs.hpp>
#include <nbla/singleton_manager.hpp>

#include <random>

namespace nbla {
/**
Singleton class for storing global context.
*/
class NBLA_API RandomManager {
  std::mt19937 rgen_;
  unsigned seed_;
  int count_;

public:
  ~RandomManager();
  /** Get random generator.
   */
  std::mt19937 &get_rand_generator();

  /** Get seed value.
   */
  unsigned int get_seed() const;

  /** Set seed and recreate random generator.
   */
  void set_seed(unsigned int seed);

  /** Get count of seed sets used for extensions to track seed update.
   */
  int get_count() const;

private:
  friend SingletonManager;
  // Never called by users.
  RandomManager();
  DISABLE_COPY_AND_ASSIGN(RandomManager);
};
}
#endif
