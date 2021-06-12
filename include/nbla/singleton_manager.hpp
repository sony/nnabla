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

#ifndef __NBLA_SINGLETON_MANAGER_HPP__
#define __NBLA_SINGLETON_MANAGER_HPP__

#include <functional>
#include <memory>
#include <mutex>
#include <thread>
#include <unordered_map>

#include <nbla/array.hpp>
#include <nbla/context.hpp>
#include <nbla/defs.hpp>
#include <nbla/synced_array.hpp>

namespace nbla {
using std::unique_ptr;
using std::unordered_map;

/** API for getting or creating and deleting any singleton classes.

In NNabla, all Singleton should be instantiated via static function
`get<SINGLETON>()`. This enables you to manage all singleton instance in one
place,
and also to delete them by your hand.
 */
class NBLA_API SingletonManager {

public:
  /** Get or create an singleton instance of a class specified by the template
     argument.

      This will create an instance, and schedule to delete the
      instance at the end of process, or let you delete by hand through erase
     function.

      @return A pointer to a singleton instance.
   */
  template <typename SINGLETON> static SINGLETON *get();

  /** Get an integer ID of a Singleton class. This can be used as an argument of
      `erase_by_id` function.
   */
  template <typename SINGLETON> static int get_id();

  /** Delete all singleton instance previously created via `get<SINGLETON>()`.
   */
  static void clear();

  /** Delete a singleton instance specified by a template argument.
   */
  template <typename SINGLETON> static void erase();

  /** Delete a singleton instance specified by an ID which can be gotten
      by `get_id\<SINGLETON\>()` function.
   */
  static void erase_by_id(int id);

private:
  int count_; ///< How many singletons in this registry.
  ///< Hash map from ID to a pair of address and deleter function.
  unordered_map<int, pair<uintptr_t, std::function<void()>>> singletons_;
  unordered_map<uintptr_t, int> adr2id_; ///< Hash map from address to ID.

  static SingletonManager
      *self_; ///< Singleton instance pointer. Never be destroyed.

  /** Get this singleton (managing singletons) class.
   */
  static SingletonManager &get_self();
  SingletonManager();
  ~SingletonManager();
  DISABLE_COPY_AND_ASSIGN(SingletonManager);
};

/** Reusable resources for ones and zeros for any devices.
*/
class NBLA_API NNabla {
  std::mutex mtx_zeros_;
  std::mutex mtx_ones_;

public:
  ~NNabla();
  /**
  Get ones array with specified size as a void pointer.
  */
  const void *ones(Size_t size, dtypes dtype, const Context &ctx);

  /**
  Get zeros array with specified size as a void pointer.
  */
  const void *zeros(Size_t size, dtypes dtype, const Context &ctx);

protected:
  unordered_map<std::thread::id, shared_ptr<SyncedArray>> ones_;
  unordered_map<std::thread::id, shared_ptr<SyncedArray>> zeros_;

private:
  friend SingletonManager;
  NNabla();
  DISABLE_COPY_AND_ASSIGN(NNabla);
};
}

#endif
