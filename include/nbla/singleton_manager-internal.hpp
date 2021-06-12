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

#ifndef __NBLA_SINGLETON_MANAGER_INTERNAL_HPP__
#define __NBLA_SINGLETON_MANAGER_INTERNAL_HPP__

#include <nbla/singleton_manager.hpp>
// #include <iostream>
#include <typeinfo>

namespace nbla {

template <typename SINGLETON> SINGLETON *SingletonManager::get() {
  static std::mutex mtx_;
  std::lock_guard<std::mutex> lock(mtx_);

  static SINGLETON *r = nullptr;
  if (r)
    return r;
  SingletonManager &s = get_self();
  // TODO: Enable debug print
  // std::cout << "Creating a singleton \"" << typeid(SINGLETON).name() << "\""
  //           << std::endl;
  r = new SINGLETON{};

  auto deleter = []() -> void {
    // TODO: Enable debug print
    // std::cout << "Deleting a singleton \"" << typeid(SINGLETON).name() <<
    // "\""
    //          << std::endl;
    delete r; // Static variable doesn't require capturing.
    r = nullptr;
  };
  int id = s.count_;
  s.singletons_.insert({id, {(uintptr_t)r, deleter}}); // Register deleter
  s.adr2id_.insert({(uintptr_t)r, id});                // Register ID
  s.count_ += 1;
  return r;
}

template <typename SINGLETON> int SingletonManager::get_id() {
  SingletonManager &s = get_self();
  auto address = (uintptr_t)(get<SINGLETON>());
  return s.adr2id_[address];
}

template <typename SINGLETON> void SingletonManager::erase() {
  erase_by_id(get_id<SINGLETON>());
}

/** Template instantiation to register a Singleton to SingletonManager.
 */
#define NBLA_INSTANTIATE_SINGLETON(API, SINGLETON_CLASS)                       \
  template API SINGLETON_CLASS *SingletonManager::get<SINGLETON_CLASS>();      \
  template API int SingletonManager::get_id<SINGLETON_CLASS>();                \
  template API void SingletonManager::erase<SINGLETON_CLASS>()
}
#endif
