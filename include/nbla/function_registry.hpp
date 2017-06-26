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

#ifndef __NBLA_FUNCTION_REGISTRY_HPP__
#define __NBLA_FUNCTION_REGISTRY_HPP__

#include <nbla/context.hpp>
#include <nbla/exception.hpp>
#include <nbla/function.hpp>
#include <nbla/preprocessor_magic.hpp>

#include <functional>
#include <memory>
#include <regex>
#include <sstream>
#include <string>
#include <vector>

namespace nbla {

/**
*/
template <typename Item>
std::string print_function_items(vector<shared_ptr<Item>> items) {
  std::ostringstream ss;
  ss << "[";
  for (auto &&item : items) {
    ss << "(" << item->rank << ",'" << item->backend << "','" << item->engine
       << "'),";
  }
  ss << "]";
  return ss.str();
}

/**
Container of FunctionDbItem that can be queried by regex.
*/
template <typename Item> class NBLA_API FunctionDb {
  vector<shared_ptr<Item>> items_;

public:
  /**
  Query function item (FunctionDbItem) by query key of device and engine
  (Regex can be used).
  */
  typename Item::function_t query(const string &backend, const string &engine) {
    std::regex re_backend(backend);
    std::regex re_engine(engine);
    vector<shared_ptr<Item>> cand(items_);
    cand.erase(std::remove_if(
                   cand.begin(), cand.end(),
                   [&](shared_ptr<Item> item) {
                     if (!std::regex_match(item->backend.begin(),
                                           item->backend.end(), re_backend)) {
                       return true;
                     }
                     if (!std::regex_match(item->engine.begin(),
                                           item->engine.end(), re_engine)) {
                       return true;
                     }
                     return false;
                   }),
               cand.end());
    NBLA_CHECK(cand.size() > 0, error_code::unclassified,
               "('%s', '%s') could not be found in %s", backend.c_str(),
               engine.c_str(), print_function_items<Item>(items_).c_str());
    // Use one that has highest priority
    return cand[cand.size() - 1]->function;
  }

  /**
  Adding function item (FunctionDbItem).
  */
  void add(shared_ptr<Item> item) {
    items_.push_back(item);
    std::sort(items_.begin(), items_.end(),
              [](shared_ptr<Item> a, shared_ptr<Item> b) -> int {
                return (a->rank < b->rank);
              }); // TODO: efficiency
  }
};

/**
Item of FunctionDb that stores query keys (backend and engine), function and
rank (if query matches multiple items, item that has highest rank is used.).
*/
template <typename Base, typename... Args> struct FunctionDbItem {
  typedef std::function<shared_ptr<Base>(const Context &ctx, Args...)>
      function_t;
  int rank;
  string backend;
  string engine;
  function_t function;

  bool operator<(const FunctionDbItem<Base, Args...> &right) {
    return this->rank < right.rank;
  }
};

/**
This is used as a base class of function registry for each function that has
numerically unique operation. See NBLA_REGISTER_FUNCTION.
*/
template <typename Base, typename... Args> class NBLA_API FunctionRegistry {
public:
  typedef FunctionDbItem<Base, Args...> item_t;

  /**
  Create a new function instance.
  */
  shared_ptr<Base> create(const Context &ctx, Args... args) {
    return function_db_.query(ctx.backend, ctx.compute_backend)(ctx, args...);
  }

  void add(shared_ptr<item_t> item) { function_db_.add(item); }

protected:
  FunctionDb<item_t> function_db_;
};

/**
This ifdef statement aim to support build on MSVC++ compiler.
The problem was `,##__VA_ARGS__` only works in GCC, but not works in MSVC
compiler. On the other hand, the workaround by NBLA_VA_ARGS for MSVC breaks
GCC preprocessor. I decided to branch very big portion of source code, but
I understand this approach is very inefficient and not maintainable. If anybody
has an idea, please let me know or PR is wellcome.
*/
#ifdef _MSC_VER
/**
*/
#define NBLA_REGISTER_FUNCTION_HEADER(NAME, ...)                               \
  NBLA_API FunctionRegistry<Function NBLA_VA_ARGS(__VA_ARGS__)>                \
      &get_##NAME##Registry();                                                 \
                                                                               \
  NBLA_API shared_ptr<Function> create_##NAME(                                 \
      NBLA_ARGDEFS(const Context &NBLA_VA_ARGS(__VA_ARGS__)));

#define NBLA_REGISTER_FUNCTION_SOURCE(NAME, ...)                               \
  NBLA_API FunctionRegistry<Function NBLA_VA_ARGS(__VA_ARGS__)>                \
      &get_##NAME##Registry() {                                                \
    static FunctionRegistry<Function NBLA_VA_ARGS(__VA_ARGS__)> registry;      \
    return registry;                                                           \
  }                                                                            \
                                                                               \
  NBLA_API shared_ptr<Function> create_##NAME(                                 \
      NBLA_ARGDEFS(const Context &NBLA_VA_ARGS(__VA_ARGS__))) {                \
    init_cpu();                                                                \
    return get_##NAME##Registry().create(                                      \
        NBLA_ARGS(const Context &NBLA_VA_ARGS(__VA_ARGS__)));                  \
  }

/**
This will be used inside init method.
*/
#define NBLA_REGISTER_FUNCTION_IMPL(BASE, CLS, RANK, BACKEND, ENGINE, ...)     \
  {                                                                            \
    std::function<shared_ptr<Function>(                                        \
        const Context &NBLA_VA_ARGS(__VA_ARGS__))>                             \
        func = [](NBLA_ARGDEFS(const Context &NBLA_VA_ARGS(__VA_ARGS__))) {    \
          return shared_ptr<Function>(                                         \
              new CLS(NBLA_ARGS(const Context &NBLA_VA_ARGS(__VA_ARGS__))));   \
        };                                                                     \
    typedef FunctionDbItem<Function NBLA_VA_ARGS(__VA_ARGS__)> item_t;         \
    get_##BASE##Registry().add(                                                \
        shared_ptr<item_t>(new item_t{RANK, BACKEND, ENGINE, func}));          \
  }

#else
/**
*/
#define NBLA_REGISTER_FUNCTION_HEADER(NAME, ...)                               \
  FunctionRegistry<Function, ##__VA_ARGS__> &get_##NAME##Registry();           \
                                                                               \
  NBLA_API shared_ptr<Function> create_##NAME(                                 \
      NBLA_ARGDEFS(const Context &, ##__VA_ARGS__));

#define NBLA_REGISTER_FUNCTION_SOURCE(NAME, ...)                               \
  FunctionRegistry<Function, ##__VA_ARGS__> &get_##NAME##Registry() {          \
    static FunctionRegistry<Function, ##__VA_ARGS__> registry;                 \
    return registry;                                                           \
  }                                                                            \
                                                                               \
  shared_ptr<Function> create_##NAME(                                          \
      NBLA_ARGDEFS(const Context &, ##__VA_ARGS__)) {                          \
    init_cpu();                                                                \
    return get_##NAME##Registry().create(                                      \
        NBLA_ARGS(const Context &, ##__VA_ARGS__));                            \
  }

/**
This will be used inside init method.
*/
#define NBLA_REGISTER_FUNCTION_IMPL(BASE, CLS, RANK, BACKEND, ENGINE, ...)     \
  {                                                                            \
    std::function<shared_ptr<Function>(const Context &, ##__VA_ARGS__)> func = \
        [](NBLA_ARGDEFS(const Context &, ##__VA_ARGS__)) {                     \
          return shared_ptr<Function>(                                         \
              new CLS(NBLA_ARGS(const Context &, ##__VA_ARGS__)));             \
        };                                                                     \
    typedef FunctionDbItem<Function, ##__VA_ARGS__> item_t;                    \
    get_##BASE##Registry().add(                                                \
        shared_ptr<item_t>(new item_t{RANK, BACKEND, ENGINE, func}));          \
  }
#endif
}
#endif
