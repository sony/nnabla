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

#ifndef __NBLA_SOLVER_REGISTRY_HPP__
#define __NBLA_SOLVER_REGISTRY_HPP__

#include <nbla/function_registry.hpp>
#include <nbla/solver.hpp>

namespace nbla {

/**
This ifdef statement aim to support build on MSVC++ compiler.
The problem was `,##__VA_ARGS__` only works in GCC, but not works in MSVC
compiler. On the other hand, the workaround by NBLA_VA_ARGS for MSVC breaks
GCC preprocessor. I decided to branch very big portion of source code, but
I understand this approach is very inefficient and not maintainable. If anybody
has an idea, please let me know or PR is welcome.
*/
#ifdef _MSC_VER
/**
*/
#define NBLA_REGISTER_SOLVER_HEADER(NAME, ...)                                 \
  NBLA_API FunctionRegistry<Solver NBLA_VA_ARGS(__VA_ARGS__)>                  \
      &get_##NAME##SolverRegistry();                                           \
                                                                               \
  NBLA_API shared_ptr<Solver> create_##NAME##Solver(                           \
      NBLA_ARGDEFS(const Context &NBLA_VA_ARGS(__VA_ARGS__)));

#define NBLA_REGISTER_SOLVER_SOURCE(NAME, ...)                                 \
  FunctionRegistry<Solver NBLA_VA_ARGS(__VA_ARGS__)>                           \
      &get_##NAME##SolverRegistry() {                                          \
    static FunctionRegistry<Solver NBLA_VA_ARGS(__VA_ARGS__)> registry;        \
    return registry;                                                           \
  }                                                                            \
                                                                               \
  shared_ptr<Solver> create_##NAME##Solver(                                    \
      NBLA_ARGDEFS(const Context &NBLA_VA_ARGS(__VA_ARGS__))) {                \
    init_cpu();                                                                \
    return get_##NAME##SolverRegistry().create(                                \
        NBLA_ARGS(const Context &NBLA_VA_ARGS(__VA_ARGS__)));                  \
  }

/**
This will be used inside init method.
*/
#define NBLA_REGISTER_SOLVER_IMPL(BASE, CLS, BACKEND, ...)                     \
  {                                                                            \
    std::function<shared_ptr<Solver>(                                          \
        const Context &NBLA_VA_ARGS(__VA_ARGS__))>                             \
        func = [](NBLA_ARGDEFS(const Context &NBLA_VA_ARGS(__VA_ARGS__))) {    \
          return shared_ptr<Solver>(                                           \
              new CLS(NBLA_ARGS(const Context &NBLA_VA_ARGS(__VA_ARGS__))));   \
        };                                                                     \
    typedef FunctionDbItem<Solver NBLA_VA_ARGS(__VA_ARGS__)> item_t;           \
    get_##BASE##SolverRegistry().add(                                          \
        shared_ptr<item_t>(new item_t{BACKEND, func}));                        \
  }
#else
/**
*/
#define NBLA_REGISTER_SOLVER_HEADER(NAME, ...)                                 \
  NBLA_API FunctionRegistry<Solver, ##__VA_ARGS__>                             \
      &get_##NAME##SolverRegistry();                                           \
                                                                               \
  NBLA_API shared_ptr<Solver> create_##NAME##Solver(                           \
      NBLA_ARGDEFS(const Context &, ##__VA_ARGS__));

#define NBLA_REGISTER_SOLVER_SOURCE(NAME, ...)                                 \
  FunctionRegistry<Solver, ##__VA_ARGS__> &get_##NAME##SolverRegistry() {      \
    static FunctionRegistry<Solver, ##__VA_ARGS__> registry;                   \
    return registry;                                                           \
  }                                                                            \
                                                                               \
  shared_ptr<Solver> create_##NAME##Solver(                                    \
      NBLA_ARGDEFS(const Context &, ##__VA_ARGS__)) {                          \
    init_cpu();                                                                \
    return get_##NAME##SolverRegistry().create(                                \
        NBLA_ARGS(const Context &, ##__VA_ARGS__));                            \
  }

/**
This will be used inside init method.
*/
#define NBLA_REGISTER_SOLVER_IMPL(BASE, CLS, BACKEND, ...)                     \
  {                                                                            \
    std::function<shared_ptr<Solver>(const Context &, ##__VA_ARGS__)> func =   \
        [](NBLA_ARGDEFS(const Context &, ##__VA_ARGS__)) {                     \
          return shared_ptr<Solver>(                                           \
              new CLS(NBLA_ARGS(const Context &, ##__VA_ARGS__)));             \
        };                                                                     \
    typedef FunctionDbItem<Solver, ##__VA_ARGS__> item_t;                      \
    get_##BASE##SolverRegistry().add(                                          \
        shared_ptr<item_t>(new item_t{BACKEND, func}));                        \
  }
#endif
}
#endif
