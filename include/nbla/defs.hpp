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

#ifndef __NBLA_DEFS_HPP__
#define __NBLA_DEFS_HPP__
// For windows support
#if defined(_MSC_VER) && !defined(__CUDACC__)
#if defined(nnabla_EXPORTS) || defined(nnabla_dbg_EXPORTS) ||                  \
    defined(nnabla_utils_EXPORTS) || defined(nnabla_utils_dbg_EXPORTS) ||      \
    defined(nnabla_cli_EXPORTS) || defined(nnabla_cli_dbg_EXPORTS)
#define NBLA_API __declspec(dllexport)
#else
#define NBLA_API __declspec(dllimport)
#endif
#else
#define NBLA_API
#endif

// C++11 gives alignas as standar
// http://en.cppreference.com/w/cpp/language/alignas
#define NBLA_ALIGN(N) alignas(N)

// Helper macro to get this class type
// To use this, <type_traits> must be included before using it.
#define NBLA_THIS_TYPE std::remove_pointer<decltype(this)>::type
#endif
