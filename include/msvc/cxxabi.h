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

#ifndef H_CXXABI_H_
#define H_CXXABI_H_

/**
   This roughly simulates a behavior of GNU cxxabi.h used in src/nbla_cli by an
   external library `cmdline`.
 */

namespace abi {
char *__cxa_demangle(const char *s, int o, int o2, int *status);
}

#endif // H_CXXABI_H_
