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

#ifndef __NBLA_INIT_HPP__
#define __NBLA_INIT_HPP__

#include <nbla/defs.hpp>

#include <string>
#include <vector>

namespace nbla {

using std::string;
using std::vector;

/**
Initialize NNabla CPU features.

User usually does not need to call this function manually.
\ingroup NNablaCoreGrp
*/
NBLA_API void init_cpu();

/**
   Clear CPU memory cache.
 */
NBLA_API void clear_cpu_memory_cache();

/**
 * Print CPU memory cache map.
 */

NBLA_API void print_cpu_memory_cache_map();

/** Get CPU array classes.
*/
NBLA_API vector<string> cpu_array_classes();

/** Set CPU array classes
*/
NBLA_API void _cpu_set_array_classes(const vector<string> &a);

NBLA_API void cpu_device_synchronize(const string &device);

NBLA_API int cpu_get_device_count();

NBLA_API vector<string> cpu_get_devices();
}
#endif
