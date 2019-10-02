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

#include <nbla/host_stream_synchronizer_registry.hpp>
#include <nbla/common.hpp>

namespace nbla {

void HostStreamSynchronizer::synchronize(const Context ctx) {
  init_cpu();
  Registry_t &registry = get_registry();

  // Extract device name e.g. "cpu" from "cpu:float" in context.
  if (ctx.backend.empty()) {
    NBLA_ERROR(error_code::unclassified, "Backend is empty.");
  }
  stringstream ss{ ctx.backend[0] };
  string key;
  getline(ss, key, ':');

  NBLA_CHECK(registry.count(key) == 1, error_code::unclassified,
    "'%s' cannot be found in host stream synchronizer.", key);

  registry[key]();
}

void HostStreamSynchronizer::add_synchronizer(const string &backend,
  Synchronizer synchronizer) {
  Registry_t &registry = get_registry();
  string key{ backend };
  registry[key] = synchronizer;
}

HostStreamSynchronizer::Registry_t &HostStreamSynchronizer::get_registry() {
  static Registry_t registry_;
  return registry_;
}
}