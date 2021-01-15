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

#include <nbla/backend_registry.hpp>
#include <nbla/common.hpp>

namespace nbla {

using std::stringstream;
using std::getline;

void BackendUtils::add_backend(const string &backend_name,
                               BackendGetter backend_getter) {
  Registry_t &registry = get_registry();
  string key{backend_name};
  registry[key] = backend_getter;
}

BackendUtils::Registry_t &BackendUtils::get_registry() {
  static Registry_t registry_;
  return registry_;
}

string get_key(const Context ctx) {
  // Extract device name e.g. "cpu" from "cpu:float" in context.
  if (ctx.backend.empty()) {
    NBLA_ERROR(error_code::unclassified, "Backend is empty.");
  }
  stringstream ss{ctx.backend[0]};
  string key;
  getline(ss, key, ':');

  return key;
}

BackendUtils::BackendGetter
BackendUtils::get_backend_getter(const Context ctx) {
  init_cpu();
  Registry_t &registry = get_registry();
  auto key = get_key(ctx);

  NBLA_CHECK(registry.count(key) == 1, error_code::unclassified,
             "'%s' cannot be found in host stream synchronizer.", key.c_str());

  return registry[key];
}

vector<string> BackendUtils::array_classes(const Context ctx) {
  return get_backend_getter(ctx)()->array_classes();
}

void BackendUtils::_set_array_classes(const Context ctx,
                                      const vector<string> &a) {
  return get_backend_getter(ctx)()->_set_array_classes(a);
}

void BackendUtils::register_array_class(const Context ctx, const string &name) {
  return get_backend_getter(ctx)()->register_array_class(name);
}

shared_ptr<Allocator> BackendUtils::caching_allocator(const Context ctx) {
  return get_backend_getter(ctx)()->caching_allocator();
}

shared_ptr<Allocator> BackendUtils::naive_allocator(const Context ctx) {
  return get_backend_getter(ctx)()->naive_allocator();
}

void BackendUtils::free_unused_host_caches(const Context ctx) {
  return get_backend_getter(ctx)()->free_unused_host_caches();
}

void BackendUtils::device_synchronize(const Context ctx) {
  return get_backend_getter(ctx)()->device_synchronize(ctx.device_id);
}

void BackendUtils::default_stream_synchronize(const Context ctx) {
  return get_backend_getter(ctx)()->default_stream_synchronize(ctx.device_id);
}

void BackendUtils::create_lms_streams(const Context ctx) {
  return get_backend_getter(ctx)()->create_lms_streams(
      std::stoi(ctx.device_id));
}
}
