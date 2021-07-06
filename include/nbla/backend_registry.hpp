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

#ifndef __NBLA_HOST_STREAM_SYNCHRONIZER_REGISTRY_HPP__
#define __NBLA_HOST_STREAM_SYNCHRONIZER_REGISTRY_HPP__

#include <nbla/backend_base.hpp>
#include <nbla/context.hpp>
#include <nbla/defs.hpp>

#include <functional>
#include <istream>
#include <memory>
#include <sstream>
#include <string>
#include <unordered_map>

namespace nbla {

using std::string;
using std::unordered_map;
using std::shared_ptr;

/** BackendUtils class

This class is never be instantiated.
 */
class NBLA_API BackendUtils {
public:
  typedef std::function<BackendBase *(void)> BackendGetter;
  typedef unordered_map<string, BackendGetter> Registry_t;

  /** Register new synchronizer
   */
  static void add_backend(const string &backend_name,
                          BackendGetter backend_getter);

  /** Call array_classes of the backend in a context.
   */
  static vector<string> array_classes(const Context ctx);

  /** Call _set_array_classes of the backend in a context.

  @note Dangerous to call. End users shouldn't call.
   */
  static void _set_array_classes(const Context ctx, const vector<string> &a);

  /** Call register_array_class of the backend in a context.
  */
  static void register_array_class(const Context ctx, const string &name);

  /** Call caching_allocator of the backend in a context.
   */
  static shared_ptr<Allocator> caching_allocator(const Context ctx);

  /** Call naive_allocator of the backend in a context.
   */
  static shared_ptr<Allocator> naive_allocator(const Context ctx);

  /** Free all unused host memory caches
   */
  static void free_unused_host_caches(const Context ctx);

  /** Call device_synchronize of the backend in a context.
   */
  static void device_synchronize(const Context ctx);

  /** Synchronize host to default stream of device.
   */
  static void default_stream_synchronize(const Context ctx);

  /** Call create_lms_streams of the backend in a context.
   */
  static void create_lms_streams(const Context ctx);

private:
  //  Never be created
  inline BackendUtils() {}

  /** Get registry of creator function.
   */
  static Registry_t &get_registry();

  /** Get backend getter.
  */
  static BackendGetter get_backend_getter(const Context ctx);
};

#define NBLA_REGISTER_BACKEND(BACKEND_NAME, BACKEND_GETTER)                    \
  { BackendUtils::add_backend(BACKEND_NAME, BACKEND_GETTER); }
}
#endif
