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

#ifndef __NBLA_HOST_STREAM_SYNCHRONIZER_REGISTRY_HPP__
#define __NBLA_HOST_STREAM_SYNCHRONIZER_REGISTRY_HPP__

#include <nbla/defs.hpp>
#include <nbla/context.hpp>

#include <string>
#include <sstream>
#include <istream>
#include <functional>
#include <unordered_map>

namespace nbla {
  
using std::string;
using std::stringstream;
using std::getline;
using std::unordered_map;

/** HostStreamSynchronizer class

This class is never be instantiated.
*/
class NBLA_API HostStreamSynchronizer {
public:
  typedef std::function<void(void)> Synchronizer;
  typedef unordered_map<string, Synchronizer> Registry_t;

  /** Synchronize host to a stream
  */
  static void synchronize(const Context ctx);

  /** Register new synchronizer
  */
  static void add_synchronizer(const string& backend,
                               Synchronizer synchronizer);

private:
  //  Never be created
  inline HostStreamSynchronizer() {}

  /** Get registry of creator function.
  */
  static Registry_t &get_registry();
};

#define NBLA_REGISTER_HOST_STREAM_SYNCHRONIZER(BACKEND, SYNCHRONIZER)   \
  { HostStreamSynchronizer::add_synchronizer(#BACKEND, SYNCHRONIZER); }
}
#endif