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

/** Context
*/
#ifndef __NBLA_CONTEXT_HPP__
#define __NBLA_CONTEXT_HPP__

#include <nbla/defs.hpp>
#include <string>

namespace nbla {

using std::string;

/** Context structure

It will be used specifying device and array class etc.
\ingroup NNablaCoreGrp
*/
class Context {
public:
  explicit NBLA_API Context(const string &backend = "cpu",
                            const string &array_class = "CpuArray",
                            const string &device_id = "0",
                            const string &compute_backend = "default");
  string backend;
  string array_class;
  string device_id;
  string compute_backend;
  Context NBLA_API &set_backend(const string &backend);
  Context NBLA_API &set_array_class(const string &array_class);
  Context NBLA_API &set_device_id(const string &device_id);
  Context NBLA_API &set_compute_backend(const string &compute_backend);

  bool operator==(const Context &rhs) const;
  bool operator!=(const Context &rhs) const;

};

/**
*/
inline string get_array_key_from_context(const Context &ctx) {
  return ctx.backend + ":" + ctx.device_id + ":" + ctx.array_class;
}
}

#endif
