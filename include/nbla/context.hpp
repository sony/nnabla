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

/** Context
*/
#ifndef __NBLA_CONTEXT_HPP__
#define __NBLA_CONTEXT_HPP__

#include <nbla/defs.hpp>
#include <string>
#include <vector>

namespace nbla {

using std::string;
using std::vector;

/** Context structure

It will be used specifying device and array class etc.
\ingroup NNablaCoreGrp
*/
class Context {
public:
  /** A compute backend descriptor passed to Function/Solver or NdArray class.

     @param[in] backend A vector of backend description. A specific
     implementation of Function/Solver will be queried by each description, the
     first matched one is used. For each element, it describes the backend of
     computation and the data type config in a format of `<backend>:<data type
     config>`. If only `backend` is given (`:<data type configuration>` is
     omitted), the default data type config (`:float`)is automatically added.
     @param[in] array_class Optional: A string expression of a preferred array
     class. Even if it is not specified an array class is chosen according to
     the default array class of an implementation of Function/Solver. Every
     Function/Solver class has a list of array classes that can be used as
     storage of the computation inputs and outputs. If the given array_class
     doesn't match any of them, the default array class of the implementation
     will be used.
     @param[in] device_id A string expression of device ID of the backend.
   */
  explicit NBLA_API Context(const vector<string> &backend = {"cpu:float"},
                            const string &array_class = "CpuArray",
                            const string &device_id = "0");
  vector<string> backend;
  string array_class;
  string device_id;
  Context NBLA_API &set_backend(const vector<string> &backend);
  Context NBLA_API &set_array_class(const string &array_class);
  Context NBLA_API &set_device_id(const string &device_id);
  string to_string() const;
};

/**
*/
inline string get_array_key_from_context(const Context &ctx) {
  return ctx.device_id + ":" + ctx.array_class;
}
}

#endif
