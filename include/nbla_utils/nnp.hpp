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

#ifndef NBLA_UTILS_NNP_HPP_
#define NBLA_UTILS_NNP_HPP_

#include <nbla/computation_graph/variable.hpp>
#include <string>

namespace nbla {
namespace utils {
namespace nnp {

// Forward dec.
class NnpImpl;
class NetworkImpl;

class Network {
  friend NnpImpl;
  std::unique_ptr<NetworkImpl> impl_;
  Network(NetworkImpl *impl);

public:
  /** Network name.
   */
  string name() const;

  /** Set batch size.
  */
  void set_batch_size(int batch_size);

  /** Get batch size.
  */
  int batch_size();

  /** Replace an arbitrary variable with name in the network with a given
      variable.

      The predecessors of the variable in the networks are dicarded, and
      replaced with the predecessors of the given variable.
   */
  void replace_variable(const string &name, CgVariablePtr variable);

  /** Get a variable by name.

      This is usually used to set or get data inside the variable.

      If requires_build_ is true, build() is called before returns the
      variable.
   */
  CgVariablePtr get_variable(const string &name);
};

class Nnp {
  std::unique_ptr<NnpImpl> impl_;

public:
  // ctor
  Nnp(const nbla::Context &ctx);
  // dtor
  ~Nnp();

  /** Add nnp|nntxt|h5 file
   */
  bool add(const string &filename);

  /** Get NetworkBuilder object associated with a network with specified name.
   */
  shared_ptr<Network> get_network(const string &name);
};
}
}
}

#endif // NBLA_UTILS_NNP_HPP_
