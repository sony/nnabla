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

#ifndef H_NBLA_UTILS_HPP_170720113455_
#define H_NBLA_UTILS_HPP_170720113455_

#include <nbla/computation_graph/variable.hpp>
#include <string>

namespace nbla_utils {
namespace NNP {

///
/// Internal class for process protobuf data.
///
class _proto_internal;

///
/// nnp
///
/// Load network and parameter from supported format to
/// std::vector<nbla::CgVariablePtr>.
///
class nnp {
protected:
  ///
  /// Internal information placeholder.
  ///
  _proto_internal *_proto;

public:
  nnp(nbla::Context &ctx);
  ~nnp();

  void set_batch_size(int batch_size);
  int get_batch_size();

  bool add(std::string filename);

  int num_of_executors();
  std::vector<std::string> get_executor_input_names(int index);
  std::vector<nbla::CgVariablePtr> get_executor_input_variables(int index);
  std::vector<nbla::CgVariablePtr>
  get_executor(int index, std::vector<nbla::CgVariablePtr> inputs);
};
};
};

#endif // H_NBLA_UTILS_HPP_170720113455_
