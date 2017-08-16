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

#ifndef H_PROTOBUF_INTERNAL_HPP_170721100717_
#define H_PROTOBUF_INTERNAL_HPP_170721100717_

// NNabla
#include <nbla/computation_graph/function.hpp>
#include <nbla/computation_graph/variable.hpp>

// Protobuf
#include <google/protobuf/io/zero_copy_stream.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <google/protobuf/text_format.h>

#include <google/protobuf/descriptor.h>
#include <google/protobuf/message.h>

// HDF5
#include <hdf5.h>
#include <hdf5_hl.h>

#include "nnabla.pb.h"

namespace nbla_utils {
namespace NNP {
class _proto_internal {
protected:
  NNablaProtoBuf *_nbla;

  static const int MAX_NAME = 1024;
  hid_t root;
  bool parse_hdf5_dataset(std::string name, hid_t did);
  bool parse_hdf5_group(hid_t gid);

  nbla::Context &_ctx;

  shared_ptr<nbla::CgFunction> create_cgfunction(const Function &func);
  shared_ptr<nbla::CgVariable> create_cgvariable(const Network &network,
                                                 std::string name);
  shared_ptr<nbla::CgVariable> create_cgvariable(const Variable &var);

public:
  int _batch_size;

  Network expand_network(const Network &orig);
  const Network &search_network(std::string name);
  const Parameter &search_parameter(std::string name);
  const Variable &search_variable(const Network &network, std::string name);

  _proto_internal(nbla::Context &ctx) : _ctx(ctx) {
    _batch_size = -1;
    _nbla = new NNablaProtoBuf();
  }
  ~_proto_internal() { delete _nbla; }

  bool add_prototxt(std::string filename);
  bool add_prototxt(char *buffer, int size);

  bool add_protobuf(std::string filename);
  bool add_protobuf(char *buffer, int size);

  bool add_hdf5(char *buffer, int size);

  int num_of_executors();
  std::vector<std::string> get_executor_input_names(int index);
  std::vector<nbla::CgVariablePtr> get_executor_input_variables(int index);
  std::vector<nbla::CgVariablePtr>
  get_executor(int index, std::vector<nbla::CgVariablePtr> inputs);
};
}
}

#endif // H_PROTOBUF_INTERNAL_HPP_170721100717_
