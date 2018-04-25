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

#ifndef NBLA_UTILS_NNP_IMPL_HPP_
#define NBLA_UTILS_NNP_IMPL_HPP_

// NNabla
#include <nbla/computation_graph/function.hpp>
#include <nbla/computation_graph/variable.hpp>
#include <nbla_utils/nnp.hpp>

// Protobuf
#include <google/protobuf/io/zero_copy_stream.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <google/protobuf/text_format.h>

#include <google/protobuf/descriptor.h>
#include <google/protobuf/message.h>

// HDF5
#ifdef NBLA_UTILS_WITH_HDF5
#include <hdf5.h>
#include <hdf5_hl.h>
#endif

#include "nnabla.pb.h"

#include <memory>
#include <unordered_map>

namespace nbla {
namespace utils {
namespace nnp {

using std::unordered_map;
using std::shared_ptr;
using std::unique_ptr;
using std::string;

// Forward dec
class NnpImpl;

// ----------------------------------------------------------------------
// NetworkImpl
// ----------------------------------------------------------------------
/** Implementation of Netwrok.
 */
class NetworkImpl {
  friend class NnpImpl;

private:
  const nbla::Context ctx_;
  // Network proto
  const ::Network network_proto_;
  unordered_map<string, const ::Variable *> variable_protos_;

  // Parameters
  const unordered_map<string, CgVariablePtr> parameters_;

  // Default batch size
  int batch_size_{-1};

  // True is set when replace_variable is called.
  bool require_build_{true};

  // Replace variable list.
  unordered_map<string, CgVariablePtr> replace_var_list_;

  // Variable map.
  unordered_map<string, CgVariablePtr> variables_;

  // Build a computation graph.
  void build();

  shared_ptr<nbla::CgVariable> get_cgvariable_or_create(const string &name);

  // Create CgFunction from Function message
  shared_ptr<nbla::CgFunction> create_cgfunction(const ::Function &func);

  // ctor
  NetworkImpl(const nbla::Context &ctx, const ::Network &network,
              const unordered_map<string, CgVariablePtr> &parameters);

public:
  string name() const;
  void set_batch_size(int batch_size);
  int batch_size() const;
  void replace_variable(const string &name, CgVariablePtr variable);
  CgVariablePtr get_variable(const string &name);
};

// ----------------------------------------------------------------------
// ExecutorImpl
// ----------------------------------------------------------------------
/** Implementation of Executor
 */
class ExecutorImpl {
  friend class NnpImpl;

private:
  // Executor proto
  const ::Executor executor_proto_;

  // Network
  shared_ptr<Network> network_;

  // Sink variable to execute forward/backard
  CgVariablePtr sink_{nullptr};

  // Get sink output
  void update_sink();

  // ctor
  ExecutorImpl(const ::Executor &executor, shared_ptr<Network> network);

public:
  string name() const;
  string network_name() const;
  void set_batch_size(int batch_size);
  int batch_size() const;
  vector<Executor::DataVariable> get_data_variables();
  vector<Executor::OutputVariable> get_output_variables();
  shared_ptr<Network> get_network();
  void execute();
};

// ----------------------------------------------------------------------
// NnpImpl
// ----------------------------------------------------------------------

/** Implementation of Nnp.
*/
class NnpImpl {
  friend class Nnp;
  const nbla::Context kCpuCtx{{"cpu:float"}, "CpuCachedArray", "0"};
  nbla::Context ctx_;
  unique_ptr<::NNablaProtoBuf> proto_;
  unordered_map<string, CgVariablePtr> parameters_;

  static const int MAX_NAME = 1024;
#ifdef NBLA_UTILS_WITH_HDF5
  hid_t root_;
  bool parse_hdf5_dataset(std::string name, hid_t did);
  bool parse_hdf5_group(hid_t gid);
#endif
  void update_parameters();
  int get_network_repeat_nest_depth(const ::Network &orig);
  std::vector<std::string> create_suffixes(std::string prefix,
                                           std::vector<std::string> ids,
                                           std::vector<int> times);
  std::vector<std::string>
  create_var_suffixes(std::map<std::string, int> repeat_info, ::Variable var);
  std::vector<std::string>
  create_func_suffixes(std::map<std::string, int> repeat_info, ::Function func);
  ::Network expand_network(const ::Network &orig);
  const ::Network &search_network(std::string name);

  NnpImpl(const nbla::Context &ctx);

public:
  ~NnpImpl() {}

  bool add_prototxt(std::string filename);
  bool add_prototxt(char *buffer, int size);
  bool add_protobuf(std::string filename);
  bool add_protobuf(char *buffer, int size);
  bool add_hdf5(char *buffer, int size);
  vector<string> get_network_names();
  shared_ptr<Network> get_network(const string &name);
  vector<string> get_executor_names();
  shared_ptr<Executor> get_executor(const string &name);
};
}
}
}

#endif // __NBLA_UTILS_NNP_IMPL_HPP__
