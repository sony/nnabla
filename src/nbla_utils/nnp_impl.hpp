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
#include <nbla/solver.hpp>
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
/** Implementation of Network.
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

  // Sink variable to execute forward/backward
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
// DatasetImpl
// ----------------------------------------------------------------------
/** Implementation of Dataset
*/
class DatasetImpl {
  friend class NnpImpl;

private:
  // Dataset proto
  const ::Dataset dataset_proto_;

  // ctor
  DatasetImpl(const ::Dataset &dataset);

  // Number of data
  int n_data_;

  // Number of stream
  int n_stream_;

  // Data shapes
  vector<Shape_t> shapes_;

  // Data names
  vector<string> data_names_;

  // Cache blocks
  vector<vector<NdArrayPtr>> cache_blocks_; // cache_block/data_column/data

public:
  string name() const;
  string uri() const;
  string cache_dir() const;
  bool create_cache_explicitly() const;
  bool overwrite_cache() const;
  bool shuffle() const;
  bool no_image_normalization() const;
  const int batch_size() const;
  const int get_num_stream() const;
  const int get_num_data() const;
  vector<string> get_data_names();
  vector<Shape_t> get_shapes();
  vector<vector<NdArrayPtr>> get_cache_blocks();
};

class DataIteratorFromCacheFiles {
public:
  // ctor
  DataIteratorFromCacheFiles(shared_ptr<DatasetImpl> dataset);
  ~DataIteratorFromCacheFiles();

private:
  vector<string> data_names_;
  vector<int> shuffle_ids_;
  vector<NdArrayPtr> dataset_; // data_column/data

  int n_data_;
  int batch_size_;
  bool shuffle_;
  int current_id_;

  void shuffle_index();
  void shuffle_dataset();

public:
  const vector<string> get_data_names() const;
  const int get_batch_size() const;
  const int get_iter_per_epoch() const;
  unordered_map<string, NdArrayPtr> next();
};

// ----------------------------------------------------------------------
// OptimizerImpl
// ----------------------------------------------------------------------
/** Implementation of Optimizer
*/
class OptimizerImpl {
  friend class NnpImpl;

private:
  // Context
  const nbla::Context ctx_;

  // Optimizer proto
  const ::Optimizer optimizer_proto_;

  // Network
  shared_ptr<Network> network_;

  // Dataset
  shared_ptr<DatasetImpl> dataset_;

  // DataIterator
  shared_ptr<DataIteratorFromCacheFiles> data_iterator_;

  // Create Solver from Solver message
  shared_ptr<Solver> solver_;
  shared_ptr<nbla::Solver> create_solver(const ::Solver &solver);

  // ctor
  OptimizerImpl(const nbla::Context &ctx, const ::Optimizer &optimizer,
                shared_ptr<Network> network, shared_ptr<DatasetImpl> dataset);

public:
  string name() const;
  string network_name() const;
  string dataset_name() const;
  const int update_interval() const;

  vector<Optimizer::DataVariable> get_data_variables();
  vector<Optimizer::GeneratorVariable> get_generator_variables();
  vector<Optimizer::LossVariable> get_loss_variables();
  vector<Optimizer::ParameterVariable> get_parameter_variables();
  shared_ptr<Network> get_network();

  string type() const;
  // Context context() const;
  float weight_decay_rate() const;
  float lr_decay() const;
  long long int lr_decay_interval() const;

  void set_parameters(const vector<Optimizer::ParameterVariable> &params,
                      bool reset = true, bool retain_state = false);
  void zero_grad();
  void update_parameters();
  float learning_rate() const;
  void set_learning_rate(float learning_rate);
  void weight_decay(float decay_rate);
  void remove_parameters(const vector<string> &keys);
  void clear_parameters();

  const float update(const int iter);
};

// ----------------------------------------------------------------------
// MonitorImpl
// ----------------------------------------------------------------------
/** Implementation of Monitor
*/
class MonitorImpl {
  friend class NnpImpl;

private:
  // Context
  const nbla::Context ctx_;

  // Monitor proto
  const ::Monitor monitor_proto_;

  // Network
  shared_ptr<Network> network_;

  // Dataset
  shared_ptr<DatasetImpl> dataset_;

  // DataIterator
  shared_ptr<DataIteratorFromCacheFiles> data_iterator_;

  // ctor
  MonitorImpl(const nbla::Context &ctx, const ::Monitor &monitor,
              shared_ptr<Network> network, shared_ptr<DatasetImpl> dataset);

public:
  string name() const;
  string network_name() const;
  string dataset_name() const;

  vector<Monitor::DataVariable> get_data_variables();
  vector<Monitor::MonitorVariable> get_monitor_variables();
  shared_ptr<Network> get_network();
  void monitor(const nbla::Context &ctx);

  const float monitor_epoch();
};

// ----------------------------------------------------------------------
// GlobalConfig
// ----------------------------------------------------------------------
/** Implementation of GlobalConfig
*/
class GlobalConfigImpl {
  friend class NnpImpl;

private:
  // GlobalConfig proto
  const ::GlobalConfig global_config_proto_;

  // Create Context from Context message
  shared_ptr<nbla::Context> default_context_;
  shared_ptr<nbla::Context> create_context(const ::Context &ctx);

  // ctor
  GlobalConfigImpl(const ::GlobalConfig &global_config);

public:
  shared_ptr<nbla::Context> default_context();
};

// ----------------------------------------------------------------------
// TrainingConfig
// ----------------------------------------------------------------------
class TrainingConfigImpl {
  friend class NnpImpl;

private:
  // TrainingConfig proto
  const ::TrainingConfig training_config_proto_;

  // ctor
  TrainingConfigImpl(const ::TrainingConfig &training_config);

public:
  const long long int max_epoch() const;
  const long long int iter_per_epoch() const;
  const bool save_best() const;
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

  bool add_archive(void *archive);
  bool add_prototxt(std::string filename);
  bool add_prototxt(char *buffer, int size);
  bool add_protobuf(std::string filename);
  bool add_protobuf(char *buffer, int size);
  bool add_hdf5(char *buffer, int size);
  vector<string> get_network_names();
  shared_ptr<Network> get_network(const string &name);
  vector<string> get_executor_names();
  shared_ptr<Executor> get_executor(const string &name);
  vector<pair<string, VariablePtr>> get_parameters();
  bool save_parameters(const string &filename);

  vector<string> get_dataset_names();
  vector<string> get_optimizer_names();
  vector<string> get_monitor_names();
  shared_ptr<DatasetImpl> get_dataset(const string &name);
  shared_ptr<Optimizer> get_optimizer(const string &name);
  shared_ptr<Monitor> get_monitor(const string &name);
  shared_ptr<TrainingConfig> get_training_config();
};
}
}
}

#endif // __NBLA_UTILS_NNP_IMPL_HPP__
