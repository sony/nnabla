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

#ifdef _WIN32
typedef int ssize_t;
#else
#include <fcntl.h>
#include <unistd.h>
#endif

#include "nnp_impl.hpp"
#include "parameters_impl.hpp"
#if defined(NBLA_UTILS_WITH_NPY)
#include "nnp_impl_dataset_npy.hpp"
#endif
#include "nnp_network_expander.hpp"

#include <nbla/computation_graph/computation_graph.hpp>

#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/text_format.h>

#include <fstream>
#include <iostream>
#include <map>

#include <nbla/function/sink.hpp>
#include <nbla/logger.hpp>

#ifdef _WIN32
#include <fcntl.h>
#include <io.h>
#define open _open
#define O_RDONLY _O_RDONLY
#endif
// Lib archive
#include <archive.h>
#include <archive_entry.h>

namespace nbla {
namespace utils {
namespace nnp {

#include <chrono>
#include <functional>
#include <utility>

// ----------------------------------------------------------------------
// NetworkImpl
// ----------------------------------------------------------------------

NetworkImpl::NetworkImpl(const nbla::Context &ctx, const ::Network &network,
                         const unordered_map<string, CgVariablePtr> &parameters)
    : ctx_(ctx), network_proto_(network), parameters_(parameters) {

  // Create hash map for faster query of proto Variable.
  for (auto it = network_proto_.variable().begin();
       it != network_proto_.variable().end(); it++) {
    variable_protos_.insert({it->name(), &(*it)});
  }
}

void NetworkImpl::build() {
  variables_.clear();
  variables_.insert(parameters_.begin(), parameters_.end());
  for (int i = 0; i < network_proto_.function_size(); i++) {
    ::Function func = network_proto_.function(i);

    NBLA_LOG_INFO("      function name:{} type:{}", func.name(), func.type());

    std::vector<nbla::CgVariablePtr> finputs;
    // std::cout << func.name() << ":";
    for (auto inp = func.input().begin(); inp != func.input().end(); inp++) {
      // std::cout << " " << *inp;
      CgVariablePtr cg_v = get_cgvariable_or_create(*inp);
      finputs.push_back(cg_v);
    }

    auto cgfunc = create_cgfunction(func);

    if (cgfunc.get() == nullptr) {
      using namespace nbla;
      NBLA_ERROR(error_code::not_implemented,
                 "Function [%s] is not supported yet", func.name().c_str());
    }

    // Create a graph connection.
    auto foutputs = nbla::connect(cgfunc, finputs, func.output_size());

    // Insert or rewire.
    // std::cout << " -->";
    for (int j = 0; j < func.output_size(); j++) {
      // std::cout << " " << func.output(j);
      auto it = variables_.find(func.output(j));
      if (it != variables_.end()) {
        // Rewire the output to an existing variable.
        // std::cout << "(r)";
        CgVariablePtr cg_v = get_cgvariable_or_create(func.output(j));
        steal_variable_from_to(foutputs[j], cg_v);
      } else {
        // Register a newly created variable
        // std::cout << "(c)";
        variables_.insert({func.output(j), foutputs[j]});
      }
    }
    // std::cout << std::endl;
  }
}

shared_ptr<nbla::CgVariable>
NetworkImpl::get_cgvariable_or_create(const string &name) {
  // Use variable set by replace_variable
  auto it_r = replace_var_list_.find(name);
  if (it_r != replace_var_list_.end()) {
    return it_r->second;
  }
  // Use variable set by replace_variable
  auto it = variables_.find(name);
  if (it != variables_.end()) {
    return it->second;
  }
  auto var_it = variable_protos_.find(name);
  NBLA_CHECK(var_it != variable_protos_.end(), error_code::value,
             "%s could not be found in variable_protos_. This does not usually "
             "happen.",
             name.c_str());
  const ::Variable *var = var_it->second;
  // Create a new on and register to variables_.
  // Create shape
  nbla::Shape_t shape(var->shape().dim().begin(), var->shape().dim().end());
  if (shape[0] == -1) {
    shape[0] = batch_size();
  }
  // TODO: set need_grad
  // std::cout << "(c)";
  auto cg_v = std::make_shared<nbla::CgVariable>(shape);
  // Register variable
  variables_.insert({name, cg_v});
  return cg_v;
}

void NetworkImpl::replace_variable(const string &name, CgVariablePtr variable) {
  require_build_ = true;
  replace_var_list_.insert({name, variable});
}

CgVariablePtr NetworkImpl::get_variable(const string &name) {
  if (require_build_) {
    build();
    require_build_ = false;
  }
  auto it = variables_.find(name);
  assert(it != variables_.end());
  return it->second;
}

string NetworkImpl::name() const { return network_proto_.name(); }

void NetworkImpl::set_batch_size(int batch_size) {
  if (batch_size_ <= 0 || batch_size_ != batch_size) {
    require_build_ = true;
  }
  batch_size_ = batch_size;
}

int NetworkImpl::batch_size() const {
  if (batch_size_ > 0) {
    return batch_size_;
  }
  assert(network_proto_.batch_size() > 0);
  return network_proto_.batch_size();
}

// ----------------------------------------------------------------------
// ExecutorImpl
// ----------------------------------------------------------------------
ExecutorImpl::ExecutorImpl(const ::Executor &executor,
                           shared_ptr<Network> network)
    : executor_proto_(executor), network_(network) {}

void ExecutorImpl::update_sink() {
  auto outputs = get_output_variables();
  if (outputs.size() == 1) {
    sink_ = outputs[0].variable;
    return;
  }
  vector<CgVariablePtr> inputs;
  for (auto it = outputs.begin(); it != outputs.end(); it++) {
    inputs.push_back(it->variable);
  }
  auto f = make_shared<CgFunction>(create_Sink(nbla::Context(), true));
  sink_ = nbla::connect(f, inputs, 1)[0];
}

string ExecutorImpl::name() const { return executor_proto_.name(); }

string ExecutorImpl::network_name() const {
  return executor_proto_.network_name();
}

void ExecutorImpl::set_batch_size(int batch_size) {
  if (batch_size == network_->batch_size()) {
    return;
  }
  network_->set_batch_size(batch_size);
  sink_ = nullptr;
}

int ExecutorImpl::batch_size() const { return network_->batch_size(); }

vector<Executor::DataVariable> ExecutorImpl::get_data_variables() {
  vector<Executor::DataVariable> ret;
  for (auto it = executor_proto_.data_variable().begin();
       it != executor_proto_.data_variable().end(); it++) {
    Executor::DataVariable v{it->variable_name(), it->data_name(),
                             network_->get_variable(it->variable_name())};
    ret.push_back(v);
  }
  return ret;
}

vector<Executor::OutputVariable> ExecutorImpl::get_output_variables() {
  vector<Executor::OutputVariable> ret;
  for (auto it = executor_proto_.output_variable().begin();
       it != executor_proto_.output_variable().end(); it++) {
    Executor::OutputVariable v{it->variable_name(), it->type(), it->data_name(),
                               network_->get_variable(it->variable_name())};
    ret.push_back(v);
  }
  NBLA_CHECK(ret.size() > 0, error_code::value,
             "Executor `%s`'s output is empty.", name().c_str());
  return ret;
}

shared_ptr<Network> ExecutorImpl::get_network() { return network_; }

void ExecutorImpl::execute() {
  if (sink_ == nullptr) {
    update_sink();
  }
  sink_->forward(true, false);
}

// ----------------------------------------------------------------------
// NnpImpl
// ----------------------------------------------------------------------
NnpImpl::NnpImpl(const nbla::Context &ctx)
    : ctx_(ctx), proto_(new NNablaProtoBuf()) {}

int NnpImpl::get_network_repeat_nest_depth(const ::Network &orig) {
  // get max nest depth.
  int max_nest_depth = -1;
  for (int i = 0; i < orig.function_size(); i++) {
    int depth = orig.function(i).repeat_id_size();
    if (depth > max_nest_depth) {
      std::cerr << "repeat nest depth exceed the maximal depth." << std::endl;
      max_nest_depth = depth;
    }
  }
  return max_nest_depth;
}

std::vector<std::string> NnpImpl::create_suffixes(std::string prefix,
                                                  std::vector<std::string> ids,
                                                  std::vector<int> times) {
  std::vector<std::string> suffixes;
  std::string id = ids.at(0);
  ids.erase(ids.begin());
  int max = times.at(0);
  times.erase(times.begin());
  for (int i = 0; i < max; i++) {
    std::string suffix = prefix + "_" + id + "[" + std::to_string(i) + "]";
    if (ids.size() > 0) {
      auto sub = create_suffixes(suffix, ids, times);
      std::copy(sub.begin(), sub.end(), std::back_inserter(suffixes));
    } else {
      suffixes.push_back(suffix);
    }
  }
  return suffixes;
}

std::vector<std::string>
NnpImpl::create_var_suffixes(std::map<std::string, int> repeat_info,
                             ::Variable var) {
  std::vector<std::string> ids;
  std::vector<int> times;
  for (int j = 0; j < var.repeat_id_size(); j++) {
    auto rid = var.repeat_id(j);
    ids.push_back(rid);
    times.push_back(repeat_info[rid]);
  }
  return create_suffixes("", ids, times);
}

std::vector<std::string>
NnpImpl::create_func_suffixes(std::map<std::string, int> repeat_info,
                              ::Function func) {
  std::vector<std::string> ids;
  std::vector<int> times;
  for (int j = 0; j < func.repeat_id_size(); j++) {
    auto rid = func.repeat_id(j);
    ids.push_back(rid);
    times.push_back(repeat_info[rid]);
  }
  return create_suffixes("", ids, times);
}

::Network NnpImpl::expand_network(const ::Network &orig) {
  auto expander = std::unique_ptr<NetworkExpander>(new NetworkExpander(orig));
  ::Network net = expander->execute();

#ifdef DEBUG_NETWORK_EXPANDER
  dump_proto_network(net);
#endif

  return net;
}

const ::Network &NnpImpl::search_network(std::string name) {
  NBLA_LOG_INFO("    Searching net {}", name);
  for (int i = 0; i < proto_->network_size(); i++) {
    if (proto_->network(i).name() == name) {
      NBLA_LOG_INFO("      Found at {}", i);
      return proto_->network(i);
    }
  }
  static const ::Network null_net;
  return null_net;
}

bool NnpImpl::add_archive(void *archive) {
  struct archive *a = (struct archive *)archive;
  struct archive_entry *entry;
  int r = ARCHIVE_OK;
  while ((r = archive_read_next_header(a, &entry)) == ARCHIVE_OK) {
    ssize_t size = (ssize_t)archive_entry_size(entry);
    char *buffer = new char[size];
    assert(buffer);
    ssize_t read_size = archive_read_data(a, buffer, size);
    if (read_size != size) {
      return false;
    }
    std::string entryname(archive_entry_pathname(entry));

    int ep = entryname.find_last_of(".");
    std::string ext = entryname.substr(ep, entryname.size() - ep);

    if (ext == ".prototxt" || ext == ".nntxt") {
      add_prototxt(buffer, size);
    } else if (ext == ".protobuf") {
      add_protobuf(buffer, size);
    } else if (ext == ".h5") {
      add_hdf5(buffer, size);
    }
    delete[] buffer;
  }
  return true;
}

void NnpImpl::update_parameters() {
  for (auto it = proto_->parameter().begin(); it != proto_->parameter().end();
       it++) {
    const string &name = it->variable_name();
    Shape_t shape(it->shape().dim().begin(), it->shape().dim().end());
    bool need_grad = it->need_grad();
    CgVariablePtr cg_v = std::make_shared<CgVariable>(shape, need_grad);
    float *data =
        cg_v->variable()->template cast_data_and_get_pointer<float>(kCpuCtx);
    auto &p_data = it->data();
    NBLA_CHECK(p_data.size() == cg_v->variable()->size(), error_code::value,
               "Inconsistent size in proto parameter %s (%d != %d)",
               name.c_str(), (int)p_data.size(), (int)cg_v->variable()->size());
    for (int i = 0; i < p_data.size(); i++) {
      data[i] = p_data[i];
    }
    parameters_.insert({name, cg_v});
  }
  proto_->clear_parameter(); // Reset all parameters consumed.
}

bool NnpImpl::add_prototxt(std::string filename) {
  int fd = open(filename.c_str(), O_RDONLY);
  google::protobuf::io::ZeroCopyInputStream *input =
      new google::protobuf::io::FileInputStream(fd);
  google::protobuf::TextFormat::Merge(input, proto_.get());
  delete input;
  close(fd);
  update_parameters();
  return true;
}

bool NnpImpl::add_prototxt(char *buffer, int size) {
  google::protobuf::io::ZeroCopyInputStream *input =
      new google::protobuf::io::ArrayInputStream(buffer, size);
  google::protobuf::TextFormat::Merge(input, proto_.get());
  delete input;
  update_parameters();
  return true;
}

bool NnpImpl::add_protobuf(std::string filename) {
  ParameterVector pv;
  bool ret = load_parameters_pb(pv, filename);
  if (!ret) {
    NBLA_ERROR(error_code::value, "Cannot load parameter file: %s",
               filename.c_str());
  }
  for (auto it = pv.begin(); it != pv.end(); ++it) {
    parameters_.insert({it->first, it->second});
  }
  return true;
}

bool NnpImpl::add_protobuf(char *buffer, int size) {
  ParameterVector pv;
  bool ret = load_parameters_pb(pv, buffer, size);
  if (!ret) {
    NBLA_ERROR(error_code::value, "Cannot load parameters from buffer.");
  }
  for (auto it = pv.begin(); it != pv.end(); ++it) {
    parameters_.insert({it->first, it->second});
  }
  return true;
}

bool NnpImpl::add_hdf5(char *buffer, int size) {
  ParameterVector pv;
  bool ret = load_parameters_h5(pv, buffer, size);
  if (!ret) {
    NBLA_ERROR(error_code::value, "Cannot load parameter from buffer.");
  }
  for (auto it = pv.begin(); it != pv.end(); ++it) {
    parameters_.insert({it->first, it->second});
  }
  return true;
}

vector<string> NnpImpl::get_network_names() {
  vector<string> list;
  for (int i = 0; i < proto_->network_size(); i++) {
    list.push_back(proto_->network(i).name());
  }
  return list;
}

shared_ptr<Network> NnpImpl::get_network(const string &name) {
  // Find network proto
  const ::Network &orig_network = search_network(name);
  NBLA_CHECK(orig_network.name() != "", error_code::value,
             "Network '%s' not found in proto", name.c_str());
  ::Network network = expand_network(orig_network);
  // Filter parameters
  unordered_map<string, CgVariablePtr> parameters;
  for (auto it = network.variable().begin(); it != network.variable().end();
       it++) {
    auto found = parameters_.find(it->name());
    if (found == parameters_.end()) {
      continue;
    }
    NBLA_LOG_INFO("Initial data of {} was found.", it->name());
    parameters.insert({found->first, found->second});
  }
  return shared_ptr<Network>(
      new Network(new NetworkImpl(ctx_, network, parameters)));
}

vector<string> NnpImpl::get_executor_names() {
  vector<string> list;
  for (auto it = proto_->executor().begin(); it != proto_->executor().end();
       it++) {
    list.push_back(it->name());
  }
  return list;
}

shared_ptr<Executor> NnpImpl::get_executor(const string &name) {
  for (auto it = proto_->executor().begin(); it != proto_->executor().end();
       it++) {
    if (it->name() != name) {
      continue;
    }
    return shared_ptr<Executor>(
        new Executor(new ExecutorImpl(*it, get_network(it->network_name()))));
  }
  NBLA_ERROR(error_code::value, "Executor `%s` not found from [%s].",
             name.c_str(),
             string_join(this->get_executor_names(), ", ").c_str());
}

vector<pair<string, VariablePtr>> NnpImpl::get_parameters() {
  vector<pair<string, VariablePtr>> parameters;
  for (auto it = parameters_.begin(); it != parameters_.end(); it++) {
    std::pair<string, VariablePtr> psv;
    psv.first = it->first;
    psv.second = it->second->variable();
    parameters.push_back(psv);
  }
  return parameters;
}

bool NnpImpl::save_parameters(const string &filename) {
  ParameterVector pv;
  for (auto it = parameters_.begin(); it != parameters_.end(); it++) {
    pv.push_back({it->first, it->second});
  }
  return nbla::utils::save_parameters(pv, filename);
}

vector<string> NnpImpl::get_optimizer_names() {
  vector<string> list;
  for (auto it = proto_->optimizer().begin(); it != proto_->optimizer().end();
       it++) {
    list.push_back(it->name());
  }
  return list;
}

shared_ptr<Optimizer> NnpImpl::get_optimizer(const string &name) {
  for (auto it = proto_->optimizer().begin(); it != proto_->optimizer().end();
       it++) {
    if (it->name() != name) {
      continue;
    }
    if (it->dataset_name_size() != 1) {
      NBLA_ERROR(error_code::value, "Currently only one dataset supported.");
    }
    return shared_ptr<Optimizer>(new Optimizer(
        new OptimizerImpl(ctx_, *it, get_network(it->network_name()),
                          get_dataset(it->dataset_name()[0]))));
  }
  NBLA_ERROR(error_code::value, "Optimizer `%s` not found", name.c_str());
}

vector<string> NnpImpl::get_dataset_names() {
  vector<string> list;
  for (auto it = proto_->dataset().begin(); it != proto_->dataset().end();
       it++) {
    list.push_back(it->name());
  }
  return list;
}

shared_ptr<DatasetImpl> NnpImpl::get_dataset(const string &name) {
  for (auto it = proto_->dataset().begin(); it != proto_->dataset().end();
       it++) {
    if (it->name() != name) {
      continue;
    }

#if defined(NBLA_UTILS_WITH_NPY)
    // Npy Support
    return shared_ptr<DatasetNpyCache>(new DatasetNpyCache(*it));
#elif defined(NBLA_UTILS_WITH_HDF5)
    // HDF5 Support
    return shared_ptr<DatasetHDF5Impl>(new DatasetHDF5Impl(*it));
#else
#warning("No cache file format is defined.");
#endif
  }
  NBLA_ERROR(error_code::value, "Dataset `%s` not found", name.c_str());
}

vector<string> NnpImpl::get_monitor_names() {
  vector<string> list;
  for (auto it = proto_->monitor().begin(); it != proto_->monitor().end();
       it++) {
    list.push_back(it->name());
  }
  return list;
}

shared_ptr<Monitor> NnpImpl::get_monitor(const string &name) {
  for (auto it = proto_->monitor().begin(); it != proto_->monitor().end();
       it++) {
    if (it->name() != name) {
      continue;
    }
    if (it->dataset_name_size() != 1) {
      NBLA_ERROR(error_code::value, "Currently only one dataset supported.");
    }
    return shared_ptr<Monitor>(
        new Monitor(new MonitorImpl(ctx_, *it, get_network(it->network_name()),
                                    get_dataset(it->dataset_name()[0]))));
  }
  NBLA_ERROR(error_code::value, "Monitor `%s` not found", name.c_str());
}

shared_ptr<TrainingConfig> NnpImpl::get_training_config() {
  return shared_ptr<TrainingConfig>(
      new TrainingConfig(new TrainingConfigImpl(proto_->training_config())));
}
}
}
}
