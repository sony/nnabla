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

#include <nbla/logger.hpp>
#include <nbla_utils/nnp.hpp>

#include <string>

#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "nnp_impl.hpp"

#include <archive.h>
#include <archive_entry.h>

namespace nbla {
namespace utils {
namespace nnp {

// ----------------------------------------------------------------------
// Network
// ----------------------------------------------------------------------
Network::Network(NetworkImpl *impl)
    : impl_(std::unique_ptr<NetworkImpl>(impl)) {}

void Network::replace_variable(const string &name, CgVariablePtr variable) {
  impl_->replace_variable(name, variable);
}

CgVariablePtr Network::get_variable(const string &name) {
  return impl_->get_variable(name);
}

string Network::name() const { return impl_->name(); }

void Network::set_batch_size(int batch_size) {
  impl_->set_batch_size(batch_size);
}

int Network::batch_size() const { return impl_->batch_size(); }

// ----------------------------------------------------------------------
// Executor
// ----------------------------------------------------------------------
Executor::Executor(ExecutorImpl *impl)
    : impl_(std::unique_ptr<ExecutorImpl>(impl)) {}
string Executor::name() const { return impl_->name(); }
string Executor::network_name() const { return impl_->network_name(); }
void Executor::set_batch_size(int batch_size) {
  impl_->set_batch_size(batch_size);
}
int Executor::batch_size() const { return impl_->batch_size(); }
vector<Executor::DataVariable> Executor::get_data_variables() {
  return impl_->get_data_variables();
}
vector<Executor::OutputVariable> Executor::get_output_variables() {
  return impl_->get_output_variables();
}
shared_ptr<Network> Executor::get_network() { return impl_->get_network(); }
void Executor::execute() { impl_->execute(); }

// ----------------------------------------------------------------------
// Optimizer
// ----------------------------------------------------------------------
Optimizer::Optimizer(OptimizerImpl *impl)
    : impl_(std::unique_ptr<OptimizerImpl>(impl)) {}

string Optimizer::name() const { return impl_->name(); }
string Optimizer::network_name() const { return impl_->network_name(); }
const int Optimizer::update_interval() const {
  return impl_->update_interval();
}
shared_ptr<Network> Optimizer::get_network() { return impl_->get_network(); }
const float Optimizer::update(const int iter) { return impl_->update(iter); }

// ----------------------------------------------------------------------
// Monitor
// ----------------------------------------------------------------------
Monitor::Monitor(MonitorImpl *impl)
    : impl_(std::unique_ptr<MonitorImpl>(impl)) {}

string Monitor::name() const { return impl_->name(); }
string Monitor::network_name() const { return impl_->network_name(); }
shared_ptr<Network> Monitor::get_network() { return impl_->get_network(); }
const float Monitor::monitor_epoch() { return impl_->monitor_epoch(); }

// ----------------------------------------------------------------------
// TrainingConfig
// ----------------------------------------------------------------------
TrainingConfig::TrainingConfig(TrainingConfigImpl *impl)
    : impl_(std::unique_ptr<TrainingConfigImpl>(impl)) {}

const long long int TrainingConfig::max_epoch() const {
  return impl_->max_epoch();
}

const long long int TrainingConfig::iter_per_epoch() const {
  return impl_->iter_per_epoch();
}

const bool TrainingConfig::save_best() const { return impl_->save_best(); }

// ----------------------------------------------------------------------
// Nnp
// ----------------------------------------------------------------------
Nnp::Nnp(const nbla::Context &ctx) : impl_(new NnpImpl(ctx)) {
  GOOGLE_PROTOBUF_VERIFY_VERSION;
}

Nnp::~Nnp() {}

bool Nnp::add(const string &filename) {
  int ep = filename.find_last_of(".");
  std::string extname = filename.substr(ep, filename.size() - ep);

  if (extname == ".prototxt" || extname == ".nntxt") {
    return impl_->add_prototxt(filename);
  } else if (extname == ".protobuf") {
    return impl_->add_protobuf(filename);
  } else if (extname == ".h5") {
    std::ifstream file(filename.c_str(), std::ios::binary | std::ios::ate);
    std::streamsize size = file.tellg();
    file.seekg(0, std::ios::beg);
    std::vector<char> buffer(size);
    if (file.read(buffer.data(), size)) {
      return impl_->add_hdf5(buffer.data(), size);
    }
  } else if (extname == ".nnp") {
    struct archive *a = archive_read_new();
    assert(a);
    archive_read_support_format_zip(a);
    int r = ARCHIVE_OK;
    r = archive_read_open_filename(a, filename.c_str(), 10240);
    assert(r == ARCHIVE_OK);
    if (r != ARCHIVE_OK) {
      return false;
    }
    bool ret = impl_->add_archive(a);
    archive_read_free(a);
    return ret;
  } else {
    std::cerr << "Error: No available file." << std::endl;
    return false;
  }

  return false;
}

bool Nnp::add(char *buffer, unsigned int size) {
  struct archive *a = archive_read_new();
  assert(a);
  archive_read_support_format_zip(a);
  int r = ARCHIVE_OK;
  r = archive_read_open_memory(a, buffer, size);
  if (r != ARCHIVE_OK) {
    return false;
  }
  bool ret = impl_->add_archive(a);
  archive_read_free(a);
  return ret;
}

vector<string> Nnp::get_network_names() { return impl_->get_network_names(); }

shared_ptr<Network> Nnp::get_network(const string &name) {
  return impl_->get_network(name);
}

vector<string> Nnp::get_executor_names() { return impl_->get_executor_names(); }

shared_ptr<Executor> Nnp::get_executor(const string &name) {
  return impl_->get_executor(name);
}

vector<pair<string, VariablePtr>> Nnp::get_parameters() {
  return impl_->get_parameters();
}

bool Nnp::save_parameters(const string &filename) {
  return impl_->save_parameters(filename);
}

vector<string> Nnp::get_optimizer_names() {
  return impl_->get_optimizer_names();
}

shared_ptr<Optimizer> Nnp::get_optimizer(const string &name) {
  return impl_->get_optimizer(name);
}

vector<string> Nnp::get_monitor_names() { return impl_->get_monitor_names(); }

shared_ptr<Monitor> Nnp::get_monitor(const string &name) {
  return impl_->get_monitor(name);
}

shared_ptr<TrainingConfig> Nnp::get_training_config() {
  return impl_->get_training_config();
}
}
}
}
