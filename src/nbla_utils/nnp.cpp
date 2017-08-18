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
#include <string>
#include <vector>

#include <fcntl.h>
#include <unistd.h>

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

int Network::batch_size() { return impl_->batch_size(); }

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
    struct archive_entry *entry;
    int r = ARCHIVE_OK;
    r = archive_read_open_filename(a, filename.c_str(), 10240);
    assert(r == ARCHIVE_OK);

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
        impl_->add_prototxt(buffer, size);
      } else if (ext == ".protobuf") {
        impl_->add_protobuf(buffer, size);
      } else if (ext == ".h5") {
        impl_->add_hdf5(buffer, size);
      }
      free(buffer);
    }
    archive_read_free(a);
    return true;
  } else {
    // TODO: unsupported
  }

  return false;
}

vector<string> Nnp::get_network_names() { return impl_->get_network_names(); }

shared_ptr<Network> Nnp::get_network(const string &name) {
  return impl_->get_network(name);
}

vector<string> Nnp::get_executor_names() { return impl_->get_executor_names(); }

shared_ptr<Executor> Nnp::get_executor(const string &name) {
  return impl_->get_executor(name);
}
}
}
}
