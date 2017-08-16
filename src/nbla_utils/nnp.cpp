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

#include "protobuf_internal.hpp"

#include <archive.h>
#include <archive_entry.h>

namespace nbla_utils {
namespace NNP {
nnp::nnp(nbla::Context &ctx) {
  GOOGLE_PROTOBUF_VERIFY_VERSION;
  _proto = new _proto_internal(ctx);
}

nnp::~nnp() { delete _proto; }

bool nnp::add(std::string filename) {
  int ep = filename.find_last_of(".");
  std::string extname = filename.substr(ep, filename.size() - ep);

  if (extname == ".prototxt" || extname == ".nntxt") {
    return _proto->add_prototxt(filename);
  } else if (extname == ".protobuf") {
    return _proto->add_protobuf(filename);
  } else if (extname == ".h5") {
    std::ifstream file(filename.c_str(), std::ios::binary | std::ios::ate);
    std::streamsize size = file.tellg();
    file.seekg(0, std::ios::beg);
    std::vector<char> buffer(size);
    if (file.read(buffer.data(), size)) {
      return _proto->add_hdf5(buffer.data(), size);
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
        _proto->add_prototxt(buffer, size);
      } else if (ext == ".protobuf") {
        _proto->add_protobuf(buffer, size);
      } else if (ext == ".h5") {
        _proto->add_hdf5(buffer, size);
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

void nnp::set_batch_size(int batch_size) { _proto->_batch_size = batch_size; }

int nnp::get_batch_size() { return _proto->_batch_size; }

int nnp::num_of_executors() { return _proto->num_of_executors(); }

std::vector<std::string> nnp::get_executor_input_names(int index) {
  return _proto->get_executor_input_names(index);
}

std::vector<nbla::CgVariablePtr> nnp::get_executor_input_variables(int index) {
  return _proto->get_executor_input_variables(index);
}

std::vector<nbla::CgVariablePtr>
nnp::get_executor(int index, std::vector<nbla::CgVariablePtr> inputs) {
  return _proto->get_executor(index, inputs);
}
}
}
