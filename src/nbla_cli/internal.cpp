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

#include <fstream>
#include <iostream>

#include "internal.hpp"

std::vector<std::string> add_files_to_nnp(nbla::utils::nnp::Nnp &nnp,
                                          std::vector<std::string> files,
                                          bool on_memory) {
  std::vector<std::string> input_files;

  for (int i = 0; i < files.size(); i++) {
    std::string arg = files[i];
    int ep = arg.find_last_of(".");
    std::string ext = arg.substr(ep, arg.size() - ep);

    if (ext == ".h5" || ext == ".nntxt" || ext == ".protobuf" ||
        ext == ".prototxt") {
      nnp.add(arg);
    } else if (ext == ".nnp") {
      if (on_memory) {

        std::ifstream file(arg, std::ios::binary | std::ios::ate);
        std::streamsize size = file.tellg();
        file.seekg(0, std::ios::beg);

        std::vector<char> buffer(size);
        if (file.read(buffer.data(), size)) {
          nnp.add(buffer.data(), size);
        }
      } else {
        nnp.add(arg);
      }
    } else {
      input_files.push_back(arg);
    }
  }
  return input_files;
}
