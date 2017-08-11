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

int main(int argc, char *argv[]) {

  if (argc < 2) {
    std::cerr << "Usage: " << argv[0] << " FILE(s)" << std::endl;
    return -1;
  }

  nbla::Context ctx; // ("cpu", "CpuArray", "0", "default");
  nbla_utils::NNP::nnp nnp(ctx);

  std::vector<std::string> input_files;
  for (int i = 1; i < argc; i++) {
    std::string arg(argv[i]);
    int ep = arg.find_last_of(".");
    std::string ext = arg.substr(ep, arg.size() - ep);

    std::cout << arg << " " << ext << std::endl;

    if (ext == ".h5" || ext == ".nnp" || ext == ".nntxt" ||
        ext == ".protobuf" || ext == ".prototxt") {
      nnp.add(arg);
    }
    if (ext == ".bin") {
      input_files.push_back(arg);
    }
  }

  int n = nnp.num_of_executors();
  if (n > 0) {
    // Get variables for input values.
    std::vector<std::string> names = nnp.get_executor_input_names(0);
    std::vector<nbla::CgVariablePtr> inputs =
        nnp.get_executor_input_variables(0);
    for (int i = 0; i < inputs.size(); i++) {
      std::cout << "Input" << i << ": " << names[i] << std::endl;
      auto var = inputs[i]->variable();

      std::string ifile = input_files[i];
      std::ifstream file(ifile.c_str(), std::ios::binary | std::ios::ate);
      std::streamsize size = file.tellg();
      file.seekg(0, std::ios::beg);

      std::cout << "Size: " << size << " iSize: " << var.get()->size()
                << std::endl;

      float *data = var->cast_data_and_get_pointer<float>(ctx);
      if ((int)size == ((int)(var.get()->size()) * sizeof(float))) {
        std::vector<float> buffer(size / sizeof(float));
        if (file.read((char *)buffer.data(), size)) {
          std::cout << "  Read data from [" << ifile << "]" << std::endl;
          for (int j = 0; j < var.get()->size(); ++j) {
            data[j] = buffer[j];
          }
        }
      }
    }

    // Get computation graph for inference.
    std::vector<nbla::CgVariablePtr> e = nnp.get_executor(0, inputs);
    e[0]->forward(true,   // clear_buffer
                  false); // clear_no_need_grad

    auto var = e[0]->variable();
    float *data = var->cast_data_and_get_pointer<float>(ctx);
    for (int i = 0; i < var.get()->size(); ++i) {
      printf("%f\n", data[i]);
    }
  }
  return 0;
}
