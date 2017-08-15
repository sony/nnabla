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

#include <cmdline.h>

static void print_usage_and_exit(const char *name) {
  std::cerr << "Usage: " << name << " (infer|dump)" << std::endl;
  std::cerr << "    " << name
            << " infer [-b BATCHSIZE] [-e EXECUTOR] input_files ..."
            << std::endl;
  std::cerr << "               input_file must be one of followings."
            << std::endl;
  std::cerr
      << "                   *.nnp      : Network structure and parameter."
      << std::endl;
  std::cerr
      << "                   *.nntxt    : Network structure in prototxt format."
      << std::endl;
  std::cerr << "                   *.prototxt : Same as nntxt." << std::endl;
  std::cerr << "                   *.h5       : Parameters in h5 format."
            << std::endl;
  std::cerr << "                   *.protobuf : Network structure and "
               "parameters in binary."
            << std::endl;
  std::cerr << "                   *.bin      : Input data." << std::endl;
  std::cerr << "    " << name << " dump input_files ..." << std::endl;
  std::cerr
      << "               input_file must be nnp, nntxt, prototxt, h5, protobuf."
      << std::endl;
  exit(-1);
}

std::vector<std::string> add_files_to_nnp(nbla::utils::nnp::Nnp &nnp,
                                          std::vector<std::string> files) {
  std::vector<std::string> input_files;

  for (int i = 0; i < files.size(); i++) {
    std::string arg = files[i];
    int ep = arg.find_last_of(".");
    std::string ext = arg.substr(ep, arg.size() - ep);

    if (ext == ".h5" || ext == ".nnp" || ext == ".nntxt" ||
        ext == ".protobuf" || ext == ".prototxt") {
      nnp.add(arg);
    } else {
      input_files.push_back(arg);
    }
  }
  return input_files;
}

bool infer(int argc, char *argv[]) {
  cmdline::parser p;
  p.add<int>("batch_size", 'b', "Batch size", false, -1);
  p.add<std::string>("executor", 'e', "Executor name (required)", true,
                     std::string());
  p.add<int>("help", 0, "Print help", false);
  if (!p.parse(argc, argv) || p.exist("help")) {
    std::cout << p.error_full() << p.usage();
    return false;
  }

  nbla::Context ctx{"cpu", "CpuCachedArray", "0", "default"};
  nbla::utils::nnp::Nnp nnp(ctx);
  std::vector<std::string> input_files = add_files_to_nnp(nnp, p.rest());

  int batch_size = p.get<int>("batch_size");
  std::string exec_name = p.get<std::string>("executor");
  std::shared_ptr<nbla::utils::nnp::Executor> exec =
      nnp.get_executor(exec_name);
  exec->set_batch_size(batch_size);

  // Get variables for input values.
  std::vector<nbla::utils::nnp::Executor::DataVariable> inputs =
      exec->get_data_variables();
  for (int i = 0; i < inputs.size(); i++) {
    std::cout << "Input" << i << ": " << inputs[i].data_name << std::endl;
    auto var = inputs[i].variable->variable();

    std::string ifile = input_files[i];
    std::ifstream file(ifile.c_str(), std::ios::binary | std::ios::ate);
    std::streamsize size = file.tellg();
    file.seekg(0, std::ios::beg);
    // TODO: other types
    float *data = var->cast_data_and_get_pointer<float>(ctx);
    if ((int)size == ((int)(var->size()) * sizeof(float))) {
      std::vector<float> buffer(size / sizeof(float));
      if (file.read((char *)buffer.data(), size)) {
        std::cout << "  Read data from [" << ifile << "]" << std::endl;
        for (int j = 0; j < var->size(); ++j) {
          data[j] = buffer[j];
        }
      }
    } else {
      std::cout << " Data size mismatch on data " << i << ". expected size is "
                << (int)(var->size()) * sizeof(float) << " but data file ["
                << ifile << "] size is " << size << "." << std::endl;
      return false;
    }
  }

  // Execute network.
  exec->execute();

  std::vector<nbla::utils::nnp::Executor::OutputVariable> outputs =
      exec->get_output_variables();
  for (auto it = outputs.begin(); it != outputs.end(); it++) {
    if (outputs.size() > 1) {
      std::cout << "Output: " << it->data_name << std::endl;
    }
    auto var = it->variable->variable();
    float *data = var->cast_data_and_get_pointer<float>(ctx);
    for (int i = 0; i < var.get()->size(); ++i) {
      printf("%f,", data[i]);
    }
    printf("\n");
  }
  return true;
}

#if 0
bool dump(int argc, char *argv[]) {
  cmdline::parser p;
  p.add<int>("batch_size", 'b', "Batch size", false, -1);
  p.add<int>("help", 0, "Print help", false);
  if (!p.parse(argc, argv) || p.exist("help")) {
    std::cout << p.error_full() << p.usage();
    return false;
  }

  nbla::Context ctx; // ("cpu", "CpuArray", "0", "default");
  nbla_utils::NNP::nnp nnp(ctx);
  add_files_to_nnp(nnp, p.rest());

  if (p.get<int>("batch_size") < 0) {
    std::cout << "Using default batch size." << std::endl;
  } else {
    std::cout << "Using batch size << " << p.get<int>("batch_size") << "."
              << std::endl;
  }

  nnp.set_batch_size(p.get<int>("batch_size"));
  int n = nnp.num_of_executors();
  std::cout << "This configuration has " << n << " executor(s)." << std::endl;
  for (int i = 0; i < n; i++) {
    std::cout << "    Executor No." << i << std::endl;

    std::vector<std::string> names = nnp.get_executor_input_names(i);
    std::vector<nbla::CgVariablePtr> inputs =
        nnp.get_executor_input_variables(i);
    for (int j = 0; j < inputs.size(); j++) {

      std::cout << "        Input No." << j << " Name:[" << names[j] << "]";
      auto v = inputs[i]->variable();
      auto shape = v->shape();

      std::cout << " Shape (";
      for (int k = 0; k < shape.size(); k++) {
        std::cout << " " << shape[k];
      }
      std::cout << " )" << std::endl;
    }
    std::vector<nbla::CgVariablePtr> e = nnp.get_executor(i, inputs);
    auto var = e[0]->variable();

    auto out_shape = var->shape();
    std::cout << "    Output: Shape (";
    for (int j = 0; j < out_shape.size(); j++) {
      std::cout << " " << out_shape[j];
    }
    std::cout << " )" << std::endl;
  }

  std::cout << "" << std::endl;

  return true;
}
#endif

int main(int argc, char *argv[]) {
  const char *command_name = argv[0];

  if (argc < 2) {
    print_usage_and_exit(command_name);
  }

  std::string command(*++argv);
  argc--;

  if (command == "infer") {
    infer(argc, argv);
#if 0
  } else if (command == "dump") {
    dump(argc, argv);
#endif
  } else {
    print_usage_and_exit(command_name);
  }

  return 0;
}
