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
            << " infer -e EXECUTOR [-b BATCHSIZE] [-o OUTPUT] input_files ..."
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

  // TODO: use Nnp::get_executor_names() to get default executor.
  p.add<std::string>("executor", 'e', "Executor name (required)", true,
                     std::string());
  p.add<std::string>(
      "output", 'o',
      "Output filename prefix, if not specified print output to stdout.", false,
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
  std::string output_filename_prefix = p.get<std::string>("output");

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
  int index = 0;
  for (auto it = outputs.begin(); it != outputs.end(); it++) {
    auto var = it->variable->variable();
    float *data = var->cast_data_and_get_pointer<float>(ctx);
    if (output_filename_prefix.size() > 0) {
      std::ofstream out;
      std::string out_filename =
          output_filename_prefix + "_" + std::to_string(index) + ".bin";
      std::cout << "Output to file [" << out_filename << "]" << std::endl;
      out.open(out_filename,
               std::ios::out | std::ios::binary | std::ios::trunc);
      out.write(reinterpret_cast<char *>(data),
                var.get()->size() * sizeof(float));
      out.close();
    } else {
      if (outputs.size() > 1) {
        std::cout << "Output: " << it->data_name << std::endl;
      }
      for (int i = 0; i < var.get()->size(); ++i) {
        printf("%f,", data[i]);
      }
      printf("\n");
    }
    index += 1;
  }
  return true;
}

bool dump(int argc, char *argv[]) {
  cmdline::parser p;
  p.add<int>("batch_size", 'b', "Batch size", false, -1);
  p.add<int>("help", 0, "Print help", false);
  if (!p.parse(argc, argv) || p.exist("help")) {
    std::cout << p.error_full() << p.usage();
    return false;
  }

  std::cout << std::endl;

  nbla::Context ctx{"cpu", "CpuCachedArray", "0", "default"};
  nbla::utils::nnp::Nnp nnp(ctx);
  std::vector<std::string> input_files = add_files_to_nnp(nnp, p.rest());

  auto names = nnp.get_executor_names();
  std::cout << "This configuration has " << names.size() << " executors."
            << std::endl;
  std::cout << std::endl;

  int i = 0, j = 0;
  for (auto it = names.begin(); it != names.end(); it++, i++) {
    std::cout << "  Executor No." << i << " Name [" << *it << "]" << std::endl;
    shared_ptr<nbla::utils::nnp::Executor> exec = nnp.get_executor(*it);
    if (p.get<int>("batch_size") < 0) {
      std::cout << "    Using default batch size " << exec->batch_size() << " ."
                << std::endl;
    } else {
      int batch_size = p.get<int>("batch_size");
      std::cout << "    Using batch size << " << batch_size << "." << std::endl;
      exec->set_batch_size(batch_size);
    }

    std::vector<nbla::utils::nnp::Executor::DataVariable> inputs =
        exec->get_data_variables();
    j = 0;
    std::cout << "     Inputs" << std::endl;
    for (auto jj = inputs.begin(); jj != inputs.end(); jj++, j++) {
      std::cout << "      Input No." << j << " Name [" << jj->data_name << "]";
      auto shape = jj->variable->variable()->shape();
      std::cout << " Shape (";
      for (int k = 0; k < shape.size(); k++) {
        std::cout << " " << shape[k];
      }
      std::cout << " )" << std::endl;
    }
    std::vector<nbla::utils::nnp::Executor::OutputVariable> outputs =
        exec->get_output_variables();
    j = 0;
    std::cout << "     Outputs" << std::endl;
    for (auto jj = outputs.begin(); jj != outputs.end(); jj++, j++) {
      std::cout << "      Output No." << j << " Name [" << jj->data_name << "]";
      auto shape = jj->variable->variable()->shape();
      std::cout << " Shape (";
      for (int k = 0; k < shape.size(); k++) {
        std::cout << " " << shape[k];
      }
      std::cout << " )" << std::endl;
    }
  }

  std::cout << "Finished" << std::endl;
  std::cout << std::endl;

  return true;
}

int main(int argc, char *argv[]) {
  const char *command_name = argv[0];

  if (argc < 2) {
    print_usage_and_exit(command_name);
  }

  std::string command(*++argv);
  argc--;

  if (command == "infer") {
    infer(argc, argv);
  } else if (command == "dump") {
    dump(argc, argv);
  } else {
    print_usage_and_exit(command_name);
  }

  return 0;
}
