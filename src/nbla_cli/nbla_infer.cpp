// Copyright (c) 2018 Sony Corporation. All Rights Reserved.
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

#include <cmdline.h>

#include <cstdint>
#include <fstream>
#include <iostream>

#include "internal.hpp"
#include "nbla_commands.hpp"

bool nbla_infer_core(nbla::Context ctx, int argc, char *argv[]) {
  cmdline::parser p;
  p.add<int>("batch_size", 'b', "Batch size", false, -1);

  // TODO: use Nnp::get_executor_names() to get default executor.
  p.add<std::string>("executor", 'e', "Executor name (required)", true,
                     std::string());

  p.add<std::string>("data_type", 't', "Parameter type (uint8 or float)", false,
                     std::string());

  p.add<std::string>(
      "output", 'o',
      "Output filename prefix, if not specified print output to stdout.", false,
      std::string());
  p.add<int>("help", 0, "Print help", false);
  p.add("on_memory", 'O', "On memory");

  bool on_memory = false;

  if (!p.parse(argc, argv) || p.exist("help")) {
    std::cout << p.error_full() << p.usage();
    return false;
  }

  if (p.exist("on_memory")) {
    on_memory = true;
  }

  nbla::utils::nnp::Nnp nnp(ctx);
  std::vector<std::string> input_files =
      add_files_to_nnp(nnp, p.rest(), on_memory);

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

    if (p.get<std::string>("data_type") == "uint8") {
      uint8_t *data = var->cast_data_and_get_pointer<uint8_t>(ctx);
      if ((int)size == ((int)(var->size()) * sizeof(uint8_t))) {
        std::vector<uint8_t> buffer(size / sizeof(uint8_t));
        if (file.read((char *)buffer.data(), size)) {
          std::cout << "  Read data from [" << ifile << "]" << std::endl;
          for (int j = 0; j < var->size(); ++j) {
            data[j] = buffer[j];
          }
        }
      } else {
        std::cout << " Data size mismatch on data " << i
                  << ". expected size is "
                  << (int)(var->size()) * sizeof(uint8_t) << " but data file ["
                  << ifile << "] size is " << size << "." << std::endl;
        return false;
      }
    } else {
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
        std::cout << " Data size mismatch on data " << i
                  << ". expected size is " << (int)(var->size()) * sizeof(float)
                  << " but data file [" << ifile << "] size is " << size << "."
                  << std::endl;
        return false;
      }
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

bool nbla_infer(int argc, char *argv[]) {
  // Create a context
  nbla::Context ctx{{"cpu:float"}, "CpuCachedArray", "0"};

  // Train
  if (!nbla_infer_core(ctx, argc, argv)) {
    return false;
  }

  return true;
}
