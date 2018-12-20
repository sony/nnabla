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

#include <cmdline.h>
#include <fstream>
#include <iostream>

#include "internal.hpp"
#include "nbla_commands.hpp"

bool nbla_dump(int argc, char *argv[]) {
  cmdline::parser p;
  p.add<int>("batch_size", 'b', "Batch size", false, -1);
  p.add<int>("help", 0, "Print help", false);
  if (!p.parse(argc, argv) || p.exist("help")) {
    std::cout << p.error_full() << p.usage();
    return false;
  }

  std::cout << std::endl;

  nbla::Context ctx{{"cpu:float"}, "CpuCachedArray", "0"};
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
