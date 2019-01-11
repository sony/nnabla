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

#include <iostream>

#include "nbla_commands.hpp"

static void print_usage_and_exit(const char *name) {
  std::cerr << "Usage: " << name << " (infer|dump|train)" << std::endl;
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
  std::cerr << "    " << name << " train input_files ..." << std::endl;
  std::cerr << "               input_file must be nnp." << std::endl;
  exit(-1);
}

int main(int argc, char *argv[]) {
  const char *command_name = argv[0];

  if (argc < 2) {
    print_usage_and_exit(command_name);
  }

  std::string command(*++argv);
  argc--;

  if (command == "infer") {
    nbla_infer(argc, argv);
  } else if (command == "dump") {
    nbla_dump(argc, argv);
  } else if (command == "train") {
    nbla_train(argc, argv);
  } else {
    print_usage_and_exit(command_name);
  }

  return 0;
}
