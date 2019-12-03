// Copyright (c) 2018 Sony Corporation. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// Include nnabla header files

#include <nbla/context.hpp>
#include <stdio.h>
#include <string.h>
using namespace nbla;

#include "dcgan_training.hpp"

/******************************************/
// Example of mnist training
/******************************************/
int main(int argc, char *argv[]) {

  // Create a context (the following setting is recommended.)
  nbla::Context ctx{{"cpu:float"}, "CpuCachedArray", "0"};

  // Execute training
  if (argc < 2 || strcmp(argv[1], "--static") == 0) {
    std::cout << "Execute training with static graph" << std::endl;
    if (!dcgan_training_with_static_graph(ctx)) {
      return -1;
    }
  } else if (strcmp(argv[1], "--dynamic") == 0) {
    std::cout << "Execute training with dynamic graph" << std::endl;
    if (!dcgan_training_with_dynamic_graph(ctx)) {
      return -1;
    }
  } else {
    std::cerr << std::endl;
    std::cerr << "Usage: " << argv[0] << " --static / --dynamic " << std::endl;
    std::cerr << std::endl;
    return -1;
  }

  return 0;
}
