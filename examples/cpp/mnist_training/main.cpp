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

#include <iostream>
#include <nbla/context.hpp>
using namespace std;
using namespace nbla;

#include "mnist_training.hpp"

/******************************************/
// Example of mnist training
/******************************************/
int main(int argc, char *argv[]) {

  // Usage message
  if (argc != 2) {
    std::cerr << std::endl;
    std::cerr << "Usage: " << argv[0] << " model.nnp " << std::endl;
    std::cerr << std::endl;
    std::cerr << "  model.nnp : model file with initialized parameters."
              << std::endl;
    std::cerr << std::endl;
    return -1;
  }

  // Create a context (the following setting is recommended.)
  nbla::Context ctx{{"cpu:float"}, "CpuCachedArray", "0"};

  // Execute training
  if (!mnist_training(ctx, argv[1])) {
    return (-1);
  }

  return 0;
}
