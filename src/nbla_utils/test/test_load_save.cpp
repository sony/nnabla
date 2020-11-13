// Copyright (c) 2019 Sony Corporation. All Rights Reserved.
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

// test_load_save.cpp

#include "gtest/gtest.h"
#include <algorithm>
#include <fstream>
#include <iostream>
#include <nbla_utils/nnp.hpp>
#include <numeric>
#include <random>
#include <string>
#include <vector>

namespace nbla {
namespace utils {

using namespace std;
using namespace nbla;

const Context kCpuCtx{{"cpu:float"}, "CpuCachedArray", "0"};

TEST(test_load_nnp_on_memory, test_load_nnp_on_memory) {
  std::ifstream file("tmp.nnp", std::ios::binary | std::ios::ate);
  std::streamsize size = file.tellg();

  // Skip when tmp.nnp does not exist.
  // Because it will generate with python test.
  if (size >= 0) {
    file.seekg(0, std::ios::beg);

    std::vector<char> buffer(size);
    if (!file.read(buffer.data(), size)) {
      ASSERT_TRUE(false);
    }

    nbla::utils::nnp::Nnp nnp(kCpuCtx);
    nnp.add(buffer.data(), size);
    std::vector<std::string> executor_names = nnp.get_executor_names();
    EXPECT_STREQ(executor_names[0].c_str(), "runtime");
    shared_ptr<nnp::Executor> executor = nnp.get_executor(executor_names[0]);
    EXPECT_NE(executor.get(), nullptr);
  } else {
    std::cout
        << "[  SKIPPED ] test_load_nnp_on_memory. 'tmp.nnp' does not generated."
        << std::endl;
  }
}
}
}
