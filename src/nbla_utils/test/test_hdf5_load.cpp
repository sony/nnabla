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

// test_hdf5_load.cpp
#ifdef NBLA_UTILS_WITH_HDF5
#include "gtest/gtest.h"
#include <vector>

#include "internal.hpp"
#include "nnp_impl.hpp"
#include <nbla_utils/nnp.hpp>

#ifdef _WIN32
#include <fcntl.h>
#include <io.h>
#define open _open
#define O_RDONLY _O_RDONLY
#endif

namespace nbla {
namespace utils {
namespace nnp {

using namespace std;

const nbla::Context kCpuCtx{{"cpu:float"}, "CpuCachedArray", "0"};

class HDF5ReaderTester : public ::testing::TestWithParam<string> {
protected:
  virtual void SetUp() {}
};

std::vector<string> test_h5_files = {"test_loop_controls/nested_loop_test",
                                     "test_loop_controls/recurrent_delay_test",
                                     "test_loop_controls/no_repeat_test"};

INSTANTIATE_TEST_CASE_P(HDF5ReaderTester, HDF5ReaderTester,
                        ::testing::ValuesIn(test_h5_files));

TEST_P(HDF5ReaderTester, TestH5Reader) {
  const string h5_files_path = GetParam();
  const string h5_file = h5_files_path + "/parameters.h5";
  const string nntxt_file = h5_files_path + "/net.nntxt";
  const vector<string> nnp_files = {nntxt_file, h5_file};

  nbla::Context ctx{{"cpu:float"}, "CpuCachedArray", "0"};
  nbla::utils::nnp::Nnp nnp(ctx);
  add_files_to_nnp(nnp, nnp_files);
}

} // namespace nnp
} // namespace utils
} // namespace nbla

#endif // ifdef NBLA_UTILS_WITH_HDF5