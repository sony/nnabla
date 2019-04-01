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

// test_dataset.cpp

#include "gtest/gtest.h"
#include <vector>

#include "nnp_impl.hpp"
#include "nnp_impl_dataset_npy.hpp"

#ifdef _WIN32
#include <fcntl.h>
#include <io.h>
#define open _open
#define O_RDONLY _O_RDONLY
#endif

namespace nbla {
namespace utils {
namespace nnp {

#ifndef NBLA_UTILS_WITH_HDF5
using namespace std;

const nbla::Context kCpuCtx{{"cpu:float"}, "CpuCachedArray", "0"};

class DataSetNpyTest
    : public ::testing::TestWithParam<std::tuple<bool, bool, int, int>> {
protected:
  virtual void SetUp() {
    dataset_.set_name("test_dataset");
    dataset_.set_uri("/home/shared/data/input_x.npy");
    dataset_.set_cache_dir("./cache_npy");
    dataset_.set_create_cache_explicitly(true);
    dataset_.set_shuffle(std::get<0>(GetParam()));
    dataset_.set_no_image_normalization(std::get<1>(GetParam()));
    dataset_.set_batch_size(std::get<2>(GetParam()));
  }

  ::Dataset dataset_;
};

std::vector<int> int_param = {1, 5};
std::vector<bool> bool_param = {true, false};

INSTANTIATE_TEST_CASE_P(AllCombinations, DataSetNpyTest,
                        ::testing::Combine(::testing::ValuesIn(bool_param),
                                           ::testing::ValuesIn(bool_param),
                                           ::testing::ValuesIn(int_param),
                                           ::testing::ValuesIn(int_param)));

string show_shape(const Shape_t &shape) {
  string message = "(";
  for (auto dim : shape) {
    message += std::to_string(dim);
    message += ",";
  }
  message += ")";
  return message;
}

void dump_ndarray(NdArrayPtr ndarray) {
  string line;
  Shape_t strides = ndarray->strides();
  float *d = ndarray->cast(get_dtype<float>(), kCpuCtx)->pointer<float>();
  for (int i = 0; i < ndarray->size(); ++i) {
    line += to_string(*d++) + " ";

    for (int j = strides.size() - 2; j >= 0; --j) {
      if ((i + 1) % strides[j] == 0) {
        cout << line << endl;
        line = "";
      }
    }
  }
}

TEST_P(DataSetNpyTest, TestDataIterator) {
  const int n_iter = std::get<3>(GetParam());

  auto dataset_npy_impl = make_shared<DatasetNpyImpl>(dataset_);
  auto data_iterator =
      make_shared<DataIteratorFromCacheFiles>(dataset_npy_impl);

  for (int i = 0; i < n_iter; ++i) {
    auto name_data = data_iterator->next();
    for (auto kv : name_data) {
      cout << kv.first << ": " << show_shape(kv.second->shape()) << endl;
      dump_ndarray(kv.second);
    }
  }

  EXPECT_TRUE(true);
}
#endif

} // namespace nnp
} // namespace utils
} // namespace nbla