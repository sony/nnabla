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
#include <algorithm>
#include <numeric>
#include <random>
#include <vector>

#include "nnp_impl.hpp"
#include "nnp_impl_dataset_npy.hpp"

#include <zlib.h>

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

// import helper functions for testing
extern bool load_variable_list(const string &path, vector<string> &data_names);
extern bool search_for_cache_files(const string &path,
                                   const vector<string> &data_names,
                                   vector<shared_ptr<CacheFile>> &cache_files);

// helper functions for testing
string show_shape(const Shape_t &shape) {
  string message = "(";
  for (auto dim : shape) {
    message += std::to_string(dim);
    message += ",";
  }
  message += ")";
  return message;
}

void dump_ndarray(NdArrayPtr ndarray, dtypes dtype) {
  string line;
  Shape_t strides = ndarray->strides();
  if (dtype == dtypes::FLOAT) {
    float *d = ndarray->cast(dtype, kCpuCtx)->pointer<float>();
    for (int i = 0; i < ndarray->size(); ++i) {
      line += to_string(*d++) + " ";

      for (int j = strides.size() - 2; j >= 0; --j) {
        if ((i + 1) % strides[j] == 0) {
          cout << line << endl;
          line = "";
        }
      }
    }
  } else if (dtype == dtypes::UBYTE) {
    uint8_t *d = ndarray->cast(dtype, kCpuCtx)->pointer<uint8_t>();
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
}

void dump_shape(Shape_t s, Shape_t t) {
  cout << "(";
  for (auto d : s) {
    cout << d << ",";
  }
  cout << ")" << endl;
  EXPECT_EQ(t, s);
}

#if 1
std::vector<int> batch_sizes = {1, 32};
std::vector<bool> shuffle_setting = {true, false};
std::vector<bool> normal_setting = {true, false};
std::vector<int> iter_nums = {1, 9};
#else
std::vector<int> batch_sizes = {128};
std::vector<bool> shuffle_setting = {false};
std::vector<bool> normal_setting = {false};
std::vector<int> iter_nums = {1};
#endif

#if 0
TEST(test_load_variable_list, normalcase) {
  string path = "./cache_npy";
  vector<string> data_names;

  EXPECT_TRUE(load_variable_list(path, data_names));
  EXPECT_EQ(4, data_names.size());
  EXPECT_EQ("x0", data_names[0]);
  EXPECT_EQ("y", data_names[1]);
}

TEST(test_search_for_cache_files, normalcase) {
  string path = "./cache_npy";
  vector<string> data_names;
  EXPECT_TRUE(load_variable_list(path, data_names));
  vector<shared_ptr<CacheFile>> cache_files;
  EXPECT_TRUE(search_for_cache_files(path, data_names, cache_files));
  EXPECT_EQ(10, cache_files.size());
}

TEST(test_cache_file_object, test_all_preload) {
  string path = "./cache_npy";
  vector<string> data_names;
  EXPECT_TRUE(load_variable_list(path, data_names));
  vector<shared_ptr<CacheFile>> cache_files;
  EXPECT_TRUE(search_for_cache_files(path, data_names, cache_files));
  EXPECT_EQ(10, cache_files.size());

  for (int i = 0; i < cache_files.size(); ++i) {
    cache_files[i]->preload();
    cout << cache_files[i]->get_name() << " preload." << endl;
    EXPECT_EQ(100, cache_files[i]->get_num_data());
    cout << "check variable x" << endl;
    const VariableDesc *var_desc = cache_files[i]->get_variable_desc("x0");
    EXPECT_EQ(1, var_desc->word_size);
    EXPECT_EQ(dtypes::UBYTE, var_desc->data_type);
    dump_shape(var_desc->shape, {100, 1, 28, 28});
    cout << "check variable y" << endl;
    var_desc = cache_files[i]->get_variable_desc("y");
    EXPECT_EQ(4, var_desc->word_size);
    EXPECT_EQ(dtypes::FLOAT, var_desc->data_type);
    dump_shape(var_desc->shape, {100, 1});
  }
}
#endif

class RingBufferTester
    : public ::testing::TestWithParam<std::tuple<int, int, bool>> {
protected:
  map<string, shared_ptr<RingBuffer>> ring_buffers_;
  vector<string> data_names_;
  int cache_file_num_{8};

  virtual void SetUp() {
    string path = "./cache_npy";
    data_names_.clear();
    EXPECT_TRUE(load_variable_list(path, data_names_));
    vector<shared_ptr<CacheFile>> cache_files;
    EXPECT_TRUE(search_for_cache_files(path, data_names_, cache_files));
    EXPECT_EQ(cache_file_num_, cache_files.size());

    for (auto f : cache_files) {
      f->preload();
    }

    bool shuffle = std::get<2>(GetParam());
    vector<int> idx_list;
    if (shuffle) {
      int data_num = cache_files[0]->get_num_data();
      idx_list.resize(data_num);
      std::iota(idx_list.begin(), idx_list.end(), 0);
      unsigned seed =
          std::chrono::system_clock::now().time_since_epoch().count();
      std::shuffle(std::begin(cache_files), std::end(cache_files),
                   std::default_random_engine(seed));
      std::shuffle(std::begin(idx_list), std::end(idx_list),
                   std::default_random_engine(seed));
    }

    for (auto n : data_names_) {
      vector<int> idx_list;
      ring_buffers_[n] = make_shared<RingBuffer>(
          cache_files, std::get<0>(GetParam()), n, idx_list, shuffle);
    }
  }
};

INSTANTIATE_TEST_CASE_P(
    ring_buffer_test, RingBufferTester,
    ::testing::Combine(::testing::ValuesIn(batch_sizes),
                       ::testing::ValuesIn(iter_nums),
                       ::testing::ValuesIn(shuffle_setting)));

TEST_P(RingBufferTester, simple_test_read_batch_data) {
  for (int i = 0; i < std::get<1>(GetParam()); ++i) {
    auto r = ring_buffers_[data_names_[0]];
    shared_ptr<VariableBuffer> v = make_shared<VariableBuffer>();
    r->read_batch_data(i, v);
    r->fill_up();
  }
}

TEST_P(RingBufferTester, full_test_read_batch_data) {
  for (int i = 0; i < std::get<1>(GetParam()); ++i) {
    for (auto kv : ring_buffers_) {
      shared_ptr<VariableBuffer> v = make_shared<VariableBuffer>();
      kv.second->read_batch_data(i, v);
      kv.second->fill_up();
    }
  }
}

TEST_P(RingBufferTester, test_iter_num_roll_back) {
  cout << std::get<0>(GetParam()) << endl;
  cout << std::get<1>(GetParam()) << endl;
  for (int i = 0; i < std::get<1>(GetParam()); ++i) {
    for (auto kv : ring_buffers_) {
      shared_ptr<VariableBuffer> v = make_shared<VariableBuffer>();
      kv.second->read_batch_data(i, v);
      kv.second->fill_up();
    }
  }

  for (int i = 0; i < std::get<1>(GetParam()); ++i) {
    for (auto kv : ring_buffers_) {
      shared_ptr<VariableBuffer> v = make_shared<VariableBuffer>();
      kv.second->read_batch_data(i, v);
      kv.second->fill_up();
    }
  }
}

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

INSTANTIATE_TEST_CASE_P(DatasetNpyCacheTest, DataSetNpyTest,
                        ::testing::Combine(::testing::ValuesIn(shuffle_setting),
                                           ::testing::ValuesIn(normal_setting),
                                           ::testing::ValuesIn(batch_sizes),
                                           ::testing::ValuesIn(iter_nums)));

TEST_P(DataSetNpyTest, TestDataIterator) {
  cout << std::get<2>(GetParam()) << endl;
  cout << std::get<3>(GetParam()) << endl;

  const int n_iter = std::get<3>(GetParam());

  auto dataset_npy_impl = make_shared<DatasetNpyCache>(dataset_);
  auto data_iterator =
      make_shared<DataIteratorFromCacheFiles>(dataset_npy_impl);

  string path = "./cache_npy";
  vector<string> data_names;
  EXPECT_TRUE(load_variable_list(path, data_names));
  int variable_num = data_names.size() - 1;

  for (int i = 0; i < n_iter; ++i) {
    auto name_data = data_iterator->next();
    auto v_y = name_data["y"];
    uint32_t *buffer = v_y->cast(nbla::get_dtype<uint32_t>(), kCpuCtx, true)
                           ->pointer<uint32_t>();
    Shape_t shape = v_y->shape();
    int batch_size = shape[0];
    for (int j = 0; j < variable_num; ++j) {
      string x_name = "x" + to_string(j);
      if (name_data.find(x_name) == name_data.end())
        break;
      auto x = name_data[x_name];
      shape = x->shape();
      shape[0] = 1;
      int item_size = compute_size_by_shape(shape);
      uint32_t *data_buffer =
          x->cast(nbla::get_dtype<uint32_t>(), kCpuCtx, true)
              ->pointer<uint32_t>();
      for (int xb = 0; xb < batch_size; ++xb) {
        uint32_t target_crc32 = *(buffer + xb * variable_num + j);
        uint32_t *data_item_buffer = data_buffer + xb * item_size;
        uint32_t c32 = crc32(0, (const unsigned char *)data_item_buffer,
                             item_size * sizeof(uint32_t));
        ASSERT_EQ(target_crc32, c32);
      }
    }
  }

  EXPECT_TRUE(true);
}

} // namespace nnp
} // namespace utils
} // namespace nbla