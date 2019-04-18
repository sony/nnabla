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

// test_cpu_array.cpp

#include "gtest/gtest.h"
#include <nbla/array.hpp>
#include <nbla/array/cpu_array.hpp>
#include <nbla/array_registry.hpp>
#include <nbla/common.hpp>
#include <nbla/cpu.hpp>

namespace nbla {

TEST(CpuArrayTest, ConstructByClass) {
  // NNabla::init();
  Context ctx;
  shared_ptr<Array> arr(new CpuArray(20, dtypes::FLOAT, ctx));
}

TEST(CpuArrayTest, ConstructByContext) {
  // NNabla::init();
  Context ctx;
  ctx.array_class = "CpuArray";
  shared_ptr<Array> array(ArrayCreator::create(20, dtypes::FLOAT, ctx));
}

class CpuArrayManipTest : public ::testing::Test {
public:
protected:
  shared_ptr<Array> arr_;
  Context ctx_;
  dtypes dtype_;
  Size_t size_;
  virtual void SetUp() {
    // NNabla::init();
    ctx_.array_class = "CpuArray";
    dtype_ = dtypes::FLOAT;
    size_ = 20;
    arr_.reset(ArrayCreator::create(size_, dtype_, ctx_));
  }

  // virtual void TearDown() {}
};

TEST_F(CpuArrayManipTest, Properties) {
  ASSERT_NE(arr_->pointer(), nullptr);
  ASSERT_EQ(arr_->const_pointer(), arr_->pointer());
  ASSERT_EQ(arr_->dtype(), dtype_);
  ASSERT_EQ(arr_->size(), size_);
}

TEST_F(CpuArrayManipTest, TypeConversion) {
  float *data = arr_->pointer<float>();
  for (int i = 0; i < arr_->size(); ++i) {
    data[i] = i - 5;
  }
  Array *arr2 = ArrayCreator::create(size_, dtypes::INT, ctx_);
  arr2->copy_from(arr_.get());
  int *data2 = arr2->pointer<int>();
  for (int i = 0; i < arr2->size(); ++i) {
    data2[i] = i - 5;
  }
  delete arr2;
}

#if 0
TEST(CpuCachedArrayTest, CacheTest) {
  Context ctx;
  ctx.array_class = "CpuArray";
  SingletonManager::get<Cpu>()->memcache().clear();
  {
    shared_ptr<Array> arr, arr2;
    // Create cached array in 512 bytes
    arr.reset(new CpuCachedArray(20, dtypes::FLOAT, ctx));
    void *ptr = arr->pointer();
    // Cache 512 byte memory into pool.
    arr = nullptr;
    ASSERT_EQ(1, SingletonManager::get<Cpu>()->memcache().count(""));
    // Pop 512 bytes memory from cache
    arr.reset(new CpuCachedArray(512 / 4, dtypes::FLOAT, ctx));
    void *ptr2 = arr->pointer();
    // Pointing the same address
    ASSERT_EQ(ptr, ptr2);
    // Cache 512 bytes memory again.
    arr = nullptr;
    ASSERT_EQ(1, SingletonManager::get<Cpu>()->memcache().count(""));
    // Create 1024 bytes memory.
    arr.reset(new CpuCachedArray(512 / 4 + 1, dtypes::FLOAT, ctx));
    ptr2 = arr->pointer();
    ASSERT_NE(ptr, ptr2);
    // Cache 1024 bytes memory into pool.
    arr = nullptr;
    // Now 2 memory objects in cache.
    ASSERT_EQ(2, SingletonManager::get<Cpu>()->memcache().count(""));
  }

  {
    CpuCachedArray b(1, dtypes::BYTE, ctx); // 1 byte
    b.pointer();
    CpuCachedArray s(1, dtypes::SHORT, ctx); // 2 bytes
    s.pointer();
    CpuCachedArray f(1, dtypes::FLOAT, ctx); // 4 bytes
    f.pointer();
    CpuCachedArray d(1, dtypes::DOUBLE, ctx); // 8 bytes
    d.pointer();
  }
  ASSERT_EQ(6, SingletonManager::get<Cpu>()->memcache().count(""));
  {
    CpuCachedArray b(1, dtypes::BYTE, ctx); // Pop from cache
    b.pointer();
    ASSERT_EQ(5, SingletonManager::get<Cpu>()->memcache().count(""));
    CpuCachedArray b_(1, dtypes::BYTE, ctx); // Create
    b_.pointer();
    ASSERT_EQ(5, SingletonManager::get<Cpu>()->memcache().count(""));
  }
  ASSERT_EQ(7, SingletonManager::get<Cpu>()->memcache().count(""));
  {
    CpuCachedArray b2(2, dtypes::BYTE, ctx); // Pop 2 bytes memory from cache
    b2.pointer();
    ASSERT_EQ(6, SingletonManager::get<Cpu>()->memcache().count(""));
    CpuCachedArray b4(4, dtypes::BYTE, ctx); // Pop 4 bytes memory from cache
    b4.pointer();
    ASSERT_EQ(5, SingletonManager::get<Cpu>()->memcache().count(""));
    CpuCachedArray b8(8, dtypes::BYTE, ctx); // Pop 8 bytes memory from cache
    b8.pointer();
    ASSERT_EQ(4, SingletonManager::get<Cpu>()->memcache().count(""));
  }
  ASSERT_EQ(7, SingletonManager::get<Cpu>()->memcache().count(""));
}
#endif
}
