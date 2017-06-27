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

// test_synced_array.cpp

#include "gtest/gtest.h"
#include <nbla/common.hpp>
#include <nbla/synced_array.hpp>

namespace nbla {

TEST(SyncedArrayTest, Create) { SyncedArray arr(15); }

class SyncedArrayManipTest : public ::testing::Test {
public:
protected:
  shared_ptr<SyncedArray> arr_;
  Context ctx_;
  dtypes dtype_;
  Size_t size_;
  virtual void SetUp() {
    ctx_.array_class = "CpuArray";
    dtype_ = dtypes::FLOAT;
    size_ = 20;
    arr_.reset(new SyncedArray(size_));
  }

  // virtual void TearDown() {}
};

TEST_F(SyncedArrayManipTest, Properties) { ASSERT_EQ(arr_->size(), size_); }

TEST_F(SyncedArrayManipTest, SyncDtypesByCastDataGetData) {
  {
    Array *arr_f = arr_->cast(dtypes::FLOAT, ctx_);
    ASSERT_EQ(arr_f->dtype(), dtypes::FLOAT);
    float *data_f = arr_f->pointer<float>();
    for (int i = 0; i < arr_->size(); ++i) {
      data_f[i] = i - 5;
    }
  }
  {
    Array *arr_d = arr_->cast(dtypes::DOUBLE, ctx_);
    ASSERT_EQ(arr_d->dtype(), dtypes::DOUBLE);
    ASSERT_EQ(arr_d->size(), arr_->size());
    double *data_d = arr_d->pointer<double>();
    ASSERT_NE(data_d, nullptr);
    for (int i = 0; i < arr_->size(); ++i) {
      EXPECT_EQ(data_d[i], i - 5);
      data_d[i] = i - 10;
    }
  }
  {
    const Array *arr_f = arr_->get(dtypes::FLOAT, ctx_);
    ASSERT_EQ(arr_f->dtype(), dtypes::FLOAT);
    ASSERT_EQ(arr_f->size(), arr_->size());
    const float *data_f = arr_f->const_pointer<float>();
    ASSERT_NE(data_f, nullptr);
    for (int i = 0; i < arr_->size(); ++i) {
      EXPECT_EQ(data_f[i], i - 10);
    }
  }
}
}
