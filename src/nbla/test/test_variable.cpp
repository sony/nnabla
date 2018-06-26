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

// test_variable.cpp

#include "gtest/gtest.h"
#include <nbla/common.hpp>
#include <nbla/variable.hpp>

namespace nbla {

TEST(VariableTest, Create) {
  // NNabla::init();
  Variable var(Shape_t{2, 3, 4});
}

class VariableManipTest : public ::testing::Test {
public:
protected:
  shared_ptr<Variable> var_;
  Context ctx_;
  dtypes dtype_;
  Shape_t shape_;
  virtual void SetUp() {
    // NNabla::init();
    ctx_.array_class = "CpuArray";
    dtype_ = dtypes::FLOAT;
    shape_ = Shape_t{2, 3, 4, 5};
    var_.reset(new Variable(shape_));
  }

  // virtual void TearDown() {}
};

TEST_F(VariableManipTest, Properties) {
  ASSERT_EQ(var_->shape(), shape_);
  ASSERT_EQ(var_->size(), compute_size_by_shape(shape_));
  for (int i = 0; i < shape_.size(); ++i) {
    Size_t size = 1;
    for (int j = i; j < shape_.size(); ++j) {
      size *= shape_[j];
    }
    ASSERT_EQ(size, var_->size(i));
  }
}

TEST_F(VariableManipTest, CastData) {
  var_->data()->cast(dtypes::FLOAT, ctx_);
  var_->grad()->cast(dtypes::FLOAT, ctx_);
}

TEST_F(VariableManipTest, GetData) {
  var_->data()->get(dtypes::FLOAT, ctx_);
  var_->grad()->get(dtypes::FLOAT, ctx_);
}

TEST_F(VariableManipTest, DataDiff) {
  {
    Array *arr_data = var_->data()->cast(dtypes::FLOAT, ctx_);
    float *data_f = arr_data->pointer<float>();
    for (int i = 0; i < arr_data->size(); ++i) {
      data_f[i] = i - 5;
    }
  }
  {
    Array *arr_grad = var_->grad()->cast(dtypes::FLOAT, ctx_);
    float *data_f = arr_grad->pointer<float>();
    for (int i = 0; i < arr_grad->size(); ++i) {
      data_f[i] = i - 10;
    }
  }
  {
    const Array *arr_data = var_->data()->get(dtypes::FLOAT, ctx_);
    const float *data_f = arr_data->const_pointer<float>();
    for (int i = 0; i < arr_data->size(); ++i) {
      EXPECT_EQ(i - 5, data_f[i]);
    }
  }
}
}
