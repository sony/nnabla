// Copyright 2021 Sony Group Corporation.
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
#include <nbla/computation_graph/variable.hpp>
#include <nbla/functions.hpp>
#include <nbla/variable.hpp>

namespace nbla {
class FunctionVariadicInputTest : public ::testing::Test {
public:
protected:
  CgVariablePtr var1_;
  CgVariablePtr var2_;
  Context ctx_;
  dtypes dtype_;
  Shape_t shape_;
  virtual void SetUp() {
    // NNabla::init();
    ctx_.array_class = "CpuArray";
    dtype_ = dtypes::FLOAT;
    shape_ = Shape_t{2, 3, 4, 5};
    var1_.reset(new CgVariable(shape_));
    var2_.reset(new CgVariable(shape_));
  }

  // virtual void TearDown() {}

  virtual void fill_data(CgVariablePtr var, float v) {
    Array *arr_data = var->variable()->data()->cast(dtypes::FLOAT, ctx_);
    float *data_f = arr_data->pointer<float>();
    for (int i = 0; i < arr_data->size(); ++i) {
      data_f[i] = v;
    }
  }

  virtual void assert_eq(CgVariablePtr var, int from, int to, float v) {
    Array *arr_data = var->variable()->data()->cast(dtypes::FLOAT, ctx_);
    float *data_f = arr_data->pointer<float>();
    ASSERT_TRUE(from >= 0);
    if (to == -1) {
      to = arr_data->size();
    } else {
      ASSERT_TRUE(to <= arr_data->size());
    }

    for (int i = from; i < to; ++i) {
      ASSERT_EQ(data_f[i], v);
    }
  }
};

TEST_F(FunctionVariadicInputTest, concatenate_2_param) {
  fill_data(var1_, 1.0);
  fill_data(var2_, 2.0);
  CgVariablePtr v = nbla::functions::concatenate({var1_, var2_}, 0);
  v->forward();
  assert_eq(v, 0, var1_->variable()->size(), 1.0);
  assert_eq(v, var1_->variable()->size(), var2_->variable()->size(), 2.0);
  auto shape = v->variable()->shape();
  EXPECT_EQ(shape[0], 4);
}

TEST_F(FunctionVariadicInputTest, concatenate_3_param) {
  fill_data(var1_, 1.0);
  fill_data(var2_, 2.0);
  vector<CgVariablePtr> v =
      nbla::functions::concatenate(ctx_, {var1_, var2_}, 0);
  v[0]->forward();
  assert_eq(v[0], 0, var1_->variable()->size(), 1.0);
  assert_eq(v[0], var1_->variable()->size(), var2_->variable()->size(), 2.0);
  auto shape = v[0]->variable()->shape();
  EXPECT_EQ(shape[0], 4);
}
}
