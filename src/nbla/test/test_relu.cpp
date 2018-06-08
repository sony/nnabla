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

// test_relu.cpp

#include "gtest/gtest.h"
#include <nbla/common.hpp>
#include <nbla/function/relu.hpp>

#include <random>

namespace nbla {

class ReLUTest : public ::testing::Test {
public:
protected:
  shared_ptr<Variable> in_;
  shared_ptr<Variable> out_;
  Context ctx_;
  dtypes dtype_;
  Shape_t shape_;
  virtual void SetUp() {
    // NNabla::init();
    ctx_.array_class = "CpuArray";
    ctx_.backend = {"cpu:float"};
    dtype_ = dtypes::FLOAT;
    shape_ = Shape_t{2, 3, 4, 5};
    in_.reset(new Variable(shape_));
    out_.reset(new Variable());

    std::random_device r;
    std::mt19937 e;
    std::normal_distribution<float> normal_dist(0, 1);
    float *data = in_->data()->cast(dtype_, ctx_)->pointer<float>();
    for (int i = 0; i < in_->size(); i++) {
      data[i] = normal_dist(e);
    }
  }

  // virtual void TearDown() {}
};

TEST_F(ReLUTest, Create) {
  shared_ptr<Function> relu = create_ReLU(ctx_, false);
  ASSERT_EQ(dtypes::FLOAT, relu->in_types()[0]);
  ASSERT_EQ(dtypes::FLOAT, relu->out_types()[0]);
}

TEST_F(ReLUTest, ForwardBackward) {
  shared_ptr<Function> relu = create_ReLU(ctx_, false);
  relu->setup(Variables{in_.get()}, Variables{out_.get()});
  ASSERT_EQ(in_->shape(), out_->shape());
  relu->forward(Variables{in_.get()}, Variables{out_.get()});
  relu->backward(Variables{in_.get()}, Variables{out_.get()}, {true}, {false});
}
}
