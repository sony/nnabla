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

// test_initializer.cpp

#include <macros.hpp>

#include <climits>
#include <gtest/gtest.h>
#include <memory>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include <nbla/computation_graph/computation_graph.hpp>
#include <nbla/exception.hpp>
#include <nbla/initializer.hpp>

using std::shared_ptr;
using std::make_shared;
using std::vector;
using std::pair;
using std::tuple;
using namespace nbla;

class InitializerTest : public ::testing::Test {
protected:
  Context ctx_;
  CgVariablePtr var_;

  virtual void SetUp() override {
    ctx_.array_class = "CpuArray";

    var_ = make_shared<CgVariable>(Shape_t({5, 5}), false);

    resetVariable();
  }

  void resetVariable() {
    // init all values by 0
    auto *d = var_->variable()->cast_data_and_get_pointer<float>(ctx_, true);

    REP(i, var_->variable()->size()) { d[i] = 0; }
  }

  template <typename T> T *initVar(Initializer *initializer) {
    initializer->initialize(var_->variable()->data());

    return var_->variable()->cast_data_and_get_pointer<T>(ctx_, false);
  }
};

// UniformInitializerTest

#define UniformInitializerParams tuple<float, float>

class UniformInitializerTest
    : public InitializerTest,
      public ::testing::WithParamInterface<UniformInitializerParams> {
protected:
  void checkUniformInitializer(const shared_ptr<UniformInitializer> &init,
                               float min, float max) {
    auto *d = initVar<float>(init.get());

    REP(i, var_->variable()->size()) {
      EXPECT_GE(max, d[i]);
      EXPECT_LE(min, d[i]);
    }
  }
};

TEST_F(UniformInitializerTest, CtorWithoutArgs) {
  // default constructor -> init by [-1, 1]
  auto initializer = make_shared<UniformInitializer>();

  checkUniformInitializer(initializer, -1, 1);
}

TEST_P(UniformInitializerTest, CtorWithArgs) {
  UniformInitializerParams p = GetParam();
  float min = std::get<0>(p);
  float max = std::get<1>(p);

  if (min > max) {
    EXPECT_THROW(UniformInitializer(min, max), Exception);
  } else {
    auto initializer = make_shared<UniformInitializer>(min, max);
    checkUniformInitializer(initializer, min, max);
  }
}

INSTANTIATE_TEST_CASE_P(Parameterize, UniformInitializerTest,
                        ::testing::Combine(::testing::Values(-1, 0, 1),
                                           ::testing::Values(-1, 0, 1)));

// ConstantInitializerTest
#define ConstantInitializerParams float

class ConstantInitializerTest
    : public InitializerTest,
      public ::testing::WithParamInterface<ConstantInitializerParams> {
protected:
  void checkConstantInitializer(const shared_ptr<ConstantInitializer> &init,
                                float value) {
    auto *d = initVar<float>(init.get());

    REP(i, var_->variable()->size()) { EXPECT_FLOAT_EQ(value, d[i]); }
  }
};

TEST_F(ConstantInitializerTest, CtorWithoutArgs) {
  // default constructor -> init by 0
  auto initializer = make_shared<ConstantInitializer>();

  checkConstantInitializer(initializer, 0.0);
}

TEST_P(ConstantInitializerTest, CtorWithArgs) {
  ConstantInitializerParams value = GetParam();

  auto initializer = make_shared<ConstantInitializer>(value);
  checkConstantInitializer(initializer, value);
}

INSTANTIATE_TEST_CASE_P(Parameterize, ConstantInitializerTest,
                        ::testing::Values(-1, 0, 1));

// NormalInitializerTest
#define NormalInitializerParams tuple<float, float>

class NormalInitializerTest
    : public InitializerTest,
      public ::testing::WithParamInterface<NormalInitializerParams> {
protected:
  void checkNormalInitializer(const shared_ptr<NormalInitializer> &init,
                              float mu, float sigma) {
    // just check call
    EXPECT_NO_THROW(initVar<float>(init.get()));
  }
};

TEST_F(NormalInitializerTest, CtorWithoutArgs) {
  // default constructor -> init by 0
  auto initializer = make_shared<NormalInitializer>();

  checkNormalInitializer(initializer, 0.0, 1.0);
}

TEST_P(NormalInitializerTest, CtorWithArgs) {
  NormalInitializerParams p = GetParam();
  float mu = std::get<0>(p);
  float sigma = std::get<1>(p);

  if (sigma < 0) {
    EXPECT_THROW(NormalInitializer(mu, sigma), Exception);
  } else {
    auto initializer = make_shared<NormalInitializer>(mu, sigma);
    checkNormalInitializer(initializer, mu, sigma);
  }
}

INSTANTIATE_TEST_CASE_P(Parameterize, NormalInitializerTest,
                        ::testing::Combine(::testing::Values(-1, 0, 1),
                                           ::testing::Values(-1, 0, 1)));

// UniformIntInitializerTest
#define UniformIntInitializerParams tuple<int, int>

class UniformIntInitializerTest
    : public InitializerTest,
      public ::testing::WithParamInterface<UniformIntInitializerParams> {
protected:
  void checkUniformIntInitializer(const shared_ptr<UniformIntInitializer> &init,
                                  int min, int max) {
    auto *d = initVar<int>(init.get());

    REP(i, var_->variable()->size()) {
      EXPECT_LE(min, d[i]);
      EXPECT_GE(max, d[i]);
    }
  }
};

TEST_F(UniformIntInitializerTest, CtorWithoutArgs) {
  // default constructor -> init by [0, INT_MAX]
  auto initializer = make_shared<UniformIntInitializer>();

  checkUniformIntInitializer(initializer, 0, INT_MAX);
}

TEST_P(UniformIntInitializerTest, CtorWithArgs) {
  UniformIntInitializerParams p = GetParam();
  int min = std::get<0>(p);
  int max = std::get<1>(p);

  if (min > max) {
    EXPECT_THROW(UniformIntInitializer(min, max), Exception);
  } else {
    auto initializer = make_shared<UniformIntInitializer>(min, max);
    checkUniformIntInitializer(initializer, min, max);
  }
}

INSTANTIATE_TEST_CASE_P(Parameterize, UniformIntInitializerTest,
                        ::testing::Combine(::testing::Range(-2, 3, 2),
                                           ::testing::Range(-3, 4, 1)));
