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

// common_fixture.hpp

#include <gtest/gtest.h>
#include <macros.hpp>
#include <memory>
#include <vector>

#include <nbla/computation_graph/computation_graph.hpp>
#include <nbla/initializer.hpp>

using std::shared_ptr;
using std::make_shared;
using std::vector;
using namespace nbla;

#ifndef ALL_COMMON_FIXTURE_HPP
#define ALL_COMMON_FIXTURE_HPP

vector<shared_ptr<Initializer>> create_init_list();

class ParametricFunctionsTest : public ::testing::Test {
protected:
  Context ctx_;

  void SetUp() override { ctx_.array_class = "CpuArray"; }

  void checkDataMinMax(const VariablePtr &var, float min, float max) {
    auto *d = var->cast_data_and_get_pointer<float>(ctx_, false);

    REP(i, var->size()) {
      EXPECT_GE(max, d[i]);
      EXPECT_LE(min, d[i]);
    }
  }

  void checkDataConstant(const VariablePtr &var, float value) {
    auto *d = var->cast_data_and_get_pointer<float>(ctx_, false);

    REP(i, var->size()) { EXPECT_FLOAT_EQ(value, d[i]); }
  }

  void checkVariablesDataSame(const VariablePtr &var, const VariablePtr &ref) {
    EXPECT_EQ(ref->shape(), var->shape());

    auto *var_d = var->cast_data_and_get_pointer<float>(ctx_, false);
    auto *ref_d = ref->cast_data_and_get_pointer<float>(ctx_, false);

    REP(i, ref->size()) { EXPECT_FLOAT_EQ(ref_d[i], var_d[i]); }
  }

  void checkVariablesGradSame(const VariablePtr &var, const VariablePtr &ref) {
    EXPECT_EQ(ref->shape(), var->shape());

    auto *var_g = var->cast_grad_and_get_pointer<float>(ctx_, false);
    auto *ref_g = ref->cast_grad_and_get_pointer<float>(ctx_, false);

    REP(i, ref->size()) { EXPECT_FLOAT_EQ(ref_g[i], var_g[i]); }
  }
};

#endif // ALL_COMMON_FIXTURE_HPP
