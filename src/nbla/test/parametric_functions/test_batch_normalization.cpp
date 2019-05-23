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

// test_batch_normalization.cpp

#include <macros.hpp>
#include <parametric_functions/common_fixture.hpp>

#include <gtest/gtest.h>
#include <memory>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include <nbla/computation_graph/computation_graph.hpp>
#include <nbla/exception.hpp>
#include <nbla/parametric_functions.hpp>

using std::shared_ptr;
using std::make_shared;
using std::tuple;
using std::pair;
using std::make_pair;
using std::vector;

using namespace nbla;
namespace PF = nbla::parametric_functions;

// input shape, fix_parameters
// currently fix_parameters dosen`t work.
#define BatchNormOptsParams tuple<Shape_t, bool>

class BatchNormOptsTest
    : public ParametricFunctionsTest,
      public ::testing::WithParamInterface<BatchNormOptsParams> {
protected:
  void checkBatchNormVariable(ParameterDirectory params, string var_name,
                              float init_value) {
    // check existence
    ASSERT_TRUE(params.get_parameter(var_name) != nullptr);

    checkDataConstant(params.get_parameter(var_name)->variable(), init_value);
  }
};

TEST_F(BatchNormOptsTest, CheckDefault) {
  auto opts = PF::BatchNormalizationOpts();

  EXPECT_FALSE(opts.fix_parameters());

  EXPECT_EQ(vector<int>({1}), opts.axes());
  EXPECT_FLOAT_EQ(0.9f, opts.decay_rate());
  EXPECT_FLOAT_EQ(1.0e-5f, opts.eps());
}

INSTANTIATE_TEST_CASE_P(ParameterizeBatchNormOpts, BatchNormOptsTest,
                        ::testing::Combine(::testing::Values(Shape_t({1, 3, 2,
                                                                      2}),
                                                             Shape_t({3, 3})),
                                           ::testing::Bool()));

TEST_P(BatchNormOptsTest, SetUp) {
  BatchNormOptsParams p = GetParam();
  Shape_t input_shape = std::get<0>(p);
  bool fix_parameters = std::get<1>(p);

  ParameterDirectory params;
  auto x = make_shared<CgVariable>(input_shape);

  // check setter/getter
  auto opts = PF::BatchNormalizationOpts();
  opts.fix_parameters(fix_parameters);

  auto h = PF::batch_normalization(x, true, params, opts);

  // fix_parameters
  // Currently this argument does nothing and there is no testing.

  // check # of values created in PF::batcc_normalization
  ASSERT_EQ(4, params.get_parameters().size());

  // check all variable are initialized correctly
  vector<pair<string, float>> var_name_list = {
      make_pair("bn/beta", 0.0f), make_pair("bn/gamma", 1.0f),
      make_pair("bn/mean", 0.0f), make_pair("bn/variance", 1.0f)};
  for (auto nv_pair : var_name_list) {
    checkBatchNormVariable(params, nv_pair.first, nv_pair.second);
  }

  // call forward/backward
  ASSERT_NO_THROW(h->forward());
  ASSERT_NO_THROW(h->backward());

  // check forward/backward values
  // todo
}
