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

// test_convolution.cpp

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

// (input shape, base_axis), with_bias, fix_parameters, w_init, b_init
// currently fix_parameters dosen`t work.
#define ConvolutionOptsParams                                                  \
  tuple<pair<Shape_t, int>, bool, bool, shared_ptr<Initializer>,               \
        shared_ptr<Initializer>>

class ConvolutionOptsTest
    : public ParametricFunctionsTest,
      public ::testing::WithParamInterface<ConvolutionOptsParams> {};

TEST_F(ConvolutionOptsTest, CheckDefault) {
  auto convolutionOpts = PF::ConvolutionOpts();

  EXPECT_TRUE(convolutionOpts.with_bias());
  EXPECT_FALSE(convolutionOpts.fix_parameters());
  EXPECT_EQ(nullptr, convolutionOpts.w_init());
  EXPECT_EQ(nullptr, convolutionOpts.b_init());

  // below is assertions for the defaults of base Opts
  EXPECT_EQ(1, convolutionOpts.group());
  EXPECT_EQ(vector<int>({0, 0}), convolutionOpts.pad());
  EXPECT_EQ(vector<int>({1, 1}), convolutionOpts.stride());
  EXPECT_EQ(vector<int>({1, 1}), convolutionOpts.dilation());
}

INSTANTIATE_TEST_CASE_P(
    ParameterizeConvolutionOpts, ConvolutionOptsTest,
    ::testing::Combine(::testing::Values(make_pair(Shape_t({2, 3, 8, 8}), 1),
                                         make_pair(Shape_t({1, 2, 3, 5, 5}),
                                                   2)),
                       ::testing::Bool(), ::testing::Bool(),
                       ::testing::ValuesIn(create_init_list()),
                       ::testing::ValuesIn(create_init_list())));

TEST_P(ConvolutionOptsTest, SetUp) {
  ConvolutionOptsParams p = GetParam();
  auto shape_axis_pair = std::get<0>(p);
  bool with_bias = std::get<1>(p);
  bool fix_parameters = std::get<2>(p);
  auto w_init = std::get<3>(p);
  auto b_init = std::get<4>(p);

  Shape_t input_shape = shape_axis_pair.first;
  int base_axis = shape_axis_pair.second;

  ParameterDirectory params;
  auto x = make_shared<CgVariable>(input_shape);

  // check setter/getter
  auto convolutionOpts = PF::ConvolutionOpts();
  convolutionOpts.with_bias(with_bias);
  convolutionOpts.fix_parameters(fix_parameters);
  convolutionOpts.w_init(w_init.get());
  convolutionOpts.b_init(b_init.get());

  EXPECT_EQ(with_bias, convolutionOpts.with_bias());
  EXPECT_EQ(fix_parameters, convolutionOpts.fix_parameters());
  EXPECT_EQ(w_init.get(), convolutionOpts.w_init());
  EXPECT_EQ(b_init.get(), convolutionOpts.b_init());

  vector<int> kernel_shape = {2, 2};
  int n_map_in = static_cast<int>(input_shape[base_axis]);
  int n_map_out = 5;
  int group = convolutionOpts.group();
  auto pad = convolutionOpts.pad();
  auto stride = convolutionOpts.stride();
  auto dilation = convolutionOpts.dilation();

  auto h = PF::convolution(x, base_axis, n_map_out, kernel_shape, params,
                           convolutionOpts);

  // bias check
  int param_size = with_bias ? 2 : 1;
  EXPECT_EQ(param_size, params.get_parameters().size());

  // fix_parameters
  // Currently this argument does nothing and there is no testing.

  // w_init
  ASSERT_TRUE(params.get_parameter("conv/W") != nullptr);
  VariablePtr w = params.get_parameter("conv/W")->variable();

  // check shape
  Shape_t w_shape = {n_map_out, int(n_map_in / group)};
  int kernel_dim_product = 1;
  for (int kernel_k : kernel_shape) {
    w_shape.push_back(kernel_k);
    kernel_dim_product *= kernel_k;
  }

  EXPECT_EQ(w_shape, w->shape());

  // check values
  if (w_init == nullptr) {
    // using default initializer: UniformInitializer(-range, range)
    float range =
        calc_uniform_lim_glorot(n_map_in, n_map_out, kernel_dim_product);
    checkDataMinMax(w, -range, range);
  } else {
    // using ConstantInitializer(1)
    checkDataConstant(w, 1);
  }

  // b_init
  if (with_bias) {
    ASSERT_TRUE(params.get_parameter("conv/b") != nullptr);
    VariablePtr b = params.get_parameter("conv/b")->variable();

    // check shape
    EXPECT_EQ(Shape_t{n_map_out}, b->shape());

    // check values
    if (b_init == nullptr) {
      // using default initializer: ConstantInitializer(0)
      checkDataConstant(b, 0);
    } else {
      // using ConstantInitializer(1)
      checkDataConstant(b, 1);
    }
  }

  // call forward/backward
  ASSERT_NO_THROW(h->forward());
  ASSERT_NO_THROW(h->backward());

  // check forward/backward values
  // todo
}
