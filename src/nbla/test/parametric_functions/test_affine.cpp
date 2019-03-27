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

// test_affine.cpp

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
using std::vector;

using namespace nbla;
namespace PF = nbla::parametric_functions;

// with_bias, fix_parameters, w_init, b_init
// currently fix_parameters dosen`t work.
#define AffineOptsParams                                                       \
  tuple<Shape_t, bool, bool, shared_ptr<Initializer>, shared_ptr<Initializer>>

class AffineOptsTest : public ParametricFunctionsTest,
                       public ::testing::WithParamInterface<AffineOptsParams> {
};

TEST_F(AffineOptsTest, CheckDefault) {
  auto affineOpts = PF::AffineOpts();

  EXPECT_TRUE(affineOpts.with_bias());
  EXPECT_FALSE(affineOpts.fix_parameters());
  EXPECT_EQ(nullptr, affineOpts.w_init());
  EXPECT_EQ(nullptr, affineOpts.b_init());
}

INSTANTIATE_TEST_CASE_P(
    ParameterizeAffineOpts, AffineOptsTest,
    ::testing::Combine(::testing::Values(Shape_t({2, 3, 8, 8}),
                                         Shape_t({5, 2})),
                       ::testing::Bool(), ::testing::Bool(),
                       ::testing::ValuesIn(create_init_list()),
                       ::testing::ValuesIn(create_init_list())));

TEST_P(AffineOptsTest, SetUp) {
  AffineOptsParams p = GetParam();
  Shape_t input_shape = std::get<0>(p);
  bool with_bias = std::get<1>(p);
  bool fix_parameters = std::get<2>(p);
  auto w_init = std::get<3>(p);
  auto b_init = std::get<4>(p);

  ParameterDirectory params;
  auto x = make_shared<CgVariable>(input_shape);

  // check setter/getter
  auto affineOpts = PF::AffineOpts();
  affineOpts.with_bias(with_bias);
  affineOpts.fix_parameters(fix_parameters);
  affineOpts.w_init(w_init.get());
  affineOpts.b_init(b_init.get());

  EXPECT_EQ(with_bias, affineOpts.with_bias());
  EXPECT_EQ(fix_parameters, affineOpts.fix_parameters());
  EXPECT_EQ(w_init.get(), affineOpts.w_init());
  EXPECT_EQ(b_init.get(), affineOpts.b_init());

  int base_axis = 1;
  int n_map_out = 5;

  auto h = PF::affine(x, base_axis, n_map_out, params, affineOpts);

  // bias check
  int param_size = with_bias ? 2 : 1;
  EXPECT_EQ(param_size, params.get_parameters().size());

  // fix_parameters
  // Currently this argument does nothing and there is no testing.

  // w_init
  ASSERT_TRUE(params.get_parameter("affine/W") != nullptr);
  VariablePtr w = params.get_parameter("affine/W")->variable();

  // check shape
  int input_dim = 1;
  FOR(i, base_axis, input_shape.size()) { input_dim *= input_shape[i]; }
  Shape_t w_shape = {input_dim, n_map_out};

  EXPECT_EQ(w_shape, w->shape());

  // check values
  if (w_init == nullptr) {
    // using default initializer: UniformInitializer(-range, range)
    float range = calc_uniform_lim_glorot(static_cast<int>(w_shape[0]),
                                          static_cast<int>(w_shape[1]), 1);
    checkDataMinMax(w, -range, range);
  } else {
    // using ConstantInitializer(1)
    checkDataConstant(w, 1);
  }

  // b_init
  if (with_bias) {
    ASSERT_TRUE(params.get_parameter("affine/b") != nullptr);
    VariablePtr b = params.get_parameter("affine/b")->variable();

    // check shape
    EXPECT_EQ(Shape_t{w_shape[1]}, b->shape());

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
