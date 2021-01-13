// Copyright (c) 2020 Sony Corporation. All Rights Reserved.
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

#include "gtest/gtest.h"
#include <chrono>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <nbla/exception.hpp>
#include <random>
#include <random>
#include <stdio.h>
#include <string>
#include <thread>
#include <vector>

#include <nbla/computation_graph/computation_graph.hpp>
#include <nbla/solver/adam.hpp>
using std::make_shared;

#include <nbla/auto_forward.hpp>
#include <nbla/functions.hpp>
#include <nbla/global_context.hpp>
#include <nbla/parametric_functions.hpp>
#include <nbla_utils/parameters.hpp>

namespace f = nbla::functions;
namespace pf = nbla::parametric_functions;
namespace utl = nbla::utils;

namespace nbla {

class RandomDataIterator {
  std::string name_;
  std::default_random_engine generator_;
  std::uniform_real_distribution<float_t> distrb_x_{0.0, 1.0};
  std::uniform_int_distribution<int> distrb_y_{0, 9};

  template <typename dist_type>
  void fill_with_random(Context ctx, CgVariablePtr x, dist_type dist) {
    float_t *p_x = x->variable()->cast_data_and_get_pointer<float_t>(ctx, true);
    for (int i = 0; i < x->variable()->size(); ++i) {
      *p_x = dist(generator_);
    }
  }

public:
  RandomDataIterator(const char *name) : name_(name) {}

  void provide_data(Context ctx, int batch_size, CgVariablePtr x,
                    CgVariablePtr y) {
    fill_with_random(ctx, x, distrb_x_);
    fill_with_random(ctx, y, distrb_y_);
  }
};

CgVariablePtr model(CgVariablePtr x, ParameterDirectory parameters) {
  auto h = pf::convolution(x, 1, 16, {5, 5}, parameters["conv1"]);
  h = f::max_pooling(h, {2, 2}, {2, 2}, true, {0, 0}, false);
  h = f::relu(h, false);
  h = pf::convolution(h, 1, 16, {5, 5}, parameters["conv2"]);
  h = f::max_pooling(h, {2, 2}, {2, 2}, true, {0, 0}, false);
  h = f::relu(h, false);
  h = pf::affine(h, 1, 50, parameters["affine3"]);
  h = f::relu(h, false);
  h = pf::affine(h, 1, 10, parameters["affine4"]);
  return h;
}

bool traing_a_model(nbla::Context ctx) {

  // Create a context for cpu
  nbla::Context cpu_ctx{{"cpu:float"}, "CpuCachedArray", "0"};

  RandomDataIterator train_data_provider("train");
  RandomDataIterator test_data_provider("test");

  // Setup context
  SingletonManager::get<GlobalContext>()->set_current_context(ctx);

  // Build network
  ParameterDirectory params;

  int batch_size = 4;
  float mean_t_loss = 0.;
  float mean_t_err = 0.;
  auto x = make_shared<CgVariable>(Shape_t({batch_size, 1, 28, 28}), false);
  auto t = make_shared<CgVariable>(Shape_t({batch_size, 1}), false);
  auto h = model(x, params);
  auto loss = f::mean(f::softmax_cross_entropy(h, t, 1), {0, 1}, false);
  auto err = f::mean(f::top_n_error(h, t, 1, 1), {0, 1}, false);

  // Setup solver and input learnable parameters
  auto adam = create_AdamSolver(ctx, 0.001, 0.9, 0.999, 1.0e-8);
  adam->set_parameters(params.get_parameters());

  int max_iter = 10;
  for (int iter = 0; iter < max_iter; iter++) {

    // Get batch and copy to input variables
    train_data_provider.provide_data(cpu_ctx, batch_size, x, t);

    // Execute forward, backward and update
    adam->zero_grad();
    loss->forward(/*clear_buffer=*/false, /*clear_no_need_grad=*/true);
    loss->variable()->grad()->fill(1.0);
    loss->backward(/*NdArrayPtr grad =*/nullptr,
                   /*bool clear_buffer = */ true);
    adam->update();

    // Get and print the average loss
    float_t *t_loss_d =
        loss->variable()->cast_data_and_get_pointer<float_t>(cpu_ctx, false);
    mean_t_loss += t_loss_d[0];

    test_data_provider.provide_data(cpu_ctx, batch_size, x, t);
    err->forward(/*clear_buffer=*/false, /*clear_no_need_grad=*/true);
    float_t *t_err_d =
        err->variable()->cast_data_and_get_pointer<float_t>(cpu_ctx, false);
    mean_t_err += t_err_d[0];
  }

  return true;
}

TEST(MultiThreadTrainingTest, TestTrainingThread8) {
  // Create a context (the following setting is recommended.)
  nbla::Context ctx{{"cpu:float"}, "CpuCachedArray", "0"};
  nbla::init_cpu();
  std::vector<std::thread> threads;

  for (int i = 0; i < 8; ++i) {
    threads.push_back(std::thread(traing_a_model, ctx));
  }

  for (auto &th : threads) {
    if (th.joinable()) {
      th.join();
    }
  }
}

} // namespace nbla
