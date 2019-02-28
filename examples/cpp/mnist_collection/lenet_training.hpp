// Copyright (c) 2018 Sony Corporation. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// Include nnabla header files

#include <fstream>
#include <iostream>
#include <random>
#include <stdio.h>
#include <string>
#include <vector>
using namespace std;

#include <nbla/computation_graph/computation_graph.hpp>
#include <nbla/solver/adam.hpp>
using namespace nbla;
using std::make_shared;

#include <nbla/functions.hpp>
#include <nbla/global_context.hpp>
#include <nbla/parametric_functions.hpp>
namespace f = nbla::functions;
namespace pf = nbla::parametric_functions;

#include "mnist_data.hpp"

/******************************************/
// Example of Lenet Model Design
/******************************************/
CgVariablePtr model(CgVariablePtr x, ParameterDirectory parameters) {
  auto h = pf::convolution(x, 1, 16, {5, 5}, parameters["conv1"]);
  h = f::max_pooling(h, {2, 2}, {2, 2}, true, {0, 0});
  h = f::relu(h, false);
  h = pf::convolution(h, 1, 16, {5, 5}, parameters["conv2"]);
  h = f::max_pooling(h, {2, 2}, {2, 2}, true, {0, 0});
  h = f::relu(h, false);
  h = pf::affine(h, 1, 50, parameters["affine3"]);
  h = f::relu(h, false);
  h = pf::affine(h, 1, 10, parameters["affine4"]);
  return h;
}

/******************************************/
// Example of Lenet Classifier Training
/******************************************/
bool lenet_training(nbla::Context ctx) {

  // Create a context for cpu
  nbla::Context cpu_ctx{{"cpu:float"}, "CpuCachedArray", "0"};

  // Create mnist data iterator
  MnistDataIterator train_data_provider("train");
  MnistDataIterator test_data_provider("test");

  // Build network
  SingletonManager::get<GlobalContext>()->set_current_context(ctx);
  ParameterDirectory params;
  int batch_size = 128;
  auto x = make_shared<CgVariable>(Shape_t({batch_size, 1, 28, 28}), false);
  auto t = make_shared<CgVariable>(Shape_t({batch_size, 1}), false);
  auto h = model(x, params);
  auto loss = f::mean(f::softmax_cross_entropy(h, t, 1), {0, 1}, false);
  auto err = f::mean(f::top_n_error(h, t, 1, 1), {0, 1}, false);

  // Setup solver and input learnable parameters
  auto adam = create_AdamSolver(ctx, 0.001, 0.9, 0.999, 1.0e-8);
  adam->set_parameters(params.get_parameters());

  // Execute training
  FILE *fp;
  fp = fopen("log.txt", "wt");
  if (fp == NULL) {
    fprintf(stderr, "Error in opening log file.");
    return false;
  }

  int max_iter = 10000;
  int n_val_iter = 10;
  float mean_t_loss = 0.;
  for (int iter = 0; iter < max_iter; iter++) {

    // Get batch and copy to input variables
    train_data_provider.provide_data(cpu_ctx, batch_size, x, t);

    // Execute forward, backward and update
    adam->zero_grad();
    loss->forward(/*clear_buffer=*/false, /*clear_no_need_grad=*/true);
    loss->variable()->grad()->fill(1.0);
    loss->backward(/*NdArrayPtr grad =*/nullptr, /*bool clear_buffer = */ true);
    adam->update();

    // Get and print the average loss
    float_t *t_loss_d =
        loss->variable()->cast_data_and_get_pointer<float_t>(cpu_ctx, false);
    mean_t_loss += t_loss_d[0];

    if ((iter + 1) % n_val_iter == 0) {

      float mean_v_err = 0.0;
      for (int v_iter = 0; v_iter < 10; v_iter++) {
        test_data_provider.provide_data(cpu_ctx, batch_size, x, t);
        err->forward(/*clear_buffer=*/false, /*clear_no_need_grad=*/true);
        float_t *v_err_d =
            err->variable()->cast_data_and_get_pointer<float_t>(cpu_ctx, false);
        mean_v_err += v_err_d[0];
      }
      mean_t_loss /= n_val_iter;
      mean_v_err /= 10;

      fprintf(fp, "iter: %d, tloss: %f, verr: %f\n", iter + 0, mean_t_loss,
              mean_v_err);
      fprintf(stdout, "iter: %d, tloss: %f, verr: %f\n", iter + 0, mean_t_loss,
              mean_v_err);
      mean_t_loss = 0;
    }
  }
  fclose(fp);
  return true;
}
