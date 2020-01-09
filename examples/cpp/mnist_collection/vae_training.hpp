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

#include <nbla/auto_forward.hpp>
#include <nbla/functions.hpp>
#include <nbla/global_context.hpp>
#include <nbla/parametric_functions.hpp>

#include <nbla_utils/parameters.hpp>

namespace f = nbla::functions;
namespace pf = nbla::parametric_functions;
namespace utl = nbla::utils;

#include "mnist_data.hpp"

/******************************************/
// Example of Lenet Model Design
/******************************************/
CgVariablePtr model(CgVariablePtr x, ParameterDirectory parameters) {

  //############################################
  // Encoder of 2 fully connected layers
  //############################################

  // Normalize input
  auto xa = x;

  // Fully connected layers, and Elu replaced from original Softplus.
  auto h1 = f::elu(pf::affine(xa, 1, 500, parameters["fc1"]), 1.0);
  auto h2 = f::elu(pf::affine(h1, 1, 500, parameters["fc2"]), 1.0);

  // The outputs are the parameters of Gauss probability density.
  auto mu = pf::affine(h2, 1, 50, parameters["fc_mu"]);
  auto logvar = pf::affine(h2, 1, 50, parameters["fc_logvar"]);
  auto sigma = f::exp(logvar * 0.5);

  // The prior variable and the reparameterization trick
  Shape_t shape_mu = mu->variable()->shape();
  vector<int> shape_z(shape_mu.size());
  for (int i = 0; i < shape_z.size(); i++)
    shape_z[i] = shape_mu[i];
  auto epsilon = f::randn(0.0, 1.0, shape_z, -1);
  auto z = mu + sigma * epsilon;

  //#############################################
  //  Decoder of 2 fully connected layers       #
  //#############################################

  // Fully connected layers, and Elu replaced from original Softplus.
  auto h3 = f::elu(pf::affine(z, 1, 500, parameters["fc3"]), 1.0);
  auto h4 = f::elu(pf::affine(h3, 1, 500, parameters["fc4"]), 1.0);

  // The outputs are the parameters of Bernoulli probabilities for each pixel.
  Shape_t shape_xa = xa->variable()->shape();
  int n_pb = 1;
  vector<int> shape_pb(shape_xa.size());
  for (int i = 0; i < shape_xa.size(); i++) {
    if (0 < i)
      n_pb *= shape_xa[i];
    shape_pb[i] = shape_xa[i];
  }
  auto h5 = pf::affine(h4, 1, n_pb, parameters["fc5"]);
  auto prob = f::reshape(h5, shape_pb, true);

  //############################################
  // Elbo components and loss objective        #
  //############################################

  // Binarized input
  auto xb = f::greater_equal_scalar(xa, 0.5);

  // E_q(z|x)[log(q(z|x))]
  // without some constant terms that will canceled after summation of loss
  auto logqz = 0.5 * f::sum(1.0 + logvar, {1}, false);

  // E_q(z|x)[log(p(z))]
  // without some constant terms that will canceled after summation of loss
  auto logpz = 0.5 * f::sum(mu * mu + sigma * sigma, {1}, false);

  // E_q(z|x)[log(p(x|z))]
  auto logpx = f::sum(f::sigmoid_cross_entropy(prob, xb), {1, 2, 3}, false);

  // Vae loss, the negative evidence lowerbound
  auto loss = f::mean(logpx + logpz - logqz, {0}, false);

  return loss;
}

/******************************************/
// Example of VAE Training
/******************************************/
bool vae_training_with_static_graph(nbla::Context ctx) {

  // Create a context for cpu
  nbla::Context cpu_ctx{{"cpu:float"}, "CpuCachedArray", "0"};

  // Create mnist data iterator
  MnistDataIterator train_data_provider("train");
  MnistDataIterator test_data_provider("test");

  // Setup context
  SingletonManager::get<GlobalContext>()->set_current_context(ctx);

  // Build network
  ParameterDirectory params;

  // Load pretrained parameter if it has.
  utl::load_parameters(params, "vae_param.protobuf");

  int batch_size = 100;
  auto x = make_shared<CgVariable>(Shape_t({batch_size, 1, 28, 28}), false);
  auto t = make_shared<CgVariable>(Shape_t({batch_size, 1}), false);
  auto loss = model(x, params);

  // Setup solver and input learnable parameters
  float learning_rate = 3.0e-4;
  auto adam = create_AdamSolver(ctx, learning_rate, 0.9, 0.999, 1.0e-8);
  adam->set_parameters(params.get_parameters());

  // Execute training
  FILE *fp;
  fp = fopen("log.txt", "wt");
  if (fp == NULL) {
    fprintf(stderr, "Error in opening log file.");
    return false;
  }

  try {
    int max_iter = 60000;
    int n_val_interval = 10;
    int n_val_iter = 10;
    float mean_loss_t = 0.;
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
      float_t *loss_d =
          loss->variable()->cast_data_and_get_pointer<float_t>(cpu_ctx, false);
      mean_loss_t += loss_d[0];

      if ((iter + 1) % n_val_interval == 0) {

        float mean_loss_v = 0.0;
        for (int v_iter = 0; v_iter < n_val_iter; v_iter++) {
          test_data_provider.provide_data(cpu_ctx, batch_size, x, t);
          loss->forward(/*clear_buffer=*/true, /*clear_no_need_grad=*/true);
          float_t *loss_d =
              loss->variable()->cast_data_and_get_pointer<float_t>(cpu_ctx,
                                                                   false);
          mean_loss_v += loss_d[0];
        }
        mean_loss_t /= n_val_interval;
        mean_loss_v /= n_val_iter;

        fprintf(fp, "iter: %d, tloss: %f, vloss: %f\n", iter + 0, mean_loss_t,
                mean_loss_v);
        fprintf(stdout, "iter: %d, tloss: %f, vloss: %f\n", iter + 0,
                mean_loss_t, mean_loss_v);
        mean_loss_t = 0;

        // Save parameters as a snapshot.
        utl::save_parameters(params, "saved_vae_param.protobuf");
      }
    }
  } catch (...) {
    cout << "Exception in vae_training_with_static_graph.\n";
    fclose(fp);
    return false;
  }
  fclose(fp);
  return true;
}

bool vae_training_with_dynamic_graph(nbla::Context ctx) {

  // Create a context for cpu
  nbla::Context cpu_ctx{{"cpu:float"}, "CpuCachedArray", "0"};

  // Create mnist data iterator
  MnistDataIterator train_data_provider("train");
  MnistDataIterator test_data_provider("test");

  // Setup context
  SingletonManager::get<GlobalContext>()->set_current_context(ctx);

  // Setup auto_forward
  SingletonManager::get<AutoForward>()->set_auto_forward(true);

  // Setup parameter
  ParameterDirectory params;

  // Load pretrained parameter if it has.
  utl::load_parameters(params, "vae_param_d.protobuf");

  // Setup solver and input learnable parameters
  float learning_rate = 3.0e-4;
  auto adam = create_AdamSolver(ctx, learning_rate, 0.9, 0.999, 1.0e-8);

  // Execute training
  FILE *fp;
  fp = fopen("log.txt", "wt");
  if (fp == NULL) {
    fprintf(stderr, "Error in opening log file.");
    return false;
  }

  try {
    int batch_size = 100;
    int max_iter = 60000;
    int n_val_interval = 10;
    int n_val_iter = 10;
    float mean_loss_t = 0.;
    for (int iter = 0; iter < max_iter; iter++) {

      // Get batch and copy to input variables
      auto x = make_shared<CgVariable>(Shape_t({batch_size, 1, 28, 28}), false);
      auto t = make_shared<CgVariable>(Shape_t({batch_size, 1}), false);
      train_data_provider.provide_data(cpu_ctx, batch_size, x, t);
      auto loss = model(x, params);

      // Execute forward, backward and update
      adam->set_parameters(params.get_parameters(), false, true);
      adam->zero_grad();
      loss->variable()->grad()->fill(1.0);
      loss->backward(/*NdArrayPtr grad =*/nullptr,
                     /*bool clear_buffer = */ true);
      adam->update();

      // Get and print the average loss
      float_t *loss_d =
          loss->variable()->cast_data_and_get_pointer<float_t>(cpu_ctx, false);
      mean_loss_t += loss_d[0];

      if ((iter + 1) % n_val_interval == 0) {

        float mean_loss_v = 0.0;
        for (int v_iter = 0; v_iter < n_val_iter; v_iter++) {
          test_data_provider.provide_data(cpu_ctx, batch_size, x, t);
          auto loss = model(x, params);
          float_t *loss_d =
              loss->variable()->cast_data_and_get_pointer<float_t>(cpu_ctx,
                                                                   false);
          mean_loss_v += loss_d[0];
        }
        mean_loss_t /= n_val_interval;
        mean_loss_v /= n_val_iter;

        fprintf(fp, "iter: %d, tloss: %f, vloss: %f\n", iter + 0, mean_loss_t,
                mean_loss_v);
        fprintf(stdout, "iter: %d, tloss: %f, vloss: %f\n", iter + 0,
                mean_loss_t, mean_loss_v);
        mean_loss_t = 0;

        // Save parameters as a snapshot.
        utl::save_parameters(params, "saved_vae_param_d.protobuf");
      }
    }
  } catch (...) {
    cout << "Exception in vae_training_with_dynamic_graph.\n";
    fclose(fp);
    return false;
  }
  fclose(fp);
  return true;
}
