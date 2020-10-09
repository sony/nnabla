// Copyright (c) 2019 Sony Corporation. All Rights Reserved.
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
std::random_device seed_gen;
std::default_random_engine engine(seed_gen());
std::uniform_real_distribution<> uniform(0.0, 1.0);
std::normal_distribution<> normal(0.0, 1.0);

/******************************************/
// Example of VAT Model
/******************************************/
CgVariablePtr mlp_net(CgVariablePtr x, int n_h, int n_y,
                      ParameterDirectory params, bool test) {
  // from python examples
  //"""
  // Function for building multi-layer-perceptron with batch_normalization
  //
  // Args:
  //    x(`~nnabla.Variable`): N-D array
  //    n_h(int): number of units in an intermediate layer
  //    n_y(int): number of classes
  //    test: operation type train=True, test=False
  //
  // Returns:
  //    ~nnabla.Variable: h
  //"""
  auto h1 = pf::affine(x, 1, n_h, params["fc1"]);
  auto r1 = f::relu(pf::batch_normalization(h1, !test, params["fc1"]), false);
  auto h2 = pf::affine(r1, 1, n_h, params["fc2"]);
  auto r2 = f::relu(pf::batch_normalization(h2, !test, params["fc3"]), false);
  auto h3 = pf::affine(r2, 1, n_y, params["fc3"]);
  return h3;
}

CgVariablePtr distance(CgVariablePtr y0, CgVariablePtr y1) {
  // from python examples
  // def distance(y0, y1):
  //"""
  // Distance function is Kullback-Leibler Divergence for categorical
  // distribution
  //"""
  return f::kl_multinomial(f::softmax(y0, 1), f::softmax(y1, 1), 1);
}

/******************************************/
// Example of Virtual Adversarial Training
/******************************************/
bool vat_training_with_static_graph(nbla::Context ctx) {
  // from python examples
  //"""
  // Main script.
  //
  // Steps:
  //  * Get and set context.
  //  * Load Dataset
  //  * Initialize DataIterator.
  //  * Create Networks
  //  *   Network for training with labeled data
  //  *   Network for training with unlabeled data
  //  *   Network for evaluation with validation data
  //  * Create Solver.
  //  * Training Loop.
  //  *   Training
  //  *     by Labeled Data
  //  *       Calculate loss with labeled data
  //  *     by Unlabeled Data
  //  *       Calculate virtual adversarial loss
  //  *       Calculate loss with virtual adversarial noise
  //  """

  // Create a context for cpu
  nbla::Context cpu_ctx{{"cpu:float"}, "CpuCachedArray", "0"};

  // Setup context
  SingletonManager::get<GlobalContext>()->set_current_context(ctx);

  //  # Create networks
  //  # Network
  ParameterDirectory params;

  // Load pretrained parameter if it has.
  utl::load_parameters(params, "vat_param.protobuf");

  int n_h = 1200;
  int n_y = 10;

  //  # Network for training with labeled data
  int batch_size_l = 100;
  auto xl = make_shared<CgVariable>(Shape_t({batch_size_l, 1, 28, 28}), false);
  auto tl = make_shared<CgVariable>(Shape_t({batch_size_l, 1}), false);
  auto yl = mlp_net(xl, n_h, n_y, params, false);
  auto loss_l = f::mean(f::softmax_cross_entropy(yl, tl, 1), {0, 1}, false);

  //  # Network for training with unlabeled data
  int batch_size_u = 250;
  float xi = 10.0;
  float eps = 1.5;
  auto xu = make_shared<CgVariable>(Shape_t({batch_size_u, 1, 28, 28}), false);
  auto tu = make_shared<CgVariable>(Shape_t({batch_size_u, 1}), false);
  auto yu = mlp_net(xu, n_h, n_y, params, false);
  auto y1 = make_shared<CgVariable>(yu->variable(), true);
  y1->set_need_grad(false);

  auto noise =
      make_shared<CgVariable>(Shape_t({batch_size_u, 1, 28, 28}), true);
  auto r =
      noise /
      f::pow_scalar((f::sum(f::pow_scalar(noise, 2, false), {1, 2, 3}, true)),
                    0.5, false);
  r->set_persistent(true);
  auto y2 = mlp_net(xu + xi * r, n_h, n_y, params, false);
  auto y3 = mlp_net(xu + eps * r, n_h, n_y, params, false);
  auto loss_k = f::mean(distance(y1, y2), {0}, false);
  auto loss_u = f::mean(distance(y1, y3), {0}, false);

  //  # Network for evaluation with validation data
  int batch_size_v = 10000;
  auto xv = make_shared<CgVariable>(Shape_t({batch_size_v, 1, 28, 28}), false);
  auto hv = mlp_net(xv, n_h, n_y, params, true);
  auto tv = make_shared<CgVariable>(Shape_t({batch_size_v, 1}), false);
  auto err = f::mean(f::top_n_error(hv, tv, 1, 1), {0, 1}, false);

  //  # Solver.
  float learning_rate = 2.0e-3;
  auto solver = create_AdamSolver(ctx, learning_rate, 0.9, 0.999, 1.0e-8);
  solver->set_parameters(params.get_parameters());

  //  # Training loop.
  MnistDataIterator train_data_iterator("train");
  MnistDataIterator valid_data_iterator("valid");
  auto labeled_set = train_data_iterator.get_batch(100);
  int max_iter = 24000;
  float learning_rate_decay = 0.9;
  int learning_rate_decay_interval = 240;
  float weight_decay = 0.;
  int max_iter_power_method = 1;
  int n_val_interval = 240;
  float mean_tloss = 0.;
  FILE *fp;
  fp = fopen("log.txt", "wt");
  if (fp == NULL) {
    fprintf(stderr, "Error in opening log file.");
    return false;
  }
  try {
    for (int iter = 0; iter < max_iter; iter++) {
      // Training with labeled data
      set_data(cpu_ctx, labeled_set, xl, tl);
      solver->zero_grad();
      loss_l->forward(/*clear_buffer=*/false, /*clear_no_need_grad=*/true);
      loss_l->variable()->grad()->fill(1.0);
      loss_l->backward(/*NdArrayPtr grad=*/nullptr, /*clear_buffer=*/true);
      solver->weight_decay(weight_decay);
      solver->update();

      // Training with unlabeled data
      train_data_iterator.provide_data(cpu_ctx, batch_size_u, xu, tu);
      yu->forward(/*clear_buffer=*/false, /*clear_no_need_grad=*/false);

      // Calculating virtual adversarial noise
      float_t *n_d =
          noise->variable()->cast_data_and_get_pointer<float_t>(cpu_ctx, true);
      for (int i = 0; i < noise->variable()->size(); i++, n_d++)
        *n_d = normal(engine);

      for (int k = 0; k < max_iter_power_method; k++) {
        r->variable()->grad()->fill(0.0);
        loss_k->forward(/*clear_buffer=*/false, /*clear_no_need_grad=*/true);
        loss_k->variable()->grad()->fill(1.0);
        loss_k->backward(/*NdArrayPtr grad=*/nullptr, /*clear_buffer=*/true);

        n_d = noise->variable()->cast_data_and_get_pointer<float_t>(cpu_ctx,
                                                                    true);
        float_t *r_g =
            r->variable()->cast_grad_and_get_pointer<float_t>(cpu_ctx, false);
        for (int i = 0; i < noise->variable()->size(); i++, n_d++, r_g++)
          *n_d = *r_g;
      }

      // Updating with virtual adversarial noise
      solver->zero_grad();
      loss_u->forward(/*clear_buffer=*/false, /*clear_no_need_grad=*/true);
      loss_u->variable()->grad()->fill(1.0);
      loss_u->backward(/*NdArrayPtr grad=*/nullptr, /*clear_buffer=*/true);
      solver->weight_decay(weight_decay);
      solver->update();

      //      ##### Learning rate update #####
      if (iter % learning_rate_decay_interval == 0) {
        solver->set_learning_rate(solver->learning_rate() *
                                  learning_rate_decay);
      }

      // Monitor loss and error rate
      float_t *loss_l_d =
          loss_l->variable()->cast_data_and_get_pointer<float_t>(cpu_ctx,
                                                                 false);
      float_t *loss_u_d =
          loss_u->variable()->cast_data_and_get_pointer<float_t>(cpu_ctx,
                                                                 false);
      mean_tloss += *loss_l_d + *loss_u_d;

      if ((iter + 1) % n_val_interval == 0) {

        mean_tloss /= n_val_interval;

        valid_data_iterator.provide_data(cpu_ctx, batch_size_v, xv, tv);
        err->forward(/*clear_buffer=*/false, /*clear_no_need_grad=*/true);
        float_t *err_d =
            err->variable()->cast_data_and_get_pointer<float_t>(cpu_ctx, false);

        fprintf(fp, "iter: %d, tloss: %f, verr: %f\n", iter + 1, mean_tloss,
                *err_d);
        fprintf(stdout, "iter: %d, tloss: %f, verr: %f\n", iter + 1, mean_tloss,
                *err_d);
        mean_tloss = 0;

        // Save parameters as a snapshot.
        utl::save_parameters(params, "saved_vat_param.protobuf");
      }
    }
  } catch (...) {
    cout << "Exception in vat_training_with_static_graph.\n";
    fclose(fp);
    return false;
  }
  fclose(fp);
  return true;
}

bool vat_training_with_dynamic_graph(nbla::Context ctx) {
  // Create a context for cpu
  nbla::Context cpu_ctx{{"cpu:float"}, "CpuCachedArray", "0"};

  // Setup context
  SingletonManager::get<GlobalContext>()->set_current_context(ctx);

  // Setup auto_forward
  SingletonManager::get<AutoForward>()->set_auto_forward(true);

  // Setup parameters
  ParameterDirectory params;

  // Load pretrained parameter if it has.
  utl::load_parameters(params, "vat_param_d.protobuf");

  // Setup solver.
  float learning_rate = 2.0e-3;
  auto solver = create_AdamSolver(ctx, learning_rate, 0.9, 0.999, 1.0e-8);

  // Setup data iterators
  MnistDataIterator train_data_iterator("train");
  MnistDataIterator valid_data_iterator("valid");
  auto labeled_set = train_data_iterator.get_batch(100);

  // Network hyper parameters
  int n_h = 1200;
  int n_y = 10;

  // Batch size of labeled, unlabeled and validation data
  int batch_size_l = 100;
  int batch_size_u = 250;
  int batch_size_v = 10000;

  // Hyper parameter for VAT
  float xi = 10.0;
  float eps = 1.5;

  //  # Training loop.
  int max_iter = 24000;
  float learning_rate_decay = 0.9;
  int learning_rate_decay_interval = 240;
  float weight_decay = 0.;
  int max_iter_power_method = 1;
  int n_val_interval = 240;
  float mean_tloss = 0.;
  FILE *fp;
  fp = fopen("log.txt", "wt");
  if (fp == NULL) {
    fprintf(stderr, "Error in opening log file.");
    return false;
  }
  try {
    for (int iter = 0; iter < max_iter; iter++) {
      // Training with labeled data
      auto xl =
          make_shared<CgVariable>(Shape_t({batch_size_l, 1, 28, 28}), false);
      auto tl = make_shared<CgVariable>(Shape_t({batch_size_l, 1}), false);
      set_data(cpu_ctx, labeled_set, xl, tl);
      auto yl = mlp_net(xl, n_h, n_y, params, false);
      auto loss_l = f::mean(f::softmax_cross_entropy(yl, tl, 1), {0, 1}, false);

      solver->set_parameters(params.get_parameters(),
                             /*reset=*/false, /*retain_state=*/true);
      solver->zero_grad();
      loss_l->variable()->grad()->fill(1.0);
      loss_l->backward(/*NdArrayPtr grad=*/nullptr, /*clear_buffer=*/true);
      solver->weight_decay(weight_decay);
      solver->update();

      // Training with unlabeled data
      auto xu =
          make_shared<CgVariable>(Shape_t({batch_size_u, 1, 28, 28}), false);
      auto tu = make_shared<CgVariable>(Shape_t({batch_size_u, 1}), false);
      train_data_iterator.provide_data(cpu_ctx, batch_size_u, xu, tu);
      auto yu = mlp_net(xu, n_h, n_y, params, false);
      auto y1 = make_shared<CgVariable>(yu->variable(), true);
      y1->set_need_grad(false);

      // Calculating virtual adversarial noise
      auto noise =
          make_shared<CgVariable>(Shape_t({batch_size_u, 1, 28, 28}), true);
      float_t *n_d =
          noise->variable()->cast_data_and_get_pointer<float_t>(cpu_ctx, true);
      for (int i = 0; i < noise->variable()->size(); i++, n_d++)
        *n_d = normal(engine);
      auto r = noise / f::pow_scalar((f::sum(f::pow_scalar(noise, 2, false),
                                             {1, 2, 3}, true)),
                                     0.5, false);
      r->set_persistent(true);

      for (int k = 0; k < max_iter_power_method; k++) {
        r->variable()->grad()->fill(0.0);
        auto y2 = mlp_net(xu + xi * r, n_h, n_y, params, false);
        auto loss_k = f::mean(distance(y1, y2), {0}, false);
        loss_k->variable()->grad()->fill(1.0);
        loss_k->backward(/*NdArrayPtr grad=*/nullptr, /*clear_buffer=*/true);

        n_d = noise->variable()->cast_data_and_get_pointer<float_t>(cpu_ctx,
                                                                    true);
        float_t *r_g =
            r->variable()->cast_grad_and_get_pointer<float_t>(cpu_ctx, false);
        for (int i = 0; i < noise->variable()->size(); i++, n_d++, r_g++)
          *n_d = *r_g;
      }

      // Updating with virtual adversarial noise
      r = noise / f::pow_scalar(
                      (f::sum(f::pow_scalar(noise, 2, false), {1, 2, 3}, true)),
                      0.5, false);
      auto y3 = mlp_net(xu + eps * r, n_h, n_y, params, false);
      auto loss_u = f::mean(distance(y1, y3), {0}, false);
      solver->set_parameters(params.get_parameters(),
                             /*reset=*/false, /*retain_state=*/true);
      solver->zero_grad();
      loss_u->variable()->grad()->fill(1.0);
      loss_u->backward(/*NdArrayPtr grad=*/nullptr, /*clear_buffer=*/true);
      solver->weight_decay(weight_decay);
      solver->update();

      //      ##### Learning rate update #####
      if (iter % learning_rate_decay_interval == 0) {
        solver->set_learning_rate(solver->learning_rate() *
                                  learning_rate_decay);
      }

      // Monitor loss and error rate
      float_t *loss_l_d =
          loss_l->variable()->cast_data_and_get_pointer<float_t>(cpu_ctx,
                                                                 false);
      float_t *loss_u_d =
          loss_u->variable()->cast_data_and_get_pointer<float_t>(cpu_ctx,
                                                                 false);
      mean_tloss += *loss_l_d + *loss_u_d;

      if ((iter + 1) % n_val_interval == 0) {

        mean_tloss /= n_val_interval;

        auto xv =
            make_shared<CgVariable>(Shape_t({batch_size_v, 1, 28, 28}), false);
        auto tv = make_shared<CgVariable>(Shape_t({batch_size_v, 1}), false);
        valid_data_iterator.provide_data(cpu_ctx, batch_size_v, xv, tv);
        auto hv = mlp_net(xv, n_h, n_y, params, true);
        auto err = f::mean(f::top_n_error(hv, tv, 1, 1), {0, 1}, false);
        float_t *err_d =
            err->variable()->cast_data_and_get_pointer<float_t>(cpu_ctx, false);

        fprintf(fp, "iter: %d, tloss: %f, verr: %f\n", iter + 1, mean_tloss,
                *err_d);
        fprintf(stdout, "iter: %d, tloss: %f, verr: %f\n", iter + 1, mean_tloss,
                *err_d);
        mean_tloss = 0;

        // Save parameters as a snapshot.
        utl::save_parameters(params, "saved_vat_param_d.protobuf");
      }
    }
  } catch (...) {
    cout << "Exception in vat_training_with_dynamic_graph.\n";
    fclose(fp);
    return false;
  }
  fclose(fp);
  return true;
}
