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

#include <assert.h>
#include <random>
#include <stdio.h>
#include <string>
#include <vector>
using namespace std;

#include <nbla/auto_forward.hpp>
#include <nbla/computation_graph/computation_graph.hpp>
#include <nbla/context.hpp>
#include <nbla/functions.hpp>
#include <nbla/global_context.hpp>
#include <nbla/parametric_functions.hpp>
#include <nbla/solver/adam.hpp>

#include <nbla_utils/parameters.hpp>

using namespace nbla;
namespace f = nbla::functions;
namespace pf = nbla::parametric_functions;
using std::make_shared;

#include "mnist_data.hpp"
using std::make_shared;
std::random_device seed_gen;
std::default_random_engine engine(seed_gen());
std::normal_distribution<> normal(0.0, 1.0);

/******************************************/
// Example of DCGAN Model Design
/******************************************/
CgVariablePtr generator(CgVariablePtr z, int max_h, bool test,
                        ParameterDirectory params) {
  // Building generator network which takes (B, Z, 1, 1) inputs and generates
  //(B, 1, 28, 28) outputs.

  assert(max_h / 4 > 0);

  //  # (Z, 1, 1) --> (256, 4, 4)
  pf::DeconvolutionOpts opts1 = pf::DeconvolutionOpts().with_bias(false);
  auto h1 = pf::deconvolution(z, 1, max_h, {4, 4}, params["deconv1"], opts1);
  auto b1 = pf::batch_normalization(h1, !test, params["deconv1"]);
  auto e1 = f::elu(b1, 1.0);

  //  # (256, 4, 4) --> (128, 8, 8)
  pf::DeconvolutionOpts opts2 =
      pf::DeconvolutionOpts().with_bias(false).pad({1, 1}).stride({2, 2});
  auto h2 =
      pf::deconvolution(e1, 1, max_h / 2, {4, 4}, params["deconv2"], opts2);
  auto b2 = pf::batch_normalization(h2, !test, params["deconv2"]);
  auto e2 = f::elu(b2, 1.0);

  //  # (128, 8, 8) --> (64, 16, 16)
  pf::DeconvolutionOpts opts3 =
      pf::DeconvolutionOpts().with_bias(false).pad({1, 1}).stride({2, 2});
  auto h3 =
      pf::deconvolution(e2, 1, max_h / 4, {4, 4}, params["deconv3"], opts3);
  auto b3 = pf::batch_normalization(h3, !test, params["deconv3"]);
  auto e3 = f::elu(b3, 1.0);

  //  # (64, 16, 16) --> (32, 28, 28)
  //  # Convolution with kernel=4, pad=3 and stride=2 transforms a 28 x 28 map
  //  # to a 16 x 16 map. Deconvolution with those parameters behaves like an
  //  # inverse operation, i.e. maps 16 x 16 to 28 x 28.
  pf::DeconvolutionOpts opts4 =
      pf::DeconvolutionOpts().with_bias(false).pad({3, 3}).stride({2, 2});
  auto h4 =
      pf::deconvolution(e3, 1, max_h / 4, {4, 4}, params["deconv4"], opts4);
  auto b4 = pf::batch_normalization(h4, !test, params["deconv4"]);
  auto e4 = f::elu(b4, 1.0);

  //  # (32, 28, 28) --> (1, 28, 28)
  pf::ConvolutionOpts opts5 = pf::ConvolutionOpts().pad({1, 1});
  auto h5 = pf::convolution(e4, 1, 1, {3, 3}, params["conv5"], opts5);
  auto x = f::sigmoid(h5);
  return x;
}

CgVariablePtr discriminator(CgVariablePtr x, int max_h, bool test,
                            ParameterDirectory params) {
  // Building discriminator network which maps a (B, 1, 28, 28) input to
  // a (B, 1).

  assert(max_h / 8 > 0);

  //  # (1, 28, 28) --> (32, 16, 16)
  pf::ConvolutionOpts opts1 =
      pf::ConvolutionOpts().pad({3, 3}).stride({2, 2}).with_bias(false);
  auto h1 = pf::convolution(x, 1, max_h / 8, {3, 3}, params["conv1"], opts1);
  auto b1 = pf::batch_normalization(h1, !test, params["conv1"]);
  auto e1 = f::elu(b1, 1.0);

  //  # (32, 16, 16) --> (64, 8, 8)
  pf::ConvolutionOpts opts2 =
      pf::ConvolutionOpts().pad({1, 1}).stride({2, 2}).with_bias(false);
  auto h2 = pf::convolution(e1, 1, max_h / 4, {3, 3}, params["conv2"], opts2);
  auto b2 = pf::batch_normalization(h2, !test, params["conv2"]);
  auto e2 = f::elu(b2, 1.0);

  //  # (64, 8, 8) --> (128, 4, 4)
  pf::ConvolutionOpts opts3 =
      pf::ConvolutionOpts().pad({1, 1}).stride({2, 2}).with_bias(false);
  auto h3 = pf::convolution(e2, 1, max_h / 2, {3, 3}, params["conv3"], opts3);
  auto b3 = pf::batch_normalization(h3, !test, params["conv3"]);
  auto e3 = f::elu(b3, 1.0);

  //  # (128, 4, 4) --> (256, 4, 4)
  pf::ConvolutionOpts opts4 =
      pf::ConvolutionOpts().pad({1, 1}).with_bias(false);
  auto h4 = pf::convolution(e3, 1, max_h, {3, 3}, params["conv4"], opts4);
  auto b4 = pf::batch_normalization(h4, !test, params["conv4"]);

  //  # (256, 4, 4) --> (1,)
  auto f1 = pf::affine(b4, 1, 1, params["fc1"]);
  return f1;
}

/******************************************/
// Simple Visualization Utils
/******************************************/
void print_array(CgVariablePtr x) {
  nbla::Context cpu_ctx{{"cpu:float"}, "CpuCachedArray", "0"};
  float_t *x_d =
      x->variable()->cast_data_and_get_pointer<float_t>(cpu_ctx, false);
  Shape_t stride_x = x->variable()->strides();
  for (int i = 0; i < x->variable()->size(); i++) {
    int k = x_d[i] < 0.5;
    printf("%d%d", k, k);
    if ((i + 1) % stride_x[2] == 0)
      printf("\n");
    if ((i + 1) % stride_x[1] == 0)
      printf("\n");
    if ((i + 1) % stride_x[0] == 0)
      printf("\n");
  }
}

/******************************************/
// Example of DCGAN Training
/******************************************/
bool dcgan_training_with_static_graph(nbla::Context ctx) {

  // Create a context for cpu
  nbla::Context cpu_ctx{{"cpu:float"}, "CpuCachedArray", "0"};

  // Setup context
  SingletonManager::get<GlobalContext>()->set_current_context(ctx);

  // Build network
  int max_h = 256;
  int batch_size = 64;
  ParameterDirectory params;

  nbla::utils::load_parameters(params, "dcgan_param.protobuf");

  //  # Fake path
  auto z = make_shared<CgVariable>(Shape_t({batch_size, 100, 1, 1}), false);
  auto fake = generator(z, max_h, false, params["gen"]);
  fake->set_persistent(true);

  auto pred_fake = discriminator(fake, max_h, false, params["dis"]);
  auto loss_gen = f::mean(
      f::sigmoid_cross_entropy(pred_fake, f::constant(1, {batch_size, 1})),
      {0, 1}, false);

  auto fake_dis = make_shared<CgVariable>(fake->variable(), true);
  fake_dis->set_need_grad(true);
  auto pred_fake_dis = discriminator(fake, max_h, false, params["dis"]);
  pred_fake_dis->set_persistent(true);
  auto loss_dis = f::mean(
      f::sigmoid_cross_entropy(pred_fake_dis, f::constant(0, {batch_size, 1})),
      {0, 1}, false);

  //  # Real path
  auto x = make_shared<CgVariable>(Shape_t({batch_size, 1, 28, 28}), false);
  auto t = make_shared<CgVariable>(Shape_t({batch_size, 1}), false);
  auto pred_real = discriminator(x, max_h, false, params["dis"]);
  loss_dis = loss_dis + f::mean(f::sigmoid_cross_entropy(
                                    pred_real, f::constant(1, {batch_size, 1})),
                                {0, 1}, false);

  //  # Create Solver.
  float learning_rate = 2.0e-4;
  auto solver_gen = create_AdamSolver(ctx, learning_rate, 0.5, 0.999, 1.0e-8);
  auto solver_dis = create_AdamSolver(ctx, learning_rate, 0.5, 0.999, 1.0e-8);
  solver_gen->set_parameters(params["gen"].get_parameters());
  solver_dis->set_parameters(params["dis"].get_parameters());

  // Create mnist data iterator
  MnistDataIterator train_data_provider("train");

  //# Training loop.
  FILE *fp;
  fp = fopen("log.txt", "wt");
  if (fp == NULL) {
    fprintf(stderr, "Error in opening log file.");
    return false;
  }

  try {
    int max_iter = 20000;
    int n_val_iter = 10;
    float weight_decay = 1.0e-4;
    float mean_loss_gen = 0.;
    float mean_loss_dis = 0.;
    for (int iter = 0; iter < max_iter; iter++) {

      // Get batch and copy to input variables
      train_data_provider.provide_data(cpu_ctx, batch_size, x, t);

      float_t *z_d =
          z->variable()->cast_data_and_get_pointer<float_t>(cpu_ctx, true);
      for (int i = 0; i < z->variable()->size(); i++, z_d++)
        *z_d = normal(engine);

      //    # Generator update.
      solver_gen->zero_grad();
      loss_gen->forward(/*clear_buffer=*/false, /*clear_no_need_grad=*/true);
      loss_gen->variable()->grad()->fill(1.0);
      loss_gen->backward(/*NdArrayPtr grad=*/nullptr, /*clear_buffer=*/true);
      solver_gen->weight_decay(weight_decay);
      solver_gen->update();

      //    # Discriminator update.
      solver_dis->zero_grad();
      loss_dis->forward(/*clear_buffer=*/false, /*clear_no_need_grad=*/true);
      loss_dis->variable()->grad()->fill(1.0);
      loss_dis->backward(/*NdArrayPtr grad=*/nullptr, /*clear_buffer=*/true);
      solver_dis->weight_decay(weight_decay);
      solver_dis->update();

      // Get and print the average loss
      float_t *loss_gen_d =
          loss_gen->variable()->cast_data_and_get_pointer<float_t>(cpu_ctx,
                                                                   false);
      mean_loss_gen += loss_gen_d[0];

      float_t *loss_dis_d =
          loss_dis->variable()->cast_data_and_get_pointer<float_t>(cpu_ctx,
                                                                   false);
      mean_loss_dis += loss_dis_d[0];

      if ((iter + 1) % n_val_iter == 0) {
        mean_loss_gen /= n_val_iter;
        mean_loss_dis /= n_val_iter;
        fprintf(fp, "iter: %d, loss_gen: %f, loss_dis: %f\n", iter + 0,
                mean_loss_gen, mean_loss_dis);
        fprintf(stdout, "iter: %d, loss_gen: %f, loss_dis: %f\n", iter + 0,
                mean_loss_gen, mean_loss_dis);
        mean_loss_gen = 0.;
        mean_loss_dis = 0.;

        nbla::utils::save_parameters(params, "saved_dcgan_param.protobuf");
      }

      // Get generated images"
      if ((iter + 1) % max_iter == 0)
        print_array(fake);
    }
  } catch (...) {
    cout << "Exception in dcgan_training_with_static_graph.\n";
    fclose(fp);
    return false;
  }
  fclose(fp);
  return true;
}

bool dcgan_training_with_dynamic_graph(nbla::Context ctx) {

  // Create a context for cpu
  nbla::Context cpu_ctx{{"cpu:float"}, "CpuCachedArray", "0"};

  // Setup context
  SingletonManager::get<GlobalContext>()->set_current_context(ctx);

  // Setup auto_forward
  SingletonManager::get<AutoForward>()->set_auto_forward(true);

  // Setup parameter
  ParameterDirectory params;

  nbla::utils::load_parameters(params, "dcgan_param_d.protobuf");

  //  # Create Solver.
  float learning_rate = 2.0e-4;
  auto solver_gen = create_AdamSolver(ctx, learning_rate, 0.5, 0.999, 1.0e-8);
  auto solver_dis = create_AdamSolver(ctx, learning_rate, 0.5, 0.999, 1.0e-8);

  // Create mnist data iterator
  MnistDataIterator train_data_provider("train");

  //# Training loop.
  FILE *fp;
  fp = fopen("log.txt", "wt");
  if (fp == NULL) {
    fprintf(stderr, "Error in opening log file.");
    return false;
  }

  try {
    int batch_size = 64;
    int max_h = 256;
    int max_iter = 20000;
    int n_val_iter = 10;
    float weight_decay = 1.0e-4;
    float mean_loss_gen = 0.;
    float mean_loss_dis = 0.;
    for (int iter = 0; iter < max_iter; iter++) {

      // Get batch and copy to input variables
      auto x = make_shared<CgVariable>(Shape_t({batch_size, 1, 28, 28}), false);
      auto t = make_shared<CgVariable>(Shape_t({batch_size, 1}), false);
      train_data_provider.provide_data(cpu_ctx, batch_size, x, t);

      auto z = make_shared<CgVariable>(Shape_t({batch_size, 100, 1, 1}), false);
      float_t *z_d =
          z->variable()->cast_data_and_get_pointer<float_t>(cpu_ctx, true);
      for (int i = 0; i < z->variable()->size(); i++, z_d++)
        *z_d = normal(engine);

      //    # Generator update.
      auto fake = generator(z, max_h, false, params["gen"]);
      fake->set_persistent(true);

      auto pred_fake = discriminator(fake, max_h, false, params["dis"]);
      auto loss_gen = f::mean(
          f::sigmoid_cross_entropy(pred_fake, f::constant(1, {batch_size, 1})),
          {0, 1}, false);

      solver_gen->set_parameters(params["gen"].get_parameters(),
                                 /*reset=*/false, /*retain_state=*/true);
      solver_gen->zero_grad();
      loss_gen->variable()->grad()->fill(1.0);
      loss_gen->backward(/*NdArrayPtr grad=*/nullptr, /*clear_buffer=*/true);
      solver_gen->weight_decay(weight_decay);
      solver_gen->update();

      //    # Discriminator update.
      auto fake_dis = make_shared<CgVariable>(fake->variable(), true);
      fake_dis->set_need_grad(true);

      auto pred_fake_dis = discriminator(fake, max_h, false, params["dis"]);
      pred_fake_dis->set_persistent(true);
      auto loss_dis =
          f::mean(f::sigmoid_cross_entropy(pred_fake_dis,
                                           f::constant(0, {batch_size, 1})),
                  {0, 1}, false);
      auto pred_real = discriminator(x, max_h, false, params["dis"]);
      loss_dis =
          loss_dis + f::mean(f::sigmoid_cross_entropy(
                                 pred_real, f::constant(1, {batch_size, 1})),
                             {0, 1}, false);

      solver_dis->set_parameters(params["dis"].get_parameters(),
                                 /*reset=*/false, /*retain_state=*/true);
      solver_dis->zero_grad();
      loss_dis->variable()->grad()->fill(1.0);
      loss_dis->backward(/*NdArrayPtr grad=*/nullptr, /*clear_buffer=*/true);
      solver_dis->weight_decay(weight_decay);
      solver_dis->update();

      // Get and print the average loss
      float_t *loss_gen_d =
          loss_gen->variable()->cast_data_and_get_pointer<float_t>(cpu_ctx,
                                                                   false);
      mean_loss_gen += loss_gen_d[0];

      float_t *loss_dis_d =
          loss_dis->variable()->cast_data_and_get_pointer<float_t>(cpu_ctx,
                                                                   false);
      mean_loss_dis += loss_dis_d[0];

      if ((iter + 1) % n_val_iter == 0) {
        mean_loss_gen /= n_val_iter;
        mean_loss_dis /= n_val_iter;
        fprintf(fp, "iter: %d, loss_gen: %f, loss_dis: %f\n", iter + 0,
                mean_loss_gen, mean_loss_dis);
        fprintf(stdout, "iter: %d, loss_gen: %f, loss_dis: %f\n", iter + 0,
                mean_loss_gen, mean_loss_dis);
        mean_loss_gen = 0.;
        mean_loss_dis = 0.;

        nbla::utils::save_parameters(params, "saved_dcgan_param_d.protobuf");
      }

      // Get generated images"
      if ((iter + 1) % max_iter == 0)
        print_array(fake);
    }
  } catch (...) {
    cout << "Exception in dcgan_training_with_dynamic_graph.\n";
    fclose(fp);
    return false;
  }
  fclose(fp);
  return true;
}
