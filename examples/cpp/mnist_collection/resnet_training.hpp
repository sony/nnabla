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
// Example of ResNet Model Design
/******************************************/
CgVariablePtr res_unit(CgVariablePtr x, bool test,
                       ParameterDirectory parameters) {
  // from python examples
  // def res_unit(x, scope):
  //      C = x.shape[1]
  //      with nn.parameter_scope(scope):
  //          with nn.parameter_scope('conv1'):
  //              h = F.elu(bn(PF.convolution(x, C / 2, (1, 1),
  //              with_bias=False)))
  //          with nn.parameter_scope('conv2'):
  //              h = F.elu(
  //                  bn(PF.convolution(h, C / 2, (3, 3), pad=(1, 1),
  //                  with_bias=False)))
  //          with nn.parameter_scope('conv3'):
  //              h = bn(PF.convolution(h, C, (1, 1), with_bias=False))
  //      return F.elu(F.add2(h, x, inplace=True))
  Shape_t shape_x = x->variable()->shape();
  const int n_channel = shape_x[1];
  pf::ConvolutionOpts opts1 = pf::ConvolutionOpts().with_bias(false);
  pf::ConvolutionOpts opts2 =
      pf::ConvolutionOpts().with_bias(false).pad({1, 1});
  pf::ConvolutionOpts opts3 = pf::ConvolutionOpts().with_bias(false);
  auto c1 =
      pf::convolution(x, 1, n_channel / 2, {1, 1}, parameters["conv1"], opts1);
  auto e1 =
      f::elu(pf::batch_normalization(c1, !test, parameters["conv1"]), 1.0);
  auto c2 =
      pf::convolution(e1, 1, n_channel / 2, {3, 3}, parameters["conv2"], opts2);
  auto e2 =
      f::elu(pf::batch_normalization(c2, !test, parameters["conv2"]), 1.0);
  auto c3 =
      pf::convolution(e2, 1, n_channel, {1, 1}, parameters["conv3"], opts3);
  auto e3 =
      f::elu(pf::batch_normalization(c3, !test, parameters["conv3"]), 1.0);
  return f::elu(f::add2(e3, x, true), 1.0);
}

CgVariablePtr augmentation(CgVariablePtr h, bool augmentation) {
  if (augmentation) {
    return f::image_augmentation(h, {1, 28, 28}, {0, 0}, 0.9, 1.1, 0.3, 1.3,
                                 0.1, false, false, 0.5, false, 1.5, 0.5, false,
                                 0.1, 0);
  } else {
    return h;
  }
}

CgVariablePtr model(CgVariablePtr x, bool test, ParameterDirectory parameters) {
  // from python exmaples
  // xa = /= 255.0;
  // xa = augmentation(xa, test, aug);
  //  # Conv1 --> 64 x 32 x 32
  //  with nn.parameter_scope("conv1"):
  //      c1 = F.elu(
  //          bn(PF.convolution(image, 64, (3, 3), pad=(3, 3),
  //          with_bias=False)))
  //  # Conv2 --> 64 x 16 x 16
  //  c2 = F.max_pooling(res_unit(c1, "conv2"), (2, 2))
  //  # Conv3 --> 64 x 8 x 8
  //  c3 = F.max_pooling(res_unit(c2, "conv3"), (2, 2))
  //  # Conv4 --> 64 x 8 x 8
  //  c4 = res_unit(c3, "conv4")
  //  # Conv5 --> 64 x 4 x 4
  //  c5 = F.max_pooling(res_unit(c4, "conv5"), (2, 2))
  //  # Conv5 --> 64 x 4 x 4
  //  c6 = res_unit(c5, "conv6")
  //  pl = F.average_pooling(c6, (4, 4))
  //  with nn.parameter_scope("classifier"):
  //      y = PF.affine(pl, 10)
  //  return y

  auto xa = x * (1.0 / 255.0);
  xa = augmentation(x, !test);

  // Conv1 --> 64 x 32 x 32
  pf::ConvolutionOpts opts = pf::ConvolutionOpts().with_bias(false).pad({3, 3});
  auto c1 = pf::convolution(x, 1, 64, {3, 3}, parameters["conv1"], opts);
  c1 = f::elu(pf::batch_normalization(c1, !test, parameters["conv1"]), 1.0);

  // Conv2 --> 64 x 16 x 16
  auto c2 =
      f::max_pooling(res_unit(c1, test, parameters["conv2"]), {2, 2}, {2, 2});

  // Conv3 --> 64 x 8 x 8
  auto c3 =
      f::max_pooling(res_unit(c2, test, parameters["conv3"]), {2, 2}, {2, 2});

  // Conv4 --> 64 x 8 x 8
  auto c4 = res_unit(c3, test, parameters["conv4"]);

  // Conv5 --> 64 x 4 x 4
  auto c5 =
      f::max_pooling(res_unit(c4, test, parameters["conv5"]), {2, 2}, {2, 2});

  // Conv6 --> 64 x 4 x 4
  auto c6 = res_unit(c5, test, parameters["conv6"]);

  // affine
  auto h = f::average_pooling(c6, {4, 4}, {4, 4});
  auto y = pf::affine(h, 1, 10, parameters["classifier"]);
  return y;
}

/******************************************/
// Example of ResNet Classifier Training
/******************************************/
bool resnet_training_with_static_graph(nbla::Context ctx) {

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
  utl::load_parameters(params, "resnet_param.protobuf");

  int batch_size = 128;
  auto x = make_shared<CgVariable>(Shape_t({batch_size, 1, 28, 28}), false);
  auto t = make_shared<CgVariable>(Shape_t({batch_size, 1}), false);
  auto h_train = model(x, false, params);
  auto loss = f::mean(f::softmax_cross_entropy(h_train, t, 1), {0, 1}, false);
  auto h_valid = model(x, true, params);
  auto err = f::mean(f::top_n_error(h_valid, t, 1, 1), {0, 1}, false);

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

  try {
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
      loss->backward(/*NdArrayPtr grad =*/nullptr,
                     /*bool clear_buffer = */ true);
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
              err->variable()->cast_data_and_get_pointer<float_t>(cpu_ctx,
                                                                  false);
          mean_v_err += v_err_d[0];
        }
        mean_t_loss /= n_val_iter;
        mean_v_err /= 10;

        fprintf(fp, "iter: %d, tloss: %f, verr: %f\n", iter + 1, mean_t_loss,
                mean_v_err);
        fprintf(stdout, "iter: %d, tloss: %f, verr: %f\n", iter + 1,
                mean_t_loss, mean_v_err);
        mean_t_loss = 0;

        // Save parameters as a snapshot.
        utl::save_parameters(params, "saved_resnet_param.protobuf");
      }
    }
  } catch (...) {
    cout << "Exception in resnet_training_with_static_graph.\n";
    fclose(fp);
    return false;
  }
  fclose(fp);
  return true;
}

bool resnet_training_with_dynamic_graph(nbla::Context ctx) {

  // Create a context for cpu
  nbla::Context cpu_ctx{{"cpu:float"}, "CpuCachedArray", "0"};

  // Create mnist data iterator
  MnistDataIterator train_data_provider("train");
  MnistDataIterator test_data_provider("test");

  // Setup context
  SingletonManager::get<GlobalContext>()->set_current_context(ctx);

  // Setup auto_forward
  SingletonManager::get<AutoForward>()->set_auto_forward(true);

  // Setup parameter space
  ParameterDirectory params;

  // Load pretrained parameter if it has.
  utl::load_parameters(params, "resnet_param_d.protobuf");

  // Setup solver and input learnable parameters
  auto adam = create_AdamSolver(ctx, 0.001, 0.9, 0.999, 1.0e-8);

  // Execute training
  FILE *fp;
  fp = fopen("log.txt", "wt");
  if (fp == NULL) {
    fprintf(stderr, "Error in opening log file.");
    return false;
  }

  try {
    int batch_size = 128;
    int max_iter = 10000;
    int n_val_iter = 10;
    float mean_t_loss = 0.;
    for (int iter = 0; iter < max_iter; iter++) {

      // Get batch and copy to input variables
      auto x = make_shared<CgVariable>(Shape_t({batch_size, 1, 28, 28}), false);
      auto t = make_shared<CgVariable>(Shape_t({batch_size, 1}), false);
      train_data_provider.provide_data(cpu_ctx, batch_size, x, t);
      auto h = model(x, false, params);
      auto loss = f::mean(f::softmax_cross_entropy(h, t, 1), {0, 1}, false);
      auto err = f::mean(f::top_n_error(h, t, 1, 1), {0, 1}, false);

      // Execute forward, backward and update
      adam->set_parameters(params.get_parameters(), false, true);
      adam->zero_grad();
      loss->variable()->grad()->fill(1.0);
      loss->backward(/*NdArrayPtr grad =*/nullptr,
                     /*bool clear_buffer = */ true);
      adam->update();

      // Get and print the average loss
      float_t *t_loss_d =
          loss->variable()->cast_data_and_get_pointer<float_t>(cpu_ctx, false);
      mean_t_loss += t_loss_d[0];

      if ((iter + 1) % n_val_iter == 0) {

        float mean_v_err = 0.0;
        for (int v_iter = 0; v_iter < 10; v_iter++) {
          test_data_provider.provide_data(cpu_ctx, batch_size, x, t);
          auto h = model(x, true, params);
          auto err = f::mean(f::top_n_error(h, t, 1, 1), {0, 1}, false);
          float_t *v_err_d =
              err->variable()->cast_data_and_get_pointer<float_t>(cpu_ctx,
                                                                  false);
          mean_v_err += v_err_d[0];
        }
        mean_t_loss /= n_val_iter;
        mean_v_err /= 10;

        fprintf(fp, "iter: %d, tloss: %f, verr: %f\n", iter + 1, mean_t_loss,
                mean_v_err);
        fprintf(stdout, "iter: %d, tloss: %f, verr: %f\n", iter + 1,
                mean_t_loss, mean_v_err);
        mean_t_loss = 0;

        // Save parameters as a snapshot.
        utl::save_parameters(params, "saved_resnet_param_d.protobuf");
      }
    }
  } catch (...) {
    cout << "Exception in resnet_training_with_dynamic_graph.\n";
    fclose(fp);
    return false;
  }
  fclose(fp);
  return true;
}
