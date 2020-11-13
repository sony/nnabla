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

/******************************************/
// Example of Siamese Model Design
/******************************************/
CgVariablePtr mnist_lenet_feature(CgVariablePtr image,
                                  ParameterDirectory params, bool test) {
  // from python example
  // def mnist_lenet_feature(image, test=False):
  //  """
  //  Construct LeNet for MNIST.
  //  """
  auto c1 = f::elu(pf::convolution(image, 1, 20, {5, 5}, params["conv1"]), 1.0);
  c1 = f::average_pooling(c1, {2, 2}, {2, 2});
  auto c2 = f::elu(pf::convolution(c1, 1, 50, {5, 5}, params["conv2"]), 1.0);
  c2 = f::average_pooling(c2, {2, 2}, {2, 2});
  auto c3 = f::elu(pf::affine(c2, 1, 500, params["fc3"]), 1.0);
  auto c4 = f::elu(pf::affine(c3, 1, 10, params["fc4"]), 1.0);
  auto c5 = pf::affine(c4, 1, 2, params["fc_embed"]);
  return c5;
}

CgVariablePtr mnist_lenet_siamese(CgVariablePtr x0, CgVariablePtr x1,
                                  ParameterDirectory params,
                                  bool test = false) {
  // from python example
  // def mnist_lenet_siamese(x0, x1, test=False):
  //  """"""
  auto h0 = mnist_lenet_feature(x0, params, test);
  auto h1 = mnist_lenet_feature(x1, params, test);
  auto h = f::squared_error(h0, h1);
  auto p = f::sum(h, {1}, true);
  return p;
}

CgVariablePtr contrastive_loss(CgVariablePtr sd, CgVariablePtr l,
                               float margin = 1.0, float eps = 1e-4) {
  // from python example
  // def contrastive_loss(sd, l, margin=1.0, eps=1e-4):
  //  """
  //  This implements contrastive loss function given squared difference `sd`
  //  and labels `l` in {0, 1}.
  //
  //  f(sd, l) = l * sd + (1 - l) * max(0, margin - sqrt(sd))^2
  //
  //  NNabla implements various basic arithmetic operations. That helps write
  //  custom operations
  //  with composition like this. This is handy, but still implementing NNabla
  //  Function in C++
  //  gives you better performance advantage.
  //  """
  auto sim_cost = l * sd;
  auto dissim_cost =
      (1 - l) *
      (f::pow_scalar(
          f::maximum_scalar(margin - f::pow_scalar(sd + eps, 0.5, false), 0),
          2.0, false));
  return sim_cost + dissim_cost;
}

class MnistSiameseDataIterator {
private:
  int batch_size_;
  bool train_;
  CgVariablePtr t1_;
  CgVariablePtr t2_;
  shared_ptr<MnistDataIterator> m1_;
  shared_ptr<MnistDataIterator> m2_;
  MnistSiameseDataIterator() {}

public:
  MnistSiameseDataIterator(int batch_size, bool train)
      : batch_size_(batch_size), train_(train) {
    t1_ = make_shared<CgVariable>(Shape_t({batch_size_, 1}), false);
    t2_ = make_shared<CgVariable>(Shape_t({batch_size_, 1}), false);
    if (train) {
      m1_ = make_shared<MnistDataIterator>("train");
      m2_ = make_shared<MnistDataIterator>("train");
    } else {
      m1_ = make_shared<MnistDataIterator>("test");
      m2_ = make_shared<MnistDataIterator>("test");
    }
  }
  ~MnistSiameseDataIterator() {}
  void provide_data(CgVariablePtr x1, CgVariablePtr x2, CgVariablePtr t) {
    nbla::Context cpu_ctx{{"cpu:float"}, "CpuCachedArray", "0"};
    m1_->provide_data(cpu_ctx, batch_size_, x1, t1_);
    m2_->provide_data(cpu_ctx, batch_size_, x2, t2_);
    uint8_t *t1_d =
        t1_->variable()->cast_data_and_get_pointer<uint8_t>(cpu_ctx, false);
    uint8_t *t2_d =
        t2_->variable()->cast_data_and_get_pointer<uint8_t>(cpu_ctx, false);
    uint8_t *t_d =
        t->variable()->cast_data_and_get_pointer<uint8_t>(cpu_ctx, true);
    for (int i = 0; i < t->variable()->size(); i++)
      t_d[i] = (t1_d[i] == t2_d[i]);
  }
};

/******************************************/
// Example of Siamese Training
/******************************************/
bool siamese_training_with_static_graph(nbla::Context ctx) {

  // Create a context for cpu
  nbla::Context cpu_ctx{{"cpu:float"}, "CpuCachedArray", "0"};

  //  # Create CNN network for both training and testing.
  float margin = 1.0; //# Margin for contrastive loss.

  // Setup context
  SingletonManager::get<GlobalContext>()->set_current_context(ctx);

  //  # Create input variables.
  ParameterDirectory params;

  // Load pretrained parameter if it has.
  utl::load_parameters(params, "siamese_param.protobuf");

  int batch_size = 128;
  auto timage0 =
      make_shared<CgVariable>(Shape_t({batch_size, 1, 28, 28}), false);
  auto timage1 =
      make_shared<CgVariable>(Shape_t({batch_size, 1, 28, 28}), false);
  auto tlabel = make_shared<CgVariable>(Shape_t({batch_size, 1}), false);

  //  # Create prediction graph.
  auto tpred = mnist_lenet_siamese(timage0, timage1, params, false);

  // # Create loss function.
  auto tloss = f::mean(contrastive_loss(tpred, tlabel, margin), {0}, false);

  //  # TEST
  //  # Create input variables.
  auto vimage0 =
      make_shared<CgVariable>(Shape_t({batch_size, 1, 28, 28}), false);
  auto vimage1 =
      make_shared<CgVariable>(Shape_t({batch_size, 1, 28, 28}), false);
  auto vlabel = make_shared<CgVariable>(Shape_t({batch_size, 1}), false);

  //  # Create prediction graph.
  auto vpred = mnist_lenet_siamese(vimage0, vimage1, params, true);
  auto vloss = f::mean(contrastive_loss(vpred, vlabel, margin), {0}, false);

  //  # Create Solver.
  float learning_rate = 1.0e-3;
  auto solver = create_AdamSolver(ctx, learning_rate, 0.9, 0.999, 1.0e-8);
  solver->set_parameters(params.get_parameters());

  // # Initialize DataIterator for MNIST.
  MnistSiameseDataIterator tdata_iterator(batch_size, true);
  MnistSiameseDataIterator vdata_iterator(batch_size, false);

  //# Training loop.
  FILE *fp;
  fp = fopen("log.txt", "wt");
  if (fp == NULL) {
    fprintf(stderr, "Error in opening log file.");
    return false;
  }

  try {
    //  # Training loop.
    int max_iter = 5000;
    int n_val_interval = 10;
    int n_val_iter = 10;
    float weight_decay = 0.;
    float mean_tloss = 0.;
    for (int iter = 0; iter < max_iter; iter++) {

      // Get batch and copy to input variables
      tdata_iterator.provide_data(timage0, timage1, tlabel);

      // # Training forward, backward and update.
      solver->zero_grad();
      tloss->forward(/*clear_buffer=*/false, /*clear_no_need_grad=*/true);
      tloss->variable()->grad()->fill(1.0);
      tloss->backward(/*NdArrayPtr grad=*/nullptr, /*clear_buffer=*/true);
      solver->weight_decay(weight_decay);
      solver->update();

      // Get and print the average loss
      float_t *tloss_d =
          tloss->variable()->cast_data_and_get_pointer<float_t>(cpu_ctx, false);
      mean_tloss += tloss_d[0];

      if ((iter + 1) % n_val_interval == 0) {
        float mean_vloss = 0.0;
        for (int i = 0; i < n_val_iter; i++) {
          vdata_iterator.provide_data(vimage0, vimage1, vlabel);
          vloss->forward(/*clear_buffer=*/true, /*clear_no_need_grad=*/true);
          float_t *vloss_d =
              vloss->variable()->cast_data_and_get_pointer<float_t>(cpu_ctx,
                                                                    false);
          mean_vloss += vloss_d[0];
        }
        mean_tloss /= n_val_interval;
        mean_vloss /= n_val_iter;
        fprintf(fp, "iter: %d, tloss: %f, vloss: %f\n", iter, mean_tloss,
                mean_vloss);
        fprintf(stdout, "iter: %d, tloss: %f, vloss: %f\n", iter, mean_tloss,
                mean_vloss);
        mean_tloss = 0.;
        mean_vloss = 0.;

        // Save parameters as a snapshot.
        utl::save_parameters(params, "saved_siamese_param.protobuf");
      }
    }
  } catch (...) {
    cout << "Exception in siamese_training_with_static_graph.\n";
    fclose(fp);
    return false;
  }
  fclose(fp);
  return true;
}

bool siamese_training_with_dynamic_graph(nbla::Context ctx) {

  // Create a context for cpu
  nbla::Context cpu_ctx{{"cpu:float"}, "CpuCachedArray", "0"};

  //  # Create CNN network for both training and testing.
  float margin = 1.0; //# Margin for contrastive loss.

  // Setup context
  SingletonManager::get<GlobalContext>()->set_current_context(ctx);

  // Setup auto_forward
  SingletonManager::get<AutoForward>()->set_auto_forward(true);

  // Setup parameter
  ParameterDirectory params;

  // Load pretrained parameter if it has.
  utl::load_parameters(params, "siamese_param_d.protobuf");

  //  # Create Solver.
  float learning_rate = 1.0e-3;
  auto solver = create_AdamSolver(ctx, learning_rate, 0.9, 0.999, 1.0e-8);

  // # Initialize DataIterator for MNIST.
  int batch_size = 128;
  MnistSiameseDataIterator tdata_iterator(batch_size, true);
  MnistSiameseDataIterator vdata_iterator(batch_size, false);

  //# Training loop.
  FILE *fp;
  fp = fopen("log.txt", "wt");
  if (fp == NULL) {
    fprintf(stderr, "Error in opening log file.");
    return false;
  }

  try {
    //  # Training loop.
    int max_iter = 5000;
    int n_val_interval = 10;
    int n_val_iter = 10;
    float weight_decay = 0.;
    float mean_tloss = 0.;
    for (int iter = 0; iter < max_iter; iter++) {

      // Get batch and copy to input variables
      auto timage0 =
          make_shared<CgVariable>(Shape_t({batch_size, 1, 28, 28}), false);
      auto timage1 =
          make_shared<CgVariable>(Shape_t({batch_size, 1, 28, 28}), false);
      auto tlabel = make_shared<CgVariable>(Shape_t({batch_size, 1}), false);
      tdata_iterator.provide_data(timage0, timage1, tlabel);

      // # Training forward, backward and update.
      auto tpred = mnist_lenet_siamese(timage0, timage1, params, false);
      auto tloss = f::mean(contrastive_loss(tpred, tlabel, margin), {0}, false);

      solver->set_parameters(params.get_parameters(),
                             /*reset=*/false, /*retain_state=*/true);
      solver->zero_grad();
      tloss->variable()->grad()->fill(1.0);
      tloss->backward(/*NdArrayPtr grad=*/nullptr, /*clear_buffer=*/true);
      solver->weight_decay(weight_decay);
      solver->update();

      // Get and print the average loss
      float_t *tloss_d =
          tloss->variable()->cast_data_and_get_pointer<float_t>(cpu_ctx, false);
      mean_tloss += tloss_d[0];

      if ((iter + 1) % n_val_interval == 0) {
        float mean_vloss = 0.0;
        for (int i = 0; i < n_val_iter; i++) {
          auto vimage0 =
              make_shared<CgVariable>(Shape_t({batch_size, 1, 28, 28}), false);
          auto vimage1 =
              make_shared<CgVariable>(Shape_t({batch_size, 1, 28, 28}), false);
          auto vlabel =
              make_shared<CgVariable>(Shape_t({batch_size, 1}), false);

          vdata_iterator.provide_data(vimage0, vimage1, vlabel);

          auto vpred = mnist_lenet_siamese(vimage0, vimage1, params, true);
          auto vloss =
              f::mean(contrastive_loss(vpred, vlabel, margin), {0}, false);
          float_t *vloss_d =
              vloss->variable()->cast_data_and_get_pointer<float_t>(cpu_ctx,
                                                                    false);
          mean_vloss += vloss_d[0];
        }
        mean_tloss /= n_val_interval;
        mean_vloss /= n_val_iter;
        fprintf(fp, "iter: %d, tloss: %f, vloss: %f\n", iter, mean_tloss,
                mean_vloss);
        fprintf(stdout, "iter: %d, tloss: %f, vloss: %f\n", iter, mean_tloss,
                mean_vloss);
        mean_tloss = 0.;
        mean_vloss = 0.;

        // Save parameters as a snapshot.
        utl::save_parameters(params, "saved_siamese_param_d.protobuf");
      }
    }
  } catch (...) {
    cout << "Exception in siamese_training_with_dynamic_graph.\n";
    fclose(fp);
    return false;
  }
  fclose(fp);
  return true;
}
