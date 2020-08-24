// Copyright (c) 2019 Sony Corporation. All Rights Reserved.
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

// test_load_save_parameters.cpp

#include "gtest/gtest.h"
#include <algorithm>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include <nbla/computation_graph/computation_graph.hpp>
#include <nbla/parametric_functions.hpp>
#include <nbla/solver/adam.hpp>
#include <nbla_utils/nnp.hpp>
#include <nbla_utils/parameters.hpp>

namespace nbla {
namespace utils {

using namespace std;
using namespace nbla;
namespace f = nbla::functions;
namespace pf = nbla::parametric_functions;

const Context kCpuCtx{{"cpu:float"}, "CpuCachedArray", "0"};
const string filename = "params.protobuf";
const string filename_h5 = "parameters.h5";

CgVariablePtr model(CgVariablePtr x, ParameterDirectory parameters) {
  auto h = pf::convolution(x, 1, 16, {3, 3}, parameters["conv1"]);
  h = f::max_pooling(h, {2, 2}, {2, 2}, true, {0, 0}, false);
  h = f::relu(h, false);
  h = pf::convolution(h, 1, 16, {3, 3}, parameters["conv2"]);
  h = f::max_pooling(h, {2, 2}, {2, 2}, true, {0, 0}, false);
  h = f::relu(h, false);
  h = pf::affine(h, 1, 50, parameters["affine3"]);
  h = f::relu(h, false);
  h = pf::affine(h, 1, 10, parameters["affine4"]);
  return h;
}

void set_input(CgVariablePtr x) {
  float_t *x_d =
      x->variable()->cast_data_and_get_pointer<float_t>(kCpuCtx, true);
  for (int i = 0; i < 28 * 28; ++i) {
    x_d[i] = 0.1 * i; // set a arbitrarily value
  }
}

void set_t(CgVariablePtr t) {
  float_t *t_d =
      t->variable()->cast_data_and_get_pointer<float_t>(kCpuCtx, true);
  for (int i = 0; i < 10; ++i) {
    t_d[i] = i; // set value from 0 to 9
  }
}

void check_result(CgVariablePtr x, CgVariablePtr y) {
  float_t *x_d =
      x->variable()->cast_data_and_get_pointer<float_t>(kCpuCtx, true);
  float_t *y_d =
      y->variable()->cast_data_and_get_pointer<float_t>(kCpuCtx, true);

  for (int i = 0; i < 10; ++i) {
    EXPECT_FLOAT_EQ(x_d[i], y_d[i]);
  }
}

void dump_parameters(ParameterDirectory &params) {
  auto pd_param = params.get_parameters();
  for (auto it = pd_param.begin(); it != pd_param.end(); it++) {
    string name = it->first;
    VariablePtr variable = it->second;
    printf("==> %s\n", name.c_str());
    float *data = variable->template cast_data_and_get_pointer<float>(kCpuCtx);
    for (int i = 0; i < variable->size(); ++i) {
      printf("%4.2f ", data[i]);
    }
    printf("\n");
  }
}

CgVariablePtr simple_train(ParameterDirectory &params) {
  // we prepared parameters by simply running a inferring work.
  int batch_size = 1;
  auto x = make_shared<CgVariable>(Shape_t({batch_size, 1, 28, 28}), false);
  auto t = make_shared<CgVariable>(Shape_t({batch_size, 1}), false);
  auto h = model(x, params);
  auto loss = f::mean(f::softmax_cross_entropy(h, t, 1), {0, 1}, false);
  auto err = f::mean(f::top_n_error(h, t, 1, 1), {0, 1}, false);
  auto adam = create_AdamSolver(kCpuCtx, 0.001, 0.9, 0.999, 1.0e-8);
  adam->set_parameters(params.get_parameters());
  adam->zero_grad();
  set_input(x);
  set_t(t);
  loss->forward();
  loss->variable()->grad()->fill(1.0);
  loss->backward(nullptr, true);
  adam->update();
  set_input(x);
  h->forward();
  return h;
}

void expect_params_equal(vector<pair<string, VariablePtr>> &a,
                         vector<pair<string, VariablePtr>> &b) {
  for (auto a_it = a.begin(); a_it != a.end(); ++a_it) {
    bool exist = false;
    for (auto b_it = b.begin(); b_it != b.end(); ++b_it) {
      if (a_it->first == b_it->first) {
        int a_size = a_it->second->size();
        int b_size = b_it->second->size();
        EXPECT_EQ(a_size, b_size);
        float *a_data =
            a_it->second->template cast_data_and_get_pointer<float>(kCpuCtx);
        float *b_data =
            b_it->second->template cast_data_and_get_pointer<float>(kCpuCtx);
        EXPECT_TRUE(memcmp(a_data, b_data, a_size) == 0);
        exist = true;
      }
    }
    EXPECT_TRUE(exist);
  }
}

CgVariablePtr simple_infer(ParameterDirectory &params) {
  int batch_size = 1;
  auto x = make_shared<CgVariable>(Shape_t({batch_size, 1, 28, 28}), false);
  auto t = make_shared<CgVariable>(Shape_t({batch_size, 1}), false);
  auto h = model(x, params);
  set_input(x);
  h->forward();
  return h;
}

TEST(test_save_and_load_parameters, test_save_load_with_simple_train) {
  ParameterDirectory train_params;
  ParameterDirectory infer_params;

  CgVariablePtr x = simple_train(train_params);
  save_parameters(train_params, filename);
  load_parameters(infer_params, filename);
  CgVariablePtr y = simple_infer(infer_params);
  check_result(x, y);
}

TEST(test_save_and_load_parameters, test_save_load_without_train) {
  ParameterDirectory train_params;
  ParameterDirectory infer_params;

  CgVariablePtr x = simple_infer(train_params);
  save_parameters(train_params, filename);
  load_parameters(infer_params, filename);
  CgVariablePtr y = simple_infer(infer_params);
  check_result(x, y);
}

#ifdef NBLA_UTILS_WITH_HDF5
TEST(test_save_and_load_parameters, test_save_load_h5) {
  ParameterDirectory train_params;
  ParameterDirectory infer_params;

  CgVariablePtr x = simple_train(train_params);
  save_parameters(train_params, filename_h5);
  load_parameters(infer_params, filename_h5);
  CgVariablePtr y = simple_infer(infer_params);
  check_result(x, y);
}

TEST(test_save_and_load_parameters, test_nnp_save_h5) {
  ParameterDirectory train_params;
  ParameterDirectory infer_params;

  CgVariablePtr x = simple_train(train_params);
  save_parameters(train_params, filename);

  nbla::utils::nnp::Nnp nnp(kCpuCtx);
  nnp.add(filename);
  nnp.save_parameters(filename_h5);
  nbla::utils::nnp::Nnp nnp_ref(kCpuCtx);
  nnp_ref.add(filename_h5);
  auto params = nnp.get_parameters();
  auto params_ref = nnp_ref.get_parameters();
  expect_params_equal(params, params_ref);

  load_parameters(infer_params, filename_h5);
  CgVariablePtr y = simple_infer(infer_params);
  check_result(x, y);
}

TEST(test_save_and_load_parameters, test_nnp_save_pb) {
  ParameterDirectory train_params;
  ParameterDirectory infer_params;

  CgVariablePtr x = simple_train(train_params);
  save_parameters(train_params, filename_h5);

  nbla::utils::nnp::Nnp nnp(kCpuCtx);
  nnp.add(filename_h5);
  nnp.save_parameters(filename);
  nbla::utils::nnp::Nnp nnp_ref(kCpuCtx);
  nnp_ref.add(filename);
  auto params = nnp.get_parameters();
  auto params_ref = nnp_ref.get_parameters();
  expect_params_equal(params, params_ref);

  load_parameters(infer_params, filename);
  CgVariablePtr y = simple_infer(infer_params);
  check_result(x, y);
}
#endif
}
}
