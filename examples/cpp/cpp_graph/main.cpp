// -*- coding:utf-8 -*-

#include <nbla/computation_graph/computation_graph.hpp>
#include <nbla/computation_graph/function.hpp>
#include <nbla/computation_graph/variable.hpp>
#include <nbla/function.hpp>
#include <nbla/function/affine.hpp>
#include <nbla/function/batch_normalization.hpp>
#include <nbla/function/relu.hpp>
#include <nbla/function/softmax.hpp>
#include <nbla/variable.hpp>
#include <stdio.h>

using namespace nbla;
using std::make_shared;

int main() {

  Shape_t shape_x = {1, 28, 28};
  int n_x = 1;
  for (int i = 0; i < shape_x.size(); i++)
    n_x *= shape_x[i];
  int n_h = 1500;
  int n_y = 10;
  int batch_size = 100;
  shape_x.insert(shape_x.begin(), batch_size);

  // Layers
  Context ctx; // ("cpu", "CpuArray", "0", "default");
  auto affine1 = make_shared<CgFunction>(create_Affine(ctx, 1));
  auto bn1 = make_shared<CgFunction>(
      create_BatchNormalization(ctx, {1}, 0, 0.0, false, false, false));
  auto relu1 = make_shared<CgFunction>(create_ReLU(ctx, true));
  auto affine2 = make_shared<CgFunction>(create_Affine(ctx, 1));
  auto bn2 = make_shared<CgFunction>(
      create_BatchNormalization(ctx, {1}, 0, 0.0, false, false, false));
  auto relu2 = make_shared<CgFunction>(create_ReLU(ctx, true));
  auto affine3 = make_shared<CgFunction>(create_Affine(ctx, 1));

  // Variables
  auto x = make_shared<CgVariable>(shape_x);

  Shape_t shape_affine_w;
  Shape_t shape_affine_b;
  Shape_t shape_bn_b;
  Shape_t shape_bn_g;
  Shape_t shape_bn_m;
  Shape_t shape_bn_v;

  shape_affine_w = {n_x, n_h};
  shape_affine_b = {n_h};
  auto affine1_w = make_shared<CgVariable>(shape_affine_w, true);
  auto affine1_b = make_shared<CgVariable>(shape_affine_b, true);

  shape_bn_b = {1, n_h};
  shape_bn_g = {1, n_h};
  shape_bn_m = {1, n_h};
  shape_bn_v = {1, n_h};
  auto bn1_b = make_shared<CgVariable>(shape_bn_b, true);
  auto bn1_g = make_shared<CgVariable>(shape_bn_g, true);
  auto bn1_m = make_shared<CgVariable>(shape_bn_m, true);
  auto bn1_v = make_shared<CgVariable>(shape_bn_v, true);

  shape_affine_w = {n_h, n_h};
  shape_affine_b = {n_h};
  auto affine2_w = make_shared<CgVariable>(shape_affine_w, true);
  auto affine2_b = make_shared<CgVariable>(shape_affine_b, true);

  shape_bn_b = {1, n_h};
  shape_bn_g = {1, n_h};
  shape_bn_m = {1, n_h};
  shape_bn_v = {1, n_h};
  auto bn2_b = make_shared<CgVariable>(shape_bn_b, true);
  auto bn2_g = make_shared<CgVariable>(shape_bn_g, true);
  auto bn2_m = make_shared<CgVariable>(shape_bn_m, true);
  auto bn2_v = make_shared<CgVariable>(shape_bn_v, true);

  shape_affine_w = {n_h, n_y};
  shape_affine_b = {n_y};
  auto affine3_w = make_shared<CgVariable>(shape_affine_w, true);
  auto affine3_b = make_shared<CgVariable>(shape_affine_b, true);

  // Graph construction
  vector<CgVariablePtr> h;
  h = connect(affine1, {x, affine1_w, affine1_b}, 1);
  h = connect(bn1, {h[0], bn1_b, bn1_g, bn1_m, bn1_v}, 1);
  h = connect(relu1, {h[0]}, 1);
  h = connect(affine2, {h[0], affine2_w, affine2_b}, 1);
  h = connect(bn2, {h[0], bn2_b, bn2_g, bn2_m, bn2_v}, 1);
  h = connect(relu2, {h[0]}, 1);
  h = connect(affine3, {h[0], affine3_w, affine3_b}, 1);

  // Forward on graph
  auto var_x = x->variable();
  float *data_x = var_x->cast_data_and_get_pointer<float>(ctx);
  for (int i = 0; i < var_x.get()->size(); ++i) {
    data_x[i] = 1;
  }

  h[0]->forward(/*clear_buffer=*/true, /*clear_no_need_grad=*/false);

  auto var_h = h[0]->variable();
  float *data_h = var_h->cast_data_and_get_pointer<float>(ctx);
  for (int i = 0; i < var_h.get()->size(); ++i) {
    printf("%f\n", data_h[i]);
  }

  return 0;
}
