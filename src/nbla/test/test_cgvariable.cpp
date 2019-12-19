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

// test_cgvariable.cpp

#include "macros.hpp"

#include "gtest/gtest.h"
#include <functional>
#include <nbla/computation_graph/computation_graph.hpp>
#include <nbla/computation_graph/variable.hpp>
#include <nbla/function.hpp>
#include <nbla/function/callback.hpp>
#include <vector>

namespace nbla {

using std::vector;
using std::make_shared;

auto ignore1 = [](void *obj, const Variables &, const Variables &) {};
auto ignore2 = [](void *obj) {};

struct MockCommunicatorBackwardCallback : public CommunicatorBackwardCallback {
  std::function<void(const CgFunctionPtr &)> on_finish_function_backward_;
  std::function<void()> on_finish_backward_;
  virtual void on_finish_function_backward(const CgFunctionPtr &ptr) {
    this->on_finish_function_backward_(ptr);
  }
  virtual void on_finish_backward() { this->on_finish_backward_(); }
};

class CgVariableTest : public ::testing::Test {
protected:
  virtual void SetUp() {
    init_cpu();
    this->ctx_.array_class = "CpuArray";
  }
  Context ctx_;
};

TEST_F(CgVariableTest, OrderOfBackwardForGraphWithoutBranches) {
  vector<std::string> order;
  auto generate_backward = [&order](std::string name) {
    return [name, &order](void *obj, const Variables &inputs,
                          const Variables &outputs,
                          const vector<bool> &propagate_down,
                          const vector<bool> &accum) { order.push_back(name); };
  };

  /* Generate network */
  auto a = make_shared<Callback>(this->ctx_, nullptr, 1, ignore1, ignore1,
                                 generate_backward("A"), ignore2);
  auto b = make_shared<Callback>(this->ctx_, nullptr, 1, ignore1, ignore1,
                                 generate_backward("B"), ignore2);
  auto c = make_shared<Callback>(this->ctx_, nullptr, 1, ignore1, ignore1,
                                 generate_backward("C"), ignore2);
  auto input = make_shared<CgVariable>(Shape_t{1, 1, 1}, true);
  auto h1 = connect(make_shared<CgFunction>(a), {input}, 1);
  auto h2 = connect(make_shared<CgFunction>(b), h1, 1);
  auto h3 = connect(make_shared<CgFunction>(c), h2, 1);
  EXPECT_EQ(1, h3.size());

  /* backward and check the order */
  h3[0]->backward();
  EXPECT_EQ(vector<std::string>({"C", "B", "A"}), order);
}

TEST_F(CgVariableTest, OrderOfBackwardForGraphWithBranches) {
  vector<std::string> order;
  auto generate_backward = [&order](std::string name) {
    return [name, &order](void *obj, const Variables &inputs,
                          const Variables &outputs,
                          const vector<bool> &propagate_down,
                          const vector<bool> &accum) { order.push_back(name); };
  };

  /* Generate network */
  auto a = make_shared<Callback>(this->ctx_, nullptr, 1, ignore1, ignore1,
                                 generate_backward("A"), ignore2);
  auto b = make_shared<Callback>(this->ctx_, nullptr, 1, ignore1, ignore1,
                                 generate_backward("B"), ignore2);
  auto c = make_shared<Callback>(this->ctx_, nullptr, 1, ignore1, ignore1,
                                 generate_backward("C"), ignore2);
  auto d = make_shared<Callback>(this->ctx_, nullptr, 1, ignore1, ignore1,
                                 generate_backward("D"), ignore2);
  auto e = make_shared<Callback>(this->ctx_, nullptr, 1, ignore1, ignore1,
                                 generate_backward("E"), ignore2);
  auto input = make_shared<CgVariable>(Shape_t{1, 1, 1}, true);
  auto h1 = connect(make_shared<CgFunction>(a), {input}, 1);
  auto h2 = connect(make_shared<CgFunction>(b), h1, 1);
  auto h3 = connect(make_shared<CgFunction>(c), h1, 1);
  auto h4 = connect(make_shared<CgFunction>(d), h2, 1);
  auto h5 = connect(make_shared<CgFunction>(e), {h3[0], h4[0]}, 1);
  EXPECT_EQ(1, h5.size());

  /* backward and check the order */
  h5[0]->backward();
  EXPECT_EQ(vector<std::string>({"E", "D", "C", "B", "A"}), order);
}

TEST_F(CgVariableTest, CommunicatorBackwardCallback) {
  auto generate_backward = [](std::string name) {
    return [name](void *obj, const Variables &inputs, const Variables &outputs,
                  const vector<bool> &propagate_down,
                  const vector<bool> &accum) {};
  };

  /* Generate network */
  auto a = make_shared<CgFunction>(
      make_shared<Callback>(this->ctx_, nullptr, 1, ignore1, ignore1,
                            generate_backward("A"), ignore2));
  auto b = make_shared<CgFunction>(
      make_shared<Callback>(this->ctx_, nullptr, 1, ignore1, ignore1,
                            generate_backward("B"), ignore2));
  auto c = make_shared<CgFunction>(
      make_shared<Callback>(this->ctx_, nullptr, 1, ignore1, ignore1,
                            generate_backward("C"), ignore2));
  auto in = make_shared<CgVariable>(Shape_t{1, 1, 1}, false);
  auto p1 = make_shared<CgVariable>(Shape_t{1, 1, 1}, true);
  auto p2 = make_shared<CgVariable>(Shape_t{1, 1, 1}, true);
  auto h1 = connect(a, {in, p1}, 1);
  h1.push_back(p2);
  auto h2 = connect(b, h1, 1);
  auto h3 = connect(c, h2, 1);
  EXPECT_EQ(1, h3.size());

  /* backward and check the order */
  vector<CgFunctionPtr> funcs;
  auto expected = vector<CgFunctionPtr>{c, b, a};
  bool is_called = false;
  auto p = make_shared<MockCommunicatorBackwardCallback>();
  p->on_finish_function_backward_ = [&funcs](const CgFunctionPtr &func) {
    funcs.push_back(func);
  };
  p->on_finish_backward_ = [&is_called]() { is_called = true; };

  h3[0]->backward(nullptr, false, {p});
  EXPECT_EQ(expected, funcs);
  EXPECT_TRUE(is_called);
}

class CgVariableDeepCopyTest : public CgVariableTest {
protected:
  CgVariablePtr x_;
  CgVariablePtr x_copy_;
  int size_;

  void SetUp() override {
    CgVariableTest::SetUp();

    x_ = make_shared<CgVariable>(Shape_t{5, 5}, false);

    // make all data values 1
    auto *d = x_->variable()->cast_data_and_get_pointer<float>(ctx_, true);
    REP(i, x_->variable()->size()) { d[i] = 1.0f; }

    // make all grad values 2
    auto *g = x_->variable()->cast_grad_and_get_pointer<float>(ctx_, true);
    REP(i, x_->variable()->size()) { g[i] = 2.0f; }

    x_copy_ = x_->create_deep_copy(ctx_, true);

    size_ = static_cast<int>(x_->variable()->size());
  }
};

TEST_F(CgVariableDeepCopyTest, CheckCopyValues) {
  ASSERT_EQ(x_->variable()->shape(), x_copy_->variable()->shape());

  // check all data values 1
  auto *d_copy =
      x_copy_->variable()->cast_data_and_get_pointer<float>(ctx_, false);
  REP(i, size_) { EXPECT_FLOAT_EQ(1.0f, d_copy[i]); }

  // check all grad values 2
  auto *g_copy =
      x_copy_->variable()->cast_grad_and_get_pointer<float>(ctx_, false);
  REP(i, size_) { EXPECT_FLOAT_EQ(2.0f, g_copy[i]); }
}

TEST_F(CgVariableDeepCopyTest, ChangeOriginalValues) {
  // change original variable`s data values
  auto *d = x_->variable()->cast_data_and_get_pointer<float>(ctx_, true);
  REP(i, size_) { d[i] = -1.0f; }

  auto *d_copy =
      x_copy_->variable()->cast_data_and_get_pointer<float>(ctx_, false);
  REP(i, size_) { EXPECT_FLOAT_EQ(1.0f, d_copy[i]); }

  // change original variable`s data values
  auto *g = x_->variable()->cast_grad_and_get_pointer<float>(ctx_, true);
  REP(i, size_) { g[i] = -2.0f; }

  auto *g_copy =
      x_copy_->variable()->cast_grad_and_get_pointer<float>(ctx_, false);
  REP(i, size_) { EXPECT_FLOAT_EQ(2.0f, g_copy[i]); }
}

TEST_F(CgVariableDeepCopyTest, ChangeCopyValues) {
  // change original variable`s data values
  auto *d_copy =
      x_copy_->variable()->cast_data_and_get_pointer<float>(ctx_, true);
  REP(i, size_) { d_copy[i] = -1.0f; }

  auto *d = x_->variable()->cast_data_and_get_pointer<float>(ctx_, false);
  REP(i, size_) { EXPECT_FLOAT_EQ(1.0f, d[i]); }

  // change original variable`s data values
  auto *g_copy =
      x_copy_->variable()->cast_grad_and_get_pointer<float>(ctx_, true);
  REP(i, size_) { g_copy[i] = -2.0f; }

  auto *g = x_->variable()->cast_grad_and_get_pointer<float>(ctx_, false);
  REP(i, size_) { EXPECT_FLOAT_EQ(2.0f, g[i]); }
}
}
