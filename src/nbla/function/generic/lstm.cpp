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

#include <nbla/function/lstm.hpp>

namespace nbla {

NBLA_REGISTER_FUNCTION_SOURCE(LSTM, int, float, bool, bool);

template <typename T>
void LSTM<T>::setup_impl(const Variables &inputs, const Variables &outputs) {

  Shape_t inshape = inputs[0]->shape();
  Shape_t hshape = inputs[1]->shape();
  Shape_t cshape = inputs[2]->shape();
  Shape_t outshape = outputs[0]->shape();

  // Check input dimensions
  NBLA_CHECK(inputs[0]->ndim() == 3, error_code::value,
             "Input x must be a 3 dimensional array with a shape of (steps, "
             "batch_size, input_size).");

  seq_len_ = inshape[0];
  batch_size_ = inshape[1];
  input_dim_ = inshape[2];
  // Assuming this function takes h as (numLayer, numD, B, M)
  hidden_size_ = inputs[1]->shape()[3];
  num_directions_ = this->bidirectional_ ? 2 : 1;

  // Check shape of h & c
  const char *error_msg_h = "Input h must be a 4 dimensional array with a "
                            "shape of (num_layers, num_directions, batch_size, "
                            "hidden_size).";
  NBLA_CHECK(inputs[1]->ndim() == 4, error_code::value, error_msg_h);
  NBLA_CHECK(hshape[0] == this->num_layers_, error_code::value, error_msg_h);
  NBLA_CHECK(hshape[1] == num_directions_, error_code::value, error_msg_h);
  NBLA_CHECK(hshape[2] == batch_size_, error_code::value, error_msg_h);
  NBLA_CHECK(hshape == cshape, error_code::value,
             "Input c must has the same shape as input h.");

  // Check weight shape at 0th layer
  Shape_t w0_shape = inputs[3]->shape();
  const char *error_msg_w0 = "Input w0 must be a 4 dimensional array with a "
                             "shape of (num_directions, 4, hidden_size, "
                             "input_size + hidden_size).";
  NBLA_CHECK(inputs[2]->ndim() == 4, error_code::value, error_msg_w0);
  NBLA_CHECK(w0_shape[0] == num_directions_, error_code::value, error_msg_w0);
  NBLA_CHECK(w0_shape[1] == 4, error_code::value, error_msg_w0);
  NBLA_CHECK(w0_shape[2] == hidden_size_, error_code::value, error_msg_w0);
  NBLA_CHECK(w0_shape[3] == hidden_size_ + input_dim_, error_code::value,
             error_msg_w0);

  weight_exists_ = true;
  bias_exists_ = true;
  if (inputs.size() == 4) {
    weight_exists_ = false;
    bias_exists_ = false;
  } else if (inputs.size() == 5) {
    Shape_t opt_shape = inputs[4]->shape();
    if (this->num_layers_ > 1 && opt_shape.size() == 5) {
      bias_exists_ = false;
    } else if (this->num_layers_ > 1 && opt_shape.size() != 5) {
      NBLA_ERROR(error_code::value,
                 "Weight argument must be passed when num_layers > 1");
    } else if (this->num_layers_ == 1 && opt_shape.size() != 4) {
      NBLA_ERROR(error_code::value,
                 "Weight argument cannot be passed when num_layers == 1");
    } else if (this->num_layers_ == 1 && opt_shape.size() == 4) {
      weight_exists_ = false;
    }
  } else if ((inputs.size() > 5) && (this->num_layers_ == 1)) {
    NBLA_ERROR(error_code::value,
               "Weight argument cannot be passed when num_layers == 1");
  }

  // Check weight shape
  if (weight_exists_) {
    Shape_t w_shape = inputs[4]->shape();
    const char *error_msg_w = "Input w must be a 5 dimensional array with a "
                              "shape of (num_layers - 1, num_directions, 4, "
                              "hidden_size, num_directions * hidden_size + "
                              "hidden_size).";
    NBLA_CHECK(inputs[4]->ndim() == 5, error_code::value, error_msg_w);
    NBLA_CHECK(w_shape[0] == this->num_layers_ - 1, error_code::value,
               error_msg_w);
    NBLA_CHECK(w_shape[1] == num_directions_, error_code::value, error_msg_w);
    NBLA_CHECK(w_shape[2] == 4, error_code::value, error_msg_w);
    NBLA_CHECK(w_shape[3] == hidden_size_, error_code::value, error_msg_w);
    NBLA_CHECK(w_shape[4] == num_directions_ * hidden_size_ + hidden_size_,
               error_code::value, error_msg_w);
  }

  // Check bias shape
  if (bias_exists_) {
    const int b_index = weight_exists_ ? 5 : 4;
    Shape_t b_shape = inputs[b_index]->shape();
    const char *error_msg_b = "Input b must be a 4 dimensional array with a "
                              "shape of (num_layers, 4, num_directions, "
                              "hidden_size).";
    NBLA_CHECK(inputs[b_index]->ndim() == 4, error_code::value, error_msg_b);
    NBLA_CHECK(b_shape[0] == this->num_layers_, error_code::value, error_msg_b);
    NBLA_CHECK(b_shape[1] == num_directions_, error_code::value, error_msg_b);
    NBLA_CHECK(b_shape[2] == 4, error_code::value, error_msg_b);
    NBLA_CHECK(b_shape[3] == hidden_size_, error_code::value, error_msg_b);
  }

  // Set output shapes
  outputs[0]->reshape({seq_len_, batch_size_, num_directions_ * hidden_size_},
                      true);
  outputs[1]->reshape(inputs[1]->shape(), true);
  outputs[2]->reshape(inputs[2]->shape(), true);
}

template <typename T>
vector<vector<CgVariablePtr>>
LSTM<T>::lstm_cell(shared_ptr<CgVariable> x, shared_ptr<CgVariable> h,
                   shared_ptr<CgVariable> c, shared_ptr<CgVariable> w,
                   shared_ptr<CgVariable> b) {
  // Graph construction
  vector<CgVariablePtr> b_vec;
  vector<int> t_axes{1, 0};
  auto concatenate = make_shared<CgFunction>(create_Concatenate(this->ctx_, 1));
  auto xh = connect(concatenate, {x, h}, 1);
  auto split = make_shared<CgFunction>(create_Split(this->ctx_, 0));
  auto w_vec = connect(split, {w}, 4);

  if (bias_exists_ == true) {
    auto split = make_shared<CgFunction>(create_Split(this->ctx_, 0));
    b_vec = connect(split, {b}, 4);
  }

  auto transpose1 =
      make_shared<CgFunction>(create_Transpose(this->ctx_, t_axes));
  auto affine1 = make_shared<CgFunction>(create_Affine(this->ctx_, 1));
  vector<CgVariablePtr> i_t;
  if (bias_exists_ == true) {
    i_t = connect(affine1,
                  {xh[0], connect(transpose1, {w_vec[0]}, 1)[0], b_vec[0]}, 1);
  } else {
    i_t = connect(affine1,
                  {
                      xh[0], connect(transpose1, {w_vec[0]}, 1)[0],
                  },
                  1);
  }

  auto transpose2 =
      make_shared<CgFunction>(create_Transpose(this->ctx_, t_axes));
  auto affine2 = make_shared<CgFunction>(create_Affine(this->ctx_, 1));
  vector<CgVariablePtr> f_t;
  if (bias_exists_ == true) {
    f_t = connect(affine2,
                  {xh[0], connect(transpose2, {w_vec[1]}, 1)[0], b_vec[1]}, 1);
  } else {
    f_t = connect(affine2,
                  {
                      xh[0], connect(transpose2, {w_vec[1]}, 1)[0],
                  },
                  1);
  }

  auto transpose3 =
      make_shared<CgFunction>(create_Transpose(this->ctx_, t_axes));
  auto affine3 = make_shared<CgFunction>(create_Affine(this->ctx_, 1));
  vector<CgVariablePtr> g_t;
  if (bias_exists_ == true) {
    g_t = connect(affine3,
                  {xh[0], connect(transpose3, {w_vec[2]}, 1)[0], b_vec[2]}, 1);
  } else {
    g_t = connect(affine3,
                  {
                      xh[0], connect(transpose3, {w_vec[2]}, 1)[0],
                  },
                  1);
  }

  auto transpose4 =
      make_shared<CgFunction>(create_Transpose(this->ctx_, t_axes));
  auto affine4 = make_shared<CgFunction>(create_Affine(this->ctx_, 1));
  vector<CgVariablePtr> o_t;
  if (bias_exists_ == true) {
    o_t = connect(affine4,
                  {xh[0], connect(transpose4, {w_vec[3]}, 1)[0], b_vec[3]}, 1);
  } else {
    o_t = connect(affine4,
                  {
                      xh[0], connect(transpose4, {w_vec[3]}, 1)[0],
                  },
                  1);
  }

  auto sigmoid1 = make_shared<CgFunction>(create_Sigmoid(this->ctx_));
  auto mul1 = make_shared<CgFunction>(create_Mul2(this->ctx_, false));
  auto tmp1 =
      connect(mul1, {connect(sigmoid1, f_t, 1)[0], c}, 1); // F.sigmoid(f_t) * c

  auto sigmoid2 = make_shared<CgFunction>(create_Sigmoid(this->ctx_));
  auto mul2 = make_shared<CgFunction>(create_Mul2(this->ctx_, false));
  auto tanh2 = make_shared<CgFunction>(create_Tanh(this->ctx_));
  auto add2 = make_shared<CgFunction>(create_Add2(this->ctx_, true));
  auto tmp2 =
      connect(mul2, {connect(sigmoid2, i_t, 1)[0], connect(tanh2, g_t, 1)[0]},
              1); // F.sigmoid(i_t) * F.tanh(g_t)

  auto c_t = connect(add2, {tmp1[0], tmp2[0]}, 1);

  auto sigmoid3 = make_shared<CgFunction>(create_Sigmoid(this->ctx_));
  auto mul3 = make_shared<CgFunction>(create_Mul2(this->ctx_, false));
  auto tanh3 = make_shared<CgFunction>(create_Tanh(this->ctx_));
  auto h_t =
      connect(mul3, {connect(sigmoid3, o_t, 1)[0], connect(tanh3, c_t, 1)[0]},
              1); // F.sigmoid(o_t) * F.tanh(c_t)

  return {h_t, c_t};
}

template <typename T>
vector<vector<CgVariablePtr>> LSTM<T>::create_fixed_length_lstm_graph(
    shared_ptr<CgVariable> in_x, shared_ptr<CgVariable> in_h,
    shared_ptr<CgVariable> in_c, shared_ptr<CgVariable> in_w0,
    shared_ptr<CgVariable> in_w, shared_ptr<CgVariable> in_b) {
  vector<CgVariablePtr> xs, hn, cn;
  vector<CgVariablePtr> ys_out, hn_out, cn_out;

  if (seq_len_ == 1) {
    auto tmp = cg_utils::get_item_nd(this->ctx_, in_x, vector<int>{0});
    xs.push_back(tmp);
  } else {
    auto split = make_shared<CgFunction>(create_Split(this->ctx_, 0));
    xs = connect(split, {in_x}, in_x->variable()->shape()[0]);
  }

  // Create graph & pass value
  for (int i = 0; i < num_layers_; i++) {
    vector<CgVariablePtr> hs;
    shared_ptr<CgVariable> hf_var, cf_var, wf_var, bf_var;
    shared_ptr<CgVariable> hb_var, cb_var, wb_var, bb_var;
    auto wi = in_w0;
    if (i > 0) {
      wi = cg_utils::get_item_nd(this->ctx_, in_w,
                                 vector<int>{i - 1}); // w[i-1];
    }

    // Forward on graph
    hf_var =
        cg_utils::get_item_nd(this->ctx_, in_h, vector<int>{i, 0}); // h0[i,0]
    cf_var =
        cg_utils::get_item_nd(this->ctx_, in_c, vector<int>{i, 0}); // c[i,0]
    wf_var = cg_utils::get_item_nd(this->ctx_, wi, vector<int>{0}); // wi[0]
    if (bias_exists_ == true) {
      bf_var =
          cg_utils::get_item_nd(this->ctx_, in_b, vector<int>{i, 0}); // b[i, 0]
    }

    for (vector<CgVariablePtr>::size_type k = 0; k < xs.size(); k++) {
      auto tmp = lstm_cell(xs[k], hf_var, cf_var, wf_var, bf_var);
      hf_var = (tmp[0])[0];
      cf_var = (tmp[1])[0];
      hs.push_back(hf_var);
    }

    hn.push_back(hf_var);
    cn.push_back(cf_var);
    if (bidirectional_ == false) {
      xs = hs;
      continue;
    }

    // backward on graph
    hb_var =
        cg_utils::get_item_nd(this->ctx_, in_h, vector<int>{i, 1}); // h0[i,1]
    cb_var =
        cg_utils::get_item_nd(this->ctx_, in_c, vector<int>{i, 1}); // c0[i,1]
    wb_var = cg_utils::get_item_nd(this->ctx_, wi, vector<int>{1}); // wi[1]
    if (bias_exists_ == true) {
      bb_var =
          cg_utils::get_item_nd(this->ctx_, in_b, vector<int>{i, 1}); // b[i, 1]
    }
    if (xs.size() > 1) {
      std::reverse(std::begin(xs), std::end(xs));
    }
    for (vector<CgVariablePtr>::size_type k = 0; k < xs.size(); k++) {
      int j = xs.size() - 1 - k;
      auto tmp = lstm_cell(xs[k], hb_var, cb_var, wb_var, bb_var);
      hb_var = (tmp[0])[0];
      cb_var = (tmp[1])[0];
      auto concatenate =
          make_shared<CgFunction>(create_Concatenate(this->ctx_, 1));
      auto h_t = connect(concatenate, {hs[j], hb_var}, 1);
      hs[j] = h_t[0];
    }
    hn.push_back(hb_var);
    cn.push_back(cb_var);
    xs = hs;
  }

  auto stack_y = make_shared<CgFunction>(create_Stack(this->ctx_, 0));
  ys_out = connect(stack_y, xs, 1);
  vector<int> shape_ys{seq_len_, batch_size_, num_directions_ * hidden_size_};
  auto reshape_ys =
      make_shared<CgFunction>(create_Reshape(this->ctx_, shape_ys, true));
  ys_out = connect(reshape_ys, ys_out, 1);

  auto stack_hn = make_shared<CgFunction>(create_Stack(this->ctx_, 0));
  hn_out = connect(stack_hn, hn, 1);
  vector<int> shape_hn{num_layers_, num_directions_, batch_size_, hidden_size_};
  auto reshape =
      make_shared<CgFunction>(create_Reshape(this->ctx_, shape_hn, true));
  hn_out = connect(reshape, hn_out, 1);

  auto stack_cn = make_shared<CgFunction>(create_Stack(this->ctx_, 0));
  cn_out = connect(stack_cn, cn, 1);
  auto reshape_cn =
      make_shared<CgFunction>(create_Reshape(this->ctx_, shape_hn, true));
  cn_out = connect(reshape_cn, cn_out, 1);

  return {ys_out, hn_out, cn_out};
}

template <typename T>
void LSTM<T>::forward_impl_training(const Variables &inputs,
                                    const Variables &outputs) {
  bool need_grad = this->training_;

  x_ = make_shared<CgVariable>(inputs[0]->view(), need_grad);  // x
  h_ = make_shared<CgVariable>(inputs[1]->view(), need_grad);  // h
  c_ = make_shared<CgVariable>(inputs[2]->view(), need_grad);  // c
  w0_ = make_shared<CgVariable>(inputs[3]->view(), need_grad); // w0

  if (inputs.size() == 5) {
    if (weight_exists_) {
      w_ = make_shared<CgVariable>(inputs[4]->view(), need_grad); // w
    } else if (bias_exists_) {
      b_ = make_shared<CgVariable>(inputs[4]->view(), need_grad); // b
    }
  }
  if (inputs.size() > 5) {
    w_ = make_shared<CgVariable>(inputs[4]->view(), need_grad); // w
    b_ = make_shared<CgVariable>(inputs[5]->view(), need_grad); // b
  }

  auto out_lstm = create_fixed_length_lstm_graph(x_, h_, c_, w0_, w_, b_);
  ys_ = out_lstm[0];
  hn_ = out_lstm[1];
  cn_ = out_lstm[2];

  auto sink = make_shared<CgFunction>(create_Sink(this->ctx_, false));
  auto dummy = connect(sink, {ys_[0], hn_[0], cn_[0]}, 1);
  dummy[0]->forward(false, true);

  cg_utils::copy_data_cgvariable_to_variable<T>(this->ctx_, ys_[0], outputs[0]);
  cg_utils::copy_data_cgvariable_to_variable<T>(this->ctx_, hn_[0], outputs[1]);
  cg_utils::copy_data_cgvariable_to_variable<T>(this->ctx_, cn_[0], outputs[2]);
}

template <typename T>
void LSTM<T>::forward_impl_inference(const Variables &inputs,
                                     const Variables &outputs) {
  bool need_grad = this->training_;

  x_ = make_shared<CgVariable>(inputs[0]->view(), need_grad);  // x
  h_ = make_shared<CgVariable>(inputs[1]->view(), need_grad);  // h
  c_ = make_shared<CgVariable>(inputs[2]->view(), need_grad);  // c
  w0_ = make_shared<CgVariable>(inputs[3]->view(), need_grad); // w0

  if (inputs.size() == 5) {
    if (weight_exists_) {
      w_ = make_shared<CgVariable>(inputs[4]->view(), need_grad); // w
    } else if (bias_exists_) {
      b_ = make_shared<CgVariable>(inputs[4]->view(), need_grad); // b
    }
  }
  if (inputs.size() > 5) {
    w_ = make_shared<CgVariable>(inputs[4]->view(), need_grad); // w
    b_ = make_shared<CgVariable>(inputs[5]->view(), need_grad); // b
  }

  auto out_lstm = create_fixed_length_lstm_graph(x_, h_, c_, w0_, w_, b_);
  ys_ = out_lstm[0];
  hn_ = out_lstm[1];
  cn_ = out_lstm[2];

  auto sink = make_shared<CgFunction>(create_Sink(this->ctx_, false));
  auto dummy = connect(sink, {ys_[0], hn_[0], cn_[0]}, 1);
  dummy[0]->forward(true, false);

  cg_utils::copy_data_cgvariable_to_variable<T>(this->ctx_, ys_[0], outputs[0]);
  cg_utils::copy_data_cgvariable_to_variable<T>(this->ctx_, hn_[0], outputs[1]);
  cg_utils::copy_data_cgvariable_to_variable<T>(this->ctx_, cn_[0], outputs[2]);
}

template <typename T>
void LSTM<T>::forward_impl(const Variables &inputs, const Variables &outputs) {
  if (this->training_) {
    forward_impl_training(inputs, outputs);
  } else {
    forward_impl_inference(inputs, outputs);
  }
}

template <typename T>
void LSTM<T>::backward_impl(const Variables &inputs, const Variables &outputs,
                            const vector<bool> &propagate_down,
                            const vector<bool> &accum) {
  if (!(propagate_down[0] || propagate_down[1] || propagate_down[2] ||
        propagate_down[3] || (inputs.size() > 4 && propagate_down[4]) ||
        (inputs.size() > 5 && propagate_down[5]))) {
    return;
  }

  NBLA_CHECK(this->training_, error_code::value,
             "Backward is called for training only.");

  if (inputs.size() > 5 && propagate_down[5]) {
    NBLA_CHECK(propagate_down[3] == propagate_down[4], error_code::value,
               "If bias is backpropagated, so should weights.");
  }

  // setting data for input variable & propagate_down & accum
  x_->variable()->set_data(inputs[0]->data());
  if (!propagate_down[0]) {
    x_->set_need_grad(false);
  }
  if (!accum[0]) {
    x_->variable()->grad()->zero();
  } else {
    x_->variable()->set_grad(inputs[0]->grad());
  }

  h_->variable()->set_data(inputs[1]->data());
  if (!propagate_down[1]) {
    h_->set_need_grad(false);
  }
  if (!accum[1]) {
    h_->variable()->grad()->zero();
  } else {
    h_->variable()->set_grad(inputs[1]->grad());
  }

  c_->variable()->set_data(inputs[2]->data());
  if (!propagate_down[2]) {
    c_->set_need_grad(false);
  }
  if (!accum[2]) {
    c_->variable()->grad()->zero();
  } else {
    c_->variable()->set_grad(inputs[2]->grad());
  }

  w0_->variable()->set_data(inputs[3]->data());
  if (!propagate_down[3]) {
    w0_->set_need_grad(false);
  }
  if (!accum[3]) {
    w0_->variable()->grad()->zero();
  } else {
    w0_->variable()->set_grad(inputs[3]->grad());
  }

  if (inputs.size() == 5) {
    if (weight_exists_) {
      w_->variable()->set_data(inputs[4]->data());
      if (!propagate_down[4]) {
        w_->set_need_grad(false);
      }
      if (!accum[4]) {
        w_->variable()->grad()->zero();
      } else {
        w_->variable()->set_grad(inputs[4]->grad());
      }
    } else if (bias_exists_) {
      b_->variable()->set_data(inputs[4]->data());
      if (!propagate_down[4]) {
        b_->set_need_grad(false);
      }
      if (!accum[4]) {
        b_->variable()->grad()->zero();
      } else {
        b_->variable()->set_grad(inputs[4]->grad());
      }
    }
  }

  if (inputs.size() == 6) {
    w_->variable()->set_data(inputs[4]->data());
    if (!propagate_down[4]) {
      w_->set_need_grad(false);
    }
    if (!accum[4]) {
      w_->variable()->grad()->zero();
    } else {
      w_->variable()->set_grad(inputs[4]->grad());
    }

    b_->variable()->set_data(inputs[5]->data());
    if (!propagate_down[5]) {
      b_->set_need_grad(false);
    }
    if (!accum[5]) {
      b_->variable()->grad()->zero();
    } else {
      b_->variable()->set_grad(inputs[5]->grad());
    }
  }

  ys_[0]->variable()->grad()->zero();
  hn_[0]->variable()->grad()->zero();
  cn_[0]->variable()->grad()->zero();

  auto sink = make_shared<CgFunction>(create_Sink(this->ctx_, false));
  auto dummy = connect(sink, {ys_[0], hn_[0], cn_[0]}, 1);

  ys_[0]->variable()->set_grad(outputs[0]->grad());
  hn_[0]->variable()->set_grad(outputs[1]->grad());
  cn_[0]->variable()->set_grad(outputs[2]->grad());

  dummy[0]->backward(nullptr, true);
}
}
