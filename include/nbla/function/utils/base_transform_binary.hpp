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

/** Base class of binary operations.
 */
#ifndef __NBLA_FUNCTION_BASE_TRANSFORM_BINARY_HPP__
#define __NBLA_FUNCTION_BASE_TRANSFORM_BINARY_HPP__

#include <nbla/cpu.hpp>
#include <nbla/function.hpp>
#include <nbla/function/broadcast.hpp>
#include <nbla/function_registry.hpp>

namespace nbla {

/**
    @todo PLACE HERE FUNCTION DOCUMENTATION.
 */
template <typename... Args>
class BaseTransformBinary : public BaseFunction<Args...> {
protected:
  shared_ptr<Function> f_bc0_, f_bc1_;
  shared_ptr<Variable> o_bc0_, o_bc1_;

public:
  BaseTransformBinary(const Context &ctx, Args... args)
      : BaseFunction<Args...>(ctx, args...) {}
  virtual ~BaseTransformBinary() {}
  virtual int min_inputs() { return 2; }
  virtual int min_outputs() { return 1; }

protected:
  virtual void setup_impl(const Variables &inputs, const Variables &outputs) {
    int ndim = inputs[0]->ndim();
    NBLA_CHECK(ndim == inputs[1]->ndim(), error_code::value,
               "Dimensions of inputs must match. "
               "inputs[0]: %d != inputs[1]: %d.",
               ndim, inputs[1]->ndim());
    Shape_t s0 = inputs[0]->shape();
    Shape_t s1 = inputs[1]->shape();
    Shape_t oshape(ndim);
    bool bc0 = false;
    bool bc1 = false;
    for (int i = 0; i < ndim; ++i) {
      if (s0[i] != s1[i]) {
        NBLA_CHECK(std::min(s0[i], s1[i]) == 1, error_code::value,
                   "Broadcast dimension must be one. shape[%d]: %d.", i,
                   std::min(s0[i], s1[i]));
        if (s0[i] == 1) {
          bc0 = true;
        }
        if (s1[i] == 1) {
          bc1 = true;
        }
      }
      oshape[i] = std::max(s0[i], s1[i]);
    }
    outputs[0]->reshape(oshape, true);
    if (bc0) {
      o_bc0_ = make_shared<Variable>(Shape_t{});
      f_bc0_ = create_Broadcast(this->ctx_,
                                vector<int>(oshape.cbegin(), oshape.cend()));
      f_bc0_->setup(Variables{inputs[0]}, Variables{o_bc0_.get()});
    }
    if (bc1) {
      o_bc1_ = make_shared<Variable>(Shape_t{});
      f_bc1_ = create_Broadcast(this->ctx_,
                                vector<int>(oshape.cbegin(), oshape.cend()));
      f_bc1_->setup(Variables{inputs[1]}, Variables{o_bc1_.get()});
    }
  }
};

template <typename T, typename BinaryOp, typename... Args>
class TransformBinary : public BaseTransformBinary<Args...> {
protected:
  BinaryOp binary_op_;

public:
  TransformBinary(const Context &ctx, Args... args)
      : BaseTransformBinary<Args...>(ctx, args...), binary_op_(args...) {}
  virtual ~TransformBinary() {}
  virtual vector<dtypes> in_types() {
    return vector<dtypes>{get_dtype<T>(), get_dtype<T>()};
  }
  virtual vector<dtypes> out_types() { return vector<dtypes>{get_dtype<T>()}; }
  virtual vector<string> allowed_array_classes() {
    return SingletonManager::get<Cpu>()->array_classes();
  }

protected:
  virtual void forward_impl(const Variables &inputs, const Variables &outputs);
  virtual void backward_impl(const Variables &inputs, const Variables &outputs,
                             const vector<bool> &propagate_down,
                             const vector<bool> &accum);
};

class BaseBinaryOp {
public:
  inline BaseBinaryOp() {}
  template <typename T> inline T operator()(const T x0, const T x1) {
    NBLA_ERROR(error_code::not_implemented,
               "Forward operation is not implemented.");
  }
  template <typename T>
  inline T g0(const T dy, const T x0, const T x1, const T y) {
    NBLA_ERROR(error_code::not_implemented,
               "Backward operation for input 0 is not implemented.");
  }
  template <typename T>
  inline T g1(const T dy, const T x0, const T x1, const T y) {
    NBLA_ERROR(error_code::not_implemented,
               "Backward operation for input 1 is not implemented.");
  }
};

template <typename T, typename BinaryOp>
void transform_binary(int size, const T *x0, const T *x1, T *y, BinaryOp op) {
  for (int idx = 0; idx < size; ++idx) {
    y[idx] = op(x0[idx], x1[idx]);
  }
}

template <typename T, typename BinaryOp, bool accum>
void transform_binary_grad0(int size, const T *dy, const T *x0, const T *x1,
                            const T *y, T *g0, BinaryOp op) {
  for (int idx = 0; idx < size; ++idx) {
    g0[idx] =
        (accum ? g0[idx] : (T)0) + op.g0(dy[idx], x0[idx], x1[idx], y[idx]);
  }
}

template <typename T, typename BinaryOp, bool accum>
void transform_binary_grad1(int size, const T *dy, const T *x0, const T *x1,
                            const T *y, T *g1, BinaryOp op) {
  for (int idx = 0; idx < size; ++idx) {
    g1[idx] =
        (accum ? g1[idx] : (T)0) + op.g1(dy[idx], x0[idx], x1[idx], y[idx]);
  }
}

template <typename T, typename BinaryOp, typename... Args>
void TransformBinary<T, BinaryOp, Args...>::forward_impl(
    const Variables &inputs, const Variables &outputs) {
  auto _get = [this](Variable *v) {
    return v->get_data_pointer<T>(this->ctx_);
  };
  if (this->f_bc0_) {
    this->f_bc0_->forward(Variables{inputs[0]}, Variables{this->o_bc0_.get()});
  }
  if (this->f_bc1_) {
    this->f_bc1_->forward(Variables{inputs[1]}, Variables{this->o_bc1_.get()});
  }
  const T *x0 = _get((this->f_bc0_) ? (this->o_bc0_.get()) : (inputs[0]));
  const T *x1 = _get((this->f_bc1_) ? (this->o_bc1_.get()) : (inputs[1]));
  T *y = outputs[0]->cast_data_and_get_pointer<T>(this->ctx_, true);
  transform_binary(outputs[0]->size(), x0, x1, y, binary_op_);
}

template <typename T, typename BinaryOp, typename... Args>
void TransformBinary<T, BinaryOp, Args...>::backward_impl(
    const Variables &inputs, const Variables &outputs,
    const vector<bool> &propagate_down, const vector<bool> &accum) {
  if (!(propagate_down[0] || propagate_down[1])) {
    return;
  }
  auto _get_data = [this](Variable *v) {
    return v->get_data_pointer<T>(this->ctx_);
  };
  auto _cast_grad = [this](Variable *v, bool wo) {
    return v->cast_grad_and_get_pointer<T>(this->ctx_, wo);
  };
  const T *x0 = _get_data((this->f_bc0_) ? (this->o_bc0_.get()) : (inputs[0]));
  const T *x1 = _get_data((this->f_bc1_) ? (this->o_bc1_.get()) : (inputs[1]));
  const T *dy = outputs[0]->get_grad_pointer<T>(this->ctx_);
  const T *y = _get_data(outputs[0]);
  Size_t size = outputs[0]->size();
  if (propagate_down[0]) {
    T *dx0 = (this->f_bc0_) ? _cast_grad(this->o_bc0_.get(), true)
                            : _cast_grad(inputs[0], !accum[0]);
    if ((!this->f_bc0_) && accum[0])
      transform_binary_grad0<T, BinaryOp, true>(size, dy, x0, x1, y, dx0,
                                                binary_op_);
    else
      transform_binary_grad0<T, BinaryOp, false>(size, dy, x0, x1, y, dx0,
                                                 binary_op_);
    if (this->f_bc0_) {
      this->f_bc0_->backward(Variables{inputs[0]},
                             Variables{this->o_bc0_.get()}, {true}, {accum[0]});
    }
  }
  if (propagate_down[1]) {
    T *dx1 = (this->f_bc1_) ? _cast_grad(this->o_bc1_.get(), true)
                            : _cast_grad(inputs[1], !accum[1]);
    if ((!this->f_bc1_) && accum[1])
      transform_binary_grad1<T, BinaryOp, true>(size, dy, x0, x1, y, dx1,
                                                binary_op_);
    else
      transform_binary_grad1<T, BinaryOp, false>(size, dy, x0, x1, y, dx1,
                                                 binary_op_);
    if (this->f_bc1_) {
      this->f_bc1_->backward(Variables{inputs[1]},
                             Variables{this->o_bc1_.get()}, {true}, {accum[1]});
    }
  }
}

#define NBLA_DEFINE_BINARY_OP_CLASS(NAME)                                      \
  class NAME##BinaryOp : public BaseBinaryOp

#define NBLA_DEFINE_BINARY_OP_FORWARD(OP)                                      \
  template <typename T> inline T operator()(const T x0, const T x1) {          \
    return OP;                                                                 \
  }
#define NBLA_DEFINE_BINARY_OP_BACKWARD(NUM, GOP)                               \
  template <typename T>                                                        \
  inline T g##NUM(const T dy, const T x0, const T x1, const T y) {             \
    return GOP;                                                                \
  }
#define NBLA_DEFINE_TRANSFORM_BINARY_CLASS_COMMON(NAME, DEP_Y_0, DEP_Y_1)      \
public:                                                                        \
  virtual ~NAME() {}                                                           \
  virtual string name() { return #NAME; }                                      \
  virtual bool grad_depends_output_data(int i, int o) const {                  \
    if (i == 0)                                                                \
      return DEP_Y_0;                                                          \
    return DEP_Y_1;                                                            \
  }

// ----------------------------------------------------------------------------
// Zero argument
// ----------------------------------------------------------------------------
#define NBLA_DEFINE_BINARY_OP_NO_GRAD(NAME, OP)                                \
  NBLA_DEFINE_BINARY_OP_CLASS(NAME) {                                          \
  public:                                                                      \
    NBLA_DEFINE_BINARY_OP_FORWARD(OP)                                          \
  }

#define NBLA_DEFINE_BINARY_OP(NAME, OP, GOP0, GOP1)                            \
  NBLA_DEFINE_BINARY_OP_CLASS(NAME) {                                          \
  public:                                                                      \
    NBLA_DEFINE_BINARY_OP_FORWARD(OP)                                          \
    NBLA_DEFINE_BINARY_OP_BACKWARD(0, GOP0)                                    \
    NBLA_DEFINE_BINARY_OP_BACKWARD(1, GOP1)                                    \
  }

#define NBLA_DEFINE_TRANSFORM_BINARY_CLASS(NAME, DEP_Y_0, DEP_Y_1)             \
  template <typename T>                                                        \
  class NAME : public TransformBinary<T, NAME##BinaryOp> {                     \
    NBLA_DEFINE_TRANSFORM_BINARY_CLASS_COMMON(NAME, DEP_Y_0, DEP_Y_1)          \
    NAME(const Context &ctx) : TransformBinary<T, NAME##BinaryOp>(ctx) {}      \
    virtual shared_ptr<Function> copy() const {                                \
      return create_##NAME(this->ctx_);                                        \
    }                                                                          \
  }

#define NBLA_DEFINE_TRANSFORM_BINARY_NO_GRAD(NAME, OP)                         \
  NBLA_REGISTER_FUNCTION_HEADER(NAME);                                         \
  NBLA_DEFINE_BINARY_OP_NO_GRAD(NAME, OP);                                     \
  NBLA_DEFINE_TRANSFORM_BINARY_CLASS(NAME, false, false)

#define NBLA_DEFINE_TRANSFORM_BINARY(NAME, OP, GOP0, GOP1, DEP_Y_0, DEP_Y_1)   \
  NBLA_REGISTER_FUNCTION_HEADER(NAME);                                         \
  NBLA_DEFINE_BINARY_OP(NAME, OP, GOP0, GOP1);                                 \
  NBLA_DEFINE_TRANSFORM_BINARY_CLASS(NAME, DEP_Y_0, DEP_Y_1)

// ----------------------------------------------------------------------------
// One argument
// ----------------------------------------------------------------------------
#define NBLA_DEFINE_BINARY_OP_1(NAME, OP, GOP0, GOP1, A0)                      \
  NBLA_DEFINE_BINARY_OP_CLASS(NAME) {                                          \
  public:                                                                      \
    A0 a0;                                                                     \
    inline NAME##BinaryOp(const A0 &a0_) : a0(a0_) {}                          \
    NBLA_DEFINE_BINARY_OP_FORWARD(OP)                                          \
    NBLA_DEFINE_BINARY_OP_BACKWARD(0, GOP0)                                    \
    NBLA_DEFINE_BINARY_OP_BACKWARD(1, GOP1)                                    \
  }

#define NBLA_DEFINE_TRANSFORM_BINARY_CLASS_1(NAME, DEP_Y_0, DEP_Y_1, A0)       \
  template <typename T>                                                        \
  class NAME : public TransformBinary<T, NAME##BinaryOp, A0> {                 \
    NBLA_DEFINE_TRANSFORM_BINARY_CLASS_COMMON(NAME, DEP_Y_0, DEP_Y_1)          \
    NAME(const Context &ctx, const A0 &a0)                                     \
        : TransformBinary<T, NAME##BinaryOp, A0>(ctx, a0) {}                   \
    virtual shared_ptr<Function> copy() const {                                \
      return create_##NAME(this->ctx_, std::get<0>(this->args_));              \
    }                                                                          \
  }

#define NBLA_DEFINE_TRANSFORM_BINARY_1(NAME, OP, GOP0, GOP1, DEP_Y_0, DEP_Y_1, \
                                       A0)                                     \
  NBLA_REGISTER_FUNCTION_HEADER(NAME, A0);                                     \
  NBLA_DEFINE_BINARY_OP_1(NAME, OP, GOP0, GOP1, A0);                           \
  NBLA_DEFINE_TRANSFORM_BINARY_CLASS_1(NAME, DEP_Y_0, DEP_Y_1, A0)
}
#endif
