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

/** Base class of unary operations.
 */
#ifndef __NBLA_FUNCTION_BASE_TRANSFORM_UNARY_HPP__
#define __NBLA_FUNCTION_BASE_TRANSFORM_UNARY_HPP__

#include <nbla/cpu.hpp>
#include <nbla/function.hpp>
#include <nbla/function_registry.hpp>

namespace nbla {

/**
    @todo PLACE HERE FUNCTION DOCUMENTATION.
 */
template <typename... Args>
class BaseTransformUnary : public BaseFunction<Args...> {
public:
  BaseTransformUnary(const Context &ctx, Args... args)
      : BaseFunction<Args...>(ctx, args...) {}
  virtual ~BaseTransformUnary() {}
  virtual int min_inputs() { return 1; }
  virtual int min_outputs() { return 1; }

protected:
  virtual void setup_impl(const Variables &inputs, const Variables &outputs) {
    outputs[0]->reshape(inputs[0]->shape(), true);
  }
};

template <typename T, typename UnaryOp, typename... Args>
class TransformUnary : public BaseTransformUnary<Args...> {
protected:
  UnaryOp unary_op_;

public:
  TransformUnary(const Context &ctx, Args... args)
      : BaseTransformUnary<Args...>(ctx, args...), unary_op_(args...) {}
  virtual ~TransformUnary() {}
  virtual vector<dtypes> in_types() { return vector<dtypes>{get_dtype<T>()}; }
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

class BaseUnaryOp {
public:
  inline BaseUnaryOp() {}
  template <typename T> inline T operator()(const T x) {
    NBLA_ERROR(error_code::not_implemented,
               "Forward operation is not implemented.");
  }
  template <typename T> inline T g(const T dy, const T x, const T y) {
    NBLA_ERROR(error_code::not_implemented,
               "Backward operation is not implemented.");
  }
};

template <typename T, typename UnaryOp>
void transform_unary(int size, const T *x, T *y, UnaryOp op) {
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
  for (int idx = 0; idx < size; ++idx) {
    y[idx] = op(x[idx]);
  }
}

template <typename T, typename UnaryOp, bool accum>
void transform_unary_grad(int size, const T *dy, const T *x, const T *y, T *g,
                          UnaryOp op) {
  for (int idx = 0; idx < size; ++idx) {
    g[idx] = (accum ? g[idx] : (T)0) + op.g(dy[idx], x[idx], y[idx]);
  }
}

template <typename T, typename UnaryOp, typename... Args>
void TransformUnary<T, UnaryOp, Args...>::forward_impl(
    const Variables &inputs, const Variables &outputs) {
  const T *x = inputs[0]->get_data_pointer<T>(this->ctx_);
  T *y = outputs[0]->cast_data_and_get_pointer<T>(this->ctx_, true);
  transform_unary(inputs[0]->size(), x, y, unary_op_);
}

template <typename T, typename UnaryOp, typename... Args>
void TransformUnary<T, UnaryOp, Args...>::backward_impl(
    const Variables &inputs, const Variables &outputs,
    const vector<bool> &propagate_down, const vector<bool> &accum) {
  if (!propagate_down[0]) {
    return;
  }
  const T *dy = outputs[0]->get_grad_pointer<T>(this->ctx_);
  const T *x = inputs[0]->get_data_pointer<T>(this->ctx_);
  const T *y = outputs[0]->get_data_pointer<T>(this->ctx_);
  Size_t size = inputs[0]->size();
  T *g = inputs[0]->cast_grad_and_get_pointer<T>(this->ctx_, !accum[0]);
  if (accum[0])
    transform_unary_grad<T, UnaryOp, true>(size, dy, x, y, g, unary_op_);
  else
    transform_unary_grad<T, UnaryOp, false>(size, dy, x, y, g, unary_op_);
}

#define NBLA_DEFINE_UNARY_OP_CLASS(NAME)                                       \
  class NAME##UnaryOp : public BaseUnaryOp

#define NBLA_DEFINE_UNARY_OP_FORWARD(OP)                                       \
  template <typename T> inline T operator()(const T x) { return OP; }

#define NBLA_DEFINE_UNARY_OP_BACKWARD(GOP)                                     \
  template <typename T> inline T g(const T dy, const T x, const T y) {         \
    return GOP;                                                                \
  }

#define NBLA_DEFINE_TRANSFORM_UNARY_CLASS_COMMON(NAME, DEP_Y)                  \
public:                                                                        \
  virtual ~NAME() {}                                                           \
  virtual string name() { return #NAME; }                                      \
  virtual bool grad_depends_output_data(int i, int o) const { return DEP_Y; }

// ----------------------------------------------------------------------------
// Zero argument
// ----------------------------------------------------------------------------
#define NBLA_DEFINE_UNARY_OP_NO_GRAD(NAME, OP)                                 \
  NBLA_DEFINE_UNARY_OP_CLASS(NAME) {                                           \
  public:                                                                      \
    NBLA_DEFINE_UNARY_OP_FORWARD(OP)                                           \
  }

#define NBLA_DEFINE_UNARY_OP(NAME, OP, GOP)                                    \
  NBLA_DEFINE_UNARY_OP_CLASS(NAME) {                                           \
  public:                                                                      \
    NBLA_DEFINE_UNARY_OP_FORWARD(OP)                                           \
    NBLA_DEFINE_UNARY_OP_BACKWARD(GOP)                                         \
  }

#define NBLA_DEFINE_TRANSFORM_UNARY_CLASS(NAME, DEP_Y)                         \
  template <typename T> class NAME : public TransformUnary<T, NAME##UnaryOp> { \
    NBLA_DEFINE_TRANSFORM_UNARY_CLASS_COMMON(NAME, DEP_Y)                      \
    NAME(const Context &ctx) : TransformUnary<T, NAME##UnaryOp>(ctx) {}        \
    virtual shared_ptr<Function> copy() const {                                \
      return create_##NAME(this->ctx_);                                        \
    }                                                                          \
  }

#define NBLA_DEFINE_TRANSFORM_UNARY_NO_GRAD(NAME, OPERATION)                   \
  NBLA_REGISTER_FUNCTION_HEADER(NAME);                                         \
  NBLA_DEFINE_UNARY_OP_NO_GRAD(NAME, OPERATION);                               \
  NBLA_DEFINE_TRANSFORM_UNARY_CLASS(NAME, false)

/**
Note : If DEP_Y is true, the gradient computation depends on output data.
*/
#define NBLA_DEFINE_TRANSFORM_UNARY(NAME, OP, GOP, DEP_Y)                      \
  NBLA_REGISTER_FUNCTION_HEADER(NAME);                                         \
  NBLA_DEFINE_UNARY_OP(NAME, OP, GOP);                                         \
  NBLA_DEFINE_TRANSFORM_UNARY_CLASS(NAME, DEP_Y)

// ----------------------------------------------------------------------------
// One argument
// ----------------------------------------------------------------------------
#define NBLA_DEFINE_TRANSFORM_UNARY_CLASS_1(NAME, DEP_Y, A0)                   \
  template <typename T>                                                        \
  class NAME : public TransformUnary<T, NAME##UnaryOp, A0> {                   \
    NBLA_DEFINE_TRANSFORM_UNARY_CLASS_COMMON(NAME, DEP_Y)                      \
    NAME(const Context &ctx, const A0 &a0)                                     \
        : TransformUnary<T, NAME##UnaryOp, A0>(ctx, a0) {}                     \
    virtual shared_ptr<Function> copy() const {                                \
      return create_##NAME(this->ctx_, std::get<0>(this->args_));              \
    }                                                                          \
  }

#define NBLA_DEFINE_UNARY_OP_1(NAME, OP, GOP, A0)                              \
  NBLA_DEFINE_UNARY_OP_CLASS(NAME) {                                           \
  public:                                                                      \
    A0 a0;                                                                     \
    inline NAME##UnaryOp(A0 a0_) : a0(a0_) {}                                  \
    NBLA_DEFINE_UNARY_OP_FORWARD(OP)                                           \
    NBLA_DEFINE_UNARY_OP_BACKWARD(GOP)                                         \
  }

#define NBLA_DEFINE_TRANSFORM_UNARY_1(NAME, OP, GOP, DEP_Y, A0)                \
  NBLA_REGISTER_FUNCTION_HEADER(NAME, A0);                                     \
  NBLA_DEFINE_UNARY_OP_1(NAME, OP, GOP, A0);                                   \
  NBLA_DEFINE_TRANSFORM_UNARY_CLASS_1(NAME, DEP_Y, A0)

#define NBLA_DEFINE_UNARY_OP_1_NO_GRAD(NAME, OP, A0)                           \
  NBLA_DEFINE_UNARY_OP_CLASS(NAME) {                                           \
  public:                                                                      \
    A0 a0;                                                                     \
    inline NAME##UnaryOp(A0 a0_) : a0(a0_) {}                                  \
    NBLA_DEFINE_UNARY_OP_FORWARD(OP)                                           \
  }

#define NBLA_DEFINE_TRANSFORM_UNARY_1_NO_GRAD(NAME, OP, A0)                    \
  NBLA_REGISTER_FUNCTION_HEADER(NAME, A0);                                     \
  NBLA_DEFINE_UNARY_OP_1_NO_GRAD(NAME, OP, A0);                                \
  NBLA_DEFINE_TRANSFORM_UNARY_CLASS_1(NAME, false, A0)
}
#endif
