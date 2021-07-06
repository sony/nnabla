// Copyright 2017,2018,2019,2020,2021 Sony Corporation.
// Copyright 2021 Sony Group Corporation.
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

#include <nbla/common.hpp>
#include <nbla/cpu.hpp>
#include <nbla/function.hpp>
#include <nbla/function_registry.hpp>
#include <nbla/half.hpp>

namespace nbla {

/**
    @todo PLACE HERE FUNCTION DOCUMENTATION.
 */
template <typename... Args>
class BaseTransformBinary : public BaseFunction<Args...> {
protected:
  const bool inplace_;

  Size_t compressed_ndim_;
  Variable strides_x0_;
  Variable strides_x1_;
  Variable strides_y_;
  Variable shape_y_;

public:
  BaseTransformBinary(const Context &ctx, bool inplace, Args... args)
      : BaseFunction<Args...>(ctx, args...), inplace_(inplace) {}
  virtual ~BaseTransformBinary() {}
  virtual int min_inputs() { return 2; }
  virtual int min_outputs() { return 1; }
  virtual int inplace_data(int i) const {
    if (!inplace_ || i > 0)
      return Function::NOT_INPLACE;
    return Function::INPLACE;
  }
  virtual int inplace_data_with(int i) const {
    // 0 is okay because never be called in the case of i != 0.
    return 0;
  }

protected:
  virtual void setup_impl(const Variables &inputs, const Variables &outputs) {
    Size_t ndim = inputs[0]->ndim();
    NBLA_CHECK(ndim == inputs[1]->ndim(), error_code::value,
               "Dimensions of inputs must match. "
               "inputs[0]: %d != inputs[1]: %d.",
               ndim, inputs[1]->ndim());
    Shape_t s0 = inputs[0]->shape();
    Shape_t s1 = inputs[1]->shape();
    Shape_t oshape(ndim);

    for (Size_t i = 0; i < ndim; ++i) {
      if (s0[i] != s1[i]) {
        NBLA_CHECK(std::min(s0[i], s1[i]) == 1, error_code::value,
                   "Broadcast dimension must be one. shape[%d]: %d.", i,
                   std::min(s0[i], s1[i]));
      }
      oshape[i] = std::max(s0[i], s1[i]);
    }
    outputs[0]->reshape(oshape, true);

    // check in-place conditions
    if (inplace_) {
      NBLA_CHECK(s0 == oshape, error_code::value,
                 "%s: Shapes of inputs[0] and output must match when "
                 "`inplace == true`.",
                 this->name().c_str());
      outputs[0]->data()->set_array(inputs[0]->data()->array());
    }

    // [ Stride computation ]
    // Broadcast for x0 and x1 are performed only by using their strides
    // while computing binary operation. For computational efficiency
    // in cuda backend, the dimensional compression into three are performed.
    // Let [z, y, x] the shape of compressed three dimensions. x-axis is
    // the innermost dimension in NNabla. If compression get success,
    // the final shape must become one of the following five patterns.
    //
    //       no broadcast    broadcast    broadcast          broadcast
    //                        y-axis       x-axis         (x and z)-axis
    //
    //  x0    [1, 1, x]      [z, y, x]    [1, y, x]    [z, y, x]  [1, y, x]
    //  x1    [1, 1, x]      [z, 1, x]    [1, y, 1]    [1, y, 1]  [z, y, 1]
    //
    // Otherwise the compressed dimenstion is more than four. Of cource,
    // the shapes of x0 and x1 are exchangable.

    // Compress the adjascent dimensions which are broadcasted in the same way.
    // In detail, they can be compressed in these three cases.
    //
    //      no broadcast   x0 broadcasted   x1 broadcasted
    //
    // s0      [a, b]          [1, 1]           [a, b]
    // s1      [a, b]          [a, b]           [1, 1]
    //
    Shape_t compressed_shape_x0;
    Shape_t compressed_shape_x1;
    Shape_t compressed_shape_y;
    Size_t tmp_size_x0 = 1;
    Size_t tmp_size_x1 = 1;
    Size_t tmp_size_y = 1;

    for (Size_t i = 0; i < ndim - 1; ++i) {
      tmp_size_x0 *= s0[i];
      tmp_size_x1 *= s1[i];
      tmp_size_y *= oshape[i];

      // Stop this adjascent compression when i- amd (i + 1)-th dimensions
      // do not share the same case among the above three cases.
      // continue above three cases.
      // i-th case
      const bool no_broadcast = (s0[i] == s1[i]);
      const bool x0_broadcast = (s0[i] == 1 && s1[i] != 1);
      const bool x1_broadcast = (s0[i] != 1 && s1[i] == 1);
      // (i + 1)-th case
      const bool no_broadcast_next = (s0[i + 1] == s1[i + 1]);
      const bool x0_broadcast_next = (s0[i + 1] == 1 && s1[i + 1] != 1);
      const bool x1_broadcast_next = (s0[i + 1] != 1 && s1[i + 1] == 1);

      if (!(no_broadcast && no_broadcast_next) &&
          !(x0_broadcast && x0_broadcast_next) &&
          !(x1_broadcast && x1_broadcast_next)) {
        // Stop the adjascent compression
        compressed_shape_x0.push_back(tmp_size_x0);
        compressed_shape_x1.push_back(tmp_size_x1);
        compressed_shape_y.push_back(tmp_size_y);
        tmp_size_x0 = 1;
        tmp_size_x1 = 1;
        tmp_size_y = 1;
      }
    }
    if (ndim > 0) {
      tmp_size_x0 *= s0[ndim - 1];
      tmp_size_x1 *= s1[ndim - 1];
      tmp_size_y *= oshape[ndim - 1];
      compressed_shape_x0.push_back(tmp_size_x0);
      compressed_shape_x1.push_back(tmp_size_x1);
      compressed_shape_y.push_back(tmp_size_y);
    }

    // if the number of the compressed dimensions are less than three,
    // pad it up to three.
    while (compressed_shape_y.size() < 3) {
      compressed_shape_x0.insert(compressed_shape_x0.begin(), 1);
      compressed_shape_x1.insert(compressed_shape_x1.begin(), 1);
      compressed_shape_y.insert(compressed_shape_y.begin(), 1);
    }
    compressed_ndim_ = compressed_shape_y.size();

    // Compress their strides
    Shape_t compressed_strides_x0(compressed_ndim_, 1);
    Shape_t compressed_strides_x1(compressed_ndim_, 1);
    Shape_t compressed_strides_y(compressed_ndim_, 1);

    for (Size_t i = compressed_ndim_ - 2; i >= 0; --i) {
      compressed_strides_x0[i] =
          compressed_strides_x0[i + 1] * compressed_shape_x0[i + 1];
      compressed_strides_x1[i] =
          compressed_strides_x1[i + 1] * compressed_shape_x1[i + 1];
      compressed_strides_y[i] =
          compressed_strides_y[i + 1] * compressed_shape_y[i + 1];
    }

    // Store the compressed strides and shape as Variable.
    // To broadcast an axis, 0 must be set to its stride instead of 1.
    strides_x0_.reshape({compressed_ndim_}, true);
    strides_x1_.reshape({compressed_ndim_}, true);
    strides_y_.reshape({compressed_ndim_}, true);
    shape_y_.reshape({compressed_ndim_}, true);

    Context cpu_ctx = Context().set_array_class(
        SingletonManager::get<Cpu>()->array_classes()[0]);
    Size_t *strides_x0 =
        strides_x0_.cast_data_and_get_pointer<Size_t>(cpu_ctx, true);
    Size_t *strides_x1 =
        strides_x1_.cast_data_and_get_pointer<Size_t>(cpu_ctx, true);
    Size_t *strides_y =
        strides_y_.cast_data_and_get_pointer<Size_t>(cpu_ctx, true);
    Size_t *shape_y = shape_y_.cast_data_and_get_pointer<Size_t>(cpu_ctx, true);

    for (Size_t i = 0; i < compressed_ndim_; ++i) {
      shape_y[i] = compressed_shape_y[i];
      strides_y[i] = compressed_strides_y[i];

      if (compressed_shape_x0[i] == compressed_shape_y[i]) {
        strides_x0[i] = compressed_strides_x0[i];
      } else {
        strides_x0[i] = 0;
      }

      if (compressed_shape_x1[i] == compressed_shape_y[i]) {
        strides_x1[i] = compressed_strides_x1[i];
      } else {
        strides_x1[i] = 0;
      }
    }
  }
};

template <typename T, typename BinaryOp, typename... Args>
class TransformBinary : public BaseTransformBinary<Args...> {
protected:
  BinaryOp binary_op_;

public:
  TransformBinary(const Context &ctx, bool inplace, Args... args)
      : BaseTransformBinary<Args...>(ctx, inplace, args...),
        binary_op_(args...) {}
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
  inline T g0(const T dy, const T x0, const T x1, const T y,
              const bool inplace) {
    NBLA_ERROR(error_code::not_implemented,
               "Backward operation for input 0 is not implemented.");
  }
  template <typename T>
  inline T g1(const T dy, const T x0, const T x1, const T y,
              const bool inplace) {
    NBLA_ERROR(error_code::not_implemented,
               "Backward operation for input 1 is not implemented.");
  }
};

template <typename T, typename BinaryOp>
void transform_binary(const Size_t size, const T *x0, const T *x1, T *y,
                      BinaryOp op, const Size_t ndim, const Size_t *strides_x0,
                      const Size_t *strides_x1, const Size_t *strides_y,
                      const Size_t *shape_y) {
  // Convert the type of intermidiate buffers from Half to float to suppress
  // a decrease in precision during computation.
  using PRECISE_T = typename force_float<T>::type;

  for (Size_t idx = 0; idx < size; ++idx) {
    Size_t idx0 = 0;
    Size_t idx1 = 0;
    for (Size_t i = 0; i < ndim; ++i) {
      Size_t dim_idx = (idx / strides_y[i]) % shape_y[i];
      idx0 += dim_idx * strides_x0[i];
      idx1 += dim_idx * strides_x1[i];
    }
    y[idx] =
        op(static_cast<PRECISE_T>(x0[idx0]), static_cast<PRECISE_T>(x1[idx1]));
  }
}

template <typename T, typename BinaryOp>
void transform_binary_grad0(const Size_t size, const T *dy, const T *x0,
                            const T *x1, const T *y, T *g0, const bool inplace,
                            BinaryOp op, const Size_t ndim,
                            const Size_t *strides_x0, const Size_t *strides_x1,
                            const Size_t *strides_y, const Size_t *shape_y) {
  // Convert the type of intermidiate buffers from Half to float to suppress
  // a decrease in precision during computation.
  using PRECISE_T = typename force_float<T>::type;

  for (Size_t idx = 0; idx < size; ++idx) {
    Size_t idx0 = 0;
    Size_t idx1 = 0;
    for (Size_t i = 0; i < ndim; ++i) {
      Size_t dim_idx = (idx / strides_y[i]) % shape_y[i];
      idx0 += dim_idx * strides_x0[i];
      idx1 += dim_idx * strides_x1[i];
    }
    g0[idx0] =
        static_cast<PRECISE_T>(g0[idx0]) +
        op.g0(static_cast<PRECISE_T>(dy[idx]), static_cast<PRECISE_T>(x0[idx0]),
              static_cast<PRECISE_T>(x1[idx1]), static_cast<PRECISE_T>(y[idx]),
              inplace);
  }
}

template <typename T, typename BinaryOp>
void transform_binary_grad1(const Size_t size, const T *dy, const T *x0,
                            const T *x1, const T *y, T *g1, const bool inplace,
                            BinaryOp op, const Size_t ndim,
                            const Size_t *strides_x0, const Size_t *strides_x1,
                            const Size_t *strides_y, const Size_t *shape_y) {
  // Convert the type of intermidiate buffers from Half to float to suppress
  // a decrease in precision during computation.
  using PRECISE_T = typename force_float<T>::type;

  for (Size_t idx = 0; idx < size; ++idx) {
    Size_t idx0 = 0;
    Size_t idx1 = 0;
    for (Size_t i = 0; i < ndim; ++i) {
      Size_t dim_idx = (idx / strides_y[i]) % shape_y[i];
      idx0 += dim_idx * strides_x0[i];
      idx1 += dim_idx * strides_x1[i];
    }
    g1[idx1] =
        static_cast<PRECISE_T>(g1[idx1]) +
        op.g1(static_cast<PRECISE_T>(dy[idx]), static_cast<PRECISE_T>(x0[idx0]),
              static_cast<PRECISE_T>(x1[idx1]), static_cast<PRECISE_T>(y[idx]),
              inplace);
  }
}

template <typename T, typename BinaryOp, typename... Args>
void TransformBinary<T, BinaryOp, Args...>::forward_impl(
    const Variables &inputs, const Variables &outputs) {
  const T *x0 = inputs[0]->get_data_pointer<T>(this->ctx_);
  const T *x1 = inputs[1]->get_data_pointer<T>(this->ctx_);
  T *y = outputs[0]->cast_data_and_get_pointer<T>(this->ctx_, !this->inplace_);
  const Size_t *strides_x0 =
      this->strides_x0_.template get_data_pointer<Size_t>(this->ctx_);
  const Size_t *strides_x1 =
      this->strides_x1_.template get_data_pointer<Size_t>(this->ctx_);
  const Size_t *strides_y =
      this->strides_y_.template get_data_pointer<Size_t>(this->ctx_);
  const Size_t *shape_y =
      this->shape_y_.template get_data_pointer<Size_t>(this->ctx_);

  transform_binary(outputs[0]->size(), x0, x1, y, binary_op_,
                   this->compressed_ndim_, strides_x0, strides_x1, strides_y,
                   shape_y);
}

template <typename T, typename BinaryOp, typename... Args>
void TransformBinary<T, BinaryOp, Args...>::backward_impl(
    const Variables &inputs, const Variables &outputs,
    const vector<bool> &propagate_down, const vector<bool> &accum) {
  if (!(propagate_down[0] || propagate_down[1])) {
    return;
  }

  const T *x0 = inputs[0]->get_data_pointer<T>(this->ctx_);
  const T *x1 = inputs[1]->get_data_pointer<T>(this->ctx_);
  const T *y = outputs[0]->get_data_pointer<T>(this->ctx_);
  const T *dy = outputs[0]->get_grad_pointer<T>(this->ctx_);
  const Size_t *strides_x0 =
      this->strides_x0_.template get_data_pointer<Size_t>(this->ctx_);
  const Size_t *strides_x1 =
      this->strides_x1_.template get_data_pointer<Size_t>(this->ctx_);
  const Size_t *strides_y =
      this->strides_y_.template get_data_pointer<Size_t>(this->ctx_);
  const Size_t *shape_y =
      this->shape_y_.template get_data_pointer<Size_t>(this->ctx_);
  Size_t size = outputs[0]->size();

  if (propagate_down[0]) {
    if (!accum[0]) {
      inputs[0]->grad()->zero();
    }
    T *dx0 = inputs[0]->cast_grad_and_get_pointer<T>(this->ctx_);
    transform_binary_grad0<T, BinaryOp>(
        size, dy, x0, x1, y, dx0, this->inplace_, binary_op_,
        this->compressed_ndim_, strides_x0, strides_x1, strides_y, shape_y);
  }
  if (propagate_down[1]) {
    if (!accum[1]) {
      inputs[1]->grad()->zero();
    }
    T *dx1 = inputs[1]->cast_grad_and_get_pointer<T>(this->ctx_);
    transform_binary_grad1<T, BinaryOp>(
        size, dy, x0, x1, y, dx1, this->inplace_, binary_op_,
        this->compressed_ndim_, strides_x0, strides_x1, strides_y, shape_y);
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
  inline T g##NUM(const T dy, const T x0, const T x1, const T y,               \
                  const bool inplace) {                                        \
    return GOP;                                                                \
  }

#define NBLA_DEFINE_TRANSFORM_BINARY_CLASS_COMMON(NAME, DEP_Y_0, DEP_Y_1,      \
                                                  DEP_X_0, DEP_X_1)            \
protected:                                                                     \
  virtual bool grad_depends_input_data_impl(int i, int j) const {              \
    if (i == 0)                                                                \
      return DEP_X_0;                                                          \
    return DEP_X_1;                                                            \
  }                                                                            \
                                                                               \
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

#define NBLA_DEFINE_TRANSFORM_BINARY_CLASS(NAME, DEP_Y_0, DEP_Y_1, DEP_X_0,    \
                                           DEP_X_1)                            \
  template <typename T>                                                        \
  class NAME : public TransformBinary<T, NAME##BinaryOp> {                     \
    NBLA_DEFINE_TRANSFORM_BINARY_CLASS_COMMON(NAME, DEP_Y_0, DEP_Y_1, DEP_X_0, \
                                              DEP_X_1)                         \
    NAME(const Context &ctx)                                                   \
        : TransformBinary<T, NAME##BinaryOp>(ctx, false) {}                    \
    virtual shared_ptr<Function> copy() const {                                \
      return create_##NAME(this->ctx_);                                        \
    }                                                                          \
  }

#define NBLA_DEFINE_TRANSFORM_BINARY_CLASS_INPLACE(                            \
    NAME, DEP_Y_0, DEP_Y_1, DEP_X_0, DEP_X_1, IGNORE_INPLACE)                  \
  template <typename T>                                                        \
  class NAME : public TransformBinary<T, NAME##BinaryOp> {                     \
    NBLA_DEFINE_TRANSFORM_BINARY_CLASS_COMMON(NAME, DEP_Y_0, DEP_Y_1, DEP_X_0, \
                                              DEP_X_1)                         \
    NAME(const Context &ctx, bool inplace)                                     \
        : TransformBinary<T, NAME##BinaryOp>(                                  \
              ctx, (IGNORE_INPLACE) ? false : inplace) {}                      \
    virtual shared_ptr<Function> copy() const {                                \
      return create_##NAME(this->ctx_, this->inplace_);                        \
    }                                                                          \
  }

#define NBLA_DEFINE_TRANSFORM_BINARY_NO_GRAD(NAME, OP)                         \
  NBLA_REGISTER_FUNCTION_HEADER(NAME);                                         \
  NBLA_DEFINE_BINARY_OP_NO_GRAD(NAME, OP);                                     \
  NBLA_DEFINE_TRANSFORM_BINARY_CLASS(NAME, false, false, false, false)

#define NBLA_DEFINE_TRANSFORM_BINARY(NAME, OP, GOP0, GOP1, DEP_Y_0, DEP_Y_1,   \
                                     DEP_X_0, DEP_X_1)                         \
  NBLA_REGISTER_FUNCTION_HEADER(NAME);                                         \
  NBLA_DEFINE_BINARY_OP(NAME, OP, GOP0, GOP1);                                 \
  NBLA_DEFINE_TRANSFORM_BINARY_CLASS(NAME, DEP_Y_0, DEP_Y_1, DEP_X_0, DEP_X_1)

#define NBLA_DEFINE_TRANSFORM_BINARY_INPLACE(                                  \
    NAME, OP, GOP0, GOP1, DEP_Y_0, DEP_Y_1, DEP_X_0, DEP_X_1, IGNORE_INPLACE)  \
  NBLA_REGISTER_FUNCTION_HEADER(NAME, bool);                                   \
  NBLA_DEFINE_BINARY_OP(NAME, OP, GOP0, GOP1);                                 \
  NBLA_DEFINE_TRANSFORM_BINARY_CLASS_INPLACE(NAME, DEP_Y_0, DEP_Y_1, DEP_X_0,  \
                                             DEP_X_1, IGNORE_INPLACE)

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

#define NBLA_DEFINE_TRANSFORM_BINARY_CLASS_1(NAME, DEP_Y_0, DEP_Y_1, DEP_X_0,  \
                                             DEP_X_1, A0)                      \
  template <typename T>                                                        \
  class NAME : public TransformBinary<T, NAME##BinaryOp, A0> {                 \
    NBLA_DEFINE_TRANSFORM_BINARY_CLASS_COMMON(NAME, DEP_Y_0, DEP_Y_1, DEP_X_0, \
                                              DEP_X_1)                         \
    NAME(const Context &ctx, const A0 &a0)                                     \
        : TransformBinary<T, NAME##BinaryOp, A0>(ctx, false, a0) {}            \
    virtual shared_ptr<Function> copy() const {                                \
      return create_##NAME(this->ctx_, std::get<0>(this->args_));              \
    }                                                                          \
  }

#define NBLA_DEFINE_TRANSFORM_BINARY_1(NAME, OP, GOP0, GOP1, DEP_Y_0, DEP_Y_1, \
                                       DEP_X_0, DEP_X_1, A0)                   \
  NBLA_REGISTER_FUNCTION_HEADER(NAME, A0);                                     \
  NBLA_DEFINE_BINARY_OP_1(NAME, OP, GOP0, GOP1, A0);                           \
  NBLA_DEFINE_TRANSFORM_BINARY_CLASS_1(NAME, DEP_Y_0, DEP_Y_1, DEP_X_0,        \
                                       DEP_X_1, A0)
}
#endif
