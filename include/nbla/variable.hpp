// Copyright 2017,2018,2019,2020,2021 Sony Corporation.
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

#ifndef __NBLA_VARIABLE_HPP__
#define __NBLA_VARIABLE_HPP__

#include <nbla/common.hpp>
#include <nbla/nd_array.hpp>

#include <memory>

using std::shared_ptr;

namespace nbla {

/** User interface for Array and passed to Function.

Users will create arrays via Variable and pass them to
Function. Variable has two array region internally, data and grad.
Data region is used as an input and/or output of Function::forward(), while
grad region is used for storing backprop error of Function::backward().

\ingroup NNablaCoreGrp
*/
class Variable {
  Shape_t shape_;   ///< Shape.
  Shape_t strides_; ///< C-contiguous strides.
  Size_t size_;     ///< Size.
  Size_t ndim_;     ///< Number of dimensions.
  NdArrayPtr data_; ///< Storing forwardprop results.
  NdArrayPtr grad_; ///< Storing backprop results.

  /** Update shape info by shape.
   */
  void update_shape_info();

public:
  typedef shared_ptr<Variable> Ptr;

  /** Create a shared_ptr instance of Variable.
   */
  template <typename... Args> static Ptr create(Args... args) {
    return make_shared<Variable>(args...);
  }

  /**
  Constructor.

  @param shape Shape.
  */
  NBLA_API Variable(const Shape_t &shape = {});

  /**
  Constructor given NdArray.

  @param data A reference of NdArray created by another can be passed.
  */
  NBLA_API Variable(NdArrayPtr data);

  /** Reshape.
   */
  NBLA_API void reshape(const vector<int64_t> &shape, bool force);

  /**
  Create a new view object without copying data.
  */
  NBLA_API Ptr view();

  /**
  Create a new view object given shape without copying data.

  @param shape Shape. The total size of the shape must match the size of this
  instance.
  */
  NBLA_API Ptr view(const Shape_t &shape);

  /**
  Return shape of variable.
  */
  inline Shape_t shape() const { return shape_; }

  /**
  Return strides of variable.
  */
  inline Shape_t strides() const { return strides_; }

  /** Size of Array (Product of shape dimensions).

  @param axis Size followed by given axis is computed.
   */
  NBLA_API Size_t size(Size_t axis = -1) const;

  /** Number of dimensions of array. */
  inline Size_t ndim() const { return ndim_; }

  /** Get data NdArray.
   */
  inline NdArrayPtr data() { return data_; }

  /** Get grad NdArray.
   */
  inline NdArrayPtr grad() { return grad_; }

  /** Set data NdArray.
   */
  NBLA_API void set_data(NdArrayPtr data);

  /** Set grad NdArray.
   */
  NBLA_API void set_grad(NdArrayPtr grad);

  /**
  A shortcut function to cast data and get pointer.

  @sa SyncedArray::cast() and Array::pointer().
  */
  template <typename T>
  T *cast_data_and_get_pointer(const Context &ctx, bool write_only = false) {
    Array *arr = data_->array()->cast(get_dtype<T>(), ctx, write_only);
    return arr->pointer<T>();
  }

  /**
  A shortcut function to cast grad and get pointer.

  @sa SyncedArray::cast() and Array::pointer().
  */
  template <typename T>
  T *cast_grad_and_get_pointer(const Context &ctx, bool write_only = false) {
    Array *arr = grad_->array()->cast(get_dtype<T>(), ctx, write_only);
    return arr->pointer<T>();
  }

  /**
  A shortcut function to get data pointer.

  @sa SyncedArray::get() and Array::const_pointer().
  */
  template <typename T> const T *get_data_pointer(const Context &ctx) {
    const Array *arr = data_->array()->get(get_dtype<T>(), ctx);
    return arr->const_pointer<T>();
  }

  /**
  A shortcut function to get grad pointer.

  @sa SyncedArray::get() and Array::const_pointer().
  */
  template <typename T> const T *get_grad_pointer(const Context &ctx) {
    const Array *arr = grad_->array()->get(get_dtype<T>(), ctx);
    return arr->const_pointer<T>();
  }

  DISABLE_COPY_AND_ASSIGN(Variable);

private:
  // This is a temporary workaround to share the member variable "mask" of
  // the function Dropout to the derivative function dropout_backward
  // without changing the backward compatibility of their user interfaces.
  // This workaround depends on GradEndFunction of the nnabla.grad scheme.
  // It guarantees the computation order (rank) from Dropout to
  // dropout_backward. However this workaround is dangerous because the
  // dependency is implicit.
  // TODO: the overall refactoring of foward/backward/nnabla.grad to solve
  //       this problem fundamentally.
  using DropoutMaskWorkaroundDependingOnGradEndFunction = Ptr;
  DropoutMaskWorkaroundDependingOnGradEndFunction dropout_mask_ = nullptr;
  // The following friend functions are used to modify Variable object without
  // changing the interface.

  // Get the Variable holding a mask, meaning the externally accessible
  // member variable of Dropout.
  NBLA_API friend Ptr get_dropout_mask(Ptr dropout_input);
  // Set the mask
  NBLA_API friend void set_dropout_mask(Variable *dropout_input,
                                        Ptr dropout_mask);
};

///< Shared pointer of Variable.
typedef Variable::Ptr VariablePtr;
}
#endif
