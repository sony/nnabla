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

/** Function interface class
 */
#ifndef __NBLA_FUNCTION_HPP__
#define __NBLA_FUNCTION_HPP__
#include <nbla/array.hpp>
#include <nbla/context.hpp>
#include <nbla/variable.hpp>

#include <memory>
#include <string>
#include <tuple>

namespace nbla {

using std::string;
using std::vector;
using std::shared_ptr;
using std::tuple;
using std::get;

/** \defgroup NNablaCoreGrp Core components of NNabla */
/*@{*/

/// Variable%s as a vector of raw pointer.
typedef vector<Variable *> Variables;

/** An interface for the units of computation.
This is extended to implement a new Function which implements forward
computation (forward() function) of the function

@f[
{\mathbf y} = f({\mathbf x})
@f]

and backward computation (backward() function)

@f[
  \Delta {\mathbf x} += \Delta {\mathbf y} \cdot \nabla_{\mathbf x} {\mathbf y}
@f]

where @f$\Delta {\mathbf x}@f$ and @f$\Delta {\mathbf y}@f$ are backpropagation
error (gradient) of the input and output variable
propagated through backward computations of descendant of computation graph, and
@f$\nabla_{\mathbf x} {\mathbf y}@f$ is a
<a href="https://en.wikipedia.org/wiki/Jacobian_matrix_and_determinant">Jacobian
matrix</a>
of the function. Note that propagated error is not substituted but
accumulated to the gradient of the input variable because we assume
@f${\mathbf x}@f$ can be used more than once by other functions. In the above
 example, the number of each of input and output variable is one, but Function
 can take multiple inputs and outputs.
*/
class NBLA_API Function {
  bool used_{false};

public:
  typedef shared_ptr<Function> Ptr;

protected:
  Context ctx_;                           ///< Storing context.
  vector<shared_ptr<Shape_t>> in_shapes;  ///< Storing input shapes.
  vector<shared_ptr<Shape_t>> out_shapes; ///< Storing output shapes.

  /** Fall back function. If this is set at instantiation of Function, behavior
  of the function will be replaced with the fall-back function.
  */
  Ptr fall_back_func_;

public:
  // Inplace level used in inplace_data function.
  static const int NOT_INPLACE = 0;
  static const int INPLACE_NOT_MODIFY = 1;
  static const int INPLACE = 2;

  /// Copying and storing Context.
  explicit Function(const Context &ctx);
  virtual ~Function() = 0;

  /** Setting up function.

  This must be called before the Function instance is used, and will do:

  - Determine Array class used according to the return of
    Function::allowed_array_classes and  the given Context.
  - Type and shape check.
  - Calling Function::setup_impl.
  - Pre-allocate memory to prevent locking in asynchronous execution in
    CUDA etc.

  @param inputs vector of Variable*
  @param outputs vector of Variable*
  */
  void setup(const Variables &inputs, const Variables &outputs);

  /** Compute forwardprop and store results into outputs' data.

  Checking shapes before calling forward_impl() which must be implemented in
  a derived function.
  @sa setup() arguments.
  */
  void forward(const Variables &inputs, const Variables &outputs);

  /** Compute backprop multiplied with outputs' grad and store it to inputs'
  grad.

  Checking shapes before calling backward_impl() which must be implemented in
  a derived function.
  @sa setup() arguments.
  */
  void backward(const Variables &inputs, const Variables &outputs,
                const vector<bool> &propagate_down, const vector<bool> &acccum);

  /** Get Context used in this function.
  */
  Context context() const;

  /** Get input dtypes.

  Last in_type will be used repeatedly if size of in_types is smaller than size
  of inputs
  */
  virtual vector<dtypes> in_types() = 0;

  /** Get output dtypes.

  Last out_type will be used repeatedly if size of out_types is smaller than
  size of outputs
  */
  virtual vector<dtypes> out_types() = 0;

  /** Get minimum number of inputs.

  This is meant to be used in setup function with in_types which is used to get
  maximum number of inputs.
  */
  virtual int min_inputs() = 0;

  /** Get minimum number of outputs.

  This is meant to be used in setup function with out_types which is used to get
  max number of outputs.
  */
  virtual int min_outputs() = 0;

  /** Get function name in string
  */
  virtual string name() = 0;

  /** Get array classes that are allowed to be specified by Context
  */
  virtual vector<string> allowed_array_classes() = 0;

  /** Dependency flag for checking if in-grad depends on out-data.

      If i=1 and o=0, checking checking if i-th input' gradient
      computation requires o-th output's data or not.

      @param[in] i Input variable index.
      @param[in] o Output variable index.

      @note If any of inputs requires an output variable data when computing
      its gradient, this function must be overridden to return appropriate
      boolean value. Otherwise, backward computation will be incorrect.
   */
  virtual bool grad_depends_output_data(int i, int o) const { return false; }
  /** Dependency flag for checking if in-grad depends on in-data.

      If i=1 and j=0, checking checking if i-th input' gradient
     computation requires j-th input's data or not.

      By default, always returns true. If override this in a sub-class, the
      computation graph engine will optimize memory usage.

      @param[in] i Input variable index.
      @param[in] j Input variable index.

   */
  virtual bool grad_depends_input_data(int i, int j) const { return true; }

  /** Get in-place-level of i-th input variable's data (see below).

      * 0 (NOT_INPLACE): Not in-placed
      * 1 (INPLACE_NOT_MODIFY): In-placed but not modified.
      * 2 (INPLACE): In-placed and modified.

      @param[in] i Input variable index.
      @retval Returns 0 by default.
      @note If a subclass uses in-place computation, the function must override
     this function.
   */
  virtual int inplace_data(int i) const { return NOT_INPLACE; }

  /** Get the output variable index where i-th variables' data in-placed to.

      @param[in] i Input variable index.
      @note This is only valid if the i-th variable is in-placed.
            The maintainer of a sub-class function must override
            this function.
   */
  virtual int inplace_data_with(int i) const {
    NBLA_ERROR(
        error_code::not_implemented,
        "This must be implemented for in-place support of this function.");
  }

  /** Get in-place-level of i-th input variable's grad (see below).

      * 0 (NOT_INPLACE): Not in-placed
      * 1 (INPLACE_NOT_MODIFY): In-placed but not modified.
      * 2 (INPLACE): In-placed and modified.

      @param[in] i Input variable index.
      @retval Returns 0 by default.
      @note If a subclass uses in-place computation, the function must override
     this function.
   */
  virtual int inplace_grad(int i) const { return NOT_INPLACE; }

  /** Get the output variable index where i-th variables' grad in-placed to.

      @param[in] i Input variable index.
      @note This is only valid if the i-th variable is in-placed.
            The maintainer of a sub-class function must override
            this function.
   */
  virtual int inplace_grad_with(int i) const {
    NBLA_ERROR(
        error_code::not_implemented,
        "This must be implemented for in-place support of this function.");
  }

  /** A flag for preventing that the graph engine clears buffers of
      input variables even if clear_buffer is true and condition mets.
   */
  virtual bool prohibit_clear_input_buffers() const { return false; }

  /** A flag for preventing that the graph engine sets input gradient buffer as
   * 0 even when accum is true.
   */
  virtual bool prohibit_zero_input_grad() const { return false; }

  /** Copy another instance of Function with the same context.
  */
  virtual shared_ptr<Function> copy() const = 0;

  /** Check whether this was already used, and turn it used.
   */
  inline bool ask_if_used_and_use() {
    bool r = used_;
    used_ = true;
    return r;
  }

protected:
  /** Implementation part of setup().

  It must do:

  - Reshape output Variable%s.
  - Allocate resources used in forward/backward computation if necessary.
  - Checking shapes and dtypes etc.
  @sa setup() for parameters
  */
  virtual void setup_impl(const Variables &inputs,
                          const Variables &outputs) = 0;

  /** Implementation part of forward().

  It must do:

  - Take data in inputs and store results into data in outputs.

  @sa setup() arguments.
  */
  virtual void forward_impl(const Variables &inputs,
                            const Variables &outputs) = 0;

  /** Implementation part of backward().

  It must do:

  - Take grad in outputs (backpropagated error from children of a computational
    graph) and compute Jacobian multiplication of this function with grad.
  - Store backprop error into grad in inputs.

  @param propagate_down Boolean array that indicates whether backprop is needed
  for an input corresponding to its index.

  @sa setup() arguments.
  */
  virtual void backward_impl(const Variables &inputs, const Variables &outputs,
                             const vector<bool> &propagate_down,
                             const vector<bool> &accum) = 0;

  DISABLE_COPY_AND_ASSIGN(Function);
};

/** Base function.

    Keep arguments.
 */
template <typename... Args> class BaseFunction : public Function {
protected:
  tuple<typename std::remove_reference<Args>::type...> args_;

public:
  BaseFunction(const Context &ctx, Args... args)
      : Function(ctx), args_(args...) {}

  /** Get number of constructor arguments.
   */
  int num_args() { return sizeof...(Args); }

  /** Get constructor arguments as a tuple.
   */
  const tuple<Args...> &get_args() { return args_; }

  /** Get a constructor argument by index.
   */
  template <int Index> auto get_arg() -> decltype(std::get<Index>(args_)) {
    return std::get<Index>(args_);
  }
};
/*@}*/

typedef Function::Ptr FunctionPtr;

/** \defgroup FunctionImplGrp Function list */
/*@{*/
/*@}*/
}
#endif
