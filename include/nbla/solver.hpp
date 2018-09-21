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

/** Solver interface class
 */
#ifndef __NBLA_SOLVER_HPP__
#define __NBLA_SOLVER_HPP__
#include <nbla/array.hpp>
#include <nbla/context.hpp>
#include <nbla/variable.hpp>

#include <memory>
#include <string>
#include <unordered_map>

namespace nbla {

using std::string;
using std::vector;
using std::shared_ptr;
using std::unordered_map;

/** \addtogroup NNablaCoreGrp */
/*@{*/

/** Solver interface which is extended to implement a new Solver class.

Solver takes care of update rule given gradients of parameters.

@f[
w_{t+1} \leftarrow w_t - g_t(\Delta w_t)
@f]

The function @f$g_t(\cdot)@f$ can have an internal state that is updated when
it is called (e.g. Adam).

*/
class NBLA_API Solver {
protected:
  /** Struct for storing both parameter Variable and update status of gradient.
   */
  struct Params {
    /** Shared pointer to parameter Variable.
     */
    VariablePtr p;

    /** Modification count of p.grad_, which tells whether or not the grad
     * region
     * is modified after the previous update.
     */
    size_t at;
  };
  Context ctx_;                          ///< Stores Context.
  unordered_map<string, Params> params_; ///< Hash map of parameters
  bool setup_called_;

public:
  /** Constructor takes at least context and parameters.

  @param ctx Context
  */
  explicit Solver(const Context &ctx);
  virtual ~Solver() = 0;

  ///< Name of Solver class, usually class name.
  virtual string name() = 0;

  ///< Learning rate
  virtual float learning_rate() = 0;

  ///< Set learning rate
  virtual void set_learning_rate(float learning_rate) = 0;

  /** Zeroing grads for all #params_. This is usually called before running
  a sequence of Function::backward() for propagating whole computation graph.
  */
  void zero_grad();

  /** Adding parameters to be optimized via solver.

      It calls set_impl().

  @param params Shared pointers of Variable%s
  @param reset Reset all parameters registered.
  @param retain_state Try to retain state (e.g. momentum) if a parameter is
  overwritten. Note that this will be ignored if reset=true.
  */
  void set_parameters(const vector<pair<string, VariablePtr>> &params,
                      bool reset = true, bool retain_state = false);

  /** Remove previously registered parameters by keys.
   */
  void remove_parameters(const vector<string> &keys);

  /** Clear all parameters.
   */
  void clear_parameters();

  /** Update all params using stored grads in #params_ by backpropagation.

  This internally calls update_impl() which must be implemented in a derived
  class.
  */
  void update();

  /** Apply weight decay to raw gradient.
  It must be called before running update() if necessary.
  This internally calls weight_decay_impl() which must be implemented in a
  derived class.

  It is equivalent to add a squared sum of weight vectors to original loss
  function.

  @f[
  L_{\rm R}({\mathbf w}) = L_{\rm orig}({\mathbf w}) + {\rm decay\_rate } \times
  ||{\mathbf w}||_2^2
  @f]

  @param decay_rate Coefficient of weight decay.
  */
  void weight_decay(float decay_rate);

  /** Check if there is any inf on the gradients which were setup.
   */
  bool check_inf_grad();

  /** Check if there is any nan on the gradients which were setup.
   */
  bool check_nan_grad();

  /** Check if there is any inf or nan on the gradients which were setup.
   */
  bool check_inf_or_nan_grad();

  /** Scale gradients,then increase the loss scale
   */
  void scale_grad(float scale);

  /** Get array classes that are allowed to be specified by Context
  */
  virtual vector<string> allowed_array_classes();

protected:
  /** Setup function implicitly called when first calling set_parameters().
   */
  void setup();

  /** Set state (e.g. momentum).

  @param key Key of parameter.
  @param param Parameter variable.
  */
  virtual void set_state_impl(const string &key, VariablePtr param) = 0;

  /** Remove state (e.g. momentum).

  @param key Key of parameter.
  */
  virtual void remove_state_impl(const string &key) = 0;

  /** Update implementation.

  @param key Key of parameter.
  @param param Parameter variable. The raw gradient of loss
  function is stored in grad region.

  */
  virtual void update_impl(const string &key, VariablePtr param) = 0;

  /** Weight decay implementation.

  @param key Key of parameter.
  @param param A parameter Variable.
  @param decay_rate Coefficient of weight decay.
  */
  virtual void weight_decay_impl(const string &key, VariablePtr param,
                                 float decay_rate) = 0;

  /** Check if there is any inf on the gradients which were setup.
   */
  virtual bool check_inf_grad_impl(const string &key, VariablePtr param) = 0;

  /** Check if there is any nan on the gradients which were setup.
   */
  virtual bool check_nan_grad_impl(const string &key, VariablePtr param) = 0;

  /** Check if there is any inf or nan on the gradients which were setup.
   */
  virtual bool check_inf_or_nan_grad_impl(const string &key,
                                          VariablePtr param) = 0;

  /** Scale gradients, then increase the loss scale
   */
  virtual void scale_grad_impl(const string &key, VariablePtr param,
                               float scale) = 0;

  DISABLE_COPY_AND_ASSIGN(Solver);
};
/*@}*/

#define NBLA_DECL_WEIGHT_DECAY()                                               \
  virtual void weight_decay_impl(const string &key, VariablePtr param,         \
                                 float decay_rate)

#define NBLA_DEF_WEIGHT_DECAY(SOLVER, WEIGHT_DECAY_FUNC)                       \
  template <typename T>                                                        \
  void SOLVER<T>::weight_decay_impl(const string &key, VariablePtr param,      \
                                    float decay_rate) {                        \
    WEIGHT_DECAY_FUNC<T>(this->ctx_, param, decay_rate);                       \
  }

#define NBLA_DECL_CHECK_INF_GRAD()                                             \
  virtual bool check_inf_grad_impl(const string &key, VariablePtr param)

#define NBLA_DEF_CHECK_INF_GRAD(SOLVER, CHECK_INF_GRAD_FUNC)                   \
  template <typename T>                                                        \
  bool SOLVER<T>::check_inf_grad_impl(const string &key, VariablePtr param) {  \
    return CHECK_INF_GRAD_FUNC<T>(this->ctx_, param);                          \
  }

#define NBLA_DECL_CHECK_NAN_GRAD()                                             \
  virtual bool check_nan_grad_impl(const string &key, VariablePtr param)

#define NBLA_DEF_CHECK_NAN_GRAD(SOLVER, CHECK_NAN_GRAD_FUNC)                   \
  template <typename T>                                                        \
  bool SOLVER<T>::check_nan_grad_impl(const string &key, VariablePtr param) {  \
    return CHECK_NAN_GRAD_FUNC<T>(this->ctx_, param);                          \
  }

#define NBLA_DECL_CHECK_INF_OR_NAN_GRAD()                                      \
  virtual bool check_inf_or_nan_grad_impl(const string &key, VariablePtr param)

#define NBLA_DEF_CHECK_INF_OR_NAN_GRAD(SOLVER, CHECK_INF_OR_NAN_GRAD_FUNC)     \
  template <typename T>                                                        \
  bool SOLVER<T>::check_inf_or_nan_grad_impl(const string &key,                \
                                             VariablePtr param) {              \
    return CHECK_INF_OR_NAN_GRAD_FUNC<T>(this->ctx_, param);                   \
  }

#define NBLA_DECL_SCALE_GRAD()                                                 \
  virtual void scale_grad_impl(const string &key, VariablePtr param,           \
                               float scale)

#define NBLA_DEF_SCALE_GRAD(SOLVER, SCALE_GRAD_FUNC)                           \
  template <typename T>                                                        \
  void SOLVER<T>::scale_grad_impl(const string &key, VariablePtr param,        \
                                  float scale) {                               \
    SCALE_GRAD_FUNC<T>(this->ctx_, param, scale);                              \
  }

/** \defgroup SolverImplGrp Solver list */
/*@{*/
/*@}*/
}
#endif
