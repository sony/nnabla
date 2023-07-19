// Copyright 2017,2018,2019,2020,2021 Sony Corporation.
// Copyright 2022 Sony Group Corporation.
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

typedef function<void(void)> update_hook_type;

/** Callback helper class for update callbacks

This is used from Python frontend.
*/
class UpdateHookWithObject {
public:
  typedef function<void(void *)> setup_callback_type;
  typedef function<void(void *)> cleanup_callback_type;
  typedef function<void(void *)> callback_type;

private:
  void *obj_{nullptr};
  callback_type callback_;
  setup_callback_type setup_callback_;
  cleanup_callback_type cleanup_callback_;

public:
  NBLA_API UpdateHookWithObject();
  NBLA_API UpdateHookWithObject(const UpdateHookWithObject &from);
  NBLA_API UpdateHookWithObject(void *obj, callback_type cb,
                                setup_callback_type setup_cb,
                                cleanup_callback_type clean_cb);
  NBLA_API ~UpdateHookWithObject();
  NBLA_API UpdateHookWithObject &operator=(const UpdateHookWithObject &rhs);

  NBLA_API void operator()();
};

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
public:
  /** Struct for storing both parameter state Variable and iteration
   */
  struct SolverState {
    unordered_map<string, VariablePtr> pstate; ///< Parameter state maps
    uint32_t t;                                ///< Iteration as state
    SolverState(){};
    SolverState(unordered_map<string, VariablePtr> pstate, uint32_t t) {
      this->pstate = pstate;
      this->t = t;
    };
    SolverState(const SolverState &state) {
      this->pstate = state.pstate;
      this->t = state.t;
    };
    SolverState &operator=(const SolverState &state) {
      this->pstate = state.pstate;
      this->t = state.t;
      return *this;
    };
  };

protected:
  /** Struct for storing a parameter variable.

      @note The previous implementation had another member variable in this
            struct to manage update status of a parameter. Now it doesn't.
   */
  struct Params {
    /** Shared pointer to parameter Variable.
     */
    VariablePtr p;
  };

  unordered_map<string, SolverState> states_; ///< Hash map of states

  Context ctx_;                          ///< Stores Context.
  unordered_map<string, Params> params_; ///< Hash map of parameters
  bool setup_called_;

  /**
     Specify whether weight decay is fused into update.
  */
  const bool weight_decay_is_fused_;

  /**
     The default value of weight decay rate (valid only if
     weight_decay_is_fused_ returns true).
     This decay rate will be used unless you call weight_decay to set a decay
     rate for the next update() call.
  */
  const float default_weight_decay_rate_;

  /**
     Weight decay rate temporarily specified by weight_decay() (valid only if
     weight_decay_is_fused_ is true).
     This decay rate value will expire after update() and will revert to the
     default_weight_decay_rate_.
  */
  float weight_decay_rate_;

  /** Constructor takes a context
   */
  Solver(const Context &ctx);

  /** Constructor to set a weight_decay_is_fused_ flag to the class instance.
   */
  Solver(const Context &ctx, bool wd_is_fused);

  /** Constructor to set a weight_decay_is_fused_ flag and the default weight
  decay rate to the class instance.

  For Solver instances which weight_decay_is_fused_ is true (the weight decay
  operation is fused into the udpate oepration), the weight_decay_rate specified
  to the constructor is used in the update operation by default. If you call
  weight_decay, the specified decay rate is stored in the instance until it's
  consumed by the update function for lazy evaluation.

  @param ctx Context
  @param wd_is_used Specify if the weight decay operation is fused into the
  update operation. The default is false.
  @param weight_decay_rate The default weight decay value. The default is 0.
  */
  Solver(const Context &ctx, bool wd_is_fused, float weight_decay_rate);

public:
  virtual ~Solver() = 0;

  ///< Name of Solver class, usually class name.
  virtual string name() = 0;

  virtual float learning_rate() = 0;

  ///< Set learning rate
  virtual void set_learning_rate(float learning_rate) = 0;

  /** Whether the weight decay is lazily evaluated at update_impl.
   */
  bool weight_decay_is_fused() const;

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

  /** Get all parameters.
   */
  vector<pair<string, VariablePtr>> get_parameters();

  /** Get all states.
   */
  vector<pair<string, SolverState>> get_states();

  /** Set states.
   */
  void set_states(const vector<pair<string, SolverState>> &params);

  /** Clear states.
   */
  void clear_state(const string &key) {
    for (auto &p : states_[key].pstate) {
      p.second->data()->array()->clear();
    }
  }

  /** Update all params using stored grads in #params_ by backpropagation.

  This internally calls update_impl() which must be implemented in a derived
  class.
  */
  void update(update_hook_type pre_callback = nullptr,
              update_hook_type post_callback = nullptr);

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
  void weight_decay(float decay_rate, update_hook_type pre_callback = nullptr,
                    update_hook_type post_callback = nullptr);

  /** Clip gradients by norm.
  The norm is calculated at each variable.
   */
  void clip_grad_by_norm(float norm, update_hook_type pre_callback = nullptr,
                         update_hook_type post_callback = nullptr);

  /** Check if there is any inf on the gradients which were setup.
   */
  bool check_inf_grad(update_hook_type pre_callback = nullptr,
                      update_hook_type post_callback = nullptr);

  /** Check if there is any nan on the gradients which were setup.
   */
  bool check_nan_grad(update_hook_type pre_callback = nullptr,
                      update_hook_type post_callback = nullptr);

  /** Check if there is any inf or nan on the gradients which were setup.
   */
  bool check_inf_or_nan_grad(update_hook_type pre_callback = nullptr,
                             update_hook_type post_callback = nullptr);

  /** Scale gradients,then increase the loss scale
   */
  void scale_grad(float scale, update_hook_type pre_callback = nullptr,
                  update_hook_type post_callback = nullptr);

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

  /**
   */
  // virtual void get_state_impl() = 0;

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

  /** Clip gradients by norm implementation.

  @param key Key of parameter.
  @param param A parameter Variable.
  @param norm A value of norm.
  */
  virtual void clip_grad_by_norm_impl(const string &key, VariablePtr param,
                                      float clip_norm) = 0;

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
                                 float decay_rate);

#define NBLA_DEF_WEIGHT_DECAY(SOLVER, WEIGHT_DECAY_FUNC)                       \
  template <typename T>                                                        \
  void SOLVER<T>::weight_decay_impl(const string &key, VariablePtr param,      \
                                    float decay_rate) {                        \
    WEIGHT_DECAY_FUNC<T>(this->ctx_, param, decay_rate);                       \
  }

#define NBLA_DECL_CLIP_GRAD_BY_NORM()                                          \
  virtual void clip_grad_by_norm_impl(const string &key, VariablePtr param,    \
                                      float clip_norm)

#define NBLA_DEF_CLIP_GRAD_BY_NORM(SOLVER, CLIP_GRAD_BY_NORM_FUNC)             \
  template <typename T>                                                        \
  void SOLVER<T>::clip_grad_by_norm_impl(const string &key, VariablePtr param, \
                                         float clip_norm) {                    \
    CLIP_GRAD_BY_NORM_FUNC<T>(this->ctx_, param, clip_norm);                   \
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
} // namespace nbla
#endif
