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

#ifndef __NBLA_COMPUTATION_GRAPH_HPP__
#define __NBLA_COMPUTATION_GRAPH_HPP__

#include <nbla/computation_graph/function.hpp>
#include <nbla/computation_graph/variable.hpp>
#include <nbla/nd_array.hpp>
#include <nbla/singleton_manager.hpp>

#include <memory>
#include <mutex>
#include <thread>

namespace nbla {

/** Create CgVariable outputs.

    Created CGVariables are held as weak_ptr by cg_f. The `need_grad`
    flags are automatically applied to created outputs.

*/
NBLA_API vector<CgVariablePtr>
create_function_outputs(CgFunctionPtr cg_f, int n_outputs = -1,
                        bool prohibit_clear_output = false);

/** Connect function to network.

    Create outputs on-demand.
 */
NBLA_API vector<CgVariablePtr> connect(CgFunctionPtr cg_f,
                                       const vector<CgVariablePtr> &inputs,
                                       int n_outputs = 1,
                                       vector<NdArrayPtr> inplace_outputs = {},
                                       bool execute = false);

/** Steal some variable properties from `from` CgVariable to `to` in order to
   rewire previously constructed graphs.

    It takes parent, need_grad flags, inplace flags, and the variable contents.

    @param[in] from Its parent function is stolen by 'to'.
    @param[in,out] to A variable 'from''s parent stolen to.
*/
NBLA_API void steal_variable_from_to(CgVariablePtr from, CgVariablePtr to);

/** Forward given variables in single inference
 * Forward all given variables with shared fclosed flags.
 */
NBLA_API void forward_all(const vector<CgVariablePtr> variables,
                          bool clear_buffer = false,
                          bool clear_no_need_grad = false,
                          function_hook_type function_pre_hook = nullptr,
                          function_hook_type function_post_hook = nullptr);

/** Clear buffer flags maintained in a global scope (per thread).
 *
 * This is used to inform nbla::Function class which buffer flag is used when
 * the function is called.
 * It's used in the forward & backward function in the nbla::CgVariable class.
 */
class NBLA_API GlobalClearBufferState {
  mutable unordered_map<std::thread::id, bool> clear_buffer_;
  mutable unordered_map<std::thread::id, bool> clear_no_need_grad_;
  mutable std::mutex mtx_;

  class NBLA_API ScopedState {
    GlobalClearBufferState *self_;
    bool clear_buffer_;
    bool clear_no_need_grad_;

  public:
    ScopedState(GlobalClearBufferState *self, bool clear_buffer,
                bool clear_no_need_grad);
    ~ScopedState();
    DISABLE_COPY_AND_ASSIGN(ScopedState);
  };

public:
  bool clear_buffer() const;
  bool clear_no_need_grad() const;
  /** Set clear buffer flags globally until the life of the returned object
     ends.

      Note that this doesn't affect any decision of clearing buffers in
      the graph exeuction. It's used to just inform anyone of the current
      clear buffer flag.
      Also, please keep in mind that the returned ScopedState instance
      shouldn't be owned by any globally maintained instance because
      it maintains a raw pointer of a global singleton instance of
      this class (GlobalClearBufferState).

      @code{.cpp}
      // Set a global clear buffer flag as true.
      auto clear_buffer_state =
        SingletonManager::get<GlobalClearBufferState>()->state(true, false);

      // The following will return true until clear_buffer_state is
      // destroyed (at exiting this scope).
      auto c = SingletonManager::get<GlobalClearBufferState>()->clear_buffer();
      @endcode
   */
  std::unique_ptr<ScopedState> state(bool clear_buffer,
                                     bool clear_no_need_grad);

private:
  friend SingletonManager;
  void set(bool clear_buffer, bool clear_no_need_grad);
  GlobalClearBufferState();
  DISABLE_COPY_AND_ASSIGN(GlobalClearBufferState);
};
}
#endif
