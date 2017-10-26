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

/** Communicator interface class
 */
#ifndef __NBLA_COMMUNICATOR_HPP__
#define __NBLA_COMMUNICATOR_HPP__
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

/** Communicator interface which is extended to implement a new Communicator
class.

Communicator exchanges gradients.

*/
class NBLA_API Communicator {
protected:
  Context ctx_;
  int rank_;
  int size_;  // number of workers.

  vector<Context> contexts_;
  vector<vector<pair<string, VariablePtr>>> device_func_named_param_;
  vector<vector<pair<string, VariablePtr>>> func_device_named_param_;

  bool initialized_ = false;
  Size_t total_params_ = 0;  // total number of parameters.

public:
  /** Constructor takes at least context and parameters.

  @param ctx Context
  */
  explicit Communicator(const Context &ctx);
  virtual ~Communicator() = 0;

  ///<  Name of Communicator class, usually class name.
  virtual string name() = 0;

  int rank();
  int size();

  /** Add context and parameters

  @param pair pair<Context, vector of pair<name, VariablePtr>
  */
  void add_context_and_parameters(
      const pair<Context, vector<pair<string, VariablePtr>>> &ctx_params);

  /** Remove previously registered parameters.
   */
  void remove_context_parameters(const pair<Context, vector<string>> &ctx_keys);

  /** Clear all parameters.
   */
  void clear_context_parameters();

  /** Initall or initrank, depending multi-threads or multi-processes.
   * This function \b MUST be called after all parameters communicated
   * are added by \e add_context_and_parameters method.
  */
  virtual void init();

  /** Check difference of the array classes.
   * Check difference between the array class of the context and that of
   * the synced_array. If it differs, the error occurs.
   */
  void check_array_class(Context ctx, VariablePtr vp);

  /** reduce.
   @param division Divide the reduced value.
   */
  virtual void reduce(bool division = true);

  /** allreduce over parameters added.
   This method is \b sync before and after iallreduce w.r.t. a host thread.
   Currently, \e iallreduce is applied to gradient regions.

  @param division Divide the reduced value.
  @param inplace Pack the arrays into one large array if flase.
   */
  virtual void allreduce(bool division = true, bool inplace=false);

  /** reducescatter.
   @param division Divide the reduced value.
   */
  virtual void reducescatter(bool division = true);

  /** broadcast.
   *
   */
  virtual void bcast();

  /** allgather.
   *
   */
  virtual void allgather();

  /** reduce asynchronously.
   @param division Divide the reduced value.
   */
  virtual void reduce_async(bool division = true);

  /** reduce asynchronously.
   @param division Divide the reduced value.
	 @param inplace Pack the arrays into one large array if flase.
   */
  virtual void allreduce_async(bool division = true, bool inplace = true);

  /** reducescatter asynchronously.
   @param division Divide the reduced value.
   */
  virtual void reducescatter_async(bool division = true);

  /** broadcast asynchronously.
   *
   */
  virtual void bcast_async();

  /** allgather asynchronously.
   *
   */
  virtual void allgather_async();

  /** Get array classes that are allowed to be specified by Context.
  */
  vector<string> allowed_array_classes();

protected:
  DISABLE_COPY_AND_ASSIGN(Communicator);
};
/*@}*/

/** \defgroup CommunicatorImplGrp Communicator list */
/*@{*/
/*@}*/
}
#endif
