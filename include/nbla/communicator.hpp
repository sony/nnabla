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

/** Communicator interface class
 */
#ifndef __NBLA_COMMUNICATOR_HPP__
#define __NBLA_COMMUNICATOR_HPP__
#include <nbla/array.hpp>
#include <nbla/computation_graph/variable.hpp>
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
  int local_rank_;
  int size_; // number of workers.

  vector<Context> contexts_;
  vector<vector<pair<string, VariablePtr>>> device_func_named_param_;
  vector<vector<pair<string, VariablePtr>>> func_device_named_param_;

  bool initialized_ = false;
  Size_t total_params_ = 0; // total number of parameters.

public:
  /** Constructor takes at least context and parameters.

  @param ctx Context
  */
  explicit Communicator(const Context &ctx);
  virtual ~Communicator() = 0;

  ///<  Name of Communicator class, usually class name.
  virtual string name() = 0;

  int rank();
  int local_rank();
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

  /** Synchronize all processes in the specified group.
  */
  virtual void barrier();

  /** Abort all processes in the specified group.
  */
  virtual void abort();

  /** Create group
   */
  virtual string new_group(pair<string, vector<int>> name_ranks_pair);

  /** List groups
   */
  virtual unordered_map<string, vector<int>> list_groups();

  /** Find groups
   */
  virtual vector<int> find_group(const string &group);

  /** Check difference of the array classes.
   * Check difference between the array class of the context and that of
   * the synced_array. If it differs, the error occurs.
   */
  void check_array_class(Context ctx, VariablePtr vp);

  /** reduce over parameters added.

        @param ndarray_list Vector of NdArrayPtr.
        @param dst Destination rank.
  @param division Divide the reduced value.
  @param inplace Pack the arrays into one large array if false.
  @param group Name of a group.
   */
  virtual void reduce(const vector<NdArrayPtr> &ndarray_list, int dst,
                      bool division = false, bool inplace = false,
                      const string &group = "world");

  /** reduce over parameters added.

  @param data NdArrayPtr.
  @param dst Destination rank.
@param division Divide the reduced value.
@param inplace Pack the arrays into one large array if false.
@param group Name of a group.
*/
  virtual void reduce(NdArrayPtr ndarray, int dst, bool division = false,
                      bool inplace = false, const string &group = "world");

  /** allreduce over parameters added.
   Deprecated. Use all_reduce.

   Currently, \e allreduce is applied to gradient regions.

  @param division Divide the reduced value.
  @param inplace Pack the arrays into one large array if false.
  @param group Name of a group.
   */
  virtual void allreduce(bool division = false, bool inplace = false);

  /** all_reduce over parameters added.

        @param ndarray_list Vector of NdArrayPtr
  @param division Divide the reduced value.
  @param inplace Pack the arrays into one large array if false.
        @param group Name of a group.

   */
  virtual void all_reduce(const vector<NdArrayPtr> &ndarray_list,
                          bool division = false, bool inplace = false,
                          const string &group = "world");

  /** all_reduce over parameters added.

  @param data NdArrayPtr
@param division Divide the reduced value.
@param inplace Pack the arrays into one large array if false.
  @param group Name of a group.
*/
  virtual void all_reduce(NdArrayPtr ndarray, bool division = false,
                          bool inplace = false, const string &group = "world");

  /** all_reduce over parameters added.

  @param ndarray_list Vector of NdArrayPtr
  @param pack_size The number of values contained in the packed data.
  @param division Divide the reduced value.
   */
  virtual CommunicatorBackwardCallbackPtr
  all_reduce_callback(const vector<NdArrayPtr> &ndarray_list, size_t pack_size,
                      bool division = false, const string &group = "world");

  /** all_reduce over parameters added.

  @param ndarray NdArrayPtr
  @param pack_size The number of values contained in the packed data.
  @param division Divide the reduced value.
   */
  virtual CommunicatorBackwardCallbackPtr
  all_reduce_callback(NdArrayPtr ndarray, size_t pack_size,
                      bool division = false, const string &group = "world");

  /** reducescatter.

   @param ndarray_list Vector of NdArrayPtr
   @param ndarray NdArrayPtr
   @param division Divide the reduced value.
   @param group Name of a group.
   */
  virtual void reduce_scatter(const vector<NdArrayPtr> &ndarray_list,
                              NdArrayPtr ndarray, bool division = false,
                              const string &group = "world");

  /** broadcast.

         @param ndarray_list Vector of NdArrayPtr.
         @param src Source rank.
         @param inplace Pack the arrays into one large array if false.
         @param group Name of a group.
   */
  virtual void bcast(const vector<NdArrayPtr> &ndarray_list, int src,
                     bool inplace = false, const string &group = "world");

  /** broadcast.

         @param data NdArrayPtr.
         @param src Source rank.
         @param inplace Pack the arrays into one large array if false.
         @param group Name of a group.
   */
  virtual void bcast(NdArrayPtr ndarray, int src, bool inplace = false,
                     const string &group = "world");

  /** all_gather.

         @param ndarray data to be sent.
         @param ndarray_list Vector of NdArrayPtr to receive data.
         @param group Name of a group.
   */
  virtual void all_gather(NdArrayPtr ndarray,
                          const vector<NdArrayPtr> &ndarray_list,
                          const string &group = "world");

  /** reduce asynchronously.
   @param division Divide the reduced value.
   */
  virtual void reduce_async(bool division = false);

  /** reduce asynchronously.
   @param division Divide the reduced value.
         @param inplace Pack the arrays into one large array if false.
   */
  virtual void allreduce_async(bool division = false, bool inplace = true);

  /** reducescatter asynchronously.
   @param division Divide the reduced value.
   */
  virtual void reducescatter_async(bool division = false);

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
