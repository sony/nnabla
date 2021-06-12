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
#ifndef __NBLA_MULTIPROCESSDATAPARALLELCOMMUNICATOR_HPP__
#define __NBLA_MULTIPROCESSDATAPARALLELCOMMUNICATOR_HPP__
#include <nbla/array.hpp>
#include <nbla/communicator.hpp>
#include <nbla/communicator_registry.hpp>
#include <nbla/context.hpp>
#include <nbla/variable.hpp>

#include <memory>
#include <string>
#include <unordered_map>

namespace nbla {

NBLA_REGISTER_COMMUNICATOR_HEADER(MultiProcessDataParallelCommunicator);

using std::string;
using std::vector;
using std::pair;
using std::shared_ptr;
using std::unordered_map;

/** \addtogroup NNablaCoreGrp */
/*@{*/

/** MultiProcessDataParallelCommunicator interface which is extended to
implement a new DataParallelcommunicator class.

MultiProcessDataParallelCommunicator exchanges gradients parameters or
parameters itself.

*/
template <typename T>
class NBLA_API MultiProcessDataParallelCommunicator : public Communicator {

public:
  /** Constructor takes at least context and parameters.

  @param ctx Context
  */
  MultiProcessDataParallelCommunicator(const Context &ctx);
  virtual ~MultiProcessDataParallelCommunicator();

  //  Name of MultiProcessDataParallelCommunicator class, usually class name.
  virtual string name() { return "MultiProcessDataParallelCommunicator"; }

  /** Adding context and parameters communicated via this class.
  @param cparams pair<Context, vector of pair<name, VariablePtr>
  */
  void add_context_and_parameters(
      const pair<Context, vector<pair<string, VariablePtr>>> &ctx_params);

  /** Remove previously registered parameters by keys.
   */
  void remove_context_parameters(const pair<Context, vector<string>> &ctx_keys);

  /** Clear all parameters.
   */
  void clear_context_parameters();

  /** Initall or initrank, depending multi-threads or multi-processes.
   * This function MUST be called after all parameters communicated
   * are added by `add_context_and_parameters` method.
  */
  virtual void init();

  virtual string new_group(pair<string, vector<int>> name_ranks_pair);
  virtual unordered_map<string, vector<int>> list_groups();
  virtual bool find_self(const string &group);
  virtual vector<int> find_group(const string &group);

  virtual void reduce(const vector<NdArrayPtr> &ndarray_list, int dst,
                      bool division = false, bool inplace = false,
                      const string &group = "world");
  virtual void reduce(NdArrayPtr ndarray, int dst, bool division = false,
                      bool inplace = false, const string &group = "world");
  virtual void allreduce(bool division = false, bool inplace = false);
  virtual void all_reduce(const vector<NdArrayPtr> &ndarray_list,
                          bool division = false, bool inplace = false,
                          const string &group = "world");
  virtual void all_reduce(NdArrayPtr ndarray, bool division = false,
                          bool inplace = false, const string &group = "world");
  virtual void reduce_scatter(const vector<NdArrayPtr> &ndarray_list,
                              NdArrayPtr ndarray, bool division = false,
                              const string &group = "world");
  virtual void bcast(const vector<NdArrayPtr> &ndarray_list, int src,
                     bool inplace = false, const string &group = "world");
  virtual void bcast(NdArrayPtr ndarray, int src, bool inplace = false,
                     const string &group = "world");
  virtual void all_gather(NdArrayPtr ndarray,
                          const vector<NdArrayPtr> &ndarray_list,
                          const string &group = "world");

  virtual void reduce_async(bool division = false);
  virtual void allreduce_async(bool division = false, bool inplace = false);
  virtual void reducescatter_async(bool division = false);
  virtual void bcast_async();
  virtual void allgather_async();

  /** Get array classes that are allowed to be specified by Context
  */
  vector<string> allowed_array_classes();

protected:
  unordered_map<string, vector<int>> groups_;

  DISABLE_COPY_AND_ASSIGN(MultiProcessDataParallelCommunicator);
};
/*@}*/

/** \defgroup MultiProcessDataParallelCommunicatorImplGrp
 * MultiProcessDataParallelCommunicator list */
/*@{*/
/*@}*/
}
#endif
