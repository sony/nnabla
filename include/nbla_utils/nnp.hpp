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

#ifndef NBLA_UTILS_NNP_HPP_
#define NBLA_UTILS_NNP_HPP_

#include <nbla/computation_graph/variable.hpp>
#include <string>

namespace nbla {
/** Utils. NNabla utilities.
*/
namespace utils {

/** \defgroup NNablaUtilsNnpGrp Utilities for NNabla format files.  */
/** \addtogroup NNablaUtilsNnpGrp */
/*@{*/

/** NNabla format file utilities.
 */
namespace nnp {

// Forward dec.
class NnpImpl;
class NetworkImpl;
class ExecutorImpl;

// ----------------------------------------------------------------------
// Network
// ----------------------------------------------------------------------

/** Network object associated with Nnp object.

    The following code will get Network instance from Nnp object nnp,
    and set batch size.
    @code{.cpp}
    shared_ptr<Network> network = nnp.get_network("net1");
    network.set_batch_size(64);
    @endcode

    The next block will get the references to the variable object in the
    computation graph by name. The computation graph is built when
    get_variable(name) is called first time, or first time since the batch size
    or network topology is changed.
    @code{.cpp}
    nbla::CgVariablePtr x = network.get_variable("input");
    nbla::CgVariablePtr y = network.get_variable("output");
    @endcode

    You can set data to a variable by accessing array data by using NNabla C++
    interface.
    @code{.cpp}
    float *data = x->variable()->cast_data_and_get_pointer<float>(
        dl.Context().set_array_class("CpuCachedArray"));
    for (int i = 0; i < x->variable()->size(); i++) {
        data[i] = ...;  // Set data
    }
    @endcode

    The forward propagation of the network can be executed at any variable by
    calling forward method. The function execution will be propagated from root
    (input) variables to to the variable.
    @code{.cpp}
    y->forward(true);
    @endcode

    Getting and displaying output are as follows.
    @code{.cpp}
    const float *out = y->variable()->get_data_pointer<float>(
        dl.Context().set_array_class("CpuCachedArray"));
    for (int i = 0; i < y->variable()->size(); i++) {
        std::cout << out[i] << ",";
    }
    std::cout << std::endl;
    @endcode

    @note The Network instance is created by a class member function
          Nnp::get_network().
          The constructor is hidden, and not called directly by users.
 */
class Network {
  friend NnpImpl;
  std::unique_ptr<NetworkImpl> impl_;
  Network(NetworkImpl *impl);

public:
  /** Network name.
   */
  string name() const;

  /** Set batch size.

      @param[in] batch_size Overwrite the default batch size in nnp file.
  */
  void set_batch_size(int batch_size);

  /** Get batch size.

      @retval Batch size. The if set_batch_size is not previously called,
      batch size written in nnp file will be returned.
  */
  int batch_size();

  /** Replace an arbitrary variable in the network with a given
      variable.

      The predecessors of the variable in the networks are dicarded, and
      replaced with the predecessors of the given variable.

      @param[in] name Name of variable in the network you are replacing.
      @param[in] variable Replaced with this.
   */
  void replace_variable(const string &name, CgVariablePtr variable);

  /** Get a variable by name.

      This is usually used to set or get data inside the variable.
      The construction of a computation graph is invoked by calling this
      if the graph is not latest or not created.

      @param[in] name Name of variable in the network.
      @retval Variable in a computation graph.
   */
  CgVariablePtr get_variable(const string &name);
};

// ----------------------------------------------------------------------
// Executor
// ----------------------------------------------------------------------

/** Executor associated with Nnp object.

    The Executor object internally stores a Network object.
 */
class Executor {
  friend NnpImpl;
  std::unique_ptr<ExecutorImpl> impl_;
  Executor(ExecutorImpl *impl);

public:
  /** Data variable container.

      The string fields corresponds to DataVariable in proto definition.
   */
  struct DataVariable {
    const string variable_name;
    const string data_name;
    const CgVariablePtr variable;
  };

  /** Output variable container.

      The string fields corresponds to OutputVariable in proto definition.
   */
  struct OutputVariable {
    const string variable_name;
    const string type;
    const string data_name;
    const CgVariablePtr variable;
  };

  /** Executor name.
   */
  string name() const;

  /** Network name.
   */
  string network_name() const;

  /** Set batch size.

      @param[in] batch_size Overwrite the default batch size in Network.
  */
  void set_batch_size(int batch_size);

  /** Get batch size.

      @retval Batch size. The if set_batch_size is not previously called,
      batch size written in the Network of NNabla format file will be returned.
  */
  int batch_size();

  /** Get data variables.

      @retval Data variables where each item holds name info and CgVariable
              instance in the Network. The data inside the CgVariable should be
              gotten via Nnabla C++ interface.
   */
  vector<DataVariable> get_data_variables();

  /** Get output variables.

      @retval Output variables where each item holds name info and CgVariable
              instance in the Network. The data inside the CgVariable should be
              gotten via Nnabla C++ interface.
   */
  vector<OutputVariable> get_output_variables();

  /** Get the reference (shared_ptr) of Network object held in this.
   */
  shared_ptr<Network> get_network();

  /** Execute the network from inputs to outputs.
   */
  void execute();
};

// ----------------------------------------------------------------------
// Nnp
// ----------------------------------------------------------------------

/** Handle of NNabla format files.

    You can create an Nnp object by passing default context as below.

    @code{.cpp}
    using namespace nbla::utils::nnp;
    nbla::Context ctx{"cpu", "CpuCachedArray", "0", "default"};
    Nnp nnp(ctx);
    @endcode

    Suppose we have network.nnp which is previously created. You can add a
    previsouly dumped NNabla format file to Nnp object.
    Nnp will parse the file format and internally store the information such as
    network architectures, learned parameters and execution settings.
    @code{.cpp}
    nnp.add("network.nnp");
    @endcode

    Suppose a network "net1" is in network.npp. The following line will create
    a Network object from the nnp file. Network can create a
    computation graph defined in NNabla format files. The created computation
    graph can be executed in C++ code. See Network doc for the usage.
    @code{.cpp}
    shared_ptr<Network> network = nnp.get_network("net1");

    ... // Use network here.
    @endcode

    Suppose an executor "exe1" is in network.npp. The following line will
    create a Executor object from NNabla format files. The Executor can also
    create a computation graph of a network associated with the Executor field
    in NNabla format files. The Executor provides easier interface to set input,
    execute the graph, and get output.
    @code{.cpp}
    shared_ptr<Executor> executor = nnp.get_executor("exe1");

    ... // Use executor here.
    @endcode
 */
class Nnp {
  std::unique_ptr<NnpImpl> impl_;

public:
  /** Constructor which sets default context.

      @param[in] ctx Default context which overwrites the config in nnp file.
   */
  Nnp(const nbla::Context &ctx);
  // dtor
  ~Nnp();

  /** Add nnp|nntxt|h5 file
   */
  bool add(const string &filename);

  /** Get Network name list from added files (nnp, nntxt etc.).
      @retval A vector of Network instance names.
   */
  vector<string> get_network_names();

  /** Get Network object from added files (nnp, nntxt etc.).

      @param[in] name Network name in loaded files (nnp, nntxt etc.)

      @retval A shared pointer of a Network instance.
   */
  shared_ptr<Network> get_network(const string &name);

  /** Get Executor name list from added files (nnp, nntxt etc.).
      @retval A vector of Executor instance names.
   */
  vector<string> get_executor_names();

  /** Get Executor object from added file(s).

      @param[in] name Executor name in loaded files (nnp, nntxt etc.)

      @retval A shared pointer of a Executor instance.
   */
  shared_ptr<Executor> get_executor(const string &name);
};
}
/*@}*/
}
}

#endif // NBLA_UTILS_NNP_HPP_
