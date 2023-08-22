// Copyright 2023 Sony Group Corporation.
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

#ifndef NBLA_UTILS_HDF5_WRAPPER_HPP_
#define NBLA_UTILS_HDF5_WRAPPER_HPP_

#include "parameters_impl.hpp"
#include <nbla/computation_graph/variable.hpp>
#include <nbla/defs.hpp>
#include <nbla/parametric_functions.hpp>
#include <string>

namespace nbla {
namespace utils {

/** load .h5 key-value pairs by filename
 */
bool load_from_h5_file(std::string h5file, std::vector<std::string> &data_names,
                       std::vector<NdArrayPtr> &ndarrays);

/** load .h5 key-value pairs from buffer
 */
bool load_from_h5_buffer(char *buffer, size_t size,
                         std::vector<std::string> &data_names,
                         std::vector<NdArrayPtr> &ndarrays);

/** load .h5 parameters by filename
 */
bool load_parameters_h5(ParameterVector &pv, std::string h5file);

/** load .h5 parameters from buffer
 */
bool load_parameters_h5(ParameterVector &pv, char *buffer, size_t size);

/** save .h5 parameters to file buffer
 */
bool save_parameters_h5(const ParameterVector &pv, char *buffer,
                        unsigned int &size);

/** save .h5 parameters to file
 */
bool save_parameters_h5(const ParameterVector &pv, std::string filename);
} // namespace utils
} // namespace nbla

#endif // NBLA_UTILS_HDF5_WRAPPER_HPP_
