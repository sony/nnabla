// Copyright 2019,2020,2021 Sony Corporation.
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

#ifndef NBLA_UTILS_PARAMETERS_HPP_
#define NBLA_UTILS_PARAMETERS_HPP_

#include <nbla/computation_graph/variable.hpp>
#include <nbla/defs.hpp>
#include <nbla/parametric_functions.hpp>
#include <string>

namespace nbla {
/** Utils. NNabla utilities.
 */
namespace utils {

/** \defgroup utilities for load and save parameters.  */
/** \addtogroup NNablaUtilsNnpGrp */
/*@{*/

/** Load parameters from a parameter file(*.h5, *.protobuf, *.prototxt, *.pb and
 * so on)
 */
NBLA_API bool load_parameters(ParameterDirectory &pd, string filename);

/** Save parameters to specified parameter file.
 */
NBLA_API bool save_parameters(ParameterDirectory &pd, string filename);

/** Load parameters from a file buffer, specified by buffer address and length.
 *  This buffer should contain the parameter file data load from a .h5 file.
 */
NBLA_API bool load_parameters_h5(ParameterDirectory &pd, char *buf, int size);

/** Load parameters from a file buffer, specified by buffer address and length.
 *  This buffer should contain the parameter file data load from a .protobuf
 * file.
 */
NBLA_API bool load_parameters_pb(ParameterDirectory &pd, char *buf, int size);

/** Save parameters to a buffer, specified by buffer address and length.
 */
NBLA_API bool save_parameters_pb(ParameterDirectory &pd, char *buf,
                                 unsigned int &size);

/** Save parameters to a file buffer, specified by buffer address and length.
 */
NBLA_API bool save_parameters_h5(ParameterDirectory &pd, char *buf,
                                 unsigned int &size);
}; // namespace utils
/*@}*/
} // namespace nbla

#endif // NBLA_UTILS_PARAMETERS_HPP_
