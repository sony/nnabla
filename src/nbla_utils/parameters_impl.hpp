// Copyright (c) 2019 Sony Corporation. All Rights Reserved.
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

#ifndef NBLA_UTILS_PARAMETERS_IMPL_HPP_
#define NBLA_UTILS_PARAMETERS_IMPL_HPP_

// NNabla
#include <nbla/computation_graph/function.hpp>
#include <nbla/computation_graph/variable.hpp>
#include <nbla/solver.hpp>
#include <string>

namespace nbla {
/** Utils. NNabla utilities.
*/
namespace utils {

/** \defgroup utilities for load and save parameters.  */
/** \addtogroup NNablaUtilsNnpGrp */
/*@{*/

/** Type for sharing implementation with different requirements.
 */
typedef vector<pair<string, CgVariablePtr>> ParameterVector;

/** load .h5 parameters from buffer
 */
bool load_parameters_h5(ParameterVector &pv, char *buffer, int size);

/** load .protobuf parameters from buffer
 */
bool load_parameters_pb(ParameterVector &pv, char *buffer, int size);

/** load .h5 parameters by filename
 */
bool load_parameters_h5(ParameterVector &pv, string filename);

/** load .protobuf parameters by filename
 */
bool load_parameters_pb(ParameterVector &pv, string filename);

/** load parameters by filename, judge file format by extension name.
 */
bool load_parameters(ParameterVector &pv, string filename);

/** save .h5 parameters to file
 */
bool save_parameters_h5(const ParameterVector &pv, string filename);

/** save .protobuf parameters to file
 */
bool save_parameters_pb(const ParameterVector &pv, string filename);

/** save parameters by filename, judge file format by extension name.
 */
bool save_parameters(const ParameterVector &pv, string filename);
};
/*@}*/
}
#endif // NBLA_UTILS_PARAMETERS_HPP_
