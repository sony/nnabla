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
#include "hdf5_wrapper.hpp"

namespace nbla {
namespace utils {
using namespace std;
bool load_from_h5_file(string h5file, vector<string> &data_names,
                       vector<NdArrayPtr> &ndarrays) {
  NBLA_ERROR(error_code::not_implemented,
             "Cannot load from .h5. HDF5 might not enabled when build.");
  return false;
}

bool load_from_h5_buffer(char *buffer, size_t size, vector<string> &data_names,
                         vector<NdArrayPtr> &ndarrays) {
  NBLA_ERROR(error_code::not_implemented,
             "Cannot load from .h5. HDF5 might not enabled when build.");
  return false;
}

bool load_parameters_h5(ParameterVector &pv, string h5file) {
  NBLA_ERROR(error_code::not_implemented,
             "Cannot load from .h5. HDF5 might not enabled when build.");
  return false;
}

bool load_parameters_h5(ParameterVector &pv, char *buffer, size_t size) {
  NBLA_ERROR(
      error_code::not_implemented,
      "Cannot load parameters from .h5. HDF5 might not enabled when build.");
  return false;
}

bool save_parameters_h5(const ParameterVector &pv, char *buffer,
                        unsigned int &size) {
  NBLA_ERROR(
      error_code::not_implemented,
      "Cannot save parameters to .h5. HDF5 might not enabled when build.");
  return false;
}

bool save_parameters_h5(const ParameterVector &pv, string filename) {
  NBLA_ERROR(
      error_code::not_implemented,
      "Cannot save parameters to .h5. HDF5 might not enabled when build.");
  return false;
}
} // namespace utils
} // namespace nbla
