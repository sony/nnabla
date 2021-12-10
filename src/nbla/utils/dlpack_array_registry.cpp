// Copyright 2020,2021 Sony Corporation.
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

#include <nbla/utils/dlpack_array_registry.hpp>
#include <nbla/utils/dlpack_utils.hpp>

namespace nbla {

using ArrayToDLDeviceType = map<string, DLDeviceType>;
using DLDeviceTypeToArray = map<DLDeviceType, string>;
using DLDeviceTypeToBackend = map<DLDeviceType, string>;

struct DlpackArrayRegistryMaps {
  DLDeviceTypeToArray device_type_to_array;
  DLDeviceTypeToBackend device_type_to_backend;
  ArrayToDLDeviceType array_to_device_type;
};

template <typename K, typename V>
void raise_error(const map<K, V> &map, const string &key_name,
                 const string &key) {
  vector<K> keys;
  for (const auto &kv : map) {
    keys.push_back(kv.first);
  }

  NBLA_ERROR(error_code::unclassified, "%s %s cannot be found in [%s].",
             key_name.c_str(), key.c_str(), string_join(keys, ", ").c_str());
}

Context DlpackArrayRegistry::create_context(const DLTensor &dl_tensor) {
  init_cpu();

  auto *maps = SingletonManager::get<DlpackArrayRegistryMaps>();

  // Compute the array size
  Size_t size = 1;
  for (int i = 0; i < dl_tensor.ndim; i++) {
    size *= dl_tensor.shape[i];
  }

  // Determine NNabla dtype from DLPack
  const auto dtype = convert_dlpack_type_to_dtype(dl_tensor.dtype);

  // Create NNabla context from DLPack
  const auto dev_t = dl_tensor.ctx.device_type;

  // Array class
  string array_class = "";
  try {
    array_class = maps->device_type_to_array.at(dev_t);
  } catch (std::out_of_range &) {
    raise_error(maps->device_type_to_array, "DLDeviceType", to_string(dev_t));
  }

  // Device ID
  const auto device_id = to_string(dl_tensor.ctx.device_id);

  // Backend
  string backend = "";
  try {
    backend = maps->device_type_to_backend.at(dev_t);
  } catch (std::out_of_range &) {
    raise_error(maps->device_type_to_backend, "DLDeviceType", to_string(dev_t));
  }

  auto str_dtpe = dtype_to_string(dtype);
  // Convert FLOAT to float, and HALF to half
  std::transform(str_dtpe.begin(), str_dtpe.end(), str_dtpe.begin(), ::tolower);
  backend = string_join(vector<string>{backend, str_dtpe}, ":");

  return Context({backend}, array_class, device_id);
}

DLDeviceType
DlpackArrayRegistry::array_to_device_type(const string &array_class) {
  auto *maps = SingletonManager::get<DlpackArrayRegistryMaps>();
  try {
    return maps->array_to_device_type.at(array_class);
  } catch (std::out_of_range &) {
    raise_error(maps->array_to_device_type, "Array class", array_class);
  }
  return kDLCPU;
}

void DlpackArrayRegistry::add_map(const DLDeviceType device_type,
                                  const string &backend,
                                  const string &array_class) {
  auto *maps = SingletonManager::get<DlpackArrayRegistryMaps>();
  maps->device_type_to_array.insert({device_type, array_class});
  maps->device_type_to_backend.insert({device_type, backend});
}

void DlpackArrayRegistry::add_map(const string &array_class,
                                  const DLDeviceType device_tyep) {
  auto *maps = SingletonManager::get<DlpackArrayRegistryMaps>();
  maps->array_to_device_type.insert({array_class, device_tyep});
}

NBLA_INSTANTIATE_SINGLETON(NBLA_API, DlpackArrayRegistryMaps);
}
