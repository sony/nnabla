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

#ifndef __NBLA_DLPACK_ARRAY_REGISTRY_HPP__
#define __NBLA_DLPACK_ARRAY_REGISTRY_HPP__

#include <dlpack/dlpack.h> // third-party
#include <nbla/context.hpp>

#include <map>

namespace nbla {

using std::map;

/** DlpackArrayCreator class this is never be instantiated.

    This creator class is used only for DlpackArray.
*/
class NBLA_API DlpackArrayRegistry {
  using ArrayToDLDeviceType = map<string, DLDeviceType>;
  using DLDeviceTypeToArray = map<DLDeviceType, string>;
  using DLDeviceTypeToBackend = map<DLDeviceType, string>;

  static DLDeviceTypeToArray device_type_to_array_;
  static DLDeviceTypeToBackend device_type_to_backend_;
  static ArrayToDLDeviceType array_to_device_type_;

public:
  /** Interface to create a context for DlpackArray */
  static Context create_context(const DLTensor &dlp);

  /** Interface to convert an array class to DLDeviceType */
  static DLDeviceType array_to_device_type(const string &array_class);

  /** Register the map from DLDeviceType to Array class and backend name */
  static void add_map(const DLDeviceType device_type, const string &backend,
                      const string &array_class);

  /** Register the map from Array class to DLDeviceType */
  static void add_map(const string &array_class,
                      const DLDeviceType device_tyep);

private:
  // Never be created
  inline DlpackArrayRegistry() {}
};

//------------------------------------------------------------------------------
// Mcros to register the information about DLDeviceType.
//------------------------------------------------------------------------------
#define NBLA_REGISTER_DLPACK_DEVICE_TYPE_TO_CONTEXT(DEVICE_TYPE, BACKEND, CLS) \
  { DlpackArrayRegistry::add_map(DEVICE_TYPE, #BACKEND, #CLS); }

#define NBLA_REGISTER_ARRAY_TO_DLPACK_DEVICE_TYPE(CLS, DEVICE_TYPE)            \
  { DlpackArrayRegistry::add_map(#CLS, DEVICE_TYPE); }
}
#endif
