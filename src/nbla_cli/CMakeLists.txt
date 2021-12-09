# Copyright 2017,2018,2019,2020,2021 Sony Corporation.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

set(LIB_NAME nnabla_cli${LIB_NAME_SUFFIX})

include(${CMAKE_SOURCE_DIR}/build-tools/cmake/Utils.cmake)
if (WIN32)
  # Use simulated cxxabi.h since it is not appeared in MSVC. 
  include_directories(${PROJECT_SOURCE_DIR}/include/msvc)
  set(CXXABI_HPP cxxabi.cpp)
endif()

if (NNABLA_UTILS_WITH_HDF5)
  add_definitions(-DNBLA_UTILS_WITH_HDF5)
endif()

if (NNABLA_UTILS_WITH_HDF5)
if (NNABLA_UTILS_STATIC_LINK_DEPS)
    set(HDF5_USE_STATIC_LIBRARIES 1)
endif()
findhdf5()
endif()
if (NNABLA_UTILS_STATIC_LINK_DEPS)
if (NNABLA_UTILS_WITH_HDF5)
    set(HDF5_LIBRARIES ${HDF5_LIBRARIES} ${CMAKE_DL_LIBS})
endif()
endif()

if (NNABLA_UTILS_WITH_HDF5)
include_directories(${HDF5_INCLUDE_DIRS})
endif()


# nbla target definition
add_executable(nbla
  internal.cpp
  nbla.cpp
  nbla_dump.cpp
  nbla_infer.cpp
  nbla_train.cpp
  ${CXXABI_HPP})

target_link_libraries(nbla ${NBLA_LIBRARY_NAME} ${NBLA_UTILS_LIBRARY_NAME})
set_property(TARGET nbla PROPERTY CXX_STANDARD 14)

# nbla_cli shared library
add_library(${LIB_NAME}
  SHARED
  internal.cpp
  nbla_train.cpp
  nbla_dump.cpp
  nbla_infer.cpp
  ${CXXABI_HPP})

target_link_libraries(${LIB_NAME} ${NBLA_LIBRARY_NAME} ${NBLA_UTILS_LIBRARY_NAME})
set_property(TARGET ${LIB_NAME} PROPERTY CXX_STANDARD 14)

install(TARGETS nbla RUNTIME DESTINATION bin)
install(TARGETS ${LIB_NAME} LIBRARY DESTINATION lib)
install(DIRECTORY ${PROJECT_SOURCE_DIR}/include/nbla_cli DESTINATION include)
include(CPack)

set(NBLA_CLI_LIBRARY_NAME ${LIB_NAME} PARENT_SCOPE)
