# Copyright (c) 2017 Sony Corporation. All Rights Reserved.
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

set(LIB_NAME nnabla_utils${LIB_NAME_SUFFIX})

include(${CMAKE_SOURCE_DIR}/build-tools/cmake/Utils.cmake)

option(NNABLA_UTILS_WITH_HDF5 "Enable HDF5 in C++ utils." OFF)
option(NNABLA_UTILS_WITH_NPY "Enable DataIterator with NPY Cache." ON)
option(NNABLA_UTILS_STATIC_LINK_DEPS "Static link C++ NNabla utils deps." OFF)
option(DEBUG_NETWORK_EXPANDER "Debug network expander." OFF)

if (DEBUG_NETWORK_EXPANDER)
  add_definitions(-DDEBUG_NETWORK_EXPANDER)
endif()

include_directories(${CMAKE_BINARY_DIR}/libraries/include)

list(APPEND nbla_utils_SOURCES
  nnp.cpp
  nnabla.pb.cc
  nnp_impl.cpp
  nnp_impl_create_function.cpp
  nnp_impl_create_solver.cpp
  nnp_impl_optimizer.cpp
  nnp_impl_monitor.cpp
  nnp_impl_configs.cpp
  nnp_network_expander.cpp
  parameters.cpp
)

if (UNIX)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread")
endif()

# NPY is prior option, if it is ON,
# data iterator is set to NPY, and hdf5
# is disabled.
if (NNABLA_UTILS_WITH_NPY)
  add_definitions(-DUSE_NPY_CACHE_DATA_ITERATOR)
  add_definitions(-DNBLA_UTILS_WITH_NPY)
  list(APPEND nbla_utils_SOURCES
    nnp_impl_dataset_npy.cpp
    data_iterator_npy.cpp
  )
elseif (NNABLA_UTILS_WITH_HDF5)
  list(APPEND nbla_utils_SOURCES
    nnp_impl_dataset_hdf5.cpp
    data_iterator.cpp
  )
else()
  list(APPEND nbla_utils_SOURCES
    data_iterator.cpp
  )
endif()

if (NNABLA_UTILS_WITH_HDF5)
  add_definitions(-DNBLA_UTILS_WITH_HDF5)
endif()

add_library(${LIB_NAME} SHARED
  ${nbla_utils_SOURCES}
)

if (WIN32)
  if (NNABLA_UTILS_WITH_HDF5)
    message(FATAL_ERROR, "HDF5 cannot be enabled on Windows.")
    error() # Undefined command to abort cmake process here.
  endif()
  find_package(LibArchive REQUIRED)
  find_package(ZLIB REQUIRED)
  find_package(Protobuf REQUIRED)
  include_directories(
    ${LibArchive_INCLUDE_DIR}
    ${PROTOBUF_INCLUDE_DIR}
    ${ZLIB_INCLUDE_DIRS}
    )
  target_link_libraries(${LIB_NAME}
    ${NBLA_LIBRARY_NAME}
    ${PROTOBUF_LIBRARY}
    ${LibArchive_LIBRARIES}
    ${ZLIB_LIBRARIES}
    )
else()
  if (${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
    if (NNABLA_UTILS_WITH_HDF5)
      message(FATAL_ERROR, "HDF5 cannot be enabled on macOS.")
      error() # Undefined command to abort cmake process here.
    endif()
  endif()
  if (ANDROID)
    include_directories(/usr/local/android/arm64/include)
    add_library(Protobuf STATIC IMPORTED)
    set_target_properties(Protobuf 
                          PROPERTIES IMPORTED_LOCATION 
                          /usr/local/android/arm64/lib/libprotobuf.a)
    
    add_library(LibArchive STATIC IMPORTED)
    set_target_properties(LibArchive 
                         PROPERTIES IMPORTED_LOCATION 
                         /usr/local/android/arm64/lib/libarchive.so) 
  else()
    find_package(Protobuf REQUIRED)
    find_package(LibArchive REQUIRED)
  endif()
  if (NNABLA_UTILS_WITH_HDF5)
    if (NNABLA_UTILS_STATIC_LINK_DEPS)
      set(HDF5_USE_STATIC_LIBRARIES 1)
    endif()
    findhdf5()
  endif()
  find_package(ZLIB REQUIRED)
  if (NNABLA_UTILS_STATIC_LINK_DEPS)
    if (NNABLA_UTILS_WITH_HDF5)
      set(HDF5_LIBRARIES ${HDF5_LIBRARIES} ${CMAKE_DL_LIBS})
    endif()
    if (${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
      string(REPLACE .dylib .a LibArchive_LIBRARIES ${LibArchive_LIBRARIES})
    else()
      string(REPLACE .so .a LibArchive_LIBRARIES ${LibArchive_LIBRARIES})
    endif()
  endif()

  include_directories(
    ${LibArchive_INCLUDE_DIR}
    ${PROTOBUF_INCLUDE_DIR}
    ${ZLIB_INCLUDE_DIRS}
    )
  if (NNABLA_UTILS_WITH_HDF5)
    include_directories(${HDF5_INCLUDE_DIRS})
  endif()

  if (ANDROID)
    target_link_libraries(${LIB_NAME}
      ${NBLA_LIBRARY_NAME}
      Protobuf
      LibArchive
      ${ZLIB_LIBRARIES}
      )
  else()
    target_link_libraries(${LIB_NAME}
      ${NBLA_LIBRARY_NAME}
      ${PROTOBUF_LIBRARY}
      ${LibArchive_LIBRARIES}
      ${ZLIB_LIBRARIES}
      )
  endif()
  if (NNABLA_UTILS_WITH_HDF5)
    target_link_libraries(${LIB_NAME}
      ${HDF5_LIBRARIES}
      ${HDF5_HL_LIBRARIES})
  endif()
endif()

set_property(TARGET ${LIB_NAME} PROPERTY CXX_STANDARD 11)


install(TARGETS ${LIB_NAME} LIBRARY DESTINATION lib)
install(DIRECTORY ${PROJECT_SOURCE_DIR}/include/nbla_utils DESTINATION include)
include(CPack)

if (BUILD_TEST)
  if (WIN32)
    set(BUILD_SHARED_LIBS ON CACHE BOOL
         "Force to build gtest with dll on windows.")
  endif()
  enable_testing()
  include_directories(${gtest_SOURCE_DIR}/include ${gtest_SOURCE_DIR})
  include_directories(${PROJECT_SOURCE_DIR}/src/nbla_utils/)
  include_directories(${PROJECT_SOURCE_DIR}/src/nbla_cli/)
  if (NNABLA_UTILS_WITH_HDF5)
    list(APPEND NBLA_UTILS_TEST_SOURCES
      test/test_expander.cpp
      test/test_hdf5_load.cpp
      test/test_load_save_parameters.cpp
      ${PROJECT_SOURCE_DIR}/src/nbla_cli/internal.cpp
    )
  endif()
  if (NNABLA_UTILS_WITH_NPY)
    if (NOT WIN32)
      # The reason not test it in WIN32:
      #   test_dataset tries to test internal classes
      #   which is not exported by NBLA_API, which causes
      #   link error.
      list(APPEND NBLA_UTILS_TEST_SOURCES
        test/test_dataset.cpp
      )
    endif()
  endif()
  list(APPEND NBLA_UTILS_TEST_SOURCES
    test/test_load_save_parameters.cpp
  )
  add_executable(test_nbla_utils ${NBLA_UTILS_TEST_SOURCES})
  add_dependencies(test_nbla_utils ${NBLA_LIBRARY_NAME})
  target_link_libraries(test_nbla_utils gtest gtest_main)
  target_link_libraries(test_nbla_utils ${NBLA_LIBRARY_NAME})
  target_link_libraries(test_nbla_utils ${LIB_NAME})
  set_property(TARGET test_nbla_utils PROPERTY CXX_STANDARD 11)
  nbla_exclude_from_all(test_nbla_utils)
  add_test(NAME test_nbla_utils COMMAND test_nbla_utils)
endif()

set(NBLA_UTILS_LIBRARY_NAME ${LIB_NAME} PARENT_SCOPE)
