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

cmake_minimum_required (VERSION 3.1)
#set(CMAKE_VERBOSE_MAKEFILE ON)

# NNabla project
project (nbla C CXX)

###############################################################################
# General settings
###############################################################################
# Using cmake scripts and modules
list(APPEND CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/build-tools/cmake/Modules)
list(APPEND CMAKE_FIND_ROOT_PATH $ENV{CMAKE_FIND_ROOT_PATH})
include(${CMAKE_SOURCE_DIR}/build-tools/cmake/Utils.cmake)

###############################################################################
option(BUILD_CPP_LIB "Build C++ Library" ON)
option(BUILD_CPP_TEST "Build testing" OFF)
option(BUILD_CPP_UTILS "Build C++ API and utilities" OFF)
option(BUILD_PYTHON_PACKAGE "Build python package" ON)
option(MAKE_MANYLINUX_WHEEL "Convert wheel to manylinux version" OFF)

option(NBLA_VERBOSE_MEMORY_USAGE "Enable debug print of memory usage" OFF)
if(NBLA_VERBOSE_MEMORY_USAGE)
  add_definitions(-DNBLA_VERBOSE_MEMORY_USAGE)
endif()

###############################################################################
# Settings
###############################################################################
if (NOT PYTHON_COMMAND_NAME)
  if(NOT PYTHON_VERSION_MAJOR)
    set(PYTHON_COMMAND_NAME "python3")
  else()
    set(PYTHON_COMMAND_NAME "python${PYTHON_VERSION_MAJOR}.${PYTHON_VERSION_MINOR}")
  endif()
endif()

find_program(PP ${PYTHON_COMMAND_NAME})
if(NOT EXISTS ${PP})
  find_program(PP "python3")
  if(NOT EXISTS ${PP})
    message(FATAL_ERROR, "python [${PYTHON_COMMAND_NAME}] not found. Please install python or check your settings.")
    error() # Place Unknown command to stop cmake process here.
  endif()
endif()
set(PYTHON_COMMAND "${PP}")


if(NOT EXISTS ${PROTOC_COMMAND})
  find_program(PROTOC_COMMAND protoc)
endif()
if(NOT EXISTS ${PROTOC_COMMAND})
  # If find_program fails.
  message(FATAL_ERROR, "protoc not found. Please install protocol buffer compiler.")
  error() # Place Unknown command to stop cmake process here.
endif()

###############################################################################
# Preprocessing
###############################################################################
execute_process(
  COMMAND ${PYTHON_COMMAND}
          ${CMAKE_SOURCE_DIR}/build-tools/cmake/get_setup_build_dir.py
          OUTPUT_VARIABLE NBLA_SETUP_LIB_DIR_RAW)
string(STRIP ${NBLA_SETUP_LIB_DIR_RAW} NBLA_SETUP_LIB_DIR)

###############################################################################
# Code generator
###############################################################################

file(STRINGS ${CMAKE_SOURCE_DIR}/VERSION.txt versions NEWLINE_CONSUME)
string(REGEX REPLACE [[^([A-z0-9.\-_]+).*$]] [[\1]] version ${versions})
string(REGEX REPLACE [[^([0-9]+)\.([0-9]+)\.(.*)$]] [[\1]] version_major ${version})
string(REGEX REPLACE [[^([0-9]+)\.([0-9]+)\.(.*)$]] [[\2]] version_minor ${version})
string(REGEX REPLACE [[^([0-9]+)\.([0-9]+)\.(.*)$]] [[\3]] version_patch ${version})

set(CPACK_PACKAGE_NAME nnabla-cpplib${LIB_NAME_SUFFIX})

if(UNIX)
  find_program(LSB_RELEASE lsb_release)
  if(EXISTS ${LSB_RELEASE})
    execute_process(COMMAND ${LSB_RELEASE} -si
      OUTPUT_VARIABLE LSB_RELEASE_ID
      OUTPUT_STRIP_TRAILING_WHITESPACE
      )
    execute_process(COMMAND ${LSB_RELEASE} -sr
      OUTPUT_VARIABLE LSB_RELEASE_RELEASE
      OUTPUT_STRIP_TRAILING_WHITESPACE
      )
    set(CPACK_SYSTEM_NAME ${LSB_RELEASE_ID}${LSB_RELEASE_RELEASE})
  else()
    find_program(UNAME uname)
    if(EXISTS ${UNAME})
      execute_process(COMMAND ${UNAME} -s
        OUTPUT_VARIABLE UNAME_SYSTEM_NAME
        OUTPUT_STRIP_TRAILING_WHITESPACE
        )
      set(CPACK_SYSTEM_NAME ${UNAME_SYSTEM_NAME})
    endif()
  endif()
  find_program(UNAME uname)
  if(EXISTS ${UNAME})
    execute_process(COMMAND ${UNAME} -m
      OUTPUT_VARIABLE UNAME_MACHINE_NAME
      OUTPUT_STRIP_TRAILING_WHITESPACE
      )
    set(CPACK_SYSTEM_NAME ${CPACK_SYSTEM_NAME}_${UNAME_MACHINE_NAME})
  endif()
endif()

set(CPACK_PACKAGE_VERSION_MAJOR ${version_major})
set(CPACK_PACKAGE_VERSION_MINOR ${version_minor})
set(CPACK_PACKAGE_VERSION_PATCH ${version_patch})
set(CPACK_PACKAGE_CONTACT "nnabla@googlegroups.com")

file(MAKE_DIRECTORY  ${CMAKE_SOURCE_DIR}/src/nbla/proto)
execute_process(
  COMMAND
  ${PYTHON_COMMAND}
  ${CMAKE_SOURCE_DIR}/build-tools/code_generator/generate.py ${version}
  RESULT_VARIABLE GENERATE_RESULT)

if (NOT ${GENERATE_RESULT} EQUAL 0)
  message(FATAL_ERROR "generate.py failed.")
endif ()

if (${CMAKE_SOURCE_DIR}/src/nbla/proto/nnabla.proto
    IS_NEWER_THAN
    ${CMAKE_SOURCE_DIR}/python/src/nnabla/utils//nnabla_pb2.py)
  execute_process(
    COMMAND
    ${PROTOC_COMMAND}
    --proto_path=${CMAKE_SOURCE_DIR}/src/nbla/proto/
    --python_out=${CMAKE_SOURCE_DIR}/python/src/nnabla/utils
    ${CMAKE_SOURCE_DIR}/src/nbla/proto/nnabla.proto
    RESULT_VARIABLE PROTOC_RESULT)
  if (NOT ${PROTOC_RESULT} EQUAL 0)
    message(FATAL_ERROR "protoc for Python failed.")
  endif ()
endif()

if (${CMAKE_SOURCE_DIR}/src/nbla/proto/nnabla.proto
    IS_NEWER_THAN
    ${CMAKE_SOURCE_DIR}/src/nbla_utils/nnabla.pb.cc)
execute_process(
  COMMAND
  ${PROTOC_COMMAND}
  --proto_path=${CMAKE_SOURCE_DIR}/src/nbla/proto/
  --cpp_out=${CMAKE_SOURCE_DIR}/src/nbla_utils
  ${CMAKE_SOURCE_DIR}/src/nbla/proto/nnabla.proto
  RESULT_VARIABLE PROTOC_RESULT)
  if (NOT ${PROTOC_RESULT} EQUAL 0)
    message(FATAL_ERROR "protoc for C++ failed.")
  endif ()
endif()

###############################################################################
# C++ Library
###############################################################################
if(BUILD_CPP_LIB)
  ###############################################################################
  # Third party libraries
  ###############################################################################
  add_subdirectory(third_party)

  if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING
      "Build type release is default on single-configuration build system like GNU make."
      FORCE)
  endif()

  # Flags
  if(MSVC)
    # Force to always compile with W4
    if(CMAKE_CXX_FLAGS MATCHES "/W[0-4]")
      string(REGEX REPLACE "/W[0-4]" "/W0" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
    else()
      set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /W0")
    endif()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /bigobj")
  endif()

  if(UNIX OR APPLE)
    find_program(CCACHE_FOUND ccache)
    if(CCACHE_FOUND)
      set_property(GLOBAL PROPERTY RULE_LAUNCH_COMPILE ccache)
      set_property(GLOBAL PROPERTY RULE_LAUNCH_LINK ccache)
    endif(CCACHE_FOUND)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC -Wall")
  endif()
  nbla_warnings_disable(CMAKE_CXX_FLAGS /wd4099)

  # Setting output directory naively
  set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
  if (WIN32)
    set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
    set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
  else()
    set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
    set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
  endif()

  ###############################################################################
  # C++ core
  ###############################################################################
  # includes
  list(APPEND NBLA_INCLUDE_DIRS
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/third_party/eigen-3.3.5
    ${CMAKE_SOURCE_DIR}/include/third_party)

  if(BUILD_CPP_UTILS)
    list(APPEND NBLA_INCLUDE_DIRS
      ${CMAKE_SOURCE_DIR}/third_party/cmdline-master)
  endif()

  include_directories(${NBLA_INCLUDE_DIRS};${PROJECT_BINARY_DIR})
  add_subdirectory(src/nbla)

  ###############################################################################
  # C++ Utilities
  ###############################################################################
  if(BUILD_CPP_UTILS)
    add_subdirectory(src/nbla_utils)
    add_subdirectory(src/nbla_cli)
    add_subdirectory(examples/cpp)
  endif()

  ###############################################################################
  # C++ Testing
  ###############################################################################
  if(BUILD_TEST)
    if (WIN32)
      set(BUILD_SHARED_LIBS ON CACHE BOOL
        "Force to build gtest with dll on Windows.")
    endif()
    enable_testing()
    include_directories(${gtest_SOURCE_DIR}/include ${gtest_SOURCE_DIR} src/nbla/test/include)
    file(GLOB NBLA_TEST_SOURCES
            src/nbla/test/test_*.cpp
            src/nbla/test/parametric_functions/*.cpp)
    add_executable(cpplibtest ${NBLA_TEST_SOURCES})
    add_dependencies(cpplibtest ${NBLA_LIBRARY_NAME})
    target_link_libraries(cpplibtest gtest gtest_main)
    target_link_libraries(cpplibtest ${NBLA_LIBRARY_NAME})
    set_property(TARGET cpplibtest PROPERTY CXX_STANDARD 14)
    nbla_exclude_from_all(cpplibtest)  # Exclude target from all or default build
    add_test(NAME cpplibtest COMMAND cpplibtest)
  endif()

  ###############################################################################
  # Generate setup.cfg
  ###############################################################################
  if (NOT (DEFINED NBLA_PYTHON_BUILD_EXT_COMPILER))
    if (WIN32)
      set(NBLA_PYTHON_BUILD_EXT_COMPILER msvc)
    elseif(UNIX OR APPLE)
      set(NBLA_PYTHON_BUILD_EXT_COMPILER unix)
    else ()
      message(FATAL_ERROR
        "Python build_ext compiler inference is only supported on Win, Unix or Apple.")
    endif()
    message("Python build_ext compiler is inferred as '${NBLA_PYTHON_BUILD_EXT_COMPILER}'.")
    message("You can specify a compiler manually setting a variable"
      " NBLA_PYTHON_BUILD_EXT_COMPILER. You can see a list of supported"
      " compiler by `python setup.py build_ext --help-compiler`.")
  endif()

  option(CONFIG_BUILD_DIR ${CMAKE_BINARY_DIR})

  # Two-step configure_file generation with generate-time property (target_file_name).
  # http://cmake.3232098.n2.nabble.com/cmp0026-file-GENERATE-and-configure-file-td7590275.html
  # Since multiple-configuration build system like Visual Studio requires
  # file(GENERATE ...) for each build config, we generate setup.cfg files for each
  # build type with suffix `-<config>`. You can switch the build type by replacing
  # the setup.cfg file by setup.cfg-<config>. The default setup.cfg is for Release build.
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/python/setup.cfg.in
    ${CMAKE_CURRENT_SOURCE_DIR}/python/setup.cfg.in.configured)

  file(GENERATE OUTPUT ${CMAKE_CURRENT_SOURCE_DIR}/python/setup.cfg-$<CONFIG>
    INPUT ${CMAKE_CURRENT_SOURCE_DIR}/python/setup.cfg.in.configured)

  file(GENERATE OUTPUT ${CMAKE_CURRENT_SOURCE_DIR}/python/setup.cfg
    INPUT ${CMAKE_CURRENT_SOURCE_DIR}/python/setup.cfg.in.configured
    CONDITION $<CONFIG:${CMAKE_BUILD_TYPE}>)

  set_property(DIRECTORY APPEND PROPERTY ADDITIONAL_MAKE_CLEAN_FILES
    ${CMAKE_CURRENT_SOURCE_DIR}/python/setup.cfg
    ${CMAKE_CURRENT_SOURCE_DIR}/python/setup.cfg.in.configured
    ${CMAKE_CURRENT_SOURCE_DIR}/python/setup.cfg-$<CONFIG>)

endif() # BUILD_CPP_LIB

###############################################################################
# Python API
###############################################################################
if(BUILD_PYTHON_PACKAGE)

  if(NOT BUILD_CPP_LIB)
    if(NOT IS_DIRECTORY ${CPPLIB_BUILD_DIR})
      message(FATAL_ERROR "Invalid CPPLIB_BUILD_DIR :[${CPPLIB_BUILD_DIR}]")
    endif()
    if(NOT EXISTS ${CPPLIB_LIBRARY})
      if (${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
	string(REPLACE .so .dylib CPPLIB_LIBRARY ${CPPLIB_LIBRARY})
	if(NOT EXISTS ${CPPLIB_LIBRARY})
	  message(FATAL_ERROR "Invalid CPPLIB_LIBRARY :[${CPPLIB_LIBRARY}]")
	endif()
      else()
	message(FATAL_ERROR "Invalid CPPLIB_LIBRARY :[${CPPLIB_LIBRARY}]")
      endif()
    endif()

    set(NBLA_LIBRARY_NAME ${CPPLIB_LIBRARY})  # No one sets if !BUILD_CPP_LIB
    set(CPPLIB_LIBRARY_NAME nnabla${LIB_NAME_SUFFIX})
    set(CPPLIB_TARGET_FILE ${CPPLIB_LIBRARY})
    get_filename_component(CPPLIB_TARGET_FILE_NAME ${CPPLIB_TARGET_FILE} NAME)

    # TODO: Debug mode in windows
    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/python/setup.cfg.build-wheel.in
      ${CMAKE_CURRENT_SOURCE_DIR}/python/setup.cfg)
  endif()

  set_property(DIRECTORY APPEND PROPERTY ADDITIONAL_MAKE_CLEAN_FILES
    ${CMAKE_CURRENT_SOURCE_DIR}/python/test/solver/__pycache__
    ${CMAKE_CURRENT_SOURCE_DIR}/python/test/__pycache__
    ${CMAKE_CURRENT_SOURCE_DIR}/python/test/function/__pycache__
    ${CMAKE_CURRENT_SOURCE_DIR}/python/test/utils/__pycache__
    ${CMAKE_BINARY_DIR}/"build")
  # http://stackoverflow.com/questions/13298504/using-cmake-with-setup-py
  file(GLOB_RECURSE NBLA_PYTHON_DEPS
    ${CMAKE_CURRENT_SOURCE_DIR}/python/src/*.py
    ${CMAKE_CURRENT_SOURCE_DIR}/python/src/*.pyx
    ${CMAKE_CURRENT_SOURCE_DIR}/python/src/*.pxd,
    ${CMAKE_CURRENT_SOURCE_DIR}/python/src/utils/dlpack/*.py,
    ${CMAKE_CURRENT_SOURCE_DIR}/python/src/utils/dlpack/*.pyx,
    ${CMAKE_CURRENT_SOURCE_DIR}/python/src/utils/dlpack/*.pxd)
  file(GLOB_RECURSE NBLA_PYTHON_BUILD_OUTPUT_CYTHON
    ${CMAKE_CURRENT_SOURCE_DIR}/python/src/*.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/python/src/*.h,
    ${CMAKE_CURRENT_SOURCE_DIR}/python/src/utils/dlpack/*.cpp,
    ${CMAKE_CURRENT_SOURCE_DIR}/python/src/utils/dlpack/*.h)

  set(NBLA_PYTHON_SETUP ${CMAKE_CURRENT_SOURCE_DIR}/python/setup.py)
  set(NNABLA_CONVERTER_SETUP ${CMAKE_CURRENT_SOURCE_DIR}/python/src/nnabla/utils/converter/setup.py)
  set(NBLA_PYTHON_OUTPUT ${CMAKE_BINARY_DIR}/build)
  set(NBLA_PYTHON_OUTPUT_WHEEL ${NBLA_PYTHON_OUTPUT}/.timestamp.wheel)
  set(NBLA_WHEEL_DEPENDS ${NBLA_PYTHON_DEPS} ${NBLA_LIBRARY_NAME})
  # Wheel
  if(UNIX)
    if(MAKE_MANYLINUX_WHEEL)
      add_custom_command(OUTPUT ${NBLA_PYTHON_OUTPUT_WHEEL}
        COMMAND ${CMAKE_COMMAND} -E remove -f ${NBLA_PYTHON_OUTPUT_CYTHON}
        COMMAND ${CMAKE_COMMAND} -E remove -f ${CMAKE_BINARY_DIR}/dist/*.whl
        COMMAND ${CMAKE_COMMAND} -E env
                WHEEL_SUFFIX=${WHEEL_SUFFIX}
                ${PYTHON_COMMAND} ${NBLA_PYTHON_SETUP} -q bdist_wheel
        COMMAND ${CMAKE_COMMAND} -E env
                WHEEL_SUFFIX=${WHEEL_SUFFIX}
                ${PYTHON_COMMAND} ${NNABLA_CONVERTER_SETUP} -q bdist_wheel
        COMMAND ${CMAKE_SOURCE_DIR}/build-tools/auditwheel-nnabla ${CMAKE_BINARY_DIR}/dist/*-linux_*.whl
        COMMAND ${CMAKE_COMMAND} -E remove -f ${CMAKE_BINARY_DIR}/dist/*-linux*.whl
        COMMAND ${CMAKE_COMMAND} -E touch ${NBLA_PYTHON_OUTPUT_WHEEL}
        DEPENDS ${NBLA_WHEEL_DEPENDS})
    else()
      add_custom_command(OUTPUT ${NBLA_PYTHON_OUTPUT_WHEEL}
        COMMAND ${CMAKE_COMMAND} -E remove -f ${NBLA_PYTHON_OUTPUT_CYTHON}
        COMMAND ${CMAKE_COMMAND} -E remove -f ${CMAKE_BINARY_DIR}/dist/*.whl
        COMMAND ${CMAKE_COMMAND} -E env
                WHEEL_SUFFIX=${WHEEL_SUFFIX}
                ${PYTHON_COMMAND} ${NBLA_PYTHON_SETUP} -q bdist_wheel
        COMMAND ${CMAKE_COMMAND} -E env
                WHEEL_SUFFIX=${WHEEL_SUFFIX}
                ${PYTHON_COMMAND} ${NNABLA_CONVERTER_SETUP} -q bdist_wheel
        COMMAND ${CMAKE_COMMAND} -E touch ${NBLA_PYTHON_OUTPUT_WHEEL}
        DEPENDS ${NBLA_WHEEL_DEPENDS})
    endif()
  else()
    add_custom_command(OUTPUT ${NBLA_PYTHON_OUTPUT_WHEEL}
      COMMAND ${CMAKE_COMMAND} -E remove -f ${NBLA_PYTHON_OUTPUT_CYTHON}
      COMMAND ${CMAKE_COMMAND} -E remove -f ${CMAKE_BINARY_DIR}/dist/*.whl
      COMMAND ${PYTHON_COMMAND} ${NBLA_PYTHON_SETUP} -q bdist_wheel
      COMMAND ${PYTHON_COMMAND} ${NNABLA_CONVERTER_SETUP} -q bdist_wheel
      COMMAND ${CMAKE_COMMAND} -E touch ${NBLA_PYTHON_OUTPUT_WHEEL}
      DEPENDS ${NBLA_WHEEL_DEPENDS})
  endif()
  if(BUILD_CPP_LIB)
    add_custom_target(wheel ALL DEPENDS ${NBLA_PYTHON_OUTPUT_WHEEL})
  else()
    add_custom_target(wheel DEPENDS ${NBLA_PYTHON_OUTPUT_WHEEL})
  endif()

  if(UNIX OR APPLE)
    add_custom_target(benchmark
      COMMAND cd ${CMAKE_CURRENT_SOURCE_DIR}/python/benchmark &&
      LD_LIBRARY_PATH=${CMAKE_BINARY_DIR}/lib:$ENV{LD_LIBRARY_PATH}
      PYTHONPATH=${CMAKE_BINARY_DIR}/build/${NBLA_SETUP_LIB_DIR}:$ENV{PYTHONPATH}
      bash -c "py.test -x")
    add_dependencies(benchmark wheel)
  endif()

  #############################################################################
  # Document
  #############################################################################
  if(UNIX OR APPLE)
    if(IS_DIRECTORY ${NNABLA_EXT_CUDA_DIRECTORY})
      if(NOT IS_ABSOLUTE ${NNABLA_EXT_CUDA_DIRECTORY})
        get_filename_component(NNABLA_EXT_CUDA_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/${NNABLA_EXT_CUDA_DIRECTORY} ABSOLUTE)
      endif()
      add_custom_target(doc
        COMMAND rm -rf ${CMAKE_BINARY_DIR}/doc
        COMMAND mkdir -p ${CMAKE_BINARY_DIR}/doc/doxygen
        COMMAND cd ${CMAKE_SOURCE_DIR} &&
                cat build-tools/doxygen/config >Doxyfile &&
                echo OUTPUT_DIRECTORY  = ${CMAKE_BINARY_DIR}/doc/doxygen >>Doxyfile &&
                doxygen && rm -f Doxyfile
        COMMAND mv ${CMAKE_BINARY_DIR}/doc/doxygen/html ${CMAKE_BINARY_DIR}/doc/html-Cpp
        COMMAND mv ${CMAKE_BINARY_DIR}/doc/doxygen/xml ${CMAKE_BINARY_DIR}/doc/xml-Cpp
        COMMAND rm -rf ${CMAKE_BINARY_DIR}/doc/doxygen
        #ext-cuda
        COMMAND mkdir -p ${CMAKE_BINARY_DIR}/doc/doxygen
        COMMAND cd ${NNABLA_EXT_CUDA_DIRECTORY} &&
                cat ${CMAKE_SOURCE_DIR}/build-tools/doxygen/config_ext_cuda >Doxyfile &&
                echo OUTPUT_DIRECTORY  = ${CMAKE_BINARY_DIR}/doc/doxygen >>Doxyfile &&
                doxygen && rm -f Doxyfile
        COMMAND mv ${CMAKE_BINARY_DIR}/doc/doxygen/html ${CMAKE_BINARY_DIR}/doc/html-Ext-Cuda-Cpp
        COMMAND mv ${CMAKE_BINARY_DIR}/doc/doxygen/xml ${CMAKE_BINARY_DIR}/doc/xml-Ext-Cuda-Cpp
        COMMAND rm -rf ${CMAKE_BINARY_DIR}/doc/doxygen
        # Generate rst
        COMMAND rm -rf ${CMAKE_SOURCE_DIR}/doc/cpp/Cpp ${CMAKE_SOURCE_DIR}/doc/cpp/Ext-Cuda-Cpp
        COMMAND cd ${CMAKE_SOURCE_DIR} &&
                python doc/dox_to_rst.py 1
        # English
        COMMAND mkdir -p ${CMAKE_BINARY_DIR}/doc/sphinx
        COMMAND cd ${CMAKE_SOURCE_DIR} &&
                PYTHONPATH=${CMAKE_BINARY_DIR}/build/${NBLA_SETUP_LIB_DIR}:$ENV{PYTHONPATH}
                sphinx-build -b html doc ${CMAKE_BINARY_DIR}/doc/sphinx -c doc
        COMMAND mv ${CMAKE_BINARY_DIR}/doc/sphinx ${CMAKE_BINARY_DIR}/doc/html
        # Japanese
        COMMAND mkdir -p ${CMAKE_BINARY_DIR}/doc/sphinx
        COMMAND cd ${CMAKE_SOURCE_DIR} &&
                PYTHONPATH=${CMAKE_BINARY_DIR}/build/${NBLA_SETUP_LIB_DIR}:$ENV{PYTHONPATH}
                sphinx-build -b html -D language=ja doc ${CMAKE_BINARY_DIR}/doc/sphinx -c doc
        COMMAND mv ${CMAKE_BINARY_DIR}/doc/sphinx ${CMAKE_BINARY_DIR}/doc/html-ja
        )
    else()
      add_custom_target(doc
        COMMAND rm -rf ${CMAKE_BINARY_DIR}/doc
        COMMAND mkdir -p ${CMAKE_BINARY_DIR}/doc/doxygen
        COMMAND cd ${CMAKE_SOURCE_DIR} &&
                cat build-tools/doxygen/config >Doxyfile &&
                echo OUTPUT_DIRECTORY  = ${CMAKE_BINARY_DIR}/doc/doxygen >>Doxyfile &&
                doxygen && rm -f Doxyfile
        COMMAND mv ${CMAKE_BINARY_DIR}/doc/doxygen/html ${CMAKE_BINARY_DIR}/doc/html-Cpp
        COMMAND mv ${CMAKE_BINARY_DIR}/doc/doxygen/xml ${CMAKE_BINARY_DIR}/doc/xml-Cpp
        COMMAND rm -rf ${CMAKE_BINARY_DIR}/doc/doxygen
        # Generate rst
        COMMAND rm -rf ${CMAKE_SOURCE_DIR}/doc/cpp/Cpp ${CMAKE_SOURCE_DIR}/doc/cpp/Ext-Cuda-Cpp
        COMMAND cd ${CMAKE_SOURCE_DIR} &&
                python doc/dox_to_rst.py 0
        # English
        COMMAND mkdir -p ${CMAKE_BINARY_DIR}/doc/sphinx
        COMMAND cd ${CMAKE_SOURCE_DIR} &&
                PYTHONPATH=${CMAKE_BINARY_DIR}/build/${NBLA_SETUP_LIB_DIR}:$ENV{PYTHONPATH}
                sphinx-build -b html doc ${CMAKE_BINARY_DIR}/doc/sphinx -c doc
        COMMAND mv ${CMAKE_BINARY_DIR}/doc/sphinx ${CMAKE_BINARY_DIR}/doc/html
        # Japanese
        COMMAND mkdir -p ${CMAKE_BINARY_DIR}/doc/sphinx
        COMMAND cd ${CMAKE_SOURCE_DIR} &&
                PYTHONPATH=${CMAKE_BINARY_DIR}/build/${NBLA_SETUP_LIB_DIR}:$ENV{PYTHONPATH}
                sphinx-build -b html -D language=ja doc ${CMAKE_BINARY_DIR}/doc/sphinx -c doc
        COMMAND mv ${CMAKE_BINARY_DIR}/doc/sphinx ${CMAKE_BINARY_DIR}/doc/html-ja
        )
    endif()
    add_dependencies(doc wheel)
    nbla_exclude_from_all(doc)
  endif()

endif() # BUILD_PYTHON_PACKAGE
