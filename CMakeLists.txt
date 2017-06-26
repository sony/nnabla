cmake_minimum_required (VERSION 3.1)

# NNabla project
project (nbla C CXX)

###############################################################################
# Settings
###############################################################################
if (NOT PYTHON_COMMAND_NAME)
  set(PYTHON_COMMAND_NAME "python")
endif()

find_program(PYTHON_COMMAND ${PYTHON_COMMAND_NAME})

find_program(PROTOC_COMMAND protoc)
if(NOT EXISTS ${PROTOC_COMMAND})
  message(FATAL_ERROR, "protoc not found. Please install protocol buffer compiler.")
  error() # Place Unknown command to stop cmake process here.
endif()

###############################################################################
# Preprocessing
###############################################################################
# Download third_party libraries
if(NOT IS_DIRECTORY ${CMAKE_SOURCE_DIR}/third_party/googletest-release-1.7.0)
  message("Downloading googletest")
  file(DOWNLOAD
    https://github.com/google/googletest/archive/release-1.7.0.zip
    ${CMAKE_SOURCE_DIR}/third_party/googletest-release-1.7.0.zip
    SHOW_PROGRESS
    STATUS status
    LOG log)
  list(GET status 0 status_code)
  list(GET status 1 status_string)
  if(NOT status_code EQUAL 0)
    message(FATAL_ERROR "Failed to download googletest. Please check network connection and proxy settings.
    status_code: ${status_code}
    status_string: ${status_string}
    log: ${log}
    ")
  endif()
  execute_process(
    COMMAND ${CMAKE_COMMAND} -E
    chdir ${CMAKE_SOURCE_DIR}/third_party
    ${CMAKE_COMMAND} -E tar xf googletest-release-1.7.0.zip)
endif()

if(NOT IS_DIRECTORY ${CMAKE_SOURCE_DIR}/third_party/eigen-eigen-de14376a69f8)
  message("Downloading eigen")
  file(DOWNLOAD
    https://bitbucket.org/eigen/eigen/get/de14376a69f8.zip
    ${CMAKE_SOURCE_DIR}/third_party/eigen-eigen-de14376a69f8.zip
    SHOW_PROGRESS
    STATUS status
    LOG log)
  list(GET status 0 status_code)
  list(GET status 1 status_string)
  if(NOT status_code EQUAL 0)
    message(FATAL_ERROR "Failed to download eigen. Please check network connection and proxy settings.
    status_code: ${status_code}
    status_string: ${status_string}
    log: ${log}
    ")
  endif()
  execute_process(
    COMMAND ${CMAKE_COMMAND} -E
    chdir ${CMAKE_SOURCE_DIR}/third_party
    ${CMAKE_COMMAND} -E tar xf eigen-eigen-de14376a69f8.zip)
endif()

if(NOT IS_DIRECTORY ${CMAKE_SOURCE_DIR}/third_party/spdlog-0.13.0)
  message("Downloading spdlog")
  file(DOWNLOAD
    https://github.com/gabime/spdlog/archive/v0.13.0.zip
    ${CMAKE_SOURCE_DIR}/third_party/spdlog-0.13.0.zip
    SHOW_PROGRESS
    STATUS status
    LOG log)
  list(GET status 0 status_code)
  list(GET status 1 status_string)
  if(NOT status_code EQUAL 0)
    message(FATAL_ERROR "Failed to download spdlog. Please check network connection and proxy settings.
    status_code: ${status_code}
    status_string: ${status_string}
    log: ${log}
    ")
  endif()
  execute_process(
    COMMAND ${CMAKE_COMMAND} -E
    chdir ${CMAKE_SOURCE_DIR}/third_party
    ${CMAKE_COMMAND} -E tar xf spdlog-0.13.0.zip)
endif()

execute_process(
  COMMAND ${PYTHON_COMMAND}
          ${CMAKE_SOURCE_DIR}/build-tools/cmake/get_setup_build_dir.py
          OUTPUT_VARIABLE NBLA_SETUP_LIB_DIR_RAW)
string(STRIP ${NBLA_SETUP_LIB_DIR_RAW} NBLA_SETUP_LIB_DIR)

###############################################################################
# Code generator
###############################################################################
file(MAKE_DIRECTORY  ${CMAKE_SOURCE_DIR}/src/nbla/proto)

execute_process(
  COMMAND ${PYTHON_COMMAND}
        ${CMAKE_CURRENT_SOURCE_DIR}/build-tools/code_generator/generate.py)

execute_process(
  COMMAND ${PROTOC_COMMAND} --proto_path=${CMAKE_SOURCE_DIR}/src/nbla/proto/
                 --python_out=${CMAKE_SOURCE_DIR}/python/src/nnabla/utils
                 ${CMAKE_SOURCE_DIR}/src/nbla/proto/nnabla.proto)
          

###############################################################################
option(BUILD_TEST "Build testing" OFF)

if (NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "Release" CACHE STRING
    "Build type release is default on single-configuration build system like GnuMake."
    FORCE)
endif()

###############################################################################
# General settings
###############################################################################
# Using cmake scripts and modules
list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/build-tools/cmake/Modules)
include(${PROJECT_SOURCE_DIR}/build-tools/cmake/Utils.cmake)

# Flags
if(MSVC)
  # Force to always compile with W4
  if(CMAKE_CXX_FLAGS MATCHES "/W[0-4]")
    string(REGEX REPLACE "/W[0-4]" "/W0" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
  else()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /W0")
  endif()
endif()

if(UNIX OR APPLE)
  find_program(CCACHE_FOUND ccache)
  if(CCACHE_FOUND)
    set_property(GLOBAL PROPERTY RULE_LAUNCH_COMPILE ccache)
    set_property(GLOBAL PROPERTY RULE_LAUNCH_LINK ccache)
  endif(CCACHE_FOUND)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC -Wall -std=gnu++0x")
endif()
nbla_warnings_disable(CMAKE_CXX_FLAGS -Wno-sign-compare -Wno-uninitialized /wd4099)

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
  ${PROJECT_SOURCE_DIR}/include
  ${PROJECT_SOURCE_DIR}/third_party/eigen-eigen-de14376a69f8
  ${PROJECT_SOURCE_DIR}/third_party/spdlog-0.13.0/include)
include_directories(${NBLA_INCLUDE_DIRS};${PROJECT_BINARY_DIR})
add_subdirectory(src/nbla)

###############################################################################
# C++ Testing
###############################################################################
if(BUILD_TEST)
  if (WIN32)
    set(BUILD_SHARED_LIBS ON CACHE BOOL
      "Force to build gtest with dll on Windows.")
  endif()
  add_subdirectory(third_party/googletest-release-1.7.0)
  enable_testing()
  include_directories(${gtest_SOURCE_DIR}/include ${gtest_SOURCE_DIR})
  file(GLOB NBLA_TEST_SOURCES src/nbla/test/test_*.cpp)
  add_executable(clibtest ${NBLA_TEST_SOURCES})
  add_dependencies(clibtest ${NBLA_LIBRARY_NAME})
  target_link_libraries(clibtest gtest gtest_main)
  target_link_libraries(clibtest ${NBLA_LIBRARY_NAME})
  set_property(TARGET clibtest PROPERTY CXX_STANDARD 11)
  nbla_exclude_from_all(clibtest)  # Exclude target from all or default build
  add_test(NAME clibtest COMMAND clibtest)
endif()

###############################################################################
# Python API
###############################################################################
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
  ${CMAKE_CURRENT_SOURCE_DIR}/python/src/*.pxd)
file(GLOB_RECURSE NBLA_PYTHON_BUILD_OUTPUT_CYTHON
  ${CMAKE_CURRENT_SOURCE_DIR}/python/src/*.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/python/src/*.h)

set(NBLA_PYTHON_SETUP ${CMAKE_CURRENT_SOURCE_DIR}/python/setup.py)
set(NBLA_PYTHON_OUTPUT ${CMAKE_BINARY_DIR}/build)
set(NBLA_PYTHON_OUTPUT_BUILD ${NBLA_PYTHON_OUTPUT}/.timestamp.build)
set(NBLA_PYTHON_OUTPUT_BDIST ${NBLA_PYTHON_OUTPUT}/.timestamp.bdist)

# Generate setup.cfg
if (NOT (DEFINED NBLA_PYTHON_BUILD_EXT_COMPILER))
  if (WIN32)
    set(NBLA_PYTHON_BUILD_EXT_COMPILER msvc)
  elseif(UNIX OR APPLE)
    set(NBLA_PYTHON_BUILD_EXT_COMPILER unix)
  else ()
    message(FATAL_ERROR
      "Python build_ext compiler inference is only supported on Win, Unix or Apple.")
  endif()
  message("Python build_ext compiler is infered as '${NBLA_PYTHON_BUILD_EXT_COMPILER}'.")
  message("You can specify a compiler manually setting a variable"
    " NBLA_PYTHON_BUILD_EXT_COMPILER. You can see a list of supported"
    " compiler by `python setup.py build_ext --help-compiler`.")
endif()
set(NBLA_PYTHON_BUILD_DIR ${CMAKE_BINARY_DIR})

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
  CONDITION $<CONFIG:Release>)

set_property(DIRECTORY APPEND PROPERTY ADDITIONAL_MAKE_CLEAN_FILES
  ${CMAKE_CURRENT_SOURCE_DIR}/python/setup.cfg
  ${CMAKE_CURRENT_SOURCE_DIR}/python/setup.cfg.in.configured
  ${CMAKE_CURRENT_SOURCE_DIR}/python/setup.cfg-$<CONFIG>)

# Build
add_custom_command(OUTPUT ${NBLA_PYTHON_OUTPUT_BUILD}
  COMMAND ${CMAKE_COMMAND} -E remove -f ${NBLA_PYTHON_OUTPUT_CYTHON}
  COMMAND ${PYTHON_COMMAND} ${NBLA_PYTHON_SETUP} build
  COMMAND ${CMAKE_COMMAND} -E touch ${NBLA_PYTHON_OUTPUT_BUILD}

  DEPENDS ${NBLA_PYTHON_DEPS} ${NBLA_LIBRARY_NAME}
  ${NBLA_PYTHON_SETUP})

add_custom_target(python ALL DEPENDS ${NBLA_PYTHON_OUTPUT_BUILD})
add_dependencies(python ${NBLA_LIBRARY_NAME})

# Bdist
add_custom_command(OUTPUT ${NBLA_PYTHON_OUTPUT_BDIST}
  COMMAND ${PYTHON_COMMAND} ${NBLA_PYTHON_SETUP} bdist_wheel
  COMMAND ${CMAKE_COMMAND} -E touch ${NBLA_PYTHON_OUTPUT_BDIST}
  DEPENDS ${NBLA_PYTHON_DEPS} ${NBLA_PYTHON_OUTPUT_BUILD})
add_custom_target(python-bdist ALL
  DEPENDS ${NBLA_PYTHON_OUTPUT_BDIST} ${NBLA_PYTHON_OUTPUT_BUILD})
add_dependencies(python-bdist python)


#############################################################################
# Python Test
#############################################################################
set(NBLA_PYTEST_OUTPUT ${NBLA_PYTHON_OUTPUT}/.timestamp.pytest)
file(GLOB_RECURSE NBLA_PYTEST_DEPS
  ${CMAKE_CURRENT_SOURCE_DIR}/python/test/*.py)
set(NBLA_PYTEST_DEPS ${NBLA_PYTEST_DEPS} ${NBLA_PYTHON_OUTPUT_BDIST})
if (WIN32)
  add_custom_command(OUTPUT ${NBLA_PYTEST_OUTPUT}
    COMMAND set PYTHONPATH=${CMAKE_BINARY_DIR}/Release\;${CMAKE_BINARY_DIR}/build/${NBLA_SETUP_LIB_DIR}
    COMMAND set PATH=%PATH%\;${CMAKE_BINARY_DIR}/bin/Release
    COMMAND cd ${CMAKE_CURRENT_SOURCE_DIR}/python/test
    COMMAND ${PYTHON_COMMAND} -m pytest
    COMMAND ${CMAKE_COMMAND} -E touch ${NBLA_PYTEST_OUTPUT}
    DEPENDS ${NBLA_PYTEST_DEPS})
else()
  add_custom_command(OUTPUT ${NBLA_PYTEST_OUTPUT}
    COMMAND cd ${CMAKE_CURRENT_SOURCE_DIR}/python/test &&
            LD_LIBRARY_PATH=${CMAKE_BINARY_DIR}/build/${NBLA_SETUP_LIB_DIR}/nnabla:$ENV{LD_LIBRARY_PATH}
            PYTHONPATH=${CMAKE_BINARY_DIR}/build/${NBLA_SETUP_LIB_DIR}:$ENV{PYTHONPATH}
            ${PYTHON_COMMAND} -m pytest
    COMMAND ${CMAKE_COMMAND} -E touch ${NBLA_PYTEST_OUTPUT}
    DEPENDS ${NBLA_PYTEST_DEPS})
endif()

add_custom_target(pytest DEPENDS ${NBLA_PYTEST_OUTPUT})
add_dependencies(pytest python-bdist)
add_test(NAME pytest COMMAND make pytest)

nbla_exclude_from_all(pytest)

if(UNIX OR APPLE)
  add_custom_target(benchmark
    COMMAND cd ${CMAKE_CURRENT_SOURCE_DIR}/python/benchmark &&
    LD_LIBRARY_PATH=${CMAKE_BINARY_DIR}/lib:$ENV{LD_LIBRARY_PATH}
    PYTHONPATH=${CMAKE_BINARY_DIR}/build/${NBLA_SETUP_LIB_DIR}:$ENV{PYTHONPATH}
    bash -c "py.test -x")
  add_dependencies(benchmark python-bdist)
endif()

if(UNIX OR APPLE)

  add_custom_target(doc ALL
    COMMAND rm -rf ${CMAKE_BINARY_DIR}/doc
    COMMAND mkdir -p ${CMAKE_BINARY_DIR}/doc/doxygen
    COMMAND cd ${CMAKE_SOURCE_DIR} &&
            cat build-tools/doxygen/config >Doxyfile &&
            echo OUTPUT_DIRECTORY  = ${CMAKE_BINARY_DIR}/doc/doxygen >>Doxyfile &&
            doxygen && rm -f Doxyfile
    COMMAND mkdir -p ${CMAKE_BINARY_DIR}/doc/sphinx
    COMMAND cd ${CMAKE_SOURCE_DIR} &&
            PYTHONPATH=${CMAKE_BINARY_DIR}/build/${NBLA_SETUP_LIB_DIR}:$ENV{PYTHONPATH}
            sphinx-build -M html doc ${CMAKE_BINARY_DIR}/doc/sphinx -c build-tools/sphinx
    COMMAND mkdir -p ${CMAKE_BINARY_DIR}/doc/protoc
    COMMAND cd ${CMAKE_SOURCE_DIR} &&
            ${PROTOC_COMMAND} --doc_out=html,index.html:${CMAKE_BINARY_DIR}/doc/protoc
            src/nbla/proto/nnabla.proto
    COMMAND mv ${CMAKE_BINARY_DIR}/doc/sphinx/doctrees ${CMAKE_BINARY_DIR}/doc
    COMMAND mv ${CMAKE_BINARY_DIR}/doc/sphinx/html ${CMAKE_BINARY_DIR}/doc
    COMMAND rm -rf ${CMAKE_BINARY_DIR}/doc/sphinx
    COMMAND mv ${CMAKE_BINARY_DIR}/doc/doxygen/html ${CMAKE_BINARY_DIR}/doc/html/doxygen
    COMMAND rm -rf ${CMAKE_BINARY_DIR}/doc/doxygen
    COMMAND mv ${CMAKE_BINARY_DIR}/doc/protoc ${CMAKE_BINARY_DIR}/doc/html/

            )
  add_dependencies(doc python)
  nbla_exclude_from_all(doc)

endif()
