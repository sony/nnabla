# Copyright 2017,2018,2019,2020,2021 Sony Corporation.
# Copyright 2021 Sony Group Corporation.
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

set(LIB_NAME nnabla${LIB_NAME_SUFFIX})

file(GLOB SOURCES RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}
  ./*.cpp
  memory/*.cpp
  array/*.cpp
  solver/*.cpp
  communicator/*.cpp
  utils/*.cpp
  function/*.cpp
  function/utils/*.cpp
  function/nontemplate/*.cpp
  computation_graph/*.cpp
  lms/*.cpp)

# add version info into DLL
if (WIN32)
  configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/version.rc.in
    ${CMAKE_CURRENT_BINARY_DIR}/version-utf8.rc
  )

  execute_process(
          COMMAND ${PYTHON_COMMAND}
          ${CMAKE_SOURCE_DIR}/build-tools/cmake/utf8_to_utf16.py
          ${CMAKE_CURRENT_BINARY_DIR}/version-utf8.rc ${CMAKE_CURRENT_BINARY_DIR}/version.rc)

  list(APPEND SOURCES ${CMAKE_CURRENT_BINARY_DIR}/version.rc)
endif()

add_library(${LIB_NAME} SHARED ${SOURCES})

target_link_libraries(${LIB_NAME} ${NBLA_LINKER_LIBS})
set_property(TARGET ${LIB_NAME} PROPERTY CXX_STANDARD 14)

install(TARGETS ${LIB_NAME} LIBRARY DESTINATION lib)
install(DIRECTORY ${PROJECT_SOURCE_DIR}/include/nbla DESTINATION include)
install(FILES ${PROJECT_SOURCE_DIR}/LICENSE ${PROJECT_SOURCE_DIR}/NOTICE.md
  DESTINATION doc)
install(FILES ${PROJECT_SOURCE_DIR}/third_party/LICENSES.md DESTINATION doc/third_party)
include(CPack)

set(NBLA_LIBRARY_NAME ${LIB_NAME} PARENT_SCOPE)
