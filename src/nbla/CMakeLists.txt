file(GLOB SOURCES RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}
  ./*.cpp
  memory/*.cpp
  array/*.cpp
  solver/*.cpp
  communicator/*.cpp
  utils/*.cpp
  function/*.cpp
  function/nontemplate/*.cpp
  computation_graph/*.cpp)

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

add_library(nnabla SHARED ${SOURCES})

target_link_libraries(nnabla ${NBLA_LINKER_LIBS})
set_property(TARGET nnabla PROPERTY CXX_STANDARD 11)

if (NOT WIN32)
  # Install library nnabla and include files.
  install(TARGETS nnabla
    LIBRARY DESTINATION lib
    )
  install(DIRECTORY ${PROJECT_SOURCE_DIR}/include/nbla
    DESTINATION include
    )
endif()

set(NBLA_LIBRARY_NAME nnabla PARENT_SCOPE)
