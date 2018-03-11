file(GLOB SOURCES RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} 
  ./*.cpp
  array/*.cpp
  solver/*.cpp
  communicator/*.cpp
  utils/*.cpp
  function/*.cpp
  function/nontemplate/*.cpp
  computation_graph/*.cpp)

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
