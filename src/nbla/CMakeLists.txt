
file(GLOB SOURCES RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} 
  ./*.cpp
  array/*.cpp
  solver/*.cpp
  function/*.cpp
  utils/*.cpp
  computation_graph/*.cpp)

add_library(nnabla SHARED ${SOURCES})
  
target_link_libraries(nnabla ${NBLA_LINKER_LIBS})
set_property(TARGET nnabla PROPERTY CXX_STANDARD 11)
set(NBLA_LIBRARY_NAME nnabla PARENT_SCOPE)
