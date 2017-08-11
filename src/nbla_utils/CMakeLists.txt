include_directories(${CMAKE_BINARY_DIR}/libraries/include)

add_library(nbla_utils SHARED 
  nnp.cpp
  nnabla.pb.cc
  protobuf_internal.cpp
  protobuf_internal_create_function.cpp
  )

if (WIN32)
  target_link_libraries(nbla_utils
    nnabla
    ${CMAKE_BINARY_DIR}/libraries/lib/libprotobuf-lite.lib
    #    ${CMAKE_BINARY_DIR}/libraries/lib/libprotobuf.lib
    #    ${CMAKE_BINARY_DIR}/libraries/lib/libprotoc.lib
    ${CMAKE_BINARY_DIR}/libraries/lib/libhdf5.lib
    ${CMAKE_BINARY_DIR}/libraries/lib/libhdf5_hl.lib
    ${CMAKE_BINARY_DIR}/libraries/lib/zlib.lib)
else()

  include(FindProtobuf)
  find_package(Protobuf REQUIRED)
  include_directories(${PROTOBUF_INCLUDE_DIR})

  set(HDF5_USE_STATIC_LIBRARIES ON)
  include(FindHDF5)
  find_package(HDF5)
  #find_package(HDF5 REQUIRED)

  include(FindLibArchive)
  find_package(LibArchive REQUIRED)
  include_directories(${LibArchive_INCLUDE_DIR})

  target_link_libraries(nbla_utils
    nnabla
    ${PROTOBUF_LIBRARY}
    ${HDF5_LIBRARY}
    ${LibArchive_LIBRARIES}
    -lhdf5-static
    -lhdf5_hl-static
    -ldl -lz)

endif()

set_property(TARGET nbla_utils PROPERTY CXX_STANDARD 11)
set(NBLA_UTILS_LIBRARY_NAME nnabla_utils PARENT_SCOPE)
