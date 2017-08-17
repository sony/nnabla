include_directories(${CMAKE_BINARY_DIR}/libraries/include)

add_library(nnabla_utils SHARED
  nnp.cpp
  nnabla.pb.cc
  nnp_impl.cpp
  nnp_impl_create_function.cpp
  )

if (WIN32)
  target_link_libraries(nnabla_utils
    nnabla
    ${CMAKE_BINARY_DIR}/libraries/lib/libprotobuf-lite.lib
    #    ${CMAKE_BINARY_DIR}/libraries/lib/libprotobuf.lib
    #    ${CMAKE_BINARY_DIR}/libraries/lib/libprotoc.lib
    ${CMAKE_BINARY_DIR}/libraries/lib/libhdf5.lib
    ${CMAKE_BINARY_DIR}/libraries/lib/libhdf5_hl.lib
    ${CMAKE_BINARY_DIR}/libraries/lib/zlib.lib)
else()

  find_package(Protobuf REQUIRED)
  find_package(HDF5 REQUIRED COMPONENTS C HL)
  find_package(LibArchive REQUIRED)

  include_directories(
    ${LibArchive_INCLUDE_DIR}
    ${HDF5_INCLUDE_DIRS}
    ${PROTOBUF_INCLUDE_DIR}
    )

  target_link_libraries(nnabla_utils
    nnabla
    ${PROTOBUF_LIBRARY}
    ${HDF5_LIBRARIES}
    ${HDF5_HL_LIBRARIES}
    ${LibArchive_LIBRARIES}
    )

endif()

set_property(TARGET nnabla_utils PROPERTY CXX_STANDARD 11)
set(NNABLA_UTILS_LIBRARY_NAME nnabla_utils PARENT_SCOPE)
