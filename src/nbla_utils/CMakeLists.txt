option(NNABLA_UTILS_STATIC_LINK_DEPS "Static link C++ NNabla utils deps. Only available dist settigs." OFF)

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
  if (NNABLA_UTILS_STATIC_LINK_DEPS)
    set(HDF5_USE_STATIC_LIBRARIES 1)
  endif()
  find_package(HDF5 REQUIRED COMPONENTS C HL)
  find_package(LibArchive REQUIRED)
  find_package(ZLIB REQUIRED)
  if (NNABLA_UTILS_STATIC_LINK_DEPS)
    set(HDF5_LIBRARIES ${HDF5_LIBRARIES} ${CMAKE_DL_LIBS})
    string(REPLACE .so .a LibArchive_LIBRARIES ${LibArchive_LIBRARIES})
  endif()

  include_directories(
    ${LibArchive_INCLUDE_DIR}
    ${HDF5_INCLUDE_DIRS}
    ${PROTOBUF_INCLUDE_DIR}
    ${ZLIB_INCLUDE_DIRS}
    )

  target_link_libraries(nnabla_utils
    nnabla
    ${PROTOBUF_LIBRARY}
    ${HDF5_LIBRARIES}
    ${HDF5_HL_LIBRARIES}
    ${LibArchive_LIBRARIES}
    ${ZLIB_LIBRARIES}
    )

endif()

set_property(TARGET nnabla_utils PROPERTY CXX_STANDARD 11)


if (NOT WIN32)
  # Install
  install(TARGETS nnabla_utils
    LIBRARY DESTINATION lib)
  install(DIRECTORY ${PROJECT_SOURCE_DIR}/include/nbla_utils
    DESTINATION include)
endif()

set(NNABLA_UTILS_LIBRARY_NAME nnabla_utils PARENT_SCOPE)
