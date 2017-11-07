option(NNABLA_UTILS_WITH_HDF5 "Enable HDF5 in C++ utils." OFF)
option(NNABLA_UTILS_STATIC_LINK_DEPS "Static link C++ NNabla utils deps." OFF)

if (NNABLA_UTILS_WITH_HDF5)
  add_definitions(-DNBLA_UTILS_WITH_HDF5)
endif()

include_directories(${CMAKE_BINARY_DIR}/libraries/include)

add_library(nnabla_utils SHARED
  nnp.cpp
  nnabla.pb.cc
  nnp_impl.cpp
  nnp_impl_create_function.cpp
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
  target_link_libraries(nnabla_utils
    nnabla
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
  find_package(Protobuf REQUIRED)
  if (NNABLA_UTILS_WITH_HDF5)
    if (NNABLA_UTILS_STATIC_LINK_DEPS)
      set(HDF5_USE_STATIC_LIBRARIES 1)
    endif()
    find_package(HDF5 REQUIRED COMPONENTS C HL)
  endif()
  find_package(LibArchive REQUIRED)
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

  target_link_libraries(nnabla_utils
    nnabla
    ${PROTOBUF_LIBRARY}
    ${LibArchive_LIBRARIES}
    ${ZLIB_LIBRARIES}
    )
  if (NNABLA_UTILS_WITH_HDF5)
    target_link_libraries(nnabla_utils
      ${HDF5_LIBRARIES}
      ${HDF5_HL_LIBRARIES})
  endif()
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
