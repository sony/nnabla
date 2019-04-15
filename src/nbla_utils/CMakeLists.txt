include(${CMAKE_SOURCE_DIR}/build-tools/cmake/Utils.cmake)

option(NNABLA_UTILS_WITH_HDF5 "Enable HDF5 in C++ utils." OFF)
option(NNABLA_UTILS_WITH_NPY "Enable NPY in C++ utils." ON)
option(NNABLA_UTILS_STATIC_LINK_DEPS "Static link C++ NNabla utils deps." OFF)
option(DEBUG_NETWORK_EXPANDER "Debug network expander." OFF)

if (DEBUG_NETWORK_EXPANDER)
  add_definitions(-DDEBUG_NETWORK_EXPANDER)
endif()

include_directories(${CMAKE_BINARY_DIR}/libraries/include)

list(APPEND nnabla_utils_SOURCES
  nnp.cpp
  nnabla.pb.cc
  nnp_impl.cpp
  nnp_impl_create_function.cpp
  nnp_impl_create_solver.cpp
  nnp_impl_optimizer.cpp
  nnp_impl_monitor.cpp
  nnp_impl_configs.cpp
  data_iterator.cpp
  nnp_network_expander.cpp
)

# NPY is prior option, if it is ON,
# data iterator is set to NPY, and hdf5
# is disabled.
if (NNABLA_UTILS_WITH_NPY)
  add_definitions(-DNBLA_UTILS_WITH_NPY)
  list(APPEND nnabla_utils_SOURCES
    nnp_impl_dataset_npy.cpp
  )
elseif (NNABLA_UTILS_WITH_HDF5)
  list(APPEND nnabla_utils_SOURCES
    nnp_impl_dataset_hdf5.cpp
  )
endif()

if (NNABLA_UTILS_WITH_HDF5)
  add_definitions(-DNBLA_UTILS_WITH_HDF5)
endif()

add_library(nnabla_utils SHARED
  ${nnabla_utils_SOURCES}
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
  if (ANDROID)
    include_directories(/usr/local/android/arm64/include)
    add_library(Protobuf STATIC IMPORTED)
    set_target_properties(Protobuf 
                          PROPERTIES IMPORTED_LOCATION 
                          /usr/local/android/arm64/lib/libprotobuf.a)
    
    add_library(LibArchive STATIC IMPORTED)
    set_target_properties(LibArchive 
                         PROPERTIES IMPORTED_LOCATION 
                         /usr/local/android/arm64/lib/libarchive.so) 
  else()
    find_package(Protobuf REQUIRED)
    find_package(LibArchive REQUIRED)
  endif()
  if (NNABLA_UTILS_WITH_HDF5)
    if (NNABLA_UTILS_STATIC_LINK_DEPS)
      set(HDF5_USE_STATIC_LIBRARIES 1)
    endif()
    findhdf5()
  endif()
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

  if (ANDROID)
    target_link_libraries(nnabla_utils
      nnabla
      Protobuf
      LibArchive
      ${ZLIB_LIBRARIES}
      )
  else()
    target_link_libraries(nnabla_utils
      nnabla
      ${PROTOBUF_LIBRARY}
      ${LibArchive_LIBRARIES}
      ${ZLIB_LIBRARIES}
      )
  endif()
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

if (BUILD_TEST)
  if (WIN32)
    set(BUILD_SHARED_LIBS ON CACHE BOOL
         "Force to build gtest with dll on windows.")
  endif()
  enable_testing()
  include_directories(${gtest_SOURCE_DIR}/include ${gtest_SOURCE_DIR})
  include_directories(${PROJECT_SOURCE_DIR}/src/nbla_utils/)
  include_directories(${PROJECT_SOURCE_DIR}/src/nbla_cli/)
  file(GLOB NBLA_UTILS_TEST_SOURCES test/test_*.cpp)
  list(APPEND NBLA_UTILS_TEST_SOURCES ${PROJECT_SOURCE_DIR}/src/nbla_cli/internal.cpp)
  add_executable(test_nbla_utils ${NBLA_UTILS_TEST_SOURCES})
  add_dependencies(test_nbla_utils ${NBLA_LIBRARY_NAME})
  target_link_libraries(test_nbla_utils gtest gtest_main)
  target_link_libraries(test_nbla_utils ${NBLA_LIBRARY_NAME})
  target_link_libraries(test_nbla_utils nnabla_utils)
  set_property(TARGET test_nbla_utils PROPERTY CXX_STANDARD 11)
  nbla_exclude_from_all(test_nbla_utils)
  add_test(NAME test_nbla_utils COMMAND test_nbla_utils)
endif()

set(NNABLA_UTILS_LIBRARY_NAME nnabla_utils PARENT_SCOPE)
