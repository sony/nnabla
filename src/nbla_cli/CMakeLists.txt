if (WIN32)
  # Use simulated cxxabi.h since it is not appeared in MSVC. 
  include_directories(${PROJECT_SOURCE_DIR}/include/msvc)
  set(CXXABI_HPP cxxabi.cpp)
endif()

if (NNABLA_UTILS_WITH_HDF5)
  add_definitions(-DNBLA_UTILS_WITH_HDF5)
endif()

if (NNABLA_UTILS_WITH_HDF5)
if (NNABLA_UTILS_STATIC_LINK_DEPS)
    set(HDF5_USE_STATIC_LIBRARIES 1)
endif()
find_package(HDF5 REQUIRED COMPONENTS C HL)
endif()
if (NNABLA_UTILS_STATIC_LINK_DEPS)
if (NNABLA_UTILS_WITH_HDF5)
    set(HDF5_LIBRARIES ${HDF5_LIBRARIES} ${CMAKE_DL_LIBS})
endif()
endif()

if (NNABLA_UTILS_WITH_HDF5)
include_directories(${HDF5_INCLUDE_DIRS})
endif()


# nbla target definition
add_executable(nbla
  internal.cpp
  nbla.cpp
  nbla_dump.cpp
  nbla_infer.cpp
  nbla_train.cpp
  ${CXXABI_HPP})

target_link_libraries(nbla nnabla nnabla_utils)
set_property(TARGET nbla PROPERTY CXX_STANDARD 11)

install(TARGETS nbla RUNTIME DESTINATION bin)


# nbla_cli shared library
add_library(nbla_cli
  SHARED
  internal.cpp
  nbla_train.cpp
  nbla_dump.cpp
  nbla_infer.cpp
  ${CXXABI_HPP})

target_link_libraries(nbla_cli nnabla nnabla_utils)
set_property(TARGET nbla_cli PROPERTY CXX_STANDARD 11)

if (NOT WIN32)
  install(TARGETS nbla_cli
    LIBRARY DESTINATION lib
    )
  install(DIRECTORY ${PROJECT_SOURCE_DIR}/include/nbla_cli
    DESTINATION include
    )
endif()

set(NBLA_CLI_LIBRARY_NAME nbla_cli PARENT_SCOPE)