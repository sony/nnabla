if (WIN32)
  # Use simulated cxxabi.h since it is not appeared in MSVC. 
  include_directories(${PROJECT_SOURCE_DIR}/include/msvc)
endif()
add_executable(nbla nbla.cpp)

target_link_libraries(nbla nnabla_utils)
set_property(TARGET nbla PROPERTY CXX_STANDARD 11)

install(TARGETS nbla RUNTIME DESTINATION bin)
