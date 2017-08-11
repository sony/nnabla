add_executable(nbla nbla.cpp)

target_link_libraries(nbla nbla_utils)
set_property(TARGET nbla PROPERTY CXX_STANDARD 11)

