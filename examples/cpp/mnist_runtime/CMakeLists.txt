add_executable(mnist_runtime mnist_runtime.cpp)
target_link_libraries(mnist_runtime nnabla nnabla_utils)
set_property(TARGET mnist_runtime PROPERTY CXX_STANDARD 11)
