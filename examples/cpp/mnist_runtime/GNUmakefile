all: mnist_runtime.cpp
	$(CXX) -std=c++11 -D TIMING -O -o mnist_runtime mnist_runtime.cpp -lnnabla -lnnabla_utils

cuda: mnist_runtime.cpp
	$(CXX) -std=c++11 -D TIMING -D WITH_CUDA -O -o mnist_runtime_cuda mnist_runtime.cpp -lnnabla -lnnabla_utils -lnnabla_cuda

clean:
	rm -f mnist_runtime
