all: mnist_runtime.cpp
	$(CXX) -std=c++11 -O -o mnist_runtime mnist_runtime.cpp -lnnabla -lnnabla_utils -lhdf5-static

clean:
	rm -f mnist_runtime
