# SPDLOG is a header dependency.
SPDLOG_INCLUDE_DIR?=../../../third_party/spdlog-0.13.0/include

all: mnist_runtime.cpp
	$(CXX) -std=c++11 -O -o mnist_runtime mnist_runtime.cpp -I${SPDLOG_INCLUDE_DIR} -lnnabla -lnnabla_utils

clean:
	rm -f mnist_runtime
