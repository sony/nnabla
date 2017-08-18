# NNabla include directory.
NNABLA_INCLUDE_DIR?=../../../include/

# SPDLOG is a header dependency.
SPDLOG_INCLUDE_DIR?=../../../third_party/spdlog-0.13.0/include

# By building NNabla from source, you should find libnnabla.so and
# libnnabla_utils.so located in the following directory.
NNABLA_LIBRARY_DIR?=../../../build/lib

INC=-I${NNABLA_INCLUDE_DIR} -I${SPDLOG_INCLUDE_DIR}
LIBS=-L${NNABLA_LIBRARY_DIR} -lnnabla -lnnabla_utils


all: mnist

mnist: mnist_runtime.cpp ${NNABLA_LIBRARY_DIR}/libnnabla.so ${NNABLA_LIBRARY_DIR}/libnnabla_utils.so
	$(CXX) -std=c++11 -O -o mnist_runtime mnist_runtime.cpp  $(INC) $(LIBS)

clean:
	rm -f *.so mnist_runtime
