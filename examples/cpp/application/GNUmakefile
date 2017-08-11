INC=-I../../../include/ \
	-I../../../third_party/spdlog-0.13.0/include

LIBS=-L../../../../output/build_py2.7-cpp_centos6/lib -lnnabla -lnbla_utils

app : app.cpp
	$(CXX) -std=c++11 -O -o app app.cpp $(INC) $(LIBS)

