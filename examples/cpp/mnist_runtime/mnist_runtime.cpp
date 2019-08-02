// Copyright (c) 2017 Sony Corporation. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <nbla_utils/nnp.hpp>

#ifdef WITH_CUDA
#include <nbla/cuda/cudnn/init.hpp>
#include <nbla/cuda/init.hpp>
#endif

#ifdef TIMING
#include <chrono>
#endif

#include <cassert>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>

/** Read PGM image with MNIST shape.

    This easy implementation does not support all possible P5 PGM format.

   @param[in] filename Path to Raw PGM file (P5). 28x28 maxval must be 255.
   @param[out] data Data will be store into this array.
*/
void read_pgm_mnist(const std::string &filename, uint8_t *data) {
  using namespace nbla;
  std::ifstream file(filename);
  std::string buff;

  // Read P5.
  getline(file, buff);
  NBLA_CHECK(buff == "P5", error_code::value, "Only P5 is supported (%s).",
             buff.c_str());
  // Read comment line
  std::getline(file, buff);
  if (!buff.empty()) {
    NBLA_CHECK(buff[0] == '#', error_code::value,
               "Comment line must start with #. (%s)", buff.c_str());
  }

  // Read image shape.
  std::getline(file, buff);
  std::stringstream ss(buff);
  int width;
  int height;
  ss >> width;
  ss >> height;
  NBLA_CHECK(width == 28 && height == 28, error_code::value,
             "Image size must be 28 x 28 (given %d x %d).", width, height);

  // Read max value.
  std::getline(file, buff);
  ss.clear();
  ss.str(buff);
  int maxval;
  ss >> maxval;
  NBLA_CHECK(maxval == 255, error_code::value, "maxVal must be 255 (given %d).",
             maxval);

  // Read image data.
  NBLA_CHECK(file.read((char *)data, width * height * sizeof(uint8_t)),
             error_code::value, "Only read %d bytes", (int)(file.gcount()));
}

int main(int argc, char *argv[]) {
  if (!(argc == 3 || argc == 4)) {
    std::cerr << "Usage: " << argv[0] << " nnp_file input_pgm" << std::endl;
    std::cerr << std::endl;
    std::cerr << "Positional arguments: " << std::endl;
    std::cerr << "  nnp_file  : .nnp file created by "
                 "examples/vision/mnist/save_nnp_classification.py."
              << std::endl;
    std::cerr << "  input_pgm : PGM (P5) file of a 28 x 28 image where pixel "
                 "values < 256."
              << std::endl;
    std::cerr << "  executor (optional) : Executor name in nnp file."
              << std::endl;
    return -1;
  }
  const std::string nnp_file(argv[1]);
  const std::string input_bin(argv[2]);
  std::string executor_name("runtime");
  if (argc == 4) {
    executor_name = argv[3];
  }

  // Create a context (the following setting is recommended.)
  nbla::Context cpu_ctx{{"cpu:float"}, "CpuCachedArray", "0"};
#ifdef WITH_CUDA
  nbla::init_cudnn();
  nbla::Context ctx{
      {"cudnn:float", "cuda:float", "cpu:float"}, "CudaCachedArray", "0"};
#else
  nbla::Context ctx = cpu_ctx;
#endif

  // Create a Nnp object
  nbla::utils::nnp::Nnp nnp(ctx);

  // Set nnp file to Nnp object.
  nnp.add(nnp_file);

  // Get an executor instance.
  auto executor = nnp.get_executor(executor_name);
  executor->set_batch_size(1); // Use batch_size = 1.

  // Get input data as a CPU array.
  nbla::CgVariablePtr x = executor->get_data_variables().at(0).variable;
  uint8_t *data = x->variable()->cast_data_and_get_pointer<uint8_t>(cpu_ctx);

  // Read input pgm file and store image data into the CPU array.
  read_pgm_mnist(input_bin, data);

#ifdef TIMING
  // The first execution of NNabla is slower due to resource allocation at
  // runtime.
  // Hence, the execution for warmup is performed if timing.
  std::cout << "Warming up..." << std::endl;
  executor->execute();
#ifdef WITH_CUDA
  nbla::cuda_device_synchronize("0");
#endif
  // Timing starts
  auto start = std::chrono::steady_clock::now();
#endif

  // Execute prediction
  std::cout << "Executing..." << std::endl;
  executor->execute();

#ifdef TIMING
#ifdef WITH_CUDA
  nbla::cuda_device_synchronize("0");
#endif
  // Timing ends
  auto end = std::chrono::steady_clock::now();
  std::cout << "Elapsed time: "
            << std::chrono::duration_cast<std::chrono::microseconds>(end -
                                                                     start)
                       .count() *
                   0.001
            << " [ms]." << std::endl;
#endif

  // Get output as a CPU array;
  nbla::CgVariablePtr y = executor->get_output_variables().at(0).variable;
  const float *y_data = y->variable()->get_data_pointer<float>(cpu_ctx);
  assert(y->variable()->size() == 10);

  int prediction = 0;
  float max_score = -1e10;
  std::cout << "Prediction scores:";
  for (int i = 0; i < 10; i++) {
    if (y_data[i] > max_score) {
      prediction = i;
      max_score = y_data[i];
    }
    std::cout << " " << std::setw(5) << y_data[i];
  }
  std::cout << std::endl;
  std::cout << "Prediction: " << prediction << std::endl;
  return 0;
}
