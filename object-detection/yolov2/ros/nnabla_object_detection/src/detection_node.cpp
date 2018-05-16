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

#include <cv_bridge/cv_bridge.h>
#include <image_transport/image_transport.h>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <ros/ros.h>
#include <sensor_msgs/image_encodings.h>

#include <nbla_utils/nnp.hpp>
#ifndef WITHOUT_CUDA
#include <nbla/cuda/cudnn/init.hpp>
#include <nbla/cuda/init.hpp>
#endif

#include <chrono>
#include <fstream>

using std::string;
using std::shared_ptr;
using std::vector;

static const std::string OPENCV_WINDOW = "Image window";

class NnablaObjectDetector {
  ros::NodeHandle nh_;
  ros::NodeHandle pnh_{"~"};
  image_transport::ImageTransport it_;
  image_transport::Subscriber image_sub_;
  image_transport::Publisher image_pub_;

  nbla::Context ctx_cpu_{{"cpu:float"}, "CpuCachedArray", "0"};
#ifdef WITHOUT_CUDA
  nbla::Context ctx_{{"cpu:float"}, "CpuCachedArray", "0"};
#else
  // TODO: devcie ID.
  // Replace with the following context if you want to use half type (FP16)
  // and TensorCore (available since Volta)
  // nbla::Context ctx_{
  //     {"cudnn:half", "cuda:half", "cpu:float"}, "CudaCachedArray", "0"};
  nbla::Context ctx_{
      {"cudnn:float", "cuda:float", "cpu:float"}, "CudaCachedArray", "0"};

#endif
  vector<cv::Scalar> colors_; // Color table for bounding boxes.
  vector<string> classes_;    // Class names
  shared_ptr<nbla::utils::nnp::Executor> executor_;

  // params
  string nnp_file_;
  string class_meta_file_;
  string executor_name_;
  string image_topic_;

  inline int transform_coord(float x, int size) {
    return std::max(0, std::min((int)(x * size), size - 1));
  }

  void read_class_meta_file() {
    std::ifstream ifs(class_meta_file_);
    assert(ifs.is_open());
    classes_.clear();
    string buff;
    while (std::getline(ifs, buff)) {
      classes_.push_back(buff);
    }
  }

  void create_color_table() {
    cv::RNG rng(313);
    colors_.clear();
    for (int i = 0; i < classes_.size(); i++) {
      colors_.push_back(cv::Scalar(rng(256), rng(256), rng(256)));
    }
  }

public:
  NnablaObjectDetector() : it_(nh_) {
    init_params();
    init_comm();
    init_nnabla();
    cv::namedWindow(OPENCV_WINDOW);
    read_class_meta_file();
    create_color_table();
  }

  void init_params() {
    pnh_.param("nnp_file", nnp_file_, std::string("yolov2.nnp"));
    pnh_.param("class_meta_file", class_meta_file_, std::string("coco.names"));
    pnh_.param("executor_name", executor_name_, std::string("runtime"));
    pnh_.param("image_topic", image_topic_, std::string("/usb_cam/image_raw"));
  }

  void init_comm() {
    image_sub_ =
        it_.subscribe(image_topic_, 1, &NnablaObjectDetector::imageCb, this);
    image_pub_ = it_.advertise("/image_converter/output_video", 1);
  }

  void init_nnabla() {
#ifndef WITHOUT_CUDA
    nbla::init_cudnn();
#endif

    nbla::utils::nnp::Nnp nnp(ctx_);
    nnp.add(nnp_file_);
    executor_ = nnp.get_executor(executor_name_);
    executor_->set_batch_size(1);
  }

  ~NnablaObjectDetector() { cv::destroyWindow(OPENCV_WINDOW); }

  void imageCb(const sensor_msgs::ImageConstPtr &msg) {
    cv_bridge::CvImagePtr cv_ptr;
    try {
      cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
    } catch (cv_bridge::Exception &e) {
      ROS_ERROR("cv_bridge exception: %s", e.what());
      return;
    }
    auto orig_size = cv_ptr->image.size();

    nbla::CgVariablePtr x = executor_->get_data_variables().at(0).variable;
    uint8_t *data = x->variable()->cast_data_and_get_pointer<uint8_t>(ctx_cpu_);
    assert(x->variable()->ndim() == 4);
    auto inshape = x->variable()->shape();
    const int C = inshape[1];
    const int H = inshape[2];
    const int W = inshape[3];
    assert(C == 3);

    cv::Mat resized;
    cv::resize(cv_ptr->image, resized, cv::Size{W, H});
    for (int hw = 0; hw < H * W; ++hw) {
      for (int c = 0; c < C; ++c) {
        data[c * H * W + hw] = resized.ptr()[hw * C + 2 - c];
      }
    }

    auto start = std::chrono::steady_clock::now();
    executor_->execute();
    auto end = std::chrono::steady_clock::now();
    ROS_INFO(
        "Input=%dx%dx%d FPS = %.1lf", H, W, C,
        1e+6 /
            std::chrono::duration_cast<std::chrono::microseconds>(end - start)
                .count());

    // Get output as a CPU array;
    nbla::CgVariablePtr y = executor_->get_output_variables().at(0).variable;
    const float *y_data = y->variable()->get_data_pointer<float>(ctx_cpu_);
    assert(y->variable()->ndim() == 3);
    auto outshape = y->variable()->shape();
    int num_classes = (int)classes_.size();
    int num_c = 5 + num_classes;
    assert(outshape[2] == num_c);

    // Draw boxes.
    // cv::Mat img_draw(cv_ptr->image);
    cv::Mat img_draw(resized);
    for (int b = 0; b < outshape[1]; ++b) {
      float score = -1;
      int class_idx = 0;
      for (int k = 0; k < num_classes; ++k) {
        const float score_k = y_data[b * num_c + 5 + k];
        if (score_k > score) {
          class_idx = k;
          score = score_k;
        }
      }
      if (score <= 0) {
        continue;
      }
      const float x = y_data[b * num_c + 0];
      const float y = y_data[b * num_c + 1];
      const float w = y_data[b * num_c + 2];
      const float h = y_data[b * num_c + 3];
#if 0
      const int x0 = transform_coord(x - w / 2, orig_size.width);
      const int y0 = transform_coord(y - h / 2, orig_size.height);
      const int x1 = transform_coord(x + w / 2, orig_size.width);
      const int y1 = transform_coord(y + h / 2, orig_size.height);
#else
      const int x0 = transform_coord(x - w / 2, W);
      const int y0 = transform_coord(y - h / 2, H);
      const int x1 = transform_coord(x + w / 2, W);
      const int y1 = transform_coord(y + h / 2, H);
#endif
      // Object detection with deep learning and OpenCV
      // https://goo.gl/q4RdcZ
      cv::rectangle(img_draw, {x0, y0}, {x1, y1}, colors_.at(class_idx), 2);
      int text_y0 = y0 + ((y0 > 30) ? -15 : 15);
      string label = nbla::format_string(
          "%s: %.2f%%", classes_.at(class_idx).c_str(), score * 100);
      cv::putText(img_draw, label, {x0, text_y0}, cv::FONT_HERSHEY_SIMPLEX, 0.5,
                  colors_.at(class_idx), 2);
      ROS_INFO("Detected: %s.", label.c_str());
    }

    // Update GUI Window
    cv::imshow(OPENCV_WINDOW, img_draw);
    cv::waitKey(3);

    // Output modified video stream
    image_pub_.publish(cv_ptr->toImageMsg());
  }
};

int main(int argc, char **argv) {
  ros::init(argc, argv, "nnabla_object_detection");
  NnablaObjectDetector ic;
  ros::spin();
  return 0;
}
