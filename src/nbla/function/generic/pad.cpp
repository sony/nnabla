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


#include <nbla/array.hpp>
#include <nbla/common.hpp>
#include <nbla/function/pad.hpp>
#include <nbla/variable.hpp>

namespace nbla {

NBLA_REGISTER_FUNCTION_SOURCE(Pad, const vector<int> &, const string &, float);

template <typename T>
void Pad<T>::setup_impl(const Variables &inputs,
                               const Variables &outputs) {
  Shape_t shape_x = inputs[0]->shape();
  Shape_t shape_y = shape_x;

  NBLA_CHECK(pad_width_.size()%2 == 0 , error_code::value,
             "pad_with should be even number.");
  NBLA_CHECK(pad_width_.size()/2 <= shape_x.size() , error_code::value,
             "pad_with %d dimensions does not match with input %d dimensions.",pad_width_.size()/2,shape_x.size());
  NBLA_CHECK(shape_x.size() == 4, error_code::value,
             "Input with more than %d dimension is not supported currently.",shape_x.size());
  NBLA_CHECK(pad_width_.size()/2 <= 2, error_code::value,
             "%d dimension padding is not supported currently.",pad_width_.size()/2);
  NBLA_CHECK(mode_.compare("constant") == 0 , error_code::value,
             "Only constant padding is supported currently.");

  //Calculate output shape
  int j=pad_width_.size()-1 , t_nd = pad_width_.size()/2;
  for(int i=shape_x.size()-1; i>=shape_x.size()-t_nd; i--,j-=2)
  {
    shape_y[i] = pad_width_[j] + shape_x[i] + pad_width_[j-1];
  }
  outputs[0]->reshape(shape_y, true);
}

template <typename T>
void Pad<T>::forward_impl(const Variables &inputs,
                               const Variables &outputs) {
  // Inputs
  const T* x = inputs[0]->get_data_pointer<T>(this->ctx_);
  T* y = outputs[0]->cast_data_and_get_pointer<T>(this->ctx_, true);
  vector<int> padding = pad_width_;

  // If pad_width_ is 1D , add dummy padding and convert to 2D padding. (i.e. Insert top and bottom padding as '0').
  if(padding.size() <= 2)
  {
     padding.insert(padding.begin(),0);
     padding.insert(padding.begin(),0);
  }

  int pad_top=padding[0],pad_left=padding[2];
  int input_height=inputs[0]->shape()[2];
  int input_width=inputs[0]->shape()[3];
  int output_height=outputs[0]->shape()[2];
  int output_width=outputs[0]->shape()[3];

  //For ND padding
  if(pad_width_.size()/2 > 2){
    //ToDo:
  }
  // For 1D and 2D padding
  else{
	for (int i = 0; i < outputs[0]->size(); ++i)
	{
		int num_channel = i / output_width;
		const int pad_h = num_channel % output_height;
		const int pad_w = i % output_width;
		const int height = pad_h - pad_top;
		const int width = pad_w - pad_left;
		num_channel /= output_height;

		if((height >= 0 && height < input_height) &&
		   (width  >= 0 && width  < input_width)) {
			y[i] = x[(num_channel * input_height + height) * input_width + width];
		}else {
			y[i] = constant_value_;
		}
	}
  }
}

template <typename T, bool accum>
void pad_backward_impl_cpu(int out_size,const T *dy, T *dx, int pad_top, int pad_left,
                                                   int input_height, int input_width,
                                                   int output_height, int output_width) {
	for (int i = 0; i < out_size; ++i)
	{
		int num_channel = i / output_width;
		const int pad_h = num_channel % output_height;
		const int pad_w = i % output_width;
		const int height = pad_h - pad_top;
		const int width = pad_w - pad_left;
		num_channel /= output_height;

		if((height >= 0 && height < input_height) &&
		   (width  >= 0 && width  < input_width)) {
			if(accum){
				dx[(num_channel * input_height + height) * input_width + width] += dy[i];
			}
			else{
				dx[(num_channel * input_height + height) * input_width + width] = dy[i];
			}
		}
	}
}

template <typename T>
void Pad<T>::backward_impl(const Variables &inputs,
                                const Variables &outputs,
                                const vector<bool> &propagate_down,
                                const vector<bool> &accum) {
  if (!(propagate_down[0])) {
    return;
  }

  // Gradient of outputs
  const T* dy = outputs[0]->get_grad_pointer<T>(this->ctx_);
  T* dx = inputs[0]->cast_grad_and_get_pointer<T>(this->ctx_, !accum[0]);
  vector<int> padding = pad_width_;

  // If 1D padding insert top and bottom padding as '0' and use 2D padding code
  if(padding.size() <= 2)
  {
     padding.insert(padding.begin(),0);
     padding.insert(padding.begin(),0);
  }

  if (accum[0])
    pad_backward_impl_cpu<T, true>(outputs[0]->size(), dy, dx, padding[0],padding[2],
                                       inputs[0]->shape()[2],inputs[0]->shape()[3],
                                       outputs[0]->shape()[2],outputs[0]->shape()[3]);
  else
    pad_backward_impl_cpu<T, false>(outputs[0]->size(), dy, dx, padding[0],padding[2],
                                       inputs[0]->shape()[2],inputs[0]->shape()[3],
                                       outputs[0]->shape()[2],outputs[0]->shape()[3]);

}

}
