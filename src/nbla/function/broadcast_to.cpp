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
// 
// *WARNING*
// THIS FILE IS AUTO-GENERATED DUMMY CODE BY CODE GENERATOR.
// PLEASE IMPLEMENT REAL CODE AND DELETE THIS MESSAGE SOON.
// If you want to change dummy code, edit following files.
// - build-tools/code_generator/function_generator/generate_src_nbla_function_cpp.py
// - build-tools/code_generator/templates/src_nbla_function_cpp_template.cpp

/** BroadcastTo
 */
#include <nbla/array.hpp>
#include <nbla/variable.hpp>
#include <nbla/function/broadcast_to.hpp>

#include <algorithm>

namespace nbla {

NBLA_REGISTER_FUNCTION_SOURCE(BroadcastTo, int);

template <typename T>
void BroadcastTo<T>::setup_impl(const Variables &inputs, const Variables &outputs) {
	const Shape_t xs = inputs[0]->shape();
	const Shape_t ys = inputs[1]->shape();
	const int xss = xs.size();
	const int yss = ys.size();
	NBLA_CHECK(xss >= yss, error_code::value,
		"BroadcastTo expects Y (variable to be broadcasted) to be smaller than or equal to X (target variable we want to fit to): %d vs %d",
		yss, xss);
	if (axis_ < 0) {
		// No axis was specified.
		// Check if y shape can fit x shape from the tail dimension
		const int xofs = xss - yss;
		for (int i=yss-1; i>=0; i--) {
			Size_t xds = xs[xofs+i];
			Size_t yds = ys[i];
			NBLA_CHECK(xds == yds, error_code::value,
				"Dimension %d's size of X and Y do not match: %d vs %d",
				xofs+i, xds, yds);
		}
	} else {
		NBLA_CHECK(axis_ < xss, error_code::value,
			"Specified axis index %d must be within the size of the actual input dimension %d",
			axis_, xss);
		// Check if y shape can fit x shape from the axis index
		for (int i=0; i<yss; i++) {
			Size_t xds = xs[i+axis_];
			Size_t yds = ys[i];
			NBLA_CHECK(xds == yds, error_code::value,
				"Dimension %d's size of X and Y do not match: %d vs %d",
				i+axis_, xds, yds);
		}
	}
	// All check passed.
	// Reshape output to fit X.
	outputs[0]->reshape(xs, true);
}

template <typename T>
void BroadcastTo<T>::forward_impl(const Variables &inputs, const Variables &outputs) {
  // TEMPLATE CODE
}

template <typename T>
void BroadcastTo<T>::backward_impl(const Variables &inputs, const Variables &outputs,
					     const vector<bool> &propagate_down,
					     const vector<bool> &accum) {
  // TEMPLATE CODE
}

// Template instantiation
template class BroadcastTo<float>;
} // namespace nbla
