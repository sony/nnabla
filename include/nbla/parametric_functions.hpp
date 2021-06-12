// Copyright 2019,2020,2021 Sony Corporation.
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
// Include nnabla header files

/** Parametric Functions
*/

#ifndef __NBLA_PARAMETRIC_FUNCTIONS_HPP__
#define __NBLA_PARAMETRIC_FUNCTIONS_HPP__

#include <random>
#include <stdio.h>
#include <string>
#include <vector>

#include <nbla/computation_graph/computation_graph.hpp>
#include <nbla/computation_graph/function.hpp>
#include <nbla/computation_graph/variable.hpp>
#include <nbla/context.hpp>
#include <nbla/function.hpp>
#include <nbla/function/affine.hpp>
#include <nbla/function/batch_normalization.hpp>
#include <nbla/function/convolution.hpp>
#include <nbla/function/deconvolution.hpp>
#include <nbla/functions.hpp>
#include <nbla/initializer.hpp>
using std::make_shared;
namespace f = nbla::functions;

namespace nbla {

CgVariablePtr make_parameter(Shape_t shape, Initializer *initializer,
                             bool need_grad);

class NBLA_API ParameterDirectory {
  typedef unordered_map<string, CgVariablePtr> dict_type;
  typedef vector<string> ordered_keys_type;

  string scope_path_;
  shared_ptr<dict_type> param_dict_;
  shared_ptr<ordered_keys_type> ordered_keys_;

  ParameterDirectory(string scope_path, shared_ptr<dict_type> param_dict,
                     shared_ptr<ordered_keys_type> ordered_keys);

public:
  // Note: the root scope is empty. No slash for consistency with Python API.
  ParameterDirectory();

public:
  ParameterDirectory operator[](string name);

  CgVariablePtr get_parameter(string name);

  vector<pair<string, VariablePtr>> get_parameters();

  CgVariablePtr get_parameter_or_create(string name, Shape_t shape,
                                        Initializer *initializer,
                                        bool need_grad);
  CgVariablePtr get_parameter_or_create(string name, CgVariablePtr variable);

  ParameterDirectory create_deep_copy();
};

namespace parametric_functions {

class NBLA_API AffineOpts {
private:
  bool with_bias_;
  bool fix_parameters_;
  Initializer *w_init_;
  Initializer *b_init_;

public:
  AffineOpts();
  AffineOpts &with_bias(bool with_bias);
  AffineOpts &fix_parameters(bool val);
  AffineOpts &w_init(Initializer *w_init);
  AffineOpts &b_init(Initializer *b_init);
  bool with_bias();
  bool fix_parameters();
  Initializer *w_init();
  Initializer *b_init();
};

class NBLA_API ConvolutionOpts {
private:
  nbla::functions::ConvolutionOpts base_;
  bool with_bias_;
  bool fix_parameters_;
  Initializer *w_init_;
  Initializer *b_init_;

public:
  ConvolutionOpts();

  ConvolutionOpts &group(int val);
  ConvolutionOpts &pad(const vector<int> &val);
  ConvolutionOpts &stride(const vector<int> &val);
  ConvolutionOpts &dilation(const vector<int> &val);
  ConvolutionOpts &channel_last(bool val);

  int group() { return base_.group(); };
  const vector<int> &pad() const { return base_.pad(); };
  const vector<int> &stride() const { return base_.stride(); };
  const vector<int> &dilation() const { return base_.dilation(); };
  bool channel_last() const { return base_.channel_last(); };

  ConvolutionOpts &with_bias(bool with_bias);
  ConvolutionOpts &fix_parameters(bool val);
  ConvolutionOpts &w_init(Initializer *w_init);
  ConvolutionOpts &b_init(Initializer *b_init);

  bool with_bias() { return with_bias_; };
  bool fix_parameters() { return fix_parameters_; };
  Initializer *w_init() { return w_init_; };
  Initializer *b_init() { return b_init_; };
};

class NBLA_API DeconvolutionOpts {
private:
  nbla::functions::DeconvolutionOpts base_;
  bool with_bias_;
  bool fix_parameters_;
  Initializer *w_init_;
  Initializer *b_init_;

public:
  DeconvolutionOpts();

  DeconvolutionOpts &group(int val);
  DeconvolutionOpts &pad(const vector<int> &val);
  DeconvolutionOpts &stride(const vector<int> &val);
  DeconvolutionOpts &dilation(const vector<int> &val);
  DeconvolutionOpts &channel_last(bool val);
  DeconvolutionOpts &output_padding(const vector<int> &val);

  int group() { return base_.group(); };
  const vector<int> &pad() const { return base_.pad(); };
  const vector<int> &stride() const { return base_.stride(); };
  const vector<int> &dilation() const { return base_.dilation(); };
  bool channel_last() const { return base_.channel_last(); };
  const vector<int> &output_padding() const { return base_.output_padding(); };

  DeconvolutionOpts &with_bias(bool with_bias);
  DeconvolutionOpts &fix_parameters(bool val);
  DeconvolutionOpts &w_init(Initializer *w_init);
  DeconvolutionOpts &b_init(Initializer *b_init);

  bool with_bias() { return with_bias_; };
  bool fix_parameters() { return fix_parameters_; };
  Initializer *w_init() { return w_init_; };
  Initializer *b_init() { return b_init_; };
};

class NBLA_API BatchNormalizationOpts {
private:
  nbla::functions::BatchNormalizationOpts base_;
  bool fix_parameters_;

public:
  BatchNormalizationOpts();

  BatchNormalizationOpts &axes(const vector<int> &val);
  BatchNormalizationOpts &decay_rate(float val);
  BatchNormalizationOpts &eps(float val);
  const vector<int> &axes() const;
  float decay_rate();
  float eps();

  BatchNormalizationOpts &fix_parameters(bool val);
  bool fix_parameters();
};

NBLA_API vector<CgVariablePtr>
affine(Context &ctx, CgVariablePtr x, int base_axis, int n_out,
       ParameterDirectory parameters, bool with_bias, bool fix_parameters,
       Initializer *w_init = nullptr, Initializer *b_init = nullptr);

NBLA_API CgVariablePtr affine(CgVariablePtr x, int base_axis, int n_out,
                              ParameterDirectory parameters,
                              AffineOpts affine_opts = AffineOpts());

NBLA_API vector<CgVariablePtr>
batch_normalization(Context &ctx, CgVariablePtr x, const vector<int> &axes,
                    float decay_rate, float eps, bool batch_stats,
                    ParameterDirectory parameters, const bool fix_parameters);

NBLA_API CgVariablePtr batch_normalization(
    CgVariablePtr x, bool batch_stat, ParameterDirectory parameters,
    BatchNormalizationOpts batch_opts = BatchNormalizationOpts());

NBLA_API vector<CgVariablePtr>
convolution(Context &ctx, CgVariablePtr x, int base_axis, int n_map_out,
            const vector<int> &kernel, const vector<int> &pad,
            const vector<int> &stride, const vector<int> &dilation, int group,
            bool channel_last, ParameterDirectory parameters, bool with_bias,
            bool fix_parameters, Initializer *w_init, Initializer *b_init);

NBLA_API CgVariablePtr
convolution(CgVariablePtr x, int base_axis, int n_map_out,
            const vector<int> &kernel, ParameterDirectory parameters,
            ConvolutionOpts conv_opts = ConvolutionOpts());

NBLA_API vector<CgVariablePtr>
deconvolution(Context &ctx, CgVariablePtr x, int base_axis, int n_map_out,
              const vector<int> &kernel, const vector<int> &pad,
              const vector<int> &stride, const vector<int> &dilation, int group,
              ParameterDirectory parameters, bool with_bias,
              bool fix_parameters, Initializer *w_init, Initializer *b_init);

NBLA_API CgVariablePtr
deconvolution(CgVariablePtr x, int base_axis, int n_map_out,
              const vector<int> &kernel, ParameterDirectory parameters,
              DeconvolutionOpts conv_opts = DeconvolutionOpts());
}
}

#endif
