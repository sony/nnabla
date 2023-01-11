// Copyright 2018,2019,2020,2021 Sony Corporation.
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
#include <nbla/computation_graph/computation_graph.hpp>
#include <nbla/computation_graph/function.hpp>
#include <nbla/computation_graph/variable.hpp>
#include <nbla/function/einsum.hpp>
#include <nbla/functions.hpp>
#include <nbla/variable.hpp>

namespace nbla {

NBLA_REGISTER_FUNCTION_SOURCE(Einsum, const string &);

namespace {
CgVariablePtr create_cgvariable_from_variable(Variable *var, bool need_grad) {
  auto cg_var = make_shared<CgVariable>(var->shape(), need_grad);
  cg_var->variable()->set_data(var->data());
  cg_var->variable()->set_grad(var->grad());
  return cg_var;
}

bool reset_cgvariable(CgVariablePtr cg_var, Variable *var) {
  bool ret = false;
  if (cg_var->variable()->data() != var->data()) {
    cg_var->variable()->set_data(var->data());
    ret = true;
  }
  if (cg_var->variable()->grad() != var->grad()) {
    cg_var->variable()->set_grad(var->grad());
    ret = true;
  }
  return ret;
}

bool need_transpose(vector<int> axes) {
  for (int i = 0; i < axes.size(); i++) {
    if (axes[i] != i) {
      return true;
    }
  }
  return false;
}
}

class EinsumGraph {
  // TermType indicates the dimensions of Variable corresponding to the label
  using TermType = map<char, vector<size_t>>;

  vector<CgVariablePtr> input_cg_variables_;
  CgVariablePtr output_cg_variable_;

  Context ctx_;
  vector<TermType> input_terms_;
  TermType output_term_;
  // last_term_index_ denotes the input index at which label last appears
  // If value is -1, indicating that it appears in the output term
  map<char, int> last_term_index_;
  set<char> labels_;

public:
  EinsumGraph(const Context &ctx, const Variables &inputs,
              const string &equation)
      : ctx_(ctx) {
    for (const auto &input : inputs) {
      input_cg_variables_.emplace_back(
          create_cgvariable_from_variable(input, true));
    }
    preprocess(equation);
    output_cg_variable_ = create_einsum_graph(input_cg_variables_);
  }

  vector<CgVariablePtr> input_cg_variables() const {
    return input_cg_variables_;
  }

  CgVariablePtr output_cg_variable() const { return output_cg_variable_; }

private:
  tuple<vector<string>, string> preprocess_equation(string equation) const {
    // Check tokens in equation
    std::regex re("([a-zA-Z, ]|\\.{3}|->)*");
    NBLA_CHECK(std::regex_match(equation, re), error_code::value,
               "Only ahphabet, commas, ellipsis(...) and '->' are allowed in "
               "equation.");

    // Remove space
    std::remove(equation.begin(), equation.end(), ' ');

    // Replace "..." to "."
    equation = std::regex_replace(equation, std::regex("\\.{3}"), ".");

    // Split equation to input_labels and output_label
    vector<string> input_labels;
    string output_label;
    {
      const auto split_pos = equation.find("->");
      string left_equation;
      if (split_pos != std::string::npos) {
        // Split equation to left_equation and output label
        left_equation = equation.substr(0, split_pos);
        output_label = equation.substr(split_pos + 2);
      } else {
        // Generate output_label
        left_equation = equation;
        map<char, int> label_count;
        bool exist_ellipsis = false;
        for (const auto &label : left_equation) {
          if (std::isalpha(label)) {
            label_count[label]++;
          } else if (label == '.') {
            exist_ellipsis = true;
          }
        }
        if (exist_ellipsis) {
          output_label = ".";
        }
        for (const auto &pair : label_count) {
          const auto label = pair.first;
          const auto count = pair.second;
          if (count == 1) {
            output_label.push_back(label);
          }
        }
      }

      // Split left_equation to input_labels
      input_labels = split_string(left_equation, ',');

      for (const auto &input_label : input_labels) {
        NBLA_CHECK(!input_label.empty(), error_code::value,
                   "equation '%s' has an empty input term.",
                   input_label.c_str());
      }
    }
    return std::make_tuple(input_labels, output_label);
  }

  size_t
  compute_ellipsis_ndim(const vector<string> &input_labels,
                        const vector<CgVariablePtr> &input_cg_variables) const {
    size_t ellipsis_ndim = 0;
    for (size_t i = 0; i < input_labels.size(); i++) {
      const auto &input_label = input_labels[i];
      NBLA_CHECK(std::count(input_label.begin(), input_label.end(), '.') < 2,
                 error_code::value,
                 "Only one ellipsis is allowed per input_label.");

      const auto input_shape = input_cg_variables[i]->variable()->shape();
      const auto ndim = input_shape.size();

      NBLA_CHECK(ndim >= input_label.size(), error_code::value,
                 "The number of %zu-th input dimensions is invalid: %zu", i,
                 ndim);

      if (input_label.find('.') != string::npos) {
        const size_t current_ellipsis_dim =
            input_shape.size() - (input_label.size() - 1);
        if (ellipsis_ndim == 0) {
          ellipsis_ndim = current_ellipsis_dim;
        } else {
          NBLA_CHECK(
              ellipsis_ndim == current_ellipsis_dim, error_code::value,
              "Ellipsis must have the same number of dimension for all inputs");
        }
      }
    }

    if (ellipsis_ndim == 0) { // If ellipsis is not found in input terms
      // Check the number of dimensions for each inputs
      for (size_t i = 0; i < input_labels.size(); i++) {
        const auto &input_label = input_labels[i];
        const auto input_shape = input_cg_variables[i]->variable()->shape();
        NBLA_CHECK(input_label.size() == input_shape.size(), error_code::value,
                   "The number of dimensions indicated by equation does not "
                   "match the number of dimensions of inputs.");
      }
    }

    return ellipsis_ndim;
  }

  static TermType create_term(const string &input_label, size_t ellipsis_ndim) {
    TermType term;
    size_t dim = 0;
    for (const auto &label : input_label) {
      if (std::isalpha(label)) {
        // alphabet
        if (term.count(label) == 0) {
          term[label] = {};
        }
        term[label].push_back(dim++);
      } else {
        // ellipsis
        vector<size_t> dims;
        for (size_t k = 0; k < ellipsis_ndim; k++) {
          dims.push_back(dim++);
        }
        term[label] = std::move(dims);
      }
    }
    return term;
  }

  static void
  validate_input_shapes(const vector<CgVariablePtr> &input_cg_variables,
                        const vector<TermType> &input_terms) {
    map<char, Size_t> label_dim_size;
    vector<Size_t> ellipsis_dim_sizes;
    for (size_t i = 0; i < input_terms.size(); i++) {
      const auto &input_term = input_terms[i];
      const auto &input_shape = input_cg_variables[i]->variable()->shape();
      for (const auto &term : input_term) {
        const auto label = term.first;
        const auto dims = term.second;
        if (std::isalpha(label)) {
          // Check size of alphabet dimension
          for (const auto &dim : dims) {
            if (label_dim_size.count(label) == 0) {
              label_dim_size[label] = input_shape[dim];
            } else {
              NBLA_CHECK(
                  label_dim_size[label] == input_shape[dim], error_code::value,
                  "Same label must have same dimension size for all Inputs.");
            }
          }
        } else {
          // Check shape of ellipsis dimension
          if (ellipsis_dim_sizes.empty()) {
            for (const auto &dim : dims)
              ellipsis_dim_sizes.push_back(input_shape[dim]);
          } else {
            for (size_t j = 0; j < dims.size(); j++) {
              const auto &dim = dims[j];
              NBLA_CHECK(ellipsis_dim_sizes[j] == input_shape[dims[j]] ||
                             ellipsis_dim_sizes[j] == 1 ||
                             input_shape[dim] == 1,
                         error_code::value, "Ellipsis must have same "
                                            "dimension size or either one is "
                                            "1 for broadcast.");
              ellipsis_dim_sizes[j] =
                  std::max(ellipsis_dim_sizes[j], input_shape[dim]);
            }
          }
        }
      }
    }
  }

  void preprocess(const string &equation) {
    // Get input labels and output label from equation
    vector<string> input_labels;
    string output_label;
    std::tie(input_labels, output_label) = preprocess_equation(equation);
    NBLA_CHECK(input_cg_variables_.size() == input_labels.size(),
               nbla::error_code::value, "The number of the input terms in "
                                        "equation does not match the inputs "
                                        "size.");

    // Compute the sizes of dimensions of ellipsis
    size_t ellipsis_ndim =
        compute_ellipsis_ndim(input_labels, input_cg_variables_);

    // Generate input_terms
    last_term_index_.clear();
    for (size_t i = 0; i < input_labels.size(); i++) {
      const auto &input_label = input_labels[i];
      const auto input_term = create_term(input_label, ellipsis_ndim);
      for (const auto &label : input_label) {
        labels_.insert(label);
        last_term_index_[label] = i;
      }
      input_terms_.push_back(std::move(input_term));
    }

    validate_input_shapes(input_cg_variables_, input_terms_);

    // Generate output_term
    {
      output_term_ = create_term(output_label, ellipsis_ndim);
      for (const auto label : output_label) {
        last_term_index_[label] = -1;
      }
    }
  }

  tuple<CgVariablePtr, TermType>
  connect_sum(CgVariablePtr x, const TermType &x_term,
              const vector<char> &reduce_labels) {
    // Make axes for Sum function
    vector<int> reduce_axes;
    for (const auto &pair : x_term) {
      const auto label = pair.first;
      const auto &label_dims = pair.second;
      const auto it =
          std::find(reduce_labels.cbegin(), reduce_labels.cend(), label);
      if (it != reduce_labels.cend()) {
        // Insert axes
        reduce_axes.insert(reduce_axes.begin(), label_dims.begin(),
                           label_dims.end());
      }
    }

    CgVariablePtr last_out = x;
    if (reduce_axes.size() > 0) {
      last_out = functions::sum(ctx_, last_out, reduce_axes, false)[0];
    }

    // Update term
    TermType reduced_term;
    for (const auto &pair : x_term) {
      const auto label = pair.first;
      const auto &label_dims = pair.second;

      const auto it =
          std::find(reduce_labels.cbegin(), reduce_labels.cend(), label);
      if (it != reduce_labels.cend())
        continue; // Remove reduce label

      // If label is not reduced, create new label_dims
      vector<size_t> new_label_dims;
      for (size_t i = 0; i < label_dims.size(); i++) {
        const auto dim = label_dims[i];

        const size_t count =
            std::count_if(reduce_axes.cbegin(), reduce_axes.cend(),
                          [&](int reduce_axis) { return dim > reduce_axis; });

        new_label_dims.push_back(dim - count);
      }
      reduced_term[label] = std::move(new_label_dims);
    }

    return std::make_tuple(last_out, reduced_term);
  }

  tuple<CgVariablePtr, TermType>
  connect_matmul(CgVariablePtr a, const TermType &a_term, CgVariablePtr b,
                 const TermType &b_term, int b_index) {
    const auto a_index = b_index - 1;
    auto last_a_term = a_term;
    auto last_b_term = b_term;

    // Classify labels to batch, M, N, and K
    vector<char> batch_labels;
    vector<char> m_labels, n_labels, k_labels;
    vector<char> a_reduce_labels, b_reduce_labels;
    for (const auto &label : labels_) {
      const bool has_a = last_a_term.count(label) == 1;
      const bool has_b = last_b_term.count(label) == 1;
      const int last_index = last_term_index_.at(label);
      if (has_a && has_b) {
        // the label appears in successors
        if (last_index > b_index || last_index == -1)
          batch_labels.push_back(label);
        else
          k_labels.push_back(label);
      } else if (has_a && !has_b) {
        // the label does not appear in a's successors
        if (last_index == a_index)
          a_reduce_labels.push_back(label);
        else
          m_labels.push_back(label);
      } else if (!has_a && has_b) {
        // the label does not appear in b's successors
        if (last_index == b_index)
          b_reduce_labels.push_back(label);
        else
          n_labels.push_back(label);
      }
    }

    CgVariablePtr last_out_a = a;
    CgVariablePtr last_out_b = b;

    // Sum
    if (a_reduce_labels.size() > 0) {
      std::tie(last_out_a, last_a_term) =
          connect_sum(last_out_a, last_a_term, a_reduce_labels);
    }
    if (b_reduce_labels.size() > 0) {
      std::tie(last_out_b, last_b_term) =
          connect_sum(last_out_b, last_b_term, b_reduce_labels);
    }

    // Compute transpose parameters and output term
    const auto a_shape = last_out_a->variable()->shape();
    const auto b_shape = last_out_b->variable()->shape();
    vector<int> batch_a_shape, batch_b_shape, m_shape, n_shape;
    int m_size = 1, n_size = 1, k_size = 1;
    vector<int> a_axes, b_axes;
    TermType matmul_output_term;

    size_t output_dim = 0;
    for (const auto &label : batch_labels) {
      vector<size_t> out_label;
      const auto &label_dims_a = last_a_term[label];
      const auto &label_dims_b = last_b_term[label];
      for (size_t i = 0; i < label_dims_a.size(); i++) {
        a_axes.push_back(label_dims_a[i]);
        b_axes.push_back(label_dims_b[i]);
        out_label.push_back(output_dim++);
        batch_a_shape.push_back(a_shape[label_dims_a[i]]);
        batch_b_shape.push_back(b_shape[label_dims_b[i]]);
      }
      matmul_output_term[label] = std::move(out_label);
    }
    for (const auto &label : m_labels) {
      vector<size_t> out_label;
      const auto &label_dims_a = last_a_term[label];
      for (size_t i = 0; i < label_dims_a.size(); i++) {
        a_axes.push_back(label_dims_a[i]);
        out_label.push_back(output_dim++);
        m_size *= a_shape[label_dims_a[i]];
        m_shape.push_back(a_shape[label_dims_a[i]]);
      }
      matmul_output_term[label] = std::move(out_label);
    }
    for (const auto &label : k_labels) {
      const auto &label_dims_a = last_a_term[label];
      const auto &label_dims_b = last_b_term[label];
      for (size_t i = 0; i < label_dims_a.size(); i++) {
        a_axes.push_back(label_dims_a[i]);
        b_axes.push_back(label_dims_b[i]);
        k_size *= a_shape[label_dims_a[i]];
      }
    }
    for (const auto &label : n_labels) {
      vector<size_t> out_label;
      const auto &label_dims_b = last_b_term[label];
      for (size_t i = 0; i < label_dims_b.size(); i++) {
        b_axes.push_back(label_dims_b[i]);
        out_label.push_back(output_dim++);
        n_size *= b_shape[label_dims_b[i]];
        n_shape.push_back(b_shape[label_dims_b[i]]);
      }
      matmul_output_term[label] = std::move(out_label);
    }

    // Transpose input matrices
    if (need_transpose(a_axes)) {
      last_out_a = functions::transpose(ctx_, last_out_a, a_axes)[0];
    }
    if (need_transpose(b_axes)) {
      last_out_b = functions::transpose(ctx_, last_out_b, b_axes)[0];
    }

    // Reshape:
    // - matrix A: [B0, ..., M0, ..., K0, ...] -> [B0, ..., M, K]
    // - matrix B: [B0, ..., K0, ..., N0, ...] -> [B0, ..., K, N]
    vector<int> shape_a(batch_a_shape);
    if (batch_a_shape.size() == 0) {
      shape_a.push_back(1);
    }
    shape_a.push_back(m_size);
    shape_a.push_back(k_size);
    vector<int> shape_b(batch_b_shape);
    if (batch_b_shape.size() == 0) {
      shape_b.push_back(1);
    }
    shape_b.push_back(k_size);
    shape_b.push_back(n_size);

    last_out_a = functions::reshape(ctx_, last_out_a, shape_a, true)[0];
    last_out_b = functions::reshape(ctx_, last_out_b, shape_b, true)[0];

    // BatchMatmul: [B, M, K] * [B, K, N] -> [B, M, N]
    CgVariablePtr last_out =
        functions::batch_matmul(ctx_, last_out_a, last_out_b, false, false)[0];

    // Reshape: [B0, ..., M, N] -> [B0, ..., M0, ..., N0, ...]
    vector<int> out_shape;
    for (int i = 0; i < batch_a_shape.size(); i++) {
      out_shape.push_back(std::max(batch_a_shape[i], batch_b_shape[i]));
    }
    out_shape.insert(out_shape.end(), m_shape.begin(), m_shape.end());
    out_shape.insert(out_shape.end(), n_shape.begin(), n_shape.end());
    last_out = functions::reshape(ctx_, last_out, out_shape, true)[0];

    return std::make_tuple(last_out, matmul_output_term);
  }

  tuple<CgVariablePtr, TermType> connect_diagonal(CgVariablePtr x,
                                                  const TermType &x_term,
                                                  int axis0, int axis1) {
    const size_t ndim = x->variable()->ndim();

    vector<int> pre_transpose_axes(ndim);
    vector<int> post_transpose_axes(ndim - 1);
    {
      size_t ind = 0;
      for (size_t i = 0; i < ndim; i++) {
        if (i != axis0 && i != axis1)
          pre_transpose_axes[ind++] = i;
      }
      pre_transpose_axes[ndim - 2] = axis0;
      pre_transpose_axes[ndim - 1] = axis1;

      ind = 0;
      for (size_t i = 0; i < ndim - 1; i++) {
        if (pre_transpose_axes[i] > axis1)
          post_transpose_axes[pre_transpose_axes[i] - 1] = ind++;
        else
          post_transpose_axes[pre_transpose_axes[i]] = ind++;
      }
    }

    CgVariablePtr last_out = x;

    // Transpose: [..., axis0, ..., axis1, ...] -> [..., axis0, axis1]
    // for matrix_diag_part
    if (need_transpose(pre_transpose_axes)) {
      last_out = functions::transpose(ctx_, last_out, pre_transpose_axes)[0];
    }

    // Diagonal: [..., axis0, axis1] -> [..., axis0]
    last_out = functions::matrix_diag_part(ctx_, last_out)[0];

    // Transpose: [..., axis0] - > [..., axis0, ...]
    if (need_transpose(post_transpose_axes)) {
      last_out = functions::transpose(ctx_, last_out, post_transpose_axes)[0];
    }

    TermType last_term;
    for (const auto &pair : x_term) {
      const auto label = pair.first;
      const auto &label_dims = pair.second;

      vector<size_t> new_label_dims;
      for (const auto label_axis : label_dims) {
        if (label_axis < axis1) {
          new_label_dims.push_back(label_axis);
        } else if (label_axis == axis1) {
          // Do nothing (remove this axis)
        } else if (label_axis > axis1) {
          new_label_dims.push_back(label_axis - 1);
        }
      }
      last_term[label] = std::move(new_label_dims);
    }

    return make_tuple(last_out, last_term);
  }

  static vector<char> get_labels(const TermType &term) {
    vector<char> labels;
    labels.reserve(term.size());
    for (const auto &pair : term) {
      labels.push_back(pair.first);
    }
    return labels;
  }

  CgVariablePtr
  create_einsum_graph(const vector<CgVariablePtr> &input_cg_variables) {
    vector<CgVariablePtr> last_outs = input_cg_variables;
    vector<TermType> last_terms = input_terms_;

    // Diagonal
    for (int i = 0; i < last_outs.size(); i++) {
      auto &input_term = last_terms[i];
      const auto labels = get_labels(input_term);

      for (const auto label : labels) {
        if (!std::isalpha(label))
          continue;

        const auto label_dims = input_term[label]; // copy
        for (int j = label_dims.size() - 1; j > 0; j--) {
          const auto curr_axis = label_dims[j];
          const auto prev_axis = label_dims[j - 1];
          std::tie(last_outs[i], last_terms[i]) = connect_diagonal(
              last_outs[i], last_terms[i], prev_axis, curr_axis);
        }
      }
    }
    input_terms_ = last_terms;

    CgVariablePtr last_out;
    TermType last_term;
    if (last_outs.size() == 1) {
      // Sum
      vector<char> reduce_labels;
      for (const auto &pair : last_terms[0]) {
        const auto label = pair.first;
        if (last_term_index_.at(label) != -1) {
          // Reduce labels is not in output_term
          reduce_labels.push_back(label);
        }
      }
      last_out = last_outs[0];
      last_term = last_terms[0];
      std::tie(last_out, last_term) =
          connect_sum(last_out, last_term, reduce_labels);

      last_terms[0] = last_term;
    } else {
      // Matmul
      last_out = last_outs[0];
      last_term = last_terms[0];
      for (int i = 1; i < last_outs.size(); i++) {
        // Matmul last_out[i - 1] and last_out[i]
        std::tie(last_out, last_term) =
            connect_matmul(last_out, last_term, last_outs[i], last_terms[i], i);
      }
    }

    if (output_term_ != last_term) {
      // Transpose
      vector<int> transpose_axes(last_out->variable()->ndim());
      for (const auto &term : last_term) {
        const auto label = term.first;
        const auto &label_dims = term.second;
        for (int i = 0; i < label_dims.size(); i++) {
          transpose_axes[output_term_[label][i]] = label_dims[i];
        }
      }
      last_out = functions::transpose(ctx_, last_out, transpose_axes)[0];
    }

    if (!last_out->has_parent()) {
      // Do nothing
      last_out = functions::identity(last_out);
    }
    return last_out;
  }

  static vector<string> split_string(const string &s, char delim) {
    vector<string> v;
    stringstream ss(s);
    string elem;
    while (std::getline(ss, elem, delim)) {
      v.push_back(elem);
    }
    return v;
  }
};

template <typename T>
void Einsum<T>::setup_impl(const Variables &inputs, const Variables &outputs) {
  EinsumGraph graph(ctx_, inputs, equation_);
  input_cg_variables_ = graph.input_cg_variables();
  last_output_cg_variable_ = graph.output_cg_variable();

  outputs[0]->reshape(last_output_cg_variable_->variable()->shape(), true);
  last_output_cg_variable_->variable()->set_data(outputs[0]->data());
  last_output_cg_variable_->variable()->set_grad(outputs[0]->grad());
  // Call all setup again to ensure inplaced variable is refer to the correct
  // array.
  unordered_set<CgFunctionPtr> fclosed;
  last_output_cg_variable_->visit_function_recursive(
      last_output_cg_variable_->parent(), fclosed, false /* as_recomputation */,
      [](CgFunctionPtr fn) { fn->setup(); });
}

template <typename T>
void Einsum<T>::forward_impl(const Variables &inputs,
                             const Variables &outputs) {
  for (int i = 0; i < inputs.size(); i++) {
    reset_cgvariable(input_cg_variables_[i], inputs[i]);
  }
  bool clear_buffer =
      SingletonManager::get<GlobalClearBufferState>()->clear_buffer();
  last_output_cg_variable_->forward(clear_buffer, false);
}

template <typename T>
void Einsum<T>::backward_impl(const Variables &inputs, const Variables &outputs,
                              const vector<bool> &propagate_down,
                              const vector<bool> &accum) {
  for (int i = 0; i < inputs.size(); i++) {
    reset_cgvariable(input_cg_variables_[i], inputs[i]);
    input_cg_variables_[i]->set_need_grad(propagate_down[i]);
  }
  // Propagate need_grad states
  unordered_set<CgFunctionPtr> fclosed;
  last_output_cg_variable_->visit_function_recursive(
      last_output_cg_variable_->parent(), fclosed, false /* recomputation */,
      [](CgFunctionPtr fn) {});
  last_output_cg_variable_->backward(outputs[0]->grad(), true);
}
}
