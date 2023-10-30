#include <nbla/array.hpp>
#include <nbla/function/trilu.hpp>
#include <nbla/variable.hpp>

namespace nbla {

NBLA_REGISTER_FUNCTION_SOURCE(Trilu, int, bool);

template <typename T>
void Trilu<T>::setup_impl(const Variables &inputs, const Variables &outputs) {
  // Check input size
  NBLA_CHECK(inputs[0]->shape().size() >= 2, error_code::unclassified,
             "Input must have at least 2 dimensions.");
  mask_ = make_shared<Variable>(inputs[0]->shape());
  set_trilu_mask(inputs);
  outputs[0]->reshape(inputs[0]->shape(), true);
}

template <typename T> void Trilu<T>::set_trilu_mask(const Variables &inputs) {
  T *m = mask_->template cast_data_and_get_pointer<T>(this->ctx_, true);
  Shape_t shape = inputs[0]->shape();
  int cols = shape[shape.size() - 1];
  int rows = shape[shape.size() - 2];
  int batch = inputs[0]->size() / (cols * rows);
  for (int b = 0; b < batch; ++b) {
    for (int i = 0; i < rows; ++i) {
      for (int j = 0; j < cols; ++j) {
        int index = (b * rows * cols) + (i * cols) + j;
        bool retain_element =
            (upper_ && (i - j <= -k_)) || (!upper_ && (i - j >= -k_));
        m[index] = retain_element ? T(1) : T(0);
      }
    }
  }
}

template <typename T>
void Trilu<T>::forward_impl(const Variables &inputs, const Variables &outputs) {
  const T *x = inputs[0]->get_data_pointer<T>(this->ctx_);
  T *y = outputs[0]->cast_data_and_get_pointer<T>(this->ctx_, true);
  T *m = mask_->template cast_data_and_get_pointer<T>(this->ctx_, true);
  // Apply the trilu mask during forward pass
  for (int i = 0; i < mask_->size(); ++i) {
    y[i] = x[i] * m[i];
  }
}

template <typename T>
void Trilu<T>::backward_impl(const Variables &inputs, const Variables &outputs,
                             const vector<bool> &propagate_down,
                             const vector<bool> &accum) {
  if (!propagate_down[0]) {
    return;
  }
  T *dx = inputs[0]->cast_grad_and_get_pointer<T>(this->ctx_, !accum[0]);
  const T *dy = outputs[0]->get_grad_pointer<T>(this->ctx_);
  const T *m = mask_->template get_data_pointer<T>(this->ctx_);
  Shape_t shape = inputs[0]->shape();
  int cols = shape[shape.size() - 1];
  int rows = shape[shape.size() - 2];
  int batch = inputs[0]->size() / (cols * rows);

  for (int b = 0; b < batch; ++b) {
    for (int i = 0; i < rows; ++i) {
      for (int j = 0; j < cols; ++j) {
        int index = (b * rows * cols) + (i * cols) + j;
        dx[index] = (accum[0] ? dx[index] : (T)0) + dy[index] * m[index];
      }
    }
  }
}
} // namespace nbla
