#include <nbla/function/utils/base_pooling.hpp>

namespace nbla {
namespace {
inline vector<int> get_stride(const vector<int> &kernel, vector<int> stride) {
  if (stride.size() == 0) {
    std::copy(kernel.cbegin(), kernel.cend(), std::back_inserter(stride));
  }
  return stride;
}

inline vector<int>
get_pooling_output_shape(const vector<int> &inshape, const vector<int> &kernel,
                         const vector<int> &stride, const vector<int> &pad,
                         bool ignore_border, bool channel_last) {
  const int s = inshape.size() - kernel.size();

  NBLA_CHECK(kernel.size() == stride.size(), error_code::value,
             "Length of kernel and stride must be same. "
             "kernel: %d != stride: %d.",
             kernel.size(), stride.size());
  NBLA_CHECK(kernel.size() <= inshape.size() - 2, error_code::value,
             "Length of kernel must be less than length of inshape - 2. There "
             "must be a batch dimension and a channel dimension in addition to "
             "spatial dimensions. "
             "kernel: %d != inshape - 2: %d.",
             kernel.size(), inshape.size() - 2);

  // TODO: support 1D
  NBLA_CHECK(kernel.size() >= 2 && kernel.size() <= 3,
             error_code::not_implemented,
             "2D and 3D Pooling are only supported so far.");

  NBLA_CHECK(kernel.size() == pad.size(), error_code::value,
             "Size of kernel and pad must be same. "
             "kernel: %d != pad: %d).",
             kernel.size(), pad.size());

  size_t first_spatial_axis = s - (channel_last ? 1 : 0);
  size_t end_spatial_axis = first_spatial_axis + kernel.size();

  vector<int> shape(kernel.size());
  for (int i = 0; i < kernel.size(); i++) {
    int w_i = static_cast<int>(inshape[i + first_spatial_axis]);
    int k_i = kernel[i];
    int s_i = stride[i];
    int p_i = pad[i];
    shape[i] = (w_i + p_i - (ignore_border ? k_i - p_i : 1)) / s_i + 1;
  }

  vector<int> outshape(inshape.size());
  for (int i = 0; i < inshape.size(); i++) {
    if ((i < first_spatial_axis) || (i >= end_spatial_axis)) {
      outshape[i] = static_cast<int>(inshape[i]);
    } else {
      outshape[i] = shape[i - first_spatial_axis];
    }
  }
  return outshape;
}
} // namespace anonymous

PoolingConfiguration::PoolingConfiguration(const vector<int> &i,
                                           const vector<int> &k,
                                           const vector<int> &s,
                                           const vector<int> &p, bool ib,
                                           bool cl)
    : inshape(i), kernel(k), stride(get_stride(k, s)), pad(p),
      ignore_border(ib), channel_last(cl),
      outshape(get_pooling_output_shape(inshape, kernel, stride, pad,
                                        ignore_border, channel_last)),
      base_axis(inshape.size() - kernel.size() - 1) {}

} // namespace nbla
