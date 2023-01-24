# Copyright 2022 Sony Corporation.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import pytest
import numpy as np
import nnabla as nn
import nnabla.functions as F
from nbla_test_utils import list_context
from nnabla.testing import assert_allclose

# Use ONNX reference only when a test case cannot be executed with ONNX Runtime.
USE_ONNX_REF = False

# Force to use ONNX reference to execute all test cases.
# NOTICE: ONNX reference and ONNX Runtime produce different results in some
# test cases. nnabla ONNXResize mimics the behavior of ONNX Runtime because
# it seems that the differences are due to ambiguousness of ONNX specification
# or a bug of ONNX reference.
FORCE_ONNX_REF = False

ctxs = list_context('ONNXResize')


def make_onnx_model(x_shape, roi, scales, sizes, mode, coord_mode,
                    cubic_coeff_a, exclude_outside, extrapolation_value,
                    nearest_mode):
    import onnx.helper as oh
    from onnx import TensorProto

    # Resize node
    sizes_name = "sizes" if len(sizes) > 0 else ""
    node = oh.make_node(
        "Resize",
        inputs=["X", "roi", "scales", sizes_name],
        outputs=["Y"],
        coordinate_transformation_mode=coord_mode,
        cubic_coeff_a=cubic_coeff_a,
        exclude_outside=exclude_outside,
        extrapolation_value=extrapolation_value,
        mode=mode,
        nearest_mode=nearest_mode
    )

    inputs = []

    # X value_info
    x_value = oh.make_tensor_value_info("X", TensorProto.FLOAT, x_shape)
    inputs.append(x_value)

    # roi value_info
    if len(roi) > 0:
        roi_value = oh.make_tensor_value_info("roi", TensorProto.FLOAT,
                                              roi.shape)
    else:
        roi_value = oh.make_tensor_value_info("roi", TensorProto.FLOAT, [])
    inputs.append(roi_value)

    # scales value_info
    x_dims = len(x_shape)
    y_shape = None
    if len(scales) > 0:
        assert len(scales) == x_dims
        y_shape = [int(x * scales[i]) for i, x in enumerate(x_shape)]
        scales_value = oh.make_tensor_value_info("scales", TensorProto.FLOAT,
                                                 scales.shape)
    else:
        scales_value = oh.make_tensor_value_info("scales", TensorProto.FLOAT,
                                                 [])
    inputs.append(scales_value)

    # sizes value_info
    if len(sizes) > 0:
        assert len(sizes) == x_dims
        y_shape = [int(x) for x in sizes]
        sizes_value = oh.make_tensor_value_info("sizes", TensorProto.INT64,
                                                sizes.shape)
    else:
        sizes_value = oh.make_tensor_value_info("sizes", TensorProto.INT64, [])
    inputs.append(sizes_value)

    # Y value_info
    outputs = [oh.make_tensor_value_info("Y", TensorProto.FLOAT, y_shape)]

    # Model
    opset = onnx.OperatorSetIdProto()
    opset.version = 11
    graph = oh.make_graph([node], "onnx_resize_test", inputs, outputs)
    model = oh.make_model(graph, opset_imports=[opset])
    onnx.checker.check_model(model)
    return model


# ONNX Reference implementation
def ref_resize_onnx_ref(x, roi, scales, sizes, mode, coord_mode, cubic_coeff_a,
                        exclude_outside, extrapolation_value, nearest_mode):
    def get_coeffs(x):
        if mode == "nearest":
            return nearest_coeffs(x, mode=nearest_mode)
        if mode == "linear":
            return linear_coeffs(x)
        if mode == "cubic":
            return cubic_coeffs(x, A=cubic_coeff_a)

    roi = None if len(roi) == 0 else roi
    if len(scales) > 0:
        output = interpolate_nd(x, get_coeffs, scale_factors=scales, roi=roi,
                                coordinate_transformation_mode=coord_mode,
                                exclude_outside=exclude_outside,
                                extrapolation_value=extrapolation_value)
    elif len(sizes) > 0:
        output = interpolate_nd(x, get_coeffs, output_size=sizes, roi=roi,
                                coordinate_transformation_mode=coord_mode,
                                exclude_outside=exclude_outside,
                                extrapolation_value=extrapolation_value)
    return output


# ONNX Runtime implementation
def ref_resize_ort(x, roi, scales, sizes, mode, coord_mode, cubic_coeff_a,
                   exclude_outside, extrapolation_value, nearest_mode):
    # Use ONNX Runtime
    model = make_onnx_model(x.shape, roi, scales, sizes, mode, coord_mode,
                            cubic_coeff_a, exclude_outside, extrapolation_value,
                            nearest_mode)
    session = ort.InferenceSession(model.SerializeToString(),
                                   providers=["CPUExecutionProvider"])
    outputs = session.run(None, {
        "X": x,
        "roi": roi,
        "scales": scales,
        "sizes": sizes,
    })
    return outputs[0]


@pytest.mark.parametrize("ctx, func_name", ctxs)
@pytest.mark.parametrize("seed", [313])
@pytest.mark.parametrize("x_shape, scales, sizes", [
    # scales
    ([1, 1, 1], [1.0, 1.0, 0.5], []),
    ([1, 1, 1], [1.0, 1.0, 2.0], []),
    ([1, 2, 3], [1.0, 1.0, 2.0], []),
    ([1, 2, 4], [1.0, 1.0, 0.5], []),
    ([1, 2, 5], [1.0, 1.0, 1.3], []),
    ([1, 2, 6], [1.0, 1.0, 0.7], []),
    ([1, 2, 3, 4], [1.0, 1.0, 2.0, 0.5], []),
    ([1, 2, 5, 7], [1.0, 1.0, 0.7, 0.8], []),
    ([1, 2, 5, 7], [1.0, 1.0, 1.3, 1.7], []),
    ([1, 2, 5, 7], [1.0, 1.0, 1.5, 1.0], []),
    ([1, 2, 3, 4, 5], [1.0, 1.0, 2.0, 0.5, 1.5], []),
    ([1, 2, 3, 4, 6], [1.0, 1.0, 0.7, 1.4, 1.0], []),
    # sizes
    ([1, 1, 1], [], [1, 1, 3]),
    ([1, 2, 3], [], [1, 2, 6]),
    ([1, 2, 4], [], [1, 2, 2]),
    ([1, 2, 5], [], [1, 2, 7]),
    ([1, 2, 6], [], [1, 2, 5]),
    ([1, 2, 3, 4], [], [1, 2, 6, 2]),
    ([1, 2, 5, 7], [], [1, 2, 4, 6]),
    ([1, 2, 5, 7], [], [1, 2, 6, 8]),
    ([1, 2, 5, 7], [], [1, 2, 7, 7]),
    ([1, 2, 3, 4, 5], [], [1, 2, 6, 2, 7]),
    ([1, 2, 3, 4, 6], [], [1, 2, 2, 5, 6]),
])
@pytest.mark.parametrize("roi", [
    # 3D roi
    [0.00, 0.00, 0.00, 1.00, 1.00, 1.00],
    [0.10, 0.20, 0.30, 1.10, 1.20, 1.30],
])
@pytest.mark.parametrize("coord_mode", [
    "half_pixel",
    "pytorch_half_pixel", "align_corners", "asymmetric",
    "tf_half_pixel_for_nn", "tf_crop_and_resize"
])
@pytest.mark.parametrize("cubic_coeff_a", [-0.5])
@pytest.mark.parametrize("extrapolation_value", [10.0])
@pytest.mark.parametrize("mode, nearest_mode, exclude_outside", [
    ("nearest", "round_prefer_floor", 0),
    ("nearest", "round_prefer_ceil", 0),
    ("nearest", "floor", 0),
    ("nearest", "ceil", 0),
    ("linear", "round_prefer_floor", 0),
    ("cubic", "round_prefer_floor", 0),
    ("cubic", "round_prefer_floor", 1),
])
def test_onnx_resize_forward(
        seed, x_shape, roi, scales, sizes, mode, coord_mode, cubic_coeff_a,
        exclude_outside, extrapolation_value, nearest_mode, ctx, func_name):

    pytest.skip('This test needs onnx, test locally.')

    import onnx
    import onnxruntime as ort
    from onnx.backend.test.case.node.resize import (
        interpolate_nd, nearest_coeffs, linear_coeffs, cubic_coeffs)

    # Compute the number of resize dimension
    if len(scales) > 0:
        num_outer_dims = next((i for i, s in enumerate(scales) if s != 1.0), 0)
    if len(sizes) > 0:
        num_outer_dims = 0
        for i, (isize, osize) in enumerate(zip(x_shape, sizes)):
            if isize != osize:
                break
            num_outer_dims += 1
    num_resize_dims = len(x_shape) - num_outer_dims

    # Check ONNX Runtime limitation
    cubic_not_2d = mode == "cubic" and num_resize_dims != 2
    if cubic_not_2d and not USE_ONNX_REF:
        pytest.skip("ONNX Runtime only supports " +
                    "2D interpolation when mode is 'cubic'")
    use_onnx_ref = (cubic_not_2d and USE_ONNX_REF) or FORCE_ONNX_REF

    if use_onnx_ref and coord_mode == "tf_half_pixel_for_nn":
        pytest.skip("ONNX reference implementation does not support " +
                    "tf_half_pixel_for_nn")

    # Slice 3D roi to target dimensions
    roi_dims = len(roi) // 2
    ndim = len(x_shape)
    roi_sta = [0.0] * num_outer_dims
    roi_end = [1.0] * num_outer_dims
    roi_sta += roi[0*roi_dims+0:0*roi_dims+num_resize_dims]
    roi_end += roi[1*roi_dims+0:1*roi_dims+num_resize_dims]
    roi = roi_sta + roi_end

    # Prepare inputs
    rng = np.random.RandomState(seed)
    x = rng.rand(*x_shape).astype(np.float32)
    roi = np.array(roi, dtype=np.float32)
    scales = np.array(scales, dtype=np.float32)
    sizes = np.array(sizes, dtype=np.int64)

    # Run nnabla function
    vx = nn.Variable.from_numpy_array(x)
    with nn.context_scope(ctx), nn.auto_forward():
        vy = F.onnx_resize(vx, roi, scales, sizes, mode, coord_mode,
                           cubic_coeff_a, exclude_outside,
                           extrapolation_value, nearest_mode)

    # Run reference
    if use_onnx_ref:
        ref = ref_resize_onnx_ref(x, roi, scales, sizes, mode, coord_mode,
                                  cubic_coeff_a, exclude_outside,
                                  extrapolation_value, nearest_mode)
    else:
        ref = ref_resize_ort(x, roi, scales, sizes, mode, coord_mode,
                             cubic_coeff_a,
                             exclude_outside, extrapolation_value, nearest_mode)

    # Check results
    assert_allclose(vy.d, ref)
    assert func_name == vy.parent.name
