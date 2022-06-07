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

import onnx
import onnxruntime as ort

ctxs = list_context('ONNXNonMaxSuppression')


def make_onnx_model(boxes_shape, scores_shape, center_point_box):
    import onnx.helper as oh
    from onnx import TensorProto

    node = oh.make_node(
        'NonMaxSuppression',
        inputs=['boxes', 'scores', 'max_output_boxes_per_class',
                'iou_threshold', 'score_threshold'],
        outputs=['selected_indices'],
        center_point_box=center_point_box
    )
    y_shape = ["num_selected_boxes", 3]
    inputs = [
        oh.make_tensor_value_info("boxes", TensorProto.FLOAT, boxes_shape),
        oh.make_tensor_value_info("scores", TensorProto.FLOAT, scores_shape),
        oh.make_tensor_value_info("max_output_boxes_per_class",
                                  TensorProto.INT64, []),
        oh.make_tensor_value_info("iou_threshold", TensorProto.FLOAT, []),
        oh.make_tensor_value_info("score_threshold", TensorProto.FLOAT, [])
    ]
    outputs = [
        oh.make_tensor_value_info("selected_indices", TensorProto.INT64,
                                  y_shape)
    ]
    graph = oh.make_graph([node], "non_max_suppression_test", inputs, outputs)
    opset = onnx.OperatorSetIdProto()
    opset.version = 11
    model = oh.make_model(graph, opset_imports=[opset])
    return model.SerializeToString()


def ref_non_max_suppression(boxes, scores, center_point_box,
                            max_output_boxes_per_class, iou_threshold,
                            score_threshold):
    model = make_onnx_model(boxes.shape, scores.shape, center_point_box)
    session = ort.InferenceSession(model, providers=["CPUExecutionProvider"])
    outputs = session.run(None, {
        "boxes": boxes,
        "scores": scores,
        "max_output_boxes_per_class": [max_output_boxes_per_class],
        "iou_threshold": [iou_threshold],
        "score_threshold": [score_threshold]
    })
    return outputs[0]


@pytest.mark.parametrize("ctx, func_name", ctxs)
@pytest.mark.parametrize("seed", [313])
@pytest.mark.parametrize("center_point_box", [0, 1])
@pytest.mark.parametrize("max_output_boxes", [0, 1, 13])
@pytest.mark.parametrize("iou_threshold", [0.0, 0.1, 0.5, 1.0])
@pytest.mark.parametrize("score_threshold", [0.4])
@pytest.mark.parametrize("batch_size, num_boxes, num_classes", [
    (1, 1, 1),
    (7, 1, 1),
    (1, 7, 1),
    (1, 1, 7),
    (2, 3, 4),
    (3, 129, 65),
])
def test_onnx_non_max_suppression_forward(
        seed, batch_size, num_boxes, num_classes, center_point_box,
        max_output_boxes, iou_threshold, score_threshold, ctx, func_name):
    rng = np.random.RandomState(seed)
    boxes = rng.rand(batch_size, num_boxes, 4).astype(np.float32)
    scores = rng.rand(batch_size, num_classes, num_boxes).astype(np.float32)

    vboxes = nn.Variable.from_numpy_array(boxes)
    vscores = nn.Variable.from_numpy_array(scores)
    with nn.context_scope(ctx), nn.auto_forward():
        voutput = F.onnx_non_max_suppression(vboxes, vscores, center_point_box,
                                             max_output_boxes, iou_threshold,
                                             score_threshold)

    ref = ref_non_max_suppression(boxes, scores, center_point_box,
                                  max_output_boxes, iou_threshold,
                                  score_threshold)

    assert_allclose(voutput.d, ref)
    assert func_name == voutput.parent.name
