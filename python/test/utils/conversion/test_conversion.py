# Copyright (c) 2017 Sony Corporation. All Rights Reserved.
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

import os
import struct
import pytest
import nnabla
import nnabla.utils.load as nnload
import onnx
import numpy as np
import pdb
from collections import OrderedDict
import caffe2.python.onnx.backend as oc2
import cntk
import cntk.ops.functions as cntkf
from nnabla.utils.converter.nnabla import NnpReader, NnpExporter
from nnabla.utils.converter.onnx import OnnxReader, OnnxExporter, onnx_model_to_nnp_protobuf

TEST_DATA_DIR = "nnabla-sample-data/conversion_data"


def run_executor(nn_net, exec_name):
    """Run specified executor and return its network"""
    exe = nn_net.executors[exec_name]
    exe.network.forward(exe.forward_sequence)
    return exe.network


def convert_onnx_to_nnp_and_compare(
        tmpdir, onnx_dir, onnx_name, nnp_name, out_name, exec_name,
        backend="caffe2",
        in_img=None, in_name="", compare_values=True, show_onnx=False, show_nnp=False,
        show_output=False, atol=1e-08):
    """Convert specified ONNX to NNP and compare each results ran by Caffe2 and NNabla"""
    path = os.path.join(onnx_dir, onnx_name)
    backend_out = None
    if backend == "caffe2":
        # Process onnx with caffe2 backend
        model = onnx.load(path)
        if show_onnx:
            print(model)
        c2out = None
        if type(in_img) is np.ndarray:
            c2out = oc2.run_model(model, [in_img])
        else:
            c2out = oc2.run_model(model, [])
        backend_out = c2out[out_name]
    elif backend == "cntk":
        n = cntkf.Function.load(path, format=cntk.ModelFormat.ONNX)
        cntk_out = None
        if type(in_img) is np.ndarray:
            cntk_out = n.eval({in_name: in_img})
        else:
            cntk_out = n.eval()
        backend_out = cntk_out[0]
    else:
        raise ValueError("Unknown backend specified")
    # Process onnx with naabla
    r = OnnxReader(path)
    nnp = r.read()
    assert nnp is not None
    assert len(nnp.other_files) == 0
    assert nnp.protobuf is not None
    if show_nnp:
        print(nnp.protobuf)

    nnpex = NnpExporter(nnp, batch_size=0)
    nnpdir = tmpdir.mkdir("nnp")
    p = os.path.join(str(nnpdir), nnp_name)
    nnpex.export_nnp(p)
    # read exported nnp and run network
    # pdb.set_trace()
    nn_net = nnload.load([p])
    if type(in_img) is np.ndarray:
        net = nn_net.executors[exec_name].network
        in_data = net.variables[in_name]
        in_data.variable_instance.d = in_img
    exe = run_executor(nn_net, exec_name)
    # in_data = exe.variables["in_data_0"]
    # print(in_data.variable_instance.d)
    nnout = exe.variables[out_name].variable_instance.d
    # print(nnout.variable_instance.d)
    # Compare both naabla and backend results
    if show_output:
        print(backend_out, nnout)
    assert backend_out.shape == nnout.shape
    if compare_values:
        assert np.allclose(backend_out, nnout, atol=atol)


def convert_nnp_to_onnx_and_compare(
        tmpdir, nnp_dir, nnp_name, onnx_name, out_name, exec_name,
        in_img=None, in_name="", compare_values=True, show_nnp=False,
        show_onnx=False, show_output=False, atol=1e-08):
    """Convert specified NNP to ONNX and compare each results ran by Caffe2 and NNabla"""
    # Process nnp with nnabla
    path = os.path.join(nnp_dir, nnp_name)
    nn_net = nnload.load([path])
    if type(in_img) is np.ndarray:
        net = nn_net.executors[exec_name].network
        in_data = net.variables[in_name]
        in_data.variable_instance.d = in_img
    exe = run_executor(nn_net, exec_name)
    nnout = exe.variables[out_name].variable_instance.d

    # Convert nnp to ONNX
    r = NnpReader(path)
    nnp = r.read()
    assert nnp is not None
    assert len(nnp.other_files) == 0
    assert nnp.protobuf is not None
    if show_nnp:
        print(nnp.protobuf)
    onnxex = OnnxExporter(nnp)
    onnxdir = tmpdir.mkdir("onnx")
    p = os.path.join(str(onnxdir), onnx_name)
    onnxex.export(p)

    # read exported onnx and run network
    model = onnx.load(p)
    if show_onnx:
        print(model)
    #pdb.set_trace()
    c2out = None
    if type(in_img) is np.ndarray:
        c2out = oc2.run_model(model, [in_img])
    else:
        c2out = oc2.run_model(model, [])
    c2 = c2out[out_name]
    # Compare both naabla and caffe2 results
    if show_output:
        print(c2, nnout)
    assert c2.shape == nnout.shape
    if compare_values:
        assert np.allclose(c2, nnout, atol=atol)


@pytest.fixture
def nnp_fixture():
    # We need to remove all parameters for each test case
    # because the buffer shape will differ while having same names
    nnabla.clear_parameters()


def test_onnx_nnp_conversion_relu(tmpdir, nnp_fixture):
    convert_onnx_to_nnp_and_compare(
        tmpdir, TEST_DATA_DIR, "relu.onnx", "relu.nnp", "out_data_1", "exec_0")


def test_nnp_onnx_conversion_relu(tmpdir, nnp_fixture):
    convert_nnp_to_onnx_and_compare(
        tmpdir, TEST_DATA_DIR, "relu.nnp", "relu.onnx", "out_data_1", "exec_0")


def test_onnx_nnp_conversion_concat(tmpdir, nnp_fixture):
    convert_onnx_to_nnp_and_compare(
        tmpdir, TEST_DATA_DIR, "concat.onnx", "concat.nnp", "out_data_1", "exec_0")


def test_nnp_onnx_conversion_concat(tmpdir, nnp_fixture):
    convert_nnp_to_onnx_and_compare(
        tmpdir, TEST_DATA_DIR, "concat.nnp", "concat.onnx", "out_data_1", "exec_0")


def test_onnx_nnp_conversion_dropout(tmpdir, nnp_fixture):
    # We do not check if the values match because a dropout
    # output yield random results
    convert_onnx_to_nnp_and_compare(
        tmpdir, TEST_DATA_DIR, "dropout.onnx", "dropout.nnp", "out_data_1", "exec_0", compare_values=False)


def test_nnp_onnx_conversion_dropout(tmpdir, nnp_fixture):
    # We do not check if the values match because a dropout
    # output yield random results
    convert_nnp_to_onnx_and_compare(
        tmpdir, TEST_DATA_DIR, "dropout.nnp", "dropout.onnx", "out_data_1", "exec_0", compare_values=False)


def test_onnx_nnp_conversion_dropout_is_test(tmpdir, nnp_fixture):
    convert_onnx_to_nnp_and_compare(
        tmpdir, TEST_DATA_DIR, "dropout_test.onnx", "dropout_test.nnp", "out_data_1", "exec_0")


def test_nnp_onnx_conversion_dropout_is_test(tmpdir, nnp_fixture):
    convert_nnp_to_onnx_and_compare(
        tmpdir, TEST_DATA_DIR, "dropout_test.nnp", "dropout_test.onnx", "out_data_1", "exec_0")


def test_onnx_nnp_conversion_maxpool(tmpdir, nnp_fixture):
    convert_onnx_to_nnp_and_compare(
        tmpdir, TEST_DATA_DIR, "maxpool.onnx", "maxpool.nnp", "out_data_1", "exec_0")


def test_nnp_onnx_conversion_maxpool(tmpdir, nnp_fixture):
    convert_nnp_to_onnx_and_compare(
        tmpdir, TEST_DATA_DIR, "maxpool.nnp", "maxpool.onnx", "out_data_1", "exec_0")


def test_onnx_nnp_conversion_maxpool_p0_s2_k2(tmpdir, nnp_fixture):
    convert_onnx_to_nnp_and_compare(
        tmpdir, TEST_DATA_DIR, "maxpool_p0_s2_k2.onnx", "maxpool_p0_s2_k2.nnp", "out_data_1", "exec_0")


def test_nnp_onnx_conversion_maxpool_p0_s2_k2(tmpdir, nnp_fixture):
    convert_nnp_to_onnx_and_compare(
        tmpdir, TEST_DATA_DIR, "maxpool_p0_s2_k2.nnp", "maxpool_p0_s2_k2.onnx", "out_data_1", "exec_0")


def test_onnx_nnp_conversion_maxpool_p0_s2_k3(tmpdir, nnp_fixture):
    convert_onnx_to_nnp_and_compare(
        tmpdir, TEST_DATA_DIR, "maxpool_p0_s2_k3.onnx", "maxpool_p0_s2_k3.nnp", "out_data_1", "exec_0")


def test_nnp_onnx_conversion_maxpool_p0_s3_k3(tmpdir, nnp_fixture):
    convert_nnp_to_onnx_and_compare(
        tmpdir, TEST_DATA_DIR, "maxpool_p0_s2_k3.nnp", "maxpool_p0_s2_k3.onnx", "out_data_1", "exec_0")


def test_onnx_nnp_conversion_conv(tmpdir, nnp_fixture):
    convert_onnx_to_nnp_and_compare(
        tmpdir, TEST_DATA_DIR, "conv.onnx", "conv.nnp", "out_data_1", "exec_0")


def test_nnp_onnx_conversion_conv(tmpdir, nnp_fixture):
    convert_nnp_to_onnx_and_compare(
        tmpdir, TEST_DATA_DIR, "conv.nnp", "conv.onnx", "out_data_1", "exec_0")


def test_onnx_nnp_conversion_gap(tmpdir, nnp_fixture):
    convert_onnx_to_nnp_and_compare(
        tmpdir, TEST_DATA_DIR, "gap.onnx", "gap.nnp", "out_data_1", "exec_0")


def test_nnp_onnx_conversion_gap(tmpdir, nnp_fixture):
    convert_nnp_to_onnx_and_compare(
        tmpdir, TEST_DATA_DIR, "gap.nnp", "gap.onnx", "out_data_1", "exec_0")


def test_onnx_nnp_conversion_softmax(tmpdir, nnp_fixture):
    convert_onnx_to_nnp_and_compare(
        tmpdir, TEST_DATA_DIR, "softmax.onnx", "softmax.nnp", "out_data_1", "exec_0")


def test_nnp_onnx_conversion_softmax(tmpdir, nnp_fixture):
    convert_nnp_to_onnx_and_compare(
        tmpdir, TEST_DATA_DIR, "softmax.nnp", "softmax.onnx", "out_data_1", "exec_0")


def test_onnx_nnp_conversion_average_pool(tmpdir, nnp_fixture):
    convert_onnx_to_nnp_and_compare(
        tmpdir, TEST_DATA_DIR, "average_pool.onnx", "average_pool.nnp", "out_data_1", "exec_0")


def test_nnp_onnx_conversion_average_pool(tmpdir, nnp_fixture):
    convert_nnp_to_onnx_and_compare(
        tmpdir, TEST_DATA_DIR, "average_pool.nnp", "average_pool.onnx", "out_data_1", "exec_0")


def test_onnx_nnp_conversion_sum(tmpdir, nnp_fixture):
    convert_onnx_to_nnp_and_compare(
        tmpdir, TEST_DATA_DIR, "sum.onnx", "sum.nnp", "out_data_1", "exec_0")


def test_nnp_onnx_conversion_sum(tmpdir, nnp_fixture):
    convert_nnp_to_onnx_and_compare(
        tmpdir, TEST_DATA_DIR, "sum.nnp", "sum.onnx", "out_data_1", "exec_0")


def test_onnx_nnp_conversion_batch_normalization(tmpdir, nnp_fixture):
    convert_onnx_to_nnp_and_compare(
        tmpdir, TEST_DATA_DIR, "batch_norm.onnx", "batch_norm.nnp", "out_data_1", "exec_0", atol=1e-05)


def test_nnp_onnx_conversion_batch_normalization(tmpdir, nnp_fixture):
    convert_nnp_to_onnx_and_compare(
        tmpdir, TEST_DATA_DIR, "batch_norm.nnp", "batch_norm.onnx", "out_data_1", "exec_0", atol=1e-05)


def test_onnx_nnp_conversion_gemm(tmpdir, nnp_fixture):
    convert_onnx_to_nnp_and_compare(
        tmpdir, TEST_DATA_DIR, "gemm.onnx", "gemm.nnp", "out_data_1", "exec_0")


def test_nnp_onnx_conversion_gemm(tmpdir, nnp_fixture):
    convert_nnp_to_onnx_and_compare(
        tmpdir, TEST_DATA_DIR, "gemm.nnp", "gemm.onnx", "out_data_1", "exec_0")


def test_onnx_nnp_conversion_add_no_broadcast(tmpdir, nnp_fixture):
    convert_onnx_to_nnp_and_compare(
        tmpdir, TEST_DATA_DIR, "add_no_broadcast.onnx", "add_no_broadcast.nnp", "out_data_1", "exec_0")


def test_nnp_onnx_conversion_add_no_broadcast(tmpdir, nnp_fixture):
    convert_nnp_to_onnx_and_compare(
        tmpdir, TEST_DATA_DIR, "add_no_broadcast.nnp", "add_no_broadcast.onnx", "out_data_1", "exec_0")


def test_onnx_nnp_conversion_mul_no_broadcast(tmpdir, nnp_fixture):
    convert_onnx_to_nnp_and_compare(
        tmpdir, TEST_DATA_DIR, "mul_no_broadcast.onnx", "mul_no_broadcast.nnp", "out_data_1", "exec_0")


def test_nnp_onnx_conversion_mul_no_broadcast(tmpdir, nnp_fixture):
    convert_nnp_to_onnx_and_compare(
        tmpdir, TEST_DATA_DIR, "mul_no_broadcast.nnp", "mul_no_broadcast.onnx", "out_data_1", "exec_0")


def test_onnx_nnp_conversion_constant(tmpdir, nnp_fixture):
    convert_onnx_to_nnp_and_compare(
        tmpdir, TEST_DATA_DIR, "constant.onnx", "constant.nnp", "Pooling33_Output_0", "exec_0")


#def test_onnx_nnp_conversion_reshape(tmpdir, nnp_fixture):
#    convert_onnx_to_nnp_and_compare(
#        tmpdir, TEST_DATA_DIR, "reshape.onnx", "reshape.nnp", "out_data_1", "exec_0")


def test_onnx_nnp_conversion_matmul(tmpdir, nnp_fixture):
    convert_onnx_to_nnp_and_compare(
        tmpdir, TEST_DATA_DIR, "matmul.onnx", "matmul.nnp", "out_data_1", "exec_0")


def test_nnp_onnx_conversion_matmul(tmpdir, nnp_fixture):
    convert_nnp_to_onnx_and_compare(
        tmpdir, TEST_DATA_DIR, "matmul.nnp", "matmul.onnx", "out_data_1", "exec_0")


def test_onnx_nnp_conversion_transpose(tmpdir, nnp_fixture):
    convert_onnx_to_nnp_and_compare(
        tmpdir, TEST_DATA_DIR, "transpose.onnx", "transpose.nnp", "out_data_1", "exec_0")


def test_nnp_onnx_conversion_transpose(tmpdir, nnp_fixture):
    convert_nnp_to_onnx_and_compare(
        tmpdir, TEST_DATA_DIR, "transpose.nnp", "transpose.onnx", "out_data_1", "exec_0")


def test_onnx_nnp_conversion_abs(tmpdir, nnp_fixture):
    convert_onnx_to_nnp_and_compare(
        tmpdir, TEST_DATA_DIR, "abs.onnx", "abs.nnp", "out_data_1", "exec_0")


def test_nnp_onnx_conversion_abs(tmpdir, nnp_fixture):
    convert_nnp_to_onnx_and_compare(
        tmpdir, TEST_DATA_DIR, "abs.nnp", "abs.onnx", "out_data_1", "exec_0")


def test_onnx_nnp_conversion_sigmoid(tmpdir, nnp_fixture):
    convert_onnx_to_nnp_and_compare(
        tmpdir, TEST_DATA_DIR, "sigmoid.onnx", "sigmoid.nnp", "out_data_1", "exec_0")


def test_nnp_onnx_conversion_sigmoid(tmpdir, nnp_fixture):
    convert_nnp_to_onnx_and_compare(
        tmpdir, TEST_DATA_DIR, "sigmoid.nnp", "sigmoid.onnx", "out_data_1", "exec_0")


def test_onnx_nnp_conversion_tanh(tmpdir, nnp_fixture):
    convert_onnx_to_nnp_and_compare(
        tmpdir, TEST_DATA_DIR, "tanh.onnx", "tanh.nnp", "out_data_1", "exec_0")


def test_nnp_onnx_conversion_tanh(tmpdir, nnp_fixture):
    convert_nnp_to_onnx_and_compare(
        tmpdir, TEST_DATA_DIR, "tanh.nnp", "tanh.onnx", "out_data_1", "exec_0")


def test_onnx_nnp_conversion_leaky_relu(tmpdir, nnp_fixture):
    convert_onnx_to_nnp_and_compare(
        tmpdir, TEST_DATA_DIR, "leaky_relu.onnx", "leaky_relu.nnp", "out_data_1", "exec_0")


def test_nnp_onnx_conversion_leaky_relu(tmpdir, nnp_fixture):
    convert_nnp_to_onnx_and_compare(
        tmpdir, TEST_DATA_DIR, "leaky_relu.nnp", "leaky_relu.onnx", "out_data_1", "exec_0")


def test_onnx_nnp_conversion_log(tmpdir, nnp_fixture):
    convert_onnx_to_nnp_and_compare(
        tmpdir, TEST_DATA_DIR, "log.onnx", "log.nnp", "out_data_1", "exec_0")


def test_nnp_onnx_conversion_log(tmpdir, nnp_fixture):
    convert_nnp_to_onnx_and_compare(
        tmpdir, TEST_DATA_DIR, "log.nnp", "log.onnx", "out_data_1", "exec_0")


def test_onnx_nnp_conversion_not(tmpdir, nnp_fixture):
    convert_onnx_to_nnp_and_compare(
        tmpdir, TEST_DATA_DIR, "not.onnx", "not.nnp", "out_data_1", "exec_0")


def test_nnp_onnx_conversion_not(tmpdir, nnp_fixture):
    convert_nnp_to_onnx_and_compare(
        tmpdir, TEST_DATA_DIR, "not.nnp", "not.onnx", "out_data_1", "exec_0")


def test_onnx_nnp_conversion_elu(tmpdir, nnp_fixture):
    convert_onnx_to_nnp_and_compare(
        tmpdir, TEST_DATA_DIR, "elu.onnx", "elu.nnp", "out_data_1", "exec_0")


def test_nnp_onnx_conversion_elu(tmpdir, nnp_fixture):
    convert_nnp_to_onnx_and_compare(
        tmpdir, TEST_DATA_DIR, "elu.nnp", "elu.onnx", "out_data_1", "exec_0")


def test_onnx_nnp_conversion_selu(tmpdir, nnp_fixture):
    convert_onnx_to_nnp_and_compare(
        tmpdir, TEST_DATA_DIR, "selu.onnx", "selu.nnp", "out_data_1", "exec_0")


def test_nnp_onnx_conversion_selu(tmpdir, nnp_fixture):
    convert_nnp_to_onnx_and_compare(
        tmpdir, TEST_DATA_DIR, "selu.nnp", "selu.onnx", "out_data_1", "exec_0")


def test_onnx_nnp_conversion_reduce_sum(tmpdir, nnp_fixture):
    convert_onnx_to_nnp_and_compare(
        tmpdir, TEST_DATA_DIR, "reduce_sum.onnx", "reduce_sum.nnp", "out_data_1", "exec_0")


def test_nnp_onnx_conversion_reduce_sum(tmpdir, nnp_fixture):
    convert_nnp_to_onnx_and_compare(
        tmpdir, TEST_DATA_DIR, "reduce_sum.nnp", "reduce_sum.onnx", "out_data_1", "exec_0")


def test_onnx_nnp_conversion_reduce_mean(tmpdir, nnp_fixture):
    convert_onnx_to_nnp_and_compare(
        tmpdir, TEST_DATA_DIR, "reduce_mean.onnx", "reduce_mean.nnp", "out_data_1", "exec_0")


def test_nnp_onnx_conversion_reduce_mean(tmpdir, nnp_fixture):
    convert_nnp_to_onnx_and_compare(
        tmpdir, TEST_DATA_DIR, "reduce_mean.nnp", "reduce_mean.onnx", "out_data_1", "exec_0")


# These following tests are invalidated due to a
# backend bug? decribed in the following issue:
# https://github.com/Microsoft/CNTK/issues/3127
#def test_onnx_nnp_conversion_reduce_min(tmpdir, nnp_fixture):
#    convert_onnx_to_nnp_and_compare(
#        tmpdir, TEST_DATA_DIR, "reduce_min.onnx", "reduce_min.nnp",
#        "ReduceElements7_Output_0", "exec_0",
#        backend="cntk")
#
#
#def test_onnx_nnp_conversion_reduce_max(tmpdir, nnp_fixture):
#    convert_onnx_to_nnp_and_compare(
#        tmpdir, TEST_DATA_DIR, "reduce_max.onnx", "reduce_max.nnp",
#        "ReduceElements7_Output_0", "exec_0",
#        backend="cntk")
#
#def test_onnx_nnp_conversion_reduce_prod(tmpdir, nnp_fixture):
#    convert_onnx_to_nnp_and_compare(
#        tmpdir, TEST_DATA_DIR, "reduce_prod.onnx", "reduce_prod.nnp",
#        "ReduceElements7_Output_0", "exec_0",
#        backend="cntk")

def test_onnx_nnp_conversion_squeezenet(tmpdir, nnp_fixture):
    img = np.random.rand(1, 3, 224, 224).astype(np.float32)
    convert_onnx_to_nnp_and_compare(
        tmpdir, TEST_DATA_DIR, "squeezenet.onnx", "squeezenet.nnp", "softmaxout_1", "exec_0",
        in_name="data_0", in_img=img)


def test_nnp_onnx_conversion_squeezenet(tmpdir, nnp_fixture):
    img = np.random.rand(1, 3, 224, 224).astype(np.float32)
    convert_nnp_to_onnx_and_compare(
        tmpdir, TEST_DATA_DIR, "squeezenet.nnp", "squeezenet.onnx", "softmaxout_1", "exec_0",
        in_name="data_0", in_img=img)


@pytest.mark.slow
def test_onnx_nnp_conversion_resnet50(tmpdir, nnp_fixture):
    img = np.random.rand(1, 3, 224, 224).astype(np.float32)
    convert_onnx_to_nnp_and_compare(
        tmpdir, TEST_DATA_DIR, "resnet50.onnx", "resnet50.nnp", "gpu_0/softmax_1", "exec_0",
        in_name="gpu_0/data_0", in_img=img, atol=1e-5)


@pytest.mark.slow
def test_nnp_onnx_conversion_resnet50(tmpdir, nnp_fixture):
    img = np.random.rand(1, 3, 224, 224).astype(np.float32)
    convert_nnp_to_onnx_and_compare(
        tmpdir, TEST_DATA_DIR, "resnet50.nnp", "resnet50.onnx", "gpu_0/softmax_1", "exec_0",
        in_name="gpu_0/data_0", in_img=img, atol=1e-5)


@pytest.mark.slow
def test_onnx_nnp_conversion_vgg19(tmpdir, nnp_fixture):
    img = np.random.rand(1, 3, 224, 224).astype(np.float32)
    convert_onnx_to_nnp_and_compare(
        tmpdir, TEST_DATA_DIR, "vgg19.onnx", "vgg19.nnp", "prob_1", "exec_0",
        in_name="data_0", in_img=img)


@pytest.mark.slow
def test_nnp_onnx_conversion_vgg19(tmpdir, nnp_fixture):
    img = np.random.rand(1, 3, 224, 224).astype(np.float32)
    convert_nnp_to_onnx_and_compare(
        tmpdir, TEST_DATA_DIR, "vgg19.nnp", "vgg19.onnx", "prob_1", "exec_0",
        in_name="data_0", in_img=img)
