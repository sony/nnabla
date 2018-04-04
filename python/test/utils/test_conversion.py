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
import onnx_caffe2.backend
from nnabla.utils.converter.nnabla import NnpReader, NnpExporter
from nnabla.utils.converter.onnx import OnnxReader, OnnxExporter, onnx_model_to_nnp_protobuf

TEST_DATA_DIR="nnabla-sample-data/conversion_data"

def run_executor(nn_net, exec_name):
    """Run specified executor and return its network"""
    exe = nn_net.executors[exec_name]
    exe.network.forward(exe.forward_sequence)
    return exe.network


def convert_onnx_to_nnp_and_compare(
        tmpdir, onnx_dir, onnx_name, nnp_name, out_name, exec_name,
        compare_values=True, show_onnx=False, show_nnp=False, show_output=False, atol=1e-08):
    """Convert specified ONNX to NNP and compare each results ran by Caffe2 and NNabla"""
    path = os.path.join(onnx_dir, onnx_name)
    # Process onnx with caffe2 backend
    model = onnx.load(path)
    if show_onnx:
        print(model)
    c2out = onnx_caffe2.backend.run_model(model, [])
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
    exe = run_executor(nn_net, exec_name)
    #in_data = exe.variables["in_data_0"]
    #print(in_data.variable_instance.d)
    nnout = exe.variables[out_name].variable_instance.d
    #print(nnout.variable_instance.d)
    # Compare both naabla and caffe2 results
    c2 = c2out[out_name]
    if show_output:
        print(c2, nnout)
    assert c2.shape == nnout.shape
    if compare_values:
        assert np.allclose(c2, nnout, atol=atol)

def convert_nnp_to_onnx_and_compare(
        tmpdir, nnp_dir, nnp_name, onnx_name, out_name, exec_name,
        compare_values=True, show_nnp=False, show_onnx=False, show_output=False, atol=1e-08):
    """Convert specified NNP to ONNX and compare each results ran by Caffe2 and NNabla"""
    # Process nnp with nnabla
    path = os.path.join(nnp_dir, nnp_name)
    nn_net = nnload.load([path])
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
    c2out = onnx_caffe2.backend.run_model(model, [])
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

def test_onnx_nnp_conversion_maxpool_no_pad(tmpdir, nnp_fixture):
    convert_onnx_to_nnp_and_compare(
        tmpdir, TEST_DATA_DIR, "maxpool_no_pad.onnx", "maxpool_no_pad.nnp", "out_data_1", "exec_0")

def test_nnp_onnx_conversion_maxpool_no_pad(tmpdir, nnp_fixture):
    convert_nnp_to_onnx_and_compare(
        tmpdir, TEST_DATA_DIR, "maxpool_no_pad.nnp", "maxpool_no_pad.onnx", "out_data_1", "exec_0")

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

#def test_onnx_nnp_conversion_constant(tmpdir, nnp_fixture):
#    convert_onnx_to_nnp_and_compare(
#        tmpdir, TEST_DATA_DIR, "constant.onnx", "constant.nnp", "Pooling33_Output_0", "exec_0")

def test_onnx_nnp_conversion_squeezenet(tmpdir, nnp_fixture):
    onnx_dir = TEST_DATA_DIR
    onnx_name = "squeezenet.onnx"
    nnp_name = "squeezenet.nnp"
    out_name = "softmaxout_1"
    exec_name = "exec_0"
    in_name = "data_0"
    show_onnx = False
    show_nnp = False
    show_output = False
    path = os.path.join(onnx_dir, onnx_name)
    # Process onnx with caffe2 backend
    model = onnx.load(path)
    if show_onnx:
        print(model)
    img = np.random.rand(1,3,224,224).astype(np.float32)
    c2out = onnx_caffe2.backend.run_model(model, [img])
    # Process onnx with naabla
    nnp = onnx_model_to_nnp_protobuf(model)
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
    nn_net = nnload.load([p])
    #pdb.set_trace()
    # set input data and run inference
    net = nn_net.executors[exec_name].network
    in_data = net.variables[in_name]
    in_data.variable_instance.d = img
    net = run_executor(nn_net, exec_name)
    #in_data = exe.variables["in_data_0"]
    #print(in_data.variable_instance.d)
    nnout = net.variables[out_name].variable_instance.d
    #print(nnout.variable_instance.d)

    # Print all the intermediate buffer shape in order
    #for k, v in net.functions.items():
    #    out = v.outputs[0]
    #    print(out.name, net.variables[out.name].variable_instance.shape)
    # Compare both naabla and caffe2 results
    c2 = c2out[out_name]
    if show_output:
        print(c2, nnout)
    assert np.allclose(c2, nnout)

def test_nnp_onnx_conversion_squeezenet(tmpdir, nnp_fixture):
    nnp_dir = TEST_DATA_DIR
    onnx_name = "squeezenet.onnx"
    nnp_name = "squeezenet.nnp"
    out_name = "softmaxout_1"
    exec_name = "exec_0"
    in_name = "data_0"
    show_onnx = False
    show_nnp = False
    show_output = False
    # Process nnp with nnabla
    path = os.path.join(nnp_dir, nnp_name)
    nn_net = nnload.load([path])
    net = nn_net.executors[exec_name].network
    in_data = net.variables[in_name]
    img = np.random.rand(1,3,224,224).astype(np.float32)
    in_data.variable_instance.d = img
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
    c2out = onnx_caffe2.backend.run_model(model, [img])
    c2 = c2out[out_name]
    # Compare both naabla and caffe2 results
    if show_output:
        print(c2, nnout)
    assert np.allclose(c2, nnout)

def test_onnx_nnp_conversion_resnet50(tmpdir, nnp_fixture):
    onnx_dir = TEST_DATA_DIR
    onnx_name = "resnet50.onnx"
    nnp_name = "resnet50.nnp"
    out_name = "gpu_0/softmax_1"
    exec_name = "exec_0"
    in_name = "gpu_0/data_0"
    show_onnx = False
    show_nnp = False
    show_output = False
    path = os.path.join(onnx_dir, onnx_name)
    # Process onnx with caffe2 backend
    model = onnx.load(path)
    if show_onnx:
        print(model)
    img = np.random.rand(1, 3, 224, 224).astype(np.float32)
    rep = onnx_caffe2.backend.prepare(model)
    c2out = rep.run([img])
    # Process onnx with naabla
    nnp = onnx_model_to_nnp_protobuf(model)
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
    nn_net = nnload.load([p])
    #pdb.set_trace()
    # set input data and run inference
    net = nn_net.executors[exec_name].network
    in_data = net.variables[in_name]
    in_data.variable_instance.d = img
    net = run_executor(nn_net, exec_name)
    #in_data = exe.variables["in_data_0"]
    #print(in_data.variable_instance.d)
    nnout = net.variables[out_name].variable_instance.d
    #print(nnout.variable_instance.d)

    # Print all the intermediate buffer shape in order
    #for k, v in net.functions.items():
    #    out = v.outputs[0]
    #    print(out.name, net.variables[out.name].variable_instance.shape)
    # Compare both naabla and caffe2 results
    c2 = c2out[out_name]
    if show_output:
        print(c2, nnout)
    assert np.allclose(c2, nnout, atol=1e-5)

def test_nnp_onnx_conversion_resnet50(tmpdir, nnp_fixture):
    nnp_dir = TEST_DATA_DIR
    onnx_name = "resnet50.onnx"
    nnp_name = "resnet50.nnp"
    out_name = "gpu_0/softmax_1"
    exec_name = "exec_0"
    in_name = "gpu_0/data_0"
    show_onnx = False
    show_nnp = False
    show_output = False
    # Process nnp with nnabla
    path = os.path.join(nnp_dir, nnp_name)
    nn_net = nnload.load([path])
    net = nn_net.executors[exec_name].network
    in_data = net.variables[in_name]
    img = np.random.rand(1,3,224,224).astype(np.float32)
    in_data.variable_instance.d = img
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
    c2out = onnx_caffe2.backend.run_model(model, [img])
    c2 = c2out[out_name]
    # Compare both naabla and caffe2 results
    if show_output:
        print(c2, nnout)
    assert np.allclose(c2, nnout, atol=1e-5)
