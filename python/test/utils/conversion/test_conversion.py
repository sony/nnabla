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

from nnabla.utils.converter.nnabla import NnpImporter, NnpExporter
from collections import OrderedDict
import os
import shutil
import pytest
import nnabla
import nnabla.utils.load as nnload
from nnabla.testing import assert_allclose

import numpy as np
CAFFE2_AVAILABLE = False
try:
    import caffe2.python.onnx.backend as oc2
    CAFFE2_AVAILABLE = True
except:
    pass

ONNX_AVAILABLE = False
try:
    import onnx
    ONNX_AVAILABLE = True
except:
    pass


try:
    from nnabla.utils.converter.onnx import OnnxImporter, OnnxExporter
except:
    ONNX_AVAILABLE = False


CNTK_AVAILABLE = False
try:
    import cntk
    import cntk.ops.functions as cntkf
    CNTK_AVAILABLE = True
except:
    print('Need to install CNTK for testing.')

# The directory of which the input ONNX files will be at
REFERENCE_AVAILABLE = False
TEST_DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(
    __file__), '..', '..', '..', '..', '..', 'nnabla-sample-data', 'reference', 'onnx'))
if os.path.exists(TEST_DATA_DIR):
    REFERENCE_AVAILABLE = True
print(TEST_DATA_DIR)

# Set a path to this parameter (preferably the same as TEST_DATA_DIR)
# if you want to update all the NNP files
DEFAULT_NNP_EXPORT_PATH = None

# Currently, the onnx exporter converter only supports opset_6 and opset_9 versions.
ONNX_EXPORT_OPSET_VERSION = ["6", "9"]
DEFAULT_SUPPORT_OPSET_VERSION = "6"

try:
    from .gen_report import gen_report
except:
    print("Cannot generate report since no gen_report module.")

    def gen_report(import_result, export_result):
        pass

import_result = {}
export_result = {}


def print_buffer_shape(net):
    for k, v in net.functions.items():
        out = v.outputs[0]
        print(out.name, net.variables[out.name].variable_instance.shape)


def run_executor(nn_net, exec_name):
    """Run specified executor and return its network"""
    exe = nn_net.executors[exec_name]
    exe.network.forward(exe.forward_sequence)
    return exe.network


def convert_onnx_to_nnp_and_compare(
        tmpdir, onnx_dir, onnx_name, nnp_name, out_name, exec_name,
        backend="caffe2",
        in_img=None, in_name="",
        compare_values=True, show_onnx=False, show_nnp=False,
        show_output=False, atol=1e-08,
        export_nnp_path=DEFAULT_NNP_EXPORT_PATH):
    """Convert specified ONNX to NNP and compare each
    results ran by Caffe2 and NNabla"""
    path = os.path.join(onnx_dir, onnx_name)
    backend_out = None
    func_name = ''
    if backend == "caffe2" and CAFFE2_AVAILABLE:
        # Process onnx with caffe2 backend
        model = onnx.load(path)
        func_name = model.graph.node[0].op_type
        if show_onnx:
            print(model)
        c2out = None
        rep = oc2.prepare(model)
        if type(in_img) is np.ndarray:
            c2out = rep.run([in_img])
        else:
            c2out = rep.run([])
        # for k in rep.workspace.Blobs():
        #     v = rep.workspace.FetchBlob(k)
        #     print(k, v.shape)
        backend_out = c2out[out_name]
    elif backend == "cntk" and CNTK_AVAILABLE:
        n = cntkf.Function.load(path, format=cntk.ModelFormat.ONNX)
        cntk_out = None
        if type(in_img) is np.ndarray:
            cntk_out = n.eval({in_name: in_img})
        else:
            cntk_out = n.eval()
        backend_out = cntk_out
    else:
        raise ValueError("Unknown backend specified")
    import_result[func_name] = 'NG'

    # Process onnx with naabla
    r = OnnxImporter(path)
    nnp = r.execute()
    assert nnp is not None
    assert len(nnp.other_files) == 0
    assert nnp.protobuf is not None
    if show_nnp:
        print(nnp.protobuf)

    nnpex = NnpExporter(nnp, batch_size=0)
    nnpdir = tmpdir.mkdir("nnp")
    p = os.path.join(str(nnpdir), nnp_name)
    nnpex._export_nnp(p)
    if export_nnp_path:
        shutil.copy2(p, export_nnp_path)
    # read exported nnp and run network
    # pdb.set_trace()
    nn_net = nnload.load([p])
    if type(in_img) is np.ndarray:
        net = nn_net.executors[exec_name].network
        in_data = net.variables[in_name]
        in_data.variable_instance.d = in_img
    exe = run_executor(nn_net, exec_name)
    # print_buffer_shape(exe)
    # in_data = exe.variables["in_data_0"]
    # print(in_data.variable_instance.d)
    nnout = exe.variables[out_name].variable_instance.d
    # print(nnout.variable_instance.d)
    # Compare both naabla and backend results
    if show_output:
        print(backend_out, nnout)
    assert backend_out.shape == nnout.shape
    if compare_values:
        assert_allclose(backend_out, nnout, atol=atol)
        import_result[func_name] = 'OK'


def convert_nnp_to_onnx_and_compare(
        tmpdir, nnp_dir, nnp_name, onnx_name, out_name, exec_name,
        backend="caffe2",
        in_img=None, in_name="", compare_values=True, show_nnp=False,
        show_onnx=False, show_output=False, atol=1e-08,
        export_onnx_path=None):
    """Convert specified NNP to ONNX and compare
    each results ran by CNTK and NNabla"""
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
    r = NnpImporter(path)
    nnp = r.execute()
    assert nnp is not None
    assert len(nnp.other_files) == 0
    assert nnp.protobuf is not None
    if show_nnp:
        print(nnp.protobuf)
    func_name = nnp.protobuf.network[0].function[0].type
    print(func_name)
    export_result[func_name] = {}
    for opset in ONNX_EXPORT_OPSET_VERSION:
        export_result[func_name][opset] = 'NG'
        onnxex = OnnxExporter(nnp, -1, opset)
        onnxdir = tmpdir.mkdir("onnx")
        p = os.path.join(str(onnxdir), onnx_name)
        onnxex.execute(p)
        if export_onnx_path:
            shutil.copy2(p, export_onnx_path)

        # read exported onnx and run network
        model = onnx.load(p)
        if show_onnx:
            print(model)
        # pdb.set_trace()
        backend_out = None
        try:
            if backend == "caffe2" and CAFFE2_AVAILABLE:
                # Process onnx with caffe2 backend
                c2out = None
                rep = oc2.prepare(model)
                if type(in_img) is np.ndarray:
                    c2out = rep.run([in_img])
                else:
                    c2out = rep.run([])
                # for k in rep.workspace.Blobs():
                #     v = rep.workspace.FetchBlob(k)
                #     print(k, v.shape)
                backend_out = c2out[out_name]
            elif backend == "cntk" and CNTK_AVAILABLE:
                n = cntkf.Function.load(p, format=cntk.ModelFormat.ONNX)
                cntk_out = None
                if type(in_img) is np.ndarray:
                    cntk_out = n.eval({in_name: in_img})
                else:
                    cntk_out = n.eval()
                backend_out = cntk_out
            else:
                raise ValueError("Unknown backend specified")
        except:
            if opset == DEFAULT_SUPPORT_OPSET_VERSION:
                raise ValueError("Process onnx with caffe2 backend failed!!!")
            else:
                print("Process onnx with caffe2 backend failed!!!")
                return
        # Compare both naabla and backend out results
        if show_output:
            print(backend_out, nnout)
        assert backend_out.shape == nnout.shape
        if compare_values:
            assert_allclose(backend_out, nnout, atol=atol)
            export_result[func_name][opset] = 'OK'
        onnxdir.remove()


@pytest.fixture
def nnp_fixture():
    # We need to remove all parameters for each test case
    # because the buffer shape will differ while having same names
    nnabla.clear_parameters()


@pytest.fixture(scope="module", autouse=True)
def gen_report_fixture():
    yield
    gen_report(import_result, export_result)


def test_onnx_nnp_conversion_concat(tmpdir, nnp_fixture):
    if not (REFERENCE_AVAILABLE and ONNX_AVAILABLE and CAFFE2_AVAILABLE):
        pytest.skip('CAFFE2 does not installed.')
    convert_onnx_to_nnp_and_compare(tmpdir, TEST_DATA_DIR,
                                    "concat.onnx", "concat.nnp",
                                    "out_data_1", "exec_0")


def test_nnp_onnx_conversion_concat(tmpdir, nnp_fixture):
    if not (REFERENCE_AVAILABLE and ONNX_AVAILABLE and CAFFE2_AVAILABLE):
        pytest.skip('CAFFE2 does not installed.')
    convert_nnp_to_onnx_and_compare(tmpdir, TEST_DATA_DIR,
                                    "concat.nnp", "concat.onnx",
                                    "out_data_1", "exec_0")


def test_onnx_nnp_conversion_conv(tmpdir, nnp_fixture):
    if not (REFERENCE_AVAILABLE and ONNX_AVAILABLE and CAFFE2_AVAILABLE):
        pytest.skip('CAFFE2 does not installed.')
    convert_onnx_to_nnp_and_compare(
        tmpdir, TEST_DATA_DIR, "conv.onnx", "conv.nnp", "out_data_1", "exec_0")


def test_nnp_onnx_conversion_conv(tmpdir, nnp_fixture):
    if not (REFERENCE_AVAILABLE and ONNX_AVAILABLE and CAFFE2_AVAILABLE):
        pytest.skip('CAFFE2 does not installed.')
    convert_nnp_to_onnx_and_compare(
        tmpdir, TEST_DATA_DIR, "conv.nnp", "conv.onnx", "out_data_1", "exec_0")


def test_onnx_nnp_conversion_dropout(tmpdir, nnp_fixture):
    if not (REFERENCE_AVAILABLE and ONNX_AVAILABLE and CAFFE2_AVAILABLE):
        pytest.skip('CAFFE2 does not installed.')
    # We do not check if the values match because a dropout
    # output yield random results
    convert_onnx_to_nnp_and_compare(tmpdir, TEST_DATA_DIR,
                                    "dropout.onnx", "dropout.nnp",
                                    "out_data_1", "exec_0",
                                    compare_values=False)


def test_nnp_onnx_conversion_dropout(tmpdir, nnp_fixture):
    if not (REFERENCE_AVAILABLE and ONNX_AVAILABLE and CAFFE2_AVAILABLE):
        pytest.skip('CAFFE2 does not installed.')
    # We do not check if the values match because a dropout
    # output yield random results
    convert_nnp_to_onnx_and_compare(tmpdir, TEST_DATA_DIR,
                                    "dropout.nnp", "dropout.onnx",
                                    "out_data_1", "exec_0",
                                    compare_values=False)


def test_onnx_nnp_conversion_dropout_is_test(tmpdir, nnp_fixture):
    if not (REFERENCE_AVAILABLE and ONNX_AVAILABLE and CAFFE2_AVAILABLE):
        pytest.skip('CAFFE2 does not installed.')
    convert_onnx_to_nnp_and_compare(tmpdir, TEST_DATA_DIR,
                                    "dropout_test.onnx", "dropout_test.nnp",
                                    "out_data_1", "exec_0")


def test_nnp_onnx_conversion_dropout_is_test(tmpdir, nnp_fixture):
    if not (REFERENCE_AVAILABLE and ONNX_AVAILABLE and CAFFE2_AVAILABLE):
        pytest.skip('CAFFE2 does not installed.')
    convert_nnp_to_onnx_and_compare(tmpdir, TEST_DATA_DIR,
                                    "dropout_test.nnp", "dropout_test.onnx",
                                    "out_data_1", "exec_0")


def test_onnx_nnp_conversion_gap(tmpdir, nnp_fixture):
    if not (REFERENCE_AVAILABLE and ONNX_AVAILABLE and CAFFE2_AVAILABLE):
        pytest.skip('CAFFE2 does not installed.')
    convert_onnx_to_nnp_and_compare(
        tmpdir, TEST_DATA_DIR, "gap.onnx", "gap.nnp", "out_data_1", "exec_0")


def test_nnp_onnx_conversion_gap(tmpdir, nnp_fixture):
    if not (REFERENCE_AVAILABLE and ONNX_AVAILABLE and CAFFE2_AVAILABLE):
        pytest.skip('CAFFE2 does not installed.')
    convert_nnp_to_onnx_and_compare(
        tmpdir, TEST_DATA_DIR, "gap.nnp", "gap.onnx", "out_data_1", "exec_0")


def test_onnx_nnp_conversion_maxpool(tmpdir, nnp_fixture):
    if not (REFERENCE_AVAILABLE and ONNX_AVAILABLE and CAFFE2_AVAILABLE):
        pytest.skip('CAFFE2 does not installed.')
    convert_onnx_to_nnp_and_compare(tmpdir, TEST_DATA_DIR,
                                    "maxpool.onnx", "maxpool.nnp",
                                    "out_data_1", "exec_0")


def test_nnp_onnx_conversion_maxpool(tmpdir, nnp_fixture):
    if not (REFERENCE_AVAILABLE and ONNX_AVAILABLE and CAFFE2_AVAILABLE):
        pytest.skip('CAFFE2 does not installed.')
    convert_nnp_to_onnx_and_compare(tmpdir, TEST_DATA_DIR,
                                    "maxpool.nnp", "maxpool.onnx",
                                    "out_data_1", "exec_0")


def test_onnx_nnp_conversion_maxpool_p0_s2_k3(tmpdir, nnp_fixture):
    if not (REFERENCE_AVAILABLE and ONNX_AVAILABLE and CAFFE2_AVAILABLE):
        pytest.skip('CAFFE2 does not installed.')
    convert_onnx_to_nnp_and_compare(tmpdir, TEST_DATA_DIR,
                                    "maxpool_p0_s2_k3.onnx",
                                    "maxpool_p0_s2_k3.nnp",
                                    "out_data_1", "exec_0")


def test_nnp_onnx_conversion_maxpool_p0_s3_k3(tmpdir, nnp_fixture):
    if not (REFERENCE_AVAILABLE and ONNX_AVAILABLE and CAFFE2_AVAILABLE):
        pytest.skip('CAFFE2 does not installed.')
    convert_nnp_to_onnx_and_compare(tmpdir, TEST_DATA_DIR,
                                    "maxpool_p0_s2_k3.nnp",
                                    "maxpool_p0_s2_k3.onnx",
                                    "out_data_1", "exec_0")


def test_onnx_nnp_conversion_relu(tmpdir, nnp_fixture):
    if not (REFERENCE_AVAILABLE and ONNX_AVAILABLE and CAFFE2_AVAILABLE):
        pytest.skip('CAFFE2 does not installed.')
    convert_onnx_to_nnp_and_compare(
        tmpdir, TEST_DATA_DIR, "relu.onnx", "relu.nnp", "out_data_1", "exec_0")


def test_nnp_onnx_conversion_relu(tmpdir, nnp_fixture):
    if not (REFERENCE_AVAILABLE and ONNX_AVAILABLE and CAFFE2_AVAILABLE):
        pytest.skip('CAFFE2 does not installed.')
    convert_nnp_to_onnx_and_compare(
        tmpdir, TEST_DATA_DIR, "relu.nnp", "relu.onnx", "out_data_1", "exec_0")


def test_onnx_nnp_conversion_softmax(tmpdir, nnp_fixture):
    if not (REFERENCE_AVAILABLE and ONNX_AVAILABLE and CAFFE2_AVAILABLE):
        pytest.skip('CAFFE2 does not installed.')
    convert_onnx_to_nnp_and_compare(tmpdir, TEST_DATA_DIR,
                                    "softmax.onnx", "softmax.nnp",
                                    "out_data_1", "exec_0")


def test_nnp_onnx_conversion_softmax(tmpdir, nnp_fixture):
    if not (REFERENCE_AVAILABLE and ONNX_AVAILABLE and CAFFE2_AVAILABLE):
        pytest.skip('CAFFE2 does not installed.')
    convert_nnp_to_onnx_and_compare(tmpdir, TEST_DATA_DIR,
                                    "softmax.nnp", "softmax.onnx",
                                    "out_data_1", "exec_0")


def test_onnx_nnp_conversion_average_pool(tmpdir, nnp_fixture):
    if not (REFERENCE_AVAILABLE and ONNX_AVAILABLE and CAFFE2_AVAILABLE):
        pytest.skip('CAFFE2 does not installed.')
    convert_onnx_to_nnp_and_compare(tmpdir, TEST_DATA_DIR,
                                    "average_pool.onnx", "average_pool.nnp",
                                    "out_data_1", "exec_0")


def test_nnp_onnx_conversion_average_pool(tmpdir, nnp_fixture):
    if not (REFERENCE_AVAILABLE and ONNX_AVAILABLE and CAFFE2_AVAILABLE):
        pytest.skip('CAFFE2 does not installed.')
    convert_nnp_to_onnx_and_compare(tmpdir, TEST_DATA_DIR,
                                    "average_pool.nnp", "average_pool.onnx",
                                    "out_data_1", "exec_0")


# def test_onnx_nnp_conversion_average_pool_p0_0_1_1_s1_k2(tmpdir, nnp_fixture):
#    if not (REFERENCE_AVAILABLE and ONNX_AVAILABLE and CAFFE2_AVAILABLE):
#        pytest.skip('CAFFE2 does not installed.')
#    convert_onnx_to_nnp_and_compare(tmpdir, TEST_DATA_DIR,
#                                    "average_pool_p0_0_1_1_s1_k2.onnx",
#                                    "average_pool_p0_0_1_1_s1_k2.nnp",
#                                    "out_data_1", "exec_0",
#                                    export_nnp_path=TEST_DATA_DIR)
#
#
# def test_nnp_onnx_conversion_average_pool_p0_0_1_1_s1_k2(tmpdir, nnp_fixture):
#    if not (REFERENCE_AVAILABLE and ONNX_AVAILABLE and CAFFE2_AVAILABLE):
#        pytest.skip('CAFFE2 does not installed.')
#    convert_nnp_to_onnx_and_compare(tmpdir, TEST_DATA_DIR,
#                                    "average_pool_p0_0_1_1_s1_k2.nnp",
#                                    "average_pool_p0_0_1_1_s1_k2.onnx",
#                                    "out_data_1", "exec_0")


def test_onnx_nnp_conversion_sum(tmpdir, nnp_fixture):
    if not (REFERENCE_AVAILABLE and ONNX_AVAILABLE and CAFFE2_AVAILABLE):
        pytest.skip('CAFFE2 does not installed.')
    convert_onnx_to_nnp_and_compare(tmpdir, TEST_DATA_DIR,
                                    "sum.onnx", "sum.nnp",
                                    "out_data_1", "exec_0")


def test_nnp_onnx_conversion_sum(tmpdir, nnp_fixture):
    if not (REFERENCE_AVAILABLE and ONNX_AVAILABLE and CAFFE2_AVAILABLE):
        pytest.skip('CAFFE2 does not installed.')
    convert_nnp_to_onnx_and_compare(
        tmpdir, TEST_DATA_DIR, "sum.nnp", "sum.onnx", "out_data_1", "exec_0")


def test_onnx_nnp_conversion_batch_normalization(tmpdir, nnp_fixture):
    if not (REFERENCE_AVAILABLE and ONNX_AVAILABLE and CAFFE2_AVAILABLE):
        pytest.skip('CAFFE2 does not installed.')
    convert_onnx_to_nnp_and_compare(tmpdir, TEST_DATA_DIR,
                                    "batch_norm.onnx", "batch_norm.nnp",
                                    "out_data_1", "exec_0", atol=1e-05)


def test_nnp_onnx_conversion_batch_normalization(tmpdir, nnp_fixture):
    if not (REFERENCE_AVAILABLE and ONNX_AVAILABLE and CAFFE2_AVAILABLE):
        pytest.skip('CAFFE2 does not installed.')
    convert_nnp_to_onnx_and_compare(tmpdir, TEST_DATA_DIR,
                                    "batch_norm.nnp", "batch_norm.onnx",
                                    "out_data_1", "exec_0", atol=1e-05)


def test_onnx_nnp_conversion_gemm(tmpdir, nnp_fixture):
    if not (REFERENCE_AVAILABLE and ONNX_AVAILABLE and CAFFE2_AVAILABLE):
        pytest.skip('CAFFE2 does not installed.')
    convert_onnx_to_nnp_and_compare(
        tmpdir, TEST_DATA_DIR, "gemm.onnx", "gemm.nnp", "out_data_1", "exec_0")


@pytest.mark.skip
def test_nnp_onnx_conversion_gemm(tmpdir, nnp_fixture):
    if not (REFERENCE_AVAILABLE and ONNX_AVAILABLE and CAFFE2_AVAILABLE):
        pytest.skip('CAFFE2 does not installed.')
    convert_nnp_to_onnx_and_compare(
        tmpdir, TEST_DATA_DIR, "gemm.nnp", "gemm.onnx", "out_data_1", "exec_0")


def test_onnx_nnp_conversion_add_no_broadcast(tmpdir, nnp_fixture):
    if not (REFERENCE_AVAILABLE and ONNX_AVAILABLE and CAFFE2_AVAILABLE):
        pytest.skip('CAFFE2 does not installed.')
    convert_onnx_to_nnp_and_compare(tmpdir, TEST_DATA_DIR,
                                    "add_no_broadcast.onnx",
                                    "add_no_broadcast.nnp",
                                    "out_data_1", "exec_0")


def test_nnp_onnx_conversion_add_no_broadcast(tmpdir, nnp_fixture):
    if not (REFERENCE_AVAILABLE and ONNX_AVAILABLE and CAFFE2_AVAILABLE):
        pytest.skip('CAFFE2 does not installed.')
    convert_nnp_to_onnx_and_compare(tmpdir, TEST_DATA_DIR,
                                    "add_no_broadcast.nnp",
                                    "add_no_broadcast.onnx",
                                    "out_data_1", "exec_0")


def test_onnx_nnp_conversion_add_broadcast_axis1(tmpdir, nnp_fixture):
    if not (REFERENCE_AVAILABLE and ONNX_AVAILABLE and CAFFE2_AVAILABLE):
        pytest.skip('CAFFE2 does not installed.')
    convert_onnx_to_nnp_and_compare(tmpdir, TEST_DATA_DIR,
                                    "add_broadcast_axis1.onnx",
                                    "add_broadcast_axis1.nnp",
                                    "out_data_1", "exec_0")


def test_nnp_onnx_conversion_add_broadcast_axis1(tmpdir, nnp_fixture):
    if not (REFERENCE_AVAILABLE and ONNX_AVAILABLE and CAFFE2_AVAILABLE):
        pytest.skip('CAFFE2 does not installed.')
    convert_nnp_to_onnx_and_compare(tmpdir, TEST_DATA_DIR,
                                    "add_broadcast_axis1.nnp",
                                    "add_broadcast_axis1.onnx",
                                    "out_data_1", "exec_0")


def test_onnx_nnp_conversion_mul_no_broadcast(tmpdir, nnp_fixture):
    if not (REFERENCE_AVAILABLE and ONNX_AVAILABLE and CAFFE2_AVAILABLE):
        pytest.skip('CAFFE2 does not installed.')
    convert_onnx_to_nnp_and_compare(tmpdir, TEST_DATA_DIR,
                                    "mul_no_broadcast.onnx",
                                    "mul_no_broadcast.nnp",
                                    "out_data_1", "exec_0")


def test_nnp_onnx_conversion_mul_no_broadcast(tmpdir, nnp_fixture):
    if not (REFERENCE_AVAILABLE and ONNX_AVAILABLE and CAFFE2_AVAILABLE):
        pytest.skip('CAFFE2 does not installed.')
    convert_nnp_to_onnx_and_compare(tmpdir, TEST_DATA_DIR,
                                    "mul_no_broadcast.nnp",
                                    "mul_no_broadcast.onnx",
                                    "out_data_1", "exec_0")


def test_onnx_nnp_conversion_mul_broadcast_axis1(tmpdir, nnp_fixture):
    if not (REFERENCE_AVAILABLE and ONNX_AVAILABLE and CAFFE2_AVAILABLE):
        pytest.skip('CAFFE2 does not installed.')
    convert_onnx_to_nnp_and_compare(tmpdir, TEST_DATA_DIR,
                                    "mul_broadcast_axis1.onnx",
                                    "mul_broadcast_axis1.nnp",
                                    "out_data_1", "exec_0")


def test_nnp_onnx_conversion_mul_broadcast_axis1(tmpdir, nnp_fixture):
    if not (REFERENCE_AVAILABLE and ONNX_AVAILABLE and CAFFE2_AVAILABLE):
        pytest.skip('CAFFE2 does not installed.')
    convert_nnp_to_onnx_and_compare(tmpdir, TEST_DATA_DIR,
                                    "mul_broadcast_axis1.nnp",
                                    "mul_broadcast_axis1.onnx",
                                    "out_data_1", "exec_0")


def test_onnx_nnp_conversion_constant(tmpdir, nnp_fixture):
    if not (REFERENCE_AVAILABLE and ONNX_AVAILABLE and CAFFE2_AVAILABLE):
        pytest.skip('CAFFE2 does not installed.')
    convert_onnx_to_nnp_and_compare(tmpdir, TEST_DATA_DIR,
                                    "constant.onnx", "constant.nnp",
                                    "Pooling33_Output_0", "exec_0")


@pytest.mark.skip(reason="Current version of reshape does not supported yet.")
def test_onnx_nnp_conversion_reshape(tmpdir, nnp_fixture):
    if not (REFERENCE_AVAILABLE and ONNX_AVAILABLE and CAFFE2_AVAILABLE):
        pytest.skip('CAFFE2 does not installed.')
    convert_onnx_to_nnp_and_compare(tmpdir, TEST_DATA_DIR,
                                    "reshape.onnx", "reshape.nnp",
                                    "out_data_1", "exec_0")


@pytest.mark.skip(reason="Current version of reshape does not supported yet.")
def test_nnp_onnx_conversion_reshape(tmpdir, nnp_fixture):
    if not (REFERENCE_AVAILABLE and ONNX_AVAILABLE and CAFFE2_AVAILABLE):
        pytest.skip('CAFFE2 does not installed.')
    convert_nnp_to_onnx_and_compare(tmpdir, TEST_DATA_DIR,
                                    "reshape.nnp", "reshape.onnx",
                                    "out_data_1", "exec_0")


def test_onnx_nnp_conversion_matmul(tmpdir, nnp_fixture):
    if not (REFERENCE_AVAILABLE and ONNX_AVAILABLE and CAFFE2_AVAILABLE):
        pytest.skip('CAFFE2 does not installed.')
    convert_onnx_to_nnp_and_compare(tmpdir, TEST_DATA_DIR,
                                    "matmul.onnx", "matmul.nnp",
                                    "out_data_1", "exec_0")


def test_nnp_onnx_conversion_matmul(tmpdir, nnp_fixture):
    if not (REFERENCE_AVAILABLE and ONNX_AVAILABLE and CAFFE2_AVAILABLE):
        pytest.skip('CAFFE2 does not installed.')
    convert_nnp_to_onnx_and_compare(tmpdir, TEST_DATA_DIR,
                                    "matmul.nnp", "matmul.onnx",
                                    "out_data_1", "exec_0")


def test_onnx_nnp_conversion_transpose(tmpdir, nnp_fixture):
    if not (REFERENCE_AVAILABLE and ONNX_AVAILABLE and CAFFE2_AVAILABLE):
        pytest.skip('CAFFE2 does not installed.')
    convert_onnx_to_nnp_and_compare(tmpdir, TEST_DATA_DIR,
                                    "transpose.onnx", "transpose.nnp",
                                    "out_data_1", "exec_0")


def test_nnp_onnx_conversion_transpose(tmpdir, nnp_fixture):
    if not (REFERENCE_AVAILABLE and ONNX_AVAILABLE and CAFFE2_AVAILABLE):
        pytest.skip('CAFFE2 does not installed.')
    convert_nnp_to_onnx_and_compare(tmpdir, TEST_DATA_DIR,
                                    "transpose.nnp", "transpose.onnx",
                                    "out_data_1", "exec_0")


def test_onnx_nnp_conversion_abs(tmpdir, nnp_fixture):
    if not (REFERENCE_AVAILABLE and ONNX_AVAILABLE and CAFFE2_AVAILABLE):
        pytest.skip('CAFFE2 does not installed.')
    convert_onnx_to_nnp_and_compare(
        tmpdir, TEST_DATA_DIR, "abs.onnx", "abs.nnp", "out_data_1", "exec_0")


def test_nnp_onnx_conversion_abs(tmpdir, nnp_fixture):
    if not (REFERENCE_AVAILABLE and ONNX_AVAILABLE and CAFFE2_AVAILABLE):
        pytest.skip('CAFFE2 does not installed.')
    convert_nnp_to_onnx_and_compare(
        tmpdir, TEST_DATA_DIR, "abs.nnp", "abs.onnx", "out_data_1", "exec_0")


def test_onnx_nnp_conversion_sigmoid(tmpdir, nnp_fixture):
    if not (REFERENCE_AVAILABLE and ONNX_AVAILABLE and CAFFE2_AVAILABLE):
        pytest.skip('CAFFE2 does not installed.')
    convert_onnx_to_nnp_and_compare(tmpdir, TEST_DATA_DIR,
                                    "sigmoid.onnx", "sigmoid.nnp",
                                    "out_data_1", "exec_0")


def test_nnp_onnx_conversion_sigmoid(tmpdir, nnp_fixture):
    if not (REFERENCE_AVAILABLE and ONNX_AVAILABLE and CAFFE2_AVAILABLE):
        pytest.skip('CAFFE2 does not installed.')
    convert_nnp_to_onnx_and_compare(tmpdir, TEST_DATA_DIR,
                                    "sigmoid.nnp", "sigmoid.onnx",
                                    "out_data_1", "exec_0")


def test_onnx_nnp_conversion_tanh(tmpdir, nnp_fixture):
    if not (REFERENCE_AVAILABLE and ONNX_AVAILABLE and CAFFE2_AVAILABLE):
        pytest.skip('CAFFE2 does not installed.')
    convert_onnx_to_nnp_and_compare(
        tmpdir, TEST_DATA_DIR, "tanh.onnx", "tanh.nnp", "out_data_1", "exec_0")


def test_nnp_onnx_conversion_tanh(tmpdir, nnp_fixture):
    if not (REFERENCE_AVAILABLE and ONNX_AVAILABLE and CAFFE2_AVAILABLE):
        pytest.skip('CAFFE2 does not installed.')
    convert_nnp_to_onnx_and_compare(
        tmpdir, TEST_DATA_DIR, "tanh.nnp", "tanh.onnx", "out_data_1", "exec_0")


def test_onnx_nnp_conversion_leaky_relu(tmpdir, nnp_fixture):
    if not (REFERENCE_AVAILABLE and ONNX_AVAILABLE and CAFFE2_AVAILABLE):
        pytest.skip('CAFFE2 does not installed.')
    convert_onnx_to_nnp_and_compare(tmpdir, TEST_DATA_DIR,
                                    "leaky_relu.onnx", "leaky_relu.nnp",
                                    "out_data_1", "exec_0")


def test_nnp_onnx_conversion_leaky_relu(tmpdir, nnp_fixture):
    if not (REFERENCE_AVAILABLE and ONNX_AVAILABLE and CAFFE2_AVAILABLE):
        pytest.skip('CAFFE2 does not installed.')
    convert_nnp_to_onnx_and_compare(tmpdir, TEST_DATA_DIR,
                                    "leaky_relu.nnp", "leaky_relu.onnx",
                                    "out_data_1", "exec_0")


def test_onnx_nnp_conversion_log(tmpdir, nnp_fixture):
    if not (REFERENCE_AVAILABLE and ONNX_AVAILABLE and CAFFE2_AVAILABLE):
        pytest.skip('CAFFE2 does not installed.')
    convert_onnx_to_nnp_and_compare(
        tmpdir, TEST_DATA_DIR, "log.onnx", "log.nnp", "out_data_1", "exec_0")


def test_nnp_onnx_conversion_log(tmpdir, nnp_fixture):
    if not (REFERENCE_AVAILABLE and ONNX_AVAILABLE and CAFFE2_AVAILABLE):
        pytest.skip('CAFFE2 does not installed.')
    convert_nnp_to_onnx_and_compare(
        tmpdir, TEST_DATA_DIR, "log.nnp", "log.onnx", "out_data_1", "exec_0")


def test_onnx_nnp_conversion_not(tmpdir, nnp_fixture):
    if not (REFERENCE_AVAILABLE and ONNX_AVAILABLE and CAFFE2_AVAILABLE):
        pytest.skip('CAFFE2 does not installed.')
    convert_onnx_to_nnp_and_compare(
        tmpdir, TEST_DATA_DIR, "not.onnx", "not.nnp", "out_data_1", "exec_0")


def test_nnp_onnx_conversion_not(tmpdir, nnp_fixture):
    if not (REFERENCE_AVAILABLE and ONNX_AVAILABLE and CAFFE2_AVAILABLE):
        pytest.skip('CAFFE2 does not installed.')
    convert_nnp_to_onnx_and_compare(
        tmpdir, TEST_DATA_DIR, "not.nnp", "not.onnx", "out_data_1", "exec_0")


def test_onnx_nnp_conversion_elu(tmpdir, nnp_fixture):
    if not (REFERENCE_AVAILABLE and ONNX_AVAILABLE and CAFFE2_AVAILABLE):
        pytest.skip('CAFFE2 does not installed.')
    convert_onnx_to_nnp_and_compare(
        tmpdir, TEST_DATA_DIR, "elu.onnx", "elu.nnp", "out_data_1", "exec_0")


def test_nnp_onnx_conversion_elu(tmpdir, nnp_fixture):
    if not (REFERENCE_AVAILABLE and ONNX_AVAILABLE and CAFFE2_AVAILABLE):
        pytest.skip('CAFFE2 does not installed.')
    convert_nnp_to_onnx_and_compare(
        tmpdir, TEST_DATA_DIR, "elu.nnp", "elu.onnx", "out_data_1", "exec_0")


def test_onnx_nnp_conversion_selu(tmpdir, nnp_fixture):
    if not (REFERENCE_AVAILABLE and ONNX_AVAILABLE and CAFFE2_AVAILABLE):
        pytest.skip('CAFFE2 does not installed.')
    convert_onnx_to_nnp_and_compare(
        tmpdir, TEST_DATA_DIR, "selu.onnx", "selu.nnp", "out_data_1", "exec_0")


def test_nnp_onnx_conversion_selu(tmpdir, nnp_fixture):
    if not (REFERENCE_AVAILABLE and ONNX_AVAILABLE and CAFFE2_AVAILABLE):
        pytest.skip('CAFFE2 does not installed.')
    convert_nnp_to_onnx_and_compare(
        tmpdir, TEST_DATA_DIR, "selu.nnp", "selu.onnx", "out_data_1", "exec_0")


def test_onnx_nnp_conversion_reduce_sum(tmpdir, nnp_fixture):
    if not (REFERENCE_AVAILABLE and ONNX_AVAILABLE and CAFFE2_AVAILABLE):
        pytest.skip('CAFFE2 does not installed.')
    convert_onnx_to_nnp_and_compare(tmpdir, TEST_DATA_DIR,
                                    "reduce_sum.onnx", "reduce_sum.nnp",
                                    "out_data_1", "exec_0")


def test_nnp_onnx_conversion_reduce_sum(tmpdir, nnp_fixture):
    if not (REFERENCE_AVAILABLE and ONNX_AVAILABLE and CAFFE2_AVAILABLE):
        pytest.skip('CAFFE2 does not installed.')
    convert_nnp_to_onnx_and_compare(tmpdir, TEST_DATA_DIR,
                                    "reduce_sum.nnp", "reduce_sum.onnx",
                                    "out_data_1", "exec_0")


def test_onnx_nnp_conversion_reduce_mean(tmpdir, nnp_fixture):
    if not (REFERENCE_AVAILABLE and ONNX_AVAILABLE and CAFFE2_AVAILABLE):
        pytest.skip('CAFFE2 does not installed.')
    convert_onnx_to_nnp_and_compare(tmpdir, TEST_DATA_DIR,
                                    "reduce_mean.onnx", "reduce_mean.nnp",
                                    "out_data_1", "exec_0")


def test_nnp_onnx_conversion_reduce_mean(tmpdir, nnp_fixture):
    if not (REFERENCE_AVAILABLE and ONNX_AVAILABLE and CAFFE2_AVAILABLE):
        pytest.skip('CAFFE2 does not installed.')
    convert_nnp_to_onnx_and_compare(tmpdir, TEST_DATA_DIR,
                                    "reduce_mean.nnp", "reduce_mean.onnx",
                                    "out_data_1", "exec_0")


def test_onnx_nnp_conversion_and_no_broadcast(tmpdir, nnp_fixture):
    if not (REFERENCE_AVAILABLE and ONNX_AVAILABLE and CAFFE2_AVAILABLE):
        pytest.skip('CAFFE2 does not installed.')
    convert_onnx_to_nnp_and_compare(tmpdir, TEST_DATA_DIR,
                                    "and_no_broadcast.onnx",
                                    "and_no_broadcast.nnp",
                                    "out_data_1", "exec_0")


def test_nnp_onnx_conversion_and_no_broadcast(tmpdir, nnp_fixture):
    if not (REFERENCE_AVAILABLE and ONNX_AVAILABLE and CAFFE2_AVAILABLE):
        pytest.skip('CAFFE2 does not installed.')
    convert_nnp_to_onnx_and_compare(tmpdir, TEST_DATA_DIR,
                                    "and_no_broadcast.nnp",
                                    "and_no_broadcast.onnx",
                                    "out_data_1", "exec_0")


def test_onnx_nnp_conversion_and_broadcast_axis1(tmpdir, nnp_fixture):
    if not (REFERENCE_AVAILABLE and ONNX_AVAILABLE and CAFFE2_AVAILABLE):
        pytest.skip('CAFFE2 does not installed.')
    convert_onnx_to_nnp_and_compare(tmpdir, TEST_DATA_DIR,
                                    "and_broadcast_axis1.onnx",
                                    "and_broadcast_axis1.nnp",
                                    "out_data_1", "exec_0")


def test_nnp_onnx_conversion_and_broadcast_axis1(tmpdir, nnp_fixture):
    if not (REFERENCE_AVAILABLE and ONNX_AVAILABLE and CAFFE2_AVAILABLE):
        pytest.skip('CAFFE2 does not installed.')
    convert_nnp_to_onnx_and_compare(tmpdir, TEST_DATA_DIR,
                                    "and_broadcast_axis1.nnp",
                                    "and_broadcast_axis1.onnx",
                                    "out_data_1", "exec_0")


def test_onnx_nnp_conversion_or_no_broadcast(tmpdir, nnp_fixture):
    if not (REFERENCE_AVAILABLE and ONNX_AVAILABLE and CAFFE2_AVAILABLE):
        pytest.skip('CAFFE2 does not installed.')
    convert_onnx_to_nnp_and_compare(tmpdir, TEST_DATA_DIR,
                                    "or_no_broadcast.onnx",
                                    "or_no_broadcast.nnp",
                                    "out_data_1", "exec_0")


def test_nnp_onnx_conversion_or_no_broadcast(tmpdir, nnp_fixture):
    if not (REFERENCE_AVAILABLE and ONNX_AVAILABLE and CAFFE2_AVAILABLE):
        pytest.skip('CAFFE2 does not installed.')
    convert_nnp_to_onnx_and_compare(tmpdir, TEST_DATA_DIR,
                                    "or_no_broadcast.nnp",
                                    "or_no_broadcast.onnx",
                                    "out_data_1", "exec_0")


def test_onnx_nnp_conversion_or_broadcast_axis1(tmpdir, nnp_fixture):
    if not (REFERENCE_AVAILABLE and ONNX_AVAILABLE and CAFFE2_AVAILABLE):
        pytest.skip('CAFFE2 does not installed.')
    convert_onnx_to_nnp_and_compare(tmpdir, TEST_DATA_DIR,
                                    "or_broadcast_axis1.onnx",
                                    "or_broadcast_axis1.nnp",
                                    "out_data_1", "exec_0")


def test_nnp_onnx_conversion_or_broadcast_axis1(tmpdir, nnp_fixture):
    if not (REFERENCE_AVAILABLE and ONNX_AVAILABLE and CAFFE2_AVAILABLE):
        pytest.skip('CAFFE2 does not installed.')
    convert_nnp_to_onnx_and_compare(tmpdir, TEST_DATA_DIR,
                                    "or_broadcast_axis1.nnp",
                                    "or_broadcast_axis1.onnx",
                                    "out_data_1", "exec_0")


def test_onnx_nnp_conversion_xor_no_broadcast(tmpdir, nnp_fixture):
    if not (REFERENCE_AVAILABLE and ONNX_AVAILABLE and CAFFE2_AVAILABLE):
        pytest.skip('CAFFE2 does not installed.')
    convert_onnx_to_nnp_and_compare(tmpdir, TEST_DATA_DIR,
                                    "xor_no_broadcast.onnx",
                                    "xor_no_broadcast.nnp",
                                    "out_data_1", "exec_0")


def test_nnp_onnx_conversion_xor_no_broadcast(tmpdir, nnp_fixture):
    if not (REFERENCE_AVAILABLE and ONNX_AVAILABLE and CAFFE2_AVAILABLE):
        pytest.skip('CAFFE2 does not installed.')
    convert_nnp_to_onnx_and_compare(tmpdir, TEST_DATA_DIR,
                                    "xor_no_broadcast.nnp",
                                    "xor_no_broadcast.onnx",
                                    "out_data_1", "exec_0")


def test_onnx_nnp_conversion_xor_broadcast_axis1(tmpdir, nnp_fixture):
    if not (REFERENCE_AVAILABLE and ONNX_AVAILABLE and CAFFE2_AVAILABLE):
        pytest.skip('CAFFE2 does not installed.')
    convert_onnx_to_nnp_and_compare(tmpdir, TEST_DATA_DIR,
                                    "xor_broadcast_axis1.onnx",
                                    "xor_broadcast_axis1.nnp",
                                    "out_data_1", "exec_0")


def test_nnp_onnx_conversion_xor_broadcast_axis1(tmpdir, nnp_fixture):
    if not (REFERENCE_AVAILABLE and ONNX_AVAILABLE and CAFFE2_AVAILABLE):
        pytest.skip('CAFFE2 does not installed.')
    convert_nnp_to_onnx_and_compare(tmpdir, TEST_DATA_DIR,
                                    "xor_broadcast_axis1.nnp",
                                    "xor_broadcast_axis1.onnx",
                                    "out_data_1", "exec_0")


def test_onnx_nnp_conversion_div_no_broadcast(tmpdir, nnp_fixture):
    if not (REFERENCE_AVAILABLE and ONNX_AVAILABLE and CAFFE2_AVAILABLE):
        pytest.skip('CAFFE2 does not installed.')
    convert_onnx_to_nnp_and_compare(tmpdir, TEST_DATA_DIR,
                                    "div_no_broadcast.onnx",
                                    "div_no_broadcast.nnp",
                                    "out_data_1", "exec_0")


def test_nnp_onnx_conversion_div_no_broadcast(tmpdir, nnp_fixture):
    if not (REFERENCE_AVAILABLE and ONNX_AVAILABLE and CAFFE2_AVAILABLE):
        pytest.skip('CAFFE2 does not installed.')
    convert_nnp_to_onnx_and_compare(tmpdir, TEST_DATA_DIR,
                                    "div_no_broadcast.nnp",
                                    "div_no_broadcast.onnx",
                                    "out_data_1", "exec_0")


def test_onnx_nnp_conversion_div_broadcast_axis1(tmpdir, nnp_fixture):
    if not (REFERENCE_AVAILABLE and ONNX_AVAILABLE and CAFFE2_AVAILABLE):
        pytest.skip('CAFFE2 does not installed.')
    convert_onnx_to_nnp_and_compare(tmpdir, TEST_DATA_DIR,
                                    "div_broadcast_axis1.onnx",
                                    "div_broadcast_axis1.nnp",
                                    "out_data_1", "exec_0")


def test_nnp_onnx_conversion_div_broadcast_axis1(tmpdir, nnp_fixture):
    if not (REFERENCE_AVAILABLE and ONNX_AVAILABLE and CAFFE2_AVAILABLE):
        pytest.skip('CAFFE2 does not installed.')
    convert_nnp_to_onnx_and_compare(tmpdir, TEST_DATA_DIR,
                                    "div_broadcast_axis1.nnp",
                                    "div_broadcast_axis1.onnx",
                                    "out_data_1", "exec_0")


def test_onnx_nnp_conversion_pow_no_broadcast(tmpdir, nnp_fixture):
    if not (REFERENCE_AVAILABLE and ONNX_AVAILABLE and CAFFE2_AVAILABLE):
        pytest.skip('CAFFE2 does not installed.')
    convert_onnx_to_nnp_and_compare(tmpdir, TEST_DATA_DIR,
                                    "pow_no_broadcast.onnx",
                                    "pow_no_broadcast.nnp",
                                    "out_data_1", "exec_0")


def test_nnp_onnx_conversion_pow_no_broadcast(tmpdir, nnp_fixture):
    if not (REFERENCE_AVAILABLE and ONNX_AVAILABLE and CAFFE2_AVAILABLE):
        pytest.skip('CAFFE2 does not installed.')
    convert_nnp_to_onnx_and_compare(tmpdir, TEST_DATA_DIR,
                                    "pow_no_broadcast.nnp",
                                    "pow_no_broadcast.onnx",
                                    "out_data_1", "exec_0")


def test_onnx_nnp_conversion_pow_broadcast_axis1(tmpdir, nnp_fixture):
    if not (REFERENCE_AVAILABLE and ONNX_AVAILABLE and CAFFE2_AVAILABLE):
        pytest.skip('CAFFE2 does not installed.')
    convert_onnx_to_nnp_and_compare(tmpdir, TEST_DATA_DIR,
                                    "pow_broadcast_axis1.onnx",
                                    "pow_broadcast_axis1.nnp",
                                    "out_data_1", "exec_0")


def test_nnp_onnx_conversion_pow_broadcast_axis1(tmpdir, nnp_fixture):
    if not (REFERENCE_AVAILABLE and ONNX_AVAILABLE and CAFFE2_AVAILABLE):
        pytest.skip('CAFFE2 does not installed.')
    convert_nnp_to_onnx_and_compare(tmpdir, TEST_DATA_DIR,
                                    "pow_broadcast_axis1.nnp",
                                    "pow_broadcast_axis1.onnx",
                                    "out_data_1", "exec_0")


def test_onnx_nnp_conversion_sub_no_broadcast(tmpdir, nnp_fixture):
    if not (REFERENCE_AVAILABLE and ONNX_AVAILABLE and CAFFE2_AVAILABLE):
        pytest.skip('CAFFE2 does not installed.')
    convert_onnx_to_nnp_and_compare(tmpdir, TEST_DATA_DIR,
                                    "sub_no_broadcast.onnx",
                                    "sub_no_broadcast.nnp",
                                    "out_data_1", "exec_0")


def test_nnp_onnx_conversion_sub_no_broadcast(tmpdir, nnp_fixture):
    if not (REFERENCE_AVAILABLE and ONNX_AVAILABLE and CAFFE2_AVAILABLE):
        pytest.skip('CAFFE2 does not installed.')
    convert_nnp_to_onnx_and_compare(tmpdir, TEST_DATA_DIR,
                                    "sub_no_broadcast.nnp",
                                    "sub_no_broadcast.onnx",
                                    "out_data_1", "exec_0")


def test_onnx_nnp_conversion_sub_broadcast_axis1(tmpdir, nnp_fixture):
    if not (REFERENCE_AVAILABLE and ONNX_AVAILABLE and CAFFE2_AVAILABLE):
        pytest.skip('CAFFE2 does not installed.')
    convert_onnx_to_nnp_and_compare(tmpdir, TEST_DATA_DIR,
                                    "sub_broadcast_axis1.onnx",
                                    "sub_broadcast_axis1.nnp",
                                    "out_data_1", "exec_0")


def test_nnp_onnx_conversion_sub_broadcast_axis1(tmpdir, nnp_fixture):
    if not (REFERENCE_AVAILABLE and ONNX_AVAILABLE and CAFFE2_AVAILABLE):
        pytest.skip('CAFFE2 does not installed.')
    convert_nnp_to_onnx_and_compare(tmpdir, TEST_DATA_DIR,
                                    "sub_broadcast_axis1.nnp",
                                    "sub_broadcast_axis1.onnx",
                                    "out_data_1", "exec_0")


def test_onnx_nnp_conversion_less_no_broadcast(tmpdir, nnp_fixture):
    if not (REFERENCE_AVAILABLE and ONNX_AVAILABLE and CAFFE2_AVAILABLE):
        pytest.skip('CAFFE2 does not installed.')
    convert_onnx_to_nnp_and_compare(tmpdir, TEST_DATA_DIR,
                                    "less_no_broadcast.onnx",
                                    "less_no_broadcast.nnp",
                                    "out_data_1", "exec_0")


def test_nnp_onnx_conversion_less_no_broadcast(tmpdir, nnp_fixture):
    if not (REFERENCE_AVAILABLE and ONNX_AVAILABLE and CAFFE2_AVAILABLE):
        pytest.skip('CAFFE2 does not installed.')
    convert_nnp_to_onnx_and_compare(tmpdir, TEST_DATA_DIR,
                                    "less_no_broadcast.nnp",
                                    "less_no_broadcast.onnx",
                                    "out_data_1", "exec_0")


def test_onnx_nnp_conversion_less_broadcast_axis1(tmpdir, nnp_fixture):
    if not (REFERENCE_AVAILABLE and ONNX_AVAILABLE and CAFFE2_AVAILABLE):
        pytest.skip('CAFFE2 does not installed.')
    convert_onnx_to_nnp_and_compare(tmpdir, TEST_DATA_DIR,
                                    "less_broadcast_axis1.onnx",
                                    "less_broadcast_axis1.nnp",
                                    "out_data_1", "exec_0")


def test_nnp_onnx_conversion_less_broadcast_axis1(tmpdir, nnp_fixture):
    if not (REFERENCE_AVAILABLE and ONNX_AVAILABLE and CAFFE2_AVAILABLE):
        pytest.skip('CAFFE2 does not installed.')
    convert_nnp_to_onnx_and_compare(tmpdir, TEST_DATA_DIR,
                                    "less_broadcast_axis1.nnp",
                                    "less_broadcast_axis1.onnx",
                                    "out_data_1", "exec_0")


def test_onnx_nnp_conversion_greater_no_broadcast(tmpdir, nnp_fixture):
    if not (REFERENCE_AVAILABLE and ONNX_AVAILABLE and CAFFE2_AVAILABLE):
        pytest.skip('CAFFE2 does not installed.')
    convert_onnx_to_nnp_and_compare(tmpdir, TEST_DATA_DIR,
                                    "greater_no_broadcast.onnx",
                                    "greater_no_broadcast.nnp",
                                    "out_data_1", "exec_0")


def test_nnp_onnx_conversion_greater_no_broadcast(tmpdir, nnp_fixture):
    if not (REFERENCE_AVAILABLE and ONNX_AVAILABLE and CAFFE2_AVAILABLE):
        pytest.skip('CAFFE2 does not installed.')
    convert_nnp_to_onnx_and_compare(tmpdir, TEST_DATA_DIR,
                                    "greater_no_broadcast.nnp",
                                    "greater_no_broadcast.onnx",
                                    "out_data_1", "exec_0")


def test_onnx_nnp_conversion_greater_broadcast_axis1(tmpdir, nnp_fixture):
    if not (REFERENCE_AVAILABLE and ONNX_AVAILABLE and CAFFE2_AVAILABLE):
        pytest.skip('CAFFE2 does not installed.')
    convert_onnx_to_nnp_and_compare(tmpdir, TEST_DATA_DIR,
                                    "greater_broadcast_axis1.onnx",
                                    "greater_broadcast_axis1.nnp",
                                    "out_data_1", "exec_0")


def test_nnp_onnx_conversion_greater_broadcast_axis1(tmpdir, nnp_fixture):
    if not (REFERENCE_AVAILABLE and ONNX_AVAILABLE and CAFFE2_AVAILABLE):
        pytest.skip('CAFFE2 does not installed.')
    convert_nnp_to_onnx_and_compare(tmpdir, TEST_DATA_DIR,
                                    "greater_broadcast_axis1.nnp",
                                    "greater_broadcast_axis1.onnx",
                                    "out_data_1", "exec_0")


def test_onnx_nnp_conversion_equal_no_broadcast_bool(tmpdir, nnp_fixture):
    if not (REFERENCE_AVAILABLE and ONNX_AVAILABLE and CAFFE2_AVAILABLE):
        pytest.skip('CAFFE2 does not installed.')
    convert_onnx_to_nnp_and_compare(tmpdir, TEST_DATA_DIR,
                                    "equal_no_broadcast_bool.onnx",
                                    "equal_no_broadcast_bool.nnp",
                                    "out_data_1", "exec_0")


def test_nnp_onnx_conversion_equal_no_broadcast_bool(tmpdir, nnp_fixture):
    if not (REFERENCE_AVAILABLE and ONNX_AVAILABLE and CAFFE2_AVAILABLE):
        pytest.skip('CAFFE2 does not installed.')
    convert_nnp_to_onnx_and_compare(tmpdir, TEST_DATA_DIR,
                                    "equal_no_broadcast_bool.nnp",
                                    "equal_no_broadcast_bool.onnx",
                                    "out_data_1", "exec_0")


def test_onnx_nnp_conversion_equal_broadcast_axis1_bool(tmpdir, nnp_fixture):
    if not (REFERENCE_AVAILABLE and ONNX_AVAILABLE and CAFFE2_AVAILABLE):
        pytest.skip('CAFFE2 does not installed.')
    convert_onnx_to_nnp_and_compare(tmpdir, TEST_DATA_DIR,
                                    "equal_broadcast_axis1_bool.onnx",
                                    "equal_broadcast_axis1_bool.nnp",
                                    "out_data_1", "exec_0")


def test_nnp_onnx_conversion_equal_broadcast_axis1_bool(tmpdir, nnp_fixture):
    if not (REFERENCE_AVAILABLE and ONNX_AVAILABLE and CAFFE2_AVAILABLE):
        pytest.skip('CAFFE2 does not installed.')
    convert_nnp_to_onnx_and_compare(tmpdir, TEST_DATA_DIR,
                                    "equal_broadcast_axis1_bool.nnp",
                                    "equal_broadcast_axis1_bool.onnx",
                                    "out_data_1", "exec_0")


def test_onnx_nnp_conversion_equal_no_broadcast_int(tmpdir, nnp_fixture):
    if not (REFERENCE_AVAILABLE and ONNX_AVAILABLE and CAFFE2_AVAILABLE):
        pytest.skip('CAFFE2 does not installed.')
    convert_onnx_to_nnp_and_compare(tmpdir, TEST_DATA_DIR,
                                    "equal_no_broadcast_int.onnx",
                                    "equal_no_broadcast_int.nnp",
                                    "out_data_1", "exec_0")


def test_nnp_onnx_conversion_equal_no_broadcast_int(tmpdir, nnp_fixture):
    if not (REFERENCE_AVAILABLE and ONNX_AVAILABLE and CAFFE2_AVAILABLE):
        pytest.skip('CAFFE2 does not installed.')
    convert_nnp_to_onnx_and_compare(tmpdir, TEST_DATA_DIR,
                                    "equal_no_broadcast_int.nnp",
                                    "equal_no_broadcast_int.onnx",
                                    "out_data_1", "exec_0")


def test_onnx_nnp_conversion_equal_broadcast_axis1_int(tmpdir, nnp_fixture):
    if not (REFERENCE_AVAILABLE and ONNX_AVAILABLE and CAFFE2_AVAILABLE):
        pytest.skip('CAFFE2 does not installed.')
    convert_onnx_to_nnp_and_compare(tmpdir, TEST_DATA_DIR,
                                    "equal_broadcast_axis1_int.onnx",
                                    "equal_broadcast_axis1_int.nnp",
                                    "out_data_1", "exec_0")


def test_nnp_onnx_conversion_equal_broadcast_axis1_int(tmpdir, nnp_fixture):
    if not (REFERENCE_AVAILABLE and ONNX_AVAILABLE and CAFFE2_AVAILABLE):
        pytest.skip('CAFFE2 does not installed.')
    convert_nnp_to_onnx_and_compare(tmpdir, TEST_DATA_DIR,
                                    "equal_broadcast_axis1_int.nnp",
                                    "equal_broadcast_axis1_int.onnx",
                                    "out_data_1", "exec_0")


def test_onnx_nnp_conversion_max(tmpdir, nnp_fixture):
    if not (REFERENCE_AVAILABLE and ONNX_AVAILABLE and CAFFE2_AVAILABLE):
        pytest.skip('CAFFE2 does not installed.')
    convert_onnx_to_nnp_and_compare(tmpdir, TEST_DATA_DIR,
                                    "max.onnx",
                                    "max.nnp",
                                    "out_data_1", "exec_0")


def test_nnp_onnx_conversion_max(tmpdir, nnp_fixture):
    if not (REFERENCE_AVAILABLE and ONNX_AVAILABLE and CAFFE2_AVAILABLE):
        pytest.skip('CAFFE2 does not installed.')
    convert_nnp_to_onnx_and_compare(tmpdir, TEST_DATA_DIR,
                                    "max.nnp",
                                    "max.onnx",
                                    "out_data_1", "exec_0")


def test_onnx_nnp_conversion_min(tmpdir, nnp_fixture):
    if not (REFERENCE_AVAILABLE and ONNX_AVAILABLE and CAFFE2_AVAILABLE):
        pytest.skip('CAFFE2 does not installed.')
    convert_onnx_to_nnp_and_compare(tmpdir, TEST_DATA_DIR,
                                    "min.onnx",
                                    "min.nnp",
                                    "out_data_1", "exec_0")


def test_nnp_onnx_conversion_min(tmpdir, nnp_fixture):
    if not (REFERENCE_AVAILABLE and ONNX_AVAILABLE and CAFFE2_AVAILABLE):
        pytest.skip('CAFFE2 does not installed.')
    convert_nnp_to_onnx_and_compare(tmpdir, TEST_DATA_DIR,
                                    "min.nnp",
                                    "min.onnx",
                                    "out_data_1", "exec_0")


def test_onnx_nnp_conversion_exp(tmpdir, nnp_fixture):
    if not (REFERENCE_AVAILABLE and ONNX_AVAILABLE and CAFFE2_AVAILABLE):
        pytest.skip('CAFFE2 does not installed.')
    convert_onnx_to_nnp_and_compare(tmpdir, TEST_DATA_DIR,
                                    "exp.onnx",
                                    "exp.nnp",
                                    "out_data_1", "exec_0")


def test_nnp_onnx_conversion_exp(tmpdir, nnp_fixture):
    if not (REFERENCE_AVAILABLE and ONNX_AVAILABLE and CAFFE2_AVAILABLE):
        pytest.skip('CAFFE2 does not installed.')
    convert_nnp_to_onnx_and_compare(tmpdir, TEST_DATA_DIR,
                                    "exp.nnp",
                                    "exp.onnx",
                                    "out_data_1", "exec_0")


def test_onnx_nnp_conversion_identity(tmpdir, nnp_fixture):
    if not (REFERENCE_AVAILABLE and ONNX_AVAILABLE and CAFFE2_AVAILABLE):
        pytest.skip('CAFFE2 does not installed.')
    convert_onnx_to_nnp_and_compare(tmpdir, TEST_DATA_DIR,
                                    "identity.onnx",
                                    "identity.nnp",
                                    "out_data_1", "exec_0")


def test_nnp_onnx_conversion_identity(tmpdir, nnp_fixture):
    if not (REFERENCE_AVAILABLE and ONNX_AVAILABLE and CAFFE2_AVAILABLE):
        pytest.skip('CAFFE2 does not installed.')
    convert_nnp_to_onnx_and_compare(tmpdir, TEST_DATA_DIR,
                                    "identity.nnp",
                                    "identity.onnx",
                                    "out_data_1", "exec_0")


def test_onnx_nnp_conversion_prelu_c1(tmpdir, nnp_fixture):
    if not (REFERENCE_AVAILABLE and ONNX_AVAILABLE and CAFFE2_AVAILABLE):
        pytest.skip('CAFFE2 does not installed.')
    convert_onnx_to_nnp_and_compare(tmpdir, TEST_DATA_DIR,
                                    "prelu_c1.onnx",
                                    "prelu_c1.nnp",
                                    "out_data_1", "exec_0")


def test_nnp_onnx_conversion_prelu_c1(tmpdir, nnp_fixture):
    if not (REFERENCE_AVAILABLE and ONNX_AVAILABLE and CAFFE2_AVAILABLE):
        pytest.skip('CAFFE2 does not installed.')
    convert_nnp_to_onnx_and_compare(tmpdir, TEST_DATA_DIR,
                                    "prelu_c1.nnp",
                                    "prelu_c1.onnx",
                                    "out_data_1", "exec_0")


def test_onnx_nnp_conversion_prelu_c3(tmpdir, nnp_fixture):
    if not (REFERENCE_AVAILABLE and ONNX_AVAILABLE and CAFFE2_AVAILABLE):
        pytest.skip('CAFFE2 does not installed.')
    convert_onnx_to_nnp_and_compare(tmpdir, TEST_DATA_DIR,
                                    "prelu_c3.onnx",
                                    "prelu_c3.nnp",
                                    "out_data_1", "exec_0")


def test_nnp_onnx_conversion_prelu_c3(tmpdir, nnp_fixture):
    if not (REFERENCE_AVAILABLE and ONNX_AVAILABLE and CAFFE2_AVAILABLE):
        pytest.skip('CAFFE2 does not installed.')
    convert_nnp_to_onnx_and_compare(tmpdir, TEST_DATA_DIR,
                                    "prelu_c3.nnp",
                                    "prelu_c3.onnx",
                                    "out_data_1", "exec_0")


def test_onnx_nnp_conversion_reciprocal(tmpdir, nnp_fixture):
    if not (REFERENCE_AVAILABLE and ONNX_AVAILABLE and CAFFE2_AVAILABLE):
        pytest.skip('CAFFE2 does not installed.')
    convert_onnx_to_nnp_and_compare(tmpdir, TEST_DATA_DIR,
                                    "reciprocal.onnx",
                                    "reciprocal.nnp",
                                    "Reciprocal4_Output_0", "exec_0")


def test_nnp_onnx_conversion_reciprocal(tmpdir, nnp_fixture):
    if not (REFERENCE_AVAILABLE and ONNX_AVAILABLE and CAFFE2_AVAILABLE):
        pytest.skip('CAFFE2 does not installed.')
    convert_nnp_to_onnx_and_compare(tmpdir, TEST_DATA_DIR,
                                    "reciprocal.nnp",
                                    "reciprocal.onnx",
                                    "Reciprocal4_Output_0", "exec_0")


def test_onnx_nnp_conversion_reduce_min(tmpdir, nnp_fixture):
    if not (REFERENCE_AVAILABLE and ONNX_AVAILABLE and CAFFE2_AVAILABLE):
        pytest.skip('CAFFE2 does not installed.')
    convert_onnx_to_nnp_and_compare(tmpdir, TEST_DATA_DIR,
                                    "reduce_min.onnx",
                                    "reduce_min.nnp",
                                    "ReduceElements7_Output_0", "exec_0")


def test_nnp_onnx_conversion_reduce_min(tmpdir, nnp_fixture):
    if not (REFERENCE_AVAILABLE and ONNX_AVAILABLE and CAFFE2_AVAILABLE):
        pytest.skip('CAFFE2 does not installed.')
    convert_nnp_to_onnx_and_compare(tmpdir, TEST_DATA_DIR,
                                    "reduce_min.nnp",
                                    "reduce_min.onnx",
                                    "ReduceElements7_Output_0", "exec_0")


def test_onnx_nnp_conversion_reduce_max(tmpdir, nnp_fixture):
    if not (REFERENCE_AVAILABLE and ONNX_AVAILABLE and CAFFE2_AVAILABLE):
        pytest.skip('CAFFE2 does not installed.')
    convert_onnx_to_nnp_and_compare(tmpdir, TEST_DATA_DIR,
                                    "reduce_max.onnx",
                                    "reduce_max.nnp",
                                    "ReduceElements7_Output_0", "exec_0")


def test_nnp_onnx_conversion_reduce_max(tmpdir, nnp_fixture):
    if not (REFERENCE_AVAILABLE and ONNX_AVAILABLE and CAFFE2_AVAILABLE):
        pytest.skip('CAFFE2 does not installed.')
    convert_nnp_to_onnx_and_compare(tmpdir, TEST_DATA_DIR,
                                    "reduce_max.nnp",
                                    "reduce_max.onnx",
                                    "ReduceElements7_Output_0", "exec_0")


def test_onnx_nnp_conversion_neg(tmpdir, nnp_fixture):
    if not (REFERENCE_AVAILABLE and ONNX_AVAILABLE and CAFFE2_AVAILABLE):
        pytest.skip('CAFFE2 does not installed.')
    convert_onnx_to_nnp_and_compare(tmpdir, TEST_DATA_DIR,
                                    "neg.onnx",
                                    "neg.nnp",
                                    "Negate4_Output_0", "exec_0")


def test_nnp_onnx_conversion_neg(tmpdir, nnp_fixture):
    if not (REFERENCE_AVAILABLE and ONNX_AVAILABLE and CAFFE2_AVAILABLE):
        pytest.skip('CAFFE2 does not installed.')
    convert_nnp_to_onnx_and_compare(tmpdir, TEST_DATA_DIR,
                                    "neg.nnp",
                                    "neg.onnx",
                                    "Negate4_Output_0", "exec_0")


def test_onnx_nnp_conversion_log_softmax(tmpdir, nnp_fixture):
    if not (REFERENCE_AVAILABLE and ONNX_AVAILABLE and CAFFE2_AVAILABLE):
        pytest.skip('CAFFE2 does not installed.')
    convert_onnx_to_nnp_and_compare(tmpdir, TEST_DATA_DIR,
                                    "log_softmax.onnx",
                                    "log_softmax.nnp",
                                    "Block17_Output_0", "exec_0")


def test_nnp_onnx_conversion_log_softmax(tmpdir, nnp_fixture):
    if not (REFERENCE_AVAILABLE and ONNX_AVAILABLE and CAFFE2_AVAILABLE):
        pytest.skip('CAFFE2 does not installed.')
    convert_nnp_to_onnx_and_compare(tmpdir, TEST_DATA_DIR,
                                    "log_softmax.nnp",
                                    "log_softmax.onnx",
                                    "Block17_Output_0", "exec_0")


def test_onnx_nnp_conversion_clip_maxNone_minNone(tmpdir, nnp_fixture):
    if not (REFERENCE_AVAILABLE and ONNX_AVAILABLE and CAFFE2_AVAILABLE):
        pytest.skip('CAFFE2 does not installed.')
    convert_onnx_to_nnp_and_compare(tmpdir, TEST_DATA_DIR,
                                    "clip_maxNone_minNone.onnx",
                                    "clip_maxNone_minNone.nnp",
                                    "out_data_1", "exec_0")


def test_nnp_onnx_conversion_clip_maxNone_minNone(tmpdir, nnp_fixture):
    if not (REFERENCE_AVAILABLE and ONNX_AVAILABLE and CAFFE2_AVAILABLE):
        pytest.skip('CAFFE2 does not installed.')
    convert_nnp_to_onnx_and_compare(tmpdir, TEST_DATA_DIR,
                                    "clip_maxNone_minNone.nnp",
                                    "clip_maxNone_minNone.onnx",
                                    "out_data_1", "exec_0")


def test_onnx_nnp_conversion_clip_max1_0_minNone(tmpdir, nnp_fixture):
    if not (REFERENCE_AVAILABLE and ONNX_AVAILABLE and CAFFE2_AVAILABLE):
        pytest.skip('CAFFE2 does not installed.')
    convert_onnx_to_nnp_and_compare(tmpdir, TEST_DATA_DIR,
                                    "clip_max1.0_minNone.onnx",
                                    "clip_max1.0_minNone.nnp",
                                    "out_data_1", "exec_0")


def test_nnp_onnx_conversion_clip_max1_0_minNone(tmpdir, nnp_fixture):
    if not (REFERENCE_AVAILABLE and ONNX_AVAILABLE and CAFFE2_AVAILABLE):
        pytest.skip('CAFFE2 does not installed.')
    convert_nnp_to_onnx_and_compare(tmpdir, TEST_DATA_DIR,
                                    "clip_max1.0_minNone.nnp",
                                    "clip_max1.0_minNone.onnx",
                                    "out_data_1", "exec_0")


def test_onnx_nnp_conversion_clip_maxNone_min_1_0(tmpdir, nnp_fixture):
    if not (REFERENCE_AVAILABLE and ONNX_AVAILABLE and CAFFE2_AVAILABLE):
        pytest.skip('CAFFE2 does not installed.')
    convert_onnx_to_nnp_and_compare(tmpdir, TEST_DATA_DIR,
                                    "clip_maxNone_min-1.0.onnx",
                                    "clip_maxNone_min-1.0.nnp",
                                    "out_data_1", "exec_0")


def test_nnp_onnx_conversion_clip_maxNone_min_1_0(tmpdir, nnp_fixture):
    if not (REFERENCE_AVAILABLE and ONNX_AVAILABLE and CAFFE2_AVAILABLE):
        pytest.skip('CAFFE2 does not installed.')
    convert_nnp_to_onnx_and_compare(tmpdir, TEST_DATA_DIR,
                                    "clip_maxNone_min-1.0.nnp",
                                    "clip_maxNone_min-1.0.onnx",
                                    "out_data_1", "exec_0")


def test_onnx_nnp_conversion_clip_max1_0_min_1_0(tmpdir, nnp_fixture):
    if not (REFERENCE_AVAILABLE and ONNX_AVAILABLE and CAFFE2_AVAILABLE):
        pytest.skip('CAFFE2 does not installed.')
    convert_onnx_to_nnp_and_compare(tmpdir, TEST_DATA_DIR,
                                    "clip_max1.0_min-1.0.onnx",
                                    "clip_max1.0_min-1.0.nnp",
                                    "out_data_1", "exec_0")


def test_nnp_onnx_conversion_clip_max1_0_min_1_0(tmpdir, nnp_fixture):
    if not (REFERENCE_AVAILABLE and ONNX_AVAILABLE and CAFFE2_AVAILABLE):
        pytest.skip('CAFFE2 does not installed.')
    convert_nnp_to_onnx_and_compare(tmpdir, TEST_DATA_DIR,
                                    "clip_max1.0_min-1.0.nnp",
                                    "clip_max1.0_min-1.0.onnx",
                                    "out_data_1", "exec_0")


def test_onnx_nnp_conversion_softplus(tmpdir, nnp_fixture):
    if not (REFERENCE_AVAILABLE and ONNX_AVAILABLE and CAFFE2_AVAILABLE):
        pytest.skip('CAFFE2 does not installed.')
    convert_onnx_to_nnp_and_compare(tmpdir, TEST_DATA_DIR,
                                    "softplus.onnx",
                                    "softplus.nnp",
                                    "out_data_1", "exec_0")


def test_nnp_onnx_conversion_softplus(tmpdir, nnp_fixture):
    if not (REFERENCE_AVAILABLE and ONNX_AVAILABLE and CAFFE2_AVAILABLE):
        pytest.skip('CAFFE2 does not installed.')
    convert_nnp_to_onnx_and_compare(tmpdir, TEST_DATA_DIR,
                                    "softplus.nnp",
                                    "softplus.onnx",
                                    "out_data_1", "exec_0")


def test_onnx_nnp_conversion_softsign(tmpdir, nnp_fixture):
    if not (REFERENCE_AVAILABLE and ONNX_AVAILABLE and CAFFE2_AVAILABLE):
        pytest.skip('CAFFE2 does not installed.')
    convert_onnx_to_nnp_and_compare(tmpdir, TEST_DATA_DIR,
                                    "softsign.onnx",
                                    "softsign.nnp",
                                    "out_data_1", "exec_0")


def test_nnp_onnx_conversion_softsign(tmpdir, nnp_fixture):
    if not (REFERENCE_AVAILABLE and ONNX_AVAILABLE and CAFFE2_AVAILABLE):
        pytest.skip('CAFFE2 does not installed.')
    convert_nnp_to_onnx_and_compare(tmpdir, TEST_DATA_DIR,
                                    "softsign.nnp",
                                    "softsign.onnx",
                                    "out_data_1", "exec_0")


def test_onnx_nnp_conversion_lrn_c4_s3(tmpdir, nnp_fixture):
    if not (REFERENCE_AVAILABLE and ONNX_AVAILABLE and CAFFE2_AVAILABLE):
        pytest.skip('CAFFE2 does not installed.')
    convert_onnx_to_nnp_and_compare(tmpdir, TEST_DATA_DIR,
                                    "lrn_c4_s3.onnx",
                                    "lrn_c4_s3.nnp",
                                    "out_data_1", "exec_0")


def test_nnp_onnx_conversion_lrn_c4_s3(tmpdir, nnp_fixture):
    if not (REFERENCE_AVAILABLE and ONNX_AVAILABLE and CAFFE2_AVAILABLE):
        pytest.skip('CAFFE2 does not installed.')
    convert_nnp_to_onnx_and_compare(tmpdir, TEST_DATA_DIR,
                                    "lrn_c4_s3.nnp",
                                    "lrn_c4_s3.onnx",
                                    "out_data_1", "exec_0",
                                    atol=1e-4)


def test_onnx_nnp_conversion_pad_mconstant_v0_pl0_0_0_1_0_1(tmpdir,
                                                            nnp_fixture):
    if not (REFERENCE_AVAILABLE and ONNX_AVAILABLE and CNTK_AVAILABLE):
        pytest.skip('CNTK does not installed.')
    convert_onnx_to_nnp_and_compare(tmpdir, TEST_DATA_DIR,
                                    "pad_mconstant_v0_pl0_0_0_1_0_1.onnx",
                                    "pad_mconstant_v0_pl0_0_0_1_0_1.nnp",
                                    "Pad4_Output_0", "exec_0",
                                    backend="cntk")


def test_nnp_onnx_conversion_pad_mconstant_v0_pl0_0_0_1_0_1(tmpdir,
                                                            nnp_fixture):
    if not (REFERENCE_AVAILABLE and ONNX_AVAILABLE and CNTK_AVAILABLE):
        pytest.skip('CNTK does not installed.')
    convert_nnp_to_onnx_and_compare(tmpdir, TEST_DATA_DIR,
                                    "pad_mconstant_v0_pl0_0_0_1_0_1.nnp",
                                    "pad_mconstant_v0_pl0_0_0_1_0_1.onnx",
                                    "Pad4_Output_0", "exec_0",
                                    backend="cntk")


def test_onnx_nnp_conversion_reduce_prod(tmpdir, nnp_fixture):
    if not (REFERENCE_AVAILABLE and ONNX_AVAILABLE and CNTK_AVAILABLE):
        pytest.skip('CNTK does not installed.')
    convert_onnx_to_nnp_and_compare(tmpdir, TEST_DATA_DIR,
                                    "reduce_prod.onnx",
                                    "reduce_prod.nnp",
                                    "ReduceElements7_Output_0", "exec_0",
                                    backend="cntk")


def test_nnp_onnx_conversion_reduce_prod(tmpdir, nnp_fixture):
    if not (REFERENCE_AVAILABLE and ONNX_AVAILABLE and CAFFE2_AVAILABLE):
        pytest.skip('CAFFE2 does not installed.')
    if not (REFERENCE_AVAILABLE and ONNX_AVAILABLE and CNTK_AVAILABLE):
        pytest.skip('CNTK does not installed.')
    convert_nnp_to_onnx_and_compare(tmpdir, TEST_DATA_DIR,
                                    "reduce_prod.nnp",
                                    "reduce_prod.onnx",
                                    "ReduceElements7_Output_0", "exec_0",
                                    backend="cntk")

# Even sized LRN is not tested because we only support
# Odd sizes for now.
# def test_onnx_nnp_conversion_lrn_c3_s2(tmpdir, nnp_fixture):
#    if not (REFERENCE_AVAILABLE and ONNX_AVAILABLE and CAFFE2_AVAILABLE):
#        pytest.skip('CAFFE2 does not installed.')
#    convert_onnx_to_nnp_and_compare(tmpdir, TEST_DATA_DIR,
#                                    "lrn_c3_s2.onnx",
#                                    "lrn_c3_s2.nnp",
#                                    "out_data_1", "exec_0")


# def test_onnx_nnp_conversion_squeezenet(tmpdir, nnp_fixture):
#    if not (REFERENCE_AVAILABLE and ONNX_AVAILABLE and CAFFE2_AVAILABLE):
#        pytest.skip('CAFFE2 does not installed.')
#     img = np.random.rand(1, 3, 224, 224).astype(np.float32)
#     convert_onnx_to_nnp_and_compare(tmpdir, TEST_DATA_DIR,
#                                     "squeezenet.onnx", "squeezenet.nnp",
#                                     "softmaxout_1", "exec_0",
#                                     in_name="data_0", in_img=img)


# def test_nnp_onnx_conversion_squeezenet(tmpdir, nnp_fixture):
#    if not (REFERENCE_AVAILABLE and ONNX_AVAILABLE and CAFFE2_AVAILABLE):
#        pytest.skip('CAFFE2 does not installed.')
#     img = np.random.rand(1, 3, 224, 224).astype(np.float32)
#     convert_nnp_to_onnx_and_compare(tmpdir, TEST_DATA_DIR,
#                                     "squeezenet.nnp", "squeezenet.onnx",
#                                     "softmaxout_1", "exec_0",
#                                     in_name="data_0", in_img=img)


# def test_nnp_onnx_conversion_lenet(tmpdir, nnp_fixture):
#    if not (REFERENCE_AVAILABLE and ONNX_AVAILABLE and CAFFE2_AVAILABLE):
#        pytest.skip('CAFFE2 does not installed.')
#     img = np.random.rand(128, 1, 28, 28).astype(np.float32)
#     convert_nnp_to_onnx_and_compare(tmpdir, TEST_DATA_DIR,
#                                     "lenet_result.nnp",
#                                     "lenet_result.onnx",
#                                     "y", "Runtime",
#                                     in_name="x", in_img=img)


# @pytest.mark.slow
# def test_onnx_nnp_conversion_inception_v2(tmpdir, nnp_fixture):
#    if not (REFERENCE_AVAILABLE and ONNX_AVAILABLE and CAFFE2_AVAILABLE):
#        pytest.skip('CAFFE2 does not installed.')
#     img = np.random.rand(1, 3, 224, 224).astype(np.float32)
#     convert_onnx_to_nnp_and_compare(tmpdir, TEST_DATA_DIR,
#                                     "inception_v2.onnx", "inception_v2.nnp",
#                                     "prob_1", "exec_0",
#                                     in_name="data_0", in_img=img)


# @pytest.mark.slow
# def test_nnp_onnx_conversion_inception_v2(tmpdir, nnp_fixture):
#    if not (REFERENCE_AVAILABLE and ONNX_AVAILABLE and CAFFE2_AVAILABLE):
#        pytest.skip('CAFFE2 does not installed.')
#     img = np.random.rand(1, 3, 224, 224).astype(np.float32)
#     convert_nnp_to_onnx_and_compare(tmpdir, TEST_DATA_DIR,
#                                     "inception_v2.nnp", "inception_v2.onnx",
#                                     "prob_1", "exec_0",
#                                     in_name="data_0", in_img=img)


# @pytest.mark.slow
# def test_onnx_nnp_conversion_densenet121(tmpdir, nnp_fixture):
#    if not (REFERENCE_AVAILABLE and ONNX_AVAILABLE and CAFFE2_AVAILABLE):
#        pytest.skip('CAFFE2 does not installed.')
#     img = np.random.rand(1, 3, 224, 224).astype(np.float32)
#     convert_onnx_to_nnp_and_compare(tmpdir, TEST_DATA_DIR,
#                                     "densenet121.onnx", "densenet121.nnp",
#                                     "fc6_1", "exec_0",
#                                     in_name="data_0", in_img=img, atol=1e-5)


# @pytest.mark.slow
# def test_nnp_onnx_conversion_densenet121(tmpdir, nnp_fixture):
#    if not (REFERENCE_AVAILABLE and ONNX_AVAILABLE and CAFFE2_AVAILABLE):
#        pytest.skip('CAFFE2 does not installed.')
#     img = np.random.rand(1, 3, 224, 224).astype(np.float32)
#     convert_nnp_to_onnx_and_compare(tmpdir, TEST_DATA_DIR,
#                                     "densenet121.nnp", "densenet121.onnx",
#                                     "fc6_1", "exec_0",
#                                     in_name="data_0", in_img=img, atol=1e-5)


# @pytest.mark.slow
# def test_onnx_nnp_conversion_resnet50(tmpdir, nnp_fixture):
#    if not (REFERENCE_AVAILABLE and ONNX_AVAILABLE and CAFFE2_AVAILABLE):
#        pytest.skip('CAFFE2 does not installed.')
#     img = np.random.rand(1, 3, 224, 224).astype(np.float32)
#     convert_onnx_to_nnp_and_compare(tmpdir, TEST_DATA_DIR,
#                                     "resnet50.onnx", "resnet50.nnp",
#                                     "gpu_0/softmax_1", "exec_0",
#                                     in_name="gpu_0/data_0", in_img=img,
#                                     atol=1e-5)


# @pytest.mark.slow
# def test_nnp_onnx_conversion_resnet50(tmpdir, nnp_fixture):
#    if not (REFERENCE_AVAILABLE and ONNX_AVAILABLE and CAFFE2_AVAILABLE):
#        pytest.skip('CAFFE2 does not installed.')
#     img = np.random.rand(1, 3, 224, 224).astype(np.float32)
#     convert_nnp_to_onnx_and_compare(tmpdir, TEST_DATA_DIR,
#                                     "resnet50.nnp", "resnet50.onnx",
#                                     "gpu_0/softmax_1", "exec_0",
#                                     in_name="gpu_0/data_0", in_img=img,
#                                     atol=1e-5)


# @pytest.mark.slow
# def test_onnx_nnp_conversion_vgg19(tmpdir, nnp_fixture):
#    if not (REFERENCE_AVAILABLE and ONNX_AVAILABLE and CAFFE2_AVAILABLE):
#        pytest.skip('CAFFE2 does not installed.')
#     img = np.random.rand(1, 3, 224, 224).astype(np.float32)
#     convert_onnx_to_nnp_and_compare(
#         tmpdir, TEST_DATA_DIR, "vgg19.onnx", "vgg19.nnp", "prob_1", "exec_0",
#         in_name="data_0", in_img=img)


# @pytest.mark.slow
# def test_nnp_onnx_conversion_vgg19(tmpdir, nnp_fixture):
#    if not (REFERENCE_AVAILABLE and ONNX_AVAILABLE and CAFFE2_AVAILABLE):
#        pytest.skip('CAFFE2 does not installed.')
#     img = np.random.rand(1, 3, 224, 224).astype(np.float32)
#     convert_nnp_to_onnx_and_compare(
#         tmpdir, TEST_DATA_DIR, "vgg19.nnp", "vgg19.onnx", "prob_1", "exec_0",
#         in_name="data_0", in_img=img)


# @pytest.mark.slow
# def test_onnx_nnp_conversion_zfnet512(tmpdir, nnp_fixture):
#    if not (REFERENCE_AVAILABLE and ONNX_AVAILABLE and CAFFE2_AVAILABLE):
#        pytest.skip('CAFFE2 does not installed.')
#     img = np.random.rand(1, 3, 224, 224).astype(np.float32)
#     convert_onnx_to_nnp_and_compare(tmpdir, TEST_DATA_DIR,
#                                     "zfnet512.onnx", "zfnet512.nnp",
#                                     "gpu_0/softmax_1", "exec_0",
#                                     in_name="gpu_0/data_0", in_img=img)


# @pytest.mark.slow
# def test_nnp_onnx_conversion_zfnet512(tmpdir, nnp_fixture):
#    if not (REFERENCE_AVAILABLE and ONNX_AVAILABLE and CAFFE2_AVAILABLE):
#        pytest.skip('CAFFE2 does not installed.')
#     img = np.random.rand(1, 3, 224, 224).astype(np.float32)
#     convert_nnp_to_onnx_and_compare(tmpdir, TEST_DATA_DIR,
#                                     "zfnet512.nnp", "zfnet512.onnx",
#                                     "gpu_0/softmax_1", "exec_0",
#                                     in_name="gpu_0/data_0", in_img=img)


# @pytest.mark.slow
# def test_onnx_nnp_conversion_bvlc_googlenet(tmpdir, nnp_fixture):
#    if not (REFERENCE_AVAILABLE and ONNX_AVAILABLE and CAFFE2_AVAILABLE):
#        pytest.skip('CAFFE2 does not installed.')
#     img = np.random.rand(1, 3, 224, 224).astype(np.float32)
#     convert_onnx_to_nnp_and_compare(tmpdir, TEST_DATA_DIR,
#                                     "bvlc_googlenet.onnx",
#                                     "bvlc_googlenet.nnp",
#                                     "prob_1", "exec_0",
#                                     in_name="data_0", in_img=img)


# @pytest.mark.slow
# def test_nnp_onnx_conversion_bvlc_googlenet(tmpdir, nnp_fixture):
#    if not (REFERENCE_AVAILABLE and ONNX_AVAILABLE and CAFFE2_AVAILABLE):
#        pytest.skip('CAFFE2 does not installed.')
#     img = np.random.rand(1, 3, 224, 224).astype(np.float32)
#     convert_nnp_to_onnx_and_compare(tmpdir, TEST_DATA_DIR,
#                                     "bvlc_googlenet.nnp",
#                                     "bvlc_googlenet.onnx",
#                                     "prob_1", "exec_0",
#                                     in_name="data_0", in_img=img,
#                                     atol=1e-5)


# @pytest.mark.slow
# def test_onnx_nnp_conversion_bvlc_caffenet(tmpdir, nnp_fixture):
#    if not (REFERENCE_AVAILABLE and ONNX_AVAILABLE and CAFFE2_AVAILABLE):
#        pytest.skip('CAFFE2 does not installed.')
#     img = np.random.rand(1, 3, 224, 224).astype(np.float32)
#     convert_onnx_to_nnp_and_compare(tmpdir, TEST_DATA_DIR,
#                                     "bvlc_caffenet.onnx", "bvlc_caffenet.nnp",
#                                     "prob_1", "exec_0",
#                                     in_name="data_0", in_img=img)


# @pytest.mark.slow
# def test_nnp_onnx_conversion_bvlc_caffenet(tmpdir, nnp_fixture):
#    if not (REFERENCE_AVAILABLE and ONNX_AVAILABLE and CAFFE2_AVAILABLE):
#        pytest.skip('CAFFE2 does not installed.')
#     img = np.random.rand(1, 3, 224, 224).astype(np.float32)
#     convert_nnp_to_onnx_and_compare(tmpdir, TEST_DATA_DIR,
#                                     "bvlc_caffenet.nnp",
#                                     "bvlc_caffenet.onnx",
#                                     "prob_1", "exec_0",
#                                     in_name="data_0", in_img=img)


# @pytest.mark.slow
# def test_onnx_nnp_conversion_shufflenet(tmpdir, nnp_fixture):
#    if not (REFERENCE_AVAILABLE and ONNX_AVAILABLE and CAFFE2_AVAILABLE):
#        pytest.skip('CAFFE2 does not installed.')
#    img = np.random.rand(1, 3, 224, 224).astype(np.float32)
#    convert_onnx_to_nnp_and_compare(tmpdir, TEST_DATA_DIR,
#                                    "shufflenet.onnx", "shufflenet.nnp",
#                                    "gpu_0/softmax_1", "exec_0",
#                                    in_name="gpu_0/data_0", in_img=img)


# @pytest.mark.slow
# def test_onnx_nnp_conversion_mnist(tmpdir, nnp_fixture):
#    if not (REFERENCE_AVAILABLE and ONNX_AVAILABLE and CNTK_AVAILABLE):
#        pytest.skip('CNTK does not installed.')
#    img = np.random.rand(1, 1, 28, 28).astype(np.float32)
#    convert_onnx_to_nnp_and_compare(tmpdir, TEST_DATA_DIR,
#                                    "mnist.onnx", "mnist.nnp",
#                                    "Plus214_Output_0", "exec_0",
#                                    in_name="Input3", in_img=img,
#                                    backend="cntk",
#                                    export_nnp_path=TEST_DATA_DIR)

# @pytest.mark.slow
# def test_onnx_nnp_conversion_inception_v1(tmpdir, nnp_fixture):
#    if not (REFERENCE_AVAILABLE and ONNX_AVAILABLE and CAFFE2_AVAILABLE):
#        pytest.skip('CAFFE2 does not installed.')
#    img = np.random.rand(1, 3, 224, 224).astype(np.float32)
#    convert_onnx_to_nnp_and_compare(tmpdir, TEST_DATA_DIR,
#                                    "inception_v1.onnx", "inception_v1.nnp",
#                                    "prob_1", "exec_0",
#                                    in_name="data_0", in_img=img)
