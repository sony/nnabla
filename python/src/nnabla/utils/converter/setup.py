# Copyright 2020,2021 Sony Corporation.
# Copyright 2021 Sony Group Corporation.
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

from __future__ import print_function

import os

from setuptools import setup

if __name__ == '__main__':
    ############################################################################
    # Get version info
    root_dir = os.path.realpath(os.path.dirname(__file__))
    a = dict()
    exec(open(os.path.join(root_dir, '../../', '_version.py')).read(), globals(), a)

    if '__version__' in a:
        __version__ = a['__version__']
    if '__author__' in a:
        __author__ = a['__author__']
    if '__email__' in a:
        __email__ = a['__email__']

    install_requires = [
        'ply',
        'tensorflow==2.5.2',
        'onnx_tf',
        'tf2onnx==1.7.2',
        'tensorflow-addons',
        'onnx==1.9.0',
        'tflite2onnx',
        'flatbuffers'
    ]

    ############################################################################
    # Package information

    pkg_name = "nnabla_converter"
    tensorflow_src_dir = os.path.join(root_dir, 'tensorflow')
    onnx_src_dir = os.path.join(root_dir, 'onnx')
    tflite_src_dir = os.path.join(root_dir, "tflite")

    package_data = {"nnabla.utils.converter.tflite": [
        'schema.fbs',
    ]}

    # Setup
    setup(
        name=pkg_name,
        description='NNabla File Format Converter',
        version=__version__,
        author=__author__,
        author_email=__email__,
        url="https://github.com/sony/nnabla",
        license='Apache License 2.0',
        classifiers=[
            'Development Status :: 5 - Production/Stable',
            'Intended Audience :: Developers',
            'Intended Audience :: Education',
            'Intended Audience :: Science/Research',
            'Topic :: Scientific/Engineering',
            'Topic :: Scientific/Engineering :: Artificial Intelligence',
            'License :: OSI Approved :: Apache Software License',
            'Programming Language :: Python :: 3.6',
            'Programming Language :: Python :: 3.7',
            'Programming Language :: Python :: 3.8'
        ],
        packages=['nnabla.utils.converter.tensorflow',
                  'nnabla.utils.converter.onnx',
                  'nnabla.utils.converter.tflite'],
        package_dir={'nnabla.utils.converter.tensorflow': tensorflow_src_dir,
                     'nnabla.utils.converter.onnx': onnx_src_dir,
                     'nnabla.utils.converter.tflite': tflite_src_dir},
        package_data=package_data,
        install_requires=install_requires,
        keywords="NNabla File Format Converter",
        python_requires='>=3.6',
    )
