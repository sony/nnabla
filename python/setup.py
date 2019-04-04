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

from __future__ import print_function

from setuptools import setup
from distutils.extension import Extension
import os
import shutil
import sys
from collections import namedtuple
import copy

setup_requires = [
    'setuptools',
    'numpy>=1.16',
    'Cython',  # Requires python-dev.
]

install_requires = setup_requires + [
    'boto3',
    'configparser',
    'contextlib2',
    'h5py',
    'protobuf>=3.6',
    'pyyaml',
    'requests',
    'scipy',
    'tqdm',
    'imageio',
    'pillow'
]


def extopts(library_name, library_dir):
    import numpy as np
    include_dir = os.path.realpath(os.path.join(
        os.path.dirname(__file__), '../include'))
    ext_opts = dict(
        include_dirs=[include_dir, np.get_include()],
        libraries=[library_name],
        library_dirs=[library_dir],
        language="c++",
        # The below definition breaks build. Use -Wcpp instead.
        # define_macros=[('NPY_NO_DEPRECATED_API', 'NPY_1_7_API_VERSION')],
    )
    if sys.platform == 'darwin':
        # macOS
        # References
        # * About @loader_path
        #   <https://wincent.com/wiki/%40executable_path%2C_%40load_path_and_%40rpath>
        # * Example of setting @loader_path
        #   <https://github.com/X-DataInitiative/tick/blob/master/setup.py>
        # * libc++
        #   <https://stackoverflow.com/questions/10116724/clang-os-x-lion-cannot-find-cstdint>
        ext_opts.update(dict(
            extra_compile_args=[
                '-std=c++11', '-stdlib=libc++', '-Wno-sign-compare',
                '-Wno-unused-function', '-Wno-mismatched-tags',
                '-mmacosx-version-min=10.7'],
            extra_link_args=['-Wl,-rpath,@loader_path/', '-stdlib=libc++',
                             '-mmacosx-version-min=10.7'],
        ))
    elif sys.platform != 'win32':
        # Linux
        ext_opts.update(dict(
            extra_compile_args=[
                '-std=c++11', '-Wno-sign-compare', '-Wno-unused-function', '-Wno-cpp'],
            runtime_library_dirs=['$ORIGIN/'],
        ))
    else:
        # Assume Windows.
        ext_opts.update(dict(extra_compile_args=['/W0', '/EHsc']))
    return ext_opts

################################################################################
# Main


if __name__ == '__main__':
    from Cython.Build import cythonize

    ############################################################################
    # Get version info
    root_dir = os.path.realpath(os.path.dirname(__file__))
    a = dict()
    exec(open(os.path.join(root_dir, 'src', 'nnabla',
                           '_version.py')).read(), globals(), a)
    if '__version__' in a:
        __version__ = a['__version__']
    if '__author__' in a:
        __author__ = a['__author__']
    if '__email__' in a:
        __email__ = a['__email__']

    ############################################################################
    # Package information

    pkg_name = "nnabla"
    if 'WHEEL_SUFFIX' in os.environ:
        pkg_name += os.environ['WHEEL_SUFFIX']

    pkg_info = dict(
        name=pkg_name,
        description='Neural Network Libraries',
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
            'Programming Language :: C++',
            'Programming Language :: Python :: 2',
            'Programming Language :: Python :: 2.7',
            'Programming Language :: Python :: 3',
            'Programming Language :: Python :: 3.5',
            'Programming Language :: Python :: 3.6',
            'Programming Language :: Python :: Implementation :: CPython',
            'Operating System :: Microsoft :: Windows',
            'Operating System :: POSIX :: Linux',
            'Operating System :: MacOS :: MacOS X'
        ],
        keywords="deep learning artificial intelligence machine learning neural network",
        python_requires='>=2.7,!=3.0.*,!=3.1.*,!=3.2.*,!=3.3.*,!=3.4.*',
    )

    ############################################################################
    # Parse setup.cfg
    from six.moves.configparser import ConfigParser
    path_cfg = os.path.join(os.path.dirname(__file__), "setup.cfg")
    if not os.path.isfile(path_cfg):
        raise ValueError(
            "`setup.cfg` does not exist. Read installation document and install using CMake.")
    cfgp = ConfigParser()
    cfgp.read(path_cfg)
    build_dir = cfgp.get("cmake", "build_dir")

    ############################################################################
    # Extension module information
    src_dir = os.path.join(root_dir, 'src')
    path_pkg = os.path.join(src_dir, 'nnabla')

    library_name = cfgp.get("cmake", "target_name")
    library_file_name = cfgp.get("cmake", "target_file_name")
    library_path = cfgp.get("cmake", "target_file")
    library_dir = os.path.dirname(library_path)

    ext_opts = extopts(library_name, library_dir)

    module_names = [
        '_variable',
        'function',
        'solver',
        'communicator',
        '_init',
        '_nd_array',
        '_computation_graph',
        '_array',
        '_arithmetic_ops',
        '_indexing']

    ext_modules = [Extension('nnabla.{}'.format(mname),
                             [os.path.join(path_pkg,
                                           '{}.pyx'.format(mname))],
                             **ext_opts) for mname in module_names]

    # Cythonize
    ext_modules = cythonize(ext_modules, compiler_directives={
                            "embedsignature": True,
                            "c_string_type": 'str',
                            "c_string_encoding": "ascii"})

    ############################################################################
    # Package data
    # Move shared libs to module
    # http://stackoverflow.com/questions/6191942/distributing-pre-built-libraries-with-python-modules
    # Packaging shared lib
    # http://stackoverflow.com/questions/6191942/distributing-pre-built-libraries-with-python-modules

    shutil.copyfile(library_path, os.path.join(path_pkg, library_file_name))
    package_data = {"nnabla": [
        library_file_name,
        'nnabla.conf',
        'utils/converter/functions.pkl',
        'models/imagenet/category_names.txt',
    ]}

    for root, dirs, files in os.walk(os.path.join(build_dir, 'bin')):
        for fn in files:
            if os.path.splitext(fn)[1] == '' or os.path.splitext(fn)[1] == '.exe':
                if not os.path.isdir(os.path.join(path_pkg, 'bin')):
                    os.makedirs(os.path.join(path_pkg, 'bin'))
                shutil.copyfile(os.path.join(root, fn),
                                os.path.join(path_pkg, 'bin', fn))
                os.chmod(os.path.join(path_pkg, 'bin', fn), 0o755)
                package_data["nnabla"].append(os.path.join('bin', fn))

    for root, dirs, files in os.walk(os.path.join(build_dir, 'lib')):
        for fn in files:
            if os.path.splitext(fn)[1] == '.so' or os.path.splitext(fn)[1] == '.dylib':
                if not os.path.isdir(os.path.join(path_pkg, 'bin')):
                    os.makedirs(os.path.join(path_pkg, 'bin'))
                shutil.copyfile(os.path.join(root, fn),
                                os.path.join(path_pkg, 'bin', fn))
                os.chmod(os.path.join(path_pkg, 'bin', fn), 0o755)
                package_data["nnabla"].append(os.path.join('bin', fn))

    export_lib = ''
    # Read NNabla lib info
    if sys.platform == 'win32':
        for root, dirs, files in os.walk(os.path.join(build_dir, 'bin')):
            for fn in files:
                if os.path.splitext(fn)[1] == '.lib':
                    shutil.copyfile(os.path.join(root, fn),
                                    os.path.join(path_pkg, fn))
                    package_data["nnabla"].append(fn)

    package_dir = {'': src_dir}

    packages = ['nnabla',
                'nnabla.contrib',
                'nnabla.experimental',
                'nnabla.experimental.graph_converters',
                'nnabla.experimental.parametric_function_class',
                'nnabla.experimental.trainers',
                'nnabla.models',
                'nnabla.models.imagenet',
                'nnabla.models.object_detection',
                'nnabla.utils',
                'nnabla.utils.cli',
                'nnabla.utils.converter',
                'nnabla.utils.converter.nnabla',
                'nnabla.utils.converter.nnablart',
                'nnabla.utils.converter.onnx',
                'nnabla.utils.converter.tensorflow',
                'nnabla.utils.factorization',
                'nnabla.utils.image_utils',
                'nnabla_ext',
                'nnabla_ext.cpu', ]

    # Setup
    setup(
        entry_points={"console_scripts":
                      ["nnabla_cli=nnabla.utils.cli.cli:main"]},
        setup_requires=setup_requires,
        install_requires=install_requires,
        extras_require={
            ':python_version == "2.7"': ['futures'],
            ':(python_version != "2.7" and python_version != "3.7")': ['onnx']
        },
        ext_modules=ext_modules,
        package_dir=package_dir,
        packages=packages,
        package_data=package_data,
        namespace_packages=['nnabla_ext'],
        **pkg_info)

    os.unlink(os.path.join(root_dir, 'src', 'nnabla', library_file_name))
    shutil.rmtree(os.path.join(root_dir, 'src',
                               'nnabla', 'dev'), ignore_errors=True)
    shutil.rmtree(os.path.join(root_dir, 'src',
                               'nnabla', 'bin'), ignore_errors=True)
