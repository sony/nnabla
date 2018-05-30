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

NNABLA_OPTIONS_INCLUDED = True

########################################################################################################################
# Environments

DOCKER_RUN_OPTS += -v $(HOME)/.ccache:/tmp/.ccache

## If your environment is under proxy uncomment following lines.
DOCKER_BUILD_ARGS = --build-arg http_proxy=${http_proxy} --build-arg https_proxy=${https_proxy}
DOCKER_RUN_OPTS += -e http_proxy=${http_proxy}
DOCKER_RUN_OPTS += -e https_proxy=${https_proxy}
DOCKER_RUN_OPTS += -e ftp_proxy=${ftp_proxy}

########################################################################################################################
# Settings
export PYTHON_VERSION_MAJOR ?= $(shell python -c 'import sys;print("{}".format(sys.version_info.major))')
DOCKER_RUN_OPTS += -e PYTHON_VERSION_MAJOR=$(PYTHON_VERSION_MAJOR)
export PYTHON_VERSION_MINOR ?= $(shell python -c 'import sys;print("{}".format(sys.version_info.minor))')
DOCKER_RUN_OPTS += -e PYTHON_VERSION_MINOR=$(PYTHON_VERSION_MINOR)

########################################################################################################################
# Build options
export NNABLA_VERSION ?= $(shell git describe --tags 2>/dev/null|sed -e 's/^v//' -e 's/-/.post/' -e 's/-/+/')
DOCKER_RUN_OPTS += -e NNABLA_VERSION=$(NNABLA_VERSION)

export NNABLA_SHORT_VERSION ?= $(shell git describe --abbrev=0 --tags 2>/dev/null|sed -e 's/^v//' -e 's/\//_/g')
DOCKER_RUN_OPTS += -e NNABLA_SHORT_VERSION=$(NNABLA_SHORT_VERSION)

NNABLA_UTILS_STATIC_LINK_DEPS ?= OFF
DOCKER_RUN_OPTS += -e NNABLA_UTILS_STATIC_LINK_DEPS=$(NNABLA_UTILS_STATIC_LINK_DEPS)

NNABLA_UTILS_WITH_HDF5 ?= OFF
DOCKER_RUN_OPTS += -e NNABLA_UTILS_WITH_HDF5=$(NNABLA_UTILS_WITH_HDF5)

MAKE_MANYLINUX_WHEEL ?= OFF
DOCKER_RUN_OPTS += -e MAKE_MANYLINUX_WHEEL=$(MAKE_MANYLINUX_WHEEL)

PARALLEL_BUILD_NUM ?= 8
DOCKER_RUN_OPTS += -e PARALLEL_BUILD_NUM=$(PARALLEL_BUILD_NUM)

########################################################################################################################
# Output directories

export BUILD_DIRECTORY_CPPLIB ?= $(NNABLA_DIRECTORY)/build$(BUILD_DIRECTORY_CPPLIB_SUFFIX)
DOCKER_RUN_OPTS += -e BUILD_DIRECTORY_CPPLIB=$(BUILD_DIRECTORY_CPPLIB)

export BUILD_DIRECTORY_WHEEL ?= $(NNABLA_DIRECTORY)/build_wheel$(BUILD_DIRECTORY_WHEEL_SUFFIX)
DOCKER_RUN_OPTS += -e BUILD_DIRECTORY_WHEEL=$(BUILD_DIRECTORY_WHEEL)

export DOC_DIRECTORY ?= $(NNABLA_DIRECTORY)/build-doc
DOCKER_RUN_OPTS += -e DOC_DIRECTORY=$(DOC_DIRECTORY)

export DOCKER_RUN_OPTS

########################################################################################################################
# Functions for makefile
define with-virtualenv
	rm -rf $(2)
	virtualenv --system-site-packages $(2)
	. $(2)/bin/activate \
	&& $(MAKE) -C $(1) $(2) $(3) $(4)\
	&& deactivate
	rm -rf $(2)
endef


