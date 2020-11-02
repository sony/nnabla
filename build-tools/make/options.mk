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

DOCKER_RUN_OPTS +=--rm
DOCKER_RUN_OPTS += -v $$(pwd):$$(pwd)
DOCKER_RUN_OPTS += -w $$(pwd)
DOCKER_RUN_OPTS += -u $$(id -u):$$(id -g)
DOCKER_RUN_OPTS += -e HOME=/tmp
DOCKER_RUN_OPTS += -e CMAKE_OPTS=$(CMAKE_OPTS)

DOCKER_RUN_OPTS += -v $(HOME)/.ccache:/tmp/.ccache

## If your environment is under proxy uncomment following lines.
DOCKER_BUILD_ARGS = --build-arg http_proxy=${http_proxy} --build-arg https_proxy=${https_proxy}
# DOCKER_BUILD_ARGS += --no-cache

DOCKER_RUN_OPTS += -e http_proxy=${http_proxy}
DOCKER_RUN_OPTS += -e https_proxy=${https_proxy}
DOCKER_RUN_OPTS += -e ftp_proxy=${ftp_proxy}

########################################################################################################################
# Settings
PYTHON_VERSION_MAJOR ?= $(shell python3 -c 'import sys;print("{}".format(sys.version_info.major))')
export PYTHON_VERSION_MAJOR
DOCKER_RUN_OPTS += -e PYTHON_VERSION_MAJOR=$(PYTHON_VERSION_MAJOR)
DOCKER_BUILD_ARGS += --build-arg PYTHON_VERSION_MAJOR=${PYTHON_VERSION_MAJOR}

PYTHON_VERSION_MINOR ?= $(shell python3 -c 'import sys;print("{}".format(sys.version_info.minor))')
export PYTHON_VERSION_MINOR
DOCKER_RUN_OPTS += -e PYTHON_VERSION_MINOR=$(PYTHON_VERSION_MINOR)
DOCKER_BUILD_ARGS += --build-arg PYTHON_VERSION_MINOR=${PYTHON_VERSION_MINOR}

ifeq ($(PYTHON_VERSION_MAJOR), 3)
	DOCKER_BUILD_ARGS += --build-arg PYTHON_LIB_SUFFIX=m
endif

########################################################################################################################
# Build options
DOCKER_RUN_OPTS += -e NNABLA_VERSION=$(NNABLA_VERSION)

NNABLA_UTILS_STATIC_LINK_DEPS ?= OFF
DOCKER_RUN_OPTS += -e NNABLA_UTILS_STATIC_LINK_DEPS=$(NNABLA_UTILS_STATIC_LINK_DEPS)

NNABLA_UTILS_WITH_HDF5 ?= OFF
DOCKER_RUN_OPTS += -e NNABLA_UTILS_WITH_HDF5=$(NNABLA_UTILS_WITH_HDF5)

MAKE_MANYLINUX_WHEEL ?= OFF
DOCKER_RUN_OPTS += -e MAKE_MANYLINUX_WHEEL=$(MAKE_MANYLINUX_WHEEL)

PARALLEL_BUILD_NUM ?= 8
DOCKER_RUN_OPTS += -e PARALLEL_BUILD_NUM=$(PARALLEL_BUILD_NUM)

SUFFIX ?=
WHEEL_SUFFIX ?= $(SUFFIX)
DOCKER_RUN_OPTS += -e WHEEL_SUFFIX=$(WHEEL_SUFFIX)

ARCH_SUFFIX ?= $(shell bash -c 'if [ "`uname -m`" == "ppc64le" ]; then echo -ppc64le ; fi')
DOCKER_RUN_OPTS += -e ARCH_SUFFIX=$(ARCH_SUFFIX)

LIB_NAME_SUFFIX ?= $(SUFFIX)
DOCKER_RUN_OPTS += -e LIB_NAME_SUFFIX=$(LIB_NAME_SUFFIX)

########################################################################################################################
# Output directories

export BUILD_DIRECTORY_CPPLIB ?= $(NNABLA_DIRECTORY)/build$(BUILD_DIRECTORY_CPPLIB_SUFFIX)
DOCKER_RUN_OPTS += -e BUILD_DIRECTORY_CPPLIB=$(BUILD_DIRECTORY_CPPLIB)

export BUILD_DIRECTORY_CPPLIB_ANDROID ?= $(NNABLA_DIRECTORY)/build-android$(BUILD_DIRECTORY_CPPLIB_ANDROID_SUFFIX)
DOCKER_RUN_OPTS += -e BUILD_DIRECTORY_CPPLIB_ANDROID=$(BUILD_DIRECTORY_CPPLIB_ANDROID)

export BUILD_DIRECTORY_WHEEL ?= $(NNABLA_DIRECTORY)/build_wheel$(BUILD_DIRECTORY_WHEEL_SUFFIX)
DOCKER_RUN_OPTS += -e BUILD_DIRECTORY_WHEEL=$(BUILD_DIRECTORY_WHEEL)

export DOC_DIRECTORY ?= $(NNABLA_DIRECTORY)/build-doc
DOCKER_RUN_OPTS += -e DOC_DIRECTORY=$(DOC_DIRECTORY)

###############################################################################
# Test options
DOCKER_RUN_OPTS += -e PYTEST_PATH_EXTRA=$(PYTEST_PATH_EXTRA)
DOCKER_RUN_OPTS += -e PYTEST_LD_LIBRARY_PATH_EXTRA=$(PYTEST_LD_LIBRARY_PATH_EXTRA)

export DOCKER_RUN_OPTS
export ARCH_SUFFIX

########################################################################################################################
# Functions for makefile
define with-venv
	rm -rf $(2)
	python$(PYTHON_VERSION_MAJOR).$(PYTHON_VERSION_MINOR) -m venv --system-site-packages $(2)
	. $(2)/bin/activate \
	&& python -m pip install -I pip \
	&& $(MAKE) -C $(1) $(3) $(4) \
	&& deactivate
	rm -rf $(2)
endef


