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

########################################################################################################################
# Suppress most of make message.
.SILENT:

NNABLA_BUILD_WITH_DOCKER_INCLUDED = True

########################################################################################################################
# General settings

NNABLA_DIRECTORY ?= $(shell pwd)
include $(NNABLA_DIRECTORY)/build-tools/make/options.mk

DOCKER_IMAGE_NAME_BASE ?= nnabla-build

DOCKER_IMAGE_AUTO_FORMAT ?= $(DOCKER_IMAGE_NAME_BASE)-auto-format
DOCKER_IMAGE_DOC ?= $(DOCKER_IMAGE_NAME_BASE)-doc
DOCKER_IMAGE_BUILD ?= $(DOCKER_IMAGE_NAME_BASE)-build
DOCKER_IMAGE_BUILD_ANDROID ?= $(DOCKER_IMAGE_NAME_BASE)-build-android
DOCKER_IMAGE_NNABLA ?= $(DOCKER_IMAGE_NAME_BASE)-nnabla
DOCKER_IMAGE_ONNX_TEST ?= $(DOCKER_IMAGE_NAME_BASE)-onnx-test

DOCKER_RUN_OPTS +=--rm
DOCKER_RUN_OPTS += -v $$(pwd):$$(pwd)
DOCKER_RUN_OPTS += -w $$(pwd)
DOCKER_RUN_OPTS += -u $$(id -u):$$(id -g)
DOCKER_RUN_OPTS += -e HOME=/tmp
DOCKER_RUN_OPTS += -e CMAKE_OPTS=$(CMAKE_OPTS)

########################################################################################################################
# Docker images
.PHONY: docker_image_auto_format
docker_image_auto_format:
	docker pull ubuntu:16.04
	cd $(NNABLA_DIRECTORY) \
	&& docker build $(DOCKER_BUILD_ARGS) -t $(DOCKER_IMAGE_AUTO_FORMAT) -f docker/development/Dockerfile.auto-format .

.PHONY: docker_image_doc
docker_image_doc:
	docker pull ubuntu:16.04
	cd $(NNABLA_DIRECTORY) \
	&& docker build $(DOCKER_BUILD_ARGS) -t $(DOCKER_IMAGE_DOC) -f docker/development/Dockerfile.document .

.PHONY: docker_image_build
docker_image_build:
	docker pull centos:6
	cd $(NNABLA_DIRECTORY) \
	&& docker build $(DOCKER_BUILD_ARGS) -t $(DOCKER_IMAGE_BUILD) \
		-f docker/development/Dockerfile.build .

.PHONY: docker_image_onnx_test
docker_image_onnx_test:
	docker pull ubuntu:16.04
	cd $(NNABLA_DIRECTORY) \
	&& docker build $(DOCKER_BUILD_ARGS) -t $(DOCKER_IMAGE_ONNX_TEST) \
		-f docker/development/Dockerfile.onnx-test .

.PHONY: docker_image_build_android
docker_image_build_android:
	docker pull ubuntu:16.04
	cd $(NNABLA_DIRECTORY) \
	&& docker build $(DOCKER_BUILD_ARGS) -t $(DOCKER_IMAGE_BUILD_ANDROID) \
		-f docker/development/Dockerfile.android .

########################################################################################################################
# Auto Format

.PHONY: bwd-nnabla-auto-format
bwd-nnabla-auto-format: docker_image_auto_format
	cd $(NNABLA_DIRECTORY) \
	&& docker run $(DOCKER_RUN_OPTS) $(DOCKER_IMAGE_AUTO_FORMAT) make -f build-tools/make/build.mk nnabla-auto-format

########################################################################################################################
# Doc
.PHONY: bwd-nnabla-doc
bwd-nnabla-doc: docker_image_doc
	cd $(NNABLA_DIRECTORY) \
	&& docker run $(DOCKER_RUN_OPTS) $(DOCKER_IMAGE_DOC) make -f build-tools/make/build.mk nnabla-doc

########################################################################################################################
# Build and test
.PHONY: bwd-nnabla-cpplib
bwd-nnabla-cpplib: docker_image_build
	cd $(NNABLA_DIRECTORY) \
	&& docker run $(DOCKER_RUN_OPTS) $(DOCKER_IMAGE_BUILD) make -f build-tools/make/build.mk nnabla-cpplib

.PHONY: bwd-nnabla-cpplib-android
bwd-nnabla-cpplib-android: docker_image_build_android
	cd $(NNABLA_DIRECTORY) \
	&& docker run $(DOCKER_RUN_OPTS) $(DOCKER_IMAGE_BUILD_ANDROID) ./build-tools/android/build_nnabla.sh -p=$(PLATFORM) -a=$(ARCHITECTURE) -n=/usr/local/src/android-ndk -e=$(ABI)

.PHONY: bwd-nnabla-test-cpplib
bwd-nnabla-test-cpplib: docker_image_build
	cd $(NNABLA_DIRECTORY) \
	&& docker run $(DOCKER_RUN_OPTS) $(DOCKER_IMAGE_BUILD) make -f build-tools/make/build.mk nnabla-test-cpplib

.PHONY: bwd-nnabla-wheel
bwd-nnabla-wheel: docker_image_build
	cd $(NNABLA_DIRECTORY) \
	&& docker run $(DOCKER_RUN_OPTS) $(DOCKER_IMAGE_BUILD) make -f build-tools/make/build.mk MAKE_MANYLINUX_WHEEL=ON nnabla-wheel

.PHONY: bwd-nnabla-test
bwd-nnabla-test: docker_image_onnx_test
	cd $(NNABLA_DIRECTORY) \
	&& docker run $(DOCKER_RUN_OPTS) $(DOCKER_IMAGE_ONNX_TEST) make -f build-tools/make/build.mk nnabla-test-local

.PHONY: bwd-nnabla-shell
bwd-nnabla-shell: docker_image_build
	cd $(NNABLA_DIRECTORY) \
	&& docker run $(DOCKER_RUN_OPTS) -it --rm ${DOCKER_IMAGE_BUILD} make nnabla-shell

########################################################################################################################
# Docker image with current nnabla
.PHONY: docker_image_nnabla
docker_image_nnabla:
	docker pull ubuntu:16.04
	cd $(NNABLA_DIRECTORY) \
	&& cp docker/development/Dockerfile.build Dockerfile \
	&& echo ADD $(shell echo build_wheel_py$(PYTHON_VERSION_MAJOR)$(PYTHON_VERSION_MINOR)/dist/*.whl) /tmp/ >>Dockerfile \
	&& echo RUN pip install /tmp/$(shell basename build_wheel_py$(PYTHON_VERSION_MAJOR)$(PYTHON_VERSION_MINOR)/dist/*.whl) >>Dockerfile \
	&& docker build $(DOCKER_BUILD_ARGS) -t $(DOCKER_IMAGE_NNABLA) . \
	&& rm -f Dockerfile
