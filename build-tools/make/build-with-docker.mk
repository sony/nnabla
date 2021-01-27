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

DOCKER_IMAGE_NAME_BASE ?= nnabla-py$(PYTHON_VERSION_MAJOR)$(PYTHON_VERSION_MINOR)

DOCKER_IMAGE_AUTO_FORMAT ?= $(DOCKER_IMAGE_NAME_BASE)-auto-format$(ARCH_SUFFIX):$(shell md5sum $(NNABLA_DIRECTORY)/docker/development/Dockerfile.auto-format |cut -d \  -f 1)
DOCKER_IMAGE_DOC ?= $(DOCKER_IMAGE_NAME_BASE)-doc$(ARCH_SUFFIX):$(shell md5sum $(NNABLA_DIRECTORY)/docker/development/Dockerfile.document |cut -d \  -f 1)
DOCKER_IMAGE_BUILD ?= $(DOCKER_IMAGE_NAME_BASE)-build$(ARCH_SUFFIX):$(shell md5sum $(NNABLA_DIRECTORY)/docker/development/Dockerfile.build$(ARCH_SUFFIX) |cut -d \  -f 1)
DOCKER_IMAGE_NNABLA ?= $(DOCKER_IMAGE_NAME_BASE)-nnabla$(ARCH_SUFFIX):$(shell md5sum $(NNABLA_DIRECTORY)/docker/development/Dockerfile.build |cut -d \  -f 1)
DOCKER_IMAGE_NNABLA_TEST ?= $(DOCKER_IMAGE_NAME_BASE)-nnabla-test$(ARCH_SUFFIX):$(shell md5sum $(NNABLA_DIRECTORY)/docker/development/Dockerfile.nnabla-test$(ARCH_SUFFIX) |cut -d \  -f 1)

########################################################################################################################
# Docker images

.PHONY: docker_image_auto_format
docker_image_auto_format:
	if ! docker image inspect $(DOCKER_IMAGE_AUTO_FORMAT) >/dev/null 2>/dev/null; then \
		docker pull ubuntu:16.04 && \
		(cd $(NNABLA_DIRECTORY) && docker build $(DOCKER_BUILD_ARGS) -t $(DOCKER_IMAGE_AUTO_FORMAT) -f docker/development/Dockerfile.auto-format .) \
	fi

.PHONY: docker_image_doc
docker_image_doc:
	if ! docker image inspect $(DOCKER_IMAGE_DOC) >/dev/null 2>/dev/null; then \
		docker pull ubuntu:16.04 && \
		( cd $(NNABLA_DIRECTORY) && docker build $(DOCKER_BUILD_ARGS) -t $(DOCKER_IMAGE_DOC) -f docker/development/Dockerfile.document . ) \
	fi

.PHONY: docker_image_build
docker_image_build:
	if ! docker image inspect $(DOCKER_IMAGE_BUILD) >/dev/null 2>/dev/null; then \
		docker pull $(shell cat $(NNABLA_DIRECTORY)/docker/development/Dockerfile.build$(ARCH_SUFFIX) |grep ^FROM |awk '{print $$2}') && \
		(cd $(NNABLA_DIRECTORY) && docker build $(DOCKER_BUILD_ARGS) -t $(DOCKER_IMAGE_BUILD) -f docker/development/Dockerfile.build$(ARCH_SUFFIX) .) \
	fi

.PHONY: docker_image_nnabla_test
docker_image_nnabla_test:
	if ! docker image inspect $(DOCKER_IMAGE_NNABLA_TEST) >/dev/null 2>/dev/null; then \
		docker pull $(shell cat $(NNABLA_DIRECTORY)/docker/development/Dockerfile.nnabla-test$(ARCH_SUFFIX) |grep ^FROM |awk '{print $$2}') && \
		(cd $(NNABLA_DIRECTORY) && docker build $(DOCKER_BUILD_ARGS) -t $(DOCKER_IMAGE_NNABLA_TEST) -f docker/development/Dockerfile.nnabla-test$(ARCH_SUFFIX) .) \
	fi

# for Android
# Compiler for diffrent ARCHITECTURE and ABI:
# CMAKE_SYSTEM_NAME = arm-linux-androideabi  ==> ARCHITECTURE = arm 	 , ABI=armeabi or ABI=armeabi-v7a
# CMAKE_SYSTEM_NAME = aarch64-linux-android  ==> ARCHITECTURE = arm64 	 , ABI=arm64-v8a
# CMAKE_SYSTEM_NAME = i686-linux-android     ==> ARCHITECTURE = x86 	 , ABI=x86
# CMAKE_SYSTEM_NAME = x86_64-linux-android   ==> ARCHITECTURE = x86_64 	 , ABI=x86_64

ANDROID_PLATFORM ?= android-26
ANDROID_ARCHITECTURE ?= arm64
ANDROID_CMAKE_SYSTEM_NAME ?= aarch64-linux-android
ANDROID_EABI ?= arm64-v8a

DOCKER_IMAGE_BUILD_ANDROID ?= $(DOCKER_IMAGE_NAME_BASE)-build-android-$(ANDROID_PLATFORM)-$(ANDROID_ARCHITECTURE)-$(ANDROID_CMAKE_SYSTEM_NAME)-$(ANDROID_EABI):$(shell md5sum $(NNABLA_DIRECTORY)/docker/development/Dockerfile.android |cut -d \  -f 1)

.PHONY: docker_image_build_android
docker_image_build_android:
	if ! docker image inspect $(DOCKER_IMAGE_BUILD_ANDROID) >/dev/null 2>/dev/null; then \
		docker pull ubuntu:16.04 && \
		(cd $(NNABLA_DIRECTORY) && docker build $(DOCKER_BUILD_ARGS) -t $(DOCKER_IMAGE_BUILD_ANDROID) \
			--build-arg ANDROID_PLATFORM=$(ANDROID_PLATFORM) \
			--build-arg ANDROID_ARCHITECTURE=$(ANDROID_ARCHITECTURE) \
			--build-arg ANDROID_CMAKE_SYSTEM_NAME=$(ANDROID_CMAKE_SYSTEM_NAME) \
			--build-arg ANDROID_EABI=$(ANDROID_EABI) \
			-f docker/development/Dockerfile.android .) \
	fi

.PHONY: docker_image_build_android_emulator
docker_image_build_android_emulator:
	docker pull gradle:4.10.0-jdk8
	cd $(NNABLA_DIRECTORY) \
	&& docker build $(DOCKER_BUILD_ARGS) -t $(DOCKER_IMAGE_BUILD_ANDROID)_test \
		-f docker/development/Dockerfile.android-test .

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
	&& docker run $(DOCKER_RUN_OPTS) $(DOCKER_IMAGE_BUILD_ANDROID) make -f build-tools/make/build.mk nnabla-cpplib-android

.PHONY: bwd-nnabla-cpplib-android-test
bwd-nnabla-cpplib-android-test: docker_image_build_android_emulator
	cd $(NNABLA_DIRECTORY) \
	&& docker run $(DOCKER_RUN_OPTS) $(DOCKER_IMAGE_BUILD_ANDROID)_test make -f build-tools/make/build.mk nnabla-cpplib-android-test

.PHONY: bwd-nnabla-test-cpplib
bwd-nnabla-test-cpplib: docker_image_build
	cd $(NNABLA_DIRECTORY) \
	&& docker run $(DOCKER_RUN_OPTS) $(DOCKER_IMAGE_BUILD) make -f build-tools/make/build.mk nnabla-test-cpplib

.PHONY: bwd-nnabla-wheel
bwd-nnabla-wheel: docker_image_build
	cd $(NNABLA_DIRECTORY) \
	&& docker run $(DOCKER_RUN_OPTS) $(DOCKER_IMAGE_BUILD) make -f build-tools/make/build.mk MAKE_MANYLINUX_WHEEL=$(MAKE_MANYLINUX_WHEEL) nnabla-wheel

.PHONY: bwd-nnabla-test
bwd-nnabla-test: docker_image_nnabla_test
	cd $(NNABLA_DIRECTORY) \
	&& docker run $(DOCKER_RUN_OPTS) $(DOCKER_IMAGE_NNABLA_TEST) make -f build-tools/make/build.mk nnabla-test

.PHONY: bwd-nnabla-shell
bwd-nnabla-shell: docker_image_build
	cd $(NNABLA_DIRECTORY) \
	&& docker run $(DOCKER_RUN_OPTS) -it --rm ${DOCKER_IMAGE_BUILD} make nnabla-shell

########################################################################################################################
# Docker image with current nnabla
.PHONY: docker_image_nnabla
docker_image_nnabla:
	docker pull ubuntu:18.04
	cd $(NNABLA_DIRECTORY) \
	&& docker build $(DOCKER_BUILD_ARGS) \
		--build-arg WHL_PATH=$$(echo build_wheel_py$(PYTHON_VERSION_MAJOR)$(PYTHON_VERSION_MINOR)/dist) \
		-f docker/runtime/Dockerfile \
		-t $(DOCKER_IMAGE_NNABLA) .

