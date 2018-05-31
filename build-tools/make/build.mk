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

NNABLA_BUILD_INCLUDED = True

NNABLA_DIRECTORY ?= $(shell pwd)
include $(NNABLA_DIRECTORY)/build-tools/make/options.mk

########################################################################################################################
# Cleaning
.PHONY: nnabla-clean
nnabla-clean:
	@git clean -fdX

.PHONY: nnabla-clean-all
nnabla-clean-all:
	@git clean -fdx

########################################################################################################################
# Auto Format
.PHONY: nnabla-auto-format
nnabla-auto-format:
	python $(NNABLA_DIRECTORY)/build-tools/auto_format .

########################################################################################################################
# Doc
.PHONY: nnabla-doc
nnabla-doc:
	mkdir -p $(DOC_DIRECTORY)
	cd $(DOC_DIRECTORY) \
	&& cmake -DBUILD_CPP_LIB=ON \
		-DBUILD_CPP_UTILS=OFF \
		-DBUILD_PYTHON_PACKAGE=ON \
		$(NNABLA_DIRECTORY)
	make -C $(DOC_DIRECTORY) -j$(PARALLEL_BUILD_NUM) all wheel doc

########################################################################################################################
# Build and test.
.PHONY: nnabla-cpplib
nnabla-cpplib:
	@mkdir -p $(BUILD_DIRECTORY_CPPLIB)
	@cd $(BUILD_DIRECTORY_CPPLIB) \
	&& cmake \
		-DPYTHON_VERSION_MAJOR=$(PYTHON_VERSION_MAJOR) \
		-DPYTHON_VERSION_MINOR=$(PYTHON_VERSION_MINOR) \
		-DBUILD_CPP_LIB=ON \
		-DBUILD_CPP_UTILS=ON \
		-DBUILD_TEST=ON \
		-DNNABLA_UTILS_STATIC_LINK_DEPS=$(NNABLA_UTILS_STATIC_LINK_DEPS) \
		-DNNABLA_UTILS_WITH_HDF5=$(NNABLA_UTILS_WITH_HDF5) \
		-DBUILD_PYTHON_PACKAGE=OFF \
		$(NNABLA_DIRECTORY)
	@$(MAKE) -C $(BUILD_DIRECTORY_CPPLIB) -j$(PARALLEL_BUILD_NUM)

.PHONY: nnabla-wheel
nnabla-wheel:
	echo @mkdir -p $(BUILD_DIRECTORY_WHEEL)
	@mkdir -p $(BUILD_DIRECTORY_WHEEL)
	@cd $(BUILD_DIRECTORY_WHEEL) \
	&& cmake \
		-DPYTHON_VERSION_MAJOR=$(PYTHON_VERSION_MAJOR) \
		-DPYTHON_VERSION_MINOR=$(PYTHON_VERSION_MINOR) \
		-DBUILD_CPP_LIB=OFF \
		-DBUILD_PYTHON_PACKAGE=ON \
		-DMAKE_MANYLINUX_WHEEL=$(MAKE_MANYLINUX_WHEEL) \
		-DCPPLIB_BUILD_DIR=$(BUILD_DIRECTORY_CPPLIB) \
		-DCPPLIB_LIBRARY=$(BUILD_DIRECTORY_CPPLIB)/lib/libnnabla.so \
		$(NNABLA_DIRECTORY)
	@$(MAKE) -C $(BUILD_DIRECTORY_WHEEL) wheel

.PHONY: nnabla-install-cpplib
nnabla-install-cpplib:
	@$(MAKE) -C $(BUILD_DIRECTORY_CPPLIB) install

.PHONY: nnabla-install
nnabla-install:
	-pip uninstall -y nnabla
	pip install $(BUILD_DIRECTORY_WHEEL)/dist/*.whl

########################################################################################################################
# Shell (for rapid development)
.PHONY: nnabla-shell
nnabla-shell:
	PS1="nnabla-build: " bash --norc -i

########################################################################################################################
# test
.PHONY: nnabla-test-cpplib
nnabla-test-cpplib: nnabla-cpplib
	@$(MAKE) -C $(BUILD_DIRECTORY_CPPLIB) cpplibtest
	@bash -c "(cd $(BUILD_DIRECTORY_CPPLIB) && ctest -R cpplibtest --output-on-failure)"

.PHONY: nnabla-test
nnabla-test:
	$(call with-virtualenv, $(NNABLA_DIRECTORY), \
				$(BUILD_DIRECTORY_WHEEL)/env, \
				-f build-tools/make/build.mk, nnabla-test-local)

.PHONY: nnabla-test-local
nnabla-test-local: nnabla-install
	@cd $(BUILD_DIRECTORY_WHEEL) \
	&& python -m pytest $(NNABLA_DIRECTORY)/python/test
