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
#.SILENT:

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
	cd $(NNABLA_DIRECTORY) && \
	python $(NNABLA_DIRECTORY)/build-tools/auto_format . --exclude \
		'\./src/nbla/(function|solver)/\w+\.cpp' \
		'\./src/nbla/init.cpp' \
		'\./python/src/nnabla/\w+\.(cpp|hpp|h|c)' \
		'\./python/src/nnabla/(solver.pyx|function.pyx|function.pxd|function_bases.py)' \
		'\./python/src/nnabla/utils/(save|load)_function.py' \
		'\./src/nbla_utils/nnp_impl_create_function.cpp' \
		'\./src/nbla_utils/nnabla\.pb\.(h|cc)'


########################################################################################################################
# Doc
.PHONY: nnabla-doc
nnabla-doc:
	mkdir -p $(DOC_DIRECTORY)
	cd $(DOC_DIRECTORY) \
	&& cmake -DBUILD_CPP_LIB=ON \
		-DBUILD_CPP_UTILS=OFF \
		-DBUILD_PYTHON_PACKAGE=ON \
		$(CMAKE_OPTS) \
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
		-DNNABLA_UTILS_WITH_NPY=ON \
		-DBUILD_PYTHON_PACKAGE=OFF \
		$(CMAKE_OPTS) \
		$(NNABLA_DIRECTORY)
	@$(MAKE) -C $(BUILD_DIRECTORY_CPPLIB) -j$(PARALLEL_BUILD_NUM)

.PHONY: nnabla-cpplib-rpm
nnabla-cpplib-rpm: nnabla-cpplib
	@cd $(BUILD_DIRECTORY_CPPLIB) && cpack -G RPM CPackConfig.cmake
	@cd $(BUILD_DIRECTORY_CPPLIB) && cpack -G TBZ2 CPackConfig.cmake

.PHONY: nnabla-cpplib-deb
nnabla-cpplib-deb: nnabla-cpplib
	@cd $(BUILD_DIRECTORY_CPPLIB) && cpack -G DEB CPackConfig.cmake
	@cd $(BUILD_DIRECTORY_CPPLIB) && cpack -G TBZ2 CPackConfig.cmake

.PHONY: nnabla-cpplib-android
nnabla-cpplib-android:
	@mkdir -p $(BUILD_DIRECTORY_CPPLIB_ANDROID)
	@cd $(BUILD_DIRECTORY_CPPLIB_ANDROID) \
	&& cmake \
	    -DCMAKE_TOOLCHAIN_FILE=$(NDK_PATH)/build/cmake/android.toolchain.cmake \
	    -DANDROID_TOOLCHAIN=clang \
	    -DCMAKE_SYSTEM_NAME=$(CMAKE_SYSTEM_NAME) \
	    -DBUILD_CPP_UTILS=ON \
	    -DBUILD_PYTHON_PACKAGE=OFF \
	    -DNNABLA_UTILS_WITH_HDF5=OFF \
	    -DANDROID_STL=c++_static \
	    -DANDROID_ABI=$(EABI) \
	    -DPYTHON_COMMAND_NAME=$(SYSTEM_PYTHON) \
	    -DPROTOC_COMMAND=$(SYSTEM_PROTOC) \
            -LA \
	    $(NNABLA_DIRECTORY)
	@$(TOOLCHAIN_INSTALL_DIR)/bin/make -C $(BUILD_DIRECTORY_CPPLIB_ANDROID) -j$(PARALLEL_BUILD_NUM)
	@rm -rf $(BUILD_DIRECTORY_CPPLIB_ANDROID)/build_$(PLATFORM)_$(ARCHITECTURE)
	@mkdir -p $(BUILD_DIRECTORY_CPPLIB_ANDROID)/build_$(PLATFORM)_$(ARCHITECTURE)/$(EABI)
	@cp $(TOOLCHAIN_INSTALL_DIR)/lib/libarchive.so $(BUILD_DIRECTORY_CPPLIB_ANDROID)/build_$(PLATFORM)_$(ARCHITECTURE)/$(EABI)
	@cp $(BUILD_DIRECTORY_CPPLIB_ANDROID)/lib/libnnabla.so $(BUILD_DIRECTORY_CPPLIB_ANDROID)/build_$(PLATFORM)_$(ARCHITECTURE)/$(EABI)
	@cp $(BUILD_DIRECTORY_CPPLIB_ANDROID)/lib/libnnabla_utils.so $(BUILD_DIRECTORY_CPPLIB_ANDROID)/build_$(PLATFORM)_$(ARCHITECTURE)/$(EABI)


.PHONY: nnabla-cpplib-android-test
nnabla-cpplib-android-test:
# Execute the binary on emulator


.PHONY: nnabla-wheel
nnabla-wheel:
	@mkdir -p $(BUILD_DIRECTORY_WHEEL)
	@cd $(BUILD_DIRECTORY_WHEEL) \
	&& cmake \
		-DPYTHON_VERSION_MAJOR=$(PYTHON_VERSION_MAJOR) \
		-DPYTHON_VERSION_MINOR=$(PYTHON_VERSION_MINOR) \
		-DWHEEL_SUFFIX=$(WHEEL_SUFFIX) \
		-DBUILD_CPP_LIB=OFF \
		-DBUILD_PYTHON_PACKAGE=ON \
		-DMAKE_MANYLINUX_WHEEL=$(MAKE_MANYLINUX_WHEEL) \
		-DCPPLIB_BUILD_DIR=$(BUILD_DIRECTORY_CPPLIB) \
		-DCPPLIB_LIBRARY=$(BUILD_DIRECTORY_CPPLIB)/lib/libnnabla.so \
		$(CMAKE_OPTS) \
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
	&& PATH=$(PYTEST_PATH_EXTRA):$(PATH) \
	LD_LIBRARY_PATH=$(PYTEST_LD_LIBRARY_PATH_EXTRA):$(LD_LIBRARY_PATH) \
	$(NNABLA_DIRECTORY)/build-tools/make/pytest.sh $(NNABLA_DIRECTORY)/python/test
