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

.PHONY: all
all: py27 py34 py35 py36

.PHONY: clean
clean:
	@rm -f Dockerfile.build.py*

.PHONY: py27
py27:
	@cat Dockerfile.build.in |sed \
		-e 's/{PYENVNAME}/py27/g' \
		-e 's/{PYENVVERSION}/2.7/g' \
		>Dockerfile.build.py27
	@cat Dockerfile.onnx-test.in |sed \
		-e 's/{PYENVNAME}/py27/g' \
		-e 's/{PYENVVERSION}/2.7/g' \
		>Dockerfile.onnx-test.py27

.PHONY: py35
py35:
	@cat Dockerfile.build.in |sed \
		-e 's/{PYENVNAME}/py35/g' \
		-e 's/{PYENVVERSION}/3.5/g' \
		>Dockerfile.build.py35
	@cat Dockerfile.onnx-test.in |sed \
		-e 's/{PYENVNAME}/py35/g' \
		-e 's/{PYENVVERSION}/3.5/g' \
		>Dockerfile.onnx-test.py35

.PHONY: py36
py36:
	@cat Dockerfile.build.in |sed \
		-e 's/{PYENVNAME}/py36/g' \
		-e 's/{PYENVVERSION}/3.6/g' \
		>Dockerfile.build.py36
	@cat Dockerfile.onnx-test.in |sed \
		-e 's/{PYENVNAME}/py36/g' \
		-e 's/{PYENVVERSION}/3.6/g' \
		>Dockerfile.onnx-test.py36
