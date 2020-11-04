# Copyright (c) 2019 Sony Corporation. All Rights Reserved.
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

all: train_lenet_classifier.cpp train_resnet_classifier.cpp train_vae.cpp train_dcgan.cpp train_siamese.cpp train_vat.cpp
	$(CXX) -std=c++11 -O -o train_lenet_classifier train_lenet_classifier.cpp -lnnabla -lz -lnnabla_utils
	$(CXX) -std=c++11 -O -o train_resnet_classifier train_resnet_classifier.cpp -lnnabla -lz -lnnabla_utils
	$(CXX) -std=c++11 -O -o train_vae train_vae.cpp -lnnabla -lz -lnnabla_utils
	$(CXX) -std=c++11 -O -o train_dcgan train_dcgan.cpp -lnnabla -lz -lnnabla_utils
	$(CXX) -std=c++11 -O -o train_siamese train_siamese.cpp -lnnabla -lz -lnnabla_utils
	$(CXX) -std=c++11 -O -o train_vat train_vat.cpp -lnnabla -lz -lnnabla_utils

lenet: train_lenet_classifier.cpp
	$(CXX) -std=c++11 -O -o train_lenet_classifier train_lenet_classifier.cpp -lnnabla -lz -lnnabla_utils

resnet: train_resnet_classifier.cpp
	$(CXX) -std=c++11 -O -o train_resnet_classifier train_resnet_classifier.cpp -lnnabla -lz -lnnabla_utils

vae: train_vae.cpp
	$(CXX) -std=c++11 -O -o train_vae train_vae.cpp -lnnabla -lz -lnnabla_utils

dcgan: train_dcgan.cpp
	$(CXX) -std=c++11 -O -o train_dcgan train_dcgan.cpp -lnnabla -lz -lnnabla_utils

siamese: train_siamese.cpp
	$(CXX) -std=c++11 -O -o train_siamese train_siamese.cpp -lnnabla -lz -lnnabla_utils

vat: train_vat.cpp
	$(CXX) -std=c++11 -O -o train_vat train_vat.cpp -lnnabla -lz -lnnabla_utils

clean:
	rm -f train_lenet_classifier
	rm -f train_resnet_classifier
	rm -f train_vae
	rm -f train_dcgan
	rm -f train_siamese
	rm -f train_vat
