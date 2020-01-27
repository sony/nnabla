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

add_executable(train_lenet_classifier train_lenet_classifier.cpp)
add_executable(train_resnet_classifier train_resnet_classifier.cpp)
add_executable(train_vae train_vae.cpp)
add_executable(train_dcgan train_dcgan.cpp)
add_executable(train_siamese train_siamese.cpp)
add_executable(train_vat train_vat.cpp)
find_package(ZLIB REQUIRED)
include_directories(${ZLIB_INCLUDE_DIRS})
target_link_libraries(train_lenet_classifier ${NBLA_LIBRARY_NAME} ${NBLA_UTILS_LIBRARY_NAME} ${ZLIB_LIBRARIES})
set_property(TARGET train_lenet_classifier PROPERTY CXX_STANDARD 11)
target_link_libraries(train_resnet_classifier ${NBLA_LIBRARY_NAME} ${NBLA_UTILS_LIBRARY_NAME} ${ZLIB_LIBRARIES})
set_property(TARGET train_resnet_classifier PROPERTY CXX_STANDARD 11)
target_link_libraries(train_vae ${NBLA_LIBRARY_NAME} ${NBLA_UTILS_LIBRARY_NAME} ${ZLIB_LIBRARIES})
set_property(TARGET train_vae PROPERTY CXX_STANDARD 11)
target_link_libraries(train_dcgan ${NBLA_LIBRARY_NAME} ${NBLA_UTILS_LIBRARY_NAME} ${ZLIB_LIBRARIES})
set_property(TARGET train_dcgan PROPERTY CXX_STANDARD 11)
target_link_libraries(train_siamese ${NBLA_LIBRARY_NAME} ${NBLA_UTILS_LIBRARY_NAME} ${ZLIB_LIBRARIES})
set_property(TARGET train_siamese PROPERTY CXX_STANDARD 11)
target_link_libraries(train_vat ${NBLA_LIBRARY_NAME} ${NBLA_UTILS_LIBRARY_NAME} ${ZLIB_LIBRARIES})
set_property(TARGET train_vat PROPERTY CXX_STANDARD 11)
