#!/bin/bash

set -xe

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
# 


# Following ABI(Application Binary Interface) can be chosen for selected architecture.
# armeabi
# armeabi-v7a
# arm64-v8a
# x86
# x86_64

# Compiler for different ARCHITECTURE and ABI:
# CMAKE_SYSTEM_NAME = arm-linux-androideabi  ==> ARCHITECTURE = arm 	 , ABI=armeabi or ABI=armeabi-v7a
# CMAKE_SYSTEM_NAME = aarch64-linux-android  ==> ARCHITECTURE = arm64 	 , ABI=arm64-v8a
# CMAKE_SYSTEM_NAME = i686-linux-android     ==> ARCHITECTURE = x86 	 , ABI=x86
# CMAKE_SYSTEM_NAME = x86_64-linux-android   ==> ARCHITECTURE = x86_64 	 , ABI=x86_64

if [ $# -lt 3 ]
  then
    echo "Usage: sh build_nnabla.sh -p=android-XX -a=[arm|arm64|x86|x86_64] -n=Path_to_your_ndk_folder -e=[armeabi|armeabi-v7a|arm64-v8a|x86|x86_64] (-e option is optional)"
    echo "Example: sh build_nnabla.sh -p=android-26 -a=arm64 -n=/usr/local/src/android-ndk-r16b -e=arm64-v8a"
    exit 1
fi

for i in "$@"
do
case $i in
    -p=*|--platform=*)
    PLATFORM="${i#*=}"
    shift # past argument=value
    ;;
    -a=*|--architecture=*)
    ARCHITECTURE="${i#*=}"
    shift # past argument=value
    ;;
    -e=*|--eabi=*)
    EABI="${i#*=}"
    shift # past argument=value
    ;;
    -n=*|--ndk=*)
    NDK_PATH="${i#*=}"
    shift # past argument=value
    ;;
    --default)
    DEFAULT=YES
    shift # past argument with no value
    ;;
    *)
          # unknown option
    ;;
esac
done

if [ "$ARCHITECTURE" = "x86_64" ]
then
    echo "Architecture chosen for build is x86_64."
    CMAKE_SYSTEM_NAME=x86_64-linux-android
    if [ "$EABI" = "" ]
    then
        echo "Setting default EABI for architecture $ARCHITECTURE."
        EABI=x86_64	
    fi
elif [ "$ARCHITECTURE" = "x86" ]
then
    echo "Architecture chosen for build is x86."
    CMAKE_SYSTEM_NAME=i686-linux-android
    if [ "$EABI" = "" ]
    then
        echo "Setting default EABI for architecture $ARCHITECTURE."
        EABI=x86	
    fi
elif [ "$ARCHITECTURE" = "arm64" ]
then
    echo "Architecture chosen for build is arm64."
    CMAKE_SYSTEM_NAME=aarch64-linux-android
    if [ "$EABI" = "" ]
    then
        echo "Setting default EABI for architecture $ARCHITECTURE."
        EABI=arm64-v8a	
    fi
elif [ "$ARCHITECTURE" = "arm" ]
then
    echo "Architecture chosen for build is arm."
    CMAKE_SYSTEM_NAME=arm-linux-androideabi
    if [ "$EABI" = "" ]
    then
        echo "Setting default EABI for architecture $ARCHITECTURE."
        EABI=armeabi-v7a	
    fi
else
    echo "Not valid architecture"
    echo "If invoking the script from make bwd-nnabla-cpplib-android, then please pass the parameters PLATFORM, ARCHITECTURE, and ABI appropriately to make command."
    echo "E.g make PLATFORM=android-26 ARCHITECTURE=arm64 ABI=arm64-v8a bwd-nnabla-cpplib-android"
    exit 1
fi
WORK_DIR=`pwd`
SYSTEM_PYTHON=`which python`
SYSTEM_PROTOC=`which protoc`
NNABLA_ROOT=$(pwd)
WORK_BUILD_DIR=$NNABLA_ROOT/build/dependencies

echo "PLATFORM  = ${PLATFORM}"
echo "ARCHITECTURE     = ${ARCHITECTURE}"
echo "EABI    = ${EABI}"
echo "NDK_PATH    = ${NDK_PATH}"
echo "WORK_DIR    = ${WORK_DIR}"
echo "Setting GCC compiler to $CMAKE_SYSTEM_NAME"

TOOLCHAIN_INSTALL_DIR=$WORK_BUILD_DIR/nnabla_$ARCHITECTURE
GCC=$CMAKE_SYSTEM_NAME-gcc
GCXX=$CMAKE_SYSTEM_NAME-c++
SYSROOT=$NDK_PATH/platforms/$PLATFORM/arch-$ARCHITECTURE

#clean up
rm -rf $NNABLA_ROOT/build
mkdir -p $WORK_BUILD_DIR

#toolchain creation and path setting
sh $NDK_PATH/build/tools/make-standalone-toolchain.sh --platform=$PLATFORM --arch=$ARCHITECTURE --install-dir=$TOOLCHAIN_INSTALL_DIR
export SYSROOT=$SYSROOT
export CC=$GCC
export CXX=$GCXX
export PATH=$TOOLCHAIN_INSTALL_DIR/bin:"$PATH"


###################### Build protobuf ######################
cd $WORK_BUILD_DIR

#download
curl -L https://github.com/google/protobuf/archive/v3.1.0.tar.gz -o protobuf-v3.1.0.tar.gz
tar xvf protobuf-v3.1.0.tar.gz
cd protobuf-3.1.0

#build
mkdir build && cd build
cmake  -DCMAKE_TOOLCHAIN_FILE=$NDK_PATH/build/cmake/android.toolchain.cmake -DANDROID_TOOLCHAIN=clang -DCMAKE_SYSTEM_NAME=$CMAKE_SYSTEM_NAME -DCMAKE_POSITION_INDEPENDENT_CODE=ON -Dprotobuf_BUILD_TESTS=OFF -DCMAKE_INSTALL_PREFIX=$TOOLCHAIN_INSTALL_DIR -DANDROID_STL=c++_static -DANDROID_ABI=$EABI ../cmake
$TOOLCHAIN_INSTALL_DIR/bin/make
$TOOLCHAIN_INSTALL_DIR/bin/make install

###################### Build libarchive ######################
cd $WORK_BUILD_DIR

#download
curl -L https://www.libarchive.org/downloads/libarchive-3.3.2.tar.gz -o libarchive-3.3.2.tar.gz
tar xvf libarchive-3.3.2.tar.gz
cd libarchive-3.3.2

#preprocess 
sed -i "/INCLUDE(CheckTypeSize)/aINCLUDE_DIRECTORIES($WORK_BUILD_DIR/libarchive-3.3.2/contrib/android/include/)" CMakeLists.txt

#build
cmake  -DCMAKE_TOOLCHAIN_FILE=$NDK_PATH/build/cmake/android.toolchain.cmake -DANDROID_TOOLCHAIN=clang -DCMAKE_SYSTEM_NAME=$CMAKE_SYSTEM_NAME -DCMAKE_POSITION_INDEPENDENT_CODE=ON -DCMAKE_INSTALL_PREFIX=$TOOLCHAIN_INSTALL_DIR -DENABLE_TEST=OFF -DANDROID_STL=c++_static -DANDROID_ABI=$EABI .

#preprocess 
sed -i "/#define HAVE_STATFS 1/a#define HAVE_STATVFS 1" config.h
sed -i "/#include \"passphrase.h\"/a#ifdef ANDROID\nint wctomb(char *s, wchar_t wc) { return wcrtomb(s,wc,NULL); }\nint mbtowc(wchar_t *pwc, const char *s, size_t n) { return mbrtowc(pwc, s, n, NULL); }\n#endif" tar/util.c

#build
$TOOLCHAIN_INSTALL_DIR/bin/make
$TOOLCHAIN_INSTALL_DIR/bin/make install

###################### Build nnabla ######################
cd $NNABLA_ROOT

#preprocess 
cp src/nbla_utils/CMakeLists.txt $WORK_BUILD_DIR/CMakeLists_bak.txt
sed -i "s/^  find_package(Protobuf REQUIRED)/#&/1" src/nbla_utils/CMakeLists.txt
sed -i "/#  find_package(Protobuf REQUIRED)/i include_directories($TOOLCHAIN_INSTALL_DIR/include)\ninclude_directories($WORK_BUILD_DIR/libarchive-3.3.2/contrib/android/include/)" src/nbla_utils/CMakeLists.txt
sed -i "/#  find_package(Protobuf REQUIRED)/a\  add_library(Protobuf STATIC IMPORTED)\n \
 set_target_properties(Protobuf \n \
                      PROPERTIES IMPORTED_LOCATION \n \
                      $TOOLCHAIN_INSTALL_DIR/lib/libprotobuf.a) \n \
 set_target_properties(Protobuf \n \
                      PROPERTIES INCLUDE_DIRECTORIES \n \
                      $WORK_BUILD_DIR/libarchive-3.3.2/contrib/android/include/)" src/nbla_utils/CMakeLists.txt

sed -i "s/^  find_package(LibArchive REQUIRED)/#&/1" src/nbla_utils/CMakeLists.txt
sed -i "/#  find_package(LibArchive REQUIRED)/a\  add_library(LibArchive STATIC IMPORTED)\n \
 set_target_properties(LibArchive \n \
                      PROPERTIES IMPORTED_LOCATION \n \
                      $TOOLCHAIN_INSTALL_DIR/lib/libarchive.so) \n \
 set_target_properties(LibArchive \n \
                      PROPERTIES INCLUDE_DIRECTORIES \n \
                      $WORK_BUILD_DIR/libarchive-3.3.2/contrib/android/include/)" src/nbla_utils/CMakeLists.txt

sed -i "s/^    \${PROTOBUF_LIBRARY}/#&/1" src/nbla_utils/CMakeLists.txt
sed -i "s/^    \${LibArchive_LIBRARIES}/#&/1" src/nbla_utils/CMakeLists.txt

sed -i "/#    \${PROTOBUF_LIBRARY}/i\    Protobuf" src/nbla_utils/CMakeLists.txt
sed -i "/#    \${LibArchive_LIBRARIES}/i\    LibArchive" src/nbla_utils/CMakeLists.txt

cp src/nbla/logger.cpp $WORK_BUILD_DIR/logger_bak.cpp
sed -i "/get_logger/a\/*" src/nbla/logger.cpp
sed -i "/return l;/a\*\/return 0;" src/nbla/logger.cpp

cp include/nbla/logger.hpp $WORK_BUILD_DIR/logger_bak.hpp
sed -i "/(...)/a\/*" include/nbla/logger.hpp
sed -i "/}/a\*\/{}" include/nbla/logger.hpp

#build
cd build
cmake .. -DCMAKE_TOOLCHAIN_FILE=$NDK_PATH/build/cmake/android.toolchain.cmake -DANDROID_TOOLCHAIN=clang -DCMAKE_SYSTEM_NAME=$CMAKE_SYSTEM_NAME -DBUILD_CPP_UTILS=ON -DBUILD_PYTHON_PACKAGE=OFF -DNNABLA_UTILS_WITH_HDF5=OFF -DANDROID_STL=c++_static -DANDROID_ABI=$EABI -DPYTHON_COMMAND_NAME=$SYSTEM_PYTHON -DPROTOC_COMMAND=$SYSTEM_PROTOC -LA
$TOOLCHAIN_INSTALL_DIR/bin/make

#revert intermediate changes
rm $NNABLA_ROOT/src/nbla_utils/CMakeLists.txt
mv $WORK_BUILD_DIR/CMakeLists_bak.txt $NNABLA_ROOT/src/nbla_utils/CMakeLists.txt

rm $NNABLA_ROOT/include/nbla/logger.hpp
mv $WORK_BUILD_DIR/logger_bak.hpp $NNABLA_ROOT/include/nbla/logger.hpp

rm $NNABLA_ROOT/src/nbla/logger.cpp
mv $WORK_BUILD_DIR/logger_bak.cpp $NNABLA_ROOT/src/nbla/logger.cpp

###################### post processing (Collect built libraries in result folder) ######################
cp $TOOLCHAIN_INSTALL_DIR/lib/libarchive.so $NNABLA_ROOT/build/lib/

#Write the contents android configuration details to file for JNI building
cd $NNABLA_ROOT/build
destfile=android_setup.cfg
echo "" > "$destfile"
echo "###############################################################" >> "$destfile"
echo "# This file is auto generated by nnabla android build system. #" >> "$destfile"
echo "###############################################################" >> "$destfile"
echo "                                                               " >> "$destfile"
echo "PLATFORM=${PLATFORM}" >> "$destfile"
echo "ARCHITECTURE=${ARCHITECTURE}" >> "$destfile"
echo "EABI=${EABI}" >> "$destfile"
echo "NDK_PATH=${NDK_PATH}" >> "$destfile"
echo "CMAKE_SYSTEM_NAME=${CMAKE_SYSTEM_NAME}" >> "$destfile"
echo "TOOLCHAIN_INSTALL_DIR=${TOOLCHAIN_INSTALL_DIR}" >> "$destfile"
echo "SYSROOT=${SYSROOT}" >> "$destfile"
echo "GCC=${GCC}" >> "$destfile"
echo "GCXX=${GCXX}" >> "$destfile"
echo "NNABLA_INCLUDE_DIR=${NNABLA_ROOT}/include" >> "$destfile"
echo "NNABLA_LIBS_DIR=$NNABLA_ROOT/build/lib" >> "$destfile"
