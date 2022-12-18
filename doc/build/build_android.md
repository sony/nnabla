# Build C++ utility libraries using Android NDK  

This document describes how to build NNabla C++ libraries using Android Native Development Kit(NDK).  
The following instruction demonstrates the build procedure on Ubuntu 16.04 using Android NDK.  
<br>
Most of the procedure is same as [Build C++ utility libraries](build_cpp_utils.md).  
There are two options to build NNabla for Android.
* Quick build using Docker
* Manual Build

## Quick build using Docker
Execute the following command at the root directory of NNabla repository.  

```shell
make bwd-nnabla-cpplib-android
```

The above make will build NNabla C++ libraries and its dependent libraries within android docker container.  
After successful build, you can find the libraries under build-android/build\_android-${PLATFORM}\_${ARCHITECTURE}/${ABI} folder of your system.
The docker container uses Android NDK version r25b by default.

You can change the parameters by adding the following arguments:

| Parameter | Default Value | Description |
| --- | --- | --- |
| ANDROID\_NDKNAME | android-ndk-r25b | Android NDK package installed in docker image |
| ANDROID\_PLATFORM | android-33 | Minimum Android API level supported by library |
| ANDROID\_ARCHITECTURE | arm64 | Target architecture |
| ANDROID\_CMAKE\_SYSTEM\_NAME | aarch64-linux-android | OS name for which cmake is to build |
| ANDROID\_EABI | arm64-v8a | CPU architecture and instruction set for Android application binary interface |

If you would like to support Oreo (API level 26), execute as follows:

```shell
make bwd-nnabla-cpplib-android ANDROID_PLATFORM=android-26
```

If you want to build docker image by yourself, then please refer build instructions of docker file at [Android Dockerfile section](https://github.com/sony/nnabla/blob/master/docker/README.md)

## Manual Build
This section explains the manual building of NNabla using android NDK.

### Requirements
Following build dependencies needs to be installed by the user manually.

* G++: `sudo apt-get install build-essential`
* CMake>=3.11: `sudo apt-get install cmake`
* Python: `sudo apt-get install python3.8 python3-pip` (Used by code generator)
  * Python packages: `sudo -H python3 -m pip install setuptools six pyyaml mako`
* curl, make and git: `sudo apt-get install curl make git`
* Android NDK: Download the required NDK version from [https://developer.android.com/ndk/downloads/](https://developer.android.com/ndk/downloads/).
  Extract the downloaded NDK into folder, and set `ANDROID_NDK_HOME` environment variable to specify this extracted directory.
 NNabla android build is tested with android-ndk-r25b, however the build should work with other NDK version as well.

* Protobuf >=3: See next section.
* libarchive: See next section.

#### Installing protobuf3 and libarchive C++ libraries and tools

##### Protobuf: Install from source.

Unlike [Python Package compilation](./build.md) which requires
`protoc` compiler only, the NNabla C++ utility library requires
protobuf C++ library too.  The following snippet running on your
terminal will build and install protobuf-3.19.4 from source.

```shell
curl -L https://github.com/google/protobuf/archive/v3.19.4.tar.gz -o protobuf-v3.19.4.tar.gz
tar xvf protobuf-v3.19.4.tar.gz
cd protobuf-3.19.4
mkdir build build-android
cd build
cmake -DCMAKE_POSITION_INDEPENDENT_CODE=ON -Dprotobuf_BUILD_TESTS=OFF -DCMAKE_CXX_STANDARD=14 ../cmake
make -j
sudo make install  # See a note below if you want your system clean.
cd ..

cd build-anrdoid
cmake \
    -DCMAKE_TOOLCHAIN_FILE=$ANDROID_NDK_HOME/build/cmake/android.toolchain.cmake  \
    -DANDROID_ABI=arm64-v8a \
    -DCMAKE_SYSTEM_NAME=aarch64-linux-android \
    -DANDROID_STL=c++_shared \
    -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
    -Dprotobuf_BUILD_TESTS=OFF \
    -DBUILD_SHARED_LIBS=ON \
    -DCMAKE_INSTALL_PREFIX=/usr/local/android/arm64 \
    ../cmake
$ANDROID_NDK_HOME/prebuilt/linux-x86_64/bin/make -j
sudo $ANDROID_NDK_HOME/prebuilt/linux-x86_64/bin/make install
```

NOTE: Installing protobuf on your system with sudo may harm a protobuf
library previously installed on your system. It is recommended to
install a protobuf to a user folder. Specify a cmake option
`-DCMAKE_INSTALL_PREFIX=<installation path>` to set your preferred
installation path (e.g. `$HOME/nnabla_build_deps`), and run `make
install` without sudo. In the NNabla C++ library compilation described
below, to help CMake system find the protobuf, pass the installation
path to an environment variable `CMAKE_FIND_ROOT_PATH`. NNabla's CMake
script is written to use the value (multiple paths can be passed with
a delimiter `;`, e.g. `CMAKE_FIND_ROOT_PATH="<path A>;<path
B>";${CMAKE_FIND_ROOT_PATH}`) as search paths to find some packages
(e.g. protobuf, hdf5, libarchive) on your system. Please don't forget
to set environment variables such as a executable path and library
paths (e.g. `PATH`, `LD_LIBRARY_PATH` etc) if you have runtime
dependencies at your custom installation path.

##### Protobuf: Install from PPA package.

Here is the procedure using an informal PPA package. If you can not
trust unofficial packages, please use the procedure to build from the
source shown above.

```shell
sudo add-apt-repository ppa:maarten-fonville/protobuf
sudo apt install protobuf-compiler libprotoc-dev libprotobuf-dev
```

##### libarchive: Install from source.

NNabla C++ utility library requires libarchive C++ library.
The following snippet running on your terminal will build
and install libarchive-3.3.2 from source.

```shell
curl -LO https://www.libarchive.org/downloads/libarchive-3.3.2.tar.gz
tar xzf libarchive-3.3.2.tar.gz
cd libarchive-3.3.2
cmake  \
    -DCMAKE_TOOLCHAIN_FILE=$ANDROID_NDK_HOME/build/cmake/android.toolchain.cmake \
    -DANDROID_ABI=arm64-v8a \
    -DCMAKE_SYSTEM_NAME=aarch64-linux-android \
    -DANDROID_STL=c++_shared \
    -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
    -DCMAKE_INSTALL_PREFIX=/usr/local/android/arm64 \
    -DENABLE_TEST=OFF \
    .
$ANDROID_NDK_HOME/prebuilt/linux-x86_64/bin/make -j
sudo $ANDROID_NDK_HOME/prebuilt/linux-x86_64/bin/make install
sudo cp contrib/android/include/* /usr/local/android/arm64/include/
```

### Build
To build NNabla using Android NDK, first clone the NNabla repository from github, then cmake and make.

```shell
git clone https://github.com/sony/nnabla
mkdir nnabla/build
cd nnabla/build
cmake \
    -DCMAKE_TOOLCHAIN_FILE=$ANDROID_NDK_HOME/build/cmake/android.toolchain.cmake \
    -DANDROID_ABI=arm64-v8a \
    -DANDROID_PLATFORM=android-33 \
    -DCMAKE_SYSTEM_NAME=aarch64-linux-android \
    -DANDROID_STL=c++_shared \
    -DBUILD_CPP_UTILS=ON \
    -DBUILD_PYTHON_PACKAGE=OFF \
    ..
$ANDROID_NDK_HOME/prebuilt/linux-x86_64/bin/make -j
```

After successful build, you can find `libnnabla.so` and `libnnabla_utils.so` under `build/lib` folder of your system.


## Test
We can verify the NNabla C++ android libraries by running the sample application in the native layer.
To do so, we need to first setup required environment as follows:

* Download and setup android studio, please refer [here](https://developer.android.com/studio/) for details.
* Add the android SDK's platform-tools folder location to path variable.
`E.g. export PATH=$PATH:/home/nnabla/Android/Sdk/platform-tools`
* Launch emulator using AVD with suitable platform and architecture or Connect android device with USB debugging option enabled.
To verify the device or emulator running , please type following command:
```$adb devices```
Your device/emulator should be listed as shown in following example.

```shell
$ adb devices
List of devices attached
CB5A2AG8N7	device
```

NNabla android build script also compiles mnist\_runtime example program located in [here](https://github.com/sony/nnabla/tree/master/examples/cpp/mnist_runtime).
The executable can be found at `build-android/bin/mnist_runtime` on successful build of NNabla.
Follow the instructions present [here](https://github.com/sony/nnabla/blob/master/examples/cpp/mnist_runtime/README.md) for creating the nnp file which is required for c++ inferencing.

Execute the following commands to run the mnist c++ inferencing sample program in native layer.

```shell
$adb push build-android/bin/mnist_runtime /data/local/tmp/
$adb push examples/cpp/mnist_runtime/lenet_010000.nnp /data/local/tmp/
$adb push examples/cpp/mnist_runtime/5.pgm /data/local/tmp/
$adb push build-android/build_${platform}_${architecture}/${ABI}/. /data/local/tmp/
$adb shell
$cd /data/local/tmp
$export LD_LIBRARY_PATH=/data/local/tmp
$./mnist_runtime lenet_010000.nnp 5.pgm
```
You should see output as follows:
```shell
SO-03H:/data/local/tmp $ ./mnist_runtime lenet_010000.nnp 5.pgm
Executing...
Prediction scores: -18.3584 -23.4614 -10.6035 8.44051 -9.19585 43.2856 5.42506 -19.7537 13.8561 -3.35031
Prediction: 5
```
Congratulations your android build is successful!
