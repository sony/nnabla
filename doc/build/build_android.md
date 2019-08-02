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
```
make bwd-nnabla-cpplib-android
```
The above make will build NNabla C++ libraries and its dependent libraries within android docker container.  
After successful build, you can find the libraries under ${NNabla_Root_Dir}/build/build_${PLATFORM}_${ARCHITECTURE}/${ABI} folder of your system.  
The docker container uses Android NDK version r16b by default.  

If you want to build docker image by yourself, then please refer build instructions of docker file at [Android Dockerfile section](https://github.com/sony/nnabla/blob/master/docker/README.md)  

## Manual Build
This section explains the manual building of NNabla using android NDK.  

### Requirements
Following build dependencies needs to be installed by the user manually.  

* G++: `sudo apt-get install build-essential`  
* CMake>=3.11: `sudo apt-get install cmake`  
* Python: `sudo apt-get install python python-pip`(Used by code generator)  
  * Python packages: PyYAML and MAKO: `sudo -H pip install pyyaml mako`  
  * Python setup tools:		      `sudo apt install python-dev python-setuptools`  
* curl, make and git: `sudo apt-get install -y curl make git`  
* Android NDK: Download the required NDK version from [https://developer.android.com/ndk/downloads/](https://developer.android.com/ndk/downloads/).  
  Extract the downloaded NDK into folder.  
 NNabla android build is tested with android-ndk-r16b, however the build should work with other NDK version as well.  

* Protobuf >=3: See below.

#### Installing protobuf3 C++ libraries and tools

##### Install from source.

Unlike [Python Package compilation](./build.md) which requires
`protoc` compiler only, the NNabla C++ utility library requires
protobuf C++ library too.  The following snippet running on your
terminal will build and install protobuf-3.1.0 from source.

```shell
curl -L https://github.com/google/protobuf/archive/v3.1.0.tar.gz -o protobuf-v3.1.0.tar.gz
tar xvf protobuf-v3.1.0.tar.gz
cd protobuf-3.1.0
mkdir build && cd build
cmake -DCMAKE_POSITION_INDEPENDENT_CODE=ON -Dprotobuf_BUILD_TESTS=OFF ../cmake
make
sudo make install  # See a note below if you want your system clean.
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

##### Install from PPA package.

Here is the procedure using an informal PPA package. If you can not
trust unofficial packages, please use the procedure to build from the
source shown above.

```shell
sudo add-apt-repository ppa:maarten-fonville/protobuf
sudo apt install protobuf-compiler libprotoc-dev libprotobuf-dev
```

### Build
To build NNabla using Android NDK, first clone the NNabla repository from git.  
```shell
git clone https://github.com/sony/nnabla
```
Execute the build_nnabla.sh script with appropriate options.  
```
cd ${NNabla_Root_Dir}/build-tools/android
sh build_nnabla.sh -p=android-XX -a=[arm|arm64|x86|x86_64] -n=Path_to_your_ndk_folder -e=[armeabi|armeabi-v7a|arm64-v8a|x86|x86_64] (-e option is optional)
E.g. sh build_nnabla.sh -p=android-26 -a=arm64 -n=/usr/local/src/android-ndk-r16b -e=arm64-v8a
```
The build_nnabla.sh script takes mainly 4 arguments,  
-p (platform) option is to specify the android API level.  
-a (architecture) option is to specify the target architecture.  
-n (ndk path) option is used to specify the path to Android NDK used for building.  
-e (ABI) option is to specify the instruction set.  
<br>
The above script will build NNabla C++ libraries and its dependent libraries.  
After successful build, you can find the libraries under ${NNabla_Root_Dir}/build/build_${PLATFORM}_${ARCHITECTURE}/${ABI} folder of your system.  
<br>

#### Troubleshooting:
The build script modifies the some of NNabla source files dynamically to add support for android build.  
At the end of the script the source modifications are reverted back.  
Stopping the script in the middle might cause for build failure for later builds.  
Request user to reset the NNabla repository and try the build again.  

## Test
We can verify the NNabla C++ android libraries by running the sample application in the native layer.  
To do so, we need to first setup required environment as follows:   
<br>
* Download and setup android studio, please refer [here](https://developer.android.com/studio/) for details.  
* Add the android SDK's platform-tools folder location to path variable.  
`E.g. export PATH=$PATH:/home/nnabla/Android/Sdk/platform-tools`  
* Launch emulator using AVD with suitable platform and architecture or Connect android device with USB debugging option enabled.  
To verify the device or emulator running , please type following command:  
```$adb devices```  
Your device/emulator should be listed as shown in following example.  
```
$ adb devices
List of devices attached
CB5A2AG8N7	device
```

NNabla android build script also compiles mnist_runtime example program located in [here](https://github.com/sony/nnabla/tree/master/examples/cpp/mnist_runtime).  
The executable can be found at ${NNabla_Root_Dir}/build/bin/mnist_runtime on successful build of NNabla.  
Follow the instructions present [here](https://github.com/sony/nnabla/blob/master/examples/cpp/mnist_runtime/README.md) for creating the nnp file which is required for c++ inferencing.  
<br>
Execute the following commands to run the mnist c++ inferencing sample program in native layer.  
```
$adb push ${NNabla_Root_Dir}/build/bin/mnist_runtime /data/local/tmp/
$adb push ${NNabla_Root_Dir}/examples/cpp/mnist_runtime/lenet_010000.nnp /data/local/tmp/
$adb push ${NNabla_Root_Dir}/examples/cpp/mnist_runtime/5.pgm /data/local/tmp/
$adb push ${NNabla_Root_Dir}/build/android/build_${platform}_${architecture}/${ABI}/libarchive.so /data/local/tmp/
$adb push ${NNabla_Root_Dir}/build/android/build_${platform}_${architecture}/${ABI}/libnnabla.so /data/local/tmp/
$adb push ${NNabla_Root_Dir}/build/android/build_${platform}_${architecture}/${ABI}/libnnabla_utils.so /data/local/tmp/
$adb shell
$cd /data/local/tmp
$export LD_LIBRARY_PATH=/data/local/tmp
$./mnist_runtime lenet_010000.nnp 5.pgm
```
You should see output as follows:  
```
SO-03H:/data/local/tmp $ ./mnist_runtime lenet_010000.nnp 5.pgm
Executing...
Prediction scores: -18.3584 -23.4614 -10.6035 8.44051 -9.19585 43.2856 5.42506 -19.7537 13.8561 -3.35031
Prediction: 5
```
Congratulations your android build is successful!.


 




