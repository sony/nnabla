# Build C++ utility libraries for Android Platform  

This document describes how to build Nnabla C++ libraries using Android Native Development Kit(NDK).  
The following instruction demonstrates the build procedure on Ubuntu 16.04 using Android NDK.  
<br>
Most of the procedure is the same as [Build C++ utility libraries](build_cpp_utils.md).  

## Requirements

The android build dependencies can be installed manually by the user or they can use the Docker file.

#### Manual installation
The following explains the manual installation of the dependencies:  

* G++: `sudo apt-get install build-essential`  
* CMake>=3.11: `sudo apt-get install cmake`  
* Python: `sudo apt-get install python python-pip`(Used by code generator)  
  * Python packages: PyYAML and MAKO: `sudo -H pip install pyyaml mako`  
  * Python setup tools:		      `sudo apt install python-dev python-setuptools`  
* curl, make and git: `sudo apt-get install -y curl make git`  
* Android NDK: Download the required NDK version from `https://developer.android.com/ndk/downloads/`.  
  Extract the downloaded NDK into folder.

<br>
Nnabla android build is tested with android-ndk-r16b, however the build should work with other NDK version as well.  

#### Build Docker image
For build instructions on the docker file please refer to [NNabla Android Docker](https://github.com/sony/nnabla/blob/master/docker/README.md)

## Build

```shell
git clone https://github.com/sony/nnabla
cd nnabla/build-tools/android
sh build_nnabla.sh -p=android-XX -a=[arm|arm64|x86|x86_64] -n=Path_to_your_ndk_folder -e=[armeabi|armeabi-v7a|arm64-v8a|x86|x86_64] (-e option is optional)
E.g. sh build_nnabla.sh -p=android-26 -a=arm64 -n=/usr/local/src/android-ndk-r16b -e=arm64-v8a 
```
The build_nnabla.sh script takes mainly 4 arguments,  
-p (platform) option is to specify the android API level.  
-a (architecture) option is to specify the target architecture.  
-n (ndk path) option is used to specify the path to Android NDK used for building.  
-e (ABI) option is to specify the instruction set.  
<br>
The above script will build Nnabla C++ libraries and its dependent libraries.  
After successful build, you can find the libraries under nnabla/build/build_${PLATFORM}_${ARCHITECTURE}/${ABI} folder of your system.  

<br>
#### Troubleshooting:
The build script modifies the nnabla source dynamically to add support for android build.  
At the end of the script the source modifications are reverted back.  
Stopping the script in the middle might cause for build failure for later builds.  
Request user to reset the nnabla repository and try the build again.  

## Test 

TODO
