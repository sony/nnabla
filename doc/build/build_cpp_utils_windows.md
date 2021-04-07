# Build C++ libraries on Windows


## Prerequisites

### Chocolatey

```bat
    choco install cmake git VisualCppBuildTools
```


```bat
    choco install cmake git visualstudio2019-workload-vctools
```


```bat
    choco install -y python3 --version=3.6.8
```

## Build

First, clone [nnabla](https://github.com/sony/nnabla) and go into the root folder.

Then, the following batch script does everything including setting up the rest of dependencies and compiling libraries.

```bat
    cmd /c nnabla\build-tools\msvc\build_cpplib.bat
```
```bat
    cmd /c nnabla\build-tools\msvc\test_nbla.bat
```

```bat
    cmd /c nnabla\build-tools\msvc\build_cpplib.bat 2019
```
```bat
    cmd /c nnabla\build-tools\msvc\test_nbla.bat 2019
```




This will setup the following dependency libraries of the NNabla C++ utility

* LibArchive
* ZLib
* Protobuf

into the `third_party` folder, and these are used when compiling and running NNabla utility library.
Note that HDF5 is not supported on Windows so far, which means you can not use a `.h5` parameter file in C++ inference/training.
(TODO: Write how to create `.protobuf` file from `.nnp` or `.h5`).

It also sets up NNabla core library and the C++ utility library (`nnabla.dll`, `nnabla_utils.dll` and their `.lib` and `.exp` files).

If you want to build with Debug mode, you have to set an environment variable `build_type` as following before running the batch script above.

```bat
set build_type=Debug
```

## Use the library in your C++ application

To build your C++ binary with NNabla C++ utilities, you need:

* Set `<nnabla root>\include` folder as include path
* Set `nnabla.lib` and `nnabla_utils.lib` as libraries (use `.dlib` when Debug mode)

At runtime, you will need the following dynamic link libraries located in a right path.

* `nnabla.dll`
* `nnabla_utils.dll`
* `zlibwapi.dll`
* `archive.dll`

Please find these libraries built in this instruction by searching them at NNabla root folder.
