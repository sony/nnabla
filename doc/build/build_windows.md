# Build Python Package on Windows

## Prerequisites

### Chocolatey

Install chocolatey with instruction in [Official page](https://chocolatey.org/）

Then, install required tools with following command.
```bat
    choco install cmake git visualstudio2019-workload-vctools visualstudio2019buildtools
```

Note: Please make sure `Microsoft Visual Studio` is installed in the path of `C:\program files (x86)` and the path exists in `C:\Program Files (x86)\Microsoft Visual Studio\2019\BuildTools\VC\Auxiliary\Build\vcvars64.bat` and `C:\Program Files "("x86")"\Microsoft Visual Studio\2019\BuildTools\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin`.

Install python3.7.9 or 3.8.10 or 3.9.13 or 3.10.8. You can refer to the following command.
```bat
    choco install -y python3 --version=3.7.9
```

We use choclatey to make the configuration as easy as possible（recommended).
If you can't or don't want to use chocolatey, please do so yourself.

- [CMake](https://cmake.org/download/)
- [Git for windows](https://gitforwindows.org/)
- [Python3](https://www.python.org/downloads/)
- [VisualC++ Build Tools 2019](https://my.visualstudio.com/Downloads?q=visual%20studio%202019&wt.mc_id=o~msft~vscom~older-downloads)

Install and set the environment variables appropriately.

### Build python package

You can build windows binary with following command.
```cmd
    git clone https://github.com/sony/nnabla.git
    cd nnabla
```

```bat
    build-tools\msvc\build_cpplib.bat
    build-tools\msvc\build_wheel.bat PYTHON_VERSION
```
The python version we tested is 3.7, 3.8 3.9 and 3.10.

Then you can run test with following.
```bat
    build-tools\msvc\test.bat PYTHON_VERSION
```

## FAQ

* Q. The compiled library and executable run on a compiled PC, but it does not work on another PC.
  * Confirm that the Microsoft Visual C++ 2019 Redistributable is installed on target PC.
