# Build Python Package on Windows

## Prerequisites

### Chocolatey

Install chocolatey with instruction in [Official page](https://chocolatey.org/)

Then, install required tools with following command.
```bat
    choco install cmake git VisualCppBuildTools
```
Install python3.6.8 or 3.7.9 or 3.8.8
```bat
    choco install -y python3 --version=3.6.8
```

We use choclatey to make the configuration as easy as possible.
If you can't or don't want to use chocolatey, please do so yourself.

- [CMake](https://cmake.org/download/)
- [Git for windows](https://gitforwindows.org/)
- [Python3](https://www.python.org/downloads/)
- [VisualC++ Build Tools 2015](https://www.microsoft.com/en-US/download/details.aspx?id=48159)

Install and set the environment variables appropriately.

### Build python package

You can build windows binary with following command.
```bat
    build-tools\msvc\build_cpplib.bat
    build-tools\msvc\build_wheel.bat PYTHON_VERSION
```
The python version we tested is 3.6, 3.7 and 3.8.

Then you can run test with following.
```bat
    build-tools\msvc\test.bat PYTHON_VERSION
```

## FAQ

* Q. The compiled library and executable run on a compiled PC, but it does not work on another PC.
  * Confirm that the Microsoft Visual C++ 2015 Redistributable is installed on target PC.
