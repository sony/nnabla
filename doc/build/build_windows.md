# Build Python Package on Windows

## Prerequisites

### Chocolatey

Install chocolatey with instruction in [Official page](https://chocolatey.org/)

Then, install required tools with following command.
```bat
    choco upgrade all
    choco install -y git visualstudio2019-workload-vctools visualstudio2019buildtools
    choco install -y zip unzip
    choco install -y pyenv-win
```
Install 3.7.9 , 3.8.10 or 3.9.4
```bat
    choco install -y python3 --version=3.7.9
```

We use choclatey to make the configuration as easy as possible.
If you can't or don't want to use chocolatey, please do so yourself.

- [Git for windows](https://gitforwindows.org/)
- [Python3](https://www.python.org/downloads/)
- [VisualC++ Build Tools 2019](https://visualstudio.microsoft.com/downloads/)

Install and set the environment variables appropriately.

Tips:
1. The cmake is included in VisualC++ Build Tools 2019 package, but build script
also depends on the cmake on the path Program Files/CMake.
2. Short filename is needed for some cases, which can be resolved by a special copy.

```bat
  powershell ^"robocopy /MIR /W:3 /R:3 /COPY:DT 'C:\Program Files (x86)\Microsoft Visual Studio\2019\BuildTools\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake' 'C:\Program Files\CMake'^"
  powershell ^"robocopy /MIR /W:3 /R:3 /COPY:DT 'C:\Program Files (x86)\Microsoft Visual Studio\2019' C:\vs\2019^"
```

### Build python package

You can build windows binary with following command.
```bat
    build-tools\msvc\build_cpplib.bat
    build-tools\msvc\build_wheel.bat PYTHON_VERSION
```
The python version we tested is 3.7, 3.8 and 3.9.

Then you can run test with following.
```bat
    build-tools\msvc\test.bat PYTHON_VERSION
```

## FAQ

* Q. The compiled library and executable run on a compiled PC, but it does not work on another PC.
  * Confirm that the Microsoft Visual C++ 2019 Redistributable is installed on target PC.
