# Build Manuals

## Quick Build Tools

Although we provide a CMake standard build workflow described in the
following sections, for users working on Linux, we also provide very
useful quick build tools based on Docker and GNU make, which enables
users to build everything including setting up dependencies, building
C++ libraries and Python package, only by a single command.

For Windows users, we also provide a quick build procedure based on
[Chocolatey](https://chocolatey.org) and batch scripts.

Note that Linux quick build tools only work for x86 architectures, and
it's only tested on Ubuntu host PC.

* [Quick Build Tools](quick_build_tools.md)

## Standard CMake workflow

### Python Package

* [Linux](build.md)
  * [With distributed execution support](build_distributed.md)
* [Windows](build_windows.md)
* [macOS](build_macos.md)

### Build C++ utilities

* [Linux/macOS](build_cpp_utils.md)
* [Windows](build_cpp_utils_windows.md)
* [Android](build_android.md)

### Extensions

* [CUDA](https://github.com/sony/nnabla-ext-cuda/tree/master/doc/build/README.md)

