# Quick build tools

We provide easiest way to build NNabla with GNU make and docker.
On windows there is helper BATCH files.

## Linux

### Prepare
You need to install just [Docker](https://docs.docker.com/install/) and `make`. 

### Build
```
$ make all
```

`make all` is as same as `make bwd-cpplib bwd-wheel`
After this you can find .whl file in `nnabla/build_wheel_py??/dist/`


### Build cpplib only

If you want only cpp libraries.
```
$ make bwd-cpplib
```
After this you can find executable file and shared library in `nnabla/build/lib/` and `nnabla/build/bin/`

### Specify python version

Prepare to specify python version.
```
$ export PYTHON_VERSION_MAJOR=2
$ export PYTHON_VERSION_MINOR=7
```

Then you can get with,
```
$ make all
```

Or you can specify every time.
```
$ make PYTHON_VERSION_MAJOR=2 PYTHON_VERSION_MINOR=7 all
```

## Windows

### Prepare

Please see [Official site](https://chocolatey.org/install)
After installing Chocolatey do following command on Administrator cmd.exe.
```
choco feature enable -n allowGlobalConfirmation
choco install cmake git miniconda3 vcbuildtools
set PATH=C:\tools\Miniconda3\Scripts;%PATH%
conda install -y pywin32 Cython=0.25 boto3 protobuf h5py ipython numpy=1.11 pip pytest scikit-image scipy wheel pyyaml mako
pip install -U tqdm
```
And make sure that `C:\tools\Miniconda3\Scripts` in your PATH environment.


### Build cpplib
```
> call build-tools\msvc\build_cpplib.bat
```

### Build wheel
```
> call build-tools\msvc\build_wheel.bat
```
