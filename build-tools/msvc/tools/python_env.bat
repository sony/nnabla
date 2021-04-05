SET PYVER=%1
IF [%PYVER%] == [] (
   ECHO Please specify Python version 3.6, 3.7 or 3.8.
   EXIT /b 255
)
FOR /F "TOKENS=1 DELIMS=." %%A IN ("%PYVER%") DO SET PYVER_MAJOR=%%A
FOR /F "TOKENS=2 DELIMS=." %%A IN ("%PYVER%") DO SET PYVER_MINOR=%%A

REM VENV
IF [%BUILDID%] == [] SET BUILDID=local
SET VENV=%CD%\build-env-%BUILDID%-py%PYVER_MAJOR%%PYVER_MINOR%

if [%PYTHON_DIR%] == [] (
   SET PYTHON_DIR=C:\Python%PYVER_MAJOR%%PYVER_MINOR%
)

IF NOT EXIST "%PYTHON_DIR%" (
   ECHO "Please install python%PYVER_MAJOR%%PYVER_MINOR% with chocolatey".
   EXIT /b 255
)

%PYTHON_DIR%\python.exe -m venv %VENV%

IF NOT EXIST "%VENV%" (
   ECHO "Failed to create virtual env".
   EXIT /b 255
)

if [%VENV_PYTHON_PKG_DIR%] == [] (
   SET VENV_PYTHON_PKG_DIR=%VENV%\\Lib\\site-packages
)

CALL %VENV%\Scripts\activate.bat

CALL python -m pip install --upgrade pip

CALL pip install ^
           Cython ^
           boto3 ^
           h5py ^
           ipython ^
           mako ^
           numpy ^
           pip ^
           protobuf ^
           pytest ^
           pywin32 ^
           pyyaml ^
           scikit-image ^
           scipy ^
           six ^
           tqdm ^
           virtualenv ^
           wheel

CALL pip install pynvml
