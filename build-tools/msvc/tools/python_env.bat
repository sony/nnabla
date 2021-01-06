SET PYVER=%1
IF [%PYVER%] == [] (
   ECHO Please specify Python version 3.6, 3.7 or 3.8.
   EXIT /b 255
)
FOR /F "TOKENS=1 DELIMS=." %%A IN ("%PYVER%") DO SET PYVER_MAJOR=%%A
FOR /F "TOKENS=2 DELIMS=." %%A IN ("%PYVER%") DO SET PYVER_MINOR=%%A

REM Miniconda
IF [%BUILDID%] == [] SET BUILDID=local
SET CONDAENV=nnabla-build-%BUILDID%-py%PYVER_MAJOR%%PYVER_MINOR%

if [%CONDA_PREFIX%] == [] (
   if NOT [%ChocolateyToolsLocation%] == [] (
      SET CONDA_PREFIX=%ChocolateyToolsLocation%\miniconda3
   )
)

IF NOT EXIST "%CONDA_PREFIX%" (
   ECHO "Please install miniconda3 with chocolatey or exec this script on Anaconda Prompt(miniconda3)".
   EXIT /b 255
)

SET BACK_CONDA_PREFIX=%CONDA_PREFIX%
CALL %CONDA_PREFIX%\Scripts\activate.bat %CONDAENV%
IF NOT [%CONDA_PREFIX%] == [%BACK_CONDA_PREFIX%] GOTO :SKIP_CONDA_ENV_CREATE

SETLOCAL
CALL %CONDA_PREFIX%\Scripts\activate.bat
CALL conda create -y -n %CONDAENV% python=%PYVER%
ENDLOCAL
CALL %CONDA_PREFIX%\Scripts\activate.bat %CONDAENV%

:SKIP_CONDA_ENV_CREATE

CALL conda install -y --update-deps ^
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

CALL conda install -y -c conda-forge pynvml
