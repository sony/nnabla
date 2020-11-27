SET PYVER=%1

FOR /F "TOKENS=1 DELIMS=." %%A IN ("%PYVER%") DO SET PYVER_MAJOR=%%A
FOR /F "TOKENS=2 DELIMS=." %%A IN ("%PYVER%") DO SET PYVER_MINOR=%%A

IF [%BUILDID%] == [] (
   ECHO BUILDID is not set
   EXIT /b
)

if [%CONDA_PREFIX%] == [] (
   if NOT [%ChocolateyToolsLocation%] == [] (
      SET CONDA_PREFIX=%ChocolateyToolsLocation%\miniconda3
   )
)

IF NOT EXIST "%CONDA_PREFIX%" (
   ECHO "Please install miniconda3 with chocolatey or exec this script on Anaconda Prompt(miniconda3)".
   EXIT /b 255
)
SET CONDAENV=nnabla-build-%BUILDID%-py%PYVER_MAJOR%%PYVER_MINOR%

IF [%BUILDID%] == [local] (
   ECHO Skip Cleanup local CONDA environment.
   EXIT /b
)

CALL %CONDA_PREFIX%\Scripts\activate.bat
CALL %CONDA_PREFIX%\Scripts\activate.bat %CONDAENV%

SET ENVNAME=%CONDA_DEFAULT_ENV%
SET ENVDIR=%CONDA_PREFIX%

ECHO Cleanup CONDA environment.
CALL conda deactivate
CALL conda env remove -y -n %CONDAENV%
IF [%ENVNAME%] == [%CONDAENV%] (
   RMDIR /s /q %ENVDIR%
)

# Ignore error.
EXIT /b 0
