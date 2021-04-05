SET PYVER=%1

FOR /F "TOKENS=1 DELIMS=." %%A IN ("%PYVER%") DO SET PYVER_MAJOR=%%A
FOR /F "TOKENS=2 DELIMS=." %%A IN ("%PYVER%") DO SET PYVER_MINOR=%%A

IF [%BUILDID%] == [] (
   ECHO BUILDID is not set
   EXIT /b
)

SET VENV=%CD%\build-env-%BUILDID%-py%PYVER_MAJOR%%PYVER_MINOR%

IF [%BUILDID%] == [local] (
   ECHO Skip Cleanup local virtual environment.
   EXIT /b
)

ECHO Cleanup virtual environment.
CALL %VENV%\Scripts\activate.bat
CALL %VENV%\Scripts\deactivate.bat

RMDIR /s /q %VENV%

REM Ignore error.
EXIT /b 0
