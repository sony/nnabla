# Adding a new function

## Overview

TODO: Write me.

## Prerequisites

TODO: Write me.

## Write a definition in YAML

TODO: Write me.

## Generate code template by CMake

TODO: Write me.

## Write your function implementation

TODO: Write me.

## Write unit testing

TODO: Write me.

NOTE: Unit testing in C++ is often demanded by embedded engineers. Writing unit testing in C++ side from scratch from now requires much effort, and imposes double-maintenance cost over C++ and Python unittests. Also, testing in Python is much easier to debug. Hence, a possible solution to this currently we are thinking is that creating a framwork generating C++ unit testing code of each functions from our Python testing framework (`function_tester`). However, we do not have any specific plan to add this feature soon. Contributions are welcome.

## (DO NOT FORGET!) Add a function doc to sphinx.

TODO: Write me.

## You want to add a CUDA function too?

We know many developers use CUDA to accelerate processing speed. See contribution guide of [CUDA extension](https://github.com/sony/nnabla-ext-cuda/blob/master/CONTRIBUTING.md).
