File format converter
=====================

Overview
--------

.. image:: file_format_converter_workflow.png

File format converter will realize Neural Network Libraries (or
Console) workflow with ONNX file format, and also NNabla C Runtime.

File format converter has following functions.

- Convert NNP valiations to valid NNP
- Convert ONNX to NNP
- Convert NNP to ONNX
- Convert NNP to NNB(Binary format for NNabla C Runtime)
- Convert NNP to C Source code for NNabla C Runtime
  
NNP
---

**NNP** is file format of NNabla.

NNP format is described at :any:`../../format`.

But with this file format converter is work with several variation of NNP.

- Standard NNP format (.nnp)
- Contents of NNP files(.nntxt, .prototxt, .h5, .protobuf)


Usage
+++++

Convert NNP to NNP
~~~~~~~~~~~~~~~~~~

Sometimes we neeed convert NNP to NNP.

Most major usecase, expand repeat or recurrent network supported by
Neural Network Console but does not supported by C++ API.

.. code-block:: none

   $ nnabla_cli convert --nnp-expand-network input.nnp output.nnp


Convert Contents of NNP to NNP
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Current version of Neural Network Console outputs .nntxt and .h5 as
training result.
Then we need to convert separated files into single NNP.

.. code-block:: none

   $ nnabla_cli convert --nnp-expand-network net.nntxt parameters.h5 output.nnp


ONNX converter
--------------

Limitation
++++++++++

- Training does not supported
- Only supports operatior set 3
- Not all functions are supported. See :any:`onnx/operator_coverage`.
- Only limited Neural Network Console projects supported.  See :any:`onnx/neural_network_console_example_coverage`.

Usage
+++++


.. code-block:: none

   $ nnabla_cli convert input.nnp output.onnx

.. code-block:: none

   $ nnabla_cli convert --nnp-expand-network input.nnp output.onnx

.. code-block:: none

   $ nnabla_cli convert input.onnx output.nnp



Work with NNabla C Runtime
--------------------------

NNB
+++

C Source Code
+++++++++++++

Usage
+++++

.. code-block:: none

   $ nnabla_cli convert input.nnp output.nnb

.. code-block:: none

   $ nnabla_cli convert --nnp-expand-network input.nnp output.nnb

.. code-block:: none

   $ nnabla_cli convert -O CSRC input.onnx output-dir



