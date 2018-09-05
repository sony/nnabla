File format converter
=====================

Overview
--------


.. blockdiag::

    blockdiag {
      default_fontsize=8
      span_width = 32;
      span_height = 20;

      NNabla1 [label = "NNabla", color="lime", shape="roundedbox", width=80, height=60, fontsize=12];
      NNabla2 [label = "Use NNabla as\nRuntime", color="lime", shape="roundedbox", width=80, height=60];
      Other [label = "Other\n(Caffe2 etc.)", shape="roundedbox", width=80, height=60];
      ONNX1 [label = "ONNX", color="mediumslateblue", width=40, height=20];
      ONNX2 [label = "ONNX", color="mediumslateblue", width=40, height=20];
      Conv1 [label = "File Format\nConverter", color="lime", shape="roundedbox", width=80, height=60, fontsize=10];
      Conv2 [label = "File Format\nConverter", color="lime", shape="roundedbox", width=80, height=60, fontsize=10];
      NNP1 [label = "NNP", color="cyan", width=40, height=20];
      NNP2 [label = "NNP", color="cyan", width=40, height=20];
      NNB [label = "NNB", color="cyan", width=40, height=20];
      CSRC [label = "C Source\ncode", color="seagreen", width=40];
      OtherRuntime [label = "Other runtime", shape="roundedbox", width=80];
      NNablaCRuntime [label = "NNabla C\nRuntime", color="lime", shape="roundedbox", width=80];
      Product [label = "Implement to\nproduct", shape="roundedbox", width=80, height=60];
      
      NNabla1 -> NNP1;
      Other -> ONNX1 -> Conv1 -> NNP1;
      NNP1 -> Conv2;
      Conv2 -> ONNX2 -> OtherRuntime;
      Conv2 -> NNB -> NNablaCRuntime;
      Conv2 -> CSRC -> Product;
      Conv2 -> NNP2 -> NNabla2;
    }


File format converter will realize Neural Network Libraries (or
Console) workflow with ONNX file format, and also NNabla C Runtime.

File format converter has following functions.

- Convert NNP variations to valid NNP
- Convert ONNX to NNP
- Convert NNP to ONNX
- Convert NNP to NNB(Binary format for NNabla C Runtime)
- Experimental: Convert NNP to C Source code for NNabla C Runtime

**IMPORTANT NOTICE**: This file format converter still has some known problems.

- Supported ONNX operator is limited. See :any:`onnx/operator_coverage`.
- Converting NNP to C Source code is still experimental. It should work but did not tested well.


Architecture
+++++++++++++


.. blockdiag::

    blockdiag {
      default_group_color = white;

      INPUT [label="<<file>>\nINPUT", color="lime"];
      OUTPUT [label="<<file>>\nOUTPUT", color="lime"];
      PROCESS [label="Process\n(Split, Expand, etc.)", shape="roundedbox"];
      proto [label="proto", color="cyan", width=60, height=20];

      
      INPUT -> proto [label="import"];
      group {
        orientation = portrait;
        proto <-> PROCESS;
      }
      proto -> OUTPUT [label="export"];
    }


This file format converter uses protobuf defined in Neural Network Libraries as intermediate format.

While this is not a generic file format converter, this is the specified converter for Neural Network Libraries.

This converter can specify both inputs and outputs for ONNX file, but if ONNX file contains a function unsupported by Neural Network Libraries, it may cause error in conversion.

This converter also provides some intermediate process functionalities. See :ref:Process.

Conversion
++++++++++

Supported Formats
^^^^^^^^^^^^^^^^^

NNP
^^^

**NNP** is file format of NNabla.

NNP format is described at :any:`../../format`.

But with this file format converter is work with several variation of NNP.

- Standard NNP format (.nnp)
- Contents of NNP files(.nntxt, .prototxt, .h5, .protobuf)


ONNX
^^^^

Limitation
++++++++++

- Training is not supported
- Only supports operator set 3
- Not all functions are supported. See :any:`onnx/operator_coverage`.
- Only limited Neural Network Console projects supported.  See :any:`onnx/neural_network_console_example_coverage`.
- In some case you must install onnx package by hand. For example you can install with command `pip install onnx` or if you want to install system wide, you can install with command `sudo -HE pip install onnx`.
  
NNB
^^^

NNB is compact binary format for NNabla C Runtime.
It is designed for `nnabla-c-runtime`_.

.. _nnabla-c-runtime: https://github.com/sony/nnabla-c-runtime


C Source Code
^^^^^^^^^^^^^

File format converter supports C source code output for `nnabla-c-runtime`_.

Process
+++++++

Expand Repeat and Recurrent
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Neural Network Console supports `LoopControl` pseudo functions `RepeatStart`_,  `RepeatEnd`_, `RecurrentInput`_, `RecurrentOutput`_ or `Delay`_.

Currently, these functions are not supported by Neural Network Libraries directly.

The file format converter expands the network and removes these pseudo functions by default.

.. _RepeatStart: https://support.dl.sony.com/docs/layer_reference/#RepeatStart
.. _RepeatEnd: https://support.dl.sony.com/docs/layer_reference/#RepeatEnd
.. _RecurrentInput: https://support.dl.sony.com/docs/layer_reference/#RecurrentInput
.. _RecurrentOutput: https://support.dl.sony.com/docs/layer_reference/#RecurrentOutput
.. _Delay: https://support.dl.sony.com/docs/layer_reference/#Delay

If you want to preserve these, specify command line option `--nnp-no-expand-network` when converting files.


Split network
^^^^^^^^^^^^^

You can split network with `--split` option.

See :ref:`Splitting network` to use this functionality.

  
Usage
-----

NNP Operation
+++++++++++++

Convert NNP to NNP
^^^^^^^^^^^^^^^^^^

Sometimes we need convert NNP to NNP.

Most major usecase, expand repeat or recurrent network supported by
Neural Network Console but does not supported by C++ API.

.. code-block:: none

   $ nnabla_cli convert --nnp-no-expand-network input.nnp output.nnp

Convert console output to single NNP file
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Current version of Neural Network Console outputs .nntxt and .h5 as
training result.

Then we need to convert separated files into single NNP and parameters
store with protobuf format.

.. code-block:: none

   $ nnabla_cli convert net.nntxt parameters.h5 output.nnp


Convert console output to single NNP file without expanding Repeat or recurrent.
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: none

   $ nnabla_cli convert --nnp-no-expand-network net.nntxt parameters.h5 output.nnp

Keep parameter format as hdf5
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: none

   $ nnabla_cli convert --nnp-no-expand-network --nnp-parameter-h5 net.nntxt parameters.h5 output.nnp

Everything into single nntxt.
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: none

   $ nnabla_cli convert --nnp-parameter-nntxt net.nntxt parameters.h5 output.nntxt

ONNX Operation
++++++++++++++

Convert NNP to ONNX
^^^^^^^^^^^^^^^^^^^

.. code-block:: none

   $ nnabla_cli convert input.nnp output.onnx

Convert ONNX to NNP
^^^^^^^^^^^^^^^^^^^

.. code-block:: none

   $ nnabla_cli convert input.onnx output.nnp


C Runtime Operation
+++++++++++++++++++

Convert NNP to NNB
^^^^^^^^^^^^^^^^^^

.. code-block:: none

   $ nnabla_cli convert input.nnp output.nnb

Convert NNP to C source code.
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: none

   $ nnabla_cli convert -O CSRC input.onnx output-dir


Splitting network
+++++++++++++++++

Splitting network is a bit complicated and can be troublesome.


NNP file could have multiple Executor networks, but Split supports only single network to split.

First, you must confirm how many Executors there are in the NNP, and specify what executor to split with `nnabla_cli dump`.

.. code-block:: none
   
    $ nnabla_cli dump squeezenet11.files/SqueezeNet-1.1/*.{nntxt,h5}
    2018-08-27 15:02:40,006 [nnabla][INFO]: Initializing CPU extension...
    Importing squeezenet11.files/SqueezeNet-1.1/net.nntxt
    Importing squeezenet11.files/SqueezeNet-1.1/parameters.h5
     Expanding Training.
     Expanding Top5Error.
     Expanding Top1Error.
     Expanding Runtime.
      Optimizer[0]: Optimizer
      Optimizer[0]:  (In) Data      variable[0]: Name:TrainingInput                  Shape:[-1, 3, 480, 480]
      Optimizer[0]:  (In) Data      variable[1]: Name:SoftmaxCrossEntropy_T          Shape:[-1, 1]
      Optimizer[0]:  (Out)Loss      variable[0]: Name:SoftmaxCrossEntropy            Shape:[-1, 1]
      Monitor  [0]: train_error
      Monitor  [0]:  (In) Data      variable[0]: Name:Input                          Shape:[-1, 3, 320, 320]
      Monitor  [0]:  (In) Data      variable[1]: Name:Top5Error_T                    Shape:[-1, 1]
      Monitor  [0]:  (Out)Monitor   variable[0]: Name:Top5Error                      Shape:[-1, 1]
      Monitor  [1]: valid_error
      Monitor  [1]:  (In) Data      variable[0]: Name:Input                          Shape:[-1, 3, 320, 320]
      Monitor  [1]:  (In) Data      variable[1]: Name:Top1rror_T                     Shape:[-1, 1]
      Monitor  [1]:  (Out)Monitor   variable[0]: Name:Top1rror                       Shape:[-1, 1]
      Executor [0]: Executor
      Executor [0]:  (In) Data      variable[0]: Name:Input                          Shape:[-1, 3, 320, 320]
      Executor [0]:  (Out)Output    variable[0]: Name:y'                             Shape:[-1, 1000]



As above output now you know only 1 executor.

Then you can show executor information with `nnabla_cli dump -E0`.

.. code-block:: none
   
    $ nnabla_cli dump -E0 squeezenet11.files/SqueezeNet-1.1/*.{nntxt,h5}
    2018-08-27 15:03:26,547 [nnabla][INFO]: Initializing CPU extension...
    Importing squeezenet11.files/SqueezeNet-1.1/net.nntxt
    Importing squeezenet11.files/SqueezeNet-1.1/parameters.h5
     Try to leave only executor[Executor].
     Expanding Runtime.
      Executor [0]: Executor
      Executor [0]:  (In) Data      variable[0]: Name:Input                          Shape:[-1, 3, 320, 320]
      Executor [0]:  (Out)Output    variable[0]: Name:y'                             Shape:[-1, 1000]

You can get list of function adding `-F` option.

.. code-block:: none
   
    $ nnabla_cli dump -FE0 squeezenet11.files/SqueezeNet-1.1/*.{nntxt,h5}
    2018-08-27 15:04:10,954 [nnabla][INFO]: Initializing CPU extension...
    Importing squeezenet11.files/SqueezeNet-1.1/net.nntxt
    Importing squeezenet11.files/SqueezeNet-1.1/parameters.h5
     Try to leave only executor[Executor].
     Expanding Runtime.
      Executor [0]: Executor
      Executor [0]:  (In) Data      variable[0]: Name:Input                          Shape:[-1, 3, 320, 320]
      Executor [0]:  (Out)Output    variable[0]: Name:y'                             Shape:[-1, 1000]
      Executor [0]:   Function[  0  ]: Type: Slice                Name: Slice
      Executor [0]:   Function[  1  ]: Type: ImageAugmentation    Name: ImageAugmentation
      Executor [0]:   Function[  2  ]: Type: MulScalar            Name: SqueezeNet/MulScalar
      Executor [0]:   Function[  3  ]: Type: AddScalar            Name: SqueezeNet/AddScalar
      Executor [0]:   Function[  4  ]: Type: Convolution          Name: SqueezeNet/Convolution
      Executor [0]:   Function[  5  ]: Type: ReLU                 Name: SqueezeNet/ReLU
      Executor [0]:   Function[  6  ]: Type: MaxPooling           Name: SqueezeNet/MaxPooling
    
        SNIP...
    
      Executor [0]:   Function[ 63  ]: Type: ReLU                 Name: SqueezeNet/FireModule_8/Expand1x1ReLU
      Executor [0]:   Function[ 64  ]: Type: Concatenate          Name: SqueezeNet/FireModule_8/Concatenate
      Executor [0]:   Function[ 65  ]: Type: Dropout              Name: SqueezeNet/Dropout
      Executor [0]:   Function[ 66  ]: Type: Convolution          Name: SqueezeNet/Convolution_2
      Executor [0]:   Function[ 67  ]: Type: ReLU                 Name: SqueezeNet/ReLU_2
      Executor [0]:   Function[ 68  ]: Type: AveragePooling       Name: SqueezeNet/AveragePooling
      Executor [0]:   Function[ 69  ]: Type: Reshape              Name: SqueezeNet/Reshape
      Executor [0]:   Function[ 70  ]: Type: Identity             Name: y'

If you want to get network without Image Augmentation, according to above output, ImageAugmentation is placed on index 2.
With splitting after index 3, you can get network without ImageAugmentation.
You must specify `-E0 -S 3-` option to `nnabla_cli convert`
This command rename output to `XXX_S_E.nnp`, XXX is original name, S is start function index, and E is end function index.

.. code-block:: none

    $ nnabla_cli convert -E0 -S 3- squeezenet11.files/SqueezeNet-1.1/*.{nntxt,h5} splitted.nnp
    2018-08-27 15:20:21,950 [nnabla][INFO]: Initializing CPU extension...
    Importing squeezenet11.files/SqueezeNet-1.1/net.nntxt
    Importing squeezenet11.files/SqueezeNet-1.1/parameters.h5
     Try to leave only executor[Executor].
     Expanding Runtime.
       Shrink 3 to 70.
        Output to [splitted_3_70.nnp]


Finally you got `splitted_3_70.nnp` as splitted output.
You can check splitted NNP with `nnabla_cli dump`

NOTE: Input shape is changed from original network. New input shape is same as start function's input.

.. code-block:: none

    $ nnabla_cli dump splitted_3_70.nnp
    2018-08-27 15:20:28,021 [nnabla][INFO]: Initializing CPU extension...
    Importing splitted_3_70.nnp
     Expanding Runtime.
      Executor [0]: Executor
      Executor [0]:  (In) Data      variable[0]: Name:SqueezeNet/MulScalar           Shape:[-1, 3, 227, 227]
      Executor [0]:  (Out)Output    variable[0]: Name:y'                             Shape:[-1, 1000]
    
Done.
