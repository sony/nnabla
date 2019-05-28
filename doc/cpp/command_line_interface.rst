C++ Command Line Interface
=========================

Nnabla has c++ version's command line interface utility which can do train, forward(inference).
Using this command line interface, developers can run train and infer without any python environment.


.. code-block:: none

    usage: nbla (infer|dump|train)


Basic functions
~~~~~~~~~~~~~~~

Forward
--------

.. code-block:: none

    usage: nbla infer -e EXECUTOR [-b BATCHSIZE] [-o OUTPUT] input_files ...

    arguments:
       -e EXECUTOR         EXECUTOR is the name of executor network.
       input_files         input_file must be one of followings.
                               *.nnp      : Network structure and parameter.
                               *.nntxt    : Network structure in prototxt format.
                               *.prototxt : Same as nntxt.
                               *.h5       : Parameters in h5 format.
                               *.protobuf : Network structure and parameters in binary.
                               *.bin      : Input data.

    optional arguments:
       -b BATCHSIZE        batch size for the input data.
       -o OUTPUT           the filename pattern of output file, default output to stdout.

    example:
        Infer using LeNet_input.bin as input, LeNet_output_0.bin as output:
           nbla infer -e Executor -b 1 LeNet.nnp LeNet_input.bin -o LeNet_output

        Infer and output the result to console:
           nbla infer -e Executor -b 1 LeNet.nnp LeNet_input.bin


Dump
-------

.. code-block:: none

    usage: nbla dump input_files ...

    arguments:
       input_files         input_files must be one of *.nnp, *.nntxt, prototxt, h5, protobuf

    example:
        Show network information by dump command:
          nbla dump LeNet.nnp

The output looks like:

.. code-block:: none

    This configuration has 1 executors.

      Executor No.0 Name [Executor]
        Using default batch size 64 .
         Inputs
          Input No.0 Name [x] Shape ( 64 1 28 28 )
         Outputs
          Output No.0 Name [y'] Shape ( 64 10 )
    Finished


Train
-----

.. code-block:: none

    usage: nbla train input_file

    arguments:
       input_file          input_file must be *.nnp