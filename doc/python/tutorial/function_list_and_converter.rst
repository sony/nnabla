
Function list and converter
=============================

``nnabla_cli`` is the command line interface of nnabla. With this command line interface, user may know current NNabla support status, and know whether or how to convert a nnabla model(e.g. *.nnp)
to other format of model(e.g. *.onnx).

The subcommand ``function_info`` provides a set of functions to output implemented function information.
With this information, you may build tailored nnabla-c-runtime library for your model, or skip some unsupported
functions for the target model.


Some simple use cases
~~~~~~~~~~~~~~~~~~~~~

Please let us introduce some simple use cases:

At first, you want to know how many functions (which functions) nnabla currently supports:

.. code-block:: none

    $ nnabla_cli function_info

You get the following list:

.. parsed-literal::

    2019-06-14 16:16:13,106 [nnabla][INFO]: Initializing CPU extension...
    NNabla command line interface (Version:1.0.18.dev1, Build:190531084842)
    LSTM
    Sub2
    Mul2
    GreaterEqual
    Sigmoid
    NotEqual
    Unpooling
    Log
    CategoricalCrossEntropy
    ...

That is the list of current nnabla all supported functions. Only function names are
shown, no more detail, only for seeking certain function by name. For the detail of each function, you have to check with online document.


As you known, nnabla's model *.nnp can be converted to a compact version, it has the postfix ``.nnb``, can be inferred by nnabla-c-runtime library. We simply named this format as ``NNB``. To know how many functions are supported in this format, you may use this command:

.. code-block:: none

    $ nnabla_cli function_info -f NNB

Similar as above, a function list is shown.


Do we simple list the functions used in a .nnp model? Yes, of course.

.. code-block:: none

    $ nnabla_cli function_info my_model.nnp

Similar as above, a function list used in this model is listed.


Then, we may know whether our model can be converted to nnabla-c-runtime model format,
or formally speaking, we can know the intersection of 2 function sets, one is the function set in .nnp and the other is nnabla-c-runtime has supported.

.. code-block:: none

    $ nnabla_cli function_info my_model.nnp -f NNB


The output looks like:

.. parsed-literal::

    2019-06-14 17:01:29,393 [nnabla][INFO]: Initializing CPU extension...
    NNabla command line interface (Version:1.0.18.dev1, Build:190531084842)
    Importing mnist_nnp/lenet_010000.nnp
     Expanding runtime.
    nnabla-c-runtime currently support the following functions in model:
    Convolution
    MulScalar
    Affine
    MaxPooling
    ReLU
    ...

Unsupported functions are also listed up if there are any in this model.


Tailored nnabla-c-runtime library
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

When implementing nnabla-c-runtime library, we hope to implement all functions
we can. But from customer's aspect, that is sometimes no need. If user only wants to use nnabla-c-runtime for enumerable models, the nnabla-c-runtime should be tailed exactly as what these models required. How to do then?

It can be implemented with the following steps:

1. generate function list
2. config your nnabla-c-runtime library
3. build nnabla-c-runtime library


1. Generate function list
-------------------------

.. code-block:: none

    $ nnabla_cli function_info my_model.nnp -f NNB -o functions.txt

This is similar as above, except that with ``-o`` parameter, which pointed out which file should be written to. (of course, the format is different from the version output to stdout, it is more compact)


2. config your nnabla-c-runtime library
---------------------------------------

User may manually modify ``functions.txt``. Then, this file is used as input, used to generate nnabla-c-runtime library's config file:

.. code-block:: none

    $ nnabla_cli function_info -c functions.txt -o nnabla-c-runtime/build-tools/code-generator/functions.yaml


As we inferred, if there is no ``-c`` parameter, a full function set will be used to generate this config file, of course, the library will finally contain all implemented functions. This is the default behavior.


3. build nnabla-c-runtime library
---------------------------------

The build process is relatively directly, as the following:

.. code-block:: none

    #> nnabla-c-runtime>mkdir build
    #> nnabla-c-runtime>cd build
    #> nnabla-c-runtime>cmake ..
    #> nnabla-c-runtime>make

The nnabla-c-runtime library ``libnnablart_functions.a`` will contain the functions what you want.



Skip functions unsupported
~~~~~~~~~~~~~~~~~~~~~~~~~~

When you want to convert ``*.nnp`` to ``*.onnx`` or ``*.nnb``, there are some functions are not supported in target function list. For example, you want to convert a network to nnabla-c-runtime. The network looks like:

.. parsed-literal::

    Affine
    Softmax
    Tanh
    Convolution
    MaxPooling
    ReLU

You do not want to use nnabla-c-runtime library's ``Convolution``, you want to split the network in 2 pieces at the point of ``Convolution``. 2 Steps are needed to do so:


1. comment out the function in functions.txt
2. convert the network with ``-c`` parameter


1. comment out the function in functions.txt
--------------------------------------------

.. parsed-literal::

    ...
    ;Affine
    ...


2. convert the network with ``-c`` parameter
--------------------------------------------


.. code-block:: none

    $ nnabla_cli convert -c functions.txt a.nnp b.nnb

Thus, the network is splitted into pieces, the output shows as the following:

.. parsed-literal::

    ...
    LeNet_036_0_5.nnb:
      input:
      - name: Input
        shape: (-1, 1, 28, 28)
      output:
      - name: Tanh_2
        shape: (-1, 30, 4, 4)
    LeNet_036_7_7.nnb:
      input:
      - name: Affine
        shape: (-1, 150)
      output:
      - name: ReLU_2
        shape: (-1, 150)
    LeNet_036_9_9.nnb:
      input:
      - name: Affine_2
        shape: (-1, 10)
      output:
      - name: Softmax
        shape: (-1, 10)

The network is split at the ``Affine`` function. Since there are 2 ``Affine`` in network, 3 sub-networks is generated.



Converting to ONNX
~~~~~~~~~~~~~~~~~~


The following commands just do similar as above, exactly to *.onnx.

List all functions supported:

.. code-block:: none

    $ nnabla_cli function_info -f ONNX


List the intersection of function sets, in a model and supported by ONNX:

.. code-block:: none

    $ nnabla_cli function_info LeNet_036.nnp -f ONNX


Split network to skip some function:

.. code-block:: none

    $ nnabla_cli convert -c functions.txt a.nnp a.onnx
