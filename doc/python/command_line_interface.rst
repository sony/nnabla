Python Command Line Interface
=============================

Nnabla has commandline interface utility whitch can do train, forward(inference),
convert param and dataset, measure performance, file format converter and so on.

.. code-block:: none

    usage: nnabla_cli [-h]
                      {train,infer,forward,encode_param,decode_param,profile,conv_dataset,compare_with_cpu,create_image_classification_dataset,upload,create_tar,function_info,dump,nnb_template,convert,version}
                      ...
    
    Command line interface for NNabla(Version 1.0.0rc2, Build 180626044347)
    
    positional arguments:
      {train,infer,forward,encode_param,decode_param,profile,conv_dataset,compare_with_cpu,create_image_classification_dataset,upload,create_tar,function_info,dump,nnb_template,convert,version}
        train               Training with NNP.
        infer               Do inference with NNP and binary data file input.
        forward             Do evaluation with NNP and test dataset.
        encode_param        Encode plain text to parameter format.
        decode_param        Decode parameter to plain text.
        profile             Profiling performance with NNP.
        conv_dataset        Convert CSV dataset to cache.
        compare_with_cpu    Compare performance between two nntxt.
        create_image_classification_dataset
                            Create dataset from image files.
        upload              Upload dataset to Neural Network Console.
        create_tar          Create tar file for Neural Network COnsole.
        function_info       Output function info.
        dump                Dump network with supported format.
        nnb_template        Generate NNB config file template.
        convert             File format converter.
        version             Print version and build number.
    
    optional arguments:
      -h, --help            show this help message and exit


Work with NNP
~~~~~~~~~~~~~

Training
--------

.. code-block:: none

    usage: nnabla_cli train [-h] -c CONFIG [-p PARAM] -o OUTDIR
    
    optional arguments:
      -h, --help            show this help message and exit
      -c CONFIG, --config CONFIG
                            path to nntxt
      -p PARAM, --param PARAM
                            path to parameter file
      -o OUTDIR, --outdir OUTDIR
                            output directory

Profile
-------

.. code-block:: none

    usage: nnabla_cli profile [-h] -c CONFIG -o OUTDIR
    
    optional arguments:
      -h, --help            show this help message and exit
      -c CONFIG, --config CONFIG
                            path to nntxt
      -o OUTDIR, --outdir OUTDIR
                            output directory


Forward
-------

.. code-block:: none

    usage: nnabla_cli forward [-h] -c CONFIG [-p PARAM] [-d DATASET] -o OUTDIR [-b BATCH_SIZE]
    
    optional arguments:
      -h, --help            show this help message and exit
      -c CONFIG, --config CONFIG
                            path to nntxt
      -p PARAM, --param PARAM
                            path to parameter file
      -d DATASET, --dataset DATASET
                            path to CSV dataset
      -o OUTDIR, --outdir OUTDIR
                            output directory
      -b BATCH_SIZE, --batch_size BATCH_SIZE
                            Batch size to use batch size in nnp file set -1.


Inference
---------

.. code-block:: none

    usage: nnabla_cli infer [-h] -c CONFIG [-o OUTPUT] [-p PARAM] [-b BATCH_SIZE] inputs [inputs ...]
    
    positional arguments:
      inputs
    
    optional arguments:
      -h, --help            show this help message and exit
      -c CONFIG, --config CONFIG
                            path to nntxt
      -o OUTPUT, --output OUTPUT
                            output file prefix
      -p PARAM, --param PARAM
                            path to parameter file
      -b BATCH_SIZE, --batch_size BATCH_SIZE
                            Batch size to use batch size in nnp file set -1.


Compare with CPU
----------------

.. code-block:: none

    usage: nnabla_cli compare_with_cpu [-h] -c CONFIG -c2 CONFIG2 -o OUTDIR
    
    optional arguments:
      -h, --help            show this help message and exit
      -c CONFIG, --config CONFIG
                            path to nntxt
      -c2 CONFIG2, --config2 CONFIG2
                            path to cpu nntxt
      -o OUTDIR, --outdir OUTDIR
                            output directory


Dataset manipulation
~~~~~~~~~~~~~~~~~~~~

Encode parameter
----------------

.. code-block:: none

    usage: nnabla_cli encode_param [-h] -i INDIR [-p PARAM]
    
    optional arguments:
      -h, --help            show this help message and exit
      -i INDIR, --indir INDIR
                            input directory
      -p PARAM, --param PARAM
                            path to parameter file


Decode parameter
----------------

.. code-block:: none

    usage: nnabla_cli decode_param [-h] [-p PARAM] -o OUTDIR
    
    optional arguments:
      -h, --help            show this help message and exit
      -p PARAM, --param PARAM
                            path to parameter file
      -o OUTDIR, --outdir OUTDIR
                            output directory


Convert dataset
---------------

.. code-block:: none

    usage: nnabla_cli conv_dataset [-h] [-F] [-S] [-N] source destination
    
    positional arguments:
      source
      destination
    
    optional arguments:
      -h, --help       show this help message and exit
      -F, --force      force overwrite destination
      -S, --shuffle    shuffle data
      -N, --normalize  normalize data range


Create image classification dataset
-----------------------------------


.. code-block:: none

    usage: nnabla_cli create_image_classification_dataset [-h] -i SOURCEDIR -o OUTDIR -c CHANNEL -w WIDTH -g HEIGHT -m MODE -s SHUFFLE -f1 FILE1 [-r1 RATIO1] [-f2 FILE2]
                                                          [-r2 RATIO2]
    
    optional arguments:
      -h, --help            show this help message and exit
      -i SOURCEDIR, --sourcedir SOURCEDIR
                            source directory with directories for each class
      -o OUTDIR, --outdir OUTDIR
                            output directory
      -c CHANNEL, --channel CHANNEL
                            number of output color channels
      -w WIDTH, --width WIDTH
                            width of output image
      -g HEIGHT, --height HEIGHT
                            height of output image
      -m MODE, --mode MODE  shaping mode (trimming or padding)
      -s SHUFFLE, --shuffle SHUFFLE
                            shuffle mode (true or false)
      -f1 FILE1, --file1 FILE1
                            output file name 1
      -r1 RATIO1, --ratio1 RATIO1
                            output file ratio(%) 1
      -f2 FILE2, --file2 FILE2
                            output file name 2
      -r2 RATIO2, --ratio2 RATIO2
                            output file ratio(%) 2

Upload dataset to Neural Network Console
----------------------------------------

.. code-block:: none

    usage: nnabla_cli upload [-h] [-e ENDPOINT] token filename
    
    positional arguments:
      token                 token for upload
      filename              filename to upload
    
    optional arguments:
      -h, --help            show this help message and exit
      -e ENDPOINT, --endpoint ENDPOINT
                            set endpoint uri

Create dataset archive for Neural Network Console
-------------------------------------------------


.. code-block:: none

    usage: nnabla_cli create_tar [-h] source destination
    
    positional arguments:
      source       CSV dataset
      destination  TAR filename
    
    optional arguments:
      -h, --help   show this help message and exit


File format converter
~~~~~~~~~~~~~~~~~~~~~


For detailed infomation please see :any:`file_format_converter/file_format_converter`.

Dump content of supported format
--------------------------------

.. code-block:: none

    usage: nnabla_cli dump [-h] [-I IMPORT_FORMAT] [--nnp-no-expand-network]
                           FILE [FILE ...]
    
    positional arguments:
      FILE                  File or directory name(s) to convert.
    
    optional arguments:
      -h, --help            show this help message and exit
      -I IMPORT_FORMAT, --import-format IMPORT_FORMAT
                            [import] import format. (one of [NNP,ONNX])
      --nnp-no-expand-network
                            [import][NNP] expand network with repeat or recurrent.


Generate NNB config file template
---------------------------------

.. code-block:: none

    usage: nnabla_cli nnb_template [-h] [-I IMPORT_FORMAT]
                                   [--nnp-no-expand-network] [-b BATCH_SIZE]
                                   [-T DEFAULT_VARIABLE_TYPE]
                                   FILE [FILE ...]
    
    positional arguments:
      FILE                  File or directory name(s) to convert.
    
    optional arguments:
      -h, --help            show this help message and exit
      -I IMPORT_FORMAT, --import-format IMPORT_FORMAT
                            [import] import format. (one of [NNP,ONNX])
      --nnp-no-expand-network
                            [import][NNP] expand network with repeat or recurrent.
      -b BATCH_SIZE, --batch-size BATCH_SIZE
                            [export] overwrite batch size.
      -T DEFAULT_VARIABLE_TYPE, --default-variable-type DEFAULT_VARIABLE_TYPE
                            Default type of variable

File format converter
---------------------

.. code-block:: none

    usage: nnabla_cli convert [-h] [-I IMPORT_FORMAT] [--nnp-no-expand-network]
                              [-O EXPORT_FORMAT] [-f] [-b BATCH_SIZE]
                              [--nnp-parameter-h5] [--nnp-parameter-nntxt]
                              [--nnp-exclude-parameter] [-T DEFAULT_VARIABLE_TYPE]
                              [-s SETTINGS]
                              FILE [FILE ...]
    
    positional arguments:
      FILE                  File or directory name(s) to convert.
    
    optional arguments:
      -h, --help            show this help message and exit
      -I IMPORT_FORMAT, --import-format IMPORT_FORMAT
                            [import] import format. (one of [NNP,ONNX])
      --nnp-no-expand-network
                            [import][NNP] expand network with repeat or recurrent.
      -O EXPORT_FORMAT, --export-format EXPORT_FORMAT
                            [export] export format. (one of [NNP,NNB,CSRC,ONNX])
      -f, --force           [export] overwrite output file.
      -b BATCH_SIZE, --batch-size BATCH_SIZE
                            [export] overwrite batch size.
      --nnp-parameter-h5    [export][NNP] store parameter with h5 format
      --nnp-parameter-nntxt
                            [export][NNP] store parameter into nntxt
      --nnp-exclude-parameter
                            [export][NNP] output without parameter
      -T DEFAULT_VARIABLE_TYPE, --default-variable-type DEFAULT_VARIABLE_TYPE
                            Default type of variable
      -s SETTINGS, --settings SETTINGS
                            Settings in YAML format file.


Development
~~~~~~~~~~~

Generate function infomation
----------------------------

.. code-block:: none

    usage: nnabla_cli function_info [-h] [dest]
    
    positional arguments:
      dest        destination filename
    
    optional arguments:
      -h, --help  show this help message and exit

Display version
---------------

.. code-block:: none

    usage: nnabla_cli version [-h]
    
    optional arguments:
      -h, --help  show this help message and exit

