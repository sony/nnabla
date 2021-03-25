Python Command Line Interface
=============================

Nnabla has command line interface utility which can do train, forward(inference),
convert param and dataset, measure performance, file format converter and so on.

.. code-block:: none

    usage: nnabla_cli [-h] [-m]
                      {train,infer,forward,encode_param,decode_param,profile,conv_dataset,compare_with_cpu,create_image_classification_dataset,upload,create_tar,function_info,optimize,dump,nnb_template,convert,plot_series,plot_timer,draw_graph,version}
                      ...
    
    Command line interface for NNabla(Version 1.0.11.dev1, Build 181226024531)
    
    positional arguments:
      {train,infer,forward,encode_param,decode_param,profile,conv_dataset,compare_with_cpu,create_image_classification_dataset,upload,create_tar,function_info,optimize,dump,nnb_template,convert,plot_series,plot_timer,draw_graph,version}
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
        create_tar          Create tar file for Neural Network Console.
        function_info       Output function info.
        optimize            Optimize pb model.
        dump                Dump network with supported format.
        nnb_template        Generate NNB config file template.
        convert             File format converter.
        plot_series         Plot *.series.txt files.
        plot_timer          Plot *.timer.txt files.
        draw_graph          Draw a graph in a NNP or nntxt file with graphviz.
        version             Print version and build number.
    
    optional arguments:
      -h, --help            show this help message and exit
      -m, --mpi             exec with mpi.


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


For detailed information please see :any:`file_format_converter/file_format_converter`.

Dump content of supported format
--------------------------------

.. code-block:: none

    usage: nnabla_cli dump [-h] [-v] [-F] [-V] [--dump-limit DUMP_LIMIT]
                           [-n DUMP_VARIABLE_NAME] [-I IMPORT_FORMAT]
                           [-E NNP_IMPORT_EXECUTOR_INDEX]
                           [--nnp-exclude-preprocess] [--nnp-no-expand-network]
                           FILE [FILE ...]
    
    positional arguments:
      FILE                  File or directory name(s) to convert.
    
    optional arguments:
      -h, --help            show this help message and exit
      -v, --dump-verbose    [dump] verbose output.
      -F, --dump-functions  [dump] dump function list.
      -V, --dump-variables  [dump] dump variable list.
      --dump-limit DUMP_LIMIT
                            [dump] limit num of items.
      -n DUMP_VARIABLE_NAME, --dump-variable-name DUMP_VARIABLE_NAME
                            [dump] Specific variable name to display.
      -I IMPORT_FORMAT, --import-format IMPORT_FORMAT
                            [import] import format. (one of [NNP,ONNX])
      -E NNP_IMPORT_EXECUTOR_INDEX, --nnp-import-executor-index NNP_IMPORT_EXECUTOR_INDEX
                            [import][NNP] import only specified executor.
      --nnp-exclude-preprocess
                            [import][NNP] EXPERIMENTAL exclude preprocess
                            functions when import.
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
                              [-s SETTINGS] [-c CONFIG] [-d DEFINE_VERSION] [--api API]
                              [--enable-optimize-pb] [--outputs OUTPUTS]
                              [--inputs INPUTS] FILE [FILE ...]
    
    positional arguments:
      FILE                  File or directory name(s) to convert.
                            (When convert ckpt format of the tensorflow model,
                            If the version of the checkpoint is V1, need to enter the `.ckpt` file,
                            otherwise need to enter the `.meta` file.)
    
    optional arguments:
      -h, --help            show this help message and exit
      -I IMPORT_FORMAT, --import-format IMPORT_FORMAT
                            [import] import format. (one of [NNP,ONNX,'TF_CKPT_V1','TF_CKPT_V2','TF_PB','SAVED_MODEL'])
      --nnp-no-expand-network
                            [import][NNP] expand network with repeat or recurrent.
      --outputs OUTPUTS
                            [import][tensorflow] The name(s) of the output nodes, comma separated.
                                                 Only needed when convert CKPT format.
      --inputs INPUTS
                            [import][tensorflow] The name(s) of the input nodes, comma separated.
                                                 Only needed when convert CKPT format.
      -O EXPORT_FORMAT, --export-format EXPORT_FORMAT
                            [export] export format. (one of [NNP,NNB,CSRC,ONNX,SAVED_MODEL,TFLITE,TF_PB],
                                     the export file format is 'CSRC' or 'SAVED_MODEL' that
                                     argument '--export-format' will have to be set!!!)
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
      -c CONFIG, --config CONFIG
                            [export] config target function list.
      -d DEFINE_VERSION, --define_version
                            [export][ONNX] define onnx opset version. e.g. opset_6
                            [export][ONNX] define convert to onnx for SNPE. e.g. opset_snpe
                            [export][ONNX] define convert to onnx for TensorRT. e.g. opset_tensorrt
                            [export][NNB] define binary format version. e.g. nnb_3
      --api API             [export][NNB] Set API Level to convert to, default is highest API Level.
      --enable-optimize-pb  [export][tensorflow] enable optimization when export to pb or tflite.

Optimize pb model
-----------------

.. code-block:: none

    usage: nnabla_cli optimize [-h] input_pb_file output_pb_file

    positional arguments:
      input_pb_file       Input pre-optimized pb model.
      output_pb_file      Output optimized pb model.


Plot Monitor class output files
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Note**:

- Plotting subcommands require matplotlib package.
- By default, the following commands show a plot on your display using a
  backend rendering engine of matplotlib depending on your environment.
  If you want to save a plot as an image or a vector data, use ``-o`` option to
  specifiy a file name where a plot is saved.

MonitorSeries
-------------

.. code-block:: none

    usage: nnabla_cli plot_series [-h] [-l LABEL] [-o OUTFILE] [-x XLABEL]
                                  [-y YLABEL] [-t TITLE] [-T YLIM_MAX]
                                  [-B YLIM_MIN] [-R XLIM_MAX] [-L XLIM_MIN]
                                  infile [infile ...]
    
    Plot *.series.txt files produced by nnabla.monitor.MonitorSeries class.
    
    Example:
    
        nnabla_cli plot_series -x "Epochs" -y "Squared error loss" -T 10 -l "config A" -l "config B" result_a/Training-loss.series.txt result_b/Training-loss.series.txt
    
    positional arguments:
      infile                Path to input file.
    
    optional arguments:
      -h, --help            show this help message and exit
      -l LABEL, --label LABEL
                            Label of each plot.
      -o OUTFILE, --outfile OUTFILE
                            Path to output file.
      -x XLABEL, --xlabel XLABEL
                            X-axis label of plot.
      -y YLABEL, --ylabel YLABEL
                            Y-axis label of plot.
      -t TITLE, --title TITLE
                            Title of plot.
      -T YLIM_MAX, --ylim-max YLIM_MAX
                            Y-axis plot range max.
      -B YLIM_MIN, --ylim-min YLIM_MIN
                            Y-axis plot range min.
      -R XLIM_MAX, --xlim-max XLIM_MAX
                            X-axis plot range max.
      -L XLIM_MIN, --xlim-min XLIM_MIN
                            X-axis plot range min.


MonitorTimeElapsed
------------------

.. code-block:: none

    usage: nnabla_cli plot_timer [-h] [-l LABEL] [-o OUTFILE] [-x XLABEL]
                                 [-y YLABEL] [-t TITLE] [-T YLIM_MAX]
                                 [-B YLIM_MIN] [-R XLIM_MAX] [-L XLIM_MIN] [-e]
                                 [-u TIME_UNIT]
                                 infile [infile ...]
    
    Plot *.timer.txt files produced by nnabla.MonitorTimeElapsed class.
    
    Example:
    
        nnabla_cli plot_timer -x "Epochs" -l "config A" -l "config B" result_a/Epoch-time.timer.txt result_b/Epoch-time.timer.txt
    
    positional arguments:
      infile                Path to input file.
    
    optional arguments:
      -h, --help            show this help message and exit
      -l LABEL, --label LABEL
                            Label of each plot.
      -o OUTFILE, --outfile OUTFILE
                            Path to output file.
      -x XLABEL, --xlabel XLABEL
                            X-axis label of plot.
      -y YLABEL, --ylabel YLABEL
                            Y-axis label of plot.
      -t TITLE, --title TITLE
                            Title of plot.
      -T YLIM_MAX, --ylim-max YLIM_MAX
                            Y-axis plot range max.
      -B YLIM_MIN, --ylim-min YLIM_MIN
                            Y-axis plot range min.
      -R XLIM_MAX, --xlim-max XLIM_MAX
                            X-axis plot range max.
      -L XLIM_MIN, --xlim-min XLIM_MIN
                            X-axis plot range min.
      -e, --elapsed         Plot total elapsed time. By default, it plots elapsed time per iteration.
      -u TIME_UNIT, --time-unit TIME_UNIT
                            Time unit chosen from {s|m|h|d}.

Draw a graph from NNP or .nntxt files
-------------------------------------

**Note**:

- This feature requires ``graphviz`` installed as a `Python package <https://graphviz.readthedocs.io/en/stable/manual.html#installation>`_. The ``graphviz`` Python is a interface to `graphviz library <https://www.graphviz.org/>`_ which is not installed by ``pip`` command. You have to install it using ``apt`` on Ubuntu for example.


.. code-block:: none

    usage: nnabla_cli draw_graph [-h] [-o OUTPUT_DIR] [-n NETWORK] [-f FORMAT]
                                 input

    Draw a graph in a NNP or nntxt file with graphviz.

    Example:

        nnabla_cli draw_graph -o output-folder path-to-nnp.nnp

    positional arguments:
      input                 Path to input nnp or nntxt.

    optional arguments:
      -h, --help            show this help message and exit
      -o OUTPUT_DIR, --output-dir OUTPUT_DIR
                            Output directory.
      -n NETWORK, --network NETWORK
                            Network names to be drawn.
      -f FORMAT, --format FORMAT
                            Graph saving format compatible with graphviz (`pdf`, `png`, ...).


Development
~~~~~~~~~~~

Generate function information
----------------------------

.. code-block:: none

    usage: nnabla_cli function_info [-h] [-o OUTFILE] [-f FUNC_SET] [-c CONFIG]
                                    [-t TARGET] [-q --query] [--nnp-no-expand-network]
                                    [--api API] [FILE] [FILE ...]

    positional arguments:
      FILE                  Path to nnp file.

    optional arguments:
      -h, --help  show this help message and exit
      -o OUTFILE, --output OUTFILE
                          output filename, *.txt or *.yaml, the default is stdout.
      -f FUNC_SET, --all_support FUNC_SET
                          select function set: NNB, ONNX, the default is nnabla.
      -c CONFIG, --config CONFIG
                          user config file for target constraint, *.txt file of the
                          function list or the "opset_" args.
      -t, --target
                          output target function list.
      -q, --query
                          query the detail of a function.
      --nnp-no-expand-network
                          [import][NNP] expand network with repeat or recurrent.
      --api API           List up api levels

Display version
---------------

.. code-block:: none

    usage: nnabla_cli version [-h]
    
    optional arguments:
      -h, --help  show this help message and exit

