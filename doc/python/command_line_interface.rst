Command Line Interface
======================

Nnabla has commandline interface utility whitch can do train, forward(inference),
convert param and dataset, measure performance and so on.

.. code-block:: none

    usage: nnabla_cli [-h]
                      {train,forward,decode_param,encode_param,profile,conv_dataset,compare_with_cpu}
                      ...
    positional arguments:
      {train,forward,decode_param,encode_param,profile,conv_dataset,compare_with_cpu,auto-format}
    
    optional arguments:
      -h, --help            show this help message and exit


Training
--------

Train with saved prototxt or nnp file.
You can resume training from saved parameter.

During the training, parameter is store every 3 minutes or 10 epochs.

If training was stopped accidentaly, you can resume training from at
most 3 minuts ago.

.. code-block:: none

    usage: nnabla_cli train [-h] [-r] -c CONFIG [-p PARAM] -o OUTDIR
    
    optional arguments:
      -h, --help            show this help message and exit
      -r, --resume          resume from last saved parameter.
      -c CONFIG, --config CONFIG
                            path to nntxt
      -p PARAM, --param PARAM
                            path to parameter file
      -o OUTDIR, --outdir OUTDIR
                            output directory

Forward
-------

.. code-block:: none

    usage: nnabla_cli forward [-h] -c CONFIG [-p PARAM] [-d DATASET] -o OUTDIR
    
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

Decode Parameter
----------------

.. code-block:: none

    usage: nnabla_cli decode_param [-h] [-p PARAM] -o OUTDIR
    
    optional arguments:
      -h, --help            show this help message and exit
      -p PARAM, --param PARAM
                            path to parameter file
      -o OUTDIR, --outdir OUTDIR
                            output directory



Encode Parameter
----------------

.. code-block:: none
    
    usage: nnabla_cli encode_param [-h] -i INDIR [-p PARAM]
    
    optional arguments:
      -h, --help            show this help message and exit
      -i INDIR, --indir INDIR
                            input directory
      -p PARAM, --param PARAM
                            path to parameter file
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


Convert Dataset
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


Compare calc speed between different context
--------------------------------------------

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
                            
