'''
Data exchange format for "Neural Network Library".

Current version of .nnp file is just a ZIP format archive file but filename extension is '.nnp'.

'.nnp' file will contain following files. If '.nnp' file contain other file, nnabla will just ignore that files.

* 'nnp_version.txt'

  * Specify version of nnp file. Version string in this file is got from nnp_version().
* '*.nntxt' (or '*.prototxt')

  * Network structure in Protocol buffer text format.
* '*.protobuf'

  * Trained parameter in Protocol buffer binary format.
* '*.h5'

  * Trained parameter in HDF5 format.(Will be obsolete soon.)



'''


def nnp_version():
    '''nnp_version

    Current version is "0.1"

    * Version history

      * Version 0.1

        * First version.

    '''
    return '0.1'
