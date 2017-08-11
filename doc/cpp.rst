C++ API
=======

The C++ API documentation is under construction. Coming soon!


Execute inference with C++
--------------------------

Here is utility class to convert from NNP file (or individual protobuf or hdf5 files) to CgVariable and CgFunction.

NNP file format is described in :doc:`format`.

.. code-block:: c++

    ///
    /// nnp
    ///
    /// Load network and parameter from supported format to
    /// std::vector<nbla::CgVariablePtr>.
    ///
    class nnp {
    protected:
      ///
      /// Internal information placeholder.
      ///
      _proto_internal *_proto;
    
    public:
      nnp(nbla::Context &ctx);
      ~nnp();
    
      void set_batch_size(int batch_size);
      int get_batch_size();
    
      bool add(std::string filename);
    
      int num_of_executors();
      std::vector<std::string> get_executor_input_names(int index);
      std::vector<nbla::CgVariablePtr> get_executor_input_variables(int index);
      std::vector<nbla::CgVariablePtr>
      get_executor(int index, std::vector<nbla::CgVariablePtr> inputs);
    };
    };
    };

Instruction to use nnp
----------------------

Inference
^^^^^^^^^


Save in python learning script.
"""""""""""""""""""""""""""""""

Here is code snippet of saving network for inference.

Input variables are following.

vpred
  Python Variable class for inference.
nnp_file
  filename to save (e.g. 'mnist_recog.nnp').

.. code-block:: python

   runtime_contents = {
        'networks': [
            {'name': 'Validation',
             'batch_size': args.batch_size,
             'variable': vpred}],
        'executors': [
            {'name': 'Runtime',
             'network': 'Validation',
             'variables': ['x', 'y']}]}

    save.save(nnp_file, runtime_contents)


Prepare instance of nnp class
"""""""""""""""""""""""""""""

You can use nnp class with default cpu context is very simple.

.. code-block:: c++

   nbla::Context ctx; // ("cpu", "CpuArray", "0", "default");
   nbla_utils::NNP::nnp nnp(ctx);


Then use 'add' method to adding network information from file.

filename
  filename. (std::string)
batch_size
  overrides batch_size defined in nnp file.  (int)

.. code-block:: c++

   nnp.add(filename);
   nnp.set_batch_size(batch_size);

   
Prepare input data
""""""""""""""""""

You can check how many executors in prepared nnp class.

.. code-block:: c++

   int nnp_num = -1;
   int n = nnp.num_of_executors();
   if( n > 0 ) {
     nnp_num = 0;
   }

If n > 0, you have executor in nnp, following example are using first executor in nnp.

And you can prepare input data as following code.

INPUTDATA
  pseude code that means INPUT data.

.. code-block:: c++

   std::vector<std::string> names = nnp.get_executor_input_names(nnp_num);
   std::vector<nbla::CgVariablePtr> inputs = nnp.get_executor_input_variables(nnp_num);
   for (int i = 0; i < inputs.size(); i++) {
     float *data = var->cast_data_and_get_pointer<float>(ctx);
     memcpy(data, INPUTDATA, var.get()->size() * sizeof(float));
   }


Exec inference
""""""""""""""

Then you can get CgVariablePtr (C++ network graph) with 'get_exetutor' method.

And, you can execute inference with 'forward' method of first member of the cpp graph.

.. code-block:: c++

   std::vector<nbla::CgVariablePtr> e = nnp.get_executor(nnp_num, inputs);
   e[0]->forward(true,   // clear_buffer
                 false); // clear_no_need_grad

Get result data
"""""""""""""""

You can get

.. code-block:: c++

   auto var = e[0]->variable();
   float *data = var->cast_data_and_get_pointer<float>(ctx);

Utility
-------

You can find 'nbla' command line utility found at nnabla install directory.
(In the near future, it must be install to same directory as 'nnabla_cli'.)

.. code-block:: bash

    $ /opt/miniconda3/envs/tmpenv/lib/python3.6/site-packages/nnabla/nbla
    Usage: /opt/miniconda3/envs/tmpenv/lib/python3.6/site-packages/nnabla/nbla (infer|dump)
        /opt/miniconda3/envs/tmpenv/lib/python3.6/site-packages/nnabla/nbla infer [-b BATCHSIZE] [-e EXECUTOR] input_files ...
                   input_file must be one of followings.
                       *.nnp      : Network structure and parameter.
                       *.nntxt    : Network structure in prototxt format.
                       *.prototxt : Same as nntxt.
                       *.h5       : Parameters in h5 format.
                       *.protobuf : Network structure and parameters in binary.
                       *.bin      : Input data.
        /opt/miniconda3/envs/tmpenv/lib/python3.6/site-packages/nnabla/nbla dump input_files ...
                   input_file must be nnp, nntxt, prototxt, h5, protobuf.
