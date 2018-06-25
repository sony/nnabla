Data Format
===========

Here is data format for exchange network structures and trained parameters.


Network Structure
-----------------

Network structure and parameter will store with Google Protocol Buffer format internally.


Overview
^^^^^^^^

Overview of network structure defined as following.

.. uml::

   skinparam monochrome true
   hide circle
   hide methods

   class NNablaProtoBuf {
     string version
     GlobalConfig global_config
     TrainingConfig training_config
     Network[] network
     Parameter[] parameter
     Dataset[] dataset
     Optimizer[] optimizer
     Monitor[] monitor
     Executor[] executor
   }

   package common <<Rectangle>> {
     class GlobalConfig {
       Context default_context
     }
  
     class Network {
       string name
       int batch_size
       RepeatInfo[] repeat_info
       Variable[] variable
       Function[] function
     }
  
     class Parameter {
       string variable_name
       Shape shape
       float[] data
       bool need_grad
     }
   }

   package training <<Rectangle>> {
     class TrainingConfig {
       int max_epoch
       int iter_per_epoch
       bool save_best
     }
  
     class Dataset {
       string name
       string type
     
       string uri
       int batch_size
       string cache_dir
       bool overwrite_cache
       bool create_cache_explicitly
     
       bool shuffle
       bool no_image_normalization
     
       string[] variable
     }
  
     class Optimizer {
       string name
     
       int order
     
       string network_name
       string dataset_name
     
       Solver solver
       int update_interval
     
       DataVariable[] data_variable
       GeneratorVariable[] generator_variable
       LossVariable[] loss_variable
       ParameterVariable[] parameter_variable
     }
     
     class Monitor {
       string name
     
       string network_name
       string dataset_name
     
       DataVariable[] data_variable
       GeneratorVariable[] generator_variable
       MonitorVariable[] monitor_variable
     }
   }   

   package inference <<Rectangle>> {
     class Executor {
       string name
     
       string network_name
     
       int num_evaluations
       string repeat_evaluation_type
     
       bool need_back_propagation
     
       DataVariable[] data_variable
       GeneratorVariable[] generator_variable
       LossVariable[] loss_variable
       OutputVariable[] output_variable
       ParameterVariable[] parameter_variable
     }
   }
   common <.. training
   common <.. inference
   
   NNablaProtoBuf "1" o-- "0,1" GlobalConfig
   NNablaProtoBuf "1" o-- "0,1" Parameter

   NNablaProtoBuf "1" o-- "0,1" TrainingConfig
   NNablaProtoBuf "1" o-- "0..*" Network
   NNablaProtoBuf "1" o-- "0..*" Dataset
   NNablaProtoBuf "1" o-- "0..*" Optimizer
   NNablaProtoBuf "1" o-- "0..*" Monitor

   NNablaProtoBuf "1" o-- "0..*" Executor


NNablaProtoBuf
  Root message of NNabla network structure.
  This message could be store GlobalConfig, TrainingConfig, Network(s), Parameter(s), Dataset(s), Optimizer(s), Monitor(s) and Executor(s).
Variable
  Internal data structure to store tensor for Neural network I/O and parameters.
GlobalConfig
  Configuration of environment that suggest to do train or inference.
TrainingConfig
  Configuration of training.
Network
  Network structure.
Parameter
  Special variable to store train result. (e.g Weight or Bias of affine layer)
Dataset
  Specify dataset for training.
Optimizer
  Define network, dataset,  and input/output variables for train.
Monitor
  Define network, dataset, and input/output variables for monitor training status..
Executor
  Define network and input/output variables for train.


Structure for Training
""""""""""""""""""""""

TBD

Structure for Inference
"""""""""""""""""""""""

TBD
  
Overall structure
^^^^^^^^^^^^^^^^^

.. uml::

   skinparam monochrome true
   hide circle
   hide methods

   class Shape {
     int[] dim
   }

   class Context {
     string backend
     string array_class
     string device_id
     string compute_backend
   }

   class GlobalConfig {
     Context default_context
   }

   class NNablaProtoBuf {
     string version
     GlobalConfig global_config
     TrainingConfig training_config
     Network[] network
     Parameter[] parameter
     Dataset[] dataset
     Optimizer[] optimizer
     Monitor[] monitor
     Executor[] executor
   }

   class TrainingConfig {
     int max_epoch
     int iter_per_epoch
     bool save_best
   }

   class Network {
     string name
     int batch_size
     RepeatInfo[] repeat_info
     Variable[] variable
     Function[] function
   }

   class RepeatInfo {
    string id
    int times
   }

   class RepeatParameter {
     string repeat_id
     int times
   }

   class RecurrentParameter {
     string repeat_id
     int length
     int axis
   }
   
   class Variable {
     string name
     string type
     string[] repeat_id
   
     Shape shape
   
     Initializer initializer
   }
   
   class Initializer {
     string type
     float multiplier
   }

   class Parameter {
     string variable_name
     Shape shape
     float[] data
     bool need_grad
   }
   
   class Dataset {
     string name
     string type
   
     string uri
     int batch_size
     string cache_dir
     bool overwrite_cache
     bool create_cache_explicitly
   
     bool shuffle
     bool no_image_normalization
   
     string[] variable
   }

   class Optimizer {
     string name
   
     int order
   
     string network_name
     string dataset_name
   
     Solver solver
     int update_interval
   
     DataVariable[] data_variable
     GeneratorVariable[] generator_variable
     LossVariable[] loss_variable
     ParameterVariable[] parameter_variable
   }
   
   class Solver {
     string type
   
     Context context
   
     float weight_decay
   
     float lr_decay
     int lr_decay_interval
   
     SolverParameter parameter
   }
   
   class DataVariable {
     string variable_name
     string data_name
   }
   
   class GeneratorVariable {
     string variable_name
     string type
     float multiplier
   }
   
   class LossVariable {
     string variable_name
   }
   
   class ParameterVariable {
     string variable_name
     float learning_rate_multiplier
   }
   
   class Monitor {
     string name
   
     string network_name
     string dataset_name
   
     DataVariable[] data_variable
     GeneratorVariable[] generator_variable
     MonitorVariable[] monitor_variable
   }
   
   class MonitorVariable {
     string variable_name
     string type
     string data_name
   
     float multiplier
   }
   
   class Executor {
     string name
   
     string network_name
   
     int num_evaluations
     string repeat_evaluation_type
   
     bool need_back_propagation
   
     DataVariable[] data_variable
     GeneratorVariable[] generator_variable
     LossVariable[] loss_variable
     OutputVariable[] output_variable
     ParameterVariable[] parameter_variable
   }
   
   class OutputVariable {
     string variable_name
     string type
     string data_name
   }
   
   class Function {
     string name
     string type
     string[] repeat_id
   
     Context context
     string[] input
     string[] output
   
     FunctionParameter parameter
   
     // Loop Functions
     RepeatParameter repeat_param
     RecurrentParameter recurrent_param
   }

   abstract class SolverParameter
   hide SolverParameter members
   
   abstract class FunctionParameter
   hide FunctionParameter members
   
   NNablaProtoBuf "1" o-- "0,1" GlobalConfig
   NNablaProtoBuf "1" o-- "0,1" TrainingConfig
   NNablaProtoBuf "1" o-- "0..*" Network
   NNablaProtoBuf "1" o-- "0..*" Parameter
   NNablaProtoBuf "1" o-- "0..*" Dataset

   NNablaProtoBuf "1" o-- "0..*" Optimizer
   NNablaProtoBuf "1" o-- "0..*" Monitor
   NNablaProtoBuf "1" o-- "0..*" Executor

   GlobalConfig "1" o-- "1" Context

   Network "1" o-- "0..*" RepeatInfo
   Network "1" o-- "0..*" Variable
   Network "1" o-- "0..*" Function

   Parameter "1" ..> "1" Variable
   Parameter "1" o-- "1" Shape

   Variable "1" o-- "1" Shape
   Variable "1" o-- "0,1" Initializer
   
   Optimizer "1" ..> "1" Network
   Optimizer "1" ..> "1" Dataset
   Optimizer "1" o-- "1" Solver
   Optimizer "1" o-- "0..*" DataVariable
   Optimizer "1" o-- "0..*" GeneratorVariable
   Optimizer "1" o-- "0..*" LossVariable
   Optimizer "1" o-- "0..*" ParameterVariable
   
   Monitor "1" ..> "1" Network
   Monitor "1" ..> "1" Dataset
   Monitor "1" o-- "1" Solver
   Monitor "1" o-- "0..*" DataVariable
   Monitor "1" o-- "0..*" GeneratorVariable
   Monitor "1" o-- "0..*" MonitorVariable
   
   Executor "1" ..> "1" Network
   Executor "1" o-- "1" Solver
   Executor "1" o-- "0..*" DataVariable
   Executor "1" o-- "0..*" GeneratorVariable
   Executor "1" o-- "0..*" LossVariable
   Executor "1" o-- "0..*" OutputVariable
   Executor "1" o-- "0..*" ParameterVariable
   
   DataVariable      "1" ..> "1" Variable
   GeneratorVariable "1" ..> "1" Variable
   LossVariable      "1" ..> "1" Variable
   ParameterVariable "1" ..> "1" Variable
   MonitorVariable   "1" ..> "1" Variable
   OutputVariable    "1" ..> "1" Variable

   Function "1" o-- "0,1" FunctionParameter
   Function "1" o-- "0,1" RepeatParameter
   Function "1" o-- "0,1" RecurrentParameter

   Solver "1" o-- "1" Context
   Solver "1" o-- "0,1" SolverParameter

Parameter
---------

From the performance point of view, parameters can be saved in HDF 5 format.

File Format and extensions
--------------------------
   
Protocol buffer text format file
  .nntxt or .prototxt
Protocol buffer serialized binary file
  .protobuf
HDF5
  .h5
NNP (ZIP archived file with above formats.)
  .nnp

