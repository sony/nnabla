=============================
Function-Level Support Status
=============================

.. contents::
   :local:
   :depth: 3

ONNX Support Status
===================

:Note: In this document, the numbers in the header of all tables represent the version of onnx opset.
:ONNX Version:
  1.6.0


Import
------

- ✓: onnx specification defined, and supported.
- X: onnx specification defined, but not support yet.
- Empty: Not defined (Support status follows latest).


Total: 93/155

.. table:: 

    ===========================  ===  ===  ===  ===  ===  ===  ===  ===  ===  ====  ====  ===============================================================  ==============================================================================================================================================================================================================================================================
           ONNX Operator          1    2    3    4    5    6    7    8    9    10    11                             NNabla Func                                                                                                                                                     Description                                                                                                                          
    ===========================  ===  ===  ===  ===  ===  ===  ===  ===  ===  ====  ====  ===============================================================  ==============================================================================================================================================================================================================================================================
     Abs                         ✓                        ✓                               Abs                                                                                                                                                                                                                                                                                                                            
     Acos                                                      ✓                          ACos                                                                                                                                                                                                                                                                                                                           
     Acosh                                                               ✓                ACosh                                                                                                                                                                                                                                                                                                                          
     Add                         ✓                        ✓    ✓                          Add2, Reshape                                                                                                                                                                                                                                                                                                                  
     And                         ✓                        ✓    ✓                          LogicalAnd, Reshape                                                                                                                                                                                                                                                                                                            
     ArgMax                      ✓                        ✓                   X     ✓     Max                                                                                                                                                                                                                                                                                                                            
     ArgMin                      ✓                        ✓                   X     ✓     Min                                                                                                                                                                                                                                                                                                                            
     Asin                                                      ✓                          ASin                                                                                                                                                                                                                                                                                                                           
     Asinh                                                               ✓                ASinh                                                                                                                                                                                                                                                                                                                          
     Atan                                                      ✓                          ATan                                                                                                                                                                                                                                                                                                                           
     Atanh                                                               ✓                ATanh                                                                                                                                                                                                                                                                                                                          
     AveragePool                 ✓                        ✓    ✓              X     X     Pad, AveragePooling                                              Not all features are verified, since some features are not supported by caffe2. Those features can be verified by ONNXRuntime when opset > 6. Some feature is not supported by Nnabla such as Pad's edge mode. if opset >= 10, the ceil_mode is not supported.
     BatchNormalization          X                        X    X         ✓                BatchNormalization                                                                                                                                                                                                                                                                                                             
     BitShift                                                                       X                                                                      Not yet implemented.                                                                                                                                                                                                                                          
     Cast                        ✓                        ✓              X                                                                                                                                                                                                                                                                                                                                      
     Ceil                        ✓                        ✓                               Ceil                                                                                                                                                                                                                                                                                                                           
     Clip                        ✓                        ✓                         ✓     MaximumScalar, MinimumScalar, Identity                                                                                                                                                                                                                                                                                         
     Compress                                                            X          X                                                                      Not yet implemented.                                                                                                                                                                                                                                          
     Concat                      ✓              ✓         ✓                         X     Concatenate                                                                                                                                                                                                                                                                                                                    
     ConcatFromSequence                                                             X                                                                      Not yet implemented.                                                                                                                                                                                                                                          
     Constant                    ✓                        ✓              X          X     Identity                                                                                                                                                                                                                                                                                                                       
     ConstantOfShape                                                     ✓                Constant                                                                                                                                                                                                                                                                                                                       
     Conv                        ✓                        ✓                         X     Convolution                                                                                                                                                                                                                                                                                                                    
     ConvInteger                                                              X                                                                            Not yet implemented.                                                                                                                                                                                                                                          
     ConvTranspose               ✓                        ✓                         X     Pad, Deconvolution                                                                                                                                                                                                                                                                                                             
     Cos                                                       ✓                          Cos                                                                                                                                                                                                                                                                                                                            
     Cosh                                                                ✓                Cosh                                                                                                                                                                                                                                                                                                                           
     CumSum                                                                         X                                                                      Not yet implemented.                                                                                                                                                                                                                                          
     DepthToSpace                ✓                        ✓                         ✓     Transpose, Reshape                                                                                                                                                                                                                                                                                                             
     DequantizeLinear                                                         X                                                                            Not yet implemented.                                                                                                                                                                                                                                          
     Det                                                                            X                                                                      Not yet implemented.                                                                                                                                                                                                                                          
     Div                         ✓                        ✓    ✓                          Reshape, Div2                                                                                                                                                                                                                                                                                                                  
     Dropout                     X                        X    ✓              X           Identity                                                                                                                                                                                                                                                                                                                       
     DynamicQuantizeLinear                                                          X                                                                      Not yet implemented.                                                                                                                                                                                                                                          
     Elu                         ✓                        ✓                               ELU                                                                                                                                                                                                                                                                                                                            
     Equal                       ✓                        ✓    ✓                    X     Equal, Reshape                                                                                                                                                                                                                                                                                                                 
     Erf                                                                 X                                                                                 Not yet implemented.                                                                                                                                                                                                                                          
     Exp                         ✓                        ✓                               Exp                                                                                                                                                                                                                                                                                                                            
     Expand                                                         ✓    ✓                Broadcast, Reshape                                                                                                                                                                                                                                                                                                             
     EyeLike                                                             X                                                                                 Not yet implemented.                                                                                                                                                                                                                                          
     Flatten                     ✓                        ✓              ✓          ✓     Reshape                                                                                                                                                                                                                                                                                                                        
     Floor                       ✓                        ✓                               Floor                                                                                                                                                                                                                                                                                                                          
     GRU                         X         X                   X                                                                                           Not yet implemented.                                                                                                                                                                                                                                          
     Gather                      ✓                        ✓                         ✓     Concatenate, Slice                                                                                                                                                                                                                                                                                                             
     GatherElements                                                                 X                                                                      Not yet implemented.                                                                                                                                                                                                                                          
     GatherND                                                                       X                                                                      Not yet implemented.                                                                                                                                                                                                                                          
     Gemm                        ✓                        ✓    ✓         ✓          ✓     MulScalar, Reshape, Add2, BatchMatmul                                                                                                                                                                                                                                                                                          
     GlobalAveragePool           ✓                        ✓                               GlobalAveragePooling                                                                                                                                                                                                                                                                                                           
     GlobalLpPool                X    X                                                                                                                    Not yet implemented.                                                                                                                                                                                                                                          
     GlobalMaxPool               X                                                                                                                         Not yet implemented.                                                                                                                                                                                                                                          
     Greater                     ✓                        ✓    ✓         ✓                Reshape, Greater                                                                                                                                                                                                                                                                                                               
     HardSigmoid                 ✓                        ✓                               MaximumScalar, HardSigmoid, MinimumScalar, AddScalar, MulScalar                                                                                                                                                                                                                                                                
     Hardmax                     ✓                        ✓                         ✓     Max, OneHot, Reshape                                                                                                                                                                                                                                                                                                           
     Identity                    ✓                        ✓                               Identity                                                                                                                                                                                                                                                                                                                       
     If                          X                                                                                                                         Not yet implemented.                                                                                                                                                                                                                                          
     InstanceNormalization       ✓                        ✓                               Concatenate, Split, BatchNormalization, Reshape                                                                                                                                                                                                                                                                                
     IsInf                                                                    ✓           IsInf                                                                                                                                                                                                                                                                                                                          
     IsNaN                                                               ✓                IsNaN                                                                                                                                                                                                                                                                                                                          
     LRN                         ✓                        ✓                               SumPooling, PowScalar, Transpose, Div2, AddScalar, MulScalar                                                                                                                                                                                                                                                                   
     LSTM                        X                             X                                                                                           Not yet implemented.                                                                                                                                                                                                                                          
     LeakyRelu                   ✓                        ✓                               LeakyReLU                                                                                                                                                                                                                                                                                                                      
     Less                        ✓                        ✓    ✓         ✓                Less, Reshape                                                                                                                                                                                                                                                                                                                  
     Log                         ✓                        ✓                               Log                                                                                                                                                                                                                                                                                                                            
     LogSoftmax                  ✓                        ✓                         ✓     Exp, Sub2, Max, Log, Add2, Sum, Reshape                                                                                                                                                                                                                                                                                        
     Loop                        X                                                  X                                                                      Not yet implemented.                                                                                                                                                                                                                                          
     LpNormalization             X                                                                                                                         Not yet implemented.                                                                                                                                                                                                                                          
     LpPool                      X    X                                             X                                                                      Not yet implemented.                                                                                                                                                                                                                                          
     MatMul                      ✓                        ✓              ✓                BatchMatmul                                                                                                                                                                                                                                                                                                                    
     MatMulInteger                                                            X                                                                            Not yet implemented.                                                                                                                                                                                                                                          
     Max                         ✓                        ✓         ✓    ✓                Maximum2                                                                                                                                                                                                                                                                                                                       
     MaxPool                     ✓                        ✓         X         X     X     Pad, MaxPooling                                                  Not all features are verified, since some features are not supported by caffe2. Those features can be verified by ONNXRuntime. if opset >= 10, the ceil_mode is not supported, dilations is not equal to 1 is not supported.                                  
     MaxRoiPool                  X                                                                                                                         Not yet implemented.                                                                                                                                                                                                                                          
     MaxUnpool                                                           X          X                                                                      Not yet implemented.                                                                                                                                                                                                                                          
     Mean                        ✓                        ✓         ✓    ✓                Stack, Mean, Broadcast                                                                                                                                                                                                                                                                                                         
     MeanVarianceNormalization                                           X                                                                                 Not yet implemented.                                                                                                                                                                                                                                          
     Min                         ✓                        ✓         ✓    ✓                Minimum2                                                                                                                                                                                                                                                                                                                       
     Mod                                                                      X                                                                            Not yet implemented.                                                                                                                                                                                                                                          
     Mul                         ✓                        ✓    ✓                          Mul2, Reshape                                                                                                                                                                                                                                                                                                                  
     Multinomial                                               X                                                                                           Not yet implemented.                                                                                                                                                                                                                                          
     Neg                         ✓                        ✓                               MulScalar                                                                                                                                                                                                                                                                                                                      
     NonMaxSuppression                                                        X     X                                                                      Not yet implemented.                                                                                                                                                                                                                                          
     NonZero                                                             X                                                                                 Not yet implemented.                                                                                                                                                                                                                                          
     Not                         ✓                        ✓                               LogicalNot                                                                                                                                                                                                                                                                                                                     
     OneHot                                                              X          X                                                                      Not yet implemented.                                                                                                                                                                                                                                          
     Or                          ✓                        ✓    ✓                          LogicalOr, Reshape                                                                                                                                                                                                                                                                                                             
     PRelu                       ✓                        ✓    X         X                PReLU                                                                                                                                                                                                                                                                                                                          
     Pad                         ✓    ✓                   ✓                         ✓     Pad                                                              Onnx required to support "edge" mode, while nnabla does not support it.                                                                                                                                                                                       
     Pow                         ✓                        ✓    ✓                          Pow2, Reshape                                                                                                                                                                                                                                                                                                                  
     QLinearConv                                                              X                                                                            Not yet implemented.                                                                                                                                                                                                                                          
     QLinearMatMul                                                            X                                                                            Not yet implemented.                                                                                                                                                                                                                                          
     QuantizeLinear                                                           X                                                                            Not yet implemented.                                                                                                                                                                                                                                          
     RNN                         X                             X                                                                                           Not yet implemented.                                                                                                                                                                                                                                          
     RandomNormal                X                                                                                                                         Not yet implemented.                                                                                                                                                                                                                                          
     RandomNormalLike            X                                                                                                                         Not yet implemented.                                                                                                                                                                                                                                          
     RandomUniform               X                                                                                                                         Not yet implemented.                                                                                                                                                                                                                                          
     RandomUniformLike           X                                                                                                                         Not yet implemented.                                                                                                                                                                                                                                          
     Range                                                                          X                                                                      Not yet implemented.                                                                                                                                                                                                                                          
     Reciprocal                  ✓                        ✓                               RDivScalar                                                                                                                                                                                                                                                                                                                     
     ReduceL1                    X                                                  X                                                                      Not yet implemented.                                                                                                                                                                                                                                          
     ReduceL2                    X                                                  X                                                                      Not yet implemented.                                                                                                                                                                                                                                          
     ReduceLogSum                X                                                  X                                                                      Not yet implemented.                                                                                                                                                                                                                                          
     ReduceLogSumExp             X                                                  X                                                                      Not yet implemented.                                                                                                                                                                                                                                          
     ReduceMax                   ✓                        ✓                         ✓     Max                                                                                                                                                                                                                                                                                                                            
     ReduceMean                  ✓                        ✓                         ✓     Mean                                                                                                                                                                                                                                                                                                                           
     ReduceMin                   ✓                        ✓                         ✓     Min                                                                                                                                                                                                                                                                                                                            
     ReduceProd                  ✓                        ✓                         ✓     Prod                                                                                                                                                                                                                                                                                                                           
     ReduceSum                   ✓                        ✓                         ✓     Sum                                                                                                                                                                                                                                                                                                                            
     ReduceSumSquare             ✓                        ✓                         ✓     Sum, PowScalar                                                                                                                                                                                                                                                                                                                 
     Relu                        ✓                        ✓                               ReLU                                                                                                                                                                                                                                                                                                                           
     Reshape                     ✓                   ✓    ✓                               Reshape                                                                                                                                                                                                                                                                                                                        
     Resize                                                                   X     X                                                                      Not yet implemented.                                                                                                                                                                                                                                          
     ReverseSequence                                                          X                                                                            Not yet implemented.                                                                                                                                                                                                                                          
     RoiAlign                                                                 X                                                                            Not yet implemented.                                                                                                                                                                                                                                          
     Round                                                                          ✓     Round                                                                                                                                                                                                                                                                                                                          
     Scan                                                           X    X          X                                                                      Not yet implemented.                                                                                                                                                                                                                                          
     Scatter                                                             X          X                                                                      Not yet implemented.                                                                                                                                                                                                                                          
     ScatterElements                                                                X                                                                      Not yet implemented.                                                                                                                                                                                                                                          
     ScatterND                                                                      X                                                                      Not yet implemented.                                                                                                                                                                                                                                          
     Selu                        ✓                        ✓                               SELU                                                                                                                                                                                                                                                                                                                           
     SequenceAt                                                                     X                                                                      Not yet implemented.                                                                                                                                                                                                                                          
     SequenceConstruct                                                              X                                                                      Not yet implemented.                                                                                                                                                                                                                                          
     SequenceErase                                                                  X                                                                      Not yet implemented.                                                                                                                                                                                                                                          
     SequenceInsert                                                                 X                                                                      Not yet implemented.                                                                                                                                                                                                                                          
     SequenceLength                                                                 X                                                                      Not yet implemented.                                                                                                                                                                                                                                          
     Shape                       X                                                                                                                         Not yet implemented.                                                                                                                                                                                                                                          
     Shrink                                                              X                                                                                 Not yet implemented.                                                                                                                                                                                                                                          
     Sigmoid                     ✓                        ✓                               Sigmoid                                                                                                                                                                                                                                                                                                                        
     Sign                                                                ✓                Sign                                                                                                                                                                                                                                                                                                                           
     Sin                                                       ✓                          Sin                                                                                                                                                                                                                                                                                                                            
     Sinh                                                                ✓                Sinh                                                                                                                                                                                                                                                                                                                           
     Size                        X                                                                                                                         Not yet implemented.                                                                                                                                                                                                                                          
     Slice                       ✓                        ✓                   ✓     X     Slice                                                                                                                                                                                                                                                                                                                          
     Softmax                     ✓                        ✓                         ✓     Exp, Sub2, Max, Div2, Sum, Reshape                                                                                                                                                                                                                                                                                             
     Softplus                    ✓                        ✓                               SoftPlus                                                                                                                                                                                                                                                                                                                       
     Softsign                    ✓                        ✓                               SoftSign                                                                                                                                                                                                                                                                                                                       
     SpaceToDepth                ✓                        ✓                               Transpose, Reshape                                                                                                                                                                                                                                                                                                             
     Split                       ✓    ✓                   ✓                         ✓     Stack, Split                                                                                                                                                                                                                                                                                                                   
     SplitToSequence                                                                X                                                                      Not yet implemented.                                                                                                                                                                                                                                          
     Sqrt                        ✓                        ✓                               PowScalar                                                                                                                                                                                                                                                                                                                      
     Squeeze                     ✓                        ✓                         ✓     Reshape                                                                                                                                                                                                                                                                                                                        
     StringNormalizer                                                         X                                                                            Not yet implemented.                                                                                                                                                                                                                                          
     Sub                         ✓                        ✓    ✓                          Sub2, Reshape                                                                                                                                                                                                                                                                                                                  
     Sum                         ✓                        ✓         ✓    ✓                Add2                                                                                                                                                                                                                                                                                                                           
     Tan                                                       ✓                          Tan                                                                                                                                                                                                                                                                                                                            
     Tanh                        ✓                        ✓                               Tanh                                                                                                                                                                                                                                                                                                                           
     TfIdfVectorizer                                                     X                                                                                 Not yet implemented.                                                                                                                                                                                                                                          
     ThresholdedRelu                                                          ✓           GreaterScalar, Constant, Where                                                                                                                                                                                                                                                                                                 
     Tile                        ✓                        ✓                               Tile                                                                                                                                                                                                                                                                                                                           
     TopK                        X                                            X     X                                                                      Not yet implemented.                                                                                                                                                                                                                                          
     Transpose                   ✓                        ✓                               Transpose                                                                                                                                                                                                                                                                                                                      
     Unique                                                                         X                                                                      Not yet implemented.                                                                                                                                                                                                                                          
     Unsqueeze                   ✓                        ✓                         ✓     Reshape                                                                                                                                                                                                                                                                                                                        
     Upsample                    ✓                        ✓    ✓         ✓    X           Unpooling                                                                                                                                                                                                                                                                                                                      
     Where                                                               ✓                Where                                                                                                                                                                                                                                                                                                                          
     Xor                         ✓                        ✓    ✓                          LogicalXor, Reshape                                                                                                                                                                                                                                                                                                            
    ===========================  ===  ===  ===  ===  ===  ===  ===  ===  ===  ====  ====  ===============================================================  ==============================================================================================================================================================================================================================================================



Export
------

- ✓: Support to export this opset.
- △: Partially support to export this opset (e.g. some cases cannot be supported, or not completely tested).
- X: Supported, but test failed.
- Empty: Not support corresponding opset version.

Total: 120/181

Neural Network Layer
^^^^^^^^^^^^^^^^^^^^

Count 11/15
 

    ===============================  ===  ===  ===  ====  ====  ========================================  ======================================================================================
            NNabla Function           6    7    9    10    11                   ONNX Op                                                        Description                                      
    ===============================  ===  ===  ===  ====  ====  ========================================  ======================================================================================
      Affine                         ✓    ✓    ✓    ✓     ✓     Gemm, Reshape                                                                                                                   
      RNN                                                                                                 Not yet implemented.                                                                  
      LSTM                                                                                                Not yet implemented.                                                                  
      GRU                                                                                                 Not yet implemented.                                                                  
      Convolution                    ✓    ✓    ✓    ✓     ✓     Conv, Reshape                                                                                                                   
      DepthwiseConvolution           ✓    ✓    ✓    ✓     ✓     Conv, Reshape                                                                                                                   
      Deconvolution                  △    △    △    △     △     ConvTranspose, Reshape                    Caffe2 and onnxruntime do not support dilations != 1.                                 
      DepthwiseDeconvolution         △    △    △    △     △     ConvTranspose, Reshape                    Caffe2 and onnxruntime do not support dilations != 1.                                 
      AdaptiveSeparableConvolution                                                                        Not yet implemented.                                                                  
      MaxPooling                     ✓    ✓    ✓    ✓     ✓     Pad, Constant, MaxPool, Reshape                                                                                                 
      AveragePooling                 △    △    △    △     △     Pad, Constant, AveragePool, Reshape       Currently only supports the cases where both ignore_border and including_pad are True.
      GlobalAveragePooling           ✓    ✓    ✓    ✓     ✓     GlobalAveragePool                                                                                                               
      SumPooling                     X    ✓    ✓    ✓     ✓     AveragePool, Constant, Pad, Mul, Reshape                                                                                        
      Unpooling                      △    ✓    ✓    ✓     ✓     Resize                                    The kernel only supports 2d on opset 6.                                               
      Embed                          ✓    ✓    ✓    ✓     ✓     Gather                                                                                                                          
    ===============================  ===  ===  ===  ====  ====  ========================================  ======================================================================================


Neural Network Activation Functions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Count 21/21
 

    =================  ===  ===  ===  ====  ====  ========================================  =================================================================================
     NNabla Function    6    7    9    10    11                   ONNX Op                                                      Description                                   
    =================  ===  ===  ===  ====  ====  ========================================  =================================================================================
      Sigmoid          ✓    ✓    ✓    ✓     ✓     Sigmoid                                                                                                                    
      Swish            ✓    ✓    ✓    ✓     ✓     Sigmoid, Mul                                                                                                               
      Tanh             ✓    ✓    ✓    ✓     ✓     Tanh                                                                                                                       
      ReLU             ✓    ✓    ✓    ✓     ✓     Relu                                                                                                                       
      LeakyReLU        ✓    ✓    ✓    ✓     ✓     LeakyRelu                                                                                                                  
      Softmax          △    ✓    ✓    ✓     ✓     ReduceMax, Exp, Div, Sub, ReduceSum       ONNX Add, Sub operator does not support multidirectional broadcasting on opset 6.
      LogSoftmax       △    ✓    ✓    ✓     ✓     ReduceMax, Exp, Log, Sub, ReduceSum                                                                                        
      ELU              ✓    ✓    ✓    ✓     ✓     Elu                                                                                                                        
      SELU             ✓    ✓    ✓    ✓     ✓     Selu                                                                                                                       
      CReLU            ✓    ✓    ✓    ✓     ✓     Neg, Relu, Concat                                                                                                          
      CELU             ✓    ✓    ✓    ✓     ✓     Neg, Elu, Concat                                                                                                           
      PReLU            ✓    ✓    ✓    ✓     ✓     PRelu, Reshape                                                                                                             
      GELU             ✓    ✓    ✓    ✓     ✓     Constant, Div, Add, Mul, Sqrt, Pow, Tanh                                                                                   
      ReLU6            ✓    ✓    ✓    ✓     ✓     Constant, Relu, Min                                                                                                        
      HardSigmoid      ✓    ✓    ✓    ✓     ✓     HardSigmoid                                                                                                                
      HardTanh         ✓    ✓    ✓    ✓     ✓     Neg, Max, Constant, Min                                                                                                    
      LogSigmoid       ✓    ✓    ✓    ✓     ✓     Sigmoid, Log                                                                                                               
      SoftPlus         ✓    ✓    ✓    ✓     ✓     Softplus                                                                                                                   
      SoftSign         ✓    ✓    ✓    ✓     ✓     Softsign                                                                                                                   
      TanhShrink       ✓    ✓    ✓    ✓     ✓     Sub, Tanh                                                                                                                  
      Sinc             X    X    ✓    ✓     ✓     Where, Constant, Div, Equal, Sin                                                                                           
    =================  ===  ===  ===  ====  ====  ========================================  =================================================================================


Normalization
^^^^^^^^^^^^^

Count 2/6
 

    ==========================  ===  ===  ===  ====  ====  ======================================================================================  =======================================================================================================
         NNabla Function         6    7    9    10    11                                          ONNX Op                                                                                        Description                                              
    ==========================  ===  ===  ===  ====  ====  ======================================================================================  =======================================================================================================
      FusedBatchNormalization   ✓    ✓    ✓    ✓     ✓     Constant, Relu, Div, Sub, ReduceMean, Add, Mul, ReduceSum, BatchNormalization, Reshape                                                                                                         
      BatchNormalization        ✓    ✓    ✓    ✓     ✓     Constant, Div, Sub, ReduceMean, Mul, ReduceSum, BatchNormalization, Reshape             In inferring stage, caffe2 mistmatch onnx 1.4.x's implementation, "in-place" feature cannot be applied.
      SyncBatchNormalization                                                                                                                       Not yet implemented.                                                                                   
      MeanSubtraction                                                                                                                              Not yet implemented.                                                                                   
      ClipGradByValue                                                                                                                              Not yet implemented.                                                                                   
      ClipGradByNorm                                                                                                                               Not yet implemented.                                                                                   
    ==========================  ===  ===  ===  ====  ====  ======================================================================================  =======================================================================================================


Reduction
^^^^^^^^^

Count 5/7
 

    =================  ===  ===  ===  ====  ====  ==========  ====================
     NNabla Function    6    7    9    10    11    ONNX Op        Description     
    =================  ===  ===  ===  ====  ====  ==========  ====================
      Sum              ✓    ✓    ✓    ✓     ✓     ReduceSum                       
      Mean             ✓    ✓    ✓    ✓     ✓     ReduceMean                      
      Max              ✓    ✓    ✓    ✓     ✓     ReduceMax                       
      Min              ✓    ✓    ✓    ✓     ✓     ReduceMin                       
      Prod             ✓    ✓    ✓    ✓     ✓     ReduceProd                      
      ReduceSum                                               Not yet implemented.
      ReduceMean                                              Not yet implemented.
    =================  ===  ===  ===  ====  ====  ==========  ====================


Arithmetic
^^^^^^^^^^

Count 11/14
 

    =================  ===  ===  ===  ====  ====  =============  ============================================================================
     NNabla Function    6    7    9    10    11      ONNX Op                                     Description                                 
    =================  ===  ===  ===  ====  ====  =============  ============================================================================
      Add2             △    ✓    ✓    ✓     ✓     Add            ONNX Add operator does not support multidirectional broadcasting on opset 6.
      AddN                                                       Not yet implemented.                                                        
      BcAdd2                                                     Not yet implemented.                                                        
      Sub2             △    ✓    ✓    ✓     ✓     Sub            ONNX Sub operator does not support multidirectional broadcasting on opset 6.
      Mul2             △    ✓    ✓    ✓     ✓     Mul            ONNX Mul operator does not support multidirectional broadcasting on opset 6.
      MulN                                                       Not yet implemented.                                                        
      Div2             △    ✓    ✓    ✓     ✓     Div            ONNX Div operator does not support multidirectional broadcasting on opset 6.
      Pow2             △    ✓    ✓    ✓     ✓     Pow            ONNX Pow operator does not support multidirectional broadcasting on opset 6.
      AddScalar        ✓    ✓    ✓    ✓     ✓     Constant, Add                                                                              
      MulScalar        ✓    ✓    ✓    ✓     ✓     Constant, Mul                                                                              
      PowScalar        ✓    ✓    ✓    ✓     ✓     Constant, Pow                                                                              
      RSubScalar       ✓    ✓    ✓    ✓     ✓     Sub, Constant                                                                              
      RDivScalar       ✓    ✓    ✓    ✓     ✓     Constant, Div                                                                              
      RPowScalar       ✓    ✓    ✓    ✓     ✓     Constant, Pow                                                                              
    =================  ===  ===  ===  ====  ====  =============  ============================================================================


Logical
^^^^^^^

Count 29/29
 

    =====================  ===  ===  ===  ====  ====  ======================  ============================================================================
       NNabla Function      6    7    9    10    11          ONNX Op                                          Description                                 
    =====================  ===  ===  ===  ====  ====  ======================  ============================================================================
      Sign                 X    X    ✓    ✓     ✓     Sign                                                                                                
      Minimum2             △    ✓    ✓    ✓     ✓     Constant, Add, Min      ONNX Add operator does not support multidirectional broadcasting on opset 6.
      Maximum2             △    ✓    ✓    ✓     ✓     Max, Add, Constant      ONNX Add operator does not support multidirectional broadcasting on opset 6.
      MinimumScalar        ✓    ✓    ✓    ✓     ✓     Constant, Add, Min                                                                                  
      MaximumScalar        ✓    ✓    ✓    ✓     ✓     Max, Add, Constant                                                                                  
      LogicalAnd           ✓    ✓    ✓    ✓     ✓     And                                                                                                 
      LogicalOr            ✓    ✓    ✓    ✓     ✓     Or                                                                                                  
      LogicalXor           ✓    ✓    ✓    ✓     ✓     Xor                                                                                                 
      Equal                ✓    ✓    ✓    ✓     ✓     Equal                                                                                               
      NotEqual             ✓    ✓    ✓    ✓     ✓     Equal, Not                                                                                          
      GreaterEqual         ✓    ✓    ✓    ✓     ✓     Not, Less                                                                                           
      Greater              ✓    ✓    ✓    ✓     ✓     Greater                                                                                             
      LessEqual            ✓    ✓    ✓    ✓     ✓     Not, Greater                                                                                        
      Less                 ✓    ✓    ✓    ✓     ✓     Less                                                                                                
      LogicalAndScalar     ✓    ✓    ✓    ✓     ✓     And, Constant                                                                                       
      LogicalOrScalar      ✓    ✓    ✓    ✓     ✓     Or, Constant                                                                                        
      LogicalXorScalar     ✓    ✓    ✓    ✓     ✓     Constant, Xor                                                                                       
      EqualScalar          ✓    ✓    ✓    ✓     ✓     Equal, Constant                                                                                     
      NotEqualScalar       ✓    ✓    ✓    ✓     ✓     Equal, Constant, Not                                                                                
      GreaterEqualScalar   ✓    ✓    ✓    ✓     ✓     Constant, Not, Less                                                                                 
      GreaterScalar        ✓    ✓    ✓    ✓     ✓     Constant, Greater                                                                                   
      LessEqualScalar      ✓    ✓    ✓    ✓     ✓     Constant, Not, Greater                                                                              
      LessScalar           ✓    ✓    ✓    ✓     ✓     Constant, Less                                                                                      
      LogicalNot           ✓    ✓    ✓    ✓     ✓     Not                                                                                                 
      IsNaN                X    X    ✓    ✓     ✓     IsNaN                                                                                               
      IsInf                X    X    X    ✓     ✓     IsInf                                                                                               
      ResetNaN             X    X    ✓    ✓     ✓     IsNaN, Constant, Where                                                                              
      ResetInf             X    X    X    ✓     ✓     Constant, IsInf, Where                                                                              
      Where                X    X    ✓    ✓     ✓     Where                                                                                               
    =====================  ===  ===  ===  ====  ====  ======================  ============================================================================


Math
^^^^

Count 22/22
 

    =================  ===  ===  ===  ====  ====  ==========================  =============
     NNabla Function    6    7    9    10    11            ONNX Op             Description 
    =================  ===  ===  ===  ====  ====  ==========================  =============
      Constant         ✓    ✓    ✓    ✓     ✓     Constant, Identity                       
      Arange           ✓    ✓    ✓    ✓     ✓     Constant, Identity                       
      Abs              ✓    ✓    ✓    ✓     ✓     Abs                                      
      Exp              ✓    ✓    ✓    ✓     ✓     Exp                                      
      Log              ✓    ✓    ✓    ✓     ✓     Log                                      
      Identity         ✓    ✓    ✓    ✓     ✓     Identity                                 
      BatchMatmul      ✓    ✓    ✓    ✓     ✓     MatMul, Transpose, Reshape               
      Round            X    X    X    X     ✓     Round                                    
      Ceil             ✓    ✓    ✓    ✓     ✓     Ceil                                     
      Floor            ✓    ✓    ✓    ✓     ✓     Floor                                    
      Sin              X    ✓    ✓    ✓     ✓     Sin                                      
      Cos              X    ✓    ✓    ✓     ✓     Cos                                      
      Tan              X    ✓    ✓    ✓     ✓     Tan                                      
      Sinh             X    X    ✓    ✓     ✓     Sinh                                     
      Cosh             X    X    ✓    ✓     ✓     Cosh                                     
      ASin             X    ✓    ✓    ✓     ✓     Asin                                     
      ACos             X    ✓    ✓    ✓     ✓     Acos                                     
      ATan             X    ✓    ✓    ✓     ✓     Atan                                     
      ATan2            X    ✓    ✓    ✓     ✓     Atan, Div                                
      ASinh            X    X    ✓    ✓     ✓     Asinh                                    
      ACosh            X    X    ✓    ✓     ✓     Acosh                                    
      ATanh            X    X    ✓    ✓     ✓     Atanh                                    
    =================  ===  ===  ===  ====  ====  ==========================  =============


Array Manipulation
^^^^^^^^^^^^^^^^^^

Count 12/21
 

    =================  ===  ===  ===  ====  ====  ===========================  ============================================================================================================================
     NNabla Function    6    7    9    10    11             ONNX Op                                                                    Description                                                         
    =================  ===  ===  ===  ====  ====  ===========================  ============================================================================================================================
      Concatenate      ✓    ✓    ✓    ✓     ✓     Concat                                                                                                                                                   
      Split            ✓    ✓    ✓    ✓     ✓     Split, Squeeze                                                                                                                                           
      Stack            ✓    ✓    ✓    ✓     ✓     Unsqueeze, Concat                                                                                                                                        
      Slice            △    △    △    △     △     Constant, Slice              ONNX slice cannot support step != 1 on opset < 10.                                                                          
      Pad              △    △    △    △     △     Pad, Constant                When the mode of the pad is reflect, if the size of the pad exceeds the input size, caffe2 and onnxruntime cannot handle it.
      Transpose        ✓    ✓    ✓    ✓     ✓     Transpose                                                                                                                                                
      Broadcast        X    X    ✓    ✓     ✓                                                                                                                                                              
      BroadcastTo      ✓    ✓    ✓    ✓     ✓                                                                                                                                                              
      Tile             ✓    ✓    ✓    ✓     △     Constant, Tile, Reshape                                                                                                                                  
      OneHot           ✓    ✓    ✓    ✓     ✓     Gather, Flatten, Reshape                                                                                                                                 
      Flip             ✓    ✓    ✓    ✓     ✓     Identity, Gather, Transpose                                                                                                                              
      Shift                                                                    Not yet implemented.                                                                                                        
      Sort                                                                     Not yet implemented.                                                                                                        
      Reshape          ✓    ✓    ✓    ✓     ✓     Constant, Reshape                                                                                                                                        
      MatrixDiag                                                               Not yet implemented.                                                                                                        
      MatrixDiagPart                                                           Not yet implemented.                                                                                                        
      BatchInv                                                                 Not yet implemented.                                                                                                        
      BatchDet                                                                 Not yet implemented.                                                                                                        
      Assign                                                                   Not yet implemented.                                                                                                        
      GatherNd                                                                 Not yet implemented.                                                                                                        
      ScatterNd                                                                Not yet implemented.                                                                                                        
    =================  ===  ===  ===  ====  ====  ===========================  ============================================================================================================================


Signal Processing
^^^^^^^^^^^^^^^^^

Count 1/3
 

    =================  ===  ===  ===  ====  ====  ===============  ====================
     NNabla Function    6    7    9    10    11       ONNX Op          Description     
    =================  ===  ===  ===  ====  ====  ===============  ====================
      Interpolate      X    X    X    X     △     Resize, Reshape                      
      FFT                                                          Not yet implemented.
      IFFT                                                         Not yet implemented.
    =================  ===  ===  ===  ====  ====  ===============  ====================


Stochasticity
^^^^^^^^^^^^^

Count 0/12
 

    ====================  ===  ===  ===  ====  ====  =========  ==================================================================================================================
      NNabla Function      6    7    9    10    11    ONNX Op                                                      Description                                                    
    ====================  ===  ===  ===  ====  ====  =========  ==================================================================================================================
      Dropout             X    X    X    X     X     Dropout    The Dropout in nnabla has no test mode and contains random parameters, so the test result is not the same as onnx.
      TopKData                                                  Not yet implemented.                                                                                              
      TopKGrad                                                  Not yet implemented.                                                                                              
      Rand                                                      Not yet implemented.                                                                                              
      Randint                                                   Not yet implemented.                                                                                              
      Randn                                                     Not yet implemented.                                                                                              
      RandomChoice                                              Not yet implemented.                                                                                              
      RandomCrop                                                Not yet implemented.                                                                                              
      RandomFlip                                                Not yet implemented.                                                                                              
      RandomShift                                               Not yet implemented.                                                                                              
      RandomErase                                               Not yet implemented.                                                                                              
      ImageAugmentation                                         Not yet implemented.                                                                                              
    ====================  ===  ===  ===  ====  ====  =========  ==================================================================================================================


Loss Functions
^^^^^^^^^^^^^^

Count 0/9
 

    ==========================  ===  ===  ===  ====  ====  =========  ====================
         NNabla Function         6    7    9    10    11    ONNX Op       Description     
    ==========================  ===  ===  ===  ====  ====  =========  ====================
      SigmoidCrossEntropy                                             Not yet implemented.
      BinaryCrossEntropy                                              Not yet implemented.
      SoftmaxCrossEntropy                                             Not yet implemented.
      CategoricalCrossEntropy                                         Not yet implemented.
      SquaredError                                                    Not yet implemented.
      AbsoluteError                                                   Not yet implemented.
      HuberLoss                                                       Not yet implemented.
      EpsilonInsensitiveLoss                                          Not yet implemented.
      KLMultinomial                                                   Not yet implemented.
    ==========================  ===  ===  ===  ====  ====  =========  ====================


Quantization Neural Network Layers
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Count 6/12
 

    ===========================  ===  ===  ===  ====  ====  =========================  ====================
          NNabla Function         6    7    9    10    11            ONNX Op               Description     
    ===========================  ===  ===  ===  ====  ====  =========================  ====================
      BinarySigmoid              X    X    ✓    ✓     ✓     Constant, Where, Greater                       
      BinaryTanh                 X    X    ✓    ✓     ✓     Constant, Where, Greater                       
      BinaryConnectAffine        ✓    ✓    ✓    ✓     ✓     Gemm, Reshape                                  
      BinaryConnectConvolution   ✓    ✓    ✓    ✓     ✓     Conv, Reshape                                  
      BinaryWeightAffine         ✓    ✓    ✓    ✓     ✓     MatMul, Add, Mul, Reshape                      
      BinaryWeightConvolution    ✓    ✓    ✓    ✓     ✓     Add, Conv, Mul, Reshape                        
      INQAffine                                                                        Not yet implemented.
      INQConvolution                                                                   Not yet implemented.
      FixedPointQuantize                                                               Not yet implemented.
      MinMaxQuantize                                                                   Not yet implemented.
      Pow2Quantize                                                                     Not yet implemented.
      Prune                                                                            Not yet implemented.
    ===========================  ===  ===  ===  ====  ====  =========================  ====================


Validation
^^^^^^^^^^

Count 0/3
 

    ==================  ===  ===  ===  ====  ====  =========  ====================
     NNabla Function     6    7    9    10    11    ONNX Op       Description     
    ==================  ===  ===  ===  ====  ====  =========  ====================
      TopNError                                               Not yet implemented.
      BinaryError                                             Not yet implemented.
      ConfusionMatrix                                         Not yet implemented.
    ==================  ===  ===  ===  ====  ====  =========  ====================


Unsupported, Special Use
^^^^^^^^^^^^^^^^^^^^^^^^

Count 0/7
 

    =====================  ===  ===  ===  ====  ====  =========  ====================
       NNabla Function      6    7    9    10    11    ONNX Op       Description     
    =====================  ===  ===  ===  ====  ====  =========  ====================
      VATNoise                                                   Not yet implemented.
      Unlink                                                     Not yet implemented.
      Sink                                                       Not yet implemented.
      NmsDetection2d                                             Not yet implemented.
      MaxPoolingBackward                                         Not yet implemented.
      WarpByFlow                                                 Not yet implemented.
      PatchCorrelation                                           Not yet implemented.
    =====================  ===  ===  ===  ====  ====  =========  ====================





Tensorflow Support Status
=========================

:Note: In this document, the numbers in the header of all tables represent the version of onnx opset.

Import
------

- ✓: Supported
- △: Partially supported
- X: Supported, but test failed.
- Empty: Not support yet.


Total: 86/120

.. table:: Tensorflow support status

    ======================================  ========  ==================================  =======================================================================================================
             Tensorflow Function             Status              NNabla Func                                                            Description                                              
    ======================================  ========  ==================================  =======================================================================================================
      Abs                                      ✓      Abs                                                                                                                                        
      Acos                                     ✓      ACos                                                                                                                                       
      Acosh                                    ✓      ACosh                                                                                                                                      
      Add                                      ✓      Add2                                                                                                                                       
      AddN                                     ✓      Add2                                                                                                                                       
      All                                                                                 Not yet implemented.                                                                                   
      Any                                                                                 Not yet implemented.                                                                                   
      ArgMax                                   ✓      Max                                                                                                                                        
      ArgMin                                   ✓      Min                                                                                                                                        
      Asin                                     ✓      ASin                                                                                                                                       
      Asinh                                    ✓      ASinh                                                                                                                                      
      Atan                                     ✓      ATan                                                                                                                                       
      Atan2                                                                               Not yet implemented.                                                                                   
      Atanh                                    ✓      ATanh                                                                                                                                      
      AvgPool                                  △      Pad, AveragePooling, Transpose      Some feature is not supported by Nnabla such as Pad's edge mode.                                       
      AvgPool3D                                                                           Not yet implemented.                                                                                   
      BatchMatMul                              ✓      Transpose, BatchMatmul                                                                                                                     
      BiasAdd                                  ✓      Add2, Reshape                                                                                                                              
      Cast                                                                                Not yet implemented.                                                                                   
      Ceil                                     ✓      Ceil                                                                                                                                       
      ConcatV2                                 ✓      Concatenate                                                                                                                                
      Const                                    ✓      Add2                                                                                                                                       
      Conv2D                                   △      Convolution, Pad, Transpose         Tensorflow require GPU to perform related test cases. This issue is recorded only for memo.            
      Conv2DBackpropFilter                                                                Not yet implemented.                                                                                   
      Conv2DBackpropInput                      △      Transpose, Deconvolution            Tensorflow require GPU to perform related test cases. This issue is recorded only for memo.            
      Conv3D                                                                              Not yet implemented.                                                                                   
      Conv3DBackpropFilterV2                                                              Not yet implemented.                                                                                   
      Conv3DBackpropInputV2                                                               Not yet implemented.                                                                                   
      Cos                                      ✓      Cos                                                                                                                                        
      Cosh                                     ✓      Cosh                                                                                                                                       
      DepthToSpace                             △      Transpose, Reshape                  Tensorflow require GPU to perform related test cases. This issue is recorded only for memo.            
      DepthwiseConv2dNative                                                               Not yet implemented.                                                                                   
      DepthwiseConv2dNativeBackpropFilter                                                 Not yet implemented.                                                                                   
      DepthwiseConv2dNativeBackpropInput                                                  Not yet implemented.                                                                                   
      Div                                      ✓      Div2                                                                                                                                       
      Elu                                      ✓      ELU                                                                                                                                        
      Equal                                    ✓      Equal                                                                                                                                      
      Erf                                                                                 Not yet implemented.                                                                                   
      Erfc                                                                                Not yet implemented.                                                                                   
      Exp                                      ✓      Exp                                                                                                                                        
      ExpandDims                               ✓      Reshape                                                                                                                                    
      Fill                                                                                Not yet implemented.                                                                                   
      Flatten                                  ✓      Reshape                                                                                                                                    
      Floor                                    ✓      Floor                                                                                                                                      
      FloorDiv                                 ✓      Floor, Div2                                                                                                                                
      FloorMod                                 ✓      Sub2, Mul2, Floor, Div2                                                                                                                    
      FusedBatchNorm                           △      BatchNormalization, Transpose       It did not pass testing for training mode.                                                             
      GatherNd                                                                            Not yet implemented.                                                                                   
      GatherV2                                                                            Not yet implemented.                                                                                   
      Greater                                  ✓      Greater                                                                                                                                    
      GreaterEqual                             ✓      LogicalNot, Less                                                                                                                           
      Identity                                 ✓      Identity                                                                                                                                   
      IsInf                                                                               Not yet implemented.                                                                                   
      IsNan                                    ✓      IsNaN                                                                                                                                      
      LeakyRelu                                ✓      LeakyReLU                                                                                                                                  
      Less                                     ✓      Less                                                                                                                                       
      LessEqual                                ✓      LogicalNot, Greater                                                                                                                        
      Log                                      ✓      Log                                                                                                                                        
      LogSoftmax                                                                          Not yet implemented.                                                                                   
      LogicalAnd                               ✓      LogicalAnd                                                                                                                                 
      LogicalNot                               ✓      LogicalNot                                                                                                                                 
      LogicalOr                                ✓      LogicalOr                                                                                                                                  
      LogicalXor                               ✓      LogicalAnd, LogicalOr, LogicalNot                                                                                                          
      MatrixBandPart                                                                      Not yet implemented.                                                                                   
      Max                                      ✓      Max                                                                                                                                        
      MaxPool                                  ✓      Pad, Transpose, MaxPooling                                                                                                                 
      MaxPool3D                                                                           Not yet implemented.                                                                                   
      MaxPoolWithArgmax                                                                   Not yet implemented.                                                                                   
      Maximum                                  ✓      Maximum2                                                                                                                                   
      Mean                                     ✓      Mean                                                                                                                                       
      Min                                      ✓      Min                                                                                                                                        
      Minimum                                  ✓      Minimum2                                                                                                                                   
      Mul                                      ✓      Mul2                                                                                                                                       
      Neg                                      ✓      MulScalar                                                                                                                                  
      NotEqual                                 ✓      Equal, LogicalNot                                                                                                                          
      OneHot                                                                              Not yet implemented.                                                                                   
      Pack                                     ✓      Concatenate, Reshape                                                                                                                       
      Pad                                      ✓      Pad                                                                                                                                        
      Pow                                      ✓      Pow2                                                                                                                                       
      Prod                                     ✓      Prod                                                                                                                                       
      RandomShuffle                                                                       Not yet implemented.                                                                                   
      RandomStandardNormal                                                                Not yet implemented.                                                                                   
      RandomUniform                                                                       Not yet implemented.                                                                                   
      RealDiv                                  ✓      Div2                                                                                                                                       
      Reciprocal                               ✓      RDivScalar                                                                                                                                 
      Relu                                     ✓      ReLU                                                                                                                                       
      Relu6                                    ✓      MinimumScalar, MaximumScalar                                                                                                               
      Reshape                                  △      Reshape                             Some test cases failed for some nnabla's implementation limitation (e.g. -1 is regarded as batch_size).
      ReverseSequence                                                                     Not yet implemented.                                                                                   
      Rsqrt                                    ✓      PowScalar, RDivScalar                                                                                                                      
      Select                                                                              Not yet implemented.                                                                                   
      Selu                                     ✓      SELU                                                                                                                                       
      Shape                                                                               Not yet implemented.                                                                                   
      Sigmoid                                  ✓      Sigmoid                                                                                                                                    
      Sign                                     ✓      Sign                                                                                                                                       
      Sin                                      ✓      Sin                                                                                                                                        
      Sinh                                     ✓      Sinh                                                                                                                                       
      Size                                                                                Not yet implemented.                                                                                   
      Slice                                    ✓      Slice                                                                                                                                      
      Softmax                                                                             Not yet implemented.                                                                                   
      Softplus                                 ✓      SoftPlus                                                                                                                                   
      Softsign                                 ✓      SoftSign                                                                                                                                   
      SpaceToDepth                             △      Transpose, Reshape                  Tensorflow require GPU to perform related test cases. This issue is recorded only for memo.            
      SplitV                                   ✓      Stack, Split                                                                                                                               
      Sqrt                                     ✓      PowScalar                                                                                                                                  
      Square                                   ✓      Mul2                                                                                                                                       
      SquaredDifference                        ✓      Sub2, Mul2                                                                                                                                 
      Squeeze                                  ✓      Reshape                                                                                                                                    
      StopGradient                             ✓      Identity                                                                                                                                   
      StridedSlice                             ✓      Slice                                                                                                                                      
      Sub                                      ✓      Sub2                                                                                                                                       
      Sum                                      ✓      Sum                                                                                                                                        
      Tan                                      ✓      Tan                                                                                                                                        
      Tanh                                     ✓      Tanh                                                                                                                                       
      Tile                                     ✓      Tile                                                                                                                                       
      TopKV2                                                                              Not yet implemented.                                                                                   
      Transpose                                ✓      Transpose                                                                                                                                  
      TruncateDiv                                                                         Not yet implemented.                                                                                   
      TruncateMod                                                                         Not yet implemented.                                                                                   
      Unpack                                   ✓      Concatenate, Stack, Split, Reshape                                                                                                         
    ======================================  ========  ==================================  =======================================================================================================





Export
------

- ✓: Supported
- △: Partially supported
- X: Supported, but test failed.
- Empty: Not support yet.


Total: 116/181

Neural Network Layer
^^^^^^^^^^^^^^^^^^^^

Count 11/15
 

    ===============================  ========  ================================================================================================================================================  ==================================================================================
            NNabla Function           Status                                                                        TF Op                                                                                                           Description                                    
    ===============================  ========  ================================================================================================================================================  ==================================================================================
      Affine                         ✓         Const, MatMul, Add, Mul, Placeholder, Reshape                                                                                                                                                                                       
      RNN                                                                                                                                                                                        Not yet implemented.                                                              
      LSTM                                                                                                                                                                                       Not yet implemented.                                                              
      GRU                                                                                                                                                                                        Not yet implemented.                                                              
      Convolution                    △         Const, Split, Conv2D, Transpose, Pad, Identity, Add, SpaceToBatchND, ConcatV2, BatchToSpaceND, Placeholder, Reshape                               The cases `dilations` and `strides` larger than 1 are not supported by tensorflow.
      DepthwiseConvolution           △         Const, Split, Conv2D, Transpose, Pad, Add, SpaceToBatchND, ConcatV2, BatchToSpaceND, Placeholder, Reshape                                         The cases `dilations` and `strides` larger than 1 are not supported by tensorflow.
      Deconvolution                  △         Const, Split, Transpose, Slice, Conv2DBackpropInput, Identity, Add, ConcatV2, Placeholder, Reshape                                                The cases `dilations` larger than 1 are not supported by tensorflow.              
      DepthwiseDeconvolution         △         Const, Split, Transpose, Slice, Conv2DBackpropInput, Add, ConcatV2, Placeholder, Reshape                                                          The cases `dilations` larger than 1 are not supported by tensorflow.              
      AdaptiveSeparableConvolution                                                                                                                                                               Not yet implemented.                                                              
      MaxPooling                     △         Const, PyFunc, Transpose, MaxPool, PadV2, MaxPool3D, Placeholder, Reshape                                                                                                                                                           
      AveragePooling                 △         Const, AvgPool3D, PyFunc, Transpose, Pad, AvgPool, Placeholder, Reshape                                                                           Currently only supports the cases both ignore_border and including_pad are True.  
      GlobalAveragePooling           ✓         Const, Mean, SplitV, Sub, Pack, Range                                                                                                                                                                                               
      SumPooling                     ✓         Const, AvgPool3D, Transpose, Pad, Mul, AvgPool, Placeholder, Reshape                                                                                                                                                                
      Unpooling                      △         Const, Transpose, Merge, StridedSlice, Reshape, LogicalAnd, NoOp, Identity, Assert, Mul, ResizeNearestNeighbor, Equal, Cast, Placeholder, Switch  The kernel only supports 2d.                                                      
      Embed                          ✓         Const, GatherV2, Placeholder                                                                                                                                                                                                        
    ===============================  ========  ================================================================================================================================================  ==================================================================================


Neural Network Activation Functions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Count 21/21
 

    =================  ========  ====================================================================================  =============
     NNabla Function    Status                                          TF Op                                           Description 
    =================  ========  ====================================================================================  =============
      Sigmoid          ✓         Sigmoid, Placeholder                                                                               
      Swish            ✓         Sigmoid, Mul, Placeholder                                                                          
      Tanh             ✓         Placeholder, Tanh                                                                                  
      ReLU             ✓         Relu, Placeholder                                                                                  
      LeakyReLU        ✓         LeakyRelu, Placeholder                                                                             
      Softmax          ✓         Const, Exp, Max, Sum, Sub, RealDiv, Placeholder                                                    
      LogSoftmax       ✓         Const, Exp, Max, Log, Sub, Sum, Placeholder                                                        
      ELU              ✓         Const, Exp, Elu, Sub, Add, Mul, GreaterEqual, Cast, Less, Placeholder                              
      SELU             ✓         Const, Maximum, Exp, Minimum, Sub, Add, Mul, Placeholder                                           
      CReLU            ✓         Const, Relu, Neg, ConcatV2, Placeholder                                                            
      CELU             ✓         Const, Exp, Elu, GreaterEqual, Neg, Sub, Add, Mul, ConcatV2, Cast, Less, Placeholder               
      PReLU            ✓         Const, Relu, Sub, Abs, Add, Mul, Placeholder, Reshape                                              
      GELU             ✓         Const, Pow, Add, Mul, Sqrt, RealDiv, Placeholder, Tanh                                             
      ReLU6            ✓         Const, Min, Relu, Pack, Placeholder                                                                
      HardSigmoid      ✓         Const, Maximum, Minimum, Add, Mul, Placeholder                                                     
      HardTanh         ✓         Const, Min, Max, Neg, Pack, Placeholder                                                            
      LogSigmoid       ✓         Sigmoid, Log, Placeholder                                                                          
      SoftPlus         ✓         Softplus, Placeholder                                                                              
      SoftSign         ✓         Softsign, Placeholder                                                                              
      TanhShrink       ✓         Sub, Placeholder, Tanh                                                                             
      Sinc             ✓         Const, Select, Equal, RealDiv, Sin, Placeholder                                                    
    =================  ========  ====================================================================================  =============


Normalization
^^^^^^^^^^^^^

Count 2/6
 

    ==========================  ========  ===========================================================================  =======================================================================================================
         NNabla Function         Status                                      TF Op                                                                                   Description                                              
    ==========================  ========  ===========================================================================  =======================================================================================================
      FusedBatchNormalization   ✓         Const, Mean, Rsqrt, Relu, Sum, Sub, Add, Mul, RealDiv, Placeholder, Reshape                                                                                                         
      BatchNormalization        ✓         Const, Mean, Rsqrt, Sum, Sub, Add, Mul, RealDiv, Placeholder, Reshape        In inferring stage, caffe2 mistmatch onnx 1.4.x's implementation, "in-place" feature cannot be applied.
      SyncBatchNormalization                                                                                           Not yet implemented.                                                                                   
      MeanSubtraction                                                                                                  Not yet implemented.                                                                                   
      ClipGradByValue                                                                                                  Not yet implemented.                                                                                   
      ClipGradByNorm                                                                                                   Not yet implemented.                                                                                   
    ==========================  ========  ===========================================================================  =======================================================================================================


Reduction
^^^^^^^^^

Count 5/7
 

    =================  ========  ========================  ====================
     NNabla Function    Status            TF Op                Description     
    =================  ========  ========================  ====================
      Sum              ✓         Const, Sum, Placeholder                       
      Mean             ✓         Const, Mean, Placeholder                      
      Max              ✓         Const, Max, Placeholder                       
      Min              ✓         Const, Min, Placeholder                       
      Prod             ✓         Const, Prod, Placeholder                      
      ReduceSum                                            Not yet implemented.
      ReduceMean                                           Not yet implemented.
    =================  ========  ========================  ====================


Arithmetic
^^^^^^^^^^

Count 11/14
 

    =================  ========  ===========================  ====================
     NNabla Function    Status              TF Op                 Description     
    =================  ========  ===========================  ====================
      Add2             ✓         Add, Placeholder                                 
      AddN                                                    Not yet implemented.
      BcAdd2                                                  Not yet implemented.
      Sub2             ✓         Sub, Placeholder                                 
      Mul2             ✓         Mul, Placeholder                                 
      MulN                                                    Not yet implemented.
      Div2             ✓         RealDiv, Placeholder                             
      Pow2             ✓         Pow, Placeholder                                 
      AddScalar        ✓         Const, Add, Placeholder                          
      MulScalar        ✓         Const, Mul, Placeholder                          
      PowScalar        ✓         Const, Pow, Placeholder                          
      RSubScalar       ✓         Const, Sub, Placeholder                          
      RDivScalar       ✓         Const, RealDiv, Placeholder                      
      RPowScalar       ✓         Const, Pow, Placeholder                          
    =================  ========  ===========================  ====================


Logical
^^^^^^^

Count 27/29
 

    =====================  ========  =====================================================  ====================
       NNabla Function      Status                           TF Op                              Description     
    =====================  ========  =====================================================  ====================
      Sign                 ✓         Sign, Placeholder                                                          
      Minimum2             ✓         Const, Min, Add, Pack, Placeholder                                         
      Maximum2             ✓         Const, Max, Add, Pack, Placeholder                                         
      MinimumScalar        ✓         Const, Min, Add, Pack, Placeholder                                         
      MaximumScalar        ✓         Const, Max, Add, Pack, Placeholder                                         
      LogicalAnd           ✓         LogicalAnd, Placeholder                                                    
      LogicalOr            ✓         LogicalOr, Placeholder                                                     
      LogicalXor           ✓         LogicalAnd, LogicalOr, LogicalNot, Placeholder                             
      Equal                ✓         Equal, Placeholder                                                         
      NotEqual             ✓         Equal, LogicalNot, Placeholder                                             
      GreaterEqual         ✓         Placeholder, LogicalNot, Less                                              
      Greater              ✓         Placeholder, Greater                                                       
      LessEqual            ✓         Placeholder, LogicalNot, Greater                                           
      Less                 ✓         Less, Placeholder                                                          
      LogicalAndScalar     ✓         Const, LogicalAnd, Placeholder                                             
      LogicalOrScalar      ✓         Const, LogicalOr, Placeholder                                              
      LogicalXorScalar     ✓         Const, LogicalNot, LogicalAnd, LogicalOr, Placeholder                      
      EqualScalar          ✓         Equal, Const, Placeholder                                                  
      NotEqualScalar       ✓         Equal, Const, LogicalNot, Placeholder                                      
      GreaterEqualScalar   ✓         Const, Placeholder, LogicalNot, Less                                       
      GreaterScalar        ✓         Const, Placeholder, Greater                                                
      LessEqualScalar      ✓         Const, Placeholder, LogicalNot, Greater                                    
      LessScalar           ✓         Const, Less, Placeholder                                                   
      LogicalNot           ✓         LogicalNot, Placeholder                                                    
      IsNaN                ✓         IsNan, Placeholder                                                         
      IsInf                X                                                                Not yet implemented.
      ResetNaN             ✓         Const, Select, IsNan, Placeholder                                          
      ResetInf             X                                                                Not yet implemented.
      Where                ✓         Select, Placeholder                                                        
    =====================  ========  =====================================================  ====================


Math
^^^^

Count 21/22
 

    =================  ========  =====================================================  ====================
     NNabla Function    Status                           TF Op                              Description     
    =================  ========  =====================================================  ====================
      Constant         ✓         Const, Identity                                                            
      Arange           ✓         Const, Identity                                                            
      Abs              ✓         Abs, Placeholder                                                           
      Exp              ✓         Exp, Placeholder                                                           
      Log              ✓         Log, Placeholder                                                           
      Identity         ✓         Identity, Placeholder                                                      
      BatchMatmul      ✓         Const, BatchMatMulV2, Transpose, Placeholder, Reshape                      
      Round            X                                                                Not yet implemented.
      Ceil             ✓         Ceil, Placeholder                                                          
      Floor            ✓         Floor, Placeholder                                                         
      Sin              ✓         Sin, Placeholder                                                           
      Cos              ✓         Cos, Placeholder                                                           
      Tan              ✓         Tan, Placeholder                                                           
      Sinh             ✓         Sinh, Placeholder                                                          
      Cosh             ✓         Placeholder, Cosh                                                          
      ASin             ✓         Asin, Placeholder                                                          
      ACos             ✓         Acos, Placeholder                                                          
      ATan             ✓         Atan, Placeholder                                                          
      ATan2            ✓         Atan, RealDiv, Placeholder                                                 
      ASinh            ✓         Placeholder, Asinh                                                         
      ACosh            ✓         Acosh, Placeholder                                                         
      ATanh            ✓         Atanh, Placeholder                                                         
    =================  ========  =====================================================  ====================


Array Manipulation
^^^^^^^^^^^^^^^^^^

Count 12/21
 

    =================  ========  =================================================  ================================================================================================================
     NNabla Function    Status                         TF Op                                                                          Description                                                   
    =================  ========  =================================================  ================================================================================================================
      Concatenate      ✓         ConcatV2, Const, Placeholder                                                                                                                                       
      Split            ✓         Const, SplitV, Squeeze, Placeholder                                                                                                                                
      Stack            ✓         ConcatV2, Const, Placeholder, ExpandDims                                                                                                                           
      Slice            △         Const, Placeholder, Slice                          step != 1" exceed the scope of onnx opset 9,  not supported.                                                    
      Pad              △         Const, MirrorPad, PadV2, Placeholder               When the mode of the pad is reflect, if the size of the pad exceeds the input size, tensorflow cannot handle it.
      Transpose        ✓         Const, Transpose, Placeholder                                                                                                                                      
      Broadcast        ✓                                                                                                                                                                            
      BroadcastTo      ✓                                                                                                                                                                            
      Tile             ✓         Const, Tile, Placeholder, Reshape                                                                                                                                  
      OneHot           ✓         Const, GatherV2, Placeholder, Reshape                                                                                                                              
      Flip             ✓         Const, Transpose, GatherV2, Identity, Placeholder                                                                                                                  
      Shift                                                                         Not yet implemented.                                                                                            
      Sort                                                                          Not yet implemented.                                                                                            
      Reshape          ✓         Const, Placeholder, Reshape                                                                                                                                        
      MatrixDiag                                                                    Not yet implemented.                                                                                            
      MatrixDiagPart                                                                Not yet implemented.                                                                                            
      BatchInv                                                                      Not yet implemented.                                                                                            
      BatchDet                                                                      Not yet implemented.                                                                                            
      Assign                                                                        Not yet implemented.                                                                                            
      GatherNd                                                                      Not yet implemented.                                                                                            
      ScatterNd                                                                     Not yet implemented.                                                                                            
    =================  ========  =================================================  ================================================================================================================


Signal Processing
^^^^^^^^^^^^^^^^^

Count 0/3
 

    =================  ========  =======  ====================
     NNabla Function    Status    TF Op       Description     
    =================  ========  =======  ====================
      Interpolate      X                  Not yet implemented.
      FFT                                 Not yet implemented.
      IFFT                                Not yet implemented.
    =================  ========  =======  ====================


Stochasticity
^^^^^^^^^^^^^

Count 0/12
 

    ====================  ========  ===========  ========================================================================================================================
      NNabla Function      Status      TF Op                                                           Description                                                       
    ====================  ========  ===========  ========================================================================================================================
      Dropout             X         Placeholder  The Dropout in nnabla has no test mode and contains random parameters, so the test result is not the same as tensorflow.
      TopKData                                   Not yet implemented.                                                                                                    
      TopKGrad                                   Not yet implemented.                                                                                                    
      Rand                                       Not yet implemented.                                                                                                    
      Randint                                    Not yet implemented.                                                                                                    
      Randn                                      Not yet implemented.                                                                                                    
      RandomChoice                               Not yet implemented.                                                                                                    
      RandomCrop                                 Not yet implemented.                                                                                                    
      RandomFlip                                 Not yet implemented.                                                                                                    
      RandomShift                                Not yet implemented.                                                                                                    
      RandomErase                                Not yet implemented.                                                                                                    
      ImageAugmentation                          Not yet implemented.                                                                                                    
    ====================  ========  ===========  ========================================================================================================================


Loss Functions
^^^^^^^^^^^^^^

Count 0/9
 

    ==========================  ========  =======  ====================
         NNabla Function         Status    TF Op       Description     
    ==========================  ========  =======  ====================
      SigmoidCrossEntropy                          Not yet implemented.
      BinaryCrossEntropy                           Not yet implemented.
      SoftmaxCrossEntropy                          Not yet implemented.
      CategoricalCrossEntropy                      Not yet implemented.
      SquaredError                                 Not yet implemented.
      AbsoluteError                                Not yet implemented.
      HuberLoss                                    Not yet implemented.
      EpsilonInsensitiveLoss                       Not yet implemented.
      KLMultinomial                                Not yet implemented.
    ==========================  ========  =======  ====================


Quantization Neural Network Layers
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Count 6/12
 

    ===========================  ========  ========================================================================================  ==================================================================================
          NNabla Function         Status                                            TF Op                                                                               Description                                    
    ===========================  ========  ========================================================================================  ==================================================================================
      BinarySigmoid              ✓         Const, Select, Placeholder, Greater                                                                                                                                         
      BinaryTanh                 ✓         Const, Select, Placeholder, Greater                                                                                                                                         
      BinaryConnectAffine        ✓         Const, MatMul, Add, Mul, Placeholder, Reshape                                                                                                                               
      BinaryConnectConvolution   △         Const, Split, Conv2D, Transpose, Pad, Identity, Add, ConcatV2, Placeholder, Reshape       The cases `dilations` and `strides` larger than 1 are not supported by tensorflow.
      BinaryWeightAffine         ✓         Const, MatMul, Add, Mul, Placeholder, Reshape                                                                                                                               
      BinaryWeightConvolution    △         Const, Split, Conv2D, Transpose, Pad, Identity, Add, Mul, ConcatV2, Placeholder, Reshape  The cases `dilations` and `strides` larger than 1 are not supported by tensorflow.
      INQAffine                                                                                                                      Not yet implemented.                                                              
      INQConvolution                                                                                                                 Not yet implemented.                                                              
      FixedPointQuantize                                                                                                             Not yet implemented.                                                              
      MinMaxQuantize                                                                                                                 Not yet implemented.                                                              
      Pow2Quantize                                                                                                                   Not yet implemented.                                                              
      Prune                                                                                                                          Not yet implemented.                                                              
    ===========================  ========  ========================================================================================  ==================================================================================


Validation
^^^^^^^^^^

Count 0/3
 

    ==================  ========  =======  ====================
     NNabla Function     Status    TF Op       Description     
    ==================  ========  =======  ====================
      TopNError                            Not yet implemented.
      BinaryError                          Not yet implemented.
      ConfusionMatrix                      Not yet implemented.
    ==================  ========  =======  ====================


Unsupported, Special Use
^^^^^^^^^^^^^^^^^^^^^^^^

Count 0/7
 

    =====================  ========  =======  ====================
       NNabla Function      Status    TF Op       Description     
    =====================  ========  =======  ====================
      VATNoise                                Not yet implemented.
      Unlink                                  Not yet implemented.
      Sink                                    Not yet implemented.
      NmsDetection2d                          Not yet implemented.
      MaxPoolingBackward                      Not yet implemented.
      WarpByFlow                              Not yet implemented.
      PatchCorrelation                        Not yet implemented.
    =====================  ========  =======  ====================




NNabla C Runtime Support Status
===============================


nnabla version: 1.0.21

- ✓: Supported
- △: Partially supported
- X: Supported, but test failed or no test data.
- Empty: Not support yet.


Export
------

Total: 56/181

Neural Network Layer
^^^^^^^^^^^^^^^^^^^^

Count 8/15
 

    ===============================  ========  =============
            NNabla Function           Status    Description 
    ===============================  ========  =============
      Affine                         ✓                      
      RNN                                                   
      LSTM                                                  
      GRU                                                   
      Convolution                    ✓                      
      DepthwiseConvolution           ✓                      
      Deconvolution                  ✓                      
      DepthwiseDeconvolution                                
      AdaptiveSeparableConvolution                          
      MaxPooling                     ✓                      
      AveragePooling                 △                      
      GlobalAveragePooling                                  
      SumPooling                     ✓                      
      Unpooling                      ✓                      
      Embed                                                 
    ===============================  ========  =============


Neural Network Activation Functions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Count 11/21
 

    =================  ========  =============
     NNabla Function    Status    Description 
    =================  ========  =============
      Sigmoid          ✓                      
      Swish            ✓                      
      Tanh             ✓                      
      ReLU             ✓                      
      LeakyReLU        ✓                      
      Softmax          ✓                      
      LogSoftmax                              
      ELU              ✓                      
      SELU             ✓                      
      CReLU            ✓                      
      CELU             ✓                      
      PReLU            ✓                      
      GELU                                    
      ReLU6                                   
      HardSigmoid                             
      HardTanh                                
      LogSigmoid                              
      SoftPlus                                
      SoftSign                                
      TanhShrink                              
      Sinc                                    
    =================  ========  =============


Normalization
^^^^^^^^^^^^^

Count 1/6
 

    ==========================  ========  =============
         NNabla Function         Status    Description 
    ==========================  ========  =============
      FusedBatchNormalization                          
      BatchNormalization        ✓                      
      SyncBatchNormalization                           
      MeanSubtraction           X                      
      ClipGradByValue                                  
      ClipGradByNorm                                   
    ==========================  ========  =============


Reduction
^^^^^^^^^

Count 1/7
 

    =================  ========  =============
     NNabla Function    Status    Description 
    =================  ========  =============
      Sum              ✓                      
      Mean                                    
      Max                                     
      Min                                     
      Prod                                    
      ReduceSum                               
      ReduceMean                              
    =================  ========  =============


Arithmetic
^^^^^^^^^^

Count 11/14
 

    =================  ========  =============
     NNabla Function    Status    Description 
    =================  ========  =============
      Add2             ✓                      
      AddN             X                      
      BcAdd2                                  
      Sub2             ✓                      
      Mul2             ✓                      
      MulN             X                      
      Div2             ✓                      
      Pow2             ✓                      
      AddScalar        ✓                      
      MulScalar        ✓                      
      PowScalar        ✓                      
      RSubScalar       ✓                      
      RDivScalar       ✓                      
      RPowScalar       ✓                      
    =================  ========  =============


Logical
^^^^^^^

Count 5/29
 

    =====================  ========  =============
       NNabla Function      Status    Description 
    =====================  ========  =============
      Sign                 ✓                      
      Minimum2             ✓                      
      Maximum2             ✓                      
      MinimumScalar        ✓                      
      MaximumScalar        ✓                      
      LogicalAnd                                  
      LogicalOr                                   
      LogicalXor                                  
      Equal                                       
      NotEqual                                    
      GreaterEqual                                
      Greater                                     
      LessEqual                                   
      Less                                        
      LogicalAndScalar                            
      LogicalOrScalar                             
      LogicalXorScalar                            
      EqualScalar                                 
      NotEqualScalar                              
      GreaterEqualScalar                          
      GreaterScalar                               
      LessEqualScalar                             
      LessScalar                                  
      LogicalNot                                  
      IsNaN                                       
      IsInf                                       
      ResetNaN                                    
      ResetInf                                    
      Where                                       
    =====================  ========  =============


Math
^^^^

Count 6/22
 

    =================  ========  =============
     NNabla Function    Status    Description 
    =================  ========  =============
      Constant                                
      Arange                                  
      Abs              ✓                      
      Exp              ✓                      
      Log              ✓                      
      Identity         ✓                      
      BatchMatmul      ✓                      
      Round            ✓                      
      Ceil                                    
      Floor                                   
      Sin                                     
      Cos                                     
      Tan                                     
      Sinh                                    
      Cosh                                    
      ASin                                    
      ACos                                    
      ATan                                    
      ATan2                                   
      ASinh                                   
      ACosh                                   
      ATanh                                   
    =================  ========  =============


Array Manipulation
^^^^^^^^^^^^^^^^^^

Count 7/21
 

    =================  ========  =============
     NNabla Function    Status    Description 
    =================  ========  =============
      Concatenate      ✓                      
      Split            ✓                      
      Stack            ✓                      
      Slice            ✓                      
      Pad                                     
      Transpose        ✓                      
      Broadcast                               
      BroadcastTo                             
      Tile                                    
      OneHot                                  
      Flip             ✓                      
      Shift            X                      
      Sort                                    
      Reshape          ✓                      
      MatrixDiag       X                      
      MatrixDiagPart   X                      
      BatchInv                                
      BatchDet                                
      Assign                                  
      GatherNd                                
      ScatterNd                               
    =================  ========  =============


Signal Processing
^^^^^^^^^^^^^^^^^

Count 0/3
 

    =================  ========  =============
     NNabla Function    Status    Description 
    =================  ========  =============
      Interpolate                             
      FFT                                     
      IFFT                                    
    =================  ========  =============


Stochasticity
^^^^^^^^^^^^^

Count 0/12
 

    ====================  ========  =============
      NNabla Function      Status    Description 
    ====================  ========  =============
      Dropout             X                      
      TopKData                                   
      TopKGrad                                   
      Rand                                       
      Randint                                    
      Randn                                      
      RandomChoice                               
      RandomCrop                                 
      RandomFlip                                 
      RandomShift                                
      RandomErase                                
      ImageAugmentation                          
    ====================  ========  =============


Loss Functions
^^^^^^^^^^^^^^

Count 0/9
 

    ==========================  ========  =============
         NNabla Function         Status    Description 
    ==========================  ========  =============
      SigmoidCrossEntropy                              
      BinaryCrossEntropy                               
      SoftmaxCrossEntropy                              
      CategoricalCrossEntropy                          
      SquaredError                                     
      AbsoluteError                                    
      HuberLoss                                        
      EpsilonInsensitiveLoss                           
      KLMultinomial                                    
    ==========================  ========  =============


Quantization Neural Network Layers
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Count 6/12
 

    ===========================  ========  =============
          NNabla Function         Status    Description 
    ===========================  ========  =============
      BinarySigmoid              ✓                      
      BinaryTanh                 ✓                      
      BinaryConnectAffine        ✓                      
      BinaryConnectConvolution   ✓                      
      BinaryWeightAffine         ✓                      
      BinaryWeightConvolution    ✓                      
      INQAffine                                         
      INQConvolution                                    
      FixedPointQuantize                                
      MinMaxQuantize                                    
      Pow2Quantize                                      
      Prune                                             
    ===========================  ========  =============


Validation
^^^^^^^^^^

Count 0/3
 

    ==================  ========  =============
     NNabla Function     Status    Description 
    ==================  ========  =============
      TopNError                                
      BinaryError                              
      ConfusionMatrix                          
    ==================  ========  =============


Unsupported, Special Use
^^^^^^^^^^^^^^^^^^^^^^^^

Count 0/7
 

    =====================  ========  =============
       NNabla Function      Status    Description 
    =====================  ========  =============
      VATNoise                                    
      Unlink                                      
      Sink                                        
      NmsDetection2d                              
      MaxPoolingBackward                          
      WarpByFlow                                  
      PatchCorrelation                            
    =====================  ========  =============



