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
     Add                         ✓                        ✓    ✓                          Reshape, Add2                                                                                                                                                                                                                                                                                                                  
     And                         ✓                        ✓    ✓                          Reshape, LogicalAnd                                                                                                                                                                                                                                                                                                            
     ArgMax                      ✓                        ✓                   X     ✓     Max                                                                                                                                                                                                                                                                                                                            
     ArgMin                      ✓                        ✓                   X     ✓     Min                                                                                                                                                                                                                                                                                                                            
     Asin                                                      ✓                          ASin                                                                                                                                                                                                                                                                                                                           
     Asinh                                                               ✓                ASinh                                                                                                                                                                                                                                                                                                                          
     Atan                                                      ✓                          ATan                                                                                                                                                                                                                                                                                                                           
     Atanh                                                               ✓                ATanh                                                                                                                                                                                                                                                                                                                          
     AveragePool                 ✓                        ✓    ✓              X     X     Pad, AveragePooling                                              Not all features are verified, since some features are not supported by caffe2. Those features can be verified by ONNXRuntime when opset > 6. Some feature is not supported by Nnabla such as Pad's edge mode. if opset >= 10, the ceil_mode is not supported.
     BatchNormalization          X                        X    X         ✓                BatchNormalization                                                                                                                                                                                                                                                                                                             
     BitShift                                                                       X                                                                      Not yet implemented.                                                                                                                                                                                                                                          
     Cast                        ✓                        ✓              X                Log, Abs                                                                                                                                                                                                                                                                                                                       
     Ceil                        ✓                        ✓                               Ceil                                                                                                                                                                                                                                                                                                                           
     Clip                        ✓                        ✓                         ✓     MinimumScalar, MaximumScalar, Identity                                                                                                                                                                                                                                                                                         
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
     DepthToSpace                ✓                        ✓                         ✓     Reshape, Transpose                                                                                                                                                                                                                                                                                                             
     DequantizeLinear                                                         X                                                                            Not yet implemented.                                                                                                                                                                                                                                          
     Det                                                                            X                                                                      Not yet implemented.                                                                                                                                                                                                                                          
     Div                         ✓                        ✓    ✓                          Reshape, Div2                                                                                                                                                                                                                                                                                                                  
     Dropout                     X                        X    ✓              X           Identity                                                                                                                                                                                                                                                                                                                       
     DynamicQuantizeLinear                                                          X                                                                      Not yet implemented.                                                                                                                                                                                                                                          
     Elu                         ✓                        ✓                               ELU                                                                                                                                                                                                                                                                                                                            
     Equal                       ✓                        ✓    ✓                    X     Reshape, Equal                                                                                                                                                                                                                                                                                                                 
     Erf                                                                 X                                                                                 Not yet implemented.                                                                                                                                                                                                                                          
     Exp                         ✓                        ✓                               Exp                                                                                                                                                                                                                                                                                                                            
     Expand                                                         ✓    ✓                Reshape, Broadcast                                                                                                                                                                                                                                                                                                             
     EyeLike                                                             X                                                                                 Not yet implemented.                                                                                                                                                                                                                                          
     Flatten                     ✓                        ✓              ✓          ✓     Reshape                                                                                                                                                                                                                                                                                                                        
     Floor                       ✓                        ✓                               Floor                                                                                                                                                                                                                                                                                                                          
     GRU                         X         X                   X                                                                                           Not yet implemented.                                                                                                                                                                                                                                          
     Gather                      ✓                        ✓                         ✓     Concatenate, Slice                                                                                                                                                                                                                                                                                                             
     GatherElements                                                                 X                                                                      Not yet implemented.                                                                                                                                                                                                                                          
     GatherND                                                                       X                                                                      Not yet implemented.                                                                                                                                                                                                                                          
     Gemm                        ✓                        ✓    ✓         ✓          ✓     Reshape, Add2, MulScalar, BatchMatmul                                                                                                                                                                                                                                                                                          
     GlobalAveragePool           ✓                        ✓                               GlobalAveragePooling                                                                                                                                                                                                                                                                                                           
     GlobalLpPool                X    X                                                                                                                    Not yet implemented.                                                                                                                                                                                                                                          
     GlobalMaxPool               X                                                                                                                         Not yet implemented.                                                                                                                                                                                                                                          
     Greater                     ✓                        ✓    ✓         ✓                Reshape, Greater                                                                                                                                                                                                                                                                                                               
     HardSigmoid                 ✓                        ✓                               MaximumScalar, MulScalar, AddScalar, HardSigmoid, MinimumScalar                                                                                                                                                                                                                                                                
     Hardmax                     ✓                        ✓                         ✓     Reshape, OneHot, Max                                                                                                                                                                                                                                                                                                           
     Identity                    ✓                        ✓                               Identity                                                                                                                                                                                                                                                                                                                       
     If                          X                                                                                                                         Not yet implemented.                                                                                                                                                                                                                                          
     InstanceNormalization       ✓                        ✓                               Split, BatchNormalization, Concatenate, Reshape                                                                                                                                                                                                                                                                                
     IsInf                                                                    ✓           IsInf                                                                                                                                                                                                                                                                                                                          
     IsNaN                                                               ✓                IsNaN                                                                                                                                                                                                                                                                                                                          
     LRN                         ✓                        ✓                               MulScalar, AddScalar, SumPooling, Transpose, PowScalar, Div2                                                                                                                                                                                                                                                                   
     LSTM                        X                             X                                                                                           Not yet implemented.                                                                                                                                                                                                                                          
     LeakyRelu                   ✓                        ✓                               LeakyReLU                                                                                                                                                                                                                                                                                                                      
     Less                        ✓                        ✓    ✓         ✓                Reshape, Less                                                                                                                                                                                                                                                                                                                  
     Log                         ✓                        ✓                               Log                                                                                                                                                                                                                                                                                                                            
     LogSoftmax                  ✓                        ✓                         ✓     Reshape, Add2, Log, Exp, Sum, Max, Sub2                                                                                                                                                                                                                                                                                        
     Loop                        X                                                  X                                                                      Not yet implemented.                                                                                                                                                                                                                                          
     LpNormalization             X                                                                                                                         Not yet implemented.                                                                                                                                                                                                                                          
     LpPool                      X    X                                             X                                                                      Not yet implemented.                                                                                                                                                                                                                                          
     MatMul                      ✓                        ✓              ✓                Reshape, BatchMatmul                                                                                                                                                                                                                                                                                                           
     MatMulInteger                                                            X                                                                            Not yet implemented.                                                                                                                                                                                                                                          
     Max                         ✓                        ✓         ✓    ✓                Maximum2                                                                                                                                                                                                                                                                                                                       
     MaxPool                     ✓                        ✓         X         X     X     Pad, MaxPooling                                                  Not all features are verified, since some features are not supported by caffe2. Those features can be verified by ONNXRuntime. if opset >= 10, the ceil_mode is not supported, dilations is not equal to 1 is not supported.                                  
     MaxRoiPool                  X                                                                                                                         Not yet implemented.                                                                                                                                                                                                                                          
     MaxUnpool                                                           X          X                                                                      Not yet implemented.                                                                                                                                                                                                                                          
     Mean                        ✓                        ✓         ✓    ✓                Mean, Broadcast, Stack                                                                                                                                                                                                                                                                                                         
     MeanVarianceNormalization                                           X                                                                                 Not yet implemented.                                                                                                                                                                                                                                          
     Min                         ✓                        ✓         ✓    ✓                Minimum2                                                                                                                                                                                                                                                                                                                       
     Mod                                                                      X                                                                            Not yet implemented.                                                                                                                                                                                                                                          
     Mul                         ✓                        ✓    ✓                          Reshape, Mul2                                                                                                                                                                                                                                                                                                                  
     Multinomial                                               X                                                                                           Not yet implemented.                                                                                                                                                                                                                                          
     Neg                         ✓                        ✓                               MulScalar                                                                                                                                                                                                                                                                                                                      
     NonMaxSuppression                                                        X     X                                                                      Not yet implemented.                                                                                                                                                                                                                                          
     NonZero                                                             X                                                                                 Not yet implemented.                                                                                                                                                                                                                                          
     Not                         ✓                        ✓                               LogicalNot                                                                                                                                                                                                                                                                                                                     
     OneHot                                                              X          X                                                                      Not yet implemented.                                                                                                                                                                                                                                          
     Or                          ✓                        ✓    ✓                          Reshape, LogicalOr                                                                                                                                                                                                                                                                                                             
     PRelu                       ✓                        ✓    X         X                PReLU                                                                                                                                                                                                                                                                                                                          
     Pad                         ✓    ✓                   ✓                         ✓     Pad                                                              Onnx required to support "edge" mode, while nnabla does not support it.                                                                                                                                                                                       
     Pow                         ✓                        ✓    ✓                          Reshape, Pow2                                                                                                                                                                                                                                                                                                                  
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
     Softmax                     ✓                        ✓                         ✓     Reshape, Exp, Sum, Max, Sub2, Div2                                                                                                                                                                                                                                                                                             
     Softplus                    ✓                        ✓                               SoftPlus                                                                                                                                                                                                                                                                                                                       
     Softsign                    ✓                        ✓                               SoftSign                                                                                                                                                                                                                                                                                                                       
     SpaceToDepth                ✓                        ✓                               Reshape, Transpose                                                                                                                                                                                                                                                                                                             
     Split                       ✓    ✓                   ✓                         ✓     Split, Stack                                                                                                                                                                                                                                                                                                                   
     SplitToSequence                                                                X                                                                      Not yet implemented.                                                                                                                                                                                                                                          
     Sqrt                        ✓                        ✓                               PowScalar                                                                                                                                                                                                                                                                                                                      
     Squeeze                     ✓                        ✓                         ✓     Reshape                                                                                                                                                                                                                                                                                                                        
     StringNormalizer                                                         X                                                                            Not yet implemented.                                                                                                                                                                                                                                          
     Sub                         ✓                        ✓    ✓                          Reshape, Sub2                                                                                                                                                                                                                                                                                                                  
     Sum                         ✓                        ✓         ✓    ✓                Add2                                                                                                                                                                                                                                                                                                                           
     Tan                                                       ✓                          Tan                                                                                                                                                                                                                                                                                                                            
     Tanh                        ✓                        ✓                               Tanh                                                                                                                                                                                                                                                                                                                           
     TfIdfVectorizer                                                     X                                                                                 Not yet implemented.                                                                                                                                                                                                                                          
     ThresholdedRelu                                                          ✓           Constant, Where, GreaterScalar                                                                                                                                                                                                                                                                                                 
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

Total: 120/173

Neural Network Layer
^^^^^^^^^^^^^^^^^^^^

Count 11/14
 

    =========================  ===  ===  ===  ====  ====  ========================================  ======================================================================================
         NNabla Function        6    7    9    10    11                   ONNX Op                                                        Description                                      
    =========================  ===  ===  ===  ====  ====  ========================================  ======================================================================================
      Affine                   ✓    ✓    ✓    ✓     ✓     Reshape, Gemm                                                                                                                   
      RNN                                                                                           Not yet implemented.                                                                  
      LSTM                                                                                          Not yet implemented.                                                                  
      GRU                                                                                           Not yet implemented.                                                                  
      Convolution              ✓    ✓    ✓    ✓     ✓     Reshape, Conv                                                                                                                   
      DepthwiseConvolution     ✓    ✓    ✓    ✓     ✓     Reshape, Conv                                                                                                                   
      Deconvolution            ✓    ✓    ✓    ✓     ✓     Reshape, ConvTranspose                                                                                                          
      DepthwiseDeconvolution   ✓    ✓    ✓    ✓     ✓     Reshape, ConvTranspose                                                                                                          
      MaxPooling               ✓    ✓    ✓    ✓     X     Reshape, Pad, MaxPool                                                                                                           
      AveragePooling           △    △    △    △     X     Reshape, Pad, AveragePool                 Currently only supports the cases where both ignore_border and including_pad are True.
      GlobalAveragePooling     ✓    ✓    ✓    ✓     ✓     GlobalAveragePool                                                                                                               
      SumPooling               X    ✓    ✓    ✓     X     Reshape, Constant, Pad, Mul, AveragePool                                                                                        
      Unpooling                △    ✓    ✓    ✓     ✓     Resize                                    The kernel only supports 2d on opset 6.                                               
      Embed                    ✓    ✓    ✓    ✓     ✓     Gather                                                                                                                          
    =========================  ===  ===  ===  ====  ====  ========================================  ======================================================================================


Neural Network Activation Functions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Count 21/21
 

    =================  ===  ===  ===  ====  ====  ========================================  =================================================================================
     NNabla Function    6    7    9    10    11                   ONNX Op                                                      Description                                   
    =================  ===  ===  ===  ====  ====  ========================================  =================================================================================
      Sigmoid          ✓    ✓    ✓    ✓     ✓     Sigmoid                                                                                                                    
      Swish            ✓    ✓    ✓    ✓     ✓     Mul, Sigmoid                                                                                                               
      Tanh             ✓    ✓    ✓    ✓     ✓     Tanh                                                                                                                       
      ReLU             ✓    ✓    ✓    ✓     ✓     Relu                                                                                                                       
      LeakyReLU        ✓    ✓    ✓    ✓     ✓     LeakyRelu                                                                                                                  
      Softmax          △    ✓    ✓    ✓     ✓     ReduceMax, ReduceSum, Div, Exp, Sub       ONNX Add, Sub operator does not support multidirectional broadcasting on opset 6.
      LogSoftmax       △    ✓    ✓    ✓     ✓     Log, ReduceMax, ReduceSum, Exp, Sub                                                                                        
      ELU              ✓    ✓    ✓    ✓     ✓     Elu                                                                                                                        
      SELU             ✓    ✓    ✓    ✓     ✓     Selu                                                                                                                       
      CReLU            ✓    ✓    ✓    ✓     ✓     Neg, Relu, Concat                                                                                                          
      CELU             ✓    ✓    ✓    ✓     ✓     Neg, Elu, Concat                                                                                                           
      PReLU            ✓    ✓    ✓    ✓     ✓     Reshape, PRelu                                                                                                             
      GELU             ✓    ✓    ✓    ✓     ✓     Mul, Constant, Tanh, Div, Pow, Add, Sqrt                                                                                   
      ReLU6            ✓    ✓    ✓    ✓     ✓     Min, Relu, Constant                                                                                                        
      HardSigmoid      ✓    ✓    ✓    ✓     ✓     HardSigmoid                                                                                                                
      HardTanh         ✓    ✓    ✓    ✓     ✓     Neg, Min, Max, Constant                                                                                                    
      LogSigmoid       ✓    ✓    ✓    ✓     ✓     Log, Sigmoid                                                                                                               
      SoftPlus         ✓    ✓    ✓    ✓     ✓     Softplus                                                                                                                   
      SoftSign         ✓    ✓    ✓    ✓     ✓     Softsign                                                                                                                   
      TanhShrink       ✓    ✓    ✓    ✓     ✓     Tanh, Sub                                                                                                                  
      Sinc             X    X    ✓    ✓     ✓     Constant, Sin, Div, Equal, Where                                                                                           
    =================  ===  ===  ===  ====  ====  ========================================  =================================================================================


Normalization
^^^^^^^^^^^^^

Count 2/6
 

    ==========================  ===  ===  ===  ====  ====  ======================================================================================  ====================
         NNabla Function         6    7    9    10    11                                          ONNX Op                                              Description     
    ==========================  ===  ===  ===  ====  ====  ======================================================================================  ====================
      FusedBatchNormalization   ✓    ✓    ✓    ✓     ✓     Mul, Constant, Reshape, ReduceSum, Div, Add, Sub, BatchNormalization, ReduceMean, Relu                      
      BatchNormalization        ✓    ✓    ✓    ✓     ✓     Reshape, Constant, Mul, ReduceSum, Div, Sub, ReduceMean, BatchNormalization                                 
      SyncBatchNormalization                                                                                                                       Not yet implemented.
      MeanSubtraction                                                                                                                              Not yet implemented.
      ClipGradByValue                                                                                                                              Not yet implemented.
      ClipGradByNorm                                                                                                                               Not yet implemented.
    ==========================  ===  ===  ===  ====  ====  ======================================================================================  ====================


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

Count 11/12
 

    =================  ===  ===  ===  ====  ====  =============  ============================================================================
     NNabla Function    6    7    9    10    11      ONNX Op                                     Description                                 
    =================  ===  ===  ===  ====  ====  =============  ============================================================================
      Add2             △    ✓    ✓    ✓     ✓     Add            ONNX Add operator does not support multidirectional broadcasting on opset 6.
      BcAdd2                                                     Not yet implemented.                                                        
      Sub2             △    ✓    ✓    ✓     ✓     Sub            ONNX Sub operator does not support multidirectional broadcasting on opset 6.
      Mul2             △    ✓    ✓    ✓     ✓     Mul            ONNX Mul operator does not support multidirectional broadcasting on opset 6.
      Div2             △    ✓    ✓    ✓     ✓     Div            ONNX Div operator does not support multidirectional broadcasting on opset 6.
      Pow2             △    ✓    ✓    ✓     ✓     Pow            ONNX Pow operator does not support multidirectional broadcasting on opset 6.
      AddScalar        ✓    ✓    ✓    ✓     ✓     Constant, Add                                                                              
      MulScalar        ✓    ✓    ✓    ✓     ✓     Mul, Constant                                                                              
      PowScalar        ✓    ✓    ✓    ✓     ✓     Constant, Pow                                                                              
      RSubScalar       ✓    ✓    ✓    ✓     ✓     Constant, Sub                                                                              
      RDivScalar       ✓    ✓    ✓    ✓     ✓     Div, Constant                                                                              
      RPowScalar       ✓    ✓    ✓    ✓     ✓     Constant, Pow                                                                              
    =================  ===  ===  ===  ====  ====  =============  ============================================================================


Logical
^^^^^^^

Count 29/29
 

    =====================  ===  ===  ===  ====  ====  ======================  ============================================================================
       NNabla Function      6    7    9    10    11          ONNX Op                                          Description                                 
    =====================  ===  ===  ===  ====  ====  ======================  ============================================================================
      Sign                 X    X    ✓    ✓     ✓     Sign                                                                                                
      Minimum2             △    ✓    ✓    ✓     ✓     Min, Constant, Add      ONNX Add operator does not support multidirectional broadcasting on opset 6.
      Maximum2             △    ✓    ✓    ✓     ✓     Constant, Max, Add      ONNX Add operator does not support multidirectional broadcasting on opset 6.
      MinimumScalar        ✓    ✓    ✓    ✓     ✓     Min, Constant, Add                                                                                  
      MaximumScalar        ✓    ✓    ✓    ✓     ✓     Constant, Max, Add                                                                                  
      LogicalAnd           ✓    ✓    ✓    ✓     ✓     And                                                                                                 
      LogicalOr            ✓    ✓    ✓    ✓     ✓     Or                                                                                                  
      LogicalXor           ✓    ✓    ✓    ✓     ✓     Xor                                                                                                 
      Equal                ✓    ✓    ✓    ✓     ✓     Equal                                                                                               
      NotEqual             ✓    ✓    ✓    ✓     ✓     Equal, Not                                                                                          
      GreaterEqual         ✓    ✓    ✓    ✓     ✓     Less, Not                                                                                           
      Greater              ✓    ✓    ✓    ✓     ✓     Greater                                                                                             
      LessEqual            ✓    ✓    ✓    ✓     ✓     Not, Greater                                                                                        
      Less                 ✓    ✓    ✓    ✓     ✓     Less                                                                                                
      LogicalAndScalar     ✓    ✓    ✓    ✓     ✓     Constant, And                                                                                       
      LogicalOrScalar      ✓    ✓    ✓    ✓     ✓     Constant, Or                                                                                        
      LogicalXorScalar     ✓    ✓    ✓    ✓     ✓     Xor, Constant                                                                                       
      EqualScalar          ✓    ✓    ✓    ✓     ✓     Equal, Constant                                                                                     
      NotEqualScalar       ✓    ✓    ✓    ✓     ✓     Equal, Not, Constant                                                                                
      GreaterEqualScalar   ✓    ✓    ✓    ✓     ✓     Constant, Less, Not                                                                                 
      GreaterScalar        ✓    ✓    ✓    ✓     ✓     Constant, Greater                                                                                   
      LessEqualScalar      ✓    ✓    ✓    ✓     ✓     Constant, Not, Greater                                                                              
      LessScalar           ✓    ✓    ✓    ✓     ✓     Constant, Less                                                                                      
      LogicalNot           ✓    ✓    ✓    ✓     ✓     Not                                                                                                 
      IsNaN                X    X    ✓    ✓     ✓     IsNaN                                                                                               
      IsInf                X    X    X    ✓     ✓     IsInf                                                                                               
      ResetNaN             X    X    ✓    ✓     ✓     IsNaN, Constant, Where                                                                              
      ResetInf             X    X    X    ✓     ✓     IsInf, Where, Constant                                                                              
      Where                X    X    ✓    ✓     ✓     Where                                                                                               
    =====================  ===  ===  ===  ====  ====  ======================  ============================================================================


Math
^^^^

Count 22/22
 

    =================  ===  ===  ===  ====  ====  ==================  =============
     NNabla Function    6    7    9    10    11        ONNX Op         Description 
    =================  ===  ===  ===  ====  ====  ==================  =============
      Constant         ✓    ✓    ✓    ✓     ✓     Constant, Identity               
      Arange           ✓    ✓    ✓    ✓     ✓     Constant, Identity               
      Abs              ✓    ✓    ✓    ✓     ✓     Abs                              
      Exp              ✓    ✓    ✓    ✓     ✓     Exp                              
      Log              ✓    ✓    ✓    ✓     ✓     Log                              
      Identity         ✓    ✓    ✓    ✓     ✓     Identity                         
      BatchMatmul      ✓    ✓    ✓    ✓     ✓     Transpose, MatMul                
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
    =================  ===  ===  ===  ====  ====  ==================  =============


Array Manipulation
^^^^^^^^^^^^^^^^^^

Count 12/19
 

    =================  ===  ===  ===  ====  ====  ===========================  ============================================================================================================================
     NNabla Function    6    7    9    10    11             ONNX Op                                                                    Description                                                         
    =================  ===  ===  ===  ====  ====  ===========================  ============================================================================================================================
      Concatenate      ✓    ✓    ✓    ✓     ✓     Concat                                                                                                                                                   
      Split            ✓    ✓    ✓    ✓     ✓     Split, Squeeze                                                                                                                                           
      Stack            ✓    ✓    ✓    ✓     ✓     Unsqueeze, Concat                                                                                                                                        
      Slice            △    △    △    △     △     Constant, Slice              ONNX slice cannot support step != 1 on opset < 10.                                                                          
      Pad              △    △    △    △     △     Constant, Pad                When the mode of the pad is reflect, if the size of the pad exceeds the input size, caffe2 and onnxruntime cannot handle it.
      Transpose        ✓    ✓    ✓    ✓     ✓     Transpose                                                                                                                                                
      Broadcast        X    X    ✓    ✓     ✓                                                                                                                                                              
      BroadcastTo      ✓    ✓    ✓    ✓     ✓                                                                                                                                                              
      Tile             ✓    ✓    ✓    ✓     ✓     Reshape, Constant, Tile                                                                                                                                  
      OneHot           ✓    ✓    ✓    ✓     ✓     Reshape, Gather, Flatten                                                                                                                                 
      Flip             ✓    ✓    ✓    ✓     ✓     Gather, Transpose, Identity                                                                                                                              
      Shift                                                                    Not yet implemented.                                                                                                        
      Sort                                                                     Not yet implemented.                                                                                                        
      Reshape          ✓    ✓    ✓    ✓     ✓     Reshape, Constant                                                                                                                                        
      MatrixDiag                                                               Not yet implemented.                                                                                                        
      MatrixDiagPart                                                           Not yet implemented.                                                                                                        
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
      Interpolate      X    X    X    △     ✓     Reshape, Resize                      
      FFT                                                          Not yet implemented.
      IFFT                                                         Not yet implemented.
    =================  ===  ===  ===  ====  ====  ===============  ====================


Stochasticity
^^^^^^^^^^^^^

Count 0/11
 

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
      BinaryConnectAffine        ✓    ✓    ✓    ✓     ✓     Reshape, Gemm                                  
      BinaryConnectConvolution   ✓    ✓    ✓    ✓     ✓     Reshape, Conv                                  
      BinaryWeightAffine         ✓    ✓    ✓    ✓     ✓     Reshape, Mul, MatMul, Add                      
      BinaryWeightConvolution    ✓    ✓    ✓    ✓     ✓     Reshape, Conv, Mul, Add                        
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

Count 0/5
 

    =====================  ===  ===  ===  ====  ====  =========  ====================
       NNabla Function      6    7    9    10    11    ONNX Op       Description     
    =====================  ===  ===  ===  ====  ====  =========  ====================
      VATNoise                                                   Not yet implemented.
      Unlink                                                     Not yet implemented.
      Sink                                                       Not yet implemented.
      NmsDetection2d                                             Not yet implemented.
      MaxPoolingBackward                                         Not yet implemented.
    =====================  ===  ===  ===  ====  ====  =========  ====================





Tensorflow Support Status
=========================

Import
------

- ✓: Supported
- △: Partially supported
- X: Supported, but test failed.
- Empty: Not support yet.


Total: 86/120

.. table:: Tensorflow support status

    ======================================  ========  =================================  ===========================================================================================
             Tensorflow Function             Status              NNabla Func                                                     Description                                        
    ======================================  ========  =================================  ===========================================================================================
      Abs                                      ✓      Abs                                                                                                                           
      Acos                                     ✓      ACos                                                                                                                          
      Acosh                                    ✓      ACosh                                                                                                                         
      Add                                      ✓      Add2                                                                                                                          
      AddN                                     ✓      Add2                                                                                                                          
      All                                                                                Not yet implemented.                                                                       
      Any                                                                                Not yet implemented.                                                                       
      ArgMax                                   ✓      Max                                                                                                                           
      ArgMin                                   ✓      Min                                                                                                                           
      Asin                                     ✓      ASin                                                                                                                          
      Asinh                                    ✓      ASinh                                                                                                                         
      Atan                                     ✓      ATan                                                                                                                          
      Atan2                                                                              Not yet implemented.                                                                       
      Atanh                                    ✓      ATanh                                                                                                                         
      AvgPool                                  △      Pad, Transpose, AveragePooling     Some feature is not supported by Nnabla such as Pad's edge mode.                           
      AvgPool3D                                                                          Not yet implemented.                                                                       
      BatchMatMul                              ✓      Transpose, BatchMatmul                                                                                                        
      BiasAdd                                  ✓      Reshape, Add2                                                                                                                 
      Cast                                                                               Not yet implemented.                                                                       
      Ceil                                     ✓      Ceil                                                                                                                          
      ConcatV2                                 ✓      Concatenate                                                                                                                   
      Const                                    ✓      Add2                                                                                                                          
      Conv2D                                   △      Convolution, Pad, Transpose        Tensorflow require GPU to perform related test cases. This issue is recorded only for memo.
      Conv2DBackpropFilter                                                               Not yet implemented.                                                                       
      Conv2DBackpropInput                      △      Deconvolution, Transpose           Tensorflow require GPU to perform related test cases. This issue is recorded only for memo.
      Conv3D                                                                             Not yet implemented.                                                                       
      Conv3DBackpropFilterV2                                                             Not yet implemented.                                                                       
      Conv3DBackpropInputV2                                                              Not yet implemented.                                                                       
      Cos                                      ✓      Cos                                                                                                                           
      Cosh                                     ✓      Cosh                                                                                                                          
      DepthToSpace                             △      Reshape, Transpose                 Tensorflow require GPU to perform related test cases. This issue is recorded only for memo.
      DepthwiseConv2dNative                                                              Not yet implemented.                                                                       
      DepthwiseConv2dNativeBackpropFilter                                                Not yet implemented.                                                                       
      DepthwiseConv2dNativeBackpropInput                                                 Not yet implemented.                                                                       
      Div                                      ✓      Div2                                                                                                                          
      Elu                                      ✓      ELU                                                                                                                           
      Equal                                    ✓      Equal                                                                                                                         
      Erf                                                                                Not yet implemented.                                                                       
      Erfc                                                                               Not yet implemented.                                                                       
      Exp                                      ✓      Exp                                                                                                                           
      ExpandDims                               ✓      Reshape                                                                                                                       
      Fill                                                                               Not yet implemented.                                                                       
      Flatten                                  ✓      Reshape                                                                                                                       
      Floor                                    ✓      Floor                                                                                                                         
      FloorDiv                                 ✓      Floor, Div2                                                                                                                   
      FloorMod                                 ✓      Floor, Mul2, Sub2, Div2                                                                                                       
      FusedBatchNorm                           △      BatchNormalization, Transpose      It did not pass testing for training mode.                                                 
      GatherNd                                                                           Not yet implemented.                                                                       
      GatherV2                                                                           Not yet implemented.                                                                       
      Greater                                  ✓      Greater                                                                                                                       
      GreaterEqual                             ✓      LogicalNot, Less                                                                                                              
      Identity                                 ✓      Identity                                                                                                                      
      IsInf                                                                              Not yet implemented.                                                                       
      IsNan                                    ✓      IsNaN                                                                                                                         
      LeakyRelu                                ✓      LeakyReLU                                                                                                                     
      Less                                     ✓      Less                                                                                                                          
      LessEqual                                ✓      LogicalNot, Greater                                                                                                           
      Log                                      ✓      Log                                                                                                                           
      LogSoftmax                                                                         Not yet implemented.                                                                       
      LogicalAnd                               ✓      LogicalAnd                                                                                                                    
      LogicalNot                               ✓      LogicalNot                                                                                                                    
      LogicalOr                                ✓      LogicalOr                                                                                                                     
      LogicalXor                               ✓      LogicalOr, LogicalNot, LogicalAnd                                                                                             
      MatrixBandPart                                                                     Not yet implemented.                                                                       
      Max                                      ✓      Max                                                                                                                           
      MaxPool                                  ✓      Pad, Transpose, MaxPooling                                                                                                    
      MaxPool3D                                                                          Not yet implemented.                                                                       
      MaxPoolWithArgmax                                                                  Not yet implemented.                                                                       
      Maximum                                  ✓      Maximum2                                                                                                                      
      Mean                                     ✓      Mean                                                                                                                          
      Min                                      ✓      Min                                                                                                                           
      Minimum                                  ✓      Minimum2                                                                                                                      
      Mul                                      ✓      Mul2                                                                                                                          
      Neg                                      ✓      MulScalar                                                                                                                     
      NotEqual                                 ✓      Equal, LogicalNot                                                                                                             
      OneHot                                                                             Not yet implemented.                                                                       
      Pack                                     ✓      Reshape, Concatenate                                                                                                          
      Pad                                      ✓      Pad                                                                                                                           
      Pow                                      ✓      Pow2                                                                                                                          
      Prod                                     ✓      Prod                                                                                                                          
      RandomShuffle                                                                      Not yet implemented.                                                                       
      RandomStandardNormal                                                               Not yet implemented.                                                                       
      RandomUniform                                                                      Not yet implemented.                                                                       
      RealDiv                                  ✓      Div2                                                                                                                          
      Reciprocal                               ✓      RDivScalar                                                                                                                    
      Relu                                     ✓      ReLU                                                                                                                          
      Relu6                                    ✓      MinimumScalar, MaximumScalar                                                                                                  
      Reshape                                  ✓      Reshape                                                                                                                       
      ReverseSequence                                                                    Not yet implemented.                                                                       
      Rsqrt                                    ✓      PowScalar, RDivScalar                                                                                                         
      Select                                                                             Not yet implemented.                                                                       
      Selu                                     ✓      SELU                                                                                                                          
      Shape                                                                              Not yet implemented.                                                                       
      Sigmoid                                  ✓      Sigmoid                                                                                                                       
      Sign                                     ✓      Sign                                                                                                                          
      Sin                                      ✓      Sin                                                                                                                           
      Sinh                                     ✓      Sinh                                                                                                                          
      Size                                                                               Not yet implemented.                                                                       
      Slice                                    ✓      Slice                                                                                                                         
      Softmax                                                                            Not yet implemented.                                                                       
      Softplus                                 ✓      SoftPlus                                                                                                                      
      Softsign                                 ✓      SoftSign                                                                                                                      
      SpaceToDepth                             △      Reshape, Transpose                 Tensorflow require GPU to perform related test cases. This issue is recorded only for memo.
      SplitV                                   ✓      Split, Concatenate, Stack                                                                                                     
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
      TopKV2                                                                             Not yet implemented.                                                                       
      Transpose                                ✓      Transpose                                                                                                                     
      TruncateDiv                                                                        Not yet implemented.                                                                       
      TruncateMod                                                                        Not yet implemented.                                                                       
      Unpack                                   ✓      Split, Concatenate, Stack                                                                                                     
    ======================================  ========  =================================  ===========================================================================================





Export
------

- ✓: Supported
- △: Partially supported
- X: Supported, but test failed.
- Empty: Not support yet.


Total: 116/173

Neural Network Layer
^^^^^^^^^^^^^^^^^^^^

Count 11/14
 

    =========================  ========  ================================================================================================================================================================================================  ==================================================================================
         NNabla Function        Status                                                                                                TF Op                                                                                                                                   Description                                    
    =========================  ========  ================================================================================================================================================================================================  ==================================================================================
      Affine                   ✓         Reshape, Mul, GatherV2, Squeeze, Equal, Placeholder, MatMul, SparseToDense, Where, Const, AddV2, Cast                                                                                                                                                                               
      RNN                                                                                                                                                                                                                                  Not yet implemented.                                                              
      LSTM                                                                                                                                                                                                                                 Not yet implemented.                                                              
      GRU                                                                                                                                                                                                                                  Not yet implemented.                                                              
      Convolution              △         Reshape, Split, Pad, BatchToSpaceND, GatherV2, Squeeze, SpaceToBatchND, Equal, ConcatV2, Placeholder, SparseToDense, Add, Where, Transpose, Identity, Const, AddV2, Cast, Conv2D                  The cases `dilations` and `strides` larger than 1 are not supported by tensorflow.
      DepthwiseConvolution     △         Reshape, Split, Pad, BatchToSpaceND, GatherV2, Squeeze, Equal, Placeholder, ConcatV2, SpaceToBatchND, Add, SparseToDense, Where, Transpose, Const, AddV2, Cast, Conv2D                            The cases `dilations` and `strides` larger than 1 are not supported by tensorflow.
      Deconvolution            △         Reshape, Split, Pad, GatherV2, Squeeze, Equal, Placeholder, ConcatV2, SparseToDense, Add, Where, Transpose, Slice, Identity, Conv2DBackpropInput, Const, AddV2, Cast                              The cases `dilations` larger than 1 are not supported by tensorflow.              
      DepthwiseDeconvolution   △         Reshape, Split, Pad, GatherV2, Squeeze, Equal, Placeholder, ConcatV2, SparseToDense, Add, Where, Transpose, Slice, Conv2DBackpropInput, Const, AddV2, Cast                                        The cases `dilations` larger than 1 are not supported by tensorflow.              
      MaxPooling               ✓         Reshape, GatherV2, Squeeze, Equal, Placeholder, SparseToDense, MaxPool3D, PadV2, Where, MaxPool, Transpose, Const, AddV2, Cast                                                                                                                                                      
      AveragePooling           △         Reshape, Pad, GatherV2, Squeeze, Equal, Placeholder, SparseToDense, AvgPool3D, Where, Transpose, Const, AddV2, Cast, AvgPool                                                                      Currently only supports the cases both ignore_border and including_pad are True.  
      GlobalAveragePooling     ✓         SplitV, Mean, Pack, Sub, Range, Const                                                                                                                                                                                                                                               
      SumPooling               ✓         Reshape, Mul, Pad, GatherV2, Squeeze, Equal, Placeholder, SparseToDense, AvgPool3D, Where, Transpose, Const, AddV2, Cast, AvgPool                                                                                                                                                   
      Unpooling                △         Switch, Mul, Assert, Reshape, ResizeNearestNeighbor, GatherV2, Equal, Merge, Placeholder, Squeeze, StridedSlice, NoOp, SparseToDense, Where, Transpose, Identity, Const, AddV2, Cast, LogicalAnd  The kernel only supports 2d.                                                      
      Embed                    ✓         GatherV2, Const, Placeholder                                                                                                                                                                                                                                                        
    =========================  ========  ================================================================================================================================================================================================  ==================================================================================


Neural Network Activation Functions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Count 21/21
 

    =================  ========  =============================================================================================================  =============
     NNabla Function    Status                                                       TF Op                                                       Description 
    =================  ========  =============================================================================================================  =============
      Sigmoid          ✓         Placeholder, Sigmoid                                                                                                        
      Swish            ✓         Mul, Placeholder, Sigmoid                                                                                                   
      Tanh             ✓         Tanh, Placeholder                                                                                                           
      ReLU             ✓         Placeholder, Relu                                                                                                           
      LeakyReLU        ✓         Placeholder, LeakyRelu                                                                                                      
      Softmax          ✓         Placeholder, Exp, Sum, Sub, Max, Const, RealDiv                                                                             
      LogSoftmax       ✓         Log, Placeholder, Exp, Sum, Sub, Max, Const                                                                                 
      ELU              ✓         Mul, Placeholder, Elu, Exp, GreaterEqual, Sub, Const, AddV2, Cast, Less                                                     
      SELU             △         Mul, Minimum, Min, Placeholder, Exp, Maximum, Sub, Max, Const, AddV2                                                        
      CReLU            ✓         Placeholder, ConcatV2, Neg, Const, Relu                                                                                     
      CELU             ✓         Mul, Placeholder, Elu, ConcatV2, Exp, Neg, GreaterEqual, Sub, Const, AddV2, Cast, Less                                      
      PReLU            ✓         Reshape, Mul, GatherV2, Squeeze, Equal, Placeholder, SparseToDense, Where, Sub, Const, AddV2, Relu, Cast, Abs               
      GELU             ✓         Mul, Tanh, Placeholder, Pow, Add, Sqrt, Const, RealDiv                                                                      
      ReLU6            ✓         Min, Pack, Placeholder, Const, Relu                                                                                         
      HardSigmoid      ✓         Mul, Minimum, Placeholder, Add, Maximum, Const                                                                              
      HardTanh         ✓         Min, Pack, Placeholder, Neg, Max, Const                                                                                     
      LogSigmoid       ✓         Placeholder, Sigmoid, Log                                                                                                   
      SoftPlus         ✓         Placeholder, Softplus                                                                                                       
      SoftSign         ✓         Placeholder, Softsign                                                                                                       
      TanhShrink       ✓         Tanh, Placeholder, Sub                                                                                                      
      Sinc             ✓         Sin, Equal, Placeholder, Const, Select, RealDiv                                                                             
    =================  ========  =============================================================================================================  =============


Normalization
^^^^^^^^^^^^^

Count 2/6
 

    ==========================  ========  ========================================================================================================================================  ====================
         NNabla Function         Status                                                                    TF Op                                                                        Description     
    ==========================  ========  ========================================================================================================================================  ====================
      FusedBatchNormalization   ✓         Reshape, Mul, Mean, GatherV2, Placeholder, Squeeze, Equal, Rsqrt, Add, SparseToDense, Sum, Where, Sub, Const, AddV2, Relu, Cast, RealDiv                      
      BatchNormalization        ✓         Reshape, Mul, GatherV2, Squeeze, Equal, Placeholder, Mean, Rsqrt, SparseToDense, Sum, Where, Sub, Const, AddV2, Cast, RealDiv                                 
      SyncBatchNormalization                                                                                                                                                        Not yet implemented.
      MeanSubtraction                                                                                                                                                               Not yet implemented.
      ClipGradByValue                                                                                                                                                               Not yet implemented.
      ClipGradByNorm                                                                                                                                                                Not yet implemented.
    ==========================  ========  ========================================================================================================================================  ====================


Reduction
^^^^^^^^^

Count 5/7
 

    =================  ========  ========================  ====================
     NNabla Function    Status            TF Op                Description     
    =================  ========  ========================  ====================
      Sum              ✓         Const, Placeholder, Sum                       
      Mean             ✓         Mean, Const, Placeholder                      
      Max              ✓         Const, Placeholder, Max                       
      Min              ✓         Const, Placeholder, Min                       
      Prod             ✓         Prod, Const, Placeholder                      
      ReduceSum                                            Not yet implemented.
      ReduceMean                                           Not yet implemented.
    =================  ========  ========================  ====================


Arithmetic
^^^^^^^^^^

Count 11/12
 

    =================  ========  ===========================  ====================
     NNabla Function    Status              TF Op                 Description     
    =================  ========  ===========================  ====================
      Add2             ✓         Placeholder, Add                                 
      BcAdd2                                                  Not yet implemented.
      Sub2             ✓         Placeholder, Sub                                 
      Mul2             ✓         Mul, Placeholder                                 
      Div2             ✓         Placeholder, RealDiv                             
      Pow2             ✓         Placeholder, Pow                                 
      AddScalar        ✓         Const, Placeholder, Add                          
      MulScalar        ✓         Mul, Const, Placeholder                          
      PowScalar        ✓         Const, Placeholder, Pow                          
      RSubScalar       ✓         Const, Placeholder, Sub                          
      RDivScalar       ✓         Const, Placeholder, RealDiv                      
      RPowScalar       ✓         Const, Placeholder, Pow                          
    =================  ========  ===========================  ====================


Logical
^^^^^^^

Count 27/29
 

    =====================  ========  =====================================================  ====================
       NNabla Function      Status                           TF Op                              Description     
    =====================  ========  =====================================================  ====================
      Sign                 ✓         Placeholder, Sign                                                          
      Minimum2             ✓         Min, Pack, Placeholder, Add, Const                                         
      Maximum2             ✓         Pack, Placeholder, Add, Max, Const                                         
      MinimumScalar        ✓         Min, Pack, Placeholder, Add, Const                                         
      MaximumScalar        ✓         Pack, Placeholder, Add, Max, Const                                         
      LogicalAnd           ✓         Placeholder, LogicalAnd                                                    
      LogicalOr            ✓         Placeholder, LogicalOr                                                     
      LogicalXor           ✓         Placeholder, LogicalOr, LogicalNot, LogicalAnd                             
      Equal                ✓         Equal, Placeholder                                                         
      NotEqual             ✓         Equal, Placeholder, LogicalNot                                             
      GreaterEqual         ✓         LogicalNot, Placeholder, Less                                              
      Greater              ✓         Placeholder, Greater                                                       
      LessEqual            ✓         Placeholder, LogicalNot, Greater                                           
      Less                 ✓         Placeholder, Less                                                          
      LogicalAndScalar     ✓         Const, Placeholder, LogicalAnd                                             
      LogicalOrScalar      ✓         Const, Placeholder, LogicalOr                                              
      LogicalXorScalar     ✓         LogicalOr, Placeholder, LogicalNot, Const, LogicalAnd                      
      EqualScalar          ✓         Const, Placeholder, Equal                                                  
      NotEqualScalar       ✓         Const, Placeholder, LogicalNot, Equal                                      
      GreaterEqualScalar   ✓         LogicalNot, Const, Placeholder, Less                                       
      GreaterScalar        ✓         Const, Placeholder, Greater                                                
      LessEqualScalar      ✓         Const, Placeholder, LogicalNot, Greater                                    
      LessScalar           ✓         Const, Placeholder, Less                                                   
      LogicalNot           ✓         Placeholder, LogicalNot                                                    
      IsNaN                ✓         IsNan, Placeholder                                                         
      IsInf                X                                                                Not yet implemented.
      ResetNaN             ✓         Const, IsNan, Select, Placeholder                                          
      ResetInf             X                                                                Not yet implemented.
      Where                ✓         Placeholder, Select                                                        
    =====================  ========  =====================================================  ====================


Math
^^^^

Count 21/22
 

    =================  ========  ============================================  ====================
     NNabla Function    Status                      TF Op                          Description     
    =================  ========  ============================================  ====================
      Constant         ✓         Const, Identity                                                   
      Arange           ✓         Const, Identity                                                   
      Abs              ✓         Placeholder, Abs                                                  
      Exp              ✓         Placeholder, Exp                                                  
      Log              ✓         Placeholder, Log                                                  
      Identity         ✓         Placeholder, Identity                                             
      BatchMatmul      ✓         Const, BatchMatMulV2, Transpose, Placeholder                      
      Round            X                                                       Not yet implemented.
      Ceil             ✓         Placeholder, Ceil                                                 
      Floor            ✓         Placeholder, Floor                                                
      Sin              ✓         Placeholder, Sin                                                  
      Cos              ✓         Cos, Placeholder                                                  
      Tan              ✓         Placeholder, Tan                                                  
      Sinh             ✓         Placeholder, Sinh                                                 
      Cosh             ✓         Placeholder, Cosh                                                 
      ASin             ✓         Placeholder, Asin                                                 
      ACos             ✓         Placeholder, Acos                                                 
      ATan             ✓         Atan, Placeholder                                                 
      ATan2            ✓         Atan, Placeholder, RealDiv                                        
      ASinh            ✓         Placeholder, Asinh                                                
      ACosh            ✓         Acosh, Placeholder                                                
      ATanh            ✓         Placeholder, Atanh                                                
    =================  ========  ============================================  ====================


Array Manipulation
^^^^^^^^^^^^^^^^^^

Count 12/19
 

    =================  ========  ==============================================================================================  ================================================================================================================
     NNabla Function    Status                                               TF Op                                                                                                 Description                                                   
    =================  ========  ==============================================================================================  ================================================================================================================
      Concatenate      ✓         Const, Placeholder, ConcatV2                                                                                                                                                                                    
      Split            ✓         Const, Placeholder, SplitV, Squeeze                                                                                                                                                                             
      Stack            ✓         Const, Placeholder, ConcatV2, ExpandDims                                                                                                                                                                        
      Slice            △         Const, Placeholder, Slice                                                                       step != 1" exceed the scope of onnx opset 9,  not supported.                                                    
      Pad              △         Const, PadV2, MirrorPad, Placeholder                                                            When the mode of the pad is reflect, if the size of the pad exceeds the input size, tensorflow cannot handle it.
      Transpose        ✓         Const, Placeholder, Transpose                                                                                                                                                                                   
      Broadcast        ✓                                                                                                                                                                                                                         
      BroadcastTo      ✓                                                                                                                                                                                                                         
      Tile             ✓         Reshape, GatherV2, Squeeze, Placeholder, Equal, SparseToDense, Where, Const, AddV2, Cast, Tile                                                                                                                  
      OneHot           ✓         Reshape, GatherV2, Squeeze, Equal, Placeholder, SparseToDense, Where, Const, AddV2, Cast                                                                                                                        
      Flip             ✓         GatherV2, Placeholder, Transpose, Identity, Const                                                                                                                                                               
      Shift                                                                                                                      Not yet implemented.                                                                                            
      Sort                                                                                                                       Not yet implemented.                                                                                            
      Reshape          ✓         Reshape, GatherV2, Squeeze, Equal, Placeholder, SparseToDense, Where, Const, AddV2, Cast                                                                                                                        
      MatrixDiag                                                                                                                 Not yet implemented.                                                                                            
      MatrixDiagPart                                                                                                             Not yet implemented.                                                                                            
      Assign                                                                                                                     Not yet implemented.                                                                                            
      GatherNd                                                                                                                   Not yet implemented.                                                                                            
      ScatterNd                                                                                                                  Not yet implemented.                                                                                            
    =================  ========  ==============================================================================================  ================================================================================================================


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

Count 0/11
 

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
 

    ===========================  ========  =====================================================================================================================================================  ==================================================================================
          NNabla Function         Status                                                                           TF Op                                                                                                             Description                                    
    ===========================  ========  =====================================================================================================================================================  ==================================================================================
      BinarySigmoid              ✓         Const, Placeholder, Select, Greater                                                                                                                                                                                                      
      BinaryTanh                 ✓         Const, Placeholder, Select, Greater                                                                                                                                                                                                      
      BinaryConnectAffine        ✓         Reshape, Mul, GatherV2, Squeeze, Equal, Placeholder, MatMul, SparseToDense, Where, Const, AddV2, Cast                                                                                                                                    
      BinaryConnectConvolution   △         Reshape, Split, Pad, GatherV2, Squeeze, Equal, Placeholder, ConcatV2, Add, SparseToDense, Where, Transpose, Identity, Const, AddV2, Cast, Conv2D       The cases `dilations` and `strides` larger than 1 are not supported by tensorflow.
      BinaryWeightAffine         ✓         Reshape, Mul, GatherV2, Squeeze, Equal, Placeholder, MatMul, SparseToDense, Add, Where, Const, AddV2, Cast                                                                                                                               
      BinaryWeightConvolution    △         Reshape, Mul, Pad, Split, GatherV2, Squeeze, Equal, Placeholder, ConcatV2, Add, SparseToDense, Where, Transpose, Identity, Const, AddV2, Cast, Conv2D  The cases `dilations` and `strides` larger than 1 are not supported by tensorflow.
      INQAffine                                                                                                                                                                                   Not yet implemented.                                                              
      INQConvolution                                                                                                                                                                              Not yet implemented.                                                              
      FixedPointQuantize                                                                                                                                                                          Not yet implemented.                                                              
      MinMaxQuantize                                                                                                                                                                              Not yet implemented.                                                              
      Pow2Quantize                                                                                                                                                                                Not yet implemented.                                                              
      Prune                                                                                                                                                                                       Not yet implemented.                                                              
    ===========================  ========  =====================================================================================================================================================  ==================================================================================


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

Count 0/5
 

    =====================  ========  =======  ====================
       NNabla Function      Status    TF Op       Description     
    =====================  ========  =======  ====================
      VATNoise                                Not yet implemented.
      Unlink                                  Not yet implemented.
      Sink                                    Not yet implemented.
      NmsDetection2d                          Not yet implemented.
      MaxPoolingBackward                      Not yet implemented.
    =====================  ========  =======  ====================




Tensorflow Lite Support Status
==============================


Export
------

- ✓: Supported
- △: Partially supported
- X: Supported, but test failed.
- Empty: Not support yet.


Total: 74/173

Neural Network Layer
^^^^^^^^^^^^^^^^^^^^

Count 2/14
 

    =========================  ========
         NNabla Function        Status 
    =========================  ========
      Affine                   X       
      RNN                              
      LSTM                             
      GRU                              
      Convolution              X       
      DepthwiseConvolution     X       
      Deconvolution            X       
      DepthwiseDeconvolution   X       
      MaxPooling               X       
      AveragePooling           X       
      GlobalAveragePooling     X       
      SumPooling               X       
      Unpooling                △       
      Embed                    △       
    =========================  ========


Neural Network Activation Functions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Count 18/21
 

    =================  ========
     NNabla Function    Status 
    =================  ========
      Sigmoid          ✓       
      Swish            ✓       
      Tanh             ✓       
      ReLU             ✓       
      LeakyReLU        ✓       
      Softmax          ✓       
      LogSoftmax       ✓       
      ELU              ✓       
      SELU             △       
      CReLU            ✓       
      CELU             ✓       
      PReLU            X       
      GELU             ✓       
      ReLU6            ✓       
      HardSigmoid      ✓       
      HardTanh         ✓       
      LogSigmoid       ✓       
      SoftPlus         X       
      SoftSign         X       
      TanhShrink       ✓       
      Sinc             ✓       
    =================  ========


Normalization
^^^^^^^^^^^^^

Count 1/6
 

    ==========================  ========
         NNabla Function         Status 
    ==========================  ========
      FusedBatchNormalization   X       
      BatchNormalization        △       
      SyncBatchNormalization            
      MeanSubtraction                   
      ClipGradByValue                   
      ClipGradByNorm                    
    ==========================  ========


Reduction
^^^^^^^^^

Count 0/7
 

    =================  ========
     NNabla Function    Status 
    =================  ========
      Sum              X       
      Mean             X       
      Max              X       
      Min              X       
      Prod             X       
      ReduceSum                
      ReduceMean               
    =================  ========


Arithmetic
^^^^^^^^^^

Count 11/12
 

    =================  ========
     NNabla Function    Status 
    =================  ========
      Add2             ✓       
      BcAdd2                   
      Sub2             ✓       
      Mul2             ✓       
      Div2             ✓       
      Pow2             ✓       
      AddScalar        ✓       
      MulScalar        ✓       
      PowScalar        ✓       
      RSubScalar       ✓       
      RDivScalar       ✓       
      RPowScalar       ✓       
    =================  ========


Logical
^^^^^^^

Count 24/29
 

    =====================  ========
       NNabla Function      Status 
    =====================  ========
      Sign                 X       
      Minimum2             ✓       
      Maximum2             ✓       
      MinimumScalar        ✓       
      MaximumScalar        ✓       
      LogicalAnd           ✓       
      LogicalOr            ✓       
      LogicalXor           ✓       
      Equal                ✓       
      NotEqual             ✓       
      GreaterEqual         ✓       
      Greater              ✓       
      LessEqual            ✓       
      Less                 ✓       
      LogicalAndScalar     ✓       
      LogicalOrScalar      ✓       
      LogicalXorScalar     ✓       
      EqualScalar          ✓       
      NotEqualScalar       ✓       
      GreaterEqualScalar   ✓       
      GreaterScalar        ✓       
      LessEqualScalar      ✓       
      LessScalar           ✓       
      LogicalNot           ✓       
      IsNaN                X       
      IsInf                X       
      ResetNaN             X       
      ResetInf             X       
      Where                ✓       
    =====================  ========


Math
^^^^

Count 8/22
 

    =================  ========
     NNabla Function    Status 
    =================  ========
      Constant         X       
      Arange           X       
      Abs              ✓       
      Exp              ✓       
      Log              ✓       
      Identity         ✓       
      BatchMatmul      X       
      Round            X       
      Ceil             ✓       
      Floor            ✓       
      Sin              ✓       
      Cos              ✓       
      Tan              X       
      Sinh             X       
      Cosh             X       
      ASin             X       
      ACos             X       
      ATan             X       
      ATan2            X       
      ASinh            X       
      ACosh            X       
      ATanh            X       
    =================  ========


Array Manipulation
^^^^^^^^^^^^^^^^^^

Count 8/19
 

    =================  ========
     NNabla Function    Status 
    =================  ========
      Concatenate      ✓       
      Split            ✓       
      Stack            ✓       
      Slice            △       
      Pad              △       
      Transpose        △       
      Broadcast        △       
      BroadcastTo      X       
      Tile             X       
      OneHot           X       
      Flip             ✓       
      Shift                    
      Sort                     
      Reshape          X       
      MatrixDiag               
      MatrixDiagPart           
      Assign                   
      GatherNd                 
      ScatterNd                
    =================  ========


Signal Processing
^^^^^^^^^^^^^^^^^

Count 0/3
 

    =================  ========
     NNabla Function    Status 
    =================  ========
      Interpolate      X       
      FFT                      
      IFFT                     
    =================  ========


Stochasticity
^^^^^^^^^^^^^

Count 0/11
 

    ====================  ========
      NNabla Function      Status 
    ====================  ========
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
      ImageAugmentation           
    ====================  ========


Loss Functions
^^^^^^^^^^^^^^

Count 0/9
 

    ==========================  ========
         NNabla Function         Status 
    ==========================  ========
      SigmoidCrossEntropy               
      BinaryCrossEntropy                
      SoftmaxCrossEntropy               
      CategoricalCrossEntropy           
      SquaredError                      
      AbsoluteError                     
      HuberLoss                         
      EpsilonInsensitiveLoss            
      KLMultinomial                     
    ==========================  ========


Quantization Neural Network Layers
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Count 2/12
 

    ===========================  ========
          NNabla Function         Status 
    ===========================  ========
      BinarySigmoid              ✓       
      BinaryTanh                 ✓       
      BinaryConnectAffine        X       
      BinaryConnectConvolution   X       
      BinaryWeightAffine         X       
      BinaryWeightConvolution    X       
      INQAffine                          
      INQConvolution                     
      FixedPointQuantize                 
      MinMaxQuantize                     
      Pow2Quantize                       
      Prune                              
    ===========================  ========


Validation
^^^^^^^^^^

Count 0/3
 

    ==================  ========
     NNabla Function     Status 
    ==================  ========
      TopNError                 
      BinaryError               
      ConfusionMatrix           
    ==================  ========


Unsupported, Special Use
^^^^^^^^^^^^^^^^^^^^^^^^

Count 0/5
 

    =====================  ========
       NNabla Function      Status 
    =====================  ========
      VATNoise                     
      Unlink                       
      Sink                         
      NmsDetection2d               
      MaxPoolingBackward           
    =====================  ========




NNabla C Runtime Support Status
===============================


NNabla version: None

- ✓: Supported
- △: Partially supported
- X: Supported, but test failed or no test data.
- Empty: Not support yet.


Export
------

Total: 56/173

Neural Network Layer
^^^^^^^^^^^^^^^^^^^^

Count 8/14
 

    =========================  ========  =============
         NNabla Function        Status    Description 
    =========================  ========  =============
      Affine                   ✓                      
      RNN                                             
      LSTM                                            
      GRU                                             
      Convolution              ✓                      
      DepthwiseConvolution     ✓                      
      Deconvolution            ✓                      
      DepthwiseDeconvolution                          
      MaxPooling               ✓                      
      AveragePooling           ✓                      
      GlobalAveragePooling                            
      SumPooling               ✓                      
      Unpooling                ✓                      
      Embed                                           
    =========================  ========  =============


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

Count 11/12
 

    =================  ========  =============
     NNabla Function    Status    Description 
    =================  ========  =============
      Add2             ✓                      
      BcAdd2                                  
      Sub2             ✓                      
      Mul2             ✓                      
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
      BatchMatmul      △                      
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

Count 7/19
 

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

Count 0/11
 

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

Count 0/5
 

    =====================  ========  =============
       NNabla Function      Status    Description 
    =====================  ========  =============
      VATNoise                                    
      Unlink                                      
      Sink                                        
      NmsDetection2d                              
      MaxPoolingBackward                          
    =====================  ========  =============



