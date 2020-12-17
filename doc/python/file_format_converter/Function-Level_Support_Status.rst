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
     AveragePool                 ✓                        ✓    ✓              X     X     AveragePooling, Pad                                              Not all features are verified, since some features are not supported by caffe2. Those features can be verified by ONNXRuntime when opset > 6. Some feature is not supported by Nnabla such as Pad's edge mode. if opset >= 10, the ceil_mode is not supported.
     BatchNormalization          X                        X    X         ✓                BatchNormalization                                                                                                                                                                                                                                                                                                             
     BitShift                                                                       X                                                                      Not yet implemented.                                                                                                                                                                                                                                          
     Cast                        ✓                        ✓              X                Abs, Log                                                                                                                                                                                                                                                                                                                       
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
     Expand                                                         ✓    ✓                Reshape, Broadcast                                                                                                                                                                                                                                                                                                             
     EyeLike                                                             X                                                                                 Not yet implemented.                                                                                                                                                                                                                                          
     Flatten                     ✓                        ✓              ✓          ✓     Reshape                                                                                                                                                                                                                                                                                                                        
     Floor                       ✓                        ✓                               Floor                                                                                                                                                                                                                                                                                                                          
     GRU                         X         X                   X                                                                                           Not yet implemented.                                                                                                                                                                                                                                          
     Gather                      ✓                        ✓                         ✓     Slice, Concatenate                                                                                                                                                                                                                                                                                                             
     GatherElements                                                                 X                                                                      Not yet implemented.                                                                                                                                                                                                                                          
     GatherND                                                                       X                                                                      Not yet implemented.                                                                                                                                                                                                                                          
     Gemm                        ✓                        ✓    ✓         ✓          ✓     BatchMatmul, Add2, Reshape, MulScalar                                                                                                                                                                                                                                                                                          
     GlobalAveragePool           ✓                        ✓                               GlobalAveragePooling                                                                                                                                                                                                                                                                                                           
     GlobalLpPool                X    X                                                                                                                    Not yet implemented.                                                                                                                                                                                                                                          
     GlobalMaxPool               X                                                                                                                         Not yet implemented.                                                                                                                                                                                                                                          
     Greater                     ✓                        ✓    ✓         ✓                Greater, Reshape                                                                                                                                                                                                                                                                                                               
     HardSigmoid                 ✓                        ✓                               HardSigmoid, MinimumScalar, MulScalar, MaximumScalar, AddScalar                                                                                                                                                                                                                                                                
     Hardmax                     ✓                        ✓                         ✓     Reshape, Max, OneHot                                                                                                                                                                                                                                                                                                           
     Identity                    ✓                        ✓                               Identity                                                                                                                                                                                                                                                                                                                       
     If                          X                                                                                                                         Not yet implemented.                                                                                                                                                                                                                                          
     InstanceNormalization       ✓                        ✓                               BatchNormalization, Split, Reshape, Concatenate                                                                                                                                                                                                                                                                                
     IsInf                                                                    ✓           IsInf                                                                                                                                                                                                                                                                                                                          
     IsNaN                                                               ✓                IsNaN                                                                                                                                                                                                                                                                                                                          
     LRN                         ✓                        ✓                               Transpose, MulScalar, Div2, SumPooling, PowScalar, AddScalar                                                                                                                                                                                                                                                                   
     LSTM                        X                             X                                                                                           Not yet implemented.                                                                                                                                                                                                                                          
     LeakyRelu                   ✓                        ✓                               LeakyReLU                                                                                                                                                                                                                                                                                                                      
     Less                        ✓                        ✓    ✓         ✓                Reshape, Less                                                                                                                                                                                                                                                                                                                  
     Log                         ✓                        ✓                               Log                                                                                                                                                                                                                                                                                                                            
     LogSoftmax                  ✓                        ✓                         ✓     Exp, Add2, Max, Sum, Sub2, Log, Reshape                                                                                                                                                                                                                                                                                        
     Loop                        X                                                  X                                                                      Not yet implemented.                                                                                                                                                                                                                                          
     LpNormalization             X                                                                                                                         Not yet implemented.                                                                                                                                                                                                                                          
     LpPool                      X    X                                             X                                                                      Not yet implemented.                                                                                                                                                                                                                                          
     MatMul                      ✓                        ✓              ✓                BatchMatmul, Reshape                                                                                                                                                                                                                                                                                                           
     MatMulInteger                                                            X                                                                            Not yet implemented.                                                                                                                                                                                                                                          
     Max                         ✓                        ✓         ✓    ✓                Maximum2                                                                                                                                                                                                                                                                                                                       
     MaxPool                     ✓                        ✓         X         X     X     Pad, MaxPooling                                                  Not all features are verified, since some features are not supported by caffe2. Those features can be verified by ONNXRuntime. if opset >= 10, the ceil_mode is not supported, dilations is not equal to 1 is not supported.                                  
     MaxRoiPool                  X                                                                                                                         Not yet implemented.                                                                                                                                                                                                                                          
     MaxUnpool                                                           X          X                                                                      Not yet implemented.                                                                                                                                                                                                                                          
     Mean                        ✓                        ✓         ✓    ✓                Mean, Stack, Broadcast                                                                                                                                                                                                                                                                                                         
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
     ReduceSumSquare             ✓                        ✓                         ✓     PowScalar, Sum                                                                                                                                                                                                                                                                                                                 
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
     Softmax                     ✓                        ✓                         ✓     Exp, Max, Div2, Sum, Sub2, Reshape                                                                                                                                                                                                                                                                                             
     Softplus                    ✓                        ✓                               SoftPlus                                                                                                                                                                                                                                                                                                                       
     Softsign                    ✓                        ✓                               SoftSign                                                                                                                                                                                                                                                                                                                       
     SpaceToDepth                ✓                        ✓                               Transpose, Reshape                                                                                                                                                                                                                                                                                                             
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
      Deconvolution            ✓    ✓    ✓    ✓     ✓     ConvTranspose, Reshape                                                                                                          
      DepthwiseDeconvolution   ✓    ✓    ✓    ✓     ✓     ConvTranspose, Reshape                                                                                                          
      MaxPooling               ✓    ✓    ✓    ✓     X     MaxPool, Reshape, Pad                                                                                                           
      AveragePooling           △    △    △    △     X     Reshape, Pad, AveragePool                 Currently only supports the cases where both ignore_border and including_pad are True.
      GlobalAveragePooling     ✓    ✓    ✓    ✓     ✓     GlobalAveragePool                                                                                                               
      SumPooling               X    ✓    ✓    ✓     X     AveragePool, Constant, Mul, Reshape, Pad                                                                                        
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
      Softmax          △    ✓    ✓    ✓     ✓     Exp, ReduceMax, Sub, Div, ReduceSum       ONNX Add, Sub operator does not support multidirectional broadcasting on opset 6.
      LogSoftmax       △    ✓    ✓    ✓     ✓     Exp, ReduceMax, Sub, Log, ReduceSum                                                                                        
      ELU              ✓    ✓    ✓    ✓     ✓     Elu                                                                                                                        
      SELU             ✓    ✓    ✓    ✓     ✓     Selu                                                                                                                       
      CReLU            ✓    ✓    ✓    ✓     ✓     Neg, Concat, Relu                                                                                                          
      CELU             ✓    ✓    ✓    ✓     ✓     Elu, Neg, Concat                                                                                                           
      PReLU            ✓    ✓    ✓    ✓     ✓     Reshape, PRelu                                                                                                             
      GELU             ✓    ✓    ✓    ✓     ✓     Sqrt, Add, Pow, Div, Tanh, Mul, Constant                                                                                   
      ReLU6            ✓    ✓    ✓    ✓     ✓     Min, Constant, Relu                                                                                                        
      HardSigmoid      ✓    ✓    ✓    ✓     ✓     HardSigmoid                                                                                                                
      HardTanh         ✓    ✓    ✓    ✓     ✓     Min, Constant, Max, Neg                                                                                                    
      LogSigmoid       ✓    ✓    ✓    ✓     ✓     Sigmoid, Log                                                                                                               
      SoftPlus         ✓    ✓    ✓    ✓     ✓     Softplus                                                                                                                   
      SoftSign         ✓    ✓    ✓    ✓     ✓     Softsign                                                                                                                   
      TanhShrink       ✓    ✓    ✓    ✓     ✓     Sub, Tanh                                                                                                                  
      Sinc             X    X    ✓    ✓     ✓     Sin, Equal, Div, Constant, Where                                                                                           
    =================  ===  ===  ===  ====  ====  ========================================  =================================================================================


Normalization
^^^^^^^^^^^^^

Count 2/6
 

    ==========================  ===  ===  ===  ====  ====  ======================================================================================  ====================
         NNabla Function         6    7    9    10    11                                          ONNX Op                                              Description     
    ==========================  ===  ===  ===  ====  ====  ======================================================================================  ====================
      FusedBatchNormalization   ✓    ✓    ✓    ✓     ✓     ReduceMean, Reshape, Add, BatchNormalization, Sub, Div, Relu, Mul, ReduceSum, Constant                      
      BatchNormalization        ✓    ✓    ✓    ✓     ✓     ReduceMean, BatchNormalization, Constant, Sub, Div, Mul, ReduceSum, Reshape                                 
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
      AddScalar        ✓    ✓    ✓    ✓     ✓     Add, Constant                                                                              
      MulScalar        ✓    ✓    ✓    ✓     ✓     Mul, Constant                                                                              
      PowScalar        ✓    ✓    ✓    ✓     ✓     Constant, Pow                                                                              
      RSubScalar       ✓    ✓    ✓    ✓     ✓     Constant, Sub                                                                              
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
      Minimum2             △    ✓    ✓    ✓     ✓     Min, Add, Constant      ONNX Add operator does not support multidirectional broadcasting on opset 6.
      Maximum2             △    ✓    ✓    ✓     ✓     Add, Constant, Max      ONNX Add operator does not support multidirectional broadcasting on opset 6.
      MinimumScalar        ✓    ✓    ✓    ✓     ✓     Min, Add, Constant                                                                                  
      MaximumScalar        ✓    ✓    ✓    ✓     ✓     Add, Constant, Max                                                                                  
      LogicalAnd           ✓    ✓    ✓    ✓     ✓     And                                                                                                 
      LogicalOr            ✓    ✓    ✓    ✓     ✓     Or                                                                                                  
      LogicalXor           ✓    ✓    ✓    ✓     ✓     Xor                                                                                                 
      Equal                ✓    ✓    ✓    ✓     ✓     Equal                                                                                               
      NotEqual             ✓    ✓    ✓    ✓     ✓     Equal, Not                                                                                          
      GreaterEqual         ✓    ✓    ✓    ✓     ✓     Not, Less                                                                                           
      Greater              ✓    ✓    ✓    ✓     ✓     Greater                                                                                             
      LessEqual            ✓    ✓    ✓    ✓     ✓     Greater, Not                                                                                        
      Less                 ✓    ✓    ✓    ✓     ✓     Less                                                                                                
      LogicalAndScalar     ✓    ✓    ✓    ✓     ✓     And, Constant                                                                                       
      LogicalOrScalar      ✓    ✓    ✓    ✓     ✓     Or, Constant                                                                                        
      LogicalXorScalar     ✓    ✓    ✓    ✓     ✓     Xor, Constant                                                                                       
      EqualScalar          ✓    ✓    ✓    ✓     ✓     Equal, Constant                                                                                     
      NotEqualScalar       ✓    ✓    ✓    ✓     ✓     Not, Equal, Constant                                                                                
      GreaterEqualScalar   ✓    ✓    ✓    ✓     ✓     Not, Constant, Less                                                                                 
      GreaterScalar        ✓    ✓    ✓    ✓     ✓     Greater, Constant                                                                                   
      LessEqualScalar      ✓    ✓    ✓    ✓     ✓     Not, Greater, Constant                                                                              
      LessScalar           ✓    ✓    ✓    ✓     ✓     Constant, Less                                                                                      
      LogicalNot           ✓    ✓    ✓    ✓     ✓     Not                                                                                                 
      IsNaN                X    X    ✓    ✓     ✓     IsNaN                                                                                               
      IsInf                X    X    X    ✓     ✓     IsInf                                                                                               
      ResetNaN             X    X    ✓    ✓     ✓     Constant, IsNaN, Where                                                                              
      ResetInf             X    X    X    ✓     ✓     IsInf, Constant, Where                                                                              
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
      ATan2            X    ✓    ✓    ✓     ✓     Div, Atan                        
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
      Split            ✓    ✓    ✓    ✓     ✓     Squeeze, Split                                                                                                                                           
      Stack            ✓    ✓    ✓    ✓     ✓     Unsqueeze, Concat                                                                                                                                        
      Slice            △    △    △    △     △     Slice, Constant              ONNX slice cannot support step != 1 on opset < 10.                                                                          
      Pad              △    △    △    △     △     Constant, Pad                When the mode of the pad is reflect, if the size of the pad exceeds the input size, caffe2 and onnxruntime cannot handle it.
      Transpose        ✓    ✓    ✓    ✓     ✓     Transpose                                                                                                                                                
      Broadcast        X    X    ✓    ✓     ✓                                                                                                                                                              
      BroadcastTo      ✓    ✓    ✓    ✓     ✓                                                                                                                                                              
      Tile             ✓    ✓    ✓    ✓     ✓     Tile, Constant, Reshape                                                                                                                                  
      OneHot           ✓    ✓    ✓    ✓     ✓     Gather, Reshape, Flatten                                                                                                                                 
      Flip             ✓    ✓    ✓    ✓     ✓     Transpose, Gather, Identity                                                                                                                              
      Shift                                                                    Not yet implemented.                                                                                                        
      Sort                                                                     Not yet implemented.                                                                                                        
      Reshape          ✓    ✓    ✓    ✓     ✓     Constant, Reshape                                                                                                                                        
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
      BinarySigmoid              X    X    ✓    ✓     ✓     Greater, Constant, Where                       
      BinaryTanh                 X    X    ✓    ✓     ✓     Greater, Constant, Where                       
      BinaryConnectAffine        ✓    ✓    ✓    ✓     ✓     Reshape, Gemm                                  
      BinaryConnectConvolution   ✓    ✓    ✓    ✓     ✓     Reshape, Conv                                  
      BinaryWeightAffine         ✓    ✓    ✓    ✓     ✓     Mul, Add, MatMul, Reshape                      
      BinaryWeightConvolution    ✓    ✓    ✓    ✓     ✓     Mul, Add, Reshape, Conv                        
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
      AvgPool                                  △      Transpose, AveragePooling, Pad     Some feature is not supported by Nnabla such as Pad's edge mode.                           
      AvgPool3D                                                                          Not yet implemented.                                                                       
      BatchMatMul                              ✓      Transpose, BatchMatmul                                                                                                        
      BiasAdd                                  ✓      Add2, Reshape                                                                                                                 
      Cast                                                                               Not yet implemented.                                                                       
      Ceil                                     ✓      Ceil                                                                                                                          
      ConcatV2                                 ✓      Concatenate                                                                                                                   
      Const                                    ✓      Add2                                                                                                                          
      Conv2D                                   △      Transpose, Convolution, Pad        Tensorflow require GPU to perform related test cases. This issue is recorded only for memo.
      Conv2DBackpropFilter                                                               Not yet implemented.                                                                       
      Conv2DBackpropInput                      △      Transpose, Deconvolution           Tensorflow require GPU to perform related test cases. This issue is recorded only for memo.
      Conv3D                                                                             Not yet implemented.                                                                       
      Conv3DBackpropFilterV2                                                             Not yet implemented.                                                                       
      Conv3DBackpropInputV2                                                              Not yet implemented.                                                                       
      Cos                                      ✓      Cos                                                                                                                           
      Cosh                                     ✓      Cosh                                                                                                                          
      DepthToSpace                             △      Transpose, Reshape                 Tensorflow require GPU to perform related test cases. This issue is recorded only for memo.
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
      FloorMod                                 ✓      Floor, Sub2, Div2, Mul2                                                                                                       
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
      LogicalXor                               ✓      LogicalAnd, LogicalNot, LogicalOr                                                                                             
      MatrixBandPart                                                                     Not yet implemented.                                                                       
      Max                                      ✓      Max                                                                                                                           
      MaxPool                                  ✓      Transpose, Pad, MaxPooling                                                                                                    
      MaxPool3D                                                                          Not yet implemented.                                                                       
      MaxPoolWithArgmax                                                                  Not yet implemented.                                                                       
      Maximum                                  ✓      Maximum2                                                                                                                      
      Mean                                     ✓      Mean                                                                                                                          
      Min                                      ✓      Min                                                                                                                           
      Minimum                                  ✓      Minimum2                                                                                                                      
      Mul                                      ✓      Mul2                                                                                                                          
      Neg                                      ✓      MulScalar                                                                                                                     
      NotEqual                                 ✓      LogicalNot, Equal                                                                                                             
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
      Relu6                                    ✓      MaximumScalar, MinimumScalar                                                                                                  
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
      SpaceToDepth                             △      Transpose, Reshape                 Tensorflow require GPU to perform related test cases. This issue is recorded only for memo.
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
      Affine                   ✓         Equal, SparseToDense, AddV2, MatMul, Cast, Placeholder, Squeeze, Const, Mul, GatherV2, Reshape, Where                                                                                                                                                                               
      RNN                                                                                                                                                                                                                                  Not yet implemented.                                                              
      LSTM                                                                                                                                                                                                                                 Not yet implemented.                                                              
      GRU                                                                                                                                                                                                                                  Not yet implemented.                                                              
      Convolution              △         Transpose, Equal, SparseToDense, BatchToSpaceND, Identity, Conv2D, AddV2, Add, Split, Cast, SpaceToBatchND, Placeholder, Squeeze, ConcatV2, Const, GatherV2, Reshape, Pad, Where                  The cases `dilations` and `strides` larger than 1 are not supported by tensorflow.
      DepthwiseConvolution     △         Transpose, Equal, SparseToDense, BatchToSpaceND, Conv2D, AddV2, Add, Split, Cast, SpaceToBatchND, Placeholder, Squeeze, ConcatV2, Const, GatherV2, Reshape, Pad, Where                            The cases `dilations` and `strides` larger than 1 are not supported by tensorflow.
      Deconvolution            △         Transpose, Equal, SparseToDense, Identity, AddV2, Add, Split, Cast, Placeholder, Squeeze, Slice, Conv2DBackpropInput, ConcatV2, Const, GatherV2, Reshape, Pad, Where                              The cases `dilations` larger than 1 are not supported by tensorflow.              
      DepthwiseDeconvolution   △         Transpose, Equal, SparseToDense, AddV2, Add, Split, Cast, Placeholder, Squeeze, Slice, Conv2DBackpropInput, ConcatV2, Const, GatherV2, Reshape, Pad, Where                                        The cases `dilations` larger than 1 are not supported by tensorflow.              
      MaxPooling               ✓         Transpose, Equal, SparseToDense, MaxPool, AddV2, Cast, Placeholder, PadV2, Squeeze, MaxPool3D, Const, GatherV2, Reshape, Where                                                                                                                                                      
      AveragePooling           △         Transpose, Equal, SparseToDense, AddV2, AvgPool3D, Cast, Placeholder, AvgPool, Squeeze, Const, GatherV2, Reshape, Pad, Where                                                                      Currently only supports the cases both ignore_border and including_pad are True.  
      GlobalAveragePooling     ✓         SplitV, Range, Pack, Sub, Const, Mean                                                                                                                                                                                                                                               
      SumPooling               ✓         Transpose, Equal, SparseToDense, AddV2, AvgPool3D, Cast, Placeholder, AvgPool, Squeeze, Const, Mul, GatherV2, Reshape, Pad, Where                                                                                                                                                   
      Unpooling                △         Transpose, ResizeNearestNeighbor, Equal, SparseToDense, Identity, Reshape, LogicalAnd, AddV2, Cast, Assert, Merge, Placeholder, Squeeze, Const, Mul, GatherV2, Switch, NoOp, StridedSlice, Where  The kernel only supports 2d.                                                      
      Embed                    ✓         GatherV2, Const, Placeholder                                                                                                                                                                                                                                                        
    =========================  ========  ================================================================================================================================================================================================  ==================================================================================


Neural Network Activation Functions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Count 21/21
 

    =================  ========  =============================================================================================================  =============
     NNabla Function    Status                                                       TF Op                                                       Description 
    =================  ========  =============================================================================================================  =============
      Sigmoid          ✓         Sigmoid, Placeholder                                                                                                        
      Swish            ✓         Mul, Sigmoid, Placeholder                                                                                                   
      Tanh             ✓         Tanh, Placeholder                                                                                                           
      ReLU             ✓         Relu, Placeholder                                                                                                           
      LeakyReLU        ✓         LeakyRelu, Placeholder                                                                                                      
      Softmax          ✓         Exp, Max, Sum, Placeholder, RealDiv, Sub, Const                                                                             
      LogSoftmax       ✓         Exp, Max, Sum, Placeholder, Sub, Const, Log                                                                                 
      ELU              ✓         Exp, Elu, AddV2, Cast, GreaterEqual, Placeholder, Sub, Const, Mul, Less                                                     
      SELU             △         Exp, Max, Min, AddV2, Minimum, Maximum, Placeholder, Sub, Const, Mul                                                        
      CReLU            ✓         Neg, Placeholder, ConcatV2, Const, Relu                                                                                     
      CELU             ✓         Exp, Elu, AddV2, Neg, GreaterEqual, Cast, Placeholder, Sub, ConcatV2, Const, Mul, Less                                      
      PReLU            ✓         Equal, SparseToDense, AddV2, Cast, Placeholder, Abs, Squeeze, Sub, Const, Relu, Mul, GatherV2, Reshape, Where               
      GELU             ✓         Sqrt, Add, RealDiv, Placeholder, Pow, Tanh, Const, Mul                                                                      
      ReLU6            ✓         Min, Placeholder, Pack, Const, Relu                                                                                         
      HardSigmoid      ✓         Add, Minimum, Maximum, Placeholder, Const, Mul                                                                              
      HardTanh         ✓         Max, Min, Neg, Placeholder, Pack, Const                                                                                     
      LogSigmoid       ✓         Sigmoid, Log, Placeholder                                                                                                   
      SoftPlus         ✓         Softplus, Placeholder                                                                                                       
      SoftSign         ✓         Softsign, Placeholder                                                                                                       
      TanhShrink       ✓         Sub, Tanh, Placeholder                                                                                                      
      Sinc             ✓         Equal, Sin, RealDiv, Placeholder, Select, Const                                                                             
    =================  ========  =============================================================================================================  =============


Normalization
^^^^^^^^^^^^^

Count 2/6
 

    ==========================  ========  ========================================================================================================================================  ====================
         NNabla Function         Status                                                                    TF Op                                                                        Description     
    ==========================  ========  ========================================================================================================================================  ====================
      FusedBatchNormalization   ✓         Equal, SparseToDense, Where, AddV2, Add, Cast, Sum, Placeholder, Rsqrt, RealDiv, Squeeze, Sub, Const, Mean, Mul, GatherV2, Reshape, Relu                      
      BatchNormalization        ✓         Equal, SparseToDense, AddV2, Cast, Sum, Placeholder, Rsqrt, Squeeze, RealDiv, Sub, Const, Mean, Mul, GatherV2, Reshape, Where                                 
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
      Sum              ✓         Const, Sum, Placeholder                       
      Mean             ✓         Mean, Const, Placeholder                      
      Max              ✓         Max, Const, Placeholder                       
      Min              ✓         Min, Const, Placeholder                       
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
      Add2             ✓         Add, Placeholder                                 
      BcAdd2                                                  Not yet implemented.
      Sub2             ✓         Sub, Placeholder                                 
      Mul2             ✓         Mul, Placeholder                                 
      Div2             ✓         RealDiv, Placeholder                             
      Pow2             ✓         Pow, Placeholder                                 
      AddScalar        ✓         Add, Const, Placeholder                          
      MulScalar        ✓         Mul, Const, Placeholder                          
      PowScalar        ✓         Pow, Const, Placeholder                          
      RSubScalar       ✓         Sub, Const, Placeholder                          
      RDivScalar       ✓         RealDiv, Const, Placeholder                      
      RPowScalar       ✓         Pow, Const, Placeholder                          
    =================  ========  ===========================  ====================


Logical
^^^^^^^

Count 27/29
 

    =====================  ========  =====================================================  ====================
       NNabla Function      Status                           TF Op                              Description     
    =====================  ========  =====================================================  ====================
      Sign                 ✓         Sign, Placeholder                                                          
      Minimum2             ✓         Min, Add, Placeholder, Pack, Const                                         
      Maximum2             ✓         Max, Add, Placeholder, Pack, Const                                         
      MinimumScalar        ✓         Min, Add, Placeholder, Pack, Const                                         
      MaximumScalar        ✓         Max, Add, Placeholder, Pack, Const                                         
      LogicalAnd           ✓         LogicalAnd, Placeholder                                                    
      LogicalOr            ✓         LogicalOr, Placeholder                                                     
      LogicalXor           ✓         LogicalAnd, LogicalNot, LogicalOr, Placeholder                             
      Equal                ✓         Equal, Placeholder                                                         
      NotEqual             ✓         LogicalNot, Equal, Placeholder                                             
      GreaterEqual         ✓         LogicalNot, Less, Placeholder                                              
      Greater              ✓         Greater, Placeholder                                                       
      LessEqual            ✓         LogicalNot, Greater, Placeholder                                           
      Less                 ✓         Less, Placeholder                                                          
      LogicalAndScalar     ✓         LogicalAnd, Const, Placeholder                                             
      LogicalOrScalar      ✓         LogicalOr, Const, Placeholder                                              
      LogicalXorScalar     ✓         LogicalAnd, LogicalNot, LogicalOr, Placeholder, Const                      
      EqualScalar          ✓         Equal, Const, Placeholder                                                  
      NotEqualScalar       ✓         LogicalNot, Equal, Const, Placeholder                                      
      GreaterEqualScalar   ✓         LogicalNot, Less, Const, Placeholder                                       
      GreaterScalar        ✓         Greater, Const, Placeholder                                                
      LessEqualScalar      ✓         LogicalNot, Greater, Const, Placeholder                                    
      LessScalar           ✓         Less, Const, Placeholder                                                   
      LogicalNot           ✓         LogicalNot, Placeholder                                                    
      IsNaN                ✓         IsNan, Placeholder                                                         
      IsInf                X                                                                Not yet implemented.
      ResetNaN             ✓         IsNan, Select, Const, Placeholder                                          
      ResetInf             X                                                                Not yet implemented.
      Where                ✓         Select, Placeholder                                                        
    =====================  ========  =====================================================  ====================


Math
^^^^

Count 21/22
 

    =================  ========  ============================================  ====================
     NNabla Function    Status                      TF Op                          Description     
    =================  ========  ============================================  ====================
      Constant         ✓         Identity, Const                                                   
      Arange           ✓         Identity, Const                                                   
      Abs              ✓         Abs, Placeholder                                                  
      Exp              ✓         Exp, Placeholder                                                  
      Log              ✓         Log, Placeholder                                                  
      Identity         ✓         Identity, Placeholder                                             
      BatchMatmul      ✓         Transpose, Placeholder, Const, BatchMatMulV2                      
      Round            X                                                       Not yet implemented.
      Ceil             ✓         Ceil, Placeholder                                                 
      Floor            ✓         Floor, Placeholder                                                
      Sin              ✓         Sin, Placeholder                                                  
      Cos              ✓         Placeholder, Cos                                                  
      Tan              ✓         Tan, Placeholder                                                  
      Sinh             ✓         Sinh, Placeholder                                                 
      Cosh             ✓         Cosh, Placeholder                                                 
      ASin             ✓         Asin, Placeholder                                                 
      ACos             ✓         Acos, Placeholder                                                 
      ATan             ✓         Placeholder, Atan                                                 
      ATan2            ✓         Placeholder, RealDiv, Atan                                        
      ASinh            ✓         Asinh, Placeholder                                                
      ACosh            ✓         Acosh, Placeholder                                                
      ATanh            ✓         Atanh, Placeholder                                                
    =================  ========  ============================================  ====================


Array Manipulation
^^^^^^^^^^^^^^^^^^

Count 12/19
 

    =================  ========  ==============================================================================================  ================================================================================================================
     NNabla Function    Status                                               TF Op                                                                                                 Description                                                   
    =================  ========  ==============================================================================================  ================================================================================================================
      Concatenate      ✓         ConcatV2, Const, Placeholder                                                                                                                                                                                    
      Split            ✓         Squeeze, SplitV, Const, Placeholder                                                                                                                                                                             
      Stack            ✓         ExpandDims, ConcatV2, Const, Placeholder                                                                                                                                                                        
      Slice            △         Slice, Const, Placeholder                                                                       step != 1" exceed the scope of onnx opset 9,  not supported.                                                    
      Pad              △         PadV2, MirrorPad, Const, Placeholder                                                            When the mode of the pad is reflect, if the size of the pad exceeds the input size, tensorflow cannot handle it.
      Transpose        ✓         Transpose, Const, Placeholder                                                                                                                                                                                   
      Broadcast        ✓                                                                                                                                                                                                                         
      BroadcastTo      ✓                                                                                                                                                                                                                         
      Tile             ✓         Equal, SparseToDense, AddV2, Cast, Placeholder, Squeeze, Const, Tile, Reshape, Where, GatherV2                                                                                                                  
      OneHot           ✓         Equal, SparseToDense, AddV2, Cast, Placeholder, Squeeze, Const, GatherV2, Reshape, Where                                                                                                                        
      Flip             ✓         Transpose, Identity, Placeholder, Const, GatherV2                                                                                                                                                               
      Shift                                                                                                                      Not yet implemented.                                                                                            
      Sort                                                                                                                       Not yet implemented.                                                                                            
      Reshape          ✓         Equal, SparseToDense, AddV2, Cast, Placeholder, Squeeze, Const, GatherV2, Reshape, Where                                                                                                                        
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
      BinarySigmoid              ✓         Greater, Select, Const, Placeholder                                                                                                                                                                                                      
      BinaryTanh                 ✓         Greater, Select, Const, Placeholder                                                                                                                                                                                                      
      BinaryConnectAffine        ✓         Equal, SparseToDense, AddV2, MatMul, Cast, Placeholder, Squeeze, Const, Mul, GatherV2, Reshape, Where                                                                                                                                    
      BinaryConnectConvolution   △         Transpose, Equal, SparseToDense, Identity, Conv2D, AddV2, Add, Split, Cast, Placeholder, Squeeze, ConcatV2, Const, GatherV2, Reshape, Pad, Where       The cases `dilations` and `strides` larger than 1 are not supported by tensorflow.
      BinaryWeightAffine         ✓         Equal, SparseToDense, AddV2, Add, MatMul, Cast, Placeholder, Squeeze, Const, Mul, GatherV2, Reshape, Where                                                                                                                               
      BinaryWeightConvolution    △         Transpose, Equal, SparseToDense, Identity, Conv2D, AddV2, Add, Split, Cast, Placeholder, Squeeze, ConcatV2, Const, Mul, GatherV2, Reshape, Pad, Where  The cases `dilations` and `strides` larger than 1 are not supported by tensorflow.
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



