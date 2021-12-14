=============================
Function-Level Support Status
=============================

.. contents::
   :local:
   :depth: 3

ONNX Support Status
===================

:Note: In this document, the numbers in the header of all tables represent the version of onnx opset.


Import
------

- ✓: onnx specification defined, and supported.
- X: onnx specification defined, but not support yet.
- Empty: Not defined (Support status follows latest).


Total: 95/159

.. table::

    ===========================  ===  ===  ===  ===  ===  ===  ===  ===  ===  ====  ====  ====  ====  ============================================================================================  =============================================================================================================================================================================================================
           ONNX Operator          1    2    3    4    5    6    7    8    9    10    11    12    13                                           NNabla Func                                                                                                                                            Description                                                                                                 
    ===========================  ===  ===  ===  ===  ===  ===  ===  ===  ===  ====  ====  ====  ====  ============================================================================================  =============================================================================================================================================================================================================
     Abs                         ✓                        ✓    ✓                                ✓     Abs                                                                                                                                                                                                                                                                                                        
     Acos                                                      ✓                                      ACos                                                                                                                                                                                                                                                                                                       
     Acosh                                                               ✓                            ACosh                                                                                                                                                                                                                                                                                                      
     Add                         ✓                        ✓    ✓                                ✓     Add2, Reshape                                                                                                                                                                                                                                                                                              
     And                         ✓                             ✓                                      LogicalAnd, Reshape                                                                                                                                                                                                                                                                                        
     ArgMax                      ✓                             ✓              X     ✓           ✓     Flip, Max, RSubScalar                                                                                                                                                                                                                                                                                      
     ArgMin                      ✓                             ✓              X     ✓           ✓     Flip, Min, RSubScalar                                                                                                                                                                                                                                                                                      
     Asin                                                      ✓                                      ASin                                                                                                                                                                                                                                                                                                       
     Asinh                                                               ✓                            ASinh                                                                                                                                                                                                                                                                                                      
     Atan                                                      ✓                                      ATan                                                                                                                                                                                                                                                                                                       
     Atanh                                                               ✓                            ATanh                                                                                                                                                                                                                                                                                                      
     AveragePool                 ✓                             ✓              X     X                 AveragePooling, Pad                                                                           Not all features are verified. Those features can be verified by ONNXRuntime when opset > 6. Some feature is not supported by Nnabla such as Pad's edge mode. if opset >= 10, the ceil_mode is not supported.
     BatchNormalization          X                        X    X         ✓                            BatchNormalization                                                                                                                                                                                                                                                                                         
     BitShift                                                                       X                                                                                                               Not yet implemented.                                                                                                                                                                                         
     Cast                        ✓                        ✓    ✓         X                      X                                                                                                                                                                                                                                                                                                                
     Ceil                        ✓                        ✓    ✓                                ✓     Ceil                                                                                                                                                                                                                                                                                                       
     Celu                                                                                       ✓     Add2, Constant, Div2, ELU, Exp, MaximumScalar, MinimumScalar, Mul2, MulScalar, Reshape, Sub2                                                                                                                                                                                                               
     Clip                        ✓                        ✓    ✓                    ✓           ✓     Identity                                                                                                                                                                                                                                                                                                   
     Compress                                                            X          X                                                                                                               Not yet implemented.                                                                                                                                                                                         
     Concat                      ✓              ✓              ✓                    X           ✓     Concatenate                                                                                                                                                                                                                                                                                                
     ConcatFromSequence                                                             X                                                                                                               Not yet implemented.                                                                                                                                                                                         
     Constant                    ✓                             ✓         X          X                 Identity                                                                                                                                                                                                                                                                                                   
     ConstantOfShape                                                     ✓                            Constant                                                                                                                                                                                                                                                                                                   
     Conv                        ✓                             ✓                    X                 Convolution                                                                                                                                                                                                                                                                                                
     ConvInteger                                                              X                                                                                                                     Not yet implemented.                                                                                                                                                                                         
     ConvTranspose               ✓                             ✓                    X                 Deconvolution, Pad                                                                                                                                                                                                                                                                                         
     Cos                                                       ✓                                      Cos                                                                                                                                                                                                                                                                                                        
     Cosh                                                                ✓                            Cosh                                                                                                                                                                                                                                                                                                       
     CumSum                                                                         X                                                                                                               Not yet implemented.                                                                                                                                                                                         
     DepthToSpace                ✓                             ✓                    ✓           ✓     Reshape, Transpose                                                                                                                                                                                                                                                                                         
     DequantizeLinear                                                         ✓                 ✓     DequantizeLinear
     Det                                                                            X                                                                                                               Not yet implemented.                                                                                                                                                                                         
     Div                         ✓                        ✓    ✓                                ✓     Div2, Reshape                                                                                                                                                                                                                                                                                              
     Dropout                     ✓                        ✓    ✓              X                 ✓     Identity                                                                                                                                                                                                                                                                                                   
     DynamicQuantizeLinear                                                          X                                                                                                               Not yet implemented.                                                                                                                                                                                         
     Einsum                                                                                     X                                                                                                   Not yet implemented.                                                                                                                                                                                         
     Elu                         ✓                        ✓    ✓                                      ELU                                                                                                                                                                                                                                                                                                        
     Equal                       ✓                             ✓                    X           X     Equal, Reshape                                                                                                                                                                                                                                                                                             
     Erf                                                                 X                      X                                                                                                   Not yet implemented.                                                                                                                                                                                         
     Exp                         ✓                        ✓    ✓                                ✓     Exp                                                                                                                                                                                                                                                                                                        
     Expand                                                         ✓    ✓                      X     Broadcast, Reshape                                                                                                                                                                                                                                                                                         
     EyeLike                                                             X                                                                                                                          Not yet implemented.                                                                                                                                                                                         
     Flatten                     ✓                             ✓         ✓          ✓           ✓     Reshape                                                                                                                                                                                                                                                                                                    
     Floor                       ✓                        ✓    ✓                                ✓     Floor                                                                                                                                                                                                                                                                                                      
     GRU                         X         X                   X                                                                                                                                    Not yet implemented.                                                                                                                                                                                         
     Gather                      ✓                             ✓                    ✓           ✓     Concatenate, Slice                                                                                                                                                                                                                                                                                         
     GatherElements                                                                 X           X                                                                                                   Not yet implemented.                                                                                                                                                                                         
     GatherND                                                                       X           X                                                                                                   Not yet implemented.                                                                                                                                                                                         
     Gemm                        ✓                        ✓    ✓         ✓          ✓           ✓     Add2, BatchMatmul, MulScalar, Reshape                                                                                                                                                                                                                                                                      
     GlobalAveragePool           ✓                             ✓                                      GlobalAveragePooling                                                                                                                                                                                                                                                                                       
     GlobalLpPool                X    X                                                                                                                                                             Not yet implemented.                                                                                                                                                                                         
     GlobalMaxPool               X                                                                                                                                                                  Not yet implemented.                                                                                                                                                                                         
     Greater                     ✓                             ✓         ✓                      ✓     Greater, Reshape                                                                                                                                                                                                                                                                                           
     GreaterOrEqual                                                                             ✓     Equal, Greater, GreaterEqual, LogicalOr, Reshape                                                                                                                                                                                                                                                           
     HardSigmoid                 ✓                        ✓    ✓                                      AddScalar, HardSigmoid, MaximumScalar, MinimumScalar, MulScalar                                                                                                                                                                                                                                            
     Hardmax                     ✓                             ✓                    ✓           ✓     Max, OneHot, Transpose                                                                                                                                                                                                                                                                                     
     Identity                    ✓                             ✓                                ✓     Identity                                                                                                                                                                                                                                                                                                   
     If                          X                                                              X                                                                                                   Not yet implemented.                                                                                                                                                                                         
     InstanceNormalization       ✓                        ✓    ✓                                      BatchNormalization, Concatenate, Reshape, Split                                                                                                                                                                                                                                                            
     IsInf                                                                    ✓                       IsInf                                                                                                                                                                                                                                                                                                      
     IsNaN                                                               ✓                      ✓     IsNaN                                                                                                                                                                                                                                                                                                      
     LRN                         ✓                             ✓                                ✓     AddScalar, Div2, MulScalar, PowScalar, SumPooling, Transpose                                                                                                                                                                                                                                               
     LSTM                        X                             X                                                                                                                                    Not yet implemented.                                                                                                                                                                                         
     LeakyRelu                   ✓                        ✓    ✓                                      LeakyReLU                                                                                                                                                                                                                                                                                                  
     Less                        ✓                             ✓         ✓                      ✓     Less, Reshape                                                                                                                                                                                                                                                                                              
     LessOrEqual                                                                                ✓     Equal, Less, LessEqual, LogicalOr, Reshape                                                                                                                                                                                                                                                                 
     Log                         ✓                        ✓    ✓                                ✓     Log                                                                                                                                                                                                                                                                                                        
     LogSoftmax                  ✓                             ✓                    ✓           ✓     Div2, Exp, Log, Max, Sub2, Sum                                                                                                                                                                                                                                                                             
     Loop                        X                                                  X                                                                                                               Not yet implemented.                                                                                                                                                                                         
     LpNormalization             X                                                                                                                                                                  Not yet implemented.                                                                                                                                                                                         
     LpPool                      X    X                                             X                                                                                                               Not yet implemented.                                                                                                                                                                                         
     MatMul                      ✓                             ✓         ✓                      ✓     BatchMatmul, Reshape                                                                                                                                                                                                                                                                                       
     MatMulInteger                                                            X                                                                                                                     Not yet implemented.                                                                                                                                                                                         
     Max                         ✓                        ✓    ✓    ✓    ✓                      ✓     Maximum2                                                                                                                                                                                                                                                                                                   
     MaxPool                     ✓                             ✓    X         X     X           ✓     MaxPooling, Pad                                                                               Not all features are verified. Those features can be verified by ONNXRuntime. if opset >= 10, the ceil_mode is not supported, dilations is not equal to 1 is not supported.                                  
     MaxRoiPool                  X                                                                                                                                                                  Not yet implemented.                                                                                                                                                                                         
     MaxUnpool                                                           X          X                                                                                                               Not yet implemented.                                                                                                                                                                                         
     Mean                        ✓                        ✓    ✓    ✓    ✓                      ✓     Identity, Mean, Stack                                                                                                                                                                                                                                                                                      
     MeanVarianceNormalization                                           X                                                                                                                          Not yet implemented.                                                                                                                                                                                         
     Min                         ✓                        ✓    ✓    ✓    ✓                      ✓     Minimum2                                                                                                                                                                                                                                                                                                   
     Mod                                                                      X                 X                                                                                                   Not yet implemented.                                                                                                                                                                                         
     Mul                         ✓                        ✓    ✓                                ✓     Mul2, Reshape                                                                                                                                                                                                                                                                                              
     Multinomial                                               X                                                                                                                                    Not yet implemented.                                                                                                                                                                                         
     Neg                         ✓                        ✓    ✓                                ✓     MulScalar                                                                                                                                                                                                                                                                                                  
     NonMaxSuppression                                                        X     X                                                                                                               Not yet implemented.                                                                                                                                                                                         
     NonZero                                                             X                      X                                                                                                   Not yet implemented.                                                                                                                                                                                         
     Not                         ✓                             ✓                                      LogicalNot                                                                                                                                                                                                                                                                                                 
     OneHot                                                              X          X                                                                                                               Not yet implemented.                                                                                                                                                                                         
     Or                          ✓                             ✓                                      LogicalOr, Reshape                                                                                                                                                                                                                                                                                         
     PRelu                       ✓                        ✓    ✓         X                            PReLU                                                                                                                                                                                                                                                                                                      
     Pad                         ✓    ✓                        ✓                    ✓                 Pad                                                                                           Onnx required to support "edge" mode, while nnabla does not support it.                                                                                                                                      
     Pow                         ✓                             ✓                                ✓     Pow2, Reshape                                                                                                                                                                                                                                                                                              
     QLinearConv                                                              X                                                                                                                     Not yet implemented.                                                                                                                                                                                         
     QLinearMatMul                                                            X                                                                                                                     Not yet implemented.                                                                                                                                                                                         
     QuantizeLinear                                                           ✓                 ✓     QuantizeLinear
     RNN                         X                             X                                                                                                                                    Not yet implemented.                                                                                                                                                                                         
     RandomNormal                X                                                                                                                                                                  Not yet implemented.                                                                                                                                                                                         
     RandomNormalLike            X                                                                                                                                                                  Not yet implemented.                                                                                                                                                                                         
     RandomUniform               X                                                                                                                                                                  Not yet implemented.                                                                                                                                                                                         
     RandomUniformLike           X                                                                                                                                                                  Not yet implemented.                                                                                                                                                                                         
     Range                                                                          X                                                                                                               Not yet implemented.                                                                                                                                                                                         
     Reciprocal                  ✓                        ✓    ✓                                ✓     RDivScalar                                                                                                                                                                                                                                                                                                 
     ReduceL1                    X                                                  X                                                                                                               Not yet implemented.                                                                                                                                                                                         
     ReduceL2                    X                                                  X                                                                                                               Not yet implemented.                                                                                                                                                                                         
     ReduceLogSum                X                                                  X                                                                                                               Not yet implemented.                                                                                                                                                                                         
     ReduceLogSumExp             X                                                  X                                                                                                               Not yet implemented.                                                                                                                                                                                         
     ReduceMax                   ✓                             ✓                    ✓                 Max                                                                                                                                                                                                                                                                                                        
     ReduceMean                  ✓                             ✓                    ✓                 Mean                                                                                                                                                                                                                                                                                                       
     ReduceMin                   ✓                             ✓                    ✓                 Min                                                                                                                                                                                                                                                                                                        
     ReduceProd                  ✓                             ✓                    ✓                 Prod                                                                                                                                                                                                                                                                                                       
     ReduceSum                   ✓                             ✓                    ✓                 Sum                                                                                                                                                                                                                                                                                                        
     ReduceSumSquare             ✓                             ✓                    ✓                 PowScalar, Sum                                                                                                                                                                                                                                                                                             
     Relu                        ✓                        ✓    ✓                                ✓     ReLU                                                                                                                                                                                                                                                                                                       
     Reshape                     ✓                   ✓         ✓                                ✓     Reshape                                                                                                                                                                                                                                                                                                    
     Resize                                                                   X     X           X                                                                                                   Not yet implemented.                                                                                                                                                                                         
     ReverseSequence                                                          X                                                                                                                     Not yet implemented.                                                                                                                                                                                         
     RoiAlign                                                                 X                                                                                                                     Not yet implemented.                                                                                                                                                                                         
     Round                                                                          ✓                 Round                                                                                                                                                                                                                                                                                                      
     Scan                                                           X    X          X                                                                                                               Not yet implemented.                                                                                                                                                                                         
     Scatter                                                             X          X                                                                                                               Not yet implemented.                                                                                                                                                                                         
     ScatterElements                                                                X                                                                                                               Not yet implemented.                                                                                                                                                                                         
     ScatterND                                                                      X           X                                                                                                   Not yet implemented.                                                                                                                                                                                         
     Selu                        ✓                        ✓    ✓                                      SELU                                                                                                                                                                                                                                                                                                       
     SequenceAt                                                                     X                                                                                                               Not yet implemented.                                                                                                                                                                                         
     SequenceConstruct                                                              X                                                                                                               Not yet implemented.                                                                                                                                                                                         
     SequenceErase                                                                  X                                                                                                               Not yet implemented.                                                                                                                                                                                         
     SequenceInsert                                                                 X                                                                                                               Not yet implemented.                                                                                                                                                                                         
     SequenceLength                                                                 X                                                                                                               Not yet implemented.                                                                                                                                                                                         
     Shape                       X                                                              X                                                                                                   Not yet implemented.                                                                                                                                                                                         
     Shrink                                                              X                                                                                                                          Not yet implemented.                                                                                                                                                                                         
     Sigmoid                     ✓                        ✓    ✓                                ✓     Sigmoid                                                                                                                                                                                                                                                                                                    
     Sign                                                                ✓                      ✓     Sign                                                                                                                                                                                                                                                                                                       
     Sin                                                       ✓                                      Sin                                                                                                                                                                                                                                                                                                        
     Sinh                                                                ✓                            Sinh                                                                                                                                                                                                                                                                                                       
     Size                        X                                                              X                                                                                                   Not yet implemented.                                                                                                                                                                                         
     Slice                       ✓                             ✓              ✓     X           X     Slice                                                                                                                                                                                                                                                                                                      
     Softmax                     ✓                             ✓                    ✓           ✓     Div2, Exp, Max, Sub2, Sum                                                                                                                                                                                                                                                                                  
     Softplus                    X                             X                                      SoftPlus                                                                                      Not yet implemented.                                                                                                                                                                                         
     Softsign                    ✓                             ✓                                      SoftSign                                                                                                                                                                                                                                                                                                   
     SpaceToDepth                ✓                             ✓                                      Reshape, Transpose                                                                                                                                                                                                                                                                                         
     Split                       ✓    ✓                        ✓                    ✓           ✓     Split, Stack                                                                                                                                                                                                                                                                                               
     SplitToSequence                                                                X                                                                                                               Not yet implemented.                                                                                                                                                                                         
     Sqrt                        ✓                        ✓    ✓                                ✓     PowScalar                                                                                                                                                                                                                                                                                                  
     Squeeze                     ✓                             ✓                    ✓           ✓     Reshape                                                                                                                                                                                                                                                                                                    
     StringNormalizer                                                         X                                                                                                                     Not yet implemented.                                                                                                                                                                                         
     Sub                         ✓                        ✓    ✓                                ✓     Reshape, Sub2                                                                                                                                                                                                                                                                                              
     Sum                         ✓                        ✓    ✓    X    X                      ✓     AddN                                                                                                                                                                                                                                                                                                       
     Tan                                                       ✓                                      Tan                                                                                                                                                                                                                                                                                                        
     Tanh                        ✓                        ✓    ✓                                ✓     Tanh                                                                                                                                                                                                                                                                                                       
     TfIdfVectorizer                                                     X                                                                                                                          Not yet implemented.                                                                                                                                                                                         
     ThresholdedRelu                                                          ✓                       Constant, GreaterScalar, Where                                                                                                                                                                                                                                                                             
     Tile                        ✓                        ✓    ✓                                ✓     Tile                                                                                                                                                                                                                                                                                                       
     TopK                        X                                            X     X                                                                                                               Not yet implemented.                                                                                                                                                                                         
     Transpose                   ✓                             ✓                                ✓     Transpose                                                                                                                                                                                                                                                                                                  
     Unique                                                                         X                                                                                                               Not yet implemented.                                                                                                                                                                                         
     Unsqueeze                   ✓                             ✓                    ✓           ✓     Reshape                                                                                                                                                                                                                                                                                                    
     Upsample                    X                             X         ✓    X                       Unpooling                                                                                                                                                                                                                                                                                                  
     Where                                                               ✓                            Where                                                                                                                                                                                                                                                                                                      
     Xor                         ✓                             ✓                                      LogicalXor, Reshape                                                                                                                                                                                                                                                                                        
    ===========================  ===  ===  ===  ===  ===  ===  ===  ===  ===  ====  ====  ====  ====  ============================================================================================  =============================================================================================================================================================================================================



Export
------

- ✓: Support to export this opset.
- △: Partially support to export this opset (e.g. some cases cannot be supported, or not completely tested).
- X: Supported, but test failed.
- Empty: Not support corresponding opset version.

Total: 124/215

Neural Network Layer
^^^^^^^^^^^^^^^^^^^^

Count 11/18


    ===============================  ===  ===  ====  ====  ====  ========================================  ======================================================================================
            NNabla Function           7    9    10    11    13                   ONNX Op                                                        Description                                      
    ===============================  ===  ===  ====  ====  ====  ========================================  ======================================================================================
      Affine                         ✓    ✓    ✓     ✓     ✓     Gemm, Reshape                                                                                                                   
      RNN                                                                                                  Not yet implemented.                                                                  
      LSTM                                                                                                 Not yet implemented.                                                                  
      GRU                                                                                                  Not yet implemented.                                                                  
      Convolution                    ✓    ✓    ✓     ✓     ✓     Conv, Reshape                                                                                                                   
      FusedConvolution                                                                                     Not yet implemented.                                                                  
      DepthwiseConvolution           ✓    ✓    ✓     ✓     ✓     Conv, Reshape                                                                                                                   
      Deconvolution                  ✓    ✓    ✓     ✓     ✓     ConvTranspose, Reshape                                                                                                          
      DepthwiseDeconvolution         ✓    ✓    ✓     ✓     ✓     ConvTranspose, Reshape                                                                                                          
      DeformableConvolution                                                                                Not yet implemented.                                                                  
      AdaptiveSeparableConvolution                                                                         Not yet implemented.                                                                  
      MaxPooling                     ✓    ✓    ✓     ✓     ✓     Constant, MaxPool, Pad, Reshape                                                                                                 
      AveragePooling                 △    △    △     △     △     AveragePool, Constant, Pad, Reshape       Currently only supports the cases where both ignore_border and including_pad are True.
      GlobalAveragePooling           ✓    ✓    ✓     ✓     ✓     GlobalAveragePool                                                                                                               
      SumPooling                     ✓    ✓    ✓     ✓     ✓     AveragePool, Constant, Mul, Pad, Reshape                                                                                        
      Unpooling                      ✓    ✓    ✓     ✓     ✓     Resize                                                                                                                          
      Embed                          ✓    ✓    ✓     ✓     ✓     Gather                                                                                                                          
      RoiAlign                                                                                             Not yet implemented.                                                                  
    ===============================  ===  ===  ====  ====  ====  ========================================  ======================================================================================


Neural Network Activation Functions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Count 20/22


    =================  ===  ===  ====  ====  ====  ========================================  ====================
     NNabla Function    7    9    10    11    13                   ONNX Op                       Description     
    =================  ===  ===  ====  ====  ====  ========================================  ====================
      Sigmoid          ✓    ✓    ✓     ✓     ✓     Sigmoid                                                       
      Swish            ✓    ✓    ✓     ✓     ✓     Mul, Sigmoid                                                  
      Tanh             ✓    ✓    ✓     ✓     ✓     Tanh                                                          
      ReLU             ✓    ✓    ✓     ✓     ✓     Relu                                                          
      LeakyReLU        ✓    ✓    ✓     ✓     ✓     LeakyRelu                                                     
      Softmax          ✓    ✓    ✓     ✓     ✓     Div, Exp, ReduceMax, ReduceSum, Sub                           
      LogSoftmax       ✓    ✓    ✓     ✓     ✓     Exp, Log, ReduceMax, ReduceSum, Sub                           
      ELU              ✓    ✓    ✓     ✓     ✓     Elu                                                           
      SELU             ✓    ✓    ✓     ✓     ✓     Selu                                                          
      CReLU            ✓    ✓    ✓     ✓     ✓     Concat, Neg, Relu                                             
      CELU             ✓    ✓    ✓     ✓     ✓     Concat, Elu, Neg                                              
      PReLU            ✓    ✓    ✓     ✓     ✓     PRelu, Reshape                                                
      GELU             ✓    ✓    ✓     ✓     ✓     Add, Constant, Div, Mul, Pow, Sqrt, Tanh                      
      Mish                                                                                   Not yet implemented.
      ReLU6            ✓    ✓    ✓     ✓     ✓     Constant, Min, Relu                                           
      HardSigmoid      ✓    ✓    ✓     ✓     ✓     HardSigmoid                                                   
      HardTanh         ✓    ✓    ✓     ✓     ✓     Constant, Max, Min, Neg                                       
      LogSigmoid       ✓    ✓    ✓     ✓     ✓     Log, Sigmoid                                                  
      SoftPlus         X    X    X     X     X     Softplus                                  Not yet implemented.
      SoftSign         ✓    ✓    ✓     ✓     ✓     Softsign                                                      
      TanhShrink       ✓    ✓    ✓     ✓     ✓     Sub, Tanh                                                     
      Sinc             X    X    X     ✓     ✓     Constant, Div, Equal, Sin, Where                              
    =================  ===  ===  ====  ====  ====  ========================================  ====================


Normalization
^^^^^^^^^^^^^

Count 7/14


    ==========================  ===  ===  ====  ====  ====  ===============================================================================================  ====================
         NNabla Function         7    9    10    11    13                                               ONNX Op                                                  Description     
    ==========================  ===  ===  ====  ====  ====  ===============================================================================================  ====================
      FusedBatchNormalization   ✓    ✓    ✓     ✓     ✓     Add, BatchNormalization, Constant, Div, Mul, ReduceMean, ReduceSum, Relu, Reshape, Squeeze, Sub                      
      BatchNormalization        ✓    ✓    ✓     ✓     ✓     BatchNormalization, Constant, Div, Mul, ReduceMean, ReduceSum, Reshape, Squeeze, Sub                                 
      GroupNormalization                                                                                                                                     Not yet implemented.
      InstanceNormalization     ✓    ✓    ✓     ✓     ✓     Add, Constant, Div, Mul, Pow, ReduceMean, ReduceSum, Reshape, Sub                                                    
      LayerNormalization        ✓    ✓    ✓     ✓     ✓     Add, Constant, Div, Mul, Pow, ReduceMean, ReduceSum, Sub                                                             
      NormNormalization                                                                                                                                      Not yet implemented.
      SyncBatchNormalization                                                                                                                                 Not yet implemented.
      TensorNormalization                                                                                                                                    Not yet implemented.
      WeightNormalization       ✓    ✓    ✓     ✓     ✓     Add, Constant, Mul, Pow, ReduceSum, Reshape                                                                          
      WeightStandardization     ✓    ✓    ✓     ✓     ✓     Add, Constant, Div, Mul, Pow, ReduceMean, ReduceSum, Sub                                                             
      SpectralNorm              ✓    ✓    ✓     ✓     ✓     Add, Constant, Div, Gemm, Pow, ReduceSum, Reshape, Sqrt, Transpose                                                   
      MeanSubtraction                                                                                                                                        Not yet implemented.
      ClipGradByValue                                                                                                                                        Not yet implemented.
      ClipGradByNorm                                                                                                                                         Not yet implemented.
    ==========================  ===  ===  ====  ====  ====  ===============================================================================================  ====================


Reduction
^^^^^^^^^

Count 5/10


    =================  ===  ===  ====  ====  ====  ==========  ====================
     NNabla Function    7    9    10    11    13    ONNX Op        Description     
    =================  ===  ===  ====  ====  ====  ==========  ====================
      Sum              ✓    ✓    ✓     ✓     ✓     ReduceSum                       
      CumSum                                                   Not yet implemented.
      Mean             ✓    ✓    ✓     ✓     ✓     ReduceMean                      
      Max              ✓    ✓    ✓     ✓     ✓     ReduceMax                       
      Min              ✓    ✓    ✓     ✓     ✓     ReduceMin                       
      Norm                                                     Not yet implemented.
      Prod             ✓    ✓    ✓     ✓     ✓     ReduceProd                      
      CumProd                                                  Not yet implemented.
      ReduceSum                                                Not yet implemented.
      ReduceMean                                               Not yet implemented.
    =================  ===  ===  ====  ====  ====  ==========  ====================


Arithmetic
^^^^^^^^^^

Count 11/14


    =================  ===  ===  ====  ====  ====  =============  ====================
     NNabla Function    7    9    10    11    13      ONNX Op         Description     
    =================  ===  ===  ====  ====  ====  =============  ====================
      Add2             ✓    ✓    ✓     ✓     ✓     Add                                
      AddN                                                        Not yet implemented.
      BcAdd2                                                      Not yet implemented.
      Sub2             ✓    ✓    ✓     ✓     ✓     Sub                                
      Mul2             ✓    ✓    ✓     ✓     ✓     Mul                                
      MulN                                                        Not yet implemented.
      Div2             ✓    ✓    ✓     ✓     ✓     Div                                
      Pow2             ✓    ✓    ✓     ✓     ✓     Pow                                
      AddScalar        ✓    ✓    ✓     ✓     ✓     Add, Constant                      
      MulScalar        ✓    ✓    ✓     ✓     ✓     Constant, Mul                      
      PowScalar        ✓    ✓    ✓     ✓     ✓     Constant, Pow                      
      RSubScalar       ✓    ✓    ✓     ✓     ✓     Constant, Sub                      
      RDivScalar       ✓    ✓    ✓     ✓     ✓     Constant, Div                      
      RPowScalar       ✓    ✓    ✓     ✓     ✓     Constant, Pow                      
    =================  ===  ===  ====  ====  ====  =============  ====================


Logical
^^^^^^^

Count 29/30


    =====================  ===  ===  ====  ====  ====  ======================  ====================
       NNabla Function      7    9    10    11    13          ONNX Op              Description     
    =====================  ===  ===  ====  ====  ====  ======================  ====================
      Sign                 X    ✓    ✓     ✓     ✓     Sign                                        
      Minimum2             ✓    ✓    ✓     ✓     ✓     Add, Constant, Min                          
      Maximum2             ✓    ✓    ✓     ✓     ✓     Add, Constant, Max                          
      MinimumScalar        ✓    ✓    ✓     ✓     ✓     Add, Constant, Min                          
      MaximumScalar        ✓    ✓    ✓     ✓     ✓     Add, Constant, Max                          
      LogicalAnd           ✓    ✓    ✓     ✓     ✓     And                                         
      LogicalOr            ✓    ✓    ✓     ✓     ✓     Or                                          
      LogicalXor           ✓    ✓    ✓     ✓     ✓     Xor                                         
      Equal                X    X    X     ✓     ✓     Equal                                       
      NotEqual             X    X    X     ✓     ✓     Equal, Not                                  
      GreaterEqual         ✓    ✓    ✓     ✓     ✓     Less, Not                                   
      Greater              ✓    ✓    ✓     ✓     ✓     Greater                                     
      LessEqual            ✓    ✓    ✓     ✓     ✓     Greater, Not                                
      Less                 ✓    ✓    ✓     ✓     ✓     Less                                        
      SearchSorted                                                             Not yet implemented.
      LogicalAndScalar     ✓    ✓    ✓     ✓     ✓     And, Constant                               
      LogicalOrScalar      ✓    ✓    ✓     ✓     ✓     Constant, Or                                
      LogicalXorScalar     ✓    ✓    ✓     ✓     ✓     Constant, Xor                               
      EqualScalar          X    X    X     ✓     ✓     Constant, Equal                             
      NotEqualScalar       X    X    X     ✓     ✓     Constant, Equal, Not                        
      GreaterEqualScalar   ✓    ✓    ✓     ✓     ✓     Constant, Less, Not                         
      GreaterScalar        ✓    ✓    ✓     ✓     ✓     Constant, Greater                           
      LessEqualScalar      ✓    ✓    ✓     ✓     ✓     Constant, Greater, Not                      
      LessScalar           ✓    ✓    ✓     ✓     ✓     Constant, Less                              
      LogicalNot           ✓    ✓    ✓     ✓     ✓     Not                                         
      IsNaN                X    ✓    ✓     ✓     ✓     IsNaN                                       
      IsInf                X    X    ✓     ✓     ✓     IsInf                                       
      ResetNaN             X    ✓    ✓     ✓     ✓     Constant, IsNaN, Where                      
      ResetInf             X    X    ✓     ✓     ✓     Constant, IsInf, Where                      
      Where                X    ✓    ✓     ✓     ✓     Where                                       
    =====================  ===  ===  ====  ====  ====  ======================  ====================


Math
^^^^

Count 22/22


    =================  ===  ===  ====  ====  ====  ==================  =============
     NNabla Function    7    9    10    11    13        ONNX Op         Description 
    =================  ===  ===  ====  ====  ====  ==================  =============
      Constant         ✓    ✓    ✓     ✓     ✓     Constant, Identity               
      Arange           ✓    ✓    ✓     ✓     ✓     Constant, Identity               
      Abs              ✓    ✓    ✓     ✓     ✓     Abs                              
      Exp              ✓    ✓    ✓     ✓     ✓     Exp                              
      Log              ✓    ✓    ✓     ✓     ✓     Log                              
      Identity         ✓    ✓    ✓     ✓     ✓     Identity                         
      BatchMatmul      ✓    ✓    ✓     ✓     ✓     MatMul, Transpose                
      Round            X    X    X     ✓     ✓     Round                            
      Ceil             ✓    ✓    ✓     ✓     ✓     Ceil                             
      Floor            ✓    ✓    ✓     ✓     ✓     Floor                            
      Sin              ✓    ✓    ✓     ✓     ✓     Sin                              
      Cos              ✓    ✓    ✓     ✓     ✓     Cos                              
      Tan              ✓    ✓    ✓     ✓     ✓     Tan                              
      Sinh             X    ✓    ✓     ✓     ✓     Sinh                             
      Cosh             X    ✓    ✓     ✓     ✓     Cosh                             
      ASin             ✓    ✓    ✓     ✓     ✓     Asin                             
      ACos             ✓    ✓    ✓     ✓     ✓     Acos                             
      ATan             ✓    ✓    ✓     ✓     ✓     Atan                             
      ATan2            ✓    ✓    ✓     ✓     ✓     Atan, Div                        
      ASinh            X    ✓    ✓     ✓     ✓     Asinh                            
      ACosh            X    ✓    ✓     ✓     ✓     Acosh                            
      ATanh            X    ✓    ✓     ✓     ✓     Atanh                            
    =================  ===  ===  ====  ====  ====  ==================  =============


Array Manipulation
^^^^^^^^^^^^^^^^^^

Count 12/30


    =====================  ===  ===  ====  ====  ====  ===========================  =================================================================================================================
       NNabla Function      7    9    10    11    13             ONNX Op                                                               Description                                                   
    =====================  ===  ===  ====  ====  ====  ===========================  =================================================================================================================
      Concatenate          ✓    ✓    ✓     ✓     ✓     Concat                                                                                                                                        
      Split                ✓    ✓    ✓     ✓     ✓     Split, Squeeze                                                                                                                                
      Stack                ✓    ✓    ✓     ✓     ✓     Concat, Unsqueeze                                                                                                                             
      Slice                △    △    ✓     ✓     ✓     Constant, Slice              ONNX slice cannot support step != 1 on opset < 10.                                                               
      Pad                  △    △    △     △     △     Constant, Pad                When the mode of the pad is reflect, if the size of the pad exceeds the input size, onnxruntime cannot handle it.
      Transpose            ✓    ✓    ✓     ✓     ✓     Transpose                                                                                                                                     
      Broadcast            X    ✓    ✓     ✓     ✓                                                                                                                                                   
      BroadcastTo          ✓    ✓    ✓     ✓     ✓                                                                                                                                                   
      Tile                 ✓    ✓    ✓     ✓     ✓     Constant, Reshape, Tile                                                                                                                       
      OneHot               X    ✓    ✓     ✓     ✓     Flatten, Gather, Reshape                                                                                                                      
      Flip                 ✓    ✓    ✓     ✓     ✓     Gather, Identity, Transpose                                                                                                                   
      Shift                                                                         Not yet implemented.                                                                                             
      Sort                                                                          Not yet implemented.                                                                                             
      Reshape              ✓    ✓    ✓     ✓     ✓     Constant, Reshape                                                                                                                             
      MatrixDiag                                                                    Not yet implemented.                                                                                             
      MatrixDiagPart                                                                Not yet implemented.                                                                                             
      Meshgrid                                                                      Not yet implemented.                                                                                             
      BatchDet                                                                      Not yet implemented.                                                                                             
      BatchInv                                                                      Not yet implemented.                                                                                             
      BatchLogdet                                                                   Not yet implemented.                                                                                             
      Assign                                                                        Not yet implemented.                                                                                             
      Gather                                                                        Not yet implemented.                                                                                             
      GatherNd                                                                      Not yet implemented.                                                                                             
      BoolGather                                                                    Not yet implemented.                                                                                             
      ScatterNd                                                                     Not yet implemented.                                                                                             
      ScatterAdd                                                                    Not yet implemented.                                                                                             
      BoolScatter                                                                   Not yet implemented.                                                                                             
      BoolFill                                                                      Not yet implemented.                                                                                             
      PackPaddedSequence                                                            Not yet implemented.                                                                                             
      PadPackedSequence                                                             Not yet implemented.                                                                                             
    =====================  ===  ===  ====  ====  ====  ===========================  =================================================================================================================


Signal Processing
^^^^^^^^^^^^^^^^^

Count 1/5


    =================  ===  ===  ====  ====  ====  =========  ====================
     NNabla Function    7    9    10    11    13    ONNX Op       Description     
    =================  ===  ===  ====  ====  ====  =========  ====================
      Interpolate      X    X    △     ✓     ✓     Resize                         
      FFT                                                     Not yet implemented.
      IFFT                                                    Not yet implemented.
      STFT                                                    Not yet implemented.
      ISTFT                                                   Not yet implemented.
    =================  ===  ===  ====  ====  ====  =========  ====================


Stochasticity
^^^^^^^^^^^^^

Count 0/15


    ====================  ===  ===  ====  ====  ====  =========  ==================================================================================================================
      NNabla Function      7    9    10    11    13    ONNX Op                                                      Description                                                    
    ====================  ===  ===  ====  ====  ====  =========  ==================================================================================================================
      Dropout             X    X    X     X     X     Dropout    The Dropout in nnabla has no test mode and contains random parameters, so the test result is not the same as onnx.
      TopKData                                                   Not yet implemented.                                                                                              
      TopKGrad                                                   Not yet implemented.                                                                                              
      Rand                                                       Not yet implemented.                                                                                              
      Randint                                                    Not yet implemented.                                                                                              
      Randn                                                      Not yet implemented.                                                                                              
      RandBinomial                                               Not yet implemented.                                                                                              
      RandBeta                                                   Not yet implemented.                                                                                              
      RandGamma                                                  Not yet implemented.                                                                                              
      RandomChoice                                               Not yet implemented.                                                                                              
      RandomCrop                                                 Not yet implemented.                                                                                              
      RandomFlip                                                 Not yet implemented.                                                                                              
      RandomShift                                                Not yet implemented.                                                                                              
      RandomErase                                                Not yet implemented.                                                                                              
      ImageAugmentation                                          Not yet implemented.                                                                                              
    ====================  ===  ===  ====  ====  ====  =========  ==================================================================================================================


Loss Functions
^^^^^^^^^^^^^^

Count 0/9


    ==========================  ===  ===  ====  ====  ====  =========  ====================
         NNabla Function         7    9    10    11    13    ONNX Op       Description     
    ==========================  ===  ===  ====  ====  ====  =========  ====================
      SigmoidCrossEntropy                                              Not yet implemented.
      BinaryCrossEntropy                                               Not yet implemented.
      SoftmaxCrossEntropy                                              Not yet implemented.
      CategoricalCrossEntropy                                          Not yet implemented.
      SquaredError                                                     Not yet implemented.
      AbsoluteError                                                    Not yet implemented.
      HuberLoss                                                        Not yet implemented.
      EpsilonInsensitiveLoss                                           Not yet implemented.
      KLMultinomial                                                    Not yet implemented.
    ==========================  ===  ===  ====  ====  ====  =========  ====================


Geometric Neural Network Layers
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Count 0/3


    =================  ===  ===  ====  ====  ====  =========  ====================
     NNabla Function    7    9    10    11    13    ONNX Op       Description     
    =================  ===  ===  ====  ====  ====  =========  ====================
      AffineGrid                                              Not yet implemented.
      WarpByGrid                                              Not yet implemented.
      WarpByFlow                                              Not yet implemented.
    =================  ===  ===  ====  ====  ====  =========  ====================


Quantization Neural Network Layers
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Count 6/14


    ===========================  ===  ===  ====  ====  ====  =========================  ====================
          NNabla Function         7    9    10    11    13            ONNX Op               Description     
    ===========================  ===  ===  ====  ====  ====  =========================  ====================
      BinarySigmoid              X    ✓    ✓     ✓     ✓     Constant, Greater, Where                       
      BinaryTanh                 X    ✓    ✓     ✓     ✓     Constant, Greater, Where                       
      BinaryConnectAffine        ✓    ✓    ✓     ✓     ✓     Gemm, Reshape                                  
      BinaryConnectConvolution   ✓    ✓    ✓     ✓     ✓     Conv, Reshape                                  
      BinaryWeightAffine         ✓    ✓    ✓     ✓     ✓     Add, MatMul, Mul, Reshape                      
      BinaryWeightConvolution    ✓    ✓    ✓     ✓     ✓     Add, Conv, Mul, Reshape                        
      INQAffine                                                                         Not yet implemented.
      INQConvolution                                                                    Not yet implemented.
      FixedPointQuantize                                                                Not yet implemented.
      MinMaxQuantize                                                                    Not yet implemented.
      Pow2Quantize                                                                      Not yet implemented.
      Prune                                                                             Not yet implemented.
      QuantizeLinear                       ✓           ✓
      DequantizeLinear                     ✓           ✓
    ===========================  ===  ===  ====  ====  ====  =========================  ====================


Validation
^^^^^^^^^^

Count 0/3


    ==================  ===  ===  ====  ====  ====  =========  ====================
     NNabla Function     7    9    10    11    13    ONNX Op       Description     
    ==================  ===  ===  ====  ====  ====  =========  ====================
      TopNError                                                Not yet implemented.
      BinaryError                                              Not yet implemented.
      ConfusionMatrix                                          Not yet implemented.
    ==================  ===  ===  ====  ====  ====  =========  ====================


Unsupported, Special Use
^^^^^^^^^^^^^^^^^^^^^^^^

Count 0/6


    =====================  ===  ===  ====  ====  ====  =========  ====================
       NNabla Function      7    9    10    11    13    ONNX Op       Description     
    =====================  ===  ===  ====  ====  ====  =========  ====================
      VATNoise                                                    Not yet implemented.
      Unlink                                                      Not yet implemented.
      Sink                                                        Not yet implemented.
      NmsDetection2d                                              Not yet implemented.
      MaxPoolingBackward                                          Not yet implemented.
      PatchCorrelation                                            Not yet implemented.
    =====================  ===  ===  ====  ====  ====  =========  ====================





Tensorflow Support Status
=========================

Import
------

- ✓: Supported
- △: Partially supported
- X: Supported, but test failed.
- Empty: Not support yet.


Total: 107/122

.. table:: Tensorflow support status

    ======================  ========  ==================================================  ====================
     Tensorflow Function     Status                      NNabla Func                          Description     
    ======================  ========  ==================================================  ====================
      Abs                      ✓      Abs                                                                     
      Acos                     ✓      ACos                                                                    
      Acosh                    ✓      ACosh                                                                   
      Add                      ✓      Add2                                                                    
      AddN                     ✓      AddN                                                                    
      All                      ✓      Greater, Min, Reshape                                                   
      Any                      ✓      Greater, Reshape, Sum                                                   
      ArgMax                   ✓      Max                                                                     
      ArgMin                   ✓      Min                                                                     
      Asin                     ✓      ASin                                                                    
      Asinh                    ✓      ASinh                                                                   
      Atan                     ✓      ATan                                                                    
      Atan2                    ✓      ATan, Add2, Div2, Mul2, Reshape, Sign, Sub2                             
      Atanh                    ✓      ATanh                                                                   
      AvgPool                  △      AveragePooling, Pad, Transpose                                          
      AvgPool3D                △      AveragePooling, Pad, Transpose                                          
      BatchMatMul              ✓      BatchMatmul, Transpose                                                  
      BatchNormalization       ✓      Add2, Mul2, PowScalar, RDivScalar, Reshape, Sub2                        
      BiasAdd                  ✓      Add2, Reshape                                                           
      BroadcastTo              ✓                                                                              
      Cast                     X      NA                                                  Not yet implemented.
      Ceil                     ✓      Ceil                                                                    
      ClipByValue              ✓      Maximum2, Minimum2, Reshape                                             
      Concat                   ✓      Concatenate                                                             
      ConcatV2                 ✓      Concatenate                                                             
      Const                    ✓      NA                                                                      
      Conv1D                   △      Convolution, Pad, Reshape, Transpose                                    
      Conv1DTranspose          △      Deconvolution, Reshape, Transpose                                       
      Conv2D                   △      Convolution, Pad, Transpose                                             
      Conv2DBackpropInput      △      Deconvolution, Transpose                                                
      Conv3D                   △      Convolution, Pad, Transpose                                             
      Conv3DBackpropInput      △      Deconvolution, Pad, Transpose                                           
      Cos                      ✓      Cos                                                                     
      Cosh                     ✓      Cosh                                                                    
      Crelu                    ✓      Concatenate, MulScalar, ReLU                                            
      Cumsum                   X                                                          Not yet implemented.
      DepthToSpace             ✓      Reshape, Transpose                                                      
      DepthwiseConv2d          △      Convolution, Pad, Reshape, Transpose                                    
      Div                      ✓      Div2                                                                    
      Elu                      ✓      ELU                                                                     
      Equal                    ✓      Equal                                                                   
      Erf                      X                                                          Not yet implemented.
      Erfc                     X                                                          Not yet implemented.
      Exp                      ✓      Exp                                                                     
      ExpandDims               ✓      Reshape                                                                 
      Floor                    ✓      Floor                                                                   
      FloorDiv                 ✓      Div2, Floor                                                             
      FloorMod                 ✓      Div2, Floor, Mul2, Sub2                                                 
      GatherNd                 X                                                          Not yet implemented.
      GatherV2                 X      Concatenate, Slice                                  Not yet implemented.
      Greater                  ✓      Greater                                                                 
      GreaterEqual             ✓      Less, LogicalNot                                                        
      Identity                 ✓      Identity                                                                
      IsInf                    ✓      IsInf                                                                   
      IsNan                    ✓      IsNaN                                                                   
      LeakyRelu                ✓      LeakyReLU                                                               
      Less                     ✓      Less                                                                    
      LessEqual                ✓      Greater, LogicalNot                                                     
      Log                      ✓      Log                                                                     
      LogSigmoid               X      MulScalar, SoftPlus                                 Not yet implemented.
      LogSoftmax               ✓      Add2, Exp, Log, Max, Reshape, Sub2, Sum, Transpose                      
      LogicalAnd               ✓      LogicalAnd                                                              
      LogicalNot               ✓      LogicalNot                                                              
      LogicalOr                ✓      LogicalOr                                                               
      LogicalXor               ✓      LogicalAnd, LogicalNot, LogicalOr                                       
      Max                      ✓      Max                                                                     
      MaxPool                  △      MaxPooling, Pad, Reshape, Transpose                                     
      MaxPool3D                △      MaxPooling, Pad, Transpose                                              
      MaxPoolWithArgmax        X                                                          Not yet implemented.
      Maximum                  ✓      Maximum2                                                                
      Mean                     ✓      Mean                                                                    
      Min                      ✓      Min                                                                     
      Minimum                  ✓      Minimum2                                                                
      Mul                      ✓      Mul2                                                                    
      Neg                      ✓      MulScalar                                                               
      NotEqual                 ✓      Equal, LogicalNot                                                       
      Pack                     ✓      Concatenate, Reshape                                                    
      Pad                      △      Pad                                                                     
      Pow                      ✓      Pow2                                                                    
      Prod                     ✓      Prod                                                                    
      RealDiv                  ✓      Div2                                                                    
      Reciprocal               ✓      RDivScalar                                                              
      Relu                     ✓      ReLU                                                                    
      Relu6                    ✓      MaximumScalar, MinimumScalar                                            
      Reshape                  ✓      Reshape                                                                 
      ReverseSequence          X                                                          Not yet implemented.
      ReverseV2                X                                                          Not yet implemented.
      Round                    ✓      Round                                                                   
      Rsqrt                    ✓      PowScalar, RDivScalar                                                   
      Selu                     ✓      SELU                                                                    
      Shape                    X                                                          Not yet implemented.
      Sigmoid                  ✓      Sigmoid                                                                 
      Sign                     ✓      Sign                                                                    
      Sin                      ✓      Sin                                                                     
      Sinh                     ✓      Sinh                                                                    
      Size                     X                                                          Not yet implemented.
      Slice                    ✓      Slice                                                                   
      Softmax                  ✓      Div2, Exp, Max, Reshape, Sub2, Sum, Transpose                           
      Softplus                 X      SoftPlus                                            Not yet implemented.
      Softsign                 ✓      SoftSign                                                                
      SpaceToDepth             ✓      Reshape, Transpose                                                      
      Split                    ✓      Split, Stack                                                            
      SplitV                   ✓      Split, Stack                                                            
      Sqrt                     ✓      PowScalar                                                               
      Square                   ✓      Mul2                                                                    
      SquaredDifference        ✓      Mul2, Sub2                                                              
      Squeeze                  ✓      Reshape                                                                 
      StopGradient             ✓      Identity                                                                
      StridedSlice             △      Slice                                                                   
      Sub                      ✓      Sub2                                                                    
      Sum                      ✓      Sum                                                                     
      Swish                    ✓      Mul2, Sigmoid                                                           
      Tan                      ✓      Tan                                                                     
      Tanh                     ✓      Tanh                                                                    
      Tile                     ✓      Tile                                                                    
      TopKV2                   X                                                          Not yet implemented.
      Transpose                ✓      Transpose                                                               
      TruncateDiv              ✓      Div2                                                                    
      TruncateMod              X                                                          Not yet implemented.
      Unpack                   ✓      Reshape, Split, Stack                                                   
      Where                    △      Where                                                                   
      ZerosLike                ✓      NA                                                                      
    ======================  ========  ==================================================  ====================





Export
------

- ✓: Supported
- △: Partially supported
- X: Supported, but test failed.
- Empty: Not support yet.


Total: 124/215

Neural Network Layer
^^^^^^^^^^^^^^^^^^^^

Count 11/18


    ===============================  ========  ==================================================================================
            NNabla Function           Status                                      Description                                    
    ===============================  ========  ==================================================================================
      Affine                         ✓                                                                                           
      RNN                                      Not yet implemented.                                                              
      LSTM                                     Not yet implemented.                                                              
      GRU                                      Not yet implemented.                                                              
      Convolution                    △         The cases `dilations` and `strides` larger than 1 are not supported by tensorflow.
      FusedConvolution                         Not yet implemented.                                                              
      DepthwiseConvolution           △         The cases `dilations` and `strides` larger than 1 are not supported by tensorflow.
      Deconvolution                  △         The cases `dilations` larger than 1 are not supported by tensorflow.              
      DepthwiseDeconvolution         △         The cases `dilations` larger than 1 are not supported by tensorflow.              
      DeformableConvolution                    Not yet implemented.                                                              
      AdaptiveSeparableConvolution             Not yet implemented.                                                              
      MaxPooling                     ✓                                                                                           
      AveragePooling                 △         Currently only supports the cases both ignore_border and including_pad are True.  
      GlobalAveragePooling           ✓                                                                                           
      SumPooling                     ✓                                                                                           
      Unpooling                      △         The kernel only supports 2d.                                                      
      Embed                          ✓                                                                                           
      RoiAlign                                 Not yet implemented.                                                              
    ===============================  ========  ==================================================================================


Neural Network Activation Functions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Count 20/22


    =================  ========  ====================
     NNabla Function    Status       Description     
    =================  ========  ====================
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
      PReLU            ✓                             
      GELU             ✓                             
      Mish                       Not yet implemented.
      ReLU6            ✓                             
      HardSigmoid      ✓                             
      HardTanh         ✓                             
      LogSigmoid       ✓                             
      SoftPlus         X         Not yet implemented.
      SoftSign         ✓                             
      TanhShrink       ✓                             
      Sinc             ✓                             
    =================  ========  ====================


Normalization
^^^^^^^^^^^^^

Count 7/14


    ==========================  ========  ====================
         NNabla Function         Status       Description     
    ==========================  ========  ====================
      FusedBatchNormalization   ✓                             
      BatchNormalization        ✓                             
      GroupNormalization                  Not yet implemented.
      InstanceNormalization     ✓                             
      LayerNormalization        ✓                             
      NormNormalization                   Not yet implemented.
      SyncBatchNormalization              Not yet implemented.
      TensorNormalization                 Not yet implemented.
      WeightNormalization       ✓                             
      WeightStandardization     ✓                             
      SpectralNorm              ✓                             
      MeanSubtraction                     Not yet implemented.
      ClipGradByValue                     Not yet implemented.
      ClipGradByNorm                      Not yet implemented.
    ==========================  ========  ====================


Reduction
^^^^^^^^^

Count 5/10


    =================  ========  ====================
     NNabla Function    Status       Description     
    =================  ========  ====================
      Sum              ✓                             
      CumSum                     Not yet implemented.
      Mean             ✓                             
      Max              ✓                             
      Min              ✓                             
      Norm                       Not yet implemented.
      Prod             ✓                             
      CumProd                    Not yet implemented.
      ReduceSum                  Not yet implemented.
      ReduceMean                 Not yet implemented.
    =================  ========  ====================


Arithmetic
^^^^^^^^^^

Count 11/14


    =================  ========  ====================
     NNabla Function    Status       Description     
    =================  ========  ====================
      Add2             ✓                             
      AddN                       Not yet implemented.
      BcAdd2                     Not yet implemented.
      Sub2             ✓                             
      Mul2             ✓                             
      MulN                       Not yet implemented.
      Div2             ✓                             
      Pow2             ✓                             
      AddScalar        ✓                             
      MulScalar        ✓                             
      PowScalar        ✓                             
      RSubScalar       ✓                             
      RDivScalar       ✓                             
      RPowScalar       ✓                             
    =================  ========  ====================


Logical
^^^^^^^

Count 29/30


    =====================  ========  ====================
       NNabla Function      Status       Description     
    =====================  ========  ====================
      Sign                 ✓                             
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
      SearchSorted                   Not yet implemented.
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
      IsNaN                ✓                             
      IsInf                ✓                             
      ResetNaN             ✓                             
      ResetInf             ✓                             
      Where                ✓                             
    =====================  ========  ====================


Math
^^^^

Count 22/22


    =================  ========  =============
     NNabla Function    Status    Description 
    =================  ========  =============
      Constant         ✓                      
      Arange           ✓                      
      Abs              ✓                      
      Exp              ✓                      
      Log              ✓                      
      Identity         ✓                      
      BatchMatmul      ✓                      
      Round            ✓                      
      Ceil             ✓                      
      Floor            ✓                      
      Sin              ✓                      
      Cos              ✓                      
      Tan              ✓                      
      Sinh             ✓                      
      Cosh             ✓                      
      ASin             ✓                      
      ACos             ✓                      
      ATan             ✓                      
      ATan2            ✓                      
      ASinh            ✓                      
      ACosh            ✓                      
      ATanh            ✓                      
    =================  ========  =============


Array Manipulation
^^^^^^^^^^^^^^^^^^

Count 12/30


    =====================  ========  ================================================================================================================
       NNabla Function      Status                                                     Description                                                   
    =====================  ========  ================================================================================================================
      Concatenate          ✓                                                                                                                         
      Split                ✓                                                                                                                         
      Stack                ✓                                                                                                                         
      Slice                ✓                                                                                                                         
      Pad                  △         When the mode of the pad is reflect, if the size of the pad exceeds the input size, tensorflow cannot handle it.
      Transpose            ✓                                                                                                                         
      Broadcast            ✓                                                                                                                         
      BroadcastTo          ✓                                                                                                                         
      Tile                 ✓                                                                                                                         
      OneHot               ✓                                                                                                                         
      Flip                 ✓                                                                                                                         
      Shift                          Not yet implemented.                                                                                            
      Sort                           Not yet implemented.                                                                                            
      Reshape              ✓                                                                                                                         
      MatrixDiag                     Not yet implemented.                                                                                            
      MatrixDiagPart                 Not yet implemented.                                                                                            
      Meshgrid                       Not yet implemented.                                                                                            
      BatchDet                       Not yet implemented.                                                                                            
      BatchInv                       Not yet implemented.                                                                                            
      BatchLogdet                    Not yet implemented.                                                                                            
      Assign                         Not yet implemented.                                                                                            
      Gather                         Not yet implemented.                                                                                            
      GatherNd                       Not yet implemented.                                                                                            
      BoolGather                     Not yet implemented.                                                                                            
      ScatterNd                      Not yet implemented.                                                                                            
      ScatterAdd                     Not yet implemented.                                                                                            
      BoolScatter                    Not yet implemented.                                                                                            
      BoolFill                       Not yet implemented.                                                                                            
      PackPaddedSequence             Not yet implemented.                                                                                            
      PadPackedSequence              Not yet implemented.                                                                                            
    =====================  ========  ================================================================================================================


Signal Processing
^^^^^^^^^^^^^^^^^

Count 1/5


    =================  ========  ====================
     NNabla Function    Status       Description     
    =================  ========  ====================
      Interpolate      △                             
      FFT                        Not yet implemented.
      IFFT                       Not yet implemented.
      STFT                       Not yet implemented.
      ISTFT                      Not yet implemented.
    =================  ========  ====================


Stochasticity
^^^^^^^^^^^^^

Count 0/15


    ====================  ========  ========================================================================================================================
      NNabla Function      Status                                                         Description                                                       
    ====================  ========  ========================================================================================================================
      Dropout             X         The Dropout in nnabla has no test mode and contains random parameters, so the test result is not the same as tensorflow.
      TopKData                      Not yet implemented.                                                                                                    
      TopKGrad                      Not yet implemented.                                                                                                    
      Rand                          Not yet implemented.                                                                                                    
      Randint                       Not yet implemented.                                                                                                    
      Randn                         Not yet implemented.                                                                                                    
      RandBinomial                  Not yet implemented.                                                                                                    
      RandBeta                      Not yet implemented.                                                                                                    
      RandGamma                     Not yet implemented.                                                                                                    
      RandomChoice                  Not yet implemented.                                                                                                    
      RandomCrop                    Not yet implemented.                                                                                                    
      RandomFlip                    Not yet implemented.                                                                                                    
      RandomShift                   Not yet implemented.                                                                                                    
      RandomErase                   Not yet implemented.                                                                                                    
      ImageAugmentation             Not yet implemented.                                                                                                    
    ====================  ========  ========================================================================================================================


Loss Functions
^^^^^^^^^^^^^^

Count 0/9


    ==========================  ========  ====================
         NNabla Function         Status       Description     
    ==========================  ========  ====================
      SigmoidCrossEntropy                 Not yet implemented.
      BinaryCrossEntropy                  Not yet implemented.
      SoftmaxCrossEntropy                 Not yet implemented.
      CategoricalCrossEntropy             Not yet implemented.
      SquaredError                        Not yet implemented.
      AbsoluteError                       Not yet implemented.
      HuberLoss                           Not yet implemented.
      EpsilonInsensitiveLoss              Not yet implemented.
      KLMultinomial                       Not yet implemented.
    ==========================  ========  ====================


Geometric Neural Network Layers
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Count 0/3


    =================  ========  ====================
     NNabla Function    Status       Description     
    =================  ========  ====================
      AffineGrid                 Not yet implemented.
      WarpByGrid                 Not yet implemented.
      WarpByFlow                 Not yet implemented.
    =================  ========  ====================


Quantization Neural Network Layers
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Count 6/14


    ===========================  ========  ==================================================================================
          NNabla Function         Status                                      Description                                    
    ===========================  ========  ==================================================================================
      BinarySigmoid              ✓                                                                                           
      BinaryTanh                 ✓                                                                                           
      BinaryConnectAffine        ✓                                                                                           
      BinaryConnectConvolution   △         The cases `dilations` and `strides` larger than 1 are not supported by tensorflow.
      BinaryWeightAffine         ✓                                                                                           
      BinaryWeightConvolution    △         The cases `dilations` and `strides` larger than 1 are not supported by tensorflow.
      INQAffine                            Not yet implemented.                                                              
      INQConvolution                       Not yet implemented.                                                              
      FixedPointQuantize                   Not yet implemented.                                                              
      MinMaxQuantize                       Not yet implemented.                                                              
      Pow2Quantize                         Not yet implemented.                                                              
      Prune                                Not yet implemented.                                                              
      QuantizeLinear                       Not yet implemented.                                                              
      DequantizeLinear                     Not yet implemented.                                                              
    ===========================  ========  ==================================================================================


Validation
^^^^^^^^^^

Count 0/3


    ==================  ========  ====================
     NNabla Function     Status       Description     
    ==================  ========  ====================
      TopNError                   Not yet implemented.
      BinaryError                 Not yet implemented.
      ConfusionMatrix             Not yet implemented.
    ==================  ========  ====================


Unsupported, Special Use
^^^^^^^^^^^^^^^^^^^^^^^^

Count 0/6


    =====================  ========  ====================
       NNabla Function      Status       Description     
    =====================  ========  ====================
      VATNoise                       Not yet implemented.
      Unlink                         Not yet implemented.
      Sink                           Not yet implemented.
      NmsDetection2d                 Not yet implemented.
      MaxPoolingBackward             Not yet implemented.
      PatchCorrelation               Not yet implemented.
    =====================  ========  ====================




Tensorflow Lite Support Status
==============================


Export
------

- ✓: Supported
- △: Partially supported
- X: Supported, but test failed.
- Empty: Not support yet.


Total: 82/215

Neural Network Layer
^^^^^^^^^^^^^^^^^^^^

Count 11/18


    ===============================  ========
            NNabla Function           Status 
    ===============================  ========
      Affine                         ✓       
      RNN                                    
      LSTM                                   
      GRU                                    
      Convolution                    △       
      FusedConvolution                       
      DepthwiseConvolution           ✓       
      Deconvolution                  △       
      DepthwiseDeconvolution         △       
      DeformableConvolution                  
      AdaptiveSeparableConvolution           
      MaxPooling                     △       
      AveragePooling                 △       
      GlobalAveragePooling           ✓       
      SumPooling                     △       
      Unpooling                      △       
      Embed                          ✓       
      RoiAlign                               
    ===============================  ========


Neural Network Activation Functions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Count 10/22


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
      ELU              △       
      SELU             X       
      CReLU            X       
      CELU             X       
      PReLU            ✓       
      GELU             X       
      Mish                     
      ReLU6            ✓       
      HardSigmoid      X       
      HardTanh         X       
      LogSigmoid       X       
      SoftPlus         X       
      SoftSign         X       
      TanhShrink       X       
      Sinc             X       
    =================  ========


Normalization
^^^^^^^^^^^^^

Count 1/14


    ==========================  ========
         NNabla Function         Status 
    ==========================  ========
      FusedBatchNormalization   X       
      BatchNormalization        ✓       
      GroupNormalization                
      InstanceNormalization     X       
      LayerNormalization        X       
      NormNormalization                 
      SyncBatchNormalization            
      TensorNormalization               
      WeightNormalization       X       
      WeightStandardization     X       
      SpectralNorm              X       
      MeanSubtraction                   
      ClipGradByValue                   
      ClipGradByNorm                    
    ==========================  ========


Reduction
^^^^^^^^^

Count 5/10


    =================  ========
     NNabla Function    Status 
    =================  ========
      Sum              ✓       
      CumSum                   
      Mean             ✓       
      Max              ✓       
      Min              ✓       
      Norm                     
      Prod             ✓       
      CumProd                  
      ReduceSum                
      ReduceMean               
    =================  ========


Arithmetic
^^^^^^^^^^

Count 11/14


    =================  ========
     NNabla Function    Status 
    =================  ========
      Add2             ✓       
      AddN                     
      BcAdd2                   
      Sub2             ✓       
      Mul2             ✓       
      MulN                     
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

Count 23/30


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
      SearchSorted                 
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
      Where                X       
    =====================  ========


Math
^^^^

Count 10/22


    =================  ========
     NNabla Function    Status 
    =================  ========
      Constant         X       
      Arange           X       
      Abs              ✓       
      Exp              ✓       
      Log              ✓       
      Identity         X       
      BatchMatmul      ✓       
      Round            ✓       
      Ceil             ✓       
      Floor            ✓       
      Sin              ✓       
      Cos              ✓       
      Tan              ✓       
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

Count 10/30


    =====================  ========
       NNabla Function      Status 
    =====================  ========
      Concatenate          ✓       
      Split                ✓       
      Stack                ✓       
      Slice                △       
      Pad                  △       
      Transpose            △       
      Broadcast            △       
      BroadcastTo          X       
      Tile                 ✓       
      OneHot               X       
      Flip                 ✓       
      Shift                        
      Sort                         
      Reshape              ✓       
      MatrixDiag                   
      MatrixDiagPart               
      Meshgrid                     
      BatchDet                     
      BatchInv                     
      BatchLogdet                  
      Assign                       
      Gather                       
      GatherNd                     
      BoolGather                   
      ScatterNd                    
      ScatterAdd                   
      BoolScatter                  
      BoolFill                     
      PackPaddedSequence           
      PadPackedSequence            
    =====================  ========


Signal Processing
^^^^^^^^^^^^^^^^^

Count 1/5


    =================  ========
     NNabla Function    Status 
    =================  ========
      Interpolate      △       
      FFT                      
      IFFT                     
      STFT                     
      ISTFT                    
    =================  ========


Stochasticity
^^^^^^^^^^^^^

Count 0/15


    ====================  ========
      NNabla Function      Status 
    ====================  ========
      Dropout             X       
      TopKData                    
      TopKGrad                    
      Rand                        
      Randint                     
      Randn                       
      RandBinomial                
      RandBeta                    
      RandGamma                   
      RandomChoice                
      RandomCrop                  
      RandomFlip                  
      RandomShift                 
      RandomErase                 
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


Geometric Neural Network Layers
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Count 0/3


    =================  ========
     NNabla Function    Status 
    =================  ========
      AffineGrid               
      WarpByGrid               
      WarpByFlow               
    =================  ========


Quantization Neural Network Layers
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Count 0/14


    ===========================  ========
          NNabla Function         Status 
    ===========================  ========
      BinarySigmoid              X       
      BinaryTanh                 X       
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
      QuantizeLinear                     
      DequantizeLinear                   
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

Count 0/6


    =====================  ========
       NNabla Function      Status 
    =====================  ========
      VATNoise                     
      Unlink                       
      Sink                         
      NmsDetection2d               
      MaxPoolingBackward           
      PatchCorrelation             
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

Total: 56/215

Neural Network Layer
^^^^^^^^^^^^^^^^^^^^

Count 8/18


    ===============================  ========  =============
            NNabla Function           Status    Description 
    ===============================  ========  =============
      Affine                         ✓                      
      RNN                                                   
      LSTM                                                  
      GRU                                                   
      Convolution                    ✓                      
      FusedConvolution                                      
      DepthwiseConvolution           ✓                      
      Deconvolution                  ✓                      
      DepthwiseDeconvolution                                
      DeformableConvolution                                 
      AdaptiveSeparableConvolution                          
      MaxPooling                     ✓                      
      AveragePooling                 ✓                      
      GlobalAveragePooling                                  
      SumPooling                     ✓                      
      Unpooling                      ✓                      
      Embed                                                 
      RoiAlign                                              
    ===============================  ========  =============


Neural Network Activation Functions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Count 11/22


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
      Mish                                    
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

Count 1/14


    ==========================  ========  =============
         NNabla Function         Status    Description 
    ==========================  ========  =============
      FusedBatchNormalization                          
      BatchNormalization        ✓                      
      GroupNormalization                               
      InstanceNormalization                            
      LayerNormalization                               
      NormNormalization                                
      SyncBatchNormalization                           
      TensorNormalization       X                      
      WeightNormalization                              
      WeightStandardization                            
      SpectralNorm                                     
      MeanSubtraction           X                      
      ClipGradByValue                                  
      ClipGradByNorm                                   
    ==========================  ========  =============


Reduction
^^^^^^^^^

Count 1/10


    =================  ========  =============
     NNabla Function    Status    Description 
    =================  ========  =============
      Sum              ✓                      
      CumSum                                  
      Mean                                    
      Max                                     
      Min                                     
      Norm                                    
      Prod                                    
      CumProd                                 
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

Count 5/30


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
      SearchSorted                                
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

Count 7/30


    =====================  ========  =============
       NNabla Function      Status    Description 
    =====================  ========  =============
      Concatenate          ✓                      
      Split                ✓                      
      Stack                ✓                      
      Slice                ✓                      
      Pad                                         
      Transpose            ✓                      
      Broadcast                                   
      BroadcastTo                                 
      Tile                                        
      OneHot                                      
      Flip                 ✓                      
      Shift                X                      
      Sort                                        
      Reshape              ✓                      
      MatrixDiag           X                      
      MatrixDiagPart       X                      
      Meshgrid                                    
      BatchDet                                    
      BatchInv                                    
      BatchLogdet                                 
      Assign                                      
      Gather                                      
      GatherNd                                    
      BoolGather                                  
      ScatterNd                                   
      ScatterAdd                                  
      BoolScatter                                 
      BoolFill                                    
      PackPaddedSequence                          
      PadPackedSequence                           
    =====================  ========  =============


Signal Processing
^^^^^^^^^^^^^^^^^

Count 0/5


    =================  ========  =============
     NNabla Function    Status    Description 
    =================  ========  =============
      Interpolate                             
      FFT                                     
      IFFT                                    
      STFT                                    
      ISTFT                                   
    =================  ========  =============


Stochasticity
^^^^^^^^^^^^^

Count 0/15


    ====================  ========  =============
      NNabla Function      Status    Description 
    ====================  ========  =============
      Dropout             X                      
      TopKData                                   
      TopKGrad                                   
      Rand                                       
      Randint                                    
      Randn                                      
      RandBinomial                               
      RandBeta                                   
      RandGamma                                  
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


Geometric Neural Network Layers
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Count 0/3


    =================  ========  =============
     NNabla Function    Status    Description 
    =================  ========  =============
      AffineGrid                              
      WarpByGrid                              
      WarpByFlow                              
    =================  ========  =============


Quantization Neural Network Layers
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Count 6/14


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
      QuantizeLinear                                    
      DequantizeLinear                                  
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

Count 0/6


    =====================  ========  =============
       NNabla Function      Status    Description 
    =====================  ========  =============
      VATNoise                                    
      Unlink                                      
      Sink                                        
      NmsDetection2d                              
      MaxPoolingBackward                          
      PatchCorrelation                            
    =====================  ========  =============



