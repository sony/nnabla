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
     Cast                        ✓                        ✓              X                Abs, Log                                                                                                                                                                                                                                                                                                                       
     Ceil                        ✓                        ✓                               Ceil                                                                                                                                                                                                                                                                                                                           
     Clip                        ✓                        ✓                         ✓     MinimumScalar, Identity, MaximumScalar                                                                                                                                                                                                                                                                                         
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
     Expand                                                         ✓    ✓                Broadcast, Reshape                                                                                                                                                                                                                                                                                                             
     EyeLike                                                             X                                                                                 Not yet implemented.                                                                                                                                                                                                                                          
     Flatten                     ✓                        ✓              ✓          ✓     Reshape                                                                                                                                                                                                                                                                                                                        
     Floor                       ✓                        ✓                               Floor                                                                                                                                                                                                                                                                                                                          
     GRU                         X         X                   X                                                                                           Not yet implemented.                                                                                                                                                                                                                                          
     Gather                      ✓                        ✓                         ✓     Concatenate, Slice                                                                                                                                                                                                                                                                                                             
     GatherElements                                                                 X                                                                      Not yet implemented.                                                                                                                                                                                                                                          
     GatherND                                                                       X                                                                      Not yet implemented.                                                                                                                                                                                                                                          
     Gemm                        ✓                        ✓    ✓         ✓          ✓     Reshape, BatchMatmul, Add2, MulScalar                                                                                                                                                                                                                                                                                          
     GlobalAveragePool           ✓                        ✓                               GlobalAveragePooling                                                                                                                                                                                                                                                                                                           
     GlobalLpPool                X    X                                                                                                                    Not yet implemented.                                                                                                                                                                                                                                          
     GlobalMaxPool               X                                                                                                                         Not yet implemented.                                                                                                                                                                                                                                          
     Greater                     ✓                        ✓    ✓         ✓                Reshape, Greater                                                                                                                                                                                                                                                                                                               
     HardSigmoid                 ✓                        ✓                               MinimumScalar, AddScalar, MaximumScalar, HardSigmoid, MulScalar                                                                                                                                                                                                                                                                
     Hardmax                     ✓                        ✓                         ✓     Max, OneHot, Reshape                                                                                                                                                                                                                                                                                                           
     Identity                    ✓                        ✓                               Identity                                                                                                                                                                                                                                                                                                                       
     If                          X                                                                                                                         Not yet implemented.                                                                                                                                                                                                                                          
     InstanceNormalization       ✓                        ✓                               Concatenate, Reshape, BatchNormalization, Split                                                                                                                                                                                                                                                                                
     IsInf                                                                    ✓           IsInf                                                                                                                                                                                                                                                                                                                          
     IsNaN                                                               ✓                IsNaN                                                                                                                                                                                                                                                                                                                          
     LRN                         ✓                        ✓                               PowScalar, Transpose, AddScalar, Div2, SumPooling, MulScalar                                                                                                                                                                                                                                                                   
     LSTM                        X                             X                                                                                           Not yet implemented.                                                                                                                                                                                                                                          
     LeakyRelu                   ✓                        ✓                               LeakyReLU                                                                                                                                                                                                                                                                                                                      
     Less                        ✓                        ✓    ✓         ✓                Reshape, Less                                                                                                                                                                                                                                                                                                                  
     Log                         ✓                        ✓                               Log                                                                                                                                                                                                                                                                                                                            
     LogSoftmax                  ✓                        ✓                         ✓     Sub2, Add2, Max, Log, Exp, Reshape, Sum                                                                                                                                                                                                                                                                                        
     Loop                        X                                                  X                                                                      Not yet implemented.                                                                                                                                                                                                                                          
     LpNormalization             X                                                                                                                         Not yet implemented.                                                                                                                                                                                                                                          
     LpPool                      X    X                                             X                                                                      Not yet implemented.                                                                                                                                                                                                                                          
     MatMul                      ✓                        ✓              ✓                BatchMatmul                                                                                                                                                                                                                                                                                                                    
     MatMulInteger                                                            X                                                                            Not yet implemented.                                                                                                                                                                                                                                          
     Max                         ✓                        ✓         ✓    ✓                Maximum2                                                                                                                                                                                                                                                                                                                       
     MaxPool                     ✓                        ✓         X         X     X     Pad, MaxPooling                                                  Not all features are verified, since some features are not supported by caffe2. Those features can be verified by ONNXRuntime. if opset >= 10, the ceil_mode is not supported, dilations is not equal to 1 is not supported.                                  
     MaxRoiPool                  X                                                                                                                         Not yet implemented.                                                                                                                                                                                                                                          
     MaxUnpool                                                           X          X                                                                      Not yet implemented.                                                                                                                                                                                                                                          
     Mean                        ✓                        ✓         ✓    ✓                Broadcast, Mean, Stack                                                                                                                                                                                                                                                                                                         
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
     Softmax                     ✓                        ✓                         ✓     Sub2, Max, Div2, Exp, Reshape, Sum                                                                                                                                                                                                                                                                                             
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
     Xor                         ✓                        ✓    ✓                          Reshape, LogicalXor                                                                                                                                                                                                                                                                                                            
    ===========================  ===  ===  ===  ===  ===  ===  ===  ===  ===  ====  ====  ===============================================================  ==============================================================================================================================================================================================================================================================



Export
------

- ✓: Support to export this opset.
- △: Partially support to export this opset (e.g. some cases cannot be supported, or not completely tested).
- X: Supported, but test failed.
- Empty: Not support corresponding opset version.

Total: 119/173

Neural Network Layer
^^^^^^^^^^^^^^^^^^^^

Count 11/14
 

    =========================  ===  ===  ===  ====  ====  ========================================  ======================================================================================
         NNabla Function        6    7    9    10    11                   ONNX Op                                                        Description                                      
    =========================  ===  ===  ===  ====  ====  ========================================  ======================================================================================
      Affine                   ✓    ✓    ✓    ✓     ✓     Gemm, Reshape                                                                                                                   
      RNN                                                                                           Not yet implemented.                                                                  
      LSTM                                                                                          Not yet implemented.                                                                  
      GRU                                                                                           Not yet implemented.                                                                  
      Convolution              ✓    ✓    ✓    ✓     ✓     Conv, Reshape                                                                                                                   
      DepthwiseConvolution     ✓    ✓    ✓    ✓     ✓     Conv, Reshape                                                                                                                   
      Deconvolution            ✓    ✓    ✓    ✓     ✓     ConvTranspose, Reshape                                                                                                          
      DepthwiseDeconvolution   ✓    ✓    ✓    ✓     ✓     ConvTranspose, Reshape                                                                                                          
      MaxPooling               ✓    ✓    ✓    ✓     X     Pad, MaxPool, Reshape                                                                                                           
      AveragePooling           △    △    △    △     X     Pad, Reshape, AveragePool                 Currently only supports the cases where both ignore_border and including_pad are True.
      GlobalAveragePooling     ✓    ✓    ✓    ✓     ✓     GlobalAveragePool                                                                                                               
      SumPooling               X    ✓    ✓    ✓     X     Mul, Pad, Constant, Reshape, AveragePool                                                                                        
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
      Swish            ✓    ✓    ✓    ✓     ✓     Sigmoid, Mul                                                                                                               
      Tanh             ✓    ✓    ✓    ✓     ✓     Tanh                                                                                                                       
      ReLU             ✓    ✓    ✓    ✓     ✓     Relu                                                                                                                       
      LeakyReLU        ✓    ✓    ✓    ✓     ✓     LeakyRelu                                                                                                                  
      Softmax          △    ✓    ✓    ✓     ✓     Div, ReduceSum, ReduceMax, Sub, Exp       ONNX Add, Sub operator does not support multidirectional broadcasting on opset 6.
      LogSoftmax       △    ✓    ✓    ✓     ✓     ReduceSum, ReduceMax, Sub, Log, Exp                                                                                        
      ELU              ✓    ✓    ✓    ✓     ✓     Elu                                                                                                                        
      SELU             ✓    ✓    ✓    ✓     ✓     Selu                                                                                                                       
      CReLU            ✓    ✓    ✓    ✓     ✓     Concat, Neg, Relu                                                                                                          
      CELU             ✓    ✓    ✓    ✓     ✓     Neg, Elu, Concat                                                                                                           
      PReLU            ✓    ✓    ✓    ✓     ✓     Reshape, PRelu                                                                                                             
      GELU             ✓    ✓    ✓    ✓     ✓     Div, Tanh, Mul, Pow, Sqrt, Constant, Add                                                                                   
      ReLU6            ✓    ✓    ✓    ✓     ✓     Min, Constant, Relu                                                                                                        
      HardSigmoid      ✓    ✓    ✓    ✓     ✓     HardSigmoid                                                                                                                
      HardTanh         ✓    ✓    ✓    ✓     ✓     Max, Neg, Min, Constant                                                                                                    
      LogSigmoid       ✓    ✓    ✓    ✓     ✓     Sigmoid, Log                                                                                                               
      SoftPlus         ✓    ✓    ✓    ✓     ✓     Softplus                                                                                                                   
      SoftSign         ✓    ✓    ✓    ✓     ✓     Softsign                                                                                                                   
      TanhShrink       ✓    ✓    ✓    ✓     ✓     Tanh, Sub                                                                                                                  
      Sinc             X    X    ✓    ✓     ✓     Div, Sin, Equal, Where, Constant                                                                                           
    =================  ===  ===  ===  ====  ====  ========================================  =================================================================================


Normalization
^^^^^^^^^^^^^

Count 1/6
 

    ==========================  ===  ===  ===  ====  ====  ===========================================================================  =======================================================================================================
         NNabla Function         6    7    9    10    11                                     ONNX Op                                                                                  Description                                              
    ==========================  ===  ===  ===  ====  ====  ===========================================================================  =======================================================================================================
      FusedBatchNormalization   X    X    X    X     X     Div, ReduceSum, ReduceMean, Mul, Sub, Constant, Relu, BatchNormalization     Not yet implemented.                                                                                   
      BatchNormalization        ✓    ✓    ✓    ✓     ✓     Div, ReduceSum, ReduceMean, Mul, Sub, Constant, Reshape, BatchNormalization  In inferring stage, caffe2 mistmatch onnx 1.4.x's implementation, "in-place" feature cannot be applied.
      SyncBatchNormalization                                                                                                            Not yet implemented.                                                                                   
      MeanSubtraction                                                                                                                   Not yet implemented.                                                                                   
      ClipGradByValue                                                                                                                   Not yet implemented.                                                                                   
      ClipGradByNorm                                                                                                                    Not yet implemented.                                                                                   
    ==========================  ===  ===  ===  ====  ====  ===========================================================================  =======================================================================================================


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
      PowScalar        ✓    ✓    ✓    ✓     ✓     Pow, Constant                                                                              
      RSubScalar       ✓    ✓    ✓    ✓     ✓     Sub, Constant                                                                              
      RDivScalar       ✓    ✓    ✓    ✓     ✓     Div, Constant                                                                              
      RPowScalar       ✓    ✓    ✓    ✓     ✓     Pow, Constant                                                                              
    =================  ===  ===  ===  ====  ====  =============  ============================================================================


Logical
^^^^^^^

Count 29/29
 

    =====================  ===  ===  ===  ====  ====  ======================  ============================================================================
       NNabla Function      6    7    9    10    11          ONNX Op                                          Description                                 
    =====================  ===  ===  ===  ====  ====  ======================  ============================================================================
      Sign                 X    X    ✓    ✓     ✓     Sign                                                                                                
      Minimum2             △    ✓    ✓    ✓     ✓     Constant, Min, Add      ONNX Add operator does not support multidirectional broadcasting on opset 6.
      Maximum2             △    ✓    ✓    ✓     ✓     Max, Constant, Add      ONNX Add operator does not support multidirectional broadcasting on opset 6.
      MinimumScalar        ✓    ✓    ✓    ✓     ✓     Constant, Min, Add                                                                                  
      MaximumScalar        ✓    ✓    ✓    ✓     ✓     Max, Constant, Add                                                                                  
      LogicalAnd           ✓    ✓    ✓    ✓     ✓     And                                                                                                 
      LogicalOr            ✓    ✓    ✓    ✓     ✓     Or                                                                                                  
      LogicalXor           ✓    ✓    ✓    ✓     ✓     Xor                                                                                                 
      Equal                ✓    ✓    ✓    ✓     ✓     Equal                                                                                               
      NotEqual             ✓    ✓    ✓    ✓     ✓     Not, Equal                                                                                          
      GreaterEqual         ✓    ✓    ✓    ✓     ✓     Not, Less                                                                                           
      Greater              ✓    ✓    ✓    ✓     ✓     Greater                                                                                             
      LessEqual            ✓    ✓    ✓    ✓     ✓     Greater, Not                                                                                        
      Less                 ✓    ✓    ✓    ✓     ✓     Less                                                                                                
      LogicalAndScalar     ✓    ✓    ✓    ✓     ✓     Constant, And                                                                                       
      LogicalOrScalar      ✓    ✓    ✓    ✓     ✓     Or, Constant                                                                                        
      LogicalXorScalar     ✓    ✓    ✓    ✓     ✓     Xor, Constant                                                                                       
      EqualScalar          ✓    ✓    ✓    ✓     ✓     Constant, Equal                                                                                     
      NotEqualScalar       ✓    ✓    ✓    ✓     ✓     Constant, Not, Equal                                                                                
      GreaterEqualScalar   ✓    ✓    ✓    ✓     ✓     Not, Constant, Less                                                                                 
      GreaterScalar        ✓    ✓    ✓    ✓     ✓     Greater, Constant                                                                                   
      LessEqualScalar      ✓    ✓    ✓    ✓     ✓     Greater, Not, Constant                                                                              
      LessScalar           ✓    ✓    ✓    ✓     ✓     Constant, Less                                                                                      
      LogicalNot           ✓    ✓    ✓    ✓     ✓     Not                                                                                                 
      IsNaN                X    X    ✓    ✓     ✓     IsNaN                                                                                               
      IsInf                X    X    X    ✓     ✓     IsInf                                                                                               
      ResetNaN             X    X    ✓    ✓     ✓     Where, IsNaN, Constant                                                                              
      ResetInf             X    X    X    ✓     ✓     Where, IsInf, Constant                                                                              
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
      BatchMatmul      ✓    ✓    ✓    ✓     ✓     MatMul, Reshape, Transpose               
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
    =================  ===  ===  ===  ====  ====  ==========================  =============


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
      Pad              △    △    △    △     △     Pad, Constant                When the mode of the pad is reflect, if the size of the pad exceeds the input size, caffe2 and onnxruntime cannot handle it.
      Transpose        ✓    ✓    ✓    ✓     ✓     Transpose                                                                                                                                                
      Broadcast        X    X    ✓    ✓     ✓                                                                                                                                                              
      BroadcastTo      ✓    ✓    ✓    ✓     ✓                                                                                                                                                              
      Tile             ✓    ✓    ✓    ✓     ✓     Tile, Reshape, Constant                                                                                                                                  
      OneHot           ✓    ✓    ✓    ✓     ✓     Reshape, Gather, Flatten                                                                                                                                 
      Flip             ✓    ✓    ✓    ✓     ✓     Transpose, Identity, Gather                                                                                                                              
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
      BinarySigmoid              X    X    ✓    ✓     ✓     Where, Greater, Constant                       
      BinaryTanh                 X    X    ✓    ✓     ✓     Where, Greater, Constant                       
      BinaryConnectAffine        ✓    ✓    ✓    ✓     ✓     Gemm, Reshape                                  
      BinaryConnectConvolution   ✓    ✓    ✓    ✓     ✓     Conv, Reshape                                  
      BinaryWeightAffine         ✓    ✓    ✓    ✓     ✓     MatMul, Reshape, Mul, Add                      
      BinaryWeightConvolution    ✓    ✓    ✓    ✓     ✓     Conv, Reshape, Mul, Add                        
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


Total: 85/120

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
      BatchMatMul                              ✓      BatchMatmul, Transpose                                                                                                                     
      BiasAdd                                  ✓      Reshape, Add2                                                                                                                              
      Cast                                                                                Not yet implemented.                                                                                   
      Ceil                                     ✓      Ceil                                                                                                                                       
      ConcatV2                                 ✓      Concatenate                                                                                                                                
      Const                                    ✓      Add2                                                                                                                                       
      Conv2D                                   △      Pad, Convolution, Transpose         Tensorflow require GPU to perform related test cases. This issue is recorded only for memo.            
      Conv2DBackpropFilter                                                                Not yet implemented.                                                                                   
      Conv2DBackpropInput                      △      Deconvolution, Transpose            Tensorflow require GPU to perform related test cases. This issue is recorded only for memo.            
      Conv3D                                                                              Not yet implemented.                                                                                   
      Conv3DBackpropFilterV2                                                              Not yet implemented.                                                                                   
      Conv3DBackpropInputV2                                                               Not yet implemented.                                                                                   
      Cos                                      ✓      Cos                                                                                                                                        
      Cosh                                     ✓      Cosh                                                                                                                                       
      DepthToSpace                             △      Reshape, Transpose                  Tensorflow require GPU to perform related test cases. This issue is recorded only for memo.            
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
      FloorMod                                 ✓      Mul2, Sub2, Floor, Div2                                                                                                                    
      FusedBatchNorm                           X      BatchNormalization, Transpose       It did not pass testing for training mode.                                                             
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
      LogicalXor                               ✓      LogicalNot, LogicalAnd, LogicalOr                                                                                                          
      MatrixBandPart                                                                      Not yet implemented.                                                                                   
      Max                                      ✓      Max                                                                                                                                        
      MaxPool                                  ✓      Pad, MaxPooling, Transpose                                                                                                                 
      MaxPool3D                                                                           Not yet implemented.                                                                                   
      MaxPoolWithArgmax                                                                   Not yet implemented.                                                                                   
      Maximum                                  ✓      Maximum2                                                                                                                                   
      Mean                                     ✓      Mean                                                                                                                                       
      Min                                      ✓      Min                                                                                                                                        
      Minimum                                  ✓      Minimum2                                                                                                                                   
      Mul                                      ✓      Mul2                                                                                                                                       
      Neg                                      ✓      MulScalar                                                                                                                                  
      NotEqual                                 ✓      LogicalNot, Equal                                                                                                                          
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
      SpaceToDepth                             △      Reshape, Transpose                  Tensorflow require GPU to perform related test cases. This issue is recorded only for memo.            
      SplitV                                   ✓      Split, Stack                                                                                                                               
      Sqrt                                     ✓      PowScalar                                                                                                                                  
      Square                                   ✓      Mul2                                                                                                                                       
      SquaredDifference                        ✓      Mul2, Sub2                                                                                                                                 
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
      Unpack                                   ✓      Concatenate, Reshape, Split, Stack                                                                                                         
    ======================================  ========  ==================================  =======================================================================================================





Export
------

- ✓: Supported
- △: Partially supported
- X: Supported, but test failed.
- Empty: Not support yet.


Total: 114/173

Neural Network Layer
^^^^^^^^^^^^^^^^^^^^

Count 11/14
 

    =========================  ========  ================================================================================================================================================  ==================================================================================
         NNabla Function        Status                                                                        TF Op                                                                                                           Description                                    
    =========================  ========  ================================================================================================================================================  ==================================================================================
      Affine                   ✓         Mul, Placeholder, Reshape, Add, MatMul, Const                                                                                                                                                                                       
      RNN                                                                                                                                                                                  Not yet implemented.                                                              
      LSTM                                                                                                                                                                                 Not yet implemented.                                                              
      GRU                                                                                                                                                                                  Not yet implemented.                                                              
      Convolution              △         Conv2D, Split, Pad, Transpose, ConcatV2, BatchToSpaceND, Placeholder, Reshape, Add, Identity, Const, SpaceToBatchND                               The cases `dilations` and `strides` larger than 1 are not supported by tensorflow.
      DepthwiseConvolution     △         Conv2D, Split, Pad, Transpose, ConcatV2, BatchToSpaceND, Placeholder, Reshape, Add, Const, SpaceToBatchND                                         The cases `dilations` and `strides` larger than 1 are not supported by tensorflow.
      Deconvolution            △         Split, Pad, Slice, Transpose, ConcatV2, Placeholder, Reshape, Add, Identity, Const, Conv2DBackpropInput                                           The cases `dilations` larger than 1 are not supported by tensorflow.              
      DepthwiseDeconvolution   △         Split, Pad, Slice, Transpose, ConcatV2, Placeholder, Reshape, Add, Const, Conv2DBackpropInput                                                     The cases `dilations` larger than 1 are not supported by tensorflow.              
      MaxPooling               ✓         MaxPool, Transpose, Placeholder, Reshape, Const, PadV2, MaxPool3D                                                                                                                                                                   
      AveragePooling           △         AvgPool, Pad, Transpose, AvgPool3D, Placeholder, Reshape, Const                                                                                   Currently only supports the cases both ignore_border and including_pad are True.  
      GlobalAveragePooling     ✓         SplitV, Mean, Sub, Range, Const, Pack                                                                                                                                                                                               
      SumPooling               ✓         AvgPool, Mul, Pad, Transpose, AvgPool3D, Placeholder, Reshape, Const                                                                                                                                                                
      Unpooling                △         Mul, Cast, Equal, NoOp, StridedSlice, Switch, LogicalAnd, Transpose, ResizeNearestNeighbor, Merge, Placeholder, Reshape, Identity, Const, Assert  The kernel only supports 2d.                                                      
      Embed                    ✓         Const, Placeholder, GatherV2                                                                                                                                                                                                        
    =========================  ========  ================================================================================================================================================  ==================================================================================


Neural Network Activation Functions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Count 21/21
 

    =================  ========  ====================================================================================  =============
     NNabla Function    Status                                          TF Op                                           Description 
    =================  ========  ====================================================================================  =============
      Sigmoid          ✓         Sigmoid, Placeholder                                                                               
      Swish            ✓         Sigmoid, Placeholder, Mul                                                                          
      Tanh             ✓         Tanh, Placeholder                                                                                  
      ReLU             ✓         Placeholder, Relu                                                                                  
      LeakyReLU        ✓         LeakyRelu, Placeholder                                                                             
      Softmax          ✓         Sub, Max, RealDiv, Placeholder, Exp, Const, Sum                                                    
      LogSoftmax       ✓         Sub, Max, Log, Placeholder, Exp, Const, Sum                                                        
      ELU              ✓         Mul, Sub, Cast, Elu, GreaterEqual, Placeholder, Add, Less, Exp, Const                              
      SELU             ✓         Maximum, Mul, Sub, Minimum, Placeholder, Add, Exp, Const                                           
      CReLU            ✓         ConcatV2, Neg, Placeholder, Relu, Const                                                            
      CELU             ✓         Mul, Sub, Cast, Elu, GreaterEqual, ConcatV2, Neg, Placeholder, Add, Less, Exp, Const               
      PReLU            ✓         Mul, Sub, Abs, Placeholder, Reshape, Add, Relu, Const                                              
      GELU             ✓         Tanh, Mul, Pow, Sqrt, RealDiv, Placeholder, Add, Const                                             
      ReLU6            ✓         Placeholder, Min, Relu, Const, Pack                                                                
      HardSigmoid      ✓         Maximum, Mul, Minimum, Placeholder, Add, Const                                                     
      HardTanh         ✓         Max, Neg, Placeholder, Min, Const, Pack                                                            
      LogSigmoid       ✓         Sigmoid, Log, Placeholder                                                                          
      SoftPlus         ✓         Softplus, Placeholder                                                                              
      SoftSign         ✓         Placeholder, Softsign                                                                              
      TanhShrink       ✓         Tanh, Placeholder, Sub                                                                             
      Sinc             ✓         Sin, Equal, RealDiv, Placeholder, Const, Select                                                    
    =================  ========  ====================================================================================  =============


Normalization
^^^^^^^^^^^^^

Count 0/6
 

    ==========================  ========  ===========================================================================  =======================================================================================================
         NNabla Function         Status                                      TF Op                                                                                   Description                                              
    ==========================  ========  ===========================================================================  =======================================================================================================
      FusedBatchNormalization   X         Mean, Mul, Sub, RealDiv, Rsqrt, Placeholder, Reshape, Add, Relu, Sum, Const  Not yet implemented.                                                                                   
      BatchNormalization        X         Mean, Mul, Sub, RealDiv, Rsqrt, Placeholder, Reshape, Add, Sum, Const        In inferring stage, caffe2 mistmatch onnx 1.4.x's implementation, "in-place" feature cannot be applied.
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
      Sum              ✓         Sum, Const, Placeholder                       
      Mean             ✓         Const, Placeholder, Mean                      
      Max              ✓         Max, Const, Placeholder                       
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
      Mul2             ✓         Placeholder, Mul                                 
      Div2             ✓         RealDiv, Placeholder                             
      Pow2             ✓         Pow, Placeholder                                 
      AddScalar        ✓         Const, Placeholder, Add                          
      MulScalar        ✓         Const, Placeholder, Mul                          
      PowScalar        ✓         Const, Pow, Placeholder                          
      RSubScalar       ✓         Const, Placeholder, Sub                          
      RDivScalar       ✓         RealDiv, Const, Placeholder                      
      RPowScalar       ✓         Const, Pow, Placeholder                          
    =================  ========  ===========================  ====================


Logical
^^^^^^^

Count 27/29
 

    =====================  ========  =====================================================  ====================
       NNabla Function      Status                           TF Op                              Description     
    =====================  ========  =====================================================  ====================
      Sign                 ✓         Sign, Placeholder                                                          
      Minimum2             ✓         Placeholder, Min, Add, Const, Pack                                         
      Maximum2             ✓         Max, Placeholder, Add, Const, Pack                                         
      MinimumScalar        ✓         Placeholder, Min, Add, Const, Pack                                         
      MaximumScalar        ✓         Max, Placeholder, Add, Const, Pack                                         
      LogicalAnd           ✓         Placeholder, LogicalAnd                                                    
      LogicalOr            ✓         Placeholder, LogicalOr                                                     
      LogicalXor           ✓         LogicalNot, Placeholder, LogicalAnd, LogicalOr                             
      Equal                ✓         Placeholder, Equal                                                         
      NotEqual             ✓         LogicalNot, Placeholder, Equal                                             
      GreaterEqual         ✓         LogicalNot, Placeholder, Less                                              
      Greater              ✓         Greater, Placeholder                                                       
      LessEqual            ✓         LogicalNot, Greater, Placeholder                                           
      Less                 ✓         Placeholder, Less                                                          
      LogicalAndScalar     ✓         Const, Placeholder, LogicalAnd                                             
      LogicalOrScalar      ✓         Const, Placeholder, LogicalOr                                              
      LogicalXorScalar     ✓         LogicalNot, LogicalAnd, LogicalOr, Placeholder, Const                      
      EqualScalar          ✓         Const, Placeholder, Equal                                                  
      NotEqualScalar       ✓         LogicalNot, Const, Placeholder, Equal                                      
      GreaterEqualScalar   ✓         LogicalNot, Const, Placeholder, Less                                       
      GreaterScalar        ✓         Const, Placeholder, Greater                                                
      LessEqualScalar      ✓         LogicalNot, Const, Placeholder, Greater                                    
      LessScalar           ✓         Const, Placeholder, Less                                                   
      LogicalNot           ✓         LogicalNot, Placeholder                                                    
      IsNaN                ✓         IsNan, Placeholder                                                         
      IsInf                X                                                                Not yet implemented.
      ResetNaN             ✓         IsNan, Const, Select, Placeholder                                          
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
      Identity         ✓         Placeholder, Identity                                                      
      BatchMatmul      ✓         Transpose, Placeholder, Reshape, BatchMatMulV2, Const                      
      Round            X                                                                Not yet implemented.
      Ceil             ✓         Ceil, Placeholder                                                          
      Floor            ✓         Placeholder, Floor                                                         
      Sin              ✓         Placeholder, Sin                                                           
      Cos              ✓         Cos, Placeholder                                                           
      Tan              ✓         Placeholder, Tan                                                           
      Sinh             ✓         Placeholder, Sinh                                                          
      Cosh             ✓         Cosh, Placeholder                                                          
      ASin             ✓         Asin, Placeholder                                                          
      ACos             ✓         Placeholder, Acos                                                          
      ATan             ✓         Atan, Placeholder                                                          
      ATan2            ✓         RealDiv, Atan, Placeholder                                                 
      ASinh            ✓         Asinh, Placeholder                                                         
      ACosh            ✓         Acosh, Placeholder                                                         
      ATanh            ✓         Atanh, Placeholder                                                         
    =================  ========  =====================================================  ====================


Array Manipulation
^^^^^^^^^^^^^^^^^^

Count 12/19
 

    =================  ========  =================================================  ================================================================================================================
     NNabla Function    Status                         TF Op                                                                          Description                                                   
    =================  ========  =================================================  ================================================================================================================
      Concatenate      ✓         ConcatV2, Const, Placeholder                                                                                                                                       
      Split            ✓         Squeeze, SplitV, Placeholder, Const                                                                                                                                
      Stack            ✓         ConcatV2, Const, Placeholder, ExpandDims                                                                                                                           
      Slice            △         Const, Placeholder, Slice                          step != 1" exceed the scope of onnx opset 9,  not supported.                                                    
      Pad              △         MirrorPad, Const, PadV2, Placeholder               When the mode of the pad is reflect, if the size of the pad exceeds the input size, tensorflow cannot handle it.
      Transpose        ✓         Const, Placeholder, Transpose                                                                                                                                      
      Broadcast        ✓                                                                                                                                                                            
      BroadcastTo      ✓                                                                                                                                                                            
      Tile             ✓         Tile, Placeholder, Reshape, Const                                                                                                                                  
      OneHot           ✓         Const, Placeholder, Reshape, GatherV2                                                                                                                              
      Flip             ✓         Transpose, Placeholder, Identity, Const, GatherV2                                                                                                                  
      Shift                                                                         Not yet implemented.                                                                                            
      Sort                                                                          Not yet implemented.                                                                                            
      Reshape          ✓         Const, Placeholder, Reshape                                                                                                                                        
      MatrixDiag                                                                    Not yet implemented.                                                                                            
      MatrixDiagPart                                                                Not yet implemented.                                                                                            
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
 

    ===========================  ========  ========================================================================================  ==================================================================================
          NNabla Function         Status                                            TF Op                                                                               Description                                    
    ===========================  ========  ========================================================================================  ==================================================================================
      BinarySigmoid              ✓         Const, Select, Greater, Placeholder                                                                                                                                         
      BinaryTanh                 ✓         Const, Select, Greater, Placeholder                                                                                                                                         
      BinaryConnectAffine        ✓         Mul, Placeholder, Reshape, Add, MatMul, Const                                                                                                                               
      BinaryConnectConvolution   △         Conv2D, Split, Pad, Transpose, ConcatV2, Placeholder, Reshape, Add, Identity, Const       The cases `dilations` and `strides` larger than 1 are not supported by tensorflow.
      BinaryWeightAffine         ✓         Mul, Placeholder, Reshape, Add, MatMul, Const                                                                                                                               
      BinaryWeightConvolution    △         Conv2D, Mul, Split, Pad, Transpose, ConcatV2, Placeholder, Reshape, Add, Identity, Const  The cases `dilations` and `strides` larger than 1 are not supported by tensorflow.
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

Count 5/14
 

    =========================  ========
         NNabla Function        Status 
    =========================  ========
      Affine                   ✓       
      RNN                              
      LSTM                             
      GRU                              
      Convolution              △       
      DepthwiseConvolution     △       
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
      SELU             ✓       
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
      BatchNormalization        ✓       
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

Count 16/29
 

    =====================  ========
       NNabla Function      Status 
    =====================  ========
      Sign                 X       
      Minimum2             ✓       
      Maximum2             ✓       
      MinimumScalar        ✓       
      MaximumScalar        ✓       
      LogicalAnd           X       
      LogicalOr            X       
      LogicalXor           X       
      Equal                ✓       
      NotEqual             ✓       
      GreaterEqual         ✓       
      Greater              ✓       
      LessEqual            ✓       
      Less                 ✓       
      LogicalAndScalar     X       
      LogicalOrScalar      X       
      LogicalXorScalar     X       
      EqualScalar          ✓       
      NotEqualScalar       ✓       
      GreaterEqualScalar   ✓       
      GreaterScalar        ✓       
      LessEqualScalar      ✓       
      LessScalar           ✓       
      LogicalNot           X       
      IsNaN                X       
      IsInf                X       
      ResetNaN             X       
      ResetInf             X       
      Where                X       
    =====================  ========


Math
^^^^

Count 9/22
 

    =================  ========
     NNabla Function    Status 
    =================  ========
      Constant         X       
      Arange           X       
      Abs              ✓       
      Exp              ✓       
      Log              ✓       
      Identity         ✓       
      BatchMatmul      △       
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

Count 10/19
 

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
      OneHot           ✓       
      Flip             △       
      Shift                    
      Sort                     
      Reshape          ✓       
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

Count 4/12
 

    ===========================  ========
          NNabla Function         Status 
    ===========================  ========
      BinarySigmoid              ✓       
      BinaryTanh                 ✓       
      BinaryConnectAffine        ✓       
      BinaryConnectConvolution   X       
      BinaryWeightAffine         ✓       
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



