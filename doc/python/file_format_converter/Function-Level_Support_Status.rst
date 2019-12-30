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


Total: 91/155

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
     Cast                        X                        X              X                                                                                 Not yet implemented.                                                                                                                                                                                                                                          
     Ceil                        ✓                        ✓                               Ceil                                                                                                                                                                                                                                                                                                                           
     Clip                        ✓                        ✓                         ✓     MinimumScalar, MaximumScalar, Identity                                                                                                                                                                                                                                                                                         
     Compress                                                            X          X                                                                      Not yet implemented.                                                                                                                                                                                                                                          
     Concat                      ✓              ✓         ✓                         X     Concatenate                                                                                                                                                                                                                                                                                                                    
     ConcatFromSequence                                                             X                                                                      Not yet implemented.                                                                                                                                                                                                                                          
     Constant                    ✓                        ✓              X          X     Identity                                                                                                                                                                                                                                                                                                                       
     ConstantOfShape                                                     ✓                Constant                                                                                                                                                                                                                                                                                                                       
     Conv                        ✓                        ✓                         X     Convolution                                                                                                                                                                                                                                                                                                                    
     ConvInteger                                                              X                                                                            Not yet implemented.                                                                                                                                                                                                                                          
     ConvTranspose               ✓                        ✓                         X     Deconvolution, Pad                                                                                                                                                                                                                                                                                                             
     Cos                                                       ✓                          Cos                                                                                                                                                                                                                                                                                                                            
     Cosh                                                                ✓                Cosh                                                                                                                                                                                                                                                                                                                           
     CumSum                                                                         X                                                                      Not yet implemented.                                                                                                                                                                                                                                          
     DepthToSpace                ✓                        ✓                         ✓     Reshape, Transpose                                                                                                                                                                                                                                                                                                             
     DequantizeLinear                                                         X                                                                            Not yet implemented.                                                                                                                                                                                                                                          
     Det                                                                            X                                                                      Not yet implemented.                                                                                                                                                                                                                                          
     Div                         ✓                        ✓    ✓                          Div2, Reshape                                                                                                                                                                                                                                                                                                                  
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
     Gather                      X                                                  X                                                                      Not yet implemented.                                                                                                                                                                                                                                          
     GatherElements                                                                 X                                                                      Not yet implemented.                                                                                                                                                                                                                                          
     GatherND                                                                       X                                                                      Not yet implemented.                                                                                                                                                                                                                                          
     Gemm                        ✓                        ✓    ✓         ✓          ✓     BatchMatmul, Add2, Reshape, MulScalar                                                                                                                                                                                                                                                                                          
     GlobalAveragePool           ✓                        ✓                               GlobalAveragePooling                                                                                                                                                                                                                                                                                                           
     GlobalLpPool                X    X                                                                                                                    Not yet implemented.                                                                                                                                                                                                                                          
     GlobalMaxPool               X                                                                                                                         Not yet implemented.                                                                                                                                                                                                                                          
     Greater                     ✓                        ✓    ✓         ✓                Greater, Reshape                                                                                                                                                                                                                                                                                                               
     HardSigmoid                 ✓                        ✓                               AddScalar, MinimumScalar, MaximumScalar, MulScalar, HardSigmoid                                                                                                                                                                                                                                                                
     Hardmax                     ✓                        ✓                         ✓     Reshape, OneHot, Max                                                                                                                                                                                                                                                                                                           
     Identity                    ✓                        ✓                               Identity                                                                                                                                                                                                                                                                                                                       
     If                          X                                                                                                                         Not yet implemented.                                                                                                                                                                                                                                          
     InstanceNormalization       ✓                        ✓                               BatchNormalization, Reshape, Concatenate, Split                                                                                                                                                                                                                                                                                
     IsInf                                                                    ✓           IsInf                                                                                                                                                                                                                                                                                                                          
     IsNaN                                                               ✓                IsNaN                                                                                                                                                                                                                                                                                                                          
     LRN                         ✓                        ✓                               AddScalar, SumPooling, Transpose, MulScalar, Div2, PowScalar                                                                                                                                                                                                                                                                   
     LSTM                        X                             X                                                                                           Not yet implemented.                                                                                                                                                                                                                                          
     LeakyRelu                   ✓                        ✓                               LeakyReLU                                                                                                                                                                                                                                                                                                                      
     Less                        ✓                        ✓    ✓         ✓                Reshape, Less                                                                                                                                                                                                                                                                                                                  
     Log                         ✓                        ✓                               Log                                                                                                                                                                                                                                                                                                                            
     LogSoftmax                  ✓                        ✓                         ✓     Sum, Sub2, Max, Reshape, Exp, Add2, Log                                                                                                                                                                                                                                                                                        
     Loop                        X                                                  X                                                                      Not yet implemented.                                                                                                                                                                                                                                          
     LpNormalization             X                                                                                                                         Not yet implemented.                                                                                                                                                                                                                                          
     LpPool                      X    X                                             X                                                                      Not yet implemented.                                                                                                                                                                                                                                          
     MatMul                      ✓                        ✓              ✓                BatchMatmul                                                                                                                                                                                                                                                                                                                    
     MatMulInteger                                                            X                                                                            Not yet implemented.                                                                                                                                                                                                                                          
     Max                         ✓                        ✓         ✓    ✓                Maximum2                                                                                                                                                                                                                                                                                                                       
     MaxPool                     ✓                        ✓         X         X     X     MaxPooling, Pad                                                  Not all features are verified, since some features are not supported by caffe2. Those features can be verified by ONNXRuntime. if opset >= 10, the ceil_mode is not supported, dilations is not equal to 1 is not supported.                                  
     MaxRoiPool                  X                                                                                                                         Not yet implemented.                                                                                                                                                                                                                                          
     MaxUnpool                                                           X          X                                                                      Not yet implemented.                                                                                                                                                                                                                                          
     Mean                        ✓                        ✓         ✓    ✓                Broadcast, Stack, Mean                                                                                                                                                                                                                                                                                                         
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
     Softmax                     ✓                        ✓                         ✓     Sum, Sub2, Max, Reshape, Div2, Exp                                                                                                                                                                                                                                                                                             
     Softplus                    ✓                        ✓                               SoftPlus                                                                                                                                                                                                                                                                                                                       
     Softsign                    ✓                        ✓                               SoftSign                                                                                                                                                                                                                                                                                                                       
     SpaceToDepth                ✓                        ✓                               Reshape, Transpose                                                                                                                                                                                                                                                                                                             
     Split                       ✓    ✓                   ✓                         ✓     Stack, Split                                                                                                                                                                                                                                                                                                                   
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

Total: 119/176

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
      Convolution              ✓    ✓    ✓    ✓     ✓     Conv, Reshape                                                                                                                   
      DepthwiseConvolution     ✓    ✓    ✓    ✓     ✓     Conv, Reshape                                                                                                                   
      Deconvolution            △    △    △    △     △     ConvTranspose, Reshape                    Caffe2 and onnxruntime do not support dilations != 1.                                 
      DepthwiseDeconvolution   △    △    △    △     △     ConvTranspose, Reshape                    Caffe2 and onnxruntime do not support dilations != 1.                                 
      MaxPooling               ✓    ✓    ✓    ✓     X     Reshape, Pad, MaxPool                                                                                                           
      AveragePooling           △    △    △    △     X     AveragePool, Reshape, Pad                 Currently only supports the cases where both ignore_border and including_pad are True.
      GlobalAveragePooling     ✓    ✓    ✓    ✓     ✓     GlobalAveragePool                                                                                                               
      SumPooling               X    ✓    ✓    ✓     X     Pad, AveragePool, Constant, Reshape, Mul                                                                                        
      Unpooling                △    ✓    ✓    X     X     Reshape, Upsample                         The kernel only supports 2d on opset 6.                                               
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
      Softmax          △    ✓    ✓    ✓     ✓     Div, ReduceMax, Sub, Exp, ReduceSum       ONNX Add, Sub operator does not support multidirectional broadcasting on opset 6.
      LogSoftmax       △    ✓    ✓    ✓     ✓     ReduceMax, Sub, Exp, Log, ReduceSum                                                                                        
      ELU              ✓    ✓    ✓    ✓     ✓     Elu                                                                                                                        
      SELU             ✓    ✓    ✓    ✓     ✓     Selu                                                                                                                       
      CReLU            ✓    ✓    ✓    ✓     ✓     Neg, Concat, Relu                                                                                                          
      CELU             ✓    ✓    ✓    ✓     ✓     Elu, Neg, Concat                                                                                                           
      PReLU            ✓    ✓    ✓    ✓     ✓     Reshape, PRelu                                                                                                             
      GELU             ✓    ✓    ✓    ✓     ✓     Pow, Add, Sqrt, Constant, Mul, Div, Tanh                                                                                   
      ReLU6            ✓    ✓    ✓    ✓     ✓     Constant, Min, Relu                                                                                                        
      HardSigmoid      ✓    ✓    ✓    ✓     ✓     HardSigmoid                                                                                                                
      HardTanh         ✓    ✓    ✓    ✓     ✓     Constant, Neg, Min, Max                                                                                                    
      LogSigmoid       ✓    ✓    ✓    ✓     ✓     Log, Sigmoid                                                                                                               
      SoftPlus         ✓    ✓    ✓    ✓     ✓     Softplus                                                                                                                   
      SoftSign         ✓    ✓    ✓    ✓     ✓     Softsign                                                                                                                   
      TanhShrink       ✓    ✓    ✓    ✓     ✓     Sub, Tanh                                                                                                                  
      Sinc             X    X    ✓    ✓     ✓     Where, Sin, Constant, Equal, Div                                                                                           
    =================  ===  ===  ===  ====  ====  ========================================  =================================================================================


Normalization
^^^^^^^^^^^^^

Count 1/6
 

    ==========================  ===  ===  ===  ====  ====  ==================================================  =======================================================================================================
         NNabla Function         6    7    9    10    11                        ONNX Op                                                                      Description                                              
    ==========================  ===  ===  ===  ====  ====  ==================================================  =======================================================================================================
      FusedBatchNormalization                                                                                  Not yet implemented.                                                                                   
      BatchNormalization        △    △    △    △     △     BatchNormalization, InstanceNormalization, Reshape  In inferring stage, caffe2 mistmatch onnx 1.4.x's implementation, "in-place" feature cannot be applied.
      SyncBatchNormalization                                                                                   Not yet implemented.                                                                                   
      MeanSubtraction                                                                                          Not yet implemented.                                                                                   
      ClipGradByValue                                                                                          Not yet implemented.                                                                                   
      ClipGradByNorm                                                                                           Not yet implemented.                                                                                   
    ==========================  ===  ===  ===  ====  ====  ==================================================  =======================================================================================================


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
      MulScalar        ✓    ✓    ✓    ✓     ✓     Constant, Mul                                                                              
      PowScalar        ✓    ✓    ✓    ✓     ✓     Pow, Constant                                                                              
      RSubScalar       ✓    ✓    ✓    ✓     ✓     Constant, Sub                                                                              
      RDivScalar       ✓    ✓    ✓    ✓     ✓     Constant, Div                                                                              
      RPowScalar       ✓    ✓    ✓    ✓     ✓     Pow, Constant                                                                              
    =================  ===  ===  ===  ====  ====  =============  ============================================================================


Logical
^^^^^^^

Count 29/29
 

    =====================  ===  ===  ===  ====  ====  ======================  ============================================================================
       NNabla Function      6    7    9    10    11          ONNX Op                                          Description                                 
    =====================  ===  ===  ===  ====  ====  ======================  ============================================================================
      Sign                 X    X    ✓    ✓     ✓     Sign                                                                                                
      Minimum2             △    ✓    ✓    ✓     ✓     Constant, Add, Min      ONNX Add operator does not support multidirectional broadcasting on opset 6.
      Maximum2             △    ✓    ✓    ✓     ✓     Constant, Add, Max      ONNX Add operator does not support multidirectional broadcasting on opset 6.
      MinimumScalar        ✓    ✓    ✓    ✓     ✓     Constant, Add, Min                                                                                  
      MaximumScalar        ✓    ✓    ✓    ✓     ✓     Constant, Add, Max                                                                                  
      LogicalAnd           ✓    ✓    ✓    ✓     ✓     And                                                                                                 
      LogicalOr            ✓    ✓    ✓    ✓     ✓     Or                                                                                                  
      LogicalXor           ✓    ✓    ✓    ✓     ✓     Xor                                                                                                 
      Equal                ✓    ✓    ✓    ✓     ✓     Equal                                                                                               
      NotEqual             ✓    ✓    ✓    ✓     ✓     Equal, Not                                                                                          
      GreaterEqual         ✓    ✓    ✓    ✓     ✓     Less, Not                                                                                           
      Greater              ✓    ✓    ✓    ✓     ✓     Greater                                                                                             
      LessEqual            ✓    ✓    ✓    ✓     ✓     Greater, Not                                                                                        
      Less                 ✓    ✓    ✓    ✓     ✓     Less                                                                                                
      LogicalAndScalar     ✓    ✓    ✓    ✓     ✓     Constant, And                                                                                       
      LogicalOrScalar      ✓    ✓    ✓    ✓     ✓     Constant, Or                                                                                        
      LogicalXorScalar     ✓    ✓    ✓    ✓     ✓     Constant, Xor                                                                                       
      EqualScalar          ✓    ✓    ✓    ✓     ✓     Constant, Equal                                                                                     
      NotEqualScalar       ✓    ✓    ✓    ✓     ✓     Constant, Equal, Not                                                                                
      GreaterEqualScalar   ✓    ✓    ✓    ✓     ✓     Constant, Less, Not                                                                                 
      GreaterScalar        ✓    ✓    ✓    ✓     ✓     Constant, Greater                                                                                   
      LessEqualScalar      ✓    ✓    ✓    ✓     ✓     Constant, Greater, Not                                                                              
      LessScalar           ✓    ✓    ✓    ✓     ✓     Constant, Less                                                                                      
      LogicalNot           ✓    ✓    ✓    ✓     ✓     Not                                                                                                 
      IsNaN                X    X    ✓    ✓     ✓     IsNaN                                                                                               
      IsInf                X    X    X    ✓     ✓     IsInf                                                                                               
      ResetNaN             X    X    ✓    ✓     ✓     Constant, Where, IsNaN                                                                              
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
      Split            ✓    ✓    ✓    ✓     ✓     Squeeze, Split                                                                                                                                           
      Stack            ✓    ✓    ✓    ✓     ✓     Concat, Unsqueeze                                                                                                                                        
      Slice            △    △    △    △     △     Constant, Slice              ONNX slice cannot support step != 1 on opset < 10.                                                                          
      Pad              △    △    △    △     △     Constant, Pad                When the mode of the pad is reflect, if the size of the pad exceeds the input size, caffe2 and onnxruntime cannot handle it.
      Transpose        ✓    ✓    ✓    ✓     ✓     Transpose                                                                                                                                                
      Broadcast        X    X    ✓    ✓     ✓                                                                                                                                                              
      BroadcastTo      ✓    ✓    ✓    ✓     ✓                                                                                                                                                              
      Tile             ✓    ✓    ✓    ✓     △     Tile, Constant, Reshape                                                                                                                                  
      OneHot           ✓    ✓    ✓    ✓     ✓     Gather, Reshape, Flatten                                                                                                                                 
      Flip             ✓    ✓    ✓    ✓     ✓     Gather, Transpose, Identity                                                                                                                              
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
      Interpolate      X    X    X    X     △     Reshape, Resize                      
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
      BinarySigmoid              X    X    ✓    ✓     ✓     Constant, Greater, Where                       
      BinaryTanh                 X    X    ✓    ✓     ✓     Constant, Greater, Where                       
      BinaryConnectAffine        ✓    ✓    ✓    ✓     ✓     Reshape, Gemm                                  
      BinaryConnectConvolution   ✓    ✓    ✓    ✓     ✓     Conv, Reshape                                  
      BinaryWeightAffine         ✓    ✓    ✓    ✓     ✓     Add, MatMul, Reshape, Mul                      
      BinaryWeightConvolution    ✓    ✓    ✓    ✓     ✓     Conv, Add, Reshape, Mul                        
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

Count 0/6
 

    =====================  ===  ===  ===  ====  ====  =========  ====================
       NNabla Function      6    7    9    10    11    ONNX Op       Description     
    =====================  ===  ===  ===  ====  ====  =========  ====================
      VATNoise                                                   Not yet implemented.
      Unlink                                                     Not yet implemented.
      Sink                                                       Not yet implemented.
      NmsDetection2d                                             Not yet implemented.
      MaxPoolingBackward                                         Not yet implemented.
      WarpByFlow                                                 Not yet implemented.
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
      BatchMatMul                              ✓      BatchMatmul, Transpose                                                                                                                     
      BiasAdd                                  ✓      Add2, Reshape                                                                                                                              
      Cast                                                                                Not yet implemented.                                                                                   
      Ceil                                     ✓      Ceil                                                                                                                                       
      ConcatV2                                 ✓      Concatenate                                                                                                                                
      Const                                    ✓      Add2                                                                                                                                       
      Conv2D                                   △      Convolution, Pad, Transpose         Tensorflow require GPU to perform related test cases. This issue is recorded only for memo.            
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
      FloorDiv                                 ✓      Div2, Floor                                                                                                                                
      FloorMod                                 ✓      Div2, Mul2, Floor, Sub2                                                                                                                    
      FusedBatchNorm                           △      BatchNormalization, Transpose       It did not pass testing for training mode.                                                             
      GatherNd                                                                            Not yet implemented.                                                                                   
      GatherV2                                                                            Not yet implemented.                                                                                   
      Greater                                  ✓      Greater                                                                                                                                    
      GreaterEqual                             ✓      Less, LogicalNot                                                                                                                           
      Identity                                 ✓      Identity                                                                                                                                   
      IsInf                                                                               Not yet implemented.                                                                                   
      IsNan                                    ✓      IsNaN                                                                                                                                      
      LeakyRelu                                ✓      LeakyReLU                                                                                                                                  
      Less                                     ✓      Less                                                                                                                                       
      LessEqual                                ✓      Greater, LogicalNot                                                                                                                        
      Log                                      ✓      Log                                                                                                                                        
      LogSoftmax                                                                          Not yet implemented.                                                                                   
      LogicalAnd                               ✓      LogicalAnd                                                                                                                                 
      LogicalNot                               ✓      LogicalNot                                                                                                                                 
      LogicalOr                                ✓      LogicalOr                                                                                                                                  
      LogicalXor                               ✓      LogicalAnd, LogicalOr, LogicalNot                                                                                                          
      MatrixBandPart                                                                      Not yet implemented.                                                                                   
      Max                                      ✓      Max                                                                                                                                        
      MaxPool                                  ✓      MaxPooling, Pad, Transpose                                                                                                                 
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
      Pack                                     ✓      Reshape, Concatenate                                                                                                                       
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
      Rsqrt                                    ✓      RDivScalar, PowScalar                                                                                                                      
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
      SplitV                                   ✓      Stack, Split                                                                                                                               
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
      Unpack                                   ✓      Stack, Reshape, Concatenate, Split                                                                                                         
    ======================================  ========  ==================================  =======================================================================================================





Export
------

- ✓: Supported
- △: Partially supported
- X: Supported, but test failed.
- Empty: Not support yet.


Total: 115/176

Neural Network Layer
^^^^^^^^^^^^^^^^^^^^

Count 11/14
 

    =========================  ========  ================================================================================================================================================  ==================================================================================
         NNabla Function        Status                                                                        TF Op                                                                                                           Description                                    
    =========================  ========  ================================================================================================================================================  ==================================================================================
      Affine                   ✓         Mul, Placeholder, Add, Reshape, MatMul, Const                                                                                                                                                                                       
      RNN                                                                                                                                                                                  Not yet implemented.                                                              
      LSTM                                                                                                                                                                                 Not yet implemented.                                                              
      GRU                                                                                                                                                                                  Not yet implemented.                                                              
      Convolution              △         Pad, Identity, Transpose, Placeholder, Add, Split, Reshape, Conv2D, Const, SpaceToBatchND, BatchToSpaceND, ConcatV2                               The cases `dilations` and `strides` larger than 1 are not supported by tensorflow.
      DepthwiseConvolution     △         Pad, Transpose, Placeholder, Add, Split, Reshape, Conv2D, Const, SpaceToBatchND, BatchToSpaceND, ConcatV2                                         The cases `dilations` and `strides` larger than 1 are not supported by tensorflow.
      Deconvolution            △         Identity, Slice, Transpose, Placeholder, Add, Split, Reshape, Const, Conv2DBackpropInput, ConcatV2                                                The cases `dilations` larger than 1 are not supported by tensorflow.              
      DepthwiseDeconvolution   △         Slice, Transpose, Placeholder, Add, Split, Reshape, Const, Conv2DBackpropInput, ConcatV2                                                          The cases `dilations` larger than 1 are not supported by tensorflow.              
      MaxPooling               ✓         Transpose, PadV2, Placeholder, Reshape, MaxPool, Const, MaxPool3D                                                                                                                                                                   
      AveragePooling           △         Pad, AvgPool3D, Transpose, Placeholder, Reshape, Const, AvgPool                                                                                   Currently only supports the cases both ignore_border and including_pad are True.  
      GlobalAveragePooling     ✓         SplitV, Sub, Mean, Const, Range, Pack                                                                                                                                                                                               
      SumPooling               ✓         Mul, Pad, AvgPool3D, Transpose, Placeholder, Reshape, Const, AvgPool                                                                                                                                                                
      Unpooling                △         Cast, Mul, Identity, NoOp, Transpose, StridedSlice, Placeholder, LogicalAnd, ResizeNearestNeighbor, Reshape, Equal, Const, Merge, Assert, Switch  The kernel only supports 2d.                                                      
      Embed                    ✓         Placeholder, Const, GatherV2                                                                                                                                                                                                        
    =========================  ========  ================================================================================================================================================  ==================================================================================


Neural Network Activation Functions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Count 21/21
 

    =================  ========  ====================================================================================  =============
     NNabla Function    Status                                          TF Op                                           Description 
    =================  ========  ====================================================================================  =============
      Sigmoid          ✓         Placeholder, Sigmoid                                                                               
      Swish            ✓         Placeholder, Mul, Sigmoid                                                                          
      Tanh             ✓         Placeholder, Tanh                                                                                  
      ReLU             ✓         Placeholder, Relu                                                                                  
      LeakyReLU        ✓         LeakyRelu, Placeholder                                                                             
      Softmax          ✓         Sum, Max, Placeholder, Sub, RealDiv, Const, Exp                                                    
      LogSoftmax       ✓         Sum, Max, Placeholder, Sub, Const, Exp, Log                                                        
      ELU              ✓         Cast, Mul, Less, Placeholder, Elu, Add, Sub, Const, Exp, GreaterEqual                              
      SELU             ✓         Mul, Placeholder, Add, Minimum, Sub, Const, Exp, Maximum                                           
      CReLU            ✓         Placeholder, Neg, Relu, Const, ConcatV2                                                            
      CELU             ✓         Cast, Mul, Less, Placeholder, Elu, Add, Neg, Sub, Const, Exp, GreaterEqual, ConcatV2               
      PReLU            ✓         Mul, Placeholder, Add, Abs, Reshape, Sub, Relu, Const                                              
      GELU             ✓         Mul, Pow, Placeholder, Add, Sqrt, RealDiv, Const, Tanh                                             
      ReLU6            ✓         Placeholder, Min, Relu, Const, Pack                                                                
      HardSigmoid      ✓         Mul, Placeholder, Add, Minimum, Const, Maximum                                                     
      HardTanh         ✓         Max, Placeholder, Neg, Min, Const, Pack                                                            
      LogSigmoid       ✓         Placeholder, Log, Sigmoid                                                                          
      SoftPlus         ✓         Placeholder, Softplus                                                                              
      SoftSign         ✓         Placeholder, Softsign                                                                              
      TanhShrink       ✓         Placeholder, Sub, Tanh                                                                             
      Sinc             ✓         Placeholder, Sin, Equal, RealDiv, Const, Select                                                    
    =================  ========  ====================================================================================  =============


Normalization
^^^^^^^^^^^^^

Count 1/6
 

    ==========================  ========  ========================================================================================  =======================================================================================================
         NNabla Function         Status                                            TF Op                                                                                          Description                                              
    ==========================  ========  ========================================================================================  =======================================================================================================
      FusedBatchNormalization                                                                                                       Not yet implemented.                                                                                   
      BatchNormalization        △         Mul, StopGradient, Placeholder, Add, Rsqrt, SquaredDifference, Reshape, Sub, Mean, Const  In inferring stage, caffe2 mistmatch onnx 1.4.x's implementation, "in-place" feature cannot be applied.
      SyncBatchNormalization                                                                                                        Not yet implemented.                                                                                   
      MeanSubtraction                                                                                                               Not yet implemented.                                                                                   
      ClipGradByValue                                                                                                               Not yet implemented.                                                                                   
      ClipGradByNorm                                                                                                                Not yet implemented.                                                                                   
    ==========================  ========  ========================================================================================  =======================================================================================================


Reduction
^^^^^^^^^

Count 5/7
 

    =================  ========  ========================  ====================
     NNabla Function    Status            TF Op                Description     
    =================  ========  ========================  ====================
      Sum              ✓         Placeholder, Sum, Const                       
      Mean             ✓         Placeholder, Mean, Const                      
      Max              ✓         Placeholder, Max, Const                       
      Min              ✓         Placeholder, Min, Const                       
      Prod             ✓         Prod, Placeholder, Const                      
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
      Div2             ✓         Placeholder, RealDiv                             
      Pow2             ✓         Pow, Placeholder                                 
      AddScalar        ✓         Placeholder, Add, Const                          
      MulScalar        ✓         Placeholder, Const, Mul                          
      PowScalar        ✓         Pow, Placeholder, Const                          
      RSubScalar       ✓         Placeholder, Sub, Const                          
      RDivScalar       ✓         Placeholder, RealDiv, Const                      
      RPowScalar       ✓         Pow, Placeholder, Const                          
    =================  ========  ===========================  ====================


Logical
^^^^^^^

Count 27/29
 

    =====================  ========  =====================================================  ====================
       NNabla Function      Status                           TF Op                              Description     
    =====================  ========  =====================================================  ====================
      Sign                 ✓         Sign, Placeholder                                                          
      Minimum2             ✓         Placeholder, Add, Min, Const, Pack                                         
      Maximum2             ✓         Max, Placeholder, Add, Const, Pack                                         
      MinimumScalar        ✓         Placeholder, Add, Min, Const, Pack                                         
      MaximumScalar        ✓         Max, Placeholder, Add, Const, Pack                                         
      LogicalAnd           ✓         LogicalAnd, Placeholder                                                    
      LogicalOr            ✓         Placeholder, LogicalOr                                                     
      LogicalXor           ✓         LogicalAnd, Placeholder, LogicalOr, LogicalNot                             
      Equal                ✓         Placeholder, Equal                                                         
      NotEqual             ✓         Placeholder, Equal, LogicalNot                                             
      GreaterEqual         ✓         Placeholder, Less, LogicalNot                                              
      Greater              ✓         Placeholder, Greater                                                       
      LessEqual            ✓         Placeholder, Greater, LogicalNot                                           
      Less                 ✓         Placeholder, Less                                                          
      LogicalAndScalar     ✓         LogicalAnd, Placeholder, Const                                             
      LogicalOrScalar      ✓         Placeholder, LogicalOr, Const                                              
      LogicalXorScalar     ✓         LogicalOr, LogicalAnd, Placeholder, LogicalNot, Const                      
      EqualScalar          ✓         Placeholder, Equal, Const                                                  
      NotEqualScalar       ✓         Placeholder, Const, Equal, LogicalNot                                      
      GreaterEqualScalar   ✓         Placeholder, Const, Less, LogicalNot                                       
      GreaterScalar        ✓         Placeholder, Greater, Const                                                
      LessEqualScalar      ✓         Placeholder, Greater, Const, LogicalNot                                    
      LessScalar           ✓         Placeholder, Less, Const                                                   
      LogicalNot           ✓         Placeholder, LogicalNot                                                    
      IsNaN                ✓         Placeholder, IsNan                                                         
      IsInf                X                                                                Not yet implemented.
      ResetNaN             ✓         Placeholder, Select, IsNan, Const                                          
      ResetInf             X                                                                Not yet implemented.
      Where                ✓         Placeholder, Select                                                        
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
      Log              ✓         Placeholder, Log                                                           
      Identity         ✓         Placeholder, Identity                                                      
      BatchMatmul      ✓         Transpose, Placeholder, BatchMatMulV2, Reshape, Const                      
      Round            X                                                                Not yet implemented.
      Ceil             ✓         Placeholder, Ceil                                                          
      Floor            ✓         Placeholder, Floor                                                         
      Sin              ✓         Placeholder, Sin                                                           
      Cos              ✓         Placeholder, Cos                                                           
      Tan              ✓         Placeholder, Tan                                                           
      Sinh             ✓         Placeholder, Sinh                                                          
      Cosh             ✓         Placeholder, Cosh                                                          
      ASin             ✓         Asin, Placeholder                                                          
      ACos             ✓         Acos, Placeholder                                                          
      ATan             ✓         Placeholder, Atan                                                          
      ATan2            ✓         Placeholder, Atan, RealDiv                                                 
      ASinh            ✓         Placeholder, Asinh                                                         
      ACosh            ✓         Placeholder, Acosh                                                         
      ATanh            ✓         Placeholder, Atanh                                                         
    =================  ========  =====================================================  ====================


Array Manipulation
^^^^^^^^^^^^^^^^^^

Count 12/21
 

    =================  ========  =================================================  ================================================================================================================
     NNabla Function    Status                         TF Op                                                                          Description                                                   
    =================  ========  =================================================  ================================================================================================================
      Concatenate      ✓         Placeholder, Const, ConcatV2                                                                                                                                       
      Split            ✓         Squeeze, Placeholder, Const, SplitV                                                                                                                                
      Stack            ✓         Placeholder, ExpandDims, Const, ConcatV2                                                                                                                           
      Slice            △         Placeholder, Slice, Const                          step != 1" exceed the scope of onnx opset 9,  not supported.                                                    
      Pad              △         MirrorPad, Placeholder, Const, PadV2               When the mode of the pad is reflect, if the size of the pad exceeds the input size, tensorflow cannot handle it.
      Transpose        ✓         Placeholder, Const, Transpose                                                                                                                                      
      Broadcast        ✓                                                                                                                                                                            
      BroadcastTo      ✓                                                                                                                                                                            
      Tile             ✓         Tile, Placeholder, Reshape, Const                                                                                                                                  
      OneHot           ✓         Const, Placeholder, Reshape, GatherV2                                                                                                                              
      Flip             ✓         Identity, Transpose, Placeholder, GatherV2, Const                                                                                                                  
      Shift                                                                         Not yet implemented.                                                                                            
      Sort                                                                          Not yet implemented.                                                                                            
      Reshape          ✓         Placeholder, Reshape, Const                                                                                                                                        
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
      BinarySigmoid              ✓         Select, Placeholder, Greater, Const                                                                                                                                         
      BinaryTanh                 ✓         Select, Placeholder, Greater, Const                                                                                                                                         
      BinaryConnectAffine        ✓         Mul, Placeholder, Add, Reshape, MatMul, Const                                                                                                                               
      BinaryConnectConvolution   △         Identity, Pad, Transpose, Placeholder, Add, Split, Reshape, Conv2D, Const, ConcatV2       The cases `dilations` and `strides` larger than 1 are not supported by tensorflow.
      BinaryWeightAffine         ✓         Placeholder, Const, Add, Reshape, MatMul, Mul                                                                                                                               
      BinaryWeightConvolution    △         Mul, Identity, Pad, Transpose, Placeholder, Add, Split, Reshape, Conv2D, Const, ConcatV2  The cases `dilations` and `strides` larger than 1 are not supported by tensorflow.
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

Count 0/6
 

    =====================  ========  =======  ====================
       NNabla Function      Status    TF Op       Description     
    =====================  ========  =======  ====================
      VATNoise                                Not yet implemented.
      Unlink                                  Not yet implemented.
      Sink                                    Not yet implemented.
      NmsDetection2d                          Not yet implemented.
      MaxPoolingBackward                      Not yet implemented.
      WarpByFlow                              Not yet implemented.
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

Total: 56/176

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
      AveragePooling           △                      
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

Count 0/6
 

    =====================  ========  =============
       NNabla Function      Status    Description 
    =====================  ========  =============
      VATNoise                                    
      Unlink                                      
      Sink                                        
      NmsDetection2d                              
      MaxPoolingBackward                          
      WarpByFlow                                  
    =====================  ========  =============



