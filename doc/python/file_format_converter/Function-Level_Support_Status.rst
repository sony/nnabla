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


Total: 93/155

.. table:: 

    ===========================  ===  ===  ===  ===  ===  ===  ===  ===  ===  ====  ====  ===============================================================  =============================================================================================================================================================================================================
           ONNX Operator          1    2    3    4    5    6    7    8    9    10    11                             NNabla Func                                                                                                                             Description                                                                                                 
    ===========================  ===  ===  ===  ===  ===  ===  ===  ===  ===  ====  ====  ===============================================================  =============================================================================================================================================================================================================
     Abs                         ✓                        ✓    ✓                          Abs                                                                                                                                                                                                                                                                           
     Acos                                                      ✓                          ACos                                                                                                                                                                                                                                                                          
     Acosh                                                               ✓                ACosh                                                                                                                                                                                                                                                                         
     Add                         ✓                        ✓    ✓                          Reshape, Add2                                                                                                                                                                                                                                                                 
     And                         ✓                             ✓                          Reshape, LogicalAnd                                                                                                                                                                                                                                                           
     ArgMax                      ✓                             ✓              X     ✓     Max                                                                                                                                                                                                                                                                           
     ArgMin                      ✓                             ✓              X     ✓     Min                                                                                                                                                                                                                                                                           
     Asin                                                      ✓                          ASin                                                                                                                                                                                                                                                                          
     Asinh                                                               ✓                ASinh                                                                                                                                                                                                                                                                         
     Atan                                                      ✓                          ATan                                                                                                                                                                                                                                                                          
     Atanh                                                               ✓                ATanh                                                                                                                                                                                                                                                                         
     AveragePool                 ✓                             ✓              X     X     Pad, AveragePooling                                              Not all features are verified. Those features can be verified by ONNXRuntime when opset > 6. Some feature is not supported by Nnabla such as Pad's edge mode. if opset >= 10, the ceil_mode is not supported.
     BatchNormalization          X                        X    X         ✓                BatchNormalization                                                                                                                                                                                                                                                            
     BitShift                                                                       X                                                                      Not yet implemented.                                                                                                                                                                                         
     Cast                        ✓                        ✓    ✓         X                Abs, Log                                                                                                                                                                                                                                                                      
     Ceil                        ✓                        ✓    ✓                          Ceil                                                                                                                                                                                                                                                                          
     Clip                        ✓                        ✓    ✓                    ✓     MinimumScalar, Identity, MaximumScalar                                                                                                                                                                                                                                        
     Compress                                                            X          X                                                                      Not yet implemented.                                                                                                                                                                                         
     Concat                      ✓              ✓              ✓                    X     Concatenate                                                                                                                                                                                                                                                                   
     ConcatFromSequence                                                             X                                                                      Not yet implemented.                                                                                                                                                                                         
     Constant                    ✓                             ✓         X          X     Identity                                                                                                                                                                                                                                                                      
     ConstantOfShape                                                     ✓                Constant                                                                                                                                                                                                                                                                      
     Conv                        ✓                             ✓                    X     Convolution                                                                                                                                                                                                                                                                   
     ConvInteger                                                              X                                                                            Not yet implemented.                                                                                                                                                                                         
     ConvTranspose               ✓                             ✓                    X     Deconvolution, Pad                                                                                                                                                                                                                                                            
     Cos                                                       ✓                          Cos                                                                                                                                                                                                                                                                           
     Cosh                                                                ✓                Cosh                                                                                                                                                                                                                                                                          
     CumSum                                                                         X                                                                      Not yet implemented.                                                                                                                                                                                         
     DepthToSpace                ✓                             ✓                    ✓     Reshape, Transpose                                                                                                                                                                                                                                                            
     DequantizeLinear                                                         X                                                                            Not yet implemented.                                                                                                                                                                                         
     Det                                                                            X                                                                      Not yet implemented.                                                                                                                                                                                         
     Div                         ✓                        ✓    ✓                          Reshape, Div2                                                                                                                                                                                                                                                                 
     Dropout                     ✓                        ✓    ✓              X           Identity                                                                                                                                                                                                                                                                      
     DynamicQuantizeLinear                                                          X                                                                      Not yet implemented.                                                                                                                                                                                         
     Elu                         ✓                        ✓    ✓                          ELU                                                                                                                                                                                                                                                                           
     Equal                       ✓                             ✓                    X     Equal, Reshape                                                                                                                                                                                                                                                                
     Erf                                                                 X                                                                                 Not yet implemented.                                                                                                                                                                                         
     Exp                         ✓                        ✓    ✓                          Exp                                                                                                                                                                                                                                                                           
     Expand                                                         ✓    ✓                Reshape, Broadcast                                                                                                                                                                                                                                                            
     EyeLike                                                             X                                                                                 Not yet implemented.                                                                                                                                                                                         
     Flatten                     ✓                             ✓         ✓          ✓     Reshape                                                                                                                                                                                                                                                                       
     Floor                       ✓                        ✓    ✓                          Floor                                                                                                                                                                                                                                                                         
     GRU                         X         X                   X                                                                                           Not yet implemented.                                                                                                                                                                                         
     Gather                      ✓                             ✓                    ✓     Slice, Concatenate                                                                                                                                                                                                                                                            
     GatherElements                                                                 X                                                                      Not yet implemented.                                                                                                                                                                                         
     GatherND                                                                       X                                                                      Not yet implemented.                                                                                                                                                                                         
     Gemm                        ✓                        ✓    ✓         ✓          ✓     MulScalar, Reshape, Add2, BatchMatmul                                                                                                                                                                                                                                         
     GlobalAveragePool           ✓                             ✓                          GlobalAveragePooling                                                                                                                                                                                                                                                          
     GlobalLpPool                X    X                                                                                                                    Not yet implemented.                                                                                                                                                                                         
     GlobalMaxPool               X                                                                                                                         Not yet implemented.                                                                                                                                                                                         
     Greater                     ✓                             ✓         ✓                Reshape, Greater                                                                                                                                                                                                                                                              
     HardSigmoid                 ✓                        ✓    ✓                          MulScalar, AddScalar, HardSigmoid, MinimumScalar, MaximumScalar                                                                                                                                                                                                               
     Hardmax                     ✓                             ✓                    ✓     Max, Reshape, OneHot                                                                                                                                                                                                                                                          
     Identity                    ✓                             ✓                          Identity                                                                                                                                                                                                                                                                      
     If                          X                                                                                                                         Not yet implemented.                                                                                                                                                                                         
     InstanceNormalization       ✓                        ✓    ✓                          BatchNormalization, Reshape, Concatenate, Split                                                                                                                                                                                                                               
     IsInf                                                                    ✓           IsInf                                                                                                                                                                                                                                                                         
     IsNaN                                                               ✓                IsNaN                                                                                                                                                                                                                                                                         
     LRN                         ✓                             ✓                          SumPooling, Transpose, MulScalar, AddScalar, PowScalar, Div2                                                                                                                                                                                                                  
     LSTM                        X                             X                                                                                           Not yet implemented.                                                                                                                                                                                         
     LeakyRelu                   ✓                        ✓    ✓                          LeakyReLU                                                                                                                                                                                                                                                                     
     Less                        ✓                             ✓         ✓                Reshape, Less                                                                                                                                                                                                                                                                 
     Log                         ✓                        ✓    ✓                          Log                                                                                                                                                                                                                                                                           
     LogSoftmax                  ✓                             ✓                    ✓     Add2, Sum, Exp, Sub2, Reshape, Max, Log                                                                                                                                                                                                                                       
     Loop                        X                                                  X                                                                      Not yet implemented.                                                                                                                                                                                         
     LpNormalization             X                                                                                                                         Not yet implemented.                                                                                                                                                                                         
     LpPool                      X    X                                             X                                                                      Not yet implemented.                                                                                                                                                                                         
     MatMul                      ✓                             ✓         ✓                Reshape, BatchMatmul                                                                                                                                                                                                                                                          
     MatMulInteger                                                            X                                                                            Not yet implemented.                                                                                                                                                                                         
     Max                         ✓                        ✓    ✓    ✓    ✓                Maximum2                                                                                                                                                                                                                                                                      
     MaxPool                     ✓                             ✓    X         X     X     MaxPooling, Pad                                                  Not all features are verified. Those features can be verified by ONNXRuntime. if opset >= 10, the ceil_mode is not supported, dilations is not equal to 1 is not supported.                                  
     MaxRoiPool                  X                                                                                                                         Not yet implemented.                                                                                                                                                                                         
     MaxUnpool                                                           X          X                                                                      Not yet implemented.                                                                                                                                                                                         
     Mean                        ✓                        ✓    ✓    ✓    ✓                Stack, Broadcast, Mean                                                                                                                                                                                                                                                        
     MeanVarianceNormalization                                           X                                                                                 Not yet implemented.                                                                                                                                                                                         
     Min                         ✓                        ✓    ✓    ✓    ✓                Minimum2                                                                                                                                                                                                                                                                      
     Mod                                                                      X                                                                            Not yet implemented.                                                                                                                                                                                         
     Mul                         ✓                        ✓    ✓                          Reshape, Mul2                                                                                                                                                                                                                                                                 
     Multinomial                                               X                                                                                           Not yet implemented.                                                                                                                                                                                         
     Neg                         ✓                        ✓    ✓                          MulScalar                                                                                                                                                                                                                                                                     
     NonMaxSuppression                                                        X     X                                                                      Not yet implemented.                                                                                                                                                                                         
     NonZero                                                             X                                                                                 Not yet implemented.                                                                                                                                                                                         
     Not                         ✓                             ✓                          LogicalNot                                                                                                                                                                                                                                                                    
     OneHot                                                              X          X                                                                      Not yet implemented.                                                                                                                                                                                         
     Or                          ✓                             ✓                          Reshape, LogicalOr                                                                                                                                                                                                                                                            
     PRelu                       ✓                        ✓    ✓         X                PReLU                                                                                                                                                                                                                                                                         
     Pad                         ✓    ✓                        ✓                    ✓     Pad                                                              Onnx required to support "edge" mode, while nnabla does not support it.                                                                                                                                      
     Pow                         ✓                             ✓                          Reshape, Pow2                                                                                                                                                                                                                                                                 
     QLinearConv                                                              X                                                                            Not yet implemented.                                                                                                                                                                                         
     QLinearMatMul                                                            X                                                                            Not yet implemented.                                                                                                                                                                                         
     QuantizeLinear                                                           X                                                                            Not yet implemented.                                                                                                                                                                                         
     RNN                         X                             X                                                                                           Not yet implemented.                                                                                                                                                                                         
     RandomNormal                X                                                                                                                         Not yet implemented.                                                                                                                                                                                         
     RandomNormalLike            X                                                                                                                         Not yet implemented.                                                                                                                                                                                         
     RandomUniform               X                                                                                                                         Not yet implemented.                                                                                                                                                                                         
     RandomUniformLike           X                                                                                                                         Not yet implemented.                                                                                                                                                                                         
     Range                                                                          X                                                                      Not yet implemented.                                                                                                                                                                                         
     Reciprocal                  ✓                        ✓    ✓                          RDivScalar                                                                                                                                                                                                                                                                    
     ReduceL1                    X                                                  X                                                                      Not yet implemented.                                                                                                                                                                                         
     ReduceL2                    X                                                  X                                                                      Not yet implemented.                                                                                                                                                                                         
     ReduceLogSum                X                                                  X                                                                      Not yet implemented.                                                                                                                                                                                         
     ReduceLogSumExp             X                                                  X                                                                      Not yet implemented.                                                                                                                                                                                         
     ReduceMax                   ✓                             ✓                    ✓     Max                                                                                                                                                                                                                                                                           
     ReduceMean                  ✓                             ✓                    ✓     Mean                                                                                                                                                                                                                                                                          
     ReduceMin                   ✓                             ✓                    ✓     Min                                                                                                                                                                                                                                                                           
     ReduceProd                  ✓                             ✓                    ✓     Prod                                                                                                                                                                                                                                                                          
     ReduceSum                   ✓                             ✓                    ✓     Sum                                                                                                                                                                                                                                                                           
     ReduceSumSquare             ✓                             ✓                    ✓     PowScalar, Sum                                                                                                                                                                                                                                                                
     Relu                        ✓                        ✓    ✓                          ReLU                                                                                                                                                                                                                                                                          
     Reshape                     ✓                   ✓         ✓                          Reshape                                                                                                                                                                                                                                                                       
     Resize                                                                   X     X                                                                      Not yet implemented.                                                                                                                                                                                         
     ReverseSequence                                                          X                                                                            Not yet implemented.                                                                                                                                                                                         
     RoiAlign                                                                 X                                                                            Not yet implemented.                                                                                                                                                                                         
     Round                                                                          ✓     Round                                                                                                                                                                                                                                                                         
     Scan                                                           X    X          X                                                                      Not yet implemented.                                                                                                                                                                                         
     Scatter                                                             X          X                                                                      Not yet implemented.                                                                                                                                                                                         
     ScatterElements                                                                X                                                                      Not yet implemented.                                                                                                                                                                                         
     ScatterND                                                                      X                                                                      Not yet implemented.                                                                                                                                                                                         
     Selu                        ✓                        ✓    ✓                          SELU                                                                                                                                                                                                                                                                          
     SequenceAt                                                                     X                                                                      Not yet implemented.                                                                                                                                                                                         
     SequenceConstruct                                                              X                                                                      Not yet implemented.                                                                                                                                                                                         
     SequenceErase                                                                  X                                                                      Not yet implemented.                                                                                                                                                                                         
     SequenceInsert                                                                 X                                                                      Not yet implemented.                                                                                                                                                                                         
     SequenceLength                                                                 X                                                                      Not yet implemented.                                                                                                                                                                                         
     Shape                       X                                                                                                                         Not yet implemented.                                                                                                                                                                                         
     Shrink                                                              X                                                                                 Not yet implemented.                                                                                                                                                                                         
     Sigmoid                     ✓                        ✓    ✓                          Sigmoid                                                                                                                                                                                                                                                                       
     Sign                                                                ✓                Sign                                                                                                                                                                                                                                                                          
     Sin                                                       ✓                          Sin                                                                                                                                                                                                                                                                           
     Sinh                                                                ✓                Sinh                                                                                                                                                                                                                                                                          
     Size                        X                                                                                                                         Not yet implemented.                                                                                                                                                                                         
     Slice                       ✓                             ✓              ✓     X     Slice                                                                                                                                                                                                                                                                         
     Softmax                     ✓                             ✓                    ✓     Sum, Exp, Sub2, Reshape, Max, Div2                                                                                                                                                                                                                                            
     Softplus                    ✓                             ✓                          SoftPlus                                                                                                                                                                                                                                                                      
     Softsign                    ✓                             ✓                          SoftSign                                                                                                                                                                                                                                                                      
     SpaceToDepth                ✓                             ✓                          Reshape, Transpose                                                                                                                                                                                                                                                            
     Split                       ✓    ✓                        ✓                    ✓     Stack, Split                                                                                                                                                                                                                                                                  
     SplitToSequence                                                                X                                                                      Not yet implemented.                                                                                                                                                                                         
     Sqrt                        ✓                        ✓    ✓                          PowScalar                                                                                                                                                                                                                                                                     
     Squeeze                     ✓                             ✓                    ✓     Reshape                                                                                                                                                                                                                                                                       
     StringNormalizer                                                         X                                                                            Not yet implemented.                                                                                                                                                                                         
     Sub                         ✓                        ✓    ✓                          Sub2, Reshape                                                                                                                                                                                                                                                                 
     Sum                         ✓                        ✓    ✓    X    X                AddN                                                                                                                                                                                                                                                                          
     Tan                                                       ✓                          Tan                                                                                                                                                                                                                                                                           
     Tanh                        ✓                        ✓    ✓                          Tanh                                                                                                                                                                                                                                                                          
     TfIdfVectorizer                                                     X                                                                                 Not yet implemented.                                                                                                                                                                                         
     ThresholdedRelu                                                          ✓           GreaterScalar, Constant, Where                                                                                                                                                                                                                                                
     Tile                        ✓                        ✓    ✓                          Tile                                                                                                                                                                                                                                                                          
     TopK                        X                                            X     X                                                                      Not yet implemented.                                                                                                                                                                                         
     Transpose                   ✓                             ✓                          Transpose                                                                                                                                                                                                                                                                     
     Unique                                                                         X                                                                      Not yet implemented.                                                                                                                                                                                         
     Unsqueeze                   ✓                             ✓                    ✓     Reshape                                                                                                                                                                                                                                                                       
     Upsample                    X                             X         ✓    X           Unpooling                                                                                                                                                                                                                                                                     
     Where                                                               ✓                Where                                                                                                                                                                                                                                                                         
     Xor                         ✓                             ✓                          Reshape, LogicalXor                                                                                                                                                                                                                                                           
    ===========================  ===  ===  ===  ===  ===  ===  ===  ===  ===  ====  ====  ===============================================================  =============================================================================================================================================================================================================



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
 

    =========================  ===  ===  ====  ====  ========================================  ======================================================================================
         NNabla Function        7    9    10    11                   ONNX Op                                                        Description                                      
    =========================  ===  ===  ====  ====  ========================================  ======================================================================================
      Affine                   ✓    ✓    ✓     ✓     Reshape, Gemm                                                                                                                   
      RNN                                                                                      Not yet implemented.                                                                  
      LSTM                                                                                     Not yet implemented.                                                                  
      GRU                                                                                      Not yet implemented.                                                                  
      Convolution              ✓    ✓    ✓     ✓     Reshape, Conv                                                                                                                   
      DepthwiseConvolution     ✓    ✓    ✓     ✓     Reshape, Conv                                                                                                                   
      Deconvolution            ✓    ✓    ✓     ✓     ConvTranspose, Reshape                                                                                                          
      DepthwiseDeconvolution   ✓    ✓    ✓     ✓     ConvTranspose, Reshape                                                                                                          
      MaxPooling               ✓    ✓    ✓     ✓     Constant, Reshape, Pad, MaxPool                                                                                                 
      AveragePooling           △    △    △     △     AveragePool, Constant, Reshape, Pad       Currently only supports the cases where both ignore_border and including_pad are True.
      GlobalAveragePooling     ✓    ✓    ✓     ✓     GlobalAveragePool                                                                                                               
      SumPooling               ✓    ✓    ✓     ✓     Constant, Mul, Pad, AveragePool, Reshape                                                                                        
      Unpooling                ✓    ✓    ✓     ✓     Resize                                                                                                                          
      Embed                    ✓    ✓    ✓     ✓     Gather                                                                                                                          
    =========================  ===  ===  ====  ====  ========================================  ======================================================================================


Neural Network Activation Functions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Count 21/21
 

    =================  ===  ===  ====  ====  ========================================  =============
     NNabla Function    7    9    10    11                   ONNX Op                    Description 
    =================  ===  ===  ====  ====  ========================================  =============
      Sigmoid          ✓    ✓    ✓     ✓     Sigmoid                                                
      Swish            ✓    ✓    ✓     ✓     Mul, Sigmoid                                           
      Tanh             ✓    ✓    ✓     ✓     Tanh                                                   
      ReLU             ✓    ✓    ✓     ✓     Relu                                                   
      LeakyReLU        ✓    ✓    ✓     ✓     LeakyRelu                                              
      Softmax          ✓    ✓    ✓     ✓     Sub, Div, Exp, ReduceSum, ReduceMax                    
      LogSoftmax       ✓    ✓    ✓     ✓     Sub, Exp, ReduceSum, ReduceMax, Log                    
      ELU              ✓    ✓    ✓     ✓     Elu                                                    
      SELU             ✓    ✓    ✓     ✓     Selu                                                   
      CReLU            ✓    ✓    ✓     ✓     Concat, Neg, Relu                                      
      CELU             ✓    ✓    ✓     ✓     Concat, Neg, Elu                                       
      PReLU            ✓    ✓    ✓     ✓     Reshape, PRelu                                         
      GELU             ✓    ✓    ✓     ✓     Constant, Mul, Add, Tanh, Div, Pow, Sqrt               
      ReLU6            ✓    ✓    ✓     ✓     Min, Constant, Relu                                    
      HardSigmoid      ✓    ✓    ✓     ✓     HardSigmoid                                            
      HardTanh         ✓    ✓    ✓     ✓     Max, Neg, Constant, Min                                
      LogSigmoid       ✓    ✓    ✓     ✓     Sigmoid, Log                                           
      SoftPlus         ✓    ✓    ✓     ✓     Softplus                                               
      SoftSign         ✓    ✓    ✓     ✓     Softsign                                               
      TanhShrink       ✓    ✓    ✓     ✓     Sub, Tanh                                              
      Sinc             X    X    X     ✓     Constant, Where, Sin, Div, Equal                       
    =================  ===  ===  ====  ====  ========================================  =============


Normalization
^^^^^^^^^^^^^

Count 2/6
 

    ==========================  ===  ===  ====  ====  ===============================================================================================  ====================
         NNabla Function         7    9    10    11                                               ONNX Op                                                  Description     
    ==========================  ===  ===  ====  ====  ===============================================================================================  ====================
      FusedBatchNormalization   ✓    ✓    ✓     ✓     Squeeze, Sub, Constant, Mul, Add, Relu, Div, BatchNormalization, Reshape, ReduceSum, ReduceMean                      
      BatchNormalization        ✓    ✓    ✓     ✓     Squeeze, Sub, Constant, Mul, Div, BatchNormalization, Reshape, ReduceSum, ReduceMean                                 
      SyncBatchNormalization                                                                                                                           Not yet implemented.
      MeanSubtraction                                                                                                                                  Not yet implemented.
      ClipGradByValue                                                                                                                                  Not yet implemented.
      ClipGradByNorm                                                                                                                                   Not yet implemented.
    ==========================  ===  ===  ====  ====  ===============================================================================================  ====================


Reduction
^^^^^^^^^

Count 5/7
 

    =================  ===  ===  ====  ====  ==========  ====================
     NNabla Function    7    9    10    11    ONNX Op        Description     
    =================  ===  ===  ====  ====  ==========  ====================
      Sum              ✓    ✓    ✓     ✓     ReduceSum                       
      Mean             ✓    ✓    ✓     ✓     ReduceMean                      
      Max              ✓    ✓    ✓     ✓     ReduceMax                       
      Min              ✓    ✓    ✓     ✓     ReduceMin                       
      Prod             ✓    ✓    ✓     ✓     ReduceProd                      
      ReduceSum                                          Not yet implemented.
      ReduceMean                                         Not yet implemented.
    =================  ===  ===  ====  ====  ==========  ====================


Arithmetic
^^^^^^^^^^

Count 11/12
 

    =================  ===  ===  ====  ====  =============  ====================
     NNabla Function    7    9    10    11      ONNX Op         Description     
    =================  ===  ===  ====  ====  =============  ====================
      Add2             ✓    ✓    ✓     ✓     Add                                
      BcAdd2                                                Not yet implemented.
      Sub2             ✓    ✓    ✓     ✓     Sub                                
      Mul2             ✓    ✓    ✓     ✓     Mul                                
      Div2             ✓    ✓    ✓     ✓     Div                                
      Pow2             ✓    ✓    ✓     ✓     Pow                                
      AddScalar        ✓    ✓    ✓     ✓     Add, Constant                      
      MulScalar        ✓    ✓    ✓     ✓     Constant, Mul                      
      PowScalar        ✓    ✓    ✓     ✓     Pow, Constant                      
      RSubScalar       ✓    ✓    ✓     ✓     Sub, Constant                      
      RDivScalar       ✓    ✓    ✓     ✓     Constant, Div                      
      RPowScalar       ✓    ✓    ✓     ✓     Pow, Constant                      
    =================  ===  ===  ====  ====  =============  ====================


Logical
^^^^^^^

Count 29/29
 

    =====================  ===  ===  ====  ====  ======================  =============
       NNabla Function      7    9    10    11          ONNX Op           Description 
    =====================  ===  ===  ====  ====  ======================  =============
      Sign                 X    ✓    ✓     ✓     Sign                                 
      Minimum2             ✓    ✓    ✓     ✓     Add, Constant, Min                   
      Maximum2             ✓    ✓    ✓     ✓     Max, Add, Constant                   
      MinimumScalar        ✓    ✓    ✓     ✓     Add, Constant, Min                   
      MaximumScalar        ✓    ✓    ✓     ✓     Max, Add, Constant                   
      LogicalAnd           ✓    ✓    ✓     ✓     And                                  
      LogicalOr            ✓    ✓    ✓     ✓     Or                                   
      LogicalXor           ✓    ✓    ✓     ✓     Xor                                  
      Equal                X    X    X     ✓     Equal                                
      NotEqual             X    X    X     ✓     Not, Equal                           
      GreaterEqual         ✓    ✓    ✓     ✓     Not, Less                            
      Greater              ✓    ✓    ✓     ✓     Greater                              
      LessEqual            ✓    ✓    ✓     ✓     Not, Greater                         
      Less                 ✓    ✓    ✓     ✓     Less                                 
      LogicalAndScalar     ✓    ✓    ✓     ✓     Constant, And                        
      LogicalOrScalar      ✓    ✓    ✓     ✓     Constant, Or                         
      LogicalXorScalar     ✓    ✓    ✓     ✓     Constant, Xor                        
      EqualScalar          X    X    X     ✓     Constant, Equal                      
      NotEqualScalar       X    X    X     ✓     Not, Equal, Constant                 
      GreaterEqualScalar   ✓    ✓    ✓     ✓     Not, Constant, Less                  
      GreaterScalar        ✓    ✓    ✓     ✓     Constant, Greater                    
      LessEqualScalar      ✓    ✓    ✓     ✓     Not, Constant, Greater               
      LessScalar           ✓    ✓    ✓     ✓     Constant, Less                       
      LogicalNot           ✓    ✓    ✓     ✓     Not                                  
      IsNaN                X    ✓    ✓     ✓     IsNaN                                
      IsInf                X    X    ✓     ✓     IsInf                                
      ResetNaN             X    ✓    ✓     ✓     Constant, Where, IsNaN               
      ResetInf             X    X    ✓     ✓     IsInf, Constant, Where               
      Where                X    ✓    ✓     ✓     Where                                
    =====================  ===  ===  ====  ====  ======================  =============


Math
^^^^

Count 22/22
 

    =================  ===  ===  ====  ====  ==================  =============
     NNabla Function    7    9    10    11        ONNX Op         Description 
    =================  ===  ===  ====  ====  ==================  =============
      Constant         ✓    ✓    ✓     ✓     Constant, Identity               
      Arange           ✓    ✓    ✓     ✓     Constant, Identity               
      Abs              ✓    ✓    ✓     ✓     Abs                              
      Exp              ✓    ✓    ✓     ✓     Exp                              
      Log              ✓    ✓    ✓     ✓     Log                              
      Identity         ✓    ✓    ✓     ✓     Identity                         
      BatchMatmul      ✓    ✓    ✓     ✓     Transpose, MatMul                
      Round            X    X    X     ✓     Round                            
      Ceil             ✓    ✓    ✓     ✓     Ceil                             
      Floor            ✓    ✓    ✓     ✓     Floor                            
      Sin              ✓    ✓    ✓     ✓     Sin                              
      Cos              ✓    ✓    ✓     ✓     Cos                              
      Tan              ✓    ✓    ✓     ✓     Tan                              
      Sinh             X    ✓    ✓     ✓     Sinh                             
      Cosh             X    ✓    ✓     ✓     Cosh                             
      ASin             ✓    ✓    ✓     ✓     Asin                             
      ACos             ✓    ✓    ✓     ✓     Acos                             
      ATan             ✓    ✓    ✓     ✓     Atan                             
      ATan2            ✓    ✓    ✓     ✓     Atan, Div                        
      ASinh            X    ✓    ✓     ✓     Asinh                            
      ACosh            X    ✓    ✓     ✓     Acosh                            
      ATanh            X    ✓    ✓     ✓     Atanh                            
    =================  ===  ===  ====  ====  ==================  =============


Array Manipulation
^^^^^^^^^^^^^^^^^^

Count 12/19
 

    =================  ===  ===  ====  ====  ===========================  =================================================================================================================
     NNabla Function    7    9    10    11             ONNX Op                                                               Description                                                   
    =================  ===  ===  ====  ====  ===========================  =================================================================================================================
      Concatenate      ✓    ✓    ✓     ✓     Concat                                                                                                                                        
      Split            ✓    ✓    ✓     ✓     Squeeze, Split                                                                                                                                
      Stack            ✓    ✓    ✓     ✓     Concat, Unsqueeze                                                                                                                             
      Slice            △    △    ✓     ✓     Slice, Constant              ONNX slice cannot support step != 1 on opset < 10.                                                               
      Pad              △    △    △     △     Constant, Pad                When the mode of the pad is reflect, if the size of the pad exceeds the input size, onnxruntime cannot handle it.
      Transpose        ✓    ✓    ✓     ✓     Transpose                                                                                                                                     
      Broadcast        X    ✓    ✓     ✓                                                                                                                                                   
      BroadcastTo      ✓    ✓    ✓     ✓                                                                                                                                                   
      Tile             ✓    ✓    ✓     ✓     Tile, Constant, Reshape                                                                                                                       
      OneHot           X    ✓    ✓     ✓     Flatten, Reshape, Gather                                                                                                                      
      Flip             ✓    ✓    ✓     ✓     Identity, Transpose, Gather                                                                                                                   
      Shift                                                               Not yet implemented.                                                                                             
      Sort                                                                Not yet implemented.                                                                                             
      Reshape          ✓    ✓    ✓     ✓     Constant, Reshape                                                                                                                             
      MatrixDiag                                                          Not yet implemented.                                                                                             
      MatrixDiagPart                                                      Not yet implemented.                                                                                             
      Assign                                                              Not yet implemented.                                                                                             
      GatherNd                                                            Not yet implemented.                                                                                             
      ScatterNd                                                           Not yet implemented.                                                                                             
    =================  ===  ===  ====  ====  ===========================  =================================================================================================================


Signal Processing
^^^^^^^^^^^^^^^^^

Count 1/3
 

    =================  ===  ===  ====  ====  ===============  ====================
     NNabla Function    7    9    10    11       ONNX Op          Description     
    =================  ===  ===  ====  ====  ===============  ====================
      Interpolate      X    X    △     ✓     Resize, Reshape                      
      FFT                                                     Not yet implemented.
      IFFT                                                    Not yet implemented.
    =================  ===  ===  ====  ====  ===============  ====================


Stochasticity
^^^^^^^^^^^^^

Count 0/11
 

    ====================  ===  ===  ====  ====  =========  ==================================================================================================================
      NNabla Function      7    9    10    11    ONNX Op                                                      Description                                                    
    ====================  ===  ===  ====  ====  =========  ==================================================================================================================
      Dropout             X    X    X     X     Dropout    The Dropout in nnabla has no test mode and contains random parameters, so the test result is not the same as onnx.
      TopKData                                             Not yet implemented.                                                                                              
      TopKGrad                                             Not yet implemented.                                                                                              
      Rand                                                 Not yet implemented.                                                                                              
      Randint                                              Not yet implemented.                                                                                              
      Randn                                                Not yet implemented.                                                                                              
      RandomChoice                                         Not yet implemented.                                                                                              
      RandomCrop                                           Not yet implemented.                                                                                              
      RandomFlip                                           Not yet implemented.                                                                                              
      RandomShift                                          Not yet implemented.                                                                                              
      ImageAugmentation                                    Not yet implemented.                                                                                              
    ====================  ===  ===  ====  ====  =========  ==================================================================================================================


Loss Functions
^^^^^^^^^^^^^^

Count 0/9
 

    ==========================  ===  ===  ====  ====  =========  ====================
         NNabla Function         7    9    10    11    ONNX Op       Description     
    ==========================  ===  ===  ====  ====  =========  ====================
      SigmoidCrossEntropy                                        Not yet implemented.
      BinaryCrossEntropy                                         Not yet implemented.
      SoftmaxCrossEntropy                                        Not yet implemented.
      CategoricalCrossEntropy                                    Not yet implemented.
      SquaredError                                               Not yet implemented.
      AbsoluteError                                              Not yet implemented.
      HuberLoss                                                  Not yet implemented.
      EpsilonInsensitiveLoss                                     Not yet implemented.
      KLMultinomial                                              Not yet implemented.
    ==========================  ===  ===  ====  ====  =========  ====================


Quantization Neural Network Layers
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Count 6/12
 

    ===========================  ===  ===  ====  ====  =========================  ====================
          NNabla Function         7    9    10    11            ONNX Op               Description     
    ===========================  ===  ===  ====  ====  =========================  ====================
      BinarySigmoid              X    ✓    ✓     ✓     Constant, Greater, Where                       
      BinaryTanh                 X    ✓    ✓     ✓     Constant, Greater, Where                       
      BinaryConnectAffine        ✓    ✓    ✓     ✓     Reshape, Gemm                                  
      BinaryConnectConvolution   ✓    ✓    ✓     ✓     Reshape, Conv                                  
      BinaryWeightAffine         ✓    ✓    ✓     ✓     Add, Reshape, Mul, MatMul                      
      BinaryWeightConvolution    ✓    ✓    ✓     ✓     Add, Mul, Reshape, Conv                        
      INQAffine                                                                   Not yet implemented.
      INQConvolution                                                              Not yet implemented.
      FixedPointQuantize                                                          Not yet implemented.
      MinMaxQuantize                                                              Not yet implemented.
      Pow2Quantize                                                                Not yet implemented.
      Prune                                                                       Not yet implemented.
    ===========================  ===  ===  ====  ====  =========================  ====================


Validation
^^^^^^^^^^

Count 0/3
 

    ==================  ===  ===  ====  ====  =========  ====================
     NNabla Function     7    9    10    11    ONNX Op       Description     
    ==================  ===  ===  ====  ====  =========  ====================
      TopNError                                          Not yet implemented.
      BinaryError                                        Not yet implemented.
      ConfusionMatrix                                    Not yet implemented.
    ==================  ===  ===  ====  ====  =========  ====================


Unsupported, Special Use
^^^^^^^^^^^^^^^^^^^^^^^^

Count 0/5
 

    =====================  ===  ===  ====  ====  =========  ====================
       NNabla Function      7    9    10    11    ONNX Op       Description     
    =====================  ===  ===  ====  ====  =========  ====================
      VATNoise                                              Not yet implemented.
      Unlink                                                Not yet implemented.
      Sink                                                  Not yet implemented.
      NmsDetection2d                                        Not yet implemented.
      MaxPoolingBackward                                    Not yet implemented.
    =====================  ===  ===  ====  ====  =========  ====================





Tensorflow Support Status
=========================

Import
------

- ✓: Supported
- △: Partially supported
- X: Supported, but test failed.
- Empty: Not support yet.


Total: 109/122

.. table:: Tensorflow support status

    ======================  ========  ==================================================  ====================
     Tensorflow Function     Status                      NNabla Func                          Description     
    ======================  ========  ==================================================  ====================
      Abs                      ✓      Abs                                                                     
      Acos                     ✓      ACos                                                                    
      Acosh                    ✓      ACosh                                                                   
      Add                      ✓      Add2                                                                    
      AddN                     ✓      AddN                                                                    
      All                      ✓      Reshape, Min, Greater                                                   
      Any                      ✓      Reshape, Sum, Greater                                                   
      ArgMax                   ✓      Max                                                                     
      ArgMin                   ✓      Min                                                                     
      Asin                     ✓      ASin                                                                    
      Asinh                    ✓      ASinh                                                                   
      Atan                     ✓      ATan                                                                    
      Atan2                    ✓      Add2, Sub2, Reshape, Mul2, Sign, ATan, Div2                             
      Atanh                    ✓      ATanh                                                                   
      AvgPool                  △      Transpose, Pad, AveragePooling                                          
      AvgPool3D                △      AveragePooling, Transpose, Pad                                          
      BatchMatMul              ✓      Transpose, BatchMatmul                                                  
      BatchNormalization       ✓      RDivScalar, Add2, Sub2, Reshape, Mul2, PowScalar                        
      BiasAdd                  ✓      Reshape, Add2                                                           
      BroadcastTo              ✓                                                                              
      Cast                     X      NA                                                  Not yet implemented.
      Ceil                     ✓      Ceil                                                                    
      ClipByValue              ✓      Maximum2, Reshape, Minimum2                                             
      Concat                   ✓      Concatenate                                                             
      ConcatV2                 ✓      Concatenate                                                             
      Const                    ✓      NA                                                                      
      Conv1D                   △      Convolution, Reshape, Transpose, Pad                                    
      Conv1DTranspose          △      Deconvolution, Reshape, Transpose                                       
      Conv2D                   △      Convolution, Transpose, Pad                                             
      Conv2DBackpropInput      △      Deconvolution, Transpose                                                
      Conv3D                   △      Convolution, Transpose, Pad                                             
      Conv3DBackpropInput      △      Deconvolution, Transpose, Pad                                           
      Cos                      ✓      Cos                                                                     
      Cosh                     ✓      Cosh                                                                    
      Crelu                    ✓      ReLU, MulScalar, Concatenate                                            
      Cumsum                   X                                                          Not yet implemented.
      DepthToSpace             ✓      Reshape, Transpose                                                      
      DepthwiseConv2d          △      Convolution, Reshape, Transpose, Pad                                    
      Div                      ✓      Div2                                                                    
      Elu                      ✓      ELU                                                                     
      Equal                    ✓      Equal                                                                   
      Erf                      X                                                          Not yet implemented.
      Erfc                     X                                                          Not yet implemented.
      Exp                      ✓      Exp                                                                     
      ExpandDims               ✓      Reshape                                                                 
      Floor                    ✓      Floor                                                                   
      FloorDiv                 ✓      Floor, Div2                                                             
      FloorMod                 ✓      Sub2, Floor, Div2, Mul2                                                 
      GatherNd                 X                                                          Not yet implemented.
      GatherV2                 X      Slice, Concatenate                                  Not yet implemented.
      Greater                  ✓      Greater                                                                 
      GreaterEqual             ✓      Less, LogicalNot                                                        
      Identity                 ✓      Identity                                                                
      IsInf                    ✓      IsInf                                                                   
      IsNan                    ✓      IsNaN                                                                   
      LeakyRelu                ✓      LeakyReLU                                                               
      Less                     ✓      Less                                                                    
      LessEqual                ✓      LogicalNot, Greater                                                     
      Log                      ✓      Log                                                                     
      LogSigmoid               ✓      SoftPlus, MulScalar                                                     
      LogSoftmax               ✓      Add2, Sum, Transpose, Exp, Sub2, Reshape, Max, Log                      
      LogicalAnd               ✓      LogicalAnd                                                              
      LogicalNot               ✓      LogicalNot                                                              
      LogicalOr                ✓      LogicalOr                                                               
      LogicalXor               ✓      LogicalAnd, LogicalNot, LogicalOr                                       
      Max                      ✓      Max                                                                     
      MaxPool                  △      Reshape, Transpose, MaxPooling, Pad                                     
      MaxPool3D                △      MaxPooling, Pad, Transpose                                              
      MaxPoolWithArgmax        X                                                          Not yet implemented.
      Maximum                  ✓      Maximum2                                                                
      Mean                     ✓      Mean                                                                    
      Min                      ✓      Min                                                                     
      Minimum                  ✓      Minimum2                                                                
      Mul                      ✓      Mul2                                                                    
      Neg                      ✓      MulScalar                                                               
      NotEqual                 ✓      Equal, LogicalNot                                                       
      Pack                     ✓      Reshape, Concatenate                                                    
      Pad                      △      Pad                                                                     
      Pow                      ✓      Pow2                                                                    
      Prod                     ✓      Prod                                                                    
      RealDiv                  ✓      Div2                                                                    
      Reciprocal               ✓      RDivScalar                                                              
      Relu                     ✓      ReLU                                                                    
      Relu6                    ✓      MinimumScalar, MaximumScalar                                            
      Reshape                  ✓      Reshape                                                                 
      ReverseSequence          X                                                          Not yet implemented.
      ReverseV2                X                                                          Not yet implemented.
      Round                    ✓      Round                                                                   
      Rsqrt                    ✓      RDivScalar, PowScalar                                                   
      Selu                     ✓      SELU                                                                    
      Shape                    X                                                          Not yet implemented.
      Sigmoid                  ✓      Sigmoid                                                                 
      Sign                     ✓      Sign                                                                    
      Sin                      ✓      Sin                                                                     
      Sinh                     ✓      Sinh                                                                    
      Size                     X                                                          Not yet implemented.
      Slice                    ✓      Slice                                                                   
      Softmax                  ✓      Transpose, Sum, Exp, Sub2, Reshape, Max, Div2                           
      Softplus                 ✓      SoftPlus                                                                
      Softsign                 ✓      SoftSign                                                                
      SpaceToDepth             ✓      Reshape, Transpose                                                      
      Split                    ✓      Stack, Split                                                            
      SplitV                   ✓      Stack, Split                                                            
      Sqrt                     ✓      PowScalar                                                               
      Square                   ✓      Mul2                                                                    
      SquaredDifference        ✓      Sub2, Mul2                                                              
      Squeeze                  ✓      Reshape                                                                 
      StopGradient             ✓      Identity                                                                
      StridedSlice             △      Slice                                                                   
      Sub                      ✓      Sub2                                                                    
      Sum                      ✓      Sum                                                                     
      Swish                    ✓      Sigmoid, Mul2                                                           
      Tan                      ✓      Tan                                                                     
      Tanh                     ✓      Tanh                                                                    
      Tile                     ✓      Tile                                                                    
      TopKV2                   X                                                          Not yet implemented.
      Transpose                ✓      Transpose                                                               
      TruncateDiv              ✓      Div2                                                                    
      TruncateMod              X                                                          Not yet implemented.
      Unpack                   ✓      Stack, Reshape, Split                                                   
      Where                    △      Where                                                                   
      ZerosLike                ✓      NA                                                                      
    ======================  ========  ==================================================  ====================





Export
------

- ✓: Supported
- △: Partially supported
- X: Supported, but test failed.
- Empty: Not support yet.


Total: 120/173

Neural Network Layer
^^^^^^^^^^^^^^^^^^^^

Count 11/14
 

    =========================  ========  ==================================================================================
         NNabla Function        Status                                      Description                                    
    =========================  ========  ==================================================================================
      Affine                   ✓                                                                                           
      RNN                                Not yet implemented.                                                              
      LSTM                               Not yet implemented.                                                              
      GRU                                Not yet implemented.                                                              
      Convolution              △         The cases `dilations` and `strides` larger than 1 are not supported by tensorflow.
      DepthwiseConvolution     △         The cases `dilations` and `strides` larger than 1 are not supported by tensorflow.
      Deconvolution            △         The cases `dilations` larger than 1 are not supported by tensorflow.              
      DepthwiseDeconvolution   △         The cases `dilations` larger than 1 are not supported by tensorflow.              
      MaxPooling               ✓                                                                                           
      AveragePooling           △         Currently only supports the cases both ignore_border and including_pad are True.  
      GlobalAveragePooling     ✓                                                                                           
      SumPooling               ✓                                                                                           
      Unpooling                △         The kernel only supports 2d.                                                      
      Embed                    ✓                                                                                           
    =========================  ========  ==================================================================================


Neural Network Activation Functions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Count 21/21
 

    =================  ========  =============
     NNabla Function    Status    Description 
    =================  ========  =============
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
      ReLU6            ✓                      
      HardSigmoid      ✓                      
      HardTanh         ✓                      
      LogSigmoid       ✓                      
      SoftPlus         ✓                      
      SoftSign         ✓                      
      TanhShrink       ✓                      
      Sinc             ✓                      
    =================  ========  =============


Normalization
^^^^^^^^^^^^^

Count 2/6
 

    ==========================  ========  ====================
         NNabla Function         Status       Description     
    ==========================  ========  ====================
      FusedBatchNormalization   ✓                             
      BatchNormalization        ✓                             
      SyncBatchNormalization              Not yet implemented.
      MeanSubtraction                     Not yet implemented.
      ClipGradByValue                     Not yet implemented.
      ClipGradByNorm                      Not yet implemented.
    ==========================  ========  ====================


Reduction
^^^^^^^^^

Count 5/7
 

    =================  ========  ====================
     NNabla Function    Status       Description     
    =================  ========  ====================
      Sum              ✓                             
      Mean             ✓                             
      Max              ✓                             
      Min              ✓                             
      Prod             ✓                             
      ReduceSum                  Not yet implemented.
      ReduceMean                 Not yet implemented.
    =================  ========  ====================


Arithmetic
^^^^^^^^^^

Count 11/12
 

    =================  ========  ====================
     NNabla Function    Status       Description     
    =================  ========  ====================
      Add2             ✓                             
      BcAdd2                     Not yet implemented.
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
    =================  ========  ====================


Logical
^^^^^^^

Count 29/29
 

    =====================  ========  =============
       NNabla Function      Status    Description 
    =====================  ========  =============
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
    =====================  ========  =============


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

Count 12/19
 

    =================  ========  ================================================================================================================
     NNabla Function    Status                                                     Description                                                   
    =================  ========  ================================================================================================================
      Concatenate      ✓                                                                                                                         
      Split            ✓                                                                                                                         
      Stack            ✓                                                                                                                         
      Slice            ✓                                                                                                                         
      Pad              △         When the mode of the pad is reflect, if the size of the pad exceeds the input size, tensorflow cannot handle it.
      Transpose        ✓                                                                                                                         
      Broadcast        ✓                                                                                                                         
      BroadcastTo      ✓                                                                                                                         
      Tile             ✓                                                                                                                         
      OneHot           ✓                                                                                                                         
      Flip             ✓                                                                                                                         
      Shift                      Not yet implemented.                                                                                            
      Sort                       Not yet implemented.                                                                                            
      Reshape          ✓                                                                                                                         
      MatrixDiag                 Not yet implemented.                                                                                            
      MatrixDiagPart             Not yet implemented.                                                                                            
      Assign                     Not yet implemented.                                                                                            
      GatherNd                   Not yet implemented.                                                                                            
      ScatterNd                  Not yet implemented.                                                                                            
    =================  ========  ================================================================================================================


Signal Processing
^^^^^^^^^^^^^^^^^

Count 1/3
 

    =================  ========  ====================
     NNabla Function    Status       Description     
    =================  ========  ====================
      Interpolate      ✓                             
      FFT                        Not yet implemented.
      IFFT                       Not yet implemented.
    =================  ========  ====================


Stochasticity
^^^^^^^^^^^^^

Count 0/11
 

    ====================  ========  ========================================================================================================================
      NNabla Function      Status                                                         Description                                                       
    ====================  ========  ========================================================================================================================
      Dropout             X         The Dropout in nnabla has no test mode and contains random parameters, so the test result is not the same as tensorflow.
      TopKData                      Not yet implemented.                                                                                                    
      TopKGrad                      Not yet implemented.                                                                                                    
      Rand                          Not yet implemented.                                                                                                    
      Randint                       Not yet implemented.                                                                                                    
      Randn                         Not yet implemented.                                                                                                    
      RandomChoice                  Not yet implemented.                                                                                                    
      RandomCrop                    Not yet implemented.                                                                                                    
      RandomFlip                    Not yet implemented.                                                                                                    
      RandomShift                   Not yet implemented.                                                                                                    
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


Quantization Neural Network Layers
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Count 6/12
 

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

Count 0/5
 

    =====================  ========  ====================
       NNabla Function      Status       Description     
    =====================  ========  ====================
      VATNoise                       Not yet implemented.
      Unlink                         Not yet implemented.
      Sink                           Not yet implemented.
      NmsDetection2d                 Not yet implemented.
      MaxPoolingBackward             Not yet implemented.
    =====================  ========  ====================




Tensorflow Lite Support Status
==============================


Export
------

- ✓: Supported
- △: Partially supported
- X: Supported, but test failed.
- Empty: Not support yet.


Total: 98/173

Neural Network Layer
^^^^^^^^^^^^^^^^^^^^

Count 8/14
 

    =========================  ========
         NNabla Function        Status 
    =========================  ========
      Affine                   ✓       
      RNN                              
      LSTM                             
      GRU                              
      Convolution              △       
      DepthwiseConvolution     △       
      Deconvolution            △       
      DepthwiseDeconvolution   △       
      MaxPooling               X       
      AveragePooling           X       
      GlobalAveragePooling     ✓       
      SumPooling               X       
      Unpooling                △       
      Embed                    ✓       
    =========================  ========


Neural Network Activation Functions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Count 20/21
 

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
      PReLU            ✓       
      GELU             ✓       
      ReLU6            ✓       
      HardSigmoid      ✓       
      HardTanh         ✓       
      LogSigmoid       ✓       
      SoftPlus         ✓       
      SoftSign         ✓       
      TanhShrink       ✓       
      Sinc             X       
    =================  ========


Normalization
^^^^^^^^^^^^^

Count 0/6
 

    ==========================  ========
         NNabla Function         Status 
    ==========================  ========
      FusedBatchNormalization   X       
      BatchNormalization        X       
      SyncBatchNormalization            
      MeanSubtraction                   
      ClipGradByValue                   
      ClipGradByNorm                    
    ==========================  ========


Reduction
^^^^^^^^^

Count 5/7
 

    =================  ========
     NNabla Function    Status 
    =================  ========
      Sum              ✓       
      Mean             ✓       
      Max              ✓       
      Min              ✓       
      Prod             ✓       
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

Count 25/29
 

    =====================  ========
       NNabla Function      Status 
    =====================  ========
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
      IsInf                X       
      ResetNaN             X       
      ResetInf             X       
      Where                X       
    =====================  ========


Math
^^^^

Count 14/22
 

    =================  ========
     NNabla Function    Status 
    =================  ========
      Constant         ✓       
      Arange           ✓       
      Abs              ✓       
      Exp              ✓       
      Log              ✓       
      Identity         ✓       
      BatchMatmul      ✓       
      Round            X       
      Ceil             ✓       
      Floor            ✓       
      Sin              ✓       
      Cos              ✓       
      Tan              ✓       
      Sinh             ✓       
      Cosh             ✓       
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

Count 11/19
 

    =================  ========
     NNabla Function    Status 
    =================  ========
      Concatenate      ✓       
      Split            ✓       
      Stack            ✓       
      Slice            △       
      Pad              X       
      Transpose        ✓       
      Broadcast        ✓       
      BroadcastTo      ✓       
      Tile             ✓       
      OneHot           ✓       
      Flip             ✓       
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
      BinarySigmoid              X       
      BinaryTanh                 X       
      BinaryConnectAffine        ✓       
      BinaryConnectConvolution   △       
      BinaryWeightAffine         ✓       
      BinaryWeightConvolution    △       
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



