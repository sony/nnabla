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
  1.4.1


Import
------

- ✓: onnx specification defined, and supported.
- X: onnx specification defined, but not support yet.
- Empty: Not defined (Support status follows latest).


Total: 81/129

.. table:: 

    ==============================  ===  ===  ===  ===  ===  ===  ===  ===  ===  ============================================================  ==============================================================================================================================================================================================================
            ONNX Operator            1    2    3    4    5    6    7    8    9                           NNabla Func                                                                                                                            Description                                                                                                  
    ==============================  ===  ===  ===  ===  ===  ===  ===  ===  ===  ============================================================  ==============================================================================================================================================================================================================
     Abs                            ✓                        ✓                   Abs                                                                                                                                                                                                                                                                         
     Acos                                                         ✓              ACos                                                                                                                                                                                                                                                                        
     Acosh                                                                  ✓    ACosh                                                                                                                                                                                                                                                                       
     Add                            ✓                        ✓    ✓              Add2, Reshape                                                                                                                                                                                                                                                               
     And                            ✓                        ✓    ✓              LogicalAnd, Reshape                                                                                                                                                                                                                                                         
     ArgMax                         ✓                        ✓                   Max                                                                                                                                                                                                                                                                         
     ArgMin                         ✓                        ✓                   Min                                                                                                                                                                                                                                                                         
     Asin                                                         ✓              ASin                                                                                                                                                                                                                                                                        
     Asinh                                                                  ✓    ASinh                                                                                                                                                                                                                                                                       
     Atan                                                         ✓              ATan                                                                                                                                                                                                                                                                        
     Atanh                                                                  ✓    ATanh                                                                                                                                                                                                                                                                       
     AveragePool                    ✓                        ✓    ✓              Pad, AveragePooling                                           Not all features are verified, since some features are not supported by caffe2. Those features can be verified by ONNXRuntime when opset > 6. Some feature is not supported by Nnabla such as Pad's edge mode.
     BatchNormalization             X                        X    X         ✓    BatchNormalization                                                                                                                                                                                                                                                          
     Cast                           X                        X              X                                                                  Not yet implemented.                                                                                                                                                                                          
     Ceil                           ✓                        ✓                   Ceil                                                                                                                                                                                                                                                                        
     Clip                           ✓                        ✓                   MinimumScalar, MaximumScalar                                                                                                                                                                                                                                                
     Compress                                                               X                                                                  Not yet implemented.                                                                                                                                                                                          
     Concat                         ✓              ✓         ✓                   Concatenate                                                                                                                                                                                                                                                                 
     Constant                       ✓                        ✓              X    Identity                                                                                                                                                                                                                                                                    
     ConstantOfShape                                                        X                                                                  Not yet implemented.                                                                                                                                                                                          
     Conv                           ✓                        ✓                   Convolution                                                                                                                                                                                                                                                                 
     ConvTranspose                  ✓                        ✓                   Pad, Deconvolution                                                                                                                                                                                                                                                          
     Cos                                                          ✓              Cos                                                                                                                                                                                                                                                                         
     Cosh                                                                   ✓    Cosh                                                                                                                                                                                                                                                                        
     DepthToSpace                   ✓                        ✓                   Transpose, Reshape                                                                                                                                                                                                                                                          
     Div                            ✓                        ✓    ✓              Div2, Reshape                                                                                                                                                                                                                                                               
     Dropout                        X                        X    ✓              Identity                                                                                                                                                                                                                                                                    
     Elu                            ✓                        ✓                   ELU                                                                                                                                                                                                                                                                         
     Equal                          ✓                        ✓    ✓              Equal, Reshape                                                                                                                                                                                                                                                              
     Erf                                                                    X                                                                  Not yet implemented.                                                                                                                                                                                          
     Exp                            ✓                        ✓                   Exp                                                                                                                                                                                                                                                                         
     Expand                                                            X                                                                       Not yet implemented.                                                                                                                                                                                          
     EyeLike                                                                X                                                                  Not yet implemented.                                                                                                                                                                                          
     Flatten                        ✓                        ✓              ✓    Reshape                                                                                                                                                                                                                                                                     
     Floor                          ✓                        ✓                   Floor                                                                                                                                                                                                                                                                       
     GRU                            X         X                   X                                                                            Not yet implemented.                                                                                                                                                                                          
     Gather                         X                                                                                                          Not yet implemented.                                                                                                                                                                                          
     Gemm                           ✓                        ✓    ✓         ✓    BatchMatmul, Add2, Broadcast, MulScalar                                                                                                                                                                                                                                     
     GlobalAveragePool              ✓                        ✓                   GlobalAveragePooling                                                                                                                                                                                                                                                        
     GlobalLpPool                   X    X                                                                                                     Not yet implemented.                                                                                                                                                                                          
     GlobalMaxPool                  X                                                                                                          Not yet implemented.                                                                                                                                                                                          
     Greater                        ✓                        ✓    ✓         ✓    Greater, Reshape                                                                                                                                                                                                                                                            
     HardSigmoid                    X                        X                                                                                 Not yet implemented.                                                                                                                                                                                          
     Hardmax                        X                                                                                                          Not yet implemented.                                                                                                                                                                                          
     Identity                       ✓                        ✓                   Identity                                                                                                                                                                                                                                                                    
     If                             X                                                                                                          Not yet implemented.                                                                                                                                                                                          
     InstanceNormalization          X                        X                                                                                 Not yet implemented.                                                                                                                                                                                          
     IsNaN                                                                  ✓    IsNaN                                                                                                                                                                                                                                                                       
     LRN                            ✓                        ✓                   PowScalar, MulScalar, Div2, SumPooling, AddScalar, Transpose                                                                                                                                                                                                                
     LSTM                           X                             X                                                                            Not yet implemented.                                                                                                                                                                                          
     LeakyRelu                      ✓                        ✓                   LeakyReLU                                                                                                                                                                                                                                                                   
     Less                           ✓                        ✓    ✓         ✓    Less, Reshape                                                                                                                                                                                                                                                               
     Log                            ✓                        ✓                   Log                                                                                                                                                                                                                                                                         
     LogSoftmax                     ✓                        ✓                   Max, Reshape, Exp, Log, Add2, Sub2, Sum                                                                                                                                                                                                                                     
     Loop                           X                                                                                                          Not yet implemented.                                                                                                                                                                                          
     LpNormalization                X                                                                                                          Not yet implemented.                                                                                                                                                                                          
     LpPool                         X    X                                                                                                     Not yet implemented.                                                                                                                                                                                          
     MatMul                         ✓                        ✓              ✓    BatchMatmul                                                                                                                                                                                                                                                                 
     Max                            ✓                        ✓         ✓    ✓    Maximum2                                                                                                                                                                                                                                                                    
     MaxPool                        ✓                        ✓         X         Pad, MaxPooling                                               Not all features are verified, since some features are not supported by caffe2. Those features can be verified by ONNXRuntime.                                                                                
     MaxRoiPool                     X                                                                                                          Not yet implemented.                                                                                                                                                                                          
     MaxUnpool                                                              X                                                                  Not yet implemented.                                                                                                                                                                                          
     Mean                           ✓                        ✓         ✓    ✓    Stack, Mean, Broadcast                                                                                                                                                                                                                                                      
     Min                            ✓                        ✓         ✓    ✓    Minimum2                                                                                                                                                                                                                                                                    
     Mul                            ✓                        ✓    ✓              Mul2, Reshape                                                                                                                                                                                                                                                               
     Multinomial                                                  X                                                                            Not yet implemented.                                                                                                                                                                                          
     Neg                            ✓                        ✓                   MulScalar                                                                                                                                                                                                                                                                   
     NonZero                                                                X                                                                  Not yet implemented.                                                                                                                                                                                          
     Not                            ✓                        ✓                   LogicalNot                                                                                                                                                                                                                                                                  
     OneHot                                                                 X                                                                  Not yet implemented.                                                                                                                                                                                          
     Or                             ✓                        ✓    ✓              LogicalOr, Reshape                                                                                                                                                                                                                                                          
     PRelu                          ✓                        ✓    X         X    PReLU                                                                                                                                                                                                                                                                       
     Pad                            ✓    ✓                   ✓                   Pad                                                           Onnx required to support "edge" mode, while nnabla does not support it.                                                                                                                                       
     Pow                            ✓                        ✓    ✓              Pow2, Reshape                                                                                                                                                                                                                                                               
     RNN                            X                             X                                                                            Not yet implemented.                                                                                                                                                                                          
     RandomNormal                   X                                                                                                          Not yet implemented.                                                                                                                                                                                          
     RandomNormalLike               X                                                                                                          Not yet implemented.                                                                                                                                                                                          
     RandomUniform                  X                                                                                                          Not yet implemented.                                                                                                                                                                                          
     RandomUniformLike              X                                                                                                          Not yet implemented.                                                                                                                                                                                          
     Reciprocal                     ✓                        ✓                   RDivScalar                                                                                                                                                                                                                                                                  
     ReduceL1                       X                                                                                                          Not yet implemented.                                                                                                                                                                                          
     ReduceL2                       X                                                                                                          Not yet implemented.                                                                                                                                                                                          
     ReduceLogSum                   X                                                                                                          Not yet implemented.                                                                                                                                                                                          
     ReduceLogSumExp                X                                                                                                          Not yet implemented.                                                                                                                                                                                          
     ReduceMax                      ✓                        ✓                   Max                                                                                                                                                                                                                                                                         
     ReduceMean                     ✓                        ✓                   Mean                                                                                                                                                                                                                                                                        
     ReduceMin                      ✓                        ✓                   Min                                                                                                                                                                                                                                                                         
     ReduceProd                     ✓                        ✓                   Prod                                                                                                                                                                                                                                                                        
     ReduceSum                      ✓                        ✓                   Sum                                                                                                                                                                                                                                                                         
     ReduceSumSquare                X                                                                                                          Not yet implemented.                                                                                                                                                                                          
     Relu                           ✓                        ✓                   ReLU                                                                                                                                                                                                                                                                        
     Reshape                        ✓                   ✓    ✓                   Reshape                                                                                                                                                                                                                                                                     
     Resize                                                                                                                                    Not yet implemented.                                                                                                                                                                                          
     Scan                                                              X    X                                                                  Not yet implemented.                                                                                                                                                                                          
     Scatter                                                                X                                                                  Not yet implemented.                                                                                                                                                                                          
     Selu                           ✓                        ✓                   SELU                                                                                                                                                                                                                                                                        
     Shape                          X                                                                                                          Not yet implemented.                                                                                                                                                                                          
     Shrink                                                                 X                                                                  Not yet implemented.                                                                                                                                                                                          
     Sigmoid                        ✓                        ✓                   Sigmoid                                                                                                                                                                                                                                                                     
     Sign                                                                   ✓    Sign                                                                                                                                                                                                                                                                        
     Sin                                                          ✓              Sin                                                                                                                                                                                                                                                                         
     Sinh                                                                   ✓    Sinh                                                                                                                                                                                                                                                                        
     Size                           X                                                                                                          Not yet implemented.                                                                                                                                                                                          
     Slice                          ✓                        ✓                   Slice                                                                                                                                                                                                                                                                       
     Softmax                        ✓                        ✓                   Max, Reshape, Exp, Div2, Sub2, Sum                                                                                                                                                                                                                                          
     Softplus                       ✓                        ✓                   AddScalar, Exp, Log                                                                                                                                                                                                                                                         
     Softsign                       ✓                        ✓                   AddScalar, Abs, Div2                                                                                                                                                                                                                                                        
     SpaceToDepth                   ✓                        ✓                   Transpose, Reshape                                                                                                                                                                                                                                                          
     Split                          ✓    ✓                   ✓                   Stack, Split                                                                                                                                                                                                                                                                
     Sqrt                           ✓                        ✓                   PowScalar                                                                                                                                                                                                                                                                   
     Squeeze                        ✓                        ✓                   Reshape                                                                                                                                                                                                                                                                     
     StringNormalizer                                                                                                                          Not yet implemented.                                                                                                                                                                                          
     Sub                            ✓                        ✓    ✓              Sub2, Reshape                                                                                                                                                                                                                                                               
     Sum                            ✓                        ✓         ✓    ✓    Add2                                                                                                                                                                                                                                                                        
     Tan                                                          ✓              Tan                                                                                                                                                                                                                                                                         
     Tanh                           ✓                        ✓                   Tanh                                                                                                                                                                                                                                                                        
     TfIdfVectorizer                                                        X                                                                  Not yet implemented.                                                                                                                                                                                          
     ThresholdedRelu                                                                                                                           Not yet implemented.                                                                                                                                                                                          
     Tile                           ✓                        ✓                   Tile                                                                                                                                                                                                                                                                        
     TopK                           X                                                                                                          Not yet implemented.                                                                                                                                                                                          
     Transpose                      ✓                        ✓                   Transpose                                                                                                                                                                                                                                                                   
     Unsqueeze                      ✓                        ✓                   Reshape                                                                                                                                                                                                                                                                     
     Upsample                       ✓                        ✓    ✓         ✓    Unpooling                                                                                                                                                                                                                                                                   
     Where                                                                  X                                                                  Not yet implemented.                                                                                                                                                                                          
     Xor                            ✓                        ✓    ✓              LogicalXor, Reshape                                                                                                                                                                                                                                                         
     experimental ATen              X                                                                                                          Not yet implemented.                                                                                                                                                                                          
     experimental GRUUnit           X                                                                                                          Not yet implemented.                                                                                                                                                                                          
     experimental GivenTensorFill   X                                                                                                          Not yet implemented.                                                                                                                                                                                          
     experimental Scale             X                                                                                                          Not yet implemented.                                                                                                                                                                                          
    ==============================  ===  ===  ===  ===  ===  ===  ===  ===  ===  ============================================================  ==============================================================================================================================================================================================================



Export
------

- ✓: Support to export this opset.
- △: Partially support to export this opset (e.g. some cases cannot be supported, or not completely tested).
- X: Supported, but test failed.
- Empty: Not support corresponding opset version.

Total: 114/172

Neural Network Layer
^^^^^^^^^^^^^^^^^^^^

Count 11/14
 

    =========================  ===  ===  ===  ========================================  ======================================================================================
         NNabla Function        6    7    9                   ONNX Op                                                        Description                                      
    =========================  ===  ===  ===  ========================================  ======================================================================================
      Affine                   ✓    ✓    ✓    Gemm, Reshape                                                                                                                   
      RNN                                                                               Not yet implemented.                                                                  
      LSTM                                                                              Not yet implemented.                                                                  
      GRU                                                                               Not yet implemented.                                                                  
      Convolution              ✓    ✓    ✓    Conv, Reshape                                                                                                                   
      DepthwiseConvolution     ✓    ✓    ✓    Conv, Reshape                                                                                                                   
      Deconvolution            △    △    △    ConvTranspose, Reshape                    Caffe2 and onnxruntime do not support dilations != 1.                                 
      DepthwiseDeconvolution   △    △    △    ConvTranspose, Reshape                    Caffe2 and onnxruntime do not support dilations != 1.                                 
      MaxPooling               ✓    ✓    ✓    Pad, MaxPool, Reshape                                                                                                           
      AveragePooling           △    △    △    Pad, AveragePool, Reshape                 Currently only supports the cases where both ignore_border and including_pad are True.
      GlobalAveragePooling     ✓    ✓    ✓    GlobalAveragePool                                                                                                               
      SumPooling               X    ✓    ✓    Mul, Reshape, AveragePool, Pad, Constant                                                                                        
      Unpooling                △    ✓    ✓    Upsample, Reshape                         The kernel only supports 2d on opset 6.                                               
      Embed                    ✓    ✓    ✓    Gather                                                                                                                          
    =========================  ===  ===  ===  ========================================  ======================================================================================


Neural Network Activation Functions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Count 21/21
 

    =================  ===  ===  ===  ========================================  =================================================================================
     NNabla Function    6    7    9                   ONNX Op                                                      Description                                   
    =================  ===  ===  ===  ========================================  =================================================================================
      Sigmoid          ✓    ✓    ✓    Sigmoid                                                                                                                    
      Swish            ✓    ✓    ✓    Mul, Sigmoid                                                                                                               
      Tanh             ✓    ✓    ✓    Tanh                                                                                                                       
      ReLU             ✓    ✓    ✓    Relu                                                                                                                       
      LeakyReLU        ✓    ✓    ✓    LeakyRelu                                                                                                                  
      Softmax          △    ✓    ✓    Sub, Exp, ReduceMax, Div, ReduceSum       ONNX Add, Sub operator does not support multidirectional broadcasting on opset 6.
      LogSoftmax       △    ✓    ✓    Sub, Exp, ReduceMax, Log, ReduceSum                                                                                        
      ELU              ✓    ✓    ✓    Elu                                                                                                                        
      SELU             ✓    ✓    ✓    Selu                                                                                                                       
      CReLU            ✓    ✓    ✓    Relu, Neg, Concat                                                                                                          
      CELU             ✓    ✓    ✓    Neg, Concat, Elu                                                                                                           
      PReLU            ✓    ✓    ✓    PRelu, Reshape                                                                                                             
      GELU             ✓    ✓    ✓    Mul, Tanh, Div, Pow, Constant, Sqrt, Add                                                                                   
      ReLU6            ✓    ✓    ✓    Min, Constant, Relu                                                                                                        
      HardSigmoid      ✓    ✓    ✓    HardSigmoid                                                                                                                
      HardTanh         ✓    ✓    ✓    Min, Neg, Max, Constant                                                                                                    
      LogSigmoid       ✓    ✓    ✓    Sigmoid, Log                                                                                                               
      SoftPlus         ✓    ✓    ✓    Softplus                                                                                                                   
      SoftSign         ✓    ✓    ✓    Softsign                                                                                                                   
      TanhShrink       ✓    ✓    ✓    Sub, Tanh                                                                                                                  
      Sinc             X    X    ✓    Equal, Sin, Div, Constant, Where                                                                                           
    =================  ===  ===  ===  ========================================  =================================================================================


Normalization
^^^^^^^^^^^^^

Count 1/6
 

    ==========================  ===  ===  ===  ==================================================  =======================================================================================================
         NNabla Function         6    7    9                        ONNX Op                                                                      Description                                              
    ==========================  ===  ===  ===  ==================================================  =======================================================================================================
      FusedBatchNormalization                                                                      Not yet implemented.                                                                                   
      BatchNormalization        △    △    △    InstanceNormalization, BatchNormalization, Reshape  In inferring stage, caffe2 mistmatch onnx 1.4.x's implementation, "in-place" feature cannot be applied.
      SyncBatchNormalization                                                                       Not yet implemented.                                                                                   
      MeanSubtraction                                                                              Not yet implemented.                                                                                   
      ClipGradByValue                                                                              Not yet implemented.                                                                                   
      ClipGradByNorm                                                                               Not yet implemented.                                                                                   
    ==========================  ===  ===  ===  ==================================================  =======================================================================================================


Reduction
^^^^^^^^^

Count 5/7
 

    =================  ===  ===  ===  ==========  ====================
     NNabla Function    6    7    9    ONNX Op        Description     
    =================  ===  ===  ===  ==========  ====================
      Sum              ✓    ✓    ✓    ReduceSum                       
      Mean             ✓    ✓    ✓    ReduceMean                      
      Max              ✓    ✓    ✓    ReduceMax                       
      Min              ✓    ✓    ✓    ReduceMin                       
      Prod             ✓    ✓    ✓    ReduceProd                      
      ReduceSum                                   Not yet implemented.
      ReduceMean                                  Not yet implemented.
    =================  ===  ===  ===  ==========  ====================


Arithmetic
^^^^^^^^^^

Count 11/12
 

    =================  ===  ===  ===  =============  ============================================================================
     NNabla Function    6    7    9      ONNX Op                                     Description                                 
    =================  ===  ===  ===  =============  ============================================================================
      Add2             △    ✓    ✓    Add            ONNX Add operator does not support multidirectional broadcasting on opset 6.
      BcAdd2                                         Not yet implemented.                                                        
      Sub2             △    ✓    ✓    Sub            ONNX Sub operator does not support multidirectional broadcasting on opset 6.
      Mul2             △    ✓    ✓    Mul            ONNX Mul operator does not support multidirectional broadcasting on opset 6.
      Div2             △    ✓    ✓    Div            ONNX Div operator does not support multidirectional broadcasting on opset 6.
      Pow2             △    ✓    ✓    Pow            ONNX Pow operator does not support multidirectional broadcasting on opset 6.
      AddScalar        ✓    ✓    ✓    Add, Constant                                                                              
      MulScalar        ✓    ✓    ✓    Mul, Constant                                                                              
      PowScalar        ✓    ✓    ✓    Constant, Pow                                                                              
      RSubScalar       ✓    ✓    ✓    Sub                                                                                        
      RDivScalar       ✓    ✓    ✓    Div                                                                                        
      RPowScalar       ✓    ✓    ✓    Pow                                                                                        
    =================  ===  ===  ===  =============  ============================================================================


Logical
^^^^^^^

Count 26/29
 

    =====================  ===  ===  ===  ======================  ============================================================================
       NNabla Function      6    7    9          ONNX Op                                          Description                                 
    =====================  ===  ===  ===  ======================  ============================================================================
      Sign                 X    X    ✓    Sign                                                                                                
      Minimum2             △    ✓    ✓    Min, Add, Constant      ONNX Add operator does not support multidirectional broadcasting on opset 6.
      Maximum2             △    ✓    ✓    Max, Add, Constant      ONNX Add operator does not support multidirectional broadcasting on opset 6.
      MinimumScalar        ✓    ✓    ✓    Min, Add, Constant                                                                                  
      MaximumScalar        ✓    ✓    ✓    Max, Add, Constant                                                                                  
      LogicalAnd           ✓    ✓    ✓    And                                                                                                 
      LogicalOr            ✓    ✓    ✓    Or                                                                                                  
      LogicalXor           ✓    ✓    ✓    Xor                                                                                                 
      Equal                ✓    ✓    ✓    Equal                                                                                               
      NotEqual             ✓    ✓    ✓    Equal, Not                                                                                          
      GreaterEqual         ✓    ✓    ✓    Less, Not                                                                                           
      Greater              ✓    ✓    ✓    Greater                                                                                             
      LessEqual            ✓    ✓    ✓    Greater, Not                                                                                        
      Less                 ✓    ✓    ✓    Less                                                                                                
      LogicalAndScalar     ✓    ✓    ✓    And, Constant                                                                                       
      LogicalOrScalar      ✓    ✓    ✓    Constant, Or                                                                                        
      LogicalXorScalar     ✓    ✓    ✓    Constant, Xor                                                                                       
      EqualScalar          ✓    ✓    ✓    Equal, Constant                                                                                     
      NotEqualScalar       ✓    ✓    ✓    Equal, Constant, Not                                                                                
      GreaterEqualScalar   ✓    ✓    ✓    Constant, Less, Not                                                                                 
      GreaterScalar        ✓    ✓    ✓    Greater, Constant                                                                                   
      LessEqualScalar      ✓    ✓    ✓    Greater, Constant, Not                                                                              
      LessScalar           ✓    ✓    ✓    Constant, Less                                                                                      
      LogicalNot           ✓    ✓    ✓    Not                                                                                                 
      IsNaN                X    X    ✓    IsNaN                                                                                               
      IsInf                                                       Not yet implemented.                                                        
      ResetNaN                                                    Not yet implemented.                                                        
      ResetInf                                                    Not yet implemented.                                                        
      Where                X    X    ✓    Where                                                                                               
    =====================  ===  ===  ===  ======================  ============================================================================


Math
^^^^

Count 21/22
 

    =================  ===  ===  ===  ==========================  ====================
     NNabla Function    6    7    9            ONNX Op                Description     
    =================  ===  ===  ===  ==========================  ====================
      Constant         ✓    ✓    ✓    Constant, Identity                              
      Arange           ✓    ✓    ✓    Constant, Identity                              
      Abs              ✓    ✓    ✓    Abs                                             
      Exp              ✓    ✓    ✓    Exp                                             
      Log              ✓    ✓    ✓    Log                                             
      Identity         ✓    ✓    ✓    Identity                                        
      BatchMatmul      ✓    ✓    ✓    Transpose, MatMul, Reshape                      
      Round                                                       Not yet implemented.
      Ceil             ✓    ✓    ✓    Ceil                                            
      Floor            ✓    ✓    ✓    Floor                                           
      Sin              X    ✓    ✓    Sin                                             
      Cos              X    ✓    ✓    Cos                                             
      Tan              X    ✓    ✓    Tan                                             
      Sinh             X    X    ✓    Sinh                                            
      Cosh             X    X    ✓    Cosh                                            
      ASin             X    ✓    ✓    Asin                                            
      ACos             X    ✓    ✓    Acos                                            
      ATan             X    ✓    ✓    Atan                                            
      ATan2            X    ✓    ✓    Div, Atan                                       
      ASinh            X    X    ✓    Asinh                                           
      ACosh            X    X    ✓    Acosh                                           
      ATanh            X    X    ✓    Atanh                                           
    =================  ===  ===  ===  ==========================  ====================


Array Manipulation
^^^^^^^^^^^^^^^^^^

Count 12/19
 

    =================  ===  ===  ===  ===========================  ============================================================================================================================
     NNabla Function    6    7    9             ONNX Op                                                                    Description                                                         
    =================  ===  ===  ===  ===========================  ============================================================================================================================
      Concatenate      ✓    ✓    ✓    Concat                                                                                                                                                   
      Split            ✓    ✓    ✓    Squeeze, Split                                                                                                                                           
      Stack            ✓    ✓    ✓    Unsqueeze, Concat                                                                                                                                        
      Slice            △    △    △    Slice                        ONNX slice cannot support step != 1 on opset < 10.                                                                          
      Pad              △    △    △    Pad                          When the mode of the pad is reflect, if the size of the pad exceeds the input size, caffe2 and onnxruntime cannot handle it.
      Transpose        ✓    ✓    ✓    Transpose                                                                                                                                                
      Broadcast        X    X    ✓                                                                                                                                                             
      BroadcastTo      ✓    ✓    ✓                                                                                                                                                             
      Tile             ✓    ✓    ✓    Tile, Constant, Reshape                                                                                                                                  
      OneHot           ✓    ✓    ✓    Gather, Flatten, Reshape                                                                                                                                 
      Flip             ✓    ✓    ✓    Gather, Transpose, Identity                                                                                                                              
      Shift                                                        Not yet implemented.                                                                                                        
      Sort                                                         Not yet implemented.                                                                                                        
      Reshape          ✓    ✓    ✓    Constant, Reshape                                                                                                                                        
      MatrixDiag                                                   Not yet implemented.                                                                                                        
      MatrixDiagPart                                               Not yet implemented.                                                                                                        
      Assign                                                       Not yet implemented.                                                                                                        
      GatherNd                                                     Not yet implemented.                                                                                                        
      ScatterNd                                                    Not yet implemented.                                                                                                        
    =================  ===  ===  ===  ===========================  ============================================================================================================================


Signal Processing
^^^^^^^^^^^^^^^^^

Count 0/3
 

    =================  ===  ===  ===  =========  ====================
     NNabla Function    6    7    9    ONNX Op       Description     
    =================  ===  ===  ===  =========  ====================
      Interpolate                                Not yet implemented.
      FFT                                        Not yet implemented.
      IFFT                                       Not yet implemented.
    =================  ===  ===  ===  =========  ====================


Stochasticity
^^^^^^^^^^^^^

Count 0/11
 

    ====================  ===  ===  ===  =========  ==================================================================================================================
      NNabla Function      6    7    9    ONNX Op                                                      Description                                                    
    ====================  ===  ===  ===  =========  ==================================================================================================================
      Dropout             X    X    X    Dropout    The Dropout in nnabla has no test mode and contains random parameters, so the test result is not the same as onnx.
      TopKData                                      Not yet implemented.                                                                                              
      TopKGrad                                      Not yet implemented.                                                                                              
      Rand                                          Not yet implemented.                                                                                              
      Randint                                       Not yet implemented.                                                                                              
      Randn                                         Not yet implemented.                                                                                              
      RandomChoice                                  Not yet implemented.                                                                                              
      RandomCrop                                    Not yet implemented.                                                                                              
      RandomFlip                                    Not yet implemented.                                                                                              
      RandomShift                                   Not yet implemented.                                                                                              
      ImageAugmentation                             Not yet implemented.                                                                                              
    ====================  ===  ===  ===  =========  ==================================================================================================================


Loss Functions
^^^^^^^^^^^^^^

Count 0/9
 

    ==========================  ===  ===  ===  =========  ====================
         NNabla Function         6    7    9    ONNX Op       Description     
    ==========================  ===  ===  ===  =========  ====================
      SigmoidCrossEntropy                                 Not yet implemented.
      BinaryCrossEntropy                                  Not yet implemented.
      SoftmaxCrossEntropy                                 Not yet implemented.
      CategoricalCrossEntropy                             Not yet implemented.
      SquaredError                                        Not yet implemented.
      AbsoluteError                                       Not yet implemented.
      HuberLoss                                           Not yet implemented.
      EpsilonInsensitiveLoss                              Not yet implemented.
      KLMultinomial                                       Not yet implemented.
    ==========================  ===  ===  ===  =========  ====================


Quantization Neural Network Layers
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Count 6/11
 

    ===========================  ===  ===  ===  =========================  ====================
          NNabla Function         6    7    9            ONNX Op               Description     
    ===========================  ===  ===  ===  =========================  ====================
      BinarySigmoid              X    X    ✓    Greater, Constant, Where                       
      BinaryTanh                 X    X    ✓    Greater, Constant, Where                       
      BinaryConnectAffine        ✓    ✓    ✓    Gemm, Reshape                                  
      BinaryConnectConvolution   ✓    ✓    ✓    Conv, Reshape                                  
      BinaryWeightAffine         ✓    ✓    ✓    Add, Mul, MatMul, Reshape                      
      BinaryWeightConvolution    ✓    ✓    ✓    Mul, Add, Conv, Reshape                        
      INQAffine                                                            Not yet implemented.
      INQConvolution                                                       Not yet implemented.
      FixedPointQuantize                                                   Not yet implemented.
      Pow2Quantize                                                         Not yet implemented.
      Prune                                                                Not yet implemented.
    ===========================  ===  ===  ===  =========================  ====================


Validation
^^^^^^^^^^

Count 0/3
 

    ==================  ===  ===  ===  =========  ====================
     NNabla Function     6    7    9    ONNX Op       Description     
    ==================  ===  ===  ===  =========  ====================
      TopNError                                   Not yet implemented.
      BinaryError                                 Not yet implemented.
      ConfusionMatrix                             Not yet implemented.
    ==================  ===  ===  ===  =========  ====================


Unsupported, Special Use
^^^^^^^^^^^^^^^^^^^^^^^^

Count 0/5
 

    =====================  ===  ===  ===  =========  ====================
       NNabla Function      6    7    9    ONNX Op       Description     
    =====================  ===  ===  ===  =========  ====================
      VATNoise                                       Not yet implemented.
      Unlink                                         Not yet implemented.
      Sink                                           Not yet implemented.
      NmsDetection2d                                 Not yet implemented.
      MaxPoolingBackward                             Not yet implemented.
    =====================  ===  ===  ===  =========  ====================





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
      AvgPool                                  △      Pad, Transpose, AveragePooling      Some feature is not supported by Nnabla such as Pad's edge mode.                                       
      AvgPool3D                                                                           Not yet implemented.                                                                                   
      BatchMatMul                              ✓      BatchMatmul, Transpose                                                                                                                     
      BiasAdd                                  ✓      Add2, Reshape                                                                                                                              
      Cast                                                                                Not yet implemented.                                                                                   
      Ceil                                     ✓      Ceil                                                                                                                                       
      ConcatV2                                 ✓      Concatenate                                                                                                                                
      Const                                    ✓      Add2                                                                                                                                       
      Conv2D                                   △      Convolution, Transpose, Pad         Tensorflow require GPU to perform related test cases. This issue is recorded only for memo.            
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
      FloorMod                                 ✓      Floor, Div2, Mul2, Sub2                                                                                                                    
      FusedBatchNorm                           △      Transpose, BatchNormalization       It did not pass testing for training mode.                                                             
      GatherNd                                                                            Not yet implemented.                                                                                   
      GatherV2                                                                            Not yet implemented.                                                                                   
      Greater                                  ✓      Greater                                                                                                                                    
      GreaterEqual                             ✓      LogicalNot, Less                                                                                                                           
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
      Softplus                                 ✓      AddScalar, Exp, Log                                                                                                                        
      Softsign                                 ✓      AddScalar, Abs, Div2                                                                                                                       
      SpaceToDepth                             △      Transpose, Reshape                  Tensorflow require GPU to perform related test cases. This issue is recorded only for memo.            
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
      Unpack                                   ✓      Stack, Split, Concatenate, Reshape                                                                                                         
    ======================================  ========  ==================================  =======================================================================================================





Export
------

- ✓: Supported
- △: Partially supported
- X: Supported, but test failed.
- Empty: Not support yet.


Total: 114/172

Neural Network Layer
^^^^^^^^^^^^^^^^^^^^

Count 11/14
 

    =========================  ========  ================================================================================================================================================  ==================================================================================
         NNabla Function        Status                                                                        TF Op                                                                                                           Description                                    
    =========================  ========  ================================================================================================================================================  ==================================================================================
      Affine                   ✓         Mul, Reshape, Const, Add, Placeholder, MatMul                                                                                                                                                                                       
      RNN                                                                                                                                                                                  Not yet implemented.                                                              
      LSTM                                                                                                                                                                                 Not yet implemented.                                                              
      GRU                                                                                                                                                                                  Not yet implemented.                                                              
      Convolution              △         Identity, BatchToSpaceND, Reshape, Const, Pad, Placeholder, Add, SpaceToBatchND, Transpose, ConcatV2, Split, Conv2D                               The cases `dilations` and `strides` larger than 1 are not supported by tensorflow.
      DepthwiseConvolution     △         BatchToSpaceND, Reshape, Const, Pad, Add, Placeholder, SpaceToBatchND, Transpose, ConcatV2, Split, Conv2D                                         The cases `dilations` and `strides` larger than 1 are not supported by tensorflow.
      Deconvolution            △         Identity, Reshape, Const, Add, Placeholder, Conv2DBackpropInput, Slice, Transpose, ConcatV2, Split                                                The cases `dilations` larger than 1 are not supported by tensorflow.              
      DepthwiseDeconvolution   △         Reshape, Const, Add, Placeholder, Conv2DBackpropInput, Slice, Transpose, ConcatV2, Split                                                          The cases `dilations` larger than 1 are not supported by tensorflow.              
      MaxPooling               ✓         MaxPool3D, MaxPool, Reshape, Const, Placeholder, PadV2, Transpose                                                                                                                                                                   
      AveragePooling           △         AvgPool3D, Reshape, AvgPool, Const, Pad, Placeholder, Transpose                                                                                   Currently only supports the cases both ignore_border and including_pad are True.  
      GlobalAveragePooling     ✓         Sub, Range, Const, Pack, SplitV, Mean                                                                                                                                                                                               
      SumPooling               ✓         Mul, AvgPool3D, Reshape, AvgPool, Const, Pad, Placeholder, Transpose                                                                                                                                                                
      Unpooling                △         StridedSlice, Mul, Merge, Identity, Reshape, LogicalAnd, Cast, Equal, ResizeNearestNeighbor, Const, Placeholder, Assert, Switch, Transpose, NoOp  The kernel only supports 2d.                                                      
      Embed                    ✓         Placeholder, GatherV2, Const                                                                                                                                                                                                        
    =========================  ========  ================================================================================================================================================  ==================================================================================


Neural Network Activation Functions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Count 21/21
 

    =================  ========  ====================================================================================  =============
     NNabla Function    Status                                          TF Op                                           Description 
    =================  ========  ====================================================================================  =============
      Sigmoid          ✓         Placeholder, Sigmoid                                                                               
      Swish            ✓         Placeholder, Sigmoid, Mul                                                                          
      Tanh             ✓         Placeholder, Tanh                                                                                  
      ReLU             ✓         Relu, Placeholder                                                                                  
      LeakyReLU        ✓         Placeholder, LeakyRelu                                                                             
      Softmax          ✓         Max, Sub, Exp, Const, Placeholder, Sum, RealDiv                                                    
      LogSoftmax       ✓         Max, Sub, Exp, Log, Const, Placeholder, Sum                                                        
      ELU              ✓         Mul, Sub, Elu, Cast, Exp, GreaterEqual, Less, Const, Placeholder, Add                              
      SELU             ✓         Mul, Maximum, Sub, Minimum, Exp, Const, Placeholder, Add                                           
      CReLU            ✓         Relu, Neg, Const, Placeholder, ConcatV2                                                            
      CELU             ✓         Mul, Sub, Elu, Cast, Neg, Exp, Const, GreaterEqual, Less, Placeholder, ConcatV2, Add               
      PReLU            ✓         Relu, Mul, Sub, Reshape, Const, Placeholder, Abs, Add                                              
      GELU             ✓         Mul, Tanh, Const, Pow, Placeholder, Sqrt, Add, RealDiv                                             
      ReLU6            ✓         Min, Relu, Const, Pack, Placeholder                                                                
      HardSigmoid      ✓         Mul, Maximum, Minimum, Const, Placeholder, Add                                                     
      HardTanh         ✓         Min, Max, Neg, Const, Pack, Placeholder                                                            
      LogSigmoid       ✓         Placeholder, Sigmoid, Log                                                                          
      SoftPlus         ✓         Placeholder, Softplus                                                                              
      SoftSign         ✓         Placeholder, Softsign                                                                              
      TanhShrink       ✓         Placeholder, Sub, Tanh                                                                             
      Sinc             ✓         Select, Equal, Const, Sin, Placeholder, RealDiv                                                    
    =================  ========  ====================================================================================  =============


Normalization
^^^^^^^^^^^^^

Count 1/6
 

    ==========================  ========  ========================================================================================  =======================================================================================================
         NNabla Function         Status                                            TF Op                                                                                          Description                                              
    ==========================  ========  ========================================================================================  =======================================================================================================
      FusedBatchNormalization                                                                                                       Not yet implemented.                                                                                   
      BatchNormalization        △         Mul, StopGradient, Sub, SquaredDifference, Reshape, Rsqrt, Const, Placeholder, Mean, Add  In inferring stage, caffe2 mistmatch onnx 1.4.x's implementation, "in-place" feature cannot be applied.
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
      Min              ✓         Min, Placeholder, Const                       
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
      Pow2             ✓         Placeholder, Pow                                 
      AddScalar        ✓         Placeholder, Add, Const                          
      MulScalar        ✓         Placeholder, Mul, Const                          
      PowScalar        ✓         Placeholder, Const, Pow                          
      RSubScalar       ✓         Placeholder, Sub, Const                          
      RDivScalar       ✓         Placeholder, RealDiv, Const                      
      RPowScalar       ✓         Placeholder, Const, Pow                          
    =================  ========  ===========================  ====================


Logical
^^^^^^^

Count 26/29
 

    =====================  ========  =====================================================  ====================
       NNabla Function      Status                           TF Op                              Description     
    =====================  ========  =====================================================  ====================
      Sign                 ✓         Placeholder, Sign                                                          
      Minimum2             ✓         Min, Const, Pack, Placeholder, Add                                         
      Maximum2             ✓         Max, Const, Pack, Placeholder, Add                                         
      MinimumScalar        ✓         Min, Const, Pack, Placeholder, Add                                         
      MaximumScalar        ✓         Max, Const, Pack, Placeholder, Add                                         
      LogicalAnd           ✓         LogicalAnd, Placeholder                                                    
      LogicalOr            ✓         LogicalOr, Placeholder                                                     
      LogicalXor           ✓         LogicalAnd, LogicalOr, Placeholder, LogicalNot                             
      Equal                ✓         Placeholder, Equal                                                         
      NotEqual             ✓         Placeholder, Equal, LogicalNot                                             
      GreaterEqual         ✓         Placeholder, LogicalNot, Less                                              
      Greater              ✓         Placeholder, Greater                                                       
      LessEqual            ✓         Placeholder, Greater, LogicalNot                                           
      Less                 ✓         Placeholder, Less                                                          
      LogicalAndScalar     ✓         LogicalAnd, Placeholder, Const                                             
      LogicalOrScalar      ✓         LogicalOr, Placeholder, Const                                              
      LogicalXorScalar     ✓         LogicalAnd, LogicalOr, Const, Placeholder, LogicalNot                      
      EqualScalar          ✓         Placeholder, Equal, Const                                                  
      NotEqualScalar       ✓         Placeholder, Equal, Const, LogicalNot                                      
      GreaterEqualScalar   ✓         Placeholder, Const, Less, LogicalNot                                       
      GreaterScalar        ✓         Placeholder, Greater, Const                                                
      LessEqualScalar      ✓         Placeholder, Greater, Const, LogicalNot                                    
      LessScalar           ✓         Placeholder, Const, Less                                                   
      LogicalNot           ✓         Placeholder, LogicalNot                                                    
      IsNaN                ✓         Placeholder, IsNan                                                         
      IsInf                                                                                 Not yet implemented.
      ResetNaN                                                                              Not yet implemented.
      ResetInf                                                                              Not yet implemented.
      Where                ✓         Placeholder, Select                                                        
    =====================  ========  =====================================================  ====================


Math
^^^^

Count 21/22
 

    =================  ========  =====================================================  ====================
     NNabla Function    Status                           TF Op                              Description     
    =================  ========  =====================================================  ====================
      Constant         ✓         Identity, Const                                                            
      Arange           ✓         Identity, Const                                                            
      Abs              ✓         Abs, Placeholder                                                           
      Exp              ✓         Exp, Placeholder                                                           
      Log              ✓         Placeholder, Log                                                           
      Identity         ✓         Placeholder, Identity                                                      
      BatchMatmul      ✓         Reshape, Const, Placeholder, BatchMatMulV2, Transpose                      
      Round                                                                             Not yet implemented.
      Ceil             ✓         Placeholder, Ceil                                                          
      Floor            ✓         Placeholder, Floor                                                         
      Sin              ✓         Placeholder, Sin                                                           
      Cos              ✓         Placeholder, Cos                                                           
      Tan              ✓         Placeholder, Tan                                                           
      Sinh             ✓         Placeholder, Sinh                                                          
      Cosh             ✓         Placeholder, Cosh                                                          
      ASin             ✓         Placeholder, Asin                                                          
      ACos             ✓         Placeholder, Acos                                                          
      ATan             ✓         Placeholder, Atan                                                          
      ATan2            ✓         Placeholder, RealDiv, Atan                                                 
      ASinh            ✓         Asinh, Placeholder                                                         
      ACosh            ✓         Placeholder, Acosh                                                         
      ATanh            ✓         Placeholder, Atanh                                                         
    =================  ========  =====================================================  ====================


Array Manipulation
^^^^^^^^^^^^^^^^^^

Count 12/19
 

    =================  ========  =================================================  ================================================================================================================
     NNabla Function    Status                         TF Op                                                                          Description                                                   
    =================  ========  =================================================  ================================================================================================================
      Concatenate      ✓         Placeholder, ConcatV2, Const                                                                                                                                       
      Split            ✓         Placeholder, Squeeze, Const, SplitV                                                                                                                                
      Stack            ✓         Placeholder, ConcatV2, ExpandDims, Const                                                                                                                           
      Slice            △         Placeholder, Const, Slice                          step != 1" exceed the scope of onnx opset 9,  not supported.                                                    
      Pad              △         MirrorPad, Placeholder, Const, PadV2               When the mode of the pad is reflect, if the size of the pad exceeds the input size, tensorflow cannot handle it.
      Transpose        ✓         Transpose, Placeholder, Const                                                                                                                                      
      Broadcast        ✓                                                                                                                                                                            
      BroadcastTo      ✓                                                                                                                                                                            
      Tile             ✓         Tile, Placeholder, Const, Reshape                                                                                                                                  
      OneHot           ✓         Placeholder, GatherV2, Const, Reshape                                                                                                                              
      Flip             ✓         Identity, GatherV2, Const, Placeholder, Transpose                                                                                                                  
      Shift                                                                         Not yet implemented.                                                                                            
      Sort                                                                          Not yet implemented.                                                                                            
      Reshape          ✓         Placeholder, Const, Reshape                                                                                                                                        
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
      Interpolate                         Not yet implemented.
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

Count 6/11
 

    ===========================  ========  ========================================================================================  ==================================================================================
          NNabla Function         Status                                            TF Op                                                                               Description                                    
    ===========================  ========  ========================================================================================  ==================================================================================
      BinarySigmoid              ✓         Placeholder, Greater, Const, Select                                                                                                                                         
      BinaryTanh                 ✓         Placeholder, Greater, Const, Select                                                                                                                                         
      BinaryConnectAffine        ✓         Mul, Reshape, Const, Add, Placeholder, MatMul                                                                                                                               
      BinaryConnectConvolution   △         Identity, Reshape, Const, Pad, Placeholder, Add, Transpose, ConcatV2, Split, Conv2D       The cases `dilations` and `strides` larger than 1 are not supported by tensorflow.
      BinaryWeightAffine         ✓         Mul, Reshape, Const, Add, Placeholder, MatMul                                                                                                                               
      BinaryWeightConvolution    △         Mul, Identity, Reshape, Const, Pad, Placeholder, Add, Transpose, ConcatV2, Split, Conv2D  The cases `dilations` and `strides` larger than 1 are not supported by tensorflow.
      INQAffine                                                                                                                      Not yet implemented.                                                              
      INQConvolution                                                                                                                 Not yet implemented.                                                              
      FixedPointQuantize                                                                                                             Not yet implemented.                                                              
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




NNabla C Runtime Support Status
===============================


nnabla version: 1.0.21

- ✓: Supported
- △: Partially supported
- X: Supported, but test failed or no test data.
- Empty: Not support yet.


Export
------

Total: 55/172

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

Count 5/22
 

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
      Round            X                      
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

Count 6/11
 

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



