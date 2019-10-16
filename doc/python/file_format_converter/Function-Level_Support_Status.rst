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
     Add                            ✓                        ✓    ✓              Reshape, Add2                                                                                                                                                                                                                                                               
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
     ConvTranspose                  ✓                        ✓                   Deconvolution, Pad                                                                                                                                                                                                                                                          
     Cos                                                          ✓              Cos                                                                                                                                                                                                                                                                         
     Cosh                                                                   ✓    Cosh                                                                                                                                                                                                                                                                        
     DepthToSpace                   ✓                        ✓                   Transpose, Reshape                                                                                                                                                                                                                                                          
     Div                            ✓                        ✓    ✓              Reshape, Div2                                                                                                                                                                                                                                                               
     Dropout                        X                        X    ✓              Identity                                                                                                                                                                                                                                                                    
     Elu                            ✓                        ✓                   ELU                                                                                                                                                                                                                                                                         
     Equal                          ✓                        ✓    ✓              Reshape, Equal                                                                                                                                                                                                                                                              
     Erf                                                                    X                                                                  Not yet implemented.                                                                                                                                                                                          
     Exp                            ✓                        ✓                   Exp                                                                                                                                                                                                                                                                         
     Expand                                                            X                                                                       Not yet implemented.                                                                                                                                                                                          
     EyeLike                                                                X                                                                  Not yet implemented.                                                                                                                                                                                          
     Flatten                        ✓                        ✓              ✓    Reshape                                                                                                                                                                                                                                                                     
     Floor                          ✓                        ✓                   Floor                                                                                                                                                                                                                                                                       
     GRU                            X         X                   X                                                                            Not yet implemented.                                                                                                                                                                                          
     Gather                         X                                                                                                          Not yet implemented.                                                                                                                                                                                          
     Gemm                           ✓                        ✓    ✓         ✓    MulScalar, Broadcast, BatchMatmul, Add2                                                                                                                                                                                                                                     
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
     LRN                            ✓                        ✓                   AddScalar, MulScalar, Transpose, PowScalar, SumPooling, Div2                                                                                                                                                                                                                
     LSTM                           X                             X                                                                            Not yet implemented.                                                                                                                                                                                          
     LeakyRelu                      ✓                        ✓                   LeakyReLU                                                                                                                                                                                                                                                                   
     Less                           ✓                        ✓    ✓         ✓    Less, Reshape                                                                                                                                                                                                                                                               
     Log                            ✓                        ✓                   Log                                                                                                                                                                                                                                                                         
     LogSoftmax                     ✓                        ✓                   Log, Sum, Reshape, Exp, Add2, Sub2, Max                                                                                                                                                                                                                                     
     Loop                           X                                                                                                          Not yet implemented.                                                                                                                                                                                          
     LpNormalization                X                                                                                                          Not yet implemented.                                                                                                                                                                                          
     LpPool                         X    X                                                                                                     Not yet implemented.                                                                                                                                                                                          
     MatMul                         ✓                        ✓              ✓    BatchMatmul                                                                                                                                                                                                                                                                 
     Max                            ✓                        ✓         ✓    ✓    Maximum2                                                                                                                                                                                                                                                                    
     MaxPool                        ✓                        ✓         X         Pad, MaxPooling                                               Not all features are verified, since some features are not supported by caffe2. Those features can be verified by ONNXRuntime.                                                                                
     MaxRoiPool                     X                                                                                                          Not yet implemented.                                                                                                                                                                                          
     MaxUnpool                                                              X                                                                  Not yet implemented.                                                                                                                                                                                          
     Mean                           ✓                        ✓         ✓    ✓    Stack, Broadcast, Mean                                                                                                                                                                                                                                                      
     Min                            ✓                        ✓         ✓    ✓    Minimum2                                                                                                                                                                                                                                                                    
     Mul                            ✓                        ✓    ✓              Mul2, Reshape                                                                                                                                                                                                                                                               
     Multinomial                                                  X                                                                            Not yet implemented.                                                                                                                                                                                          
     Neg                            ✓                        ✓                   MulScalar                                                                                                                                                                                                                                                                   
     NonZero                                                                X                                                                  Not yet implemented.                                                                                                                                                                                          
     Not                            ✓                        ✓                   LogicalNot                                                                                                                                                                                                                                                                  
     OneHot                                                                 X                                                                  Not yet implemented.                                                                                                                                                                                          
     Or                             ✓                        ✓    ✓              Reshape, LogicalOr                                                                                                                                                                                                                                                          
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
     Softmax                        ✓                        ✓                   Sum, Reshape, Exp, Sub2, Max, Div2                                                                                                                                                                                                                                          
     Softplus                       ✓                        ✓                   Log, AddScalar, Exp                                                                                                                                                                                                                                                         
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
     Xor                            ✓                        ✓    ✓              Reshape, LogicalXor                                                                                                                                                                                                                                                         
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

Total: 81/172

Neural Network Layer
^^^^^^^^^^^^^^^^^^^^

Count 10/14
 

    =========================  ===  ===  ===  ========================================  ======================================================================================
         NNabla Function        6    7    9                   ONNX Op                                                        Description                                      
    =========================  ===  ===  ===  ========================================  ======================================================================================
      Affine                   ✓    ✓    ✓    Gemm, Reshape                                                                                                                   
      RNN                                                                               Not yet implemented.                                                                  
      LSTM                                                                              Not yet implemented.                                                                  
      GRU                                                                               Not yet implemented.                                                                  
      Convolution              ✓    ✓    ✓    Reshape, Conv                                                                                                                   
      DepthwiseConvolution     ✓    ✓    ✓    Reshape, Conv                                                                                                                   
      Deconvolution            △    △    △    ConvTranspose, Reshape                    Caffe2 and onnxruntime do not support dilations != 1.                                 
      DepthwiseDeconvolution   △    △    △    ConvTranspose, Reshape                    Caffe2 and onnxruntime do not support dilations != 1.                                 
      MaxPooling               ✓    ✓    ✓    Pad, Reshape, MaxPool                                                                                                           
      AveragePooling           △    △    △    Pad, AveragePool, Reshape                 Currently only supports the cases where both ignore_border and including_pad are True.
      GlobalAveragePooling     ✓    ✓    ✓    GlobalAveragePool                                                                                                               
      SumPooling               X    ✓    ✓    Pad, Reshape, AveragePool, Constant, Mul                                                                                        
      Unpooling                △    ✓    ✓    Upsample, Reshape                         The kernel only supports 2d on opset 6.                                               
      Embed                                                                             Not yet implemented.                                                                  
    =========================  ===  ===  ===  ========================================  ======================================================================================


Neural Network Activation Functions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Count 8/21
 

    =================  ===  ===  ===  ===================================  =================================================================================
     NNabla Function    6    7    9                 ONNX Op                                                   Description                                   
    =================  ===  ===  ===  ===================================  =================================================================================
      Sigmoid          ✓    ✓    ✓    Sigmoid                                                                                                               
      Swish                                                                Not yet implemented.                                                             
      Tanh             ✓    ✓    ✓    Tanh                                                                                                                  
      ReLU             ✓    ✓    ✓    Relu                                                                                                                  
      LeakyReLU        ✓    ✓    ✓    LeakyRelu                                                                                                             
      Softmax          △    ✓    ✓    Exp, Sub, ReduceMax, ReduceSum, Div  ONNX Add, Sub operator does not support multidirectional broadcasting on opset 6.
      LogSoftmax                                                           Not yet implemented.                                                             
      ELU              ✓    ✓    ✓    Elu                                                                                                                   
      SELU             ✓    ✓    ✓    Selu                                                                                                                  
      CReLU                                                                Not yet implemented.                                                             
      CELU                                                                 Not yet implemented.                                                             
      PReLU            ✓    ✓    ✓    PRelu, Reshape                                                                                                        
      GELU                                                                 Not yet implemented.                                                             
      ReLU6                                                                Not yet implemented.                                                             
      HardSigmoid                                                          Not yet implemented.                                                             
      HardTanh                                                             Not yet implemented.                                                             
      LogSigmoid                                                           Not yet implemented.                                                             
      SoftPlus                                                             Not yet implemented.                                                             
      SoftSign                                                             Not yet implemented.                                                             
      TanhShrink                                                           Not yet implemented.                                                             
      Sinc                                                                 Not yet implemented.                                                             
    =================  ===  ===  ===  ===================================  =================================================================================


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
      AddScalar        ✓    ✓    ✓    Constant, Add                                                                              
      MulScalar        ✓    ✓    ✓    Constant, Mul                                                                              
      PowScalar        ✓    ✓    ✓    Constant, Pow                                                                              
      RSubScalar       ✓    ✓    ✓    Sub                                                                                        
      RDivScalar       ✓    ✓    ✓    Div                                                                                        
      RPowScalar       ✓    ✓    ✓    Pow                                                                                        
    =================  ===  ===  ===  =============  ============================================================================


Logical
^^^^^^^

Count 12/29
 

    =====================  ===  ===  ===  ==================  ============================================================================
       NNabla Function      6    7    9        ONNX Op                                        Description                                 
    =====================  ===  ===  ===  ==================  ============================================================================
      Sign                 X    X    ✓    Sign                                                                                            
      Minimum2             △    ✓    ✓    Add, Min, Constant  ONNX Add operator does not support multidirectional broadcasting on opset 6.
      Maximum2             △    ✓    ✓    Max, Constant, Add  ONNX Add operator does not support multidirectional broadcasting on opset 6.
      MinimumScalar        ✓    ✓    ✓    Add, Min, Constant                                                                              
      MaximumScalar        ✓    ✓    ✓    Max, Constant, Add                                                                              
      LogicalAnd           ✓    ✓    ✓    And                                                                                             
      LogicalOr            ✓    ✓    ✓    Or                                                                                              
      LogicalXor           ✓    ✓    ✓    Xor                                                                                             
      Equal                ✓    ✓    ✓    Equal                                                                                           
      NotEqual                                                Not yet implemented.                                                        
      GreaterEqual                                            Not yet implemented.                                                        
      Greater              ✓    ✓    ✓    Greater                                                                                         
      LessEqual                                               Not yet implemented.                                                        
      Less                 ✓    ✓    ✓    Less                                                                                            
      LogicalAndScalar                                        Not yet implemented.                                                        
      LogicalOrScalar                                         Not yet implemented.                                                        
      LogicalXorScalar                                        Not yet implemented.                                                        
      EqualScalar                                             Not yet implemented.                                                        
      NotEqualScalar                                          Not yet implemented.                                                        
      GreaterEqualScalar                                      Not yet implemented.                                                        
      GreaterScalar                                           Not yet implemented.                                                        
      LessEqualScalar                                         Not yet implemented.                                                        
      LessScalar                                              Not yet implemented.                                                        
      LogicalNot           ✓    ✓    ✓    Not                                                                                             
      IsNaN                                                   Not yet implemented.                                                        
      IsInf                                                   Not yet implemented.                                                        
      ResetNaN                                                Not yet implemented.                                                        
      ResetInf                                                Not yet implemented.                                                        
      Where                                                   Not yet implemented.                                                        
    =====================  ===  ===  ===  ==================  ============================================================================


Math
^^^^

Count 19/22
 

    =================  ===  ===  ===  ==========================  ====================
     NNabla Function    6    7    9            ONNX Op                Description     
    =================  ===  ===  ===  ==========================  ====================
      Constant                                                    Not yet implemented.
      Arange                                                      Not yet implemented.
      Abs              ✓    ✓    ✓    Abs                                             
      Exp              ✓    ✓    ✓    Exp                                             
      Log              ✓    ✓    ✓    Log                                             
      Identity         ✓    ✓    ✓    Identity                                        
      BatchMatmul      ✓    ✓    ✓    MatMul, Transpose, Reshape                      
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
      ATan2            X    ✓    ✓    Atan, Div                                       
      ASinh            X    X    ✓    Asinh                                           
      ACosh            X    X    ✓    Acosh                                           
      ATanh            X    X    ✓    Atanh                                           
    =================  ===  ===  ===  ==========================  ====================


Array Manipulation
^^^^^^^^^^^^^^^^^^

Count 9/19
 

    =================  ===  ===  ===  ===========================  ============================================================================================================================
     NNabla Function    6    7    9             ONNX Op                                                                    Description                                                         
    =================  ===  ===  ===  ===========================  ============================================================================================================================
      Concatenate      ✓    ✓    ✓    Concat                                                                                                                                                   
      Split            ✓    ✓    ✓    Squeeze, Split                                                                                                                                           
      Stack            ✓    ✓    ✓    Concat, Unsqueeze                                                                                                                                        
      Slice            △    △    △    Slice                        ONNX slice cannot support step != 1 on opset < 10.                                                                          
      Pad              △    △    △    Pad                          When the mode of the pad is reflect, if the size of the pad exceeds the input size, caffe2 and onnxruntime cannot handle it.
      Transpose        ✓    ✓    ✓    Transpose                                                                                                                                                
      Broadcast                                                    Not yet implemented.                                                                                                        
      BroadcastTo      ✓    ✓    ✓                                                                                                                                                             
      Tile                                                         Not yet implemented.                                                                                                        
      OneHot                                                       Not yet implemented.                                                                                                        
      Flip             ✓    ✓    ✓    Transpose, Identity, Gather                                                                                                                              
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
      BinaryConnectConvolution   ✓    ✓    ✓    Reshape, Conv                                  
      BinaryWeightAffine         ✓    ✓    ✓    MatMul, Mul, Reshape, Add                      
      BinaryWeightConvolution    ✓    ✓    ✓    Mul, Reshape, Add, Conv                        
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
      BatchMatMul                              ✓      Transpose, BatchMatmul                                                                                                                     
      BiasAdd                                  ✓      Reshape, Add2                                                                                                                              
      Cast                                                                                Not yet implemented.                                                                                   
      Ceil                                     ✓      Ceil                                                                                                                                       
      ConcatV2                                 ✓      Concatenate                                                                                                                                
      Const                                    ✓      Add2                                                                                                                                       
      Conv2D                                   △      Pad, Transpose, Convolution         Tensorflow require GPU to perform related test cases. This issue is recorded only for memo.            
      Conv2DBackpropFilter                                                                Not yet implemented.                                                                                   
      Conv2DBackpropInput                      △      Deconvolution, Transpose            Tensorflow require GPU to perform related test cases. This issue is recorded only for memo.            
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
      FloorMod                                 ✓      Mul2, Sub2, Floor, Div2                                                                                                                    
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
      LessEqual                                ✓      LogicalNot, Greater                                                                                                                        
      Log                                      ✓      Log                                                                                                                                        
      LogSoftmax                                                                          Not yet implemented.                                                                                   
      LogicalAnd                               ✓      LogicalAnd                                                                                                                                 
      LogicalNot                               ✓      LogicalNot                                                                                                                                 
      LogicalOr                                ✓      LogicalOr                                                                                                                                  
      LogicalXor                               ✓      LogicalNot, LogicalAnd, LogicalOr                                                                                                          
      MatrixBandPart                                                                      Not yet implemented.                                                                                   
      Max                                      ✓      Max                                                                                                                                        
      MaxPool                                  ✓      Transpose, MaxPooling, Pad                                                                                                                 
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
      Softplus                                 ✓      Log, AddScalar, Exp                                                                                                                        
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
      Unpack                                   ✓      Stack, Concatenate, Reshape, Split                                                                                                         
    ======================================  ========  ==================================  =======================================================================================================





Export
------

- ✓: Supported
- △: Partially supported
- X: Supported, but test failed.
- Empty: Not support yet.


Total: 81/172

Neural Network Layer
^^^^^^^^^^^^^^^^^^^^

Count 10/14
 

    =========================  ========  ================================================================================================================================================  ==================================================================================
         NNabla Function        Status                                                                        TF Op                                                                                                           Description                                    
    =========================  ========  ================================================================================================================================================  ==================================================================================
      Affine                   ✓         MatMul, Placeholder, Reshape, Add, Const, Mul                                                                                                                                                                                       
      RNN                                                                                                                                                                                  Not yet implemented.                                                              
      LSTM                                                                                                                                                                                 Not yet implemented.                                                              
      GRU                                                                                                                                                                                  Not yet implemented.                                                              
      Convolution              △         BatchToSpaceND, Placeholder, Pad, SpaceToBatchND, Reshape, Split, Add, ConcatV2, Const, Transpose, Identity, Conv2D                               The cases `dilations` and `strides` larger than 1 are not supported by tensorflow.
      DepthwiseConvolution     △         BatchToSpaceND, Placeholder, Pad, SpaceToBatchND, Reshape, Split, Add, ConcatV2, Const, Transpose, Conv2D                                         The cases `dilations` and `strides` larger than 1 are not supported by tensorflow.
      Deconvolution            △         Placeholder, Reshape, Split, Add, ConcatV2, Const, Transpose, Conv2DBackpropInput, Slice, Identity                                                The cases `dilations` larger than 1 are not supported by tensorflow.              
      DepthwiseDeconvolution   △         Placeholder, Reshape, Split, Add, ConcatV2, Const, Transpose, Conv2DBackpropInput, Slice                                                          The cases `dilations` larger than 1 are not supported by tensorflow.              
      MaxPooling               ✓         Placeholder, Reshape, PadV2, MaxPool, Const, Transpose, MaxPool3D                                                                                                                                                                   
      AveragePooling           △         Placeholder, Pad, AvgPool, Reshape, Const, Transpose, AvgPool3D                                                                                   Currently only supports the cases both ignore_border and including_pad are True.  
      GlobalAveragePooling     ✓         Pack, SplitV, Sub, Mean, Range, Const                                                                                                                                                                                               
      SumPooling               ✓         Placeholder, Pad, AvgPool, Reshape, AvgPool3D, Const, Transpose, Mul                                                                                                                                                                
      Unpooling                △         Placeholder, Switch, Equal, Reshape, ResizeNearestNeighbor, Cast, NoOp, Assert, Transpose, Const, Merge, LogicalAnd, StridedSlice, Identity, Mul  The kernel only supports 2d.                                                      
      Embed                                                                                                                                                                                Not yet implemented.                                                              
    =========================  ========  ================================================================================================================================================  ==================================================================================


Neural Network Activation Functions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Count 8/21
 

    =================  ========  =====================================================================  ====================
     NNabla Function    Status                                   TF Op                                      Description     
    =================  ========  =====================================================================  ====================
      Sigmoid          ✓         Placeholder, Sigmoid                                                                       
      Swish                                                                                             Not yet implemented.
      Tanh             ✓         Tanh, Placeholder                                                                          
      ReLU             ✓         Placeholder, Relu                                                                          
      LeakyReLU        ✓         Placeholder, LeakyRelu                                                                     
      Softmax          ✓         Sum, Placeholder, Exp, Sub, Const, RealDiv, Max                                            
      LogSoftmax                                                                                        Not yet implemented.
      ELU              ✓         Placeholder, Add, Sub, Exp, Mul, Elu, Const, GreaterEqual, Less, Cast                      
      SELU             ✓         Placeholder, Add, Sub, Exp, Const, Minimum, Mul, Maximum                                   
      CReLU                                                                                             Not yet implemented.
      CELU                                                                                              Not yet implemented.
      PReLU            ✓         Placeholder, Reshape, Add, Sub, Abs, Const, Mul, Relu                                      
      GELU                                                                                              Not yet implemented.
      ReLU6                                                                                             Not yet implemented.
      HardSigmoid                                                                                       Not yet implemented.
      HardTanh                                                                                          Not yet implemented.
      LogSigmoid                                                                                        Not yet implemented.
      SoftPlus                                                                                          Not yet implemented.
      SoftSign                                                                                          Not yet implemented.
      TanhShrink                                                                                        Not yet implemented.
      Sinc                                                                                              Not yet implemented.
    =================  ========  =====================================================================  ====================


Normalization
^^^^^^^^^^^^^

Count 1/6
 

    ==========================  ========  ========================================================================================  =======================================================================================================
         NNabla Function         Status                                            TF Op                                                                                          Description                                              
    ==========================  ========  ========================================================================================  =======================================================================================================
      FusedBatchNormalization                                                                                                       Not yet implemented.                                                                                   
      BatchNormalization        △         Placeholder, Reshape, Add, Sub, StopGradient, Mean, Rsqrt, Const, Mul, SquaredDifference  In inferring stage, caffe2 mistmatch onnx 1.4.x's implementation, "in-place" feature cannot be applied.
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
      Sum              ✓         Sum, Placeholder, Const                       
      Mean             ✓         Const, Placeholder, Mean                      
      Max              ✓         Const, Placeholder, Max                       
      Min              ✓         Const, Placeholder, Min                       
      Prod             ✓         Const, Placeholder, Prod                      
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
      AddScalar        ✓         Const, Placeholder, Add                          
      MulScalar        ✓         Const, Placeholder, Mul                          
      PowScalar        ✓         Const, Placeholder, Pow                          
      RSubScalar       ✓         Const, Placeholder, Sub                          
      RDivScalar       ✓         Const, Placeholder, RealDiv                      
      RPowScalar       ✓         Const, Placeholder, Pow                          
    =================  ========  ===========================  ====================


Logical
^^^^^^^

Count 12/29
 

    =====================  ========  ==============================================  ====================
       NNabla Function      Status                       TF Op                           Description     
    =====================  ========  ==============================================  ====================
      Sign                 ✓         Placeholder, Sign                                                   
      Minimum2             ✓         Placeholder, Min, Pack, Add, Const                                  
      Maximum2             ✓         Placeholder, Pack, Add, Const, Max                                  
      MinimumScalar        ✓         Placeholder, Min, Pack, Add, Const                                  
      MaximumScalar        ✓         Placeholder, Pack, Add, Const, Max                                  
      LogicalAnd           ✓         Placeholder, LogicalAnd                                             
      LogicalOr            ✓         Placeholder, LogicalOr                                              
      LogicalXor           ✓         LogicalNot, Placeholder, LogicalAnd, LogicalOr                      
      Equal                ✓         Placeholder, Equal                                                  
      NotEqual                                                                       Not yet implemented.
      GreaterEqual                                                                   Not yet implemented.
      Greater              ✓         Greater, Placeholder                                                
      LessEqual                                                                      Not yet implemented.
      Less                 ✓         Placeholder, Less                                                   
      LogicalAndScalar                                                               Not yet implemented.
      LogicalOrScalar                                                                Not yet implemented.
      LogicalXorScalar                                                               Not yet implemented.
      EqualScalar                                                                    Not yet implemented.
      NotEqualScalar                                                                 Not yet implemented.
      GreaterEqualScalar                                                             Not yet implemented.
      GreaterScalar                                                                  Not yet implemented.
      LessEqualScalar                                                                Not yet implemented.
      LessScalar                                                                     Not yet implemented.
      LogicalNot           ✓         LogicalNot, Placeholder                                             
      IsNaN                                                                          Not yet implemented.
      IsInf                                                                          Not yet implemented.
      ResetNaN                                                                       Not yet implemented.
      ResetInf                                                                       Not yet implemented.
      Where                                                                          Not yet implemented.
    =====================  ========  ==============================================  ====================


Math
^^^^

Count 19/22
 

    =================  ========  =====================================================  ====================
     NNabla Function    Status                           TF Op                              Description     
    =================  ========  =====================================================  ====================
      Constant                                                                          Not yet implemented.
      Arange                                                                            Not yet implemented.
      Abs              ✓         Placeholder, Abs                                                           
      Exp              ✓         Placeholder, Exp                                                           
      Log              ✓         Log, Placeholder                                                           
      Identity         ✓         Placeholder, Identity                                                      
      BatchMatmul      ✓         Placeholder, Reshape, Const, Transpose, BatchMatMulV2                      
      Round                                                                             Not yet implemented.
      Ceil             ✓         Ceil, Placeholder                                                          
      Floor            ✓         Placeholder, Floor                                                         
      Sin              ✓         Sin, Placeholder                                                           
      Cos              ✓         Placeholder, Cos                                                           
      Tan              ✓         Placeholder, Tan                                                           
      Sinh             ✓         Placeholder, Sinh                                                          
      Cosh             ✓         Cosh, Placeholder                                                          
      ASin             ✓         Placeholder, Asin                                                          
      ACos             ✓         Placeholder, Acos                                                          
      ATan             ✓         Placeholder, Atan                                                          
      ATan2            ✓         Placeholder, Atan, RealDiv                                                 
      ASinh            ✓         Placeholder, Asinh                                                         
      ACosh            ✓         Placeholder, Acosh                                                         
      ATanh            ✓         Placeholder, Atanh                                                         
    =================  ========  =====================================================  ====================


Array Manipulation
^^^^^^^^^^^^^^^^^^

Count 9/19
 

    =================  ========  =================================================  ================================================================================================================
     NNabla Function    Status                         TF Op                                                                          Description                                                   
    =================  ========  =================================================  ================================================================================================================
      Concatenate      ✓         ConcatV2, Placeholder, Const                                                                                                                                       
      Split            ✓         Const, Placeholder, SplitV, Squeeze                                                                                                                                
      Stack            ✓         ConcatV2, Placeholder, ExpandDims, Const                                                                                                                           
      Slice            △         Const, Placeholder, Slice                          step != 1" exceed the scope of onnx opset 9,  not supported.                                                    
      Pad              △         Const, Placeholder, MirrorPad, PadV2               When the mode of the pad is reflect, if the size of the pad exceeds the input size, tensorflow cannot handle it.
      Transpose        ✓         Const, Placeholder, Transpose                                                                                                                                      
      Broadcast                                                                     Not yet implemented.                                                                                            
      BroadcastTo      ✓                                                                                                                                                                            
      Tile                                                                          Not yet implemented.                                                                                            
      OneHot                                                                        Not yet implemented.                                                                                            
      Flip             ✓         Placeholder, GatherV2, Const, Transpose, Identity                                                                                                                  
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
      BinarySigmoid              ✓         Const, Greater, Placeholder, Select                                                                                                                                         
      BinaryTanh                 ✓         Const, Greater, Placeholder, Select                                                                                                                                         
      BinaryConnectAffine        ✓         MatMul, Placeholder, Reshape, Add, Const, Mul                                                                                                                               
      BinaryConnectConvolution   △         Placeholder, Pad, Reshape, Split, Add, ConcatV2, Const, Transpose, Identity, Conv2D       The cases `dilations` and `strides` larger than 1 are not supported by tensorflow.
      BinaryWeightAffine         ✓         MatMul, Placeholder, Reshape, Add, Const, Mul                                                                                                                               
      BinaryWeightConvolution    △         Placeholder, Pad, Reshape, Split, Add, ConcatV2, Const, Transpose, Identity, Mul, Conv2D  The cases `dilations` and `strides` larger than 1 are not supported by tensorflow.
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

Total: 52/172

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

Count 8/21
 

    =================  ========  =============
     NNabla Function    Status    Description 
    =================  ========  =============
      Sigmoid          ✓                      
      Swish            X                      
      Tanh             ✓                      
      ReLU             ✓                      
      LeakyReLU        ✓                      
      Softmax          ✓                      
      LogSoftmax                              
      ELU              ✓                      
      SELU             ✓                      
      CReLU            X                      
      CELU             X                      
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



