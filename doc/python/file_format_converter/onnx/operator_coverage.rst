ONNX Support Status
===================

:Note: In this document, the numbers in the header of all tables represent the version of onnx opset.

Import
------

- ✓: onnx specification defined, and supported.
- X: onnx specification defined, but not support yet.
- Empty: Not defined (Support status follows latest).

:ONNX Version Info:
  - Version: 1.4.1

Total: 81/129

.. table:: 

    ==============================  ===  ===  ===  ===  ===  ===  ===  ===  ===  ============================================================  ==============================================================================================================================================================================================================
            ONNX Operator            1    2    3    4    5    6    7    8    9                           NNabla Func                                                                                                                            Description                                                                                                  
    ==============================  ===  ===  ===  ===  ===  ===  ===  ===  ===  ============================================================  ==============================================================================================================================================================================================================
     Abs                            ✓                        ✓                   Abs                                                                                                                                                                                                                                                                         
     Acos                                                         ✓              ACos                                                                                                                                                                                                                                                                        
     Acosh                                                                  ✓    ACosh                                                                                                                                                                                                                                                                       
     Add                            ✓                        ✓    ✓              Reshape, Add2                                                                                                                                                                                                                                                               
     And                            ✓                        ✓    ✓              Reshape, LogicalAnd                                                                                                                                                                                                                                                         
     ArgMax                         ✓                        ✓                   Max                                                                                                                                                                                                                                                                         
     ArgMin                         ✓                        ✓                   Min                                                                                                                                                                                                                                                                         
     Asin                                                         ✓              ASin                                                                                                                                                                                                                                                                        
     Asinh                                                                  ✓    ASinh                                                                                                                                                                                                                                                                       
     Atan                                                         ✓              ATan                                                                                                                                                                                                                                                                        
     Atanh                                                                  ✓    ATanh                                                                                                                                                                                                                                                                       
     AveragePool                    ✓                        ✓    ✓              AveragePooling, Pad                                           Not all features are verified, since some features are not supported by caffe2. Those features can be verified by ONNXRuntime when opset > 6. Some feature is not supported by Nnabla such as Pad's edge mode.
     BatchNormalization             X                        X    X         ✓    BatchNormalization                                                                                                                                                                                                                                                          
     Cast                           X                        X              X                                                                  Not yet implemented.                                                                                                                                                                                          
     Ceil                           ✓                        ✓                   Ceil                                                                                                                                                                                                                                                                        
     Clip                           ✓                        ✓                   MaximumScalar, MinimumScalar                                                                                                                                                                                                                                                
     Compress                                                               X                                                                  Not yet implemented.                                                                                                                                                                                          
     Concat                         ✓              ✓         ✓                   Concatenate                                                                                                                                                                                                                                                                 
     Constant                       ✓                        ✓              X    Identity                                                                                                                                                                                                                                                                    
     ConstantOfShape                                                        X                                                                  Not yet implemented.                                                                                                                                                                                          
     Conv                           ✓                        ✓                   Convolution                                                                                                                                                                                                                                                                 
     ConvTranspose                  ✓                        ✓                   Pad, Deconvolution                                                                                                                                                                                                                                                          
     Cos                                                          ✓              Cos                                                                                                                                                                                                                                                                         
     Cosh                                                                   ✓    Cosh                                                                                                                                                                                                                                                                        
     DepthToSpace                   ✓                        ✓                   Reshape, Transpose                                                                                                                                                                                                                                                          
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
     Gemm                           ✓                        ✓    ✓         ✓    Add2, BatchMatmul, MulScalar, Broadcast                                                                                                                                                                                                                                     
     GlobalAveragePool              ✓                        ✓                   GlobalAveragePooling                                                                                                                                                                                                                                                        
     GlobalLpPool                   X    X                                                                                                     Not yet implemented.                                                                                                                                                                                          
     GlobalMaxPool                  X                                                                                                          Not yet implemented.                                                                                                                                                                                          
     Greater                        ✓                        ✓    ✓         ✓    Reshape, Greater                                                                                                                                                                                                                                                            
     HardSigmoid                    X                        X                                                                                 Not yet implemented.                                                                                                                                                                                          
     Hardmax                        X                                                                                                          Not yet implemented.                                                                                                                                                                                          
     Identity                       ✓                        ✓                   Identity                                                                                                                                                                                                                                                                    
     If                             X                                                                                                          Not yet implemented.                                                                                                                                                                                          
     InstanceNormalization          X                        X                                                                                 Not yet implemented.                                                                                                                                                                                          
     IsNaN                                                                  ✓    IsNaN                                                                                                                                                                                                                                                                       
     LRN                            ✓                        ✓                   Div2, AddScalar, SumPooling, MulScalar, PowScalar, Transpose                                                                                                                                                                                                                
     LSTM                           X                             X                                                                            Not yet implemented.                                                                                                                                                                                          
     LeakyRelu                      ✓                        ✓                   LeakyReLU                                                                                                                                                                                                                                                                   
     Less                           ✓                        ✓    ✓         ✓    Reshape, Less                                                                                                                                                                                                                                                               
     Log                            ✓                        ✓                   Log                                                                                                                                                                                                                                                                         
     LogSoftmax                     ✓                        ✓                   Sub2, Add2, Sum, Max, Reshape, Log, Exp                                                                                                                                                                                                                                     
     Loop                           X                                                                                                          Not yet implemented.                                                                                                                                                                                          
     LpNormalization                X                                                                                                          Not yet implemented.                                                                                                                                                                                          
     LpPool                         X    X                                                                                                     Not yet implemented.                                                                                                                                                                                          
     MatMul                         ✓                        ✓              ✓    BatchMatmul                                                                                                                                                                                                                                                                 
     Max                            ✓                        ✓         ✓    ✓    Maximum2                                                                                                                                                                                                                                                                    
     MaxPool                        ✓                        ✓         X         MaxPooling, Pad                                               Not all features are verified, since some features are not supported by caffe2. Those features can be verified by ONNXRuntime.                                                                                
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
     Pow                            ✓                        ✓    ✓              Reshape, Pow2                                                                                                                                                                                                                                                               
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
     Softmax                        ✓                        ✓                   Sub2, Div2, Sum, Max, Reshape, Exp                                                                                                                                                                                                                                          
     Softplus                       ✓                        ✓                   Log, Exp, AddScalar                                                                                                                                                                                                                                                         
     Softsign                       ✓                        ✓                   Div2, Abs, AddScalar                                                                                                                                                                                                                                                        
     SpaceToDepth                   ✓                        ✓                   Reshape, Transpose                                                                                                                                                                                                                                                          
     Split                          ✓    ✓                   ✓                   Split, Stack                                                                                                                                                                                                                                                                
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

:NNabla Version Info:
  - Version: 1.0.21

Total: 81/172

Neural Network Layer
^^^^^^^^^^^^^^^^^^^^

Count 10/14
 

    =========================  ===  ===  ===  ========================================  ======================================================================================
         NNabla Function        6    7    9                   ONNX Op                                                        Description                                      
    =========================  ===  ===  ===  ========================================  ======================================================================================
      Affine                   ✓    ✓    ✓    Reshape, Gemm                                                                                                                   
      RNN                                                                               Not yet implemented.                                                                  
      LSTM                                                                              Not yet implemented.                                                                  
      GRU                                                                               Not yet implemented.                                                                  
      Convolution              ✓    ✓    ✓    Reshape, Conv                                                                                                                   
      DepthwiseConvolution     ✓    ✓    ✓    Reshape, Conv                                                                                                                   
      Deconvolution            △    △    △    Reshape, ConvTranspose                    Caffe2 and onnxruntime do not support dilations != 1.                                 
      DepthwiseDeconvolution   △    △    △    Reshape, ConvTranspose                    Caffe2 and onnxruntime do not support dilations != 1.                                 
      MaxPooling               ✓    ✓    ✓    Reshape, MaxPool, Pad                                                                                                           
      AveragePooling           △    △    △    Reshape, AveragePool, Pad                 Currently only supports the cases where both ignore_border and including_pad are True.
      GlobalAveragePooling     ✓    ✓    ✓    GlobalAveragePool                                                                                                               
      SumPooling               X    ✓    ✓    Reshape, Constant, AveragePool, Pad, Mul                                                                                        
      Unpooling                △    ✓    ✓    Reshape, Upsample                         The kernel only supports 2d on opset 6.                                               
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
      Softmax          △    ✓    ✓    Div, Sub, ReduceMax, ReduceSum, Exp  ONNX Add, Sub operator does not support multidirectional broadcasting on opset 6.
      LogSoftmax                                                           Not yet implemented.                                                             
      ELU              ✓    ✓    ✓    Elu                                                                                                                   
      SELU             ✓    ✓    ✓    Selu                                                                                                                  
      CReLU                                                                Not yet implemented.                                                             
      CELU                                                                 Not yet implemented.                                                             
      PReLU            ✓    ✓    ✓    Reshape, PRelu                                                                                                        
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
      BatchNormalization        △    △    △    Reshape, InstanceNormalization, BatchNormalization  In inferring stage, caffe2 mistmatch onnx 1.4.x's implementation, "in-place" feature cannot be applied.
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

Count 12/29
 

    =====================  ===  ===  ===  ==================  ============================================================================
       NNabla Function      6    7    9        ONNX Op                                        Description                                 
    =====================  ===  ===  ===  ==================  ============================================================================
      Sign                 X    X    ✓    Sign                                                                                            
      Minimum2             △    ✓    ✓    Add, Min, Constant  ONNX Add operator does not support multidirectional broadcasting on opset 6.
      Maximum2             △    ✓    ✓    Add, Max, Constant  ONNX Add operator does not support multidirectional broadcasting on opset 6.
      MinimumScalar        ✓    ✓    ✓    Add, Min, Constant                                                                              
      MaximumScalar        ✓    ✓    ✓    Add, Max, Constant                                                                              
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
      BatchMatmul      ✓    ✓    ✓    Reshape, MatMul, Transpose                      
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
      Flip             ✓    ✓    ✓    Identity, Gather, Transpose                                                                                                                              
      Shift                                                        Not yet implemented.                                                                                                        
      Sort                                                         Not yet implemented.                                                                                                        
      Reshape          ✓    ✓    ✓    Reshape, Constant                                                                                                                                        
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
      BinaryConnectAffine        ✓    ✓    ✓    Reshape, Gemm                                  
      BinaryConnectConvolution   ✓    ✓    ✓    Reshape, Conv                                  
      BinaryWeightAffine         ✓    ✓    ✓    Reshape, Add, MatMul, Mul                      
      BinaryWeightConvolution    ✓    ✓    ✓    Reshape, Conv, Mul, Add                        
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



