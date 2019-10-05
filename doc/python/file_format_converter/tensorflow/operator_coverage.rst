Tensorflow Support Status
=========================


nnabla version: 1.0.21

- ✓: Supported
- △: Partially supported
- X: Supported, but test failed.
- Empty: Not support yet.


Import
------

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
      BiasAdd                                  ✓      Reshape, Add2                                                                                                                              
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
      FloorMod                                 ✓      Sub2, Div2, Floor, Mul2                                                                                                                    
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
      LogicalXor                               ✓      LogicalOr, LogicalAnd, LogicalNot                                                                                                          
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
      Relu6                                    ✓      MaximumScalar, MinimumScalar                                                                                                               
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
      Softplus                                 ✓      Log, Exp, AddScalar                                                                                                                        
      Softsign                                 ✓      Div2, Abs, AddScalar                                                                                                                       
      SpaceToDepth                             △      Reshape, Transpose                  Tensorflow require GPU to perform related test cases. This issue is recorded only for memo.            
      SplitV                                   ✓      Split, Stack                                                                                                                               
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
      Unpack                                   ✓      Reshape, Split, Stack, Concatenate                                                                                                         
    ======================================  ========  ==================================  =======================================================================================================




Export
------

Total: 81/172

Neural Network Layer
^^^^^^^^^^^^^^^^^^^^

Count 10/14
 

    =========================  ========  ================================================================================================================================================  ==================================================================================
         NNabla Function        Status                                                                        TF Op                                                                                                           Description                                    
    =========================  ========  ================================================================================================================================================  ==================================================================================
      Affine                   ✓         Placeholder, Add, Const, Reshape, MatMul, Mul                                                                                                                                                                                       
      RNN                                                                                                                                                                                  Not yet implemented.                                                              
      LSTM                                                                                                                                                                                 Not yet implemented.                                                              
      GRU                                                                                                                                                                                  Not yet implemented.                                                              
      Convolution              △         Identity, Placeholder, SpaceToBatchND, Add, BatchToSpaceND, Const, ConcatV2, Reshape, Split, Pad, Conv2D, Transpose                               The cases `dilations` and `strides` larger than 1 are not supported by tensorflow.
      DepthwiseConvolution     △         Placeholder, SpaceToBatchND, Add, BatchToSpaceND, Const, ConcatV2, Reshape, Split, Pad, Conv2D, Transpose                                         The cases `dilations` and `strides` larger than 1 are not supported by tensorflow.
      Deconvolution            △         Slice, Identity, Conv2DBackpropInput, Placeholder, Add, Const, ConcatV2, Reshape, Split, Transpose                                                The cases `dilations` larger than 1 are not supported by tensorflow.              
      DepthwiseDeconvolution   △         Slice, Conv2DBackpropInput, Placeholder, Add, Const, ConcatV2, Reshape, Split, Transpose                                                          The cases `dilations` larger than 1 are not supported by tensorflow.              
      MaxPooling               ✓         PadV2, MaxPool3D, Placeholder, Const, Reshape, MaxPool, Transpose                                                                                                                                                                   
      AveragePooling           △         AvgPool3D, Placeholder, AvgPool, Const, Reshape, Pad, Transpose                                                                                   Currently only supports the cases both ignore_border and including_pad are True.  
      GlobalAveragePooling     ✓         Mean, SplitV, Sub, Const, Pack, Range                                                                                                                                                                                               
      SumPooling               ✓         AvgPool3D, Placeholder, AvgPool, Const, Reshape, Pad, Mul, Transpose                                                                                                                                                                
      Unpooling                △         StridedSlice, Identity, LogicalAnd, Mul, Placeholder, Equal, NoOp, Cast, ResizeNearestNeighbor, Const, Reshape, Switch, Merge, Assert, Transpose  The kernel only supports 2d.                                                      
      Embed                                                                                                                                                                                Not yet implemented.                                                              
    =========================  ========  ================================================================================================================================================  ==================================================================================


Neural Network Activation Functions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Count 8/21
 

    =================  ========  =====================================================================  ====================
     NNabla Function    Status                                   TF Op                                      Description     
    =================  ========  =====================================================================  ====================
      Sigmoid          ✓         Sigmoid, Placeholder                                                                       
      Swish                                                                                             Not yet implemented.
      Tanh             ✓         Tanh, Placeholder                                                                          
      ReLU             ✓         Relu, Placeholder                                                                          
      LeakyReLU        ✓         LeakyRelu, Placeholder                                                                     
      Softmax          ✓         Placeholder, Sum, Max, Sub, Const, Exp, RealDiv                                            
      LogSoftmax                                                                                        Not yet implemented.
      ELU              ✓         Less, Placeholder, Add, Sub, Const, Elu, Cast, Exp, Mul, GreaterEqual                      
      SELU             ✓         Placeholder, Maximum, Add, Sub, Const, Minimum, Exp, Mul                                   
      CReLU                                                                                             Not yet implemented.
      CELU                                                                                              Not yet implemented.
      PReLU            ✓         Relu, Placeholder, Add, Sub, Const, Abs, Reshape, Mul                                      
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
      BatchNormalization        △         SquaredDifference, Placeholder, Mean, Add, Sub, Const, Rsqrt, Reshape, StopGradient, Mul  In inferring stage, caffe2 mistmatch onnx 1.4.x's implementation, "in-place" feature cannot be applied.
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
      Sum              ✓         Sum, Const, Placeholder                       
      Mean             ✓         Const, Placeholder, Mean                      
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
      Pow2             ✓         Placeholder, Pow                                 
      AddScalar        ✓         Add, Const, Placeholder                          
      MulScalar        ✓         Mul, Const, Placeholder                          
      PowScalar        ✓         Pow, Const, Placeholder                          
      RSubScalar       ✓         Sub, Const, Placeholder                          
      RDivScalar       ✓         RealDiv, Const, Placeholder                      
      RPowScalar       ✓         Pow, Const, Placeholder                          
    =================  ========  ===========================  ====================


Logical
^^^^^^^

Count 12/29
 

    =====================  ========  ==============================================  ====================
       NNabla Function      Status                       TF Op                           Description     
    =====================  ========  ==============================================  ====================
      Sign                 ✓         Sign, Placeholder                                                   
      Minimum2             ✓         Min, Placeholder, Add, Const, Pack                                  
      Maximum2             ✓         Placeholder, Add, Max, Const, Pack                                  
      MinimumScalar        ✓         Min, Placeholder, Add, Const, Pack                                  
      MaximumScalar        ✓         Placeholder, Add, Max, Const, Pack                                  
      LogicalAnd           ✓         LogicalAnd, Placeholder                                             
      LogicalOr            ✓         LogicalOr, Placeholder                                              
      LogicalXor           ✓         LogicalOr, LogicalAnd, LogicalNot, Placeholder                      
      Equal                ✓         Placeholder, Equal                                                  
      NotEqual                                                                       Not yet implemented.
      GreaterEqual                                                                   Not yet implemented.
      Greater              ✓         Greater, Placeholder                                                
      LessEqual                                                                      Not yet implemented.
      Less                 ✓         Less, Placeholder                                                   
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
      Exp              ✓         Exp, Placeholder                                                           
      Log              ✓         Log, Placeholder                                                           
      Identity         ✓         Identity, Placeholder                                                      
      BatchMatmul      ✓         BatchMatMulV2, Placeholder, Const, Reshape, Transpose                      
      Round                                                                             Not yet implemented.
      Ceil             ✓         Ceil, Placeholder                                                          
      Floor            ✓         Floor, Placeholder                                                         
      Sin              ✓         Sin, Placeholder                                                           
      Cos              ✓         Cos, Placeholder                                                           
      Tan              ✓         Tan, Placeholder                                                           
      Sinh             ✓         Sinh, Placeholder                                                          
      Cosh             ✓         Cosh, Placeholder                                                          
      ASin             ✓         Asin, Placeholder                                                          
      ACos             ✓         Acos, Placeholder                                                          
      ATan             ✓         Atan, Placeholder                                                          
      ATan2            ✓         Atan, RealDiv, Placeholder                                                 
      ASinh            ✓         Asinh, Placeholder                                                         
      ACosh            ✓         Acosh, Placeholder                                                         
      ATanh            ✓         Placeholder, Atanh                                                         
    =================  ========  =====================================================  ====================


Array Manipulation
^^^^^^^^^^^^^^^^^^

Count 9/19
 

    =================  ========  =================================================  ================================================================================================================
     NNabla Function    Status                         TF Op                                                                          Description                                                   
    =================  ========  =================================================  ================================================================================================================
      Concatenate      ✓         Const, Placeholder, ConcatV2                                                                                                                                       
      Split            ✓         Squeeze, SplitV, Const, Placeholder                                                                                                                                
      Stack            ✓         ExpandDims, Const, Placeholder, ConcatV2                                                                                                                           
      Slice            △         Slice, Const, Placeholder                          step != 1" exceed the scope of onnx opset 9,  not supported.                                                    
      Pad              △         PadV2, MirrorPad, Const, Placeholder               When the mode of the pad is reflect, if the size of the pad exceeds the input size, tensorflow cannot handle it.
      Transpose        ✓         Const, Placeholder, Transpose                                                                                                                                      
      Broadcast                                                                     Not yet implemented.                                                                                            
      BroadcastTo      ✓                                                                                                                                                                            
      Tile                                                                          Not yet implemented.                                                                                            
      OneHot                                                                        Not yet implemented.                                                                                            
      Flip             ✓         Identity, Placeholder, Const, GatherV2, Transpose                                                                                                                  
      Shift                                                                         Not yet implemented.                                                                                            
      Sort                                                                          Not yet implemented.                                                                                            
      Reshape          ✓         Reshape, Const, Placeholder                                                                                                                                        
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
      BinarySigmoid              ✓         Greater, Const, Select, Placeholder                                                                                                                                         
      BinaryTanh                 ✓         Greater, Const, Select, Placeholder                                                                                                                                         
      BinaryConnectAffine        ✓         Placeholder, Add, Const, Reshape, MatMul, Mul                                                                                                                               
      BinaryConnectConvolution   △         Identity, Placeholder, Add, Const, ConcatV2, Reshape, Split, Pad, Conv2D, Transpose       The cases `dilations` and `strides` larger than 1 are not supported by tensorflow.
      BinaryWeightAffine         ✓         Placeholder, Add, Const, Reshape, MatMul, Mul                                                                                                                               
      BinaryWeightConvolution    △         Identity, Placeholder, Add, Const, ConcatV2, Reshape, Split, Pad, Mul, Conv2D, Transpose  The cases `dilations` and `strides` larger than 1 are not supported by tensorflow.
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



