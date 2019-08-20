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

    ======================================  ========  =======================================================================================================
             Tensorflow Function             Status                                                 Description                                              
    ======================================  ========  =======================================================================================================
      Abs                                      ✓                                                                                                             
      Acos                                     ✓                                                                                                             
      Acosh                                    ✓                                                                                                             
      Add                                      ✓                                                                                                             
      AddN                                     ✓                                                                                                             
      All                                                                                                                                                    
      Any                                                                                                                                                    
      ArgMax                                   ✓                                                                                                             
      ArgMin                                   ✓                                                                                                             
      Asin                                     ✓                                                                                                             
      Asinh                                    ✓                                                                                                             
      Atan                                     ✓                                                                                                             
      Atan2                                                                                                                                                  
      Atanh                                    ✓                                                                                                             
      AvgPool                                  △      Some feature is not supported by Nnabla such as Pad's edge mode.                                       
      AvgPool3D                                                                                                                                              
      BatchMatMul                              ✓                                                                                                             
      BiasAdd                                  ✓                                                                                                             
      Cast                                                                                                                                                   
      Ceil                                     ✓                                                                                                             
      ConcatV2                                 ✓                                                                                                             
      Const                                    ✓                                                                                                             
      Conv2D                                   △      Tensorflow require GPU to perform related test cases. This issue is recorded only for memo.            
      Conv2DBackpropFilter                                                                                                                                   
      Conv2DBackpropInput                      △      Tensorflow require GPU to perform related test cases. This issue is recorded only for memo.            
      Conv3D                                                                                                                                                 
      Conv3DBackpropFilterV2                                                                                                                                 
      Conv3DBackpropInputV2                                                                                                                                  
      Cos                                      ✓                                                                                                             
      Cosh                                     ✓                                                                                                             
      DepthToSpace                             △      Tensorflow require GPU to perform related test cases. This issue is recorded only for memo.            
      DepthwiseConv2dNative                                                                                                                                  
      DepthwiseConv2dNativeBackpropFilter                                                                                                                    
      DepthwiseConv2dNativeBackpropInput                                                                                                                     
      Div                                      ✓                                                                                                             
      Elu                                      ✓                                                                                                             
      Equal                                    ✓                                                                                                             
      Erf                                                                                                                                                    
      Erfc                                                                                                                                                   
      Exp                                      ✓                                                                                                             
      ExpandDims                               ✓                                                                                                             
      Fill                                                                                                                                                   
      Flatten                                  ✓                                                                                                             
      Floor                                    ✓                                                                                                             
      FloorDiv                                 ✓                                                                                                             
      FloorMod                                 ✓                                                                                                             
      FusedBatchNorm                           △      It did not pass testing for training mode.                                                             
      GatherNd                                                                                                                                               
      GatherV2                                                                                                                                               
      Greater                                  ✓                                                                                                             
      GreaterEqual                             ✓                                                                                                             
      Identity                                 ✓                                                                                                             
      IsInf                                                                                                                                                  
      IsNan                                    ✓                                                                                                             
      LeakyRelu                                ✓                                                                                                             
      Less                                     ✓                                                                                                             
      LessEqual                                ✓                                                                                                             
      Log                                      ✓                                                                                                             
      LogSoftmax                                                                                                                                             
      LogicalAnd                               ✓                                                                                                             
      LogicalNot                               ✓                                                                                                             
      LogicalOr                                ✓                                                                                                             
      LogicalXor                               ✓                                                                                                             
      MatrixBandPart                                                                                                                                         
      Max                                      ✓                                                                                                             
      MaxPool                                  ✓                                                                                                             
      MaxPool3D                                                                                                                                              
      MaxPoolWithArgmax                                                                                                                                      
      Maximum                                  ✓                                                                                                             
      Mean                                     ✓                                                                                                             
      Min                                      ✓                                                                                                             
      Minimum                                  ✓                                                                                                             
      Mul                                      ✓                                                                                                             
      Neg                                      ✓                                                                                                             
      NotEqual                                 ✓                                                                                                             
      OneHot                                                                                                                                                 
      Pack                                     ✓                                                                                                             
      Pad                                      ✓                                                                                                             
      Pow                                      ✓                                                                                                             
      Prod                                     ✓                                                                                                             
      RandomShuffle                                                                                                                                          
      RandomStandardNormal                                                                                                                                   
      RandomUniform                                                                                                                                          
      RealDiv                                  ✓                                                                                                             
      Reciprocal                               ✓                                                                                                             
      Relu                                     ✓                                                                                                             
      Relu6                                    ✓                                                                                                             
      Reshape                                  △      Some test cases failed for some nnabla's implementation limitation (e.g. -1 is regarded as batch_size).
      ReverseSequence                                                                                                                                        
      Rsqrt                                    ✓                                                                                                             
      Select                                                                                                                                                 
      Selu                                     ✓                                                                                                             
      Shape                                                                                                                                                  
      Sigmoid                                  ✓                                                                                                             
      Sign                                     ✓                                                                                                             
      Sin                                      ✓                                                                                                             
      Sinh                                     ✓                                                                                                             
      Size                                                                                                                                                   
      Slice                                    ✓                                                                                                             
      Softmax                                                                                                                                                
      Softplus                                 ✓                                                                                                             
      Softsign                                 ✓                                                                                                             
      SpaceToDepth                             △      Tensorflow require GPU to perform related test cases. This issue is recorded only for memo.            
      SplitV                                   ✓                                                                                                             
      Sqrt                                     ✓                                                                                                             
      Square                                   ✓                                                                                                             
      SquaredDifference                        ✓                                                                                                             
      Squeeze                                  ✓                                                                                                             
      StopGradient                             ✓                                                                                                             
      StridedSlice                             ✓                                                                                                             
      Sub                                      ✓                                                                                                             
      Sum                                      ✓                                                                                                             
      Tan                                      ✓                                                                                                             
      Tanh                                     ✓                                                                                                             
      Tile                                     ✓                                                                                                             
      TopKV2                                                                                                                                                 
      Transpose                                ✓                                                                                                             
      TruncateDiv                                                                                                                                            
      TruncateMod                                                                                                                                            
      Unpack                                   ✓                                                                                                             
    ======================================  ========  =======================================================================================================




Export
------

Total: 82/172

.. table:: Neural Network Layer

Count 10/14
 

    =========================  ========  ==================================================================================
         NNabla Function        Status                                      Description                                    
    =========================  ========  ==================================================================================
      Affine                   ✓                                                                                           
      RNN                                                                                                                  
      LSTM                                                                                                                 
      GRU                                                                                                                  
      Convolution              △         The cases `dilations` and `strides` larger than 1 are not supported by tensorflow.
      DepthwiseConvolution     △         The cases `dilations` and `strides` larger than 1 are not supported by tensorflow.
      Deconvolution            △         The cases `dilations` larger than 1 are not supported by tensorflow.              
      DepthwiseDeconvolution   △         The cases `dilations` larger than 1 are not supported by tensorflow.              
      MaxPooling               ✓                                                                                           
      AveragePooling           △         Currently only supports the cases both ignore_border and including_pad are True.  
      GlobalAveragePooling     ✓                                                                                           
      SumPooling               ✓                                                                                           
      Unpooling                △         The kernel only supports 2d.                                                      
      Embed                                                                                                                
    =========================  ========  ==================================================================================


.. table:: Neural Network Activation Functions

Count 8/21
 

    =================  ========  =============
     NNabla Function    Status    Description 
    =================  ========  =============
      Sigmoid          ✓                      
      Swish                                   
      Tanh             ✓                      
      ReLU             ✓                      
      LeakyReLU        ✓                      
      Softmax          ✓                      
      LogSoftmax                              
      ELU              ✓                      
      SELU             ✓                      
      CReLU                                   
      CELU                                    
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


.. table:: Normalization

Count 1/6
 

    ==========================  ========  =======================================================================================================
         NNabla Function         Status                                                 Description                                              
    ==========================  ========  =======================================================================================================
      FusedBatchNormalization                                                                                                                    
      BatchNormalization        △         In inferring stage, caffe2 mistmatch onnx 1.4.x's implementation, "in-place" feature cannot be applied.
      SyncBatchNormalization                                                                                                                     
      MeanSubtraction                                                                                                                            
      ClipGradByValue                                                                                                                            
      ClipGradByNorm                                                                                                                             
    ==========================  ========  =======================================================================================================


.. table:: Reduction

Count 5/7
 

    =================  ========  =============
     NNabla Function    Status    Description 
    =================  ========  =============
      Sum              ✓                      
      Mean             ✓                      
      Max              ✓                      
      Min              ✓                      
      Prod             ✓                      
      ReduceSum                               
      ReduceMean                              
    =================  ========  =============


.. table:: Arithmetic

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


.. table:: Logical

Count 12/29
 

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
      NotEqual                                    
      GreaterEqual                                
      Greater              ✓                      
      LessEqual                                   
      Less                 ✓                      
      LogicalAndScalar                            
      LogicalOrScalar                             
      LogicalXorScalar                            
      EqualScalar                                 
      NotEqualScalar                              
      GreaterEqualScalar                          
      GreaterScalar                               
      LessEqualScalar                             
      LessScalar                                  
      LogicalNot           ✓                      
      IsNaN                                       
      IsInf                                       
      ResetNaN                                    
      ResetInf                                    
      Where                                       
    =====================  ========  =============


.. table:: Math

Count 19/22
 

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
      Round                                   
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


.. table:: Array Manipulation

Count 9/19
 

    =================  ========  ================================================================================================================
     NNabla Function    Status                                                     Description                                                   
    =================  ========  ================================================================================================================
      Concatenate      ✓                                                                                                                         
      Split            ✓                                                                                                                         
      Stack            ✓                                                                                                                         
      Slice            △         step != 1" exceed the scope of onnx opset 9,  not supported.                                                    
      Pad              △         When the mode of the pad is reflect, if the size of the pad exceeds the input size, tensorflow cannot handle it.
      Transpose        ✓                                                                                                                         
      Broadcast                                                                                                                                  
      BroadcastTo      ✓                                                                                                                         
      Tile                                                                                                                                       
      OneHot                                                                                                                                     
      Flip             ✓                                                                                                                         
      Shift                                                                                                                                      
      Sort                                                                                                                                       
      Reshape          ✓                                                                                                                         
      MatrixDiag                                                                                                                                 
      MatrixDiagPart                                                                                                                             
      Assign                                                                                                                                     
      GatherNd                                                                                                                                   
      ScatterNd                                                                                                                                  
    =================  ========  ================================================================================================================


.. table:: Signal Processing

Count 0/3
 

    =================  ========  =============
     NNabla Function    Status    Description 
    =================  ========  =============
      Interpolate                             
      FFT                                     
      IFFT                                    
    =================  ========  =============


.. table:: Stochasticity

Count 1/11
 

    ====================  ========  =============
      NNabla Function      Status    Description 
    ====================  ========  =============
      Dropout             ✓                      
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


.. table:: Loss Functions

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


.. table:: Quantization Neural Network Layers

Count 6/11
 

    ===========================  ========  ==================================================================================
          NNabla Function         Status                                      Description                                    
    ===========================  ========  ==================================================================================
      BinarySigmoid              ✓                                                                                           
      BinaryTanh                 ✓                                                                                           
      BinaryConnectAffine        ✓                                                                                           
      BinaryConnectConvolution   △         The cases `dilations` and `strides` larger than 1 are not supported by tensorflow.
      BinaryWeightAffine         ✓                                                                                           
      BinaryWeightConvolution    △         The cases `dilations` and `strides` larger than 1 are not supported by tensorflow.
      INQAffine                                                                                                              
      INQConvolution                                                                                                         
      FixedPointQuantize                                                                                                     
      Pow2Quantize                                                                                                           
      Prune                                                                                                                  
    ===========================  ========  ==================================================================================


.. table:: Validation

Count 0/3
 

    ==================  ========  =============
     NNabla Function     Status    Description 
    ==================  ========  =============
      TopNError                                
      BinaryError                              
      ConfusionMatrix                          
    ==================  ========  =============


.. table:: Unsupported, Special Use

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



