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

    ==============================  ===  ===  ===  ===  ===  ===  ===  ===  ===  ==============================================================================================================================================================================================================
            ONNX Operator            1    2    3    4    5    6    7    8    9                                                                                                    Description                                                                                                  
    ==============================  ===  ===  ===  ===  ===  ===  ===  ===  ===  ==============================================================================================================================================================================================================
     Abs                            ✓                        ✓                                                                                                                                                                                                                                 
     Acos                                                         ✓                                                                                                                                                                                                                            
     Acosh                                                                  ✓                                                                                                                                                                                                                  
     Add                            ✓                        ✓    ✓                                                                                                                                                                                                                            
     And                            ✓                        ✓    ✓                                                                                                                                                                                                                            
     ArgMax                         ✓                        ✓                                                                                                                                                                                                                                 
     ArgMin                         ✓                        ✓                                                                                                                                                                                                                                 
     Asin                                                         ✓                                                                                                                                                                                                                            
     Asinh                                                                  ✓                                                                                                                                                                                                                  
     Atan                                                         ✓                                                                                                                                                                                                                            
     Atanh                                                                  ✓                                                                                                                                                                                                                  
     AveragePool                    ✓                        ✓    ✓              Not all features are verified, since some features are not supported by caffe2. Those features can be verified by ONNXRuntime when opset > 6. Some feature is not supported by Nnabla such as Pad's edge mode.
     BatchNormalization             X                        X    X         ✓                                                                                                                                                                                                                  
     Cast                           X                        X              X                                                                                                                                                                                                                  
     Ceil                           ✓                        ✓                                                                                                                                                                                                                                 
     Clip                           ✓                        ✓                                                                                                                                                                                                                                 
     Compress                                                               X                                                                                                                                                                                                                  
     Concat                         ✓              ✓         ✓                                                                                                                                                                                                                                 
     Constant                       ✓                        ✓              X                                                                                                                                                                                                                  
     ConstantOfShape                                                        X                                                                                                                                                                                                                  
     Conv                           ✓                        ✓                                                                                                                                                                                                                                 
     ConvTranspose                  ✓                        ✓                                                                                                                                                                                                                                 
     Cos                                                          ✓                                                                                                                                                                                                                            
     Cosh                                                                   ✓                                                                                                                                                                                                                  
     DepthToSpace                   ✓                        ✓                                                                                                                                                                                                                                 
     Div                            ✓                        ✓    ✓                                                                                                                                                                                                                            
     Dropout                        X                        X    ✓                                                                                                                                                                                                                            
     Elu                            ✓                        ✓                                                                                                                                                                                                                                 
     Equal                          ✓                        ✓    ✓                                                                                                                                                                                                                            
     Erf                                                                    X                                                                                                                                                                                                                  
     Exp                            ✓                        ✓                                                                                                                                                                                                                                 
     Expand                                                            X                                                                                                                                                                                                                       
     EyeLike                                                                X                                                                                                                                                                                                                  
     Flatten                        ✓                        ✓              ✓                                                                                                                                                                                                                  
     Floor                          ✓                        ✓                                                                                                                                                                                                                                 
     GRU                            X         X                   X                                                                                                                                                                                                                            
     Gather                         X                                                                                                                                                                                                                                                          
     Gemm                           ✓                        ✓    ✓         ✓                                                                                                                                                                                                                  
     GlobalAveragePool              ✓                        ✓                                                                                                                                                                                                                                 
     GlobalLpPool                   X    X                                                                                                                                                                                                                                                     
     GlobalMaxPool                  X                                                                                                                                                                                                                                                          
     Greater                        ✓                        ✓    ✓         ✓                                                                                                                                                                                                                  
     HardSigmoid                    X                        X                                                                                                                                                                                                                                 
     Hardmax                        X                                                                                                                                                                                                                                                          
     Identity                       ✓                        ✓                                                                                                                                                                                                                                 
     If                             X                                                                                                                                                                                                                                                          
     InstanceNormalization          X                        X                                                                                                                                                                                                                                 
     IsNaN                                                                  ✓                                                                                                                                                                                                                  
     LRN                            ✓                        ✓                                                                                                                                                                                                                                 
     LSTM                           X                             X                                                                                                                                                                                                                            
     LeakyRelu                      ✓                        ✓                                                                                                                                                                                                                                 
     Less                           ✓                        ✓    ✓         ✓                                                                                                                                                                                                                  
     Log                            ✓                        ✓                                                                                                                                                                                                                                 
     LogSoftmax                     ✓                        ✓                                                                                                                                                                                                                                 
     Loop                           X                                                                                                                                                                                                                                                          
     LpNormalization                X                                                                                                                                                                                                                                                          
     LpPool                         X    X                                                                                                                                                                                                                                                     
     MatMul                         ✓                        ✓              ✓                                                                                                                                                                                                                  
     Max                            ✓                        ✓         ✓    ✓                                                                                                                                                                                                                  
     MaxPool                        ✓                        ✓         X         Not all features are verified, since some features are not supported by caffe2. Those features can be verified by ONNXRuntime.                                                                                
     MaxRoiPool                     X                                                                                                                                                                                                                                                          
     MaxUnpool                                                              X                                                                                                                                                                                                                  
     Mean                           ✓                        ✓         ✓    ✓                                                                                                                                                                                                                  
     Min                            ✓                        ✓         ✓    ✓                                                                                                                                                                                                                  
     Mul                            ✓                        ✓    ✓                                                                                                                                                                                                                            
     Multinomial                                                  X                                                                                                                                                                                                                            
     Neg                            ✓                        ✓                                                                                                                                                                                                                                 
     NonZero                                                                X                                                                                                                                                                                                                  
     Not                            ✓                        ✓                                                                                                                                                                                                                                 
     OneHot                                                                 X                                                                                                                                                                                                                  
     Or                             ✓                        ✓    ✓                                                                                                                                                                                                                            
     PRelu                          ✓                        ✓    X         X                                                                                                                                                                                                                  
     Pad                            ✓    ✓                   ✓                   Onnx required to support "edge" mode, while nnabla does not support it.                                                                                                                                       
     Pow                            ✓                        ✓    ✓                                                                                                                                                                                                                            
     RNN                            X                             X                                                                                                                                                                                                                            
     RandomNormal                   X                                                                                                                                                                                                                                                          
     RandomNormalLike               X                                                                                                                                                                                                                                                          
     RandomUniform                  X                                                                                                                                                                                                                                                          
     RandomUniformLike              X                                                                                                                                                                                                                                                          
     Reciprocal                     ✓                        ✓                                                                                                                                                                                                                                 
     ReduceL1                       X                                                                                                                                                                                                                                                          
     ReduceL2                       X                                                                                                                                                                                                                                                          
     ReduceLogSum                   X                                                                                                                                                                                                                                                          
     ReduceLogSumExp                X                                                                                                                                                                                                                                                          
     ReduceMax                      ✓                        ✓                                                                                                                                                                                                                                 
     ReduceMean                     ✓                        ✓                                                                                                                                                                                                                                 
     ReduceMin                      ✓                        ✓                                                                                                                                                                                                                                 
     ReduceProd                     ✓                        ✓                                                                                                                                                                                                                                 
     ReduceSum                      ✓                        ✓                                                                                                                                                                                                                                 
     ReduceSumSquare                X                                                                                                                                                                                                                                                          
     Relu                           ✓                        ✓                                                                                                                                                                                                                                 
     Reshape                        ✓                   ✓    ✓                                                                                                                                                                                                                                 
     Resize                                                                                                                                                                                                                                                                                    
     Scan                                                              X    X                                                                                                                                                                                                                  
     Scatter                                                                X                                                                                                                                                                                                                  
     Selu                           ✓                        ✓                                                                                                                                                                                                                                 
     Shape                          X                                                                                                                                                                                                                                                          
     Shrink                                                                 X                                                                                                                                                                                                                  
     Sigmoid                        ✓                        ✓                                                                                                                                                                                                                                 
     Sign                                                                   ✓                                                                                                                                                                                                                  
     Sin                                                          ✓                                                                                                                                                                                                                            
     Sinh                                                                   ✓                                                                                                                                                                                                                  
     Size                           X                                                                                                                                                                                                                                                          
     Slice                          ✓                        ✓                                                                                                                                                                                                                                 
     Softmax                        ✓                        ✓                                                                                                                                                                                                                                 
     Softplus                       ✓                        ✓                                                                                                                                                                                                                                 
     Softsign                       ✓                        ✓                                                                                                                                                                                                                                 
     SpaceToDepth                   ✓                        ✓                                                                                                                                                                                                                                 
     Split                          ✓    ✓                   ✓                                                                                                                                                                                                                                 
     Sqrt                           ✓                        ✓                                                                                                                                                                                                                                 
     Squeeze                        ✓                        ✓                                                                                                                                                                                                                                 
     StringNormalizer                                                                                                                                                                                                                                                                          
     Sub                            ✓                        ✓    ✓                                                                                                                                                                                                                            
     Sum                            ✓                        ✓         ✓    ✓                                                                                                                                                                                                                  
     Tan                                                          ✓                                                                                                                                                                                                                            
     Tanh                           ✓                        ✓                                                                                                                                                                                                                                 
     TfIdfVectorizer                                                        X                                                                                                                                                                                                                  
     ThresholdedRelu                                                                                                                                                                                                                                                                           
     Tile                           ✓                        ✓                                                                                                                                                                                                                                 
     TopK                           X                                                                                                                                                                                                                                                          
     Transpose                      ✓                        ✓                                                                                                                                                                                                                                 
     Unsqueeze                      ✓                        ✓                                                                                                                                                                                                                                 
     Upsample                       ✓                        ✓    ✓         ✓                                                                                                                                                                                                                  
     Where                                                                  X                                                                                                                                                                                                                  
     Xor                            ✓                        ✓    ✓                                                                                                                                                                                                                            
     experimental ATen              X                                                                                                                                                                                                                                                          
     experimental GRUUnit           X                                                                                                                                                                                                                                                          
     experimental GivenTensorFill   X                                                                                                                                                                                                                                                          
     experimental Scale             X                                                                                                                                                                                                                                                          
    ==============================  ===  ===  ===  ===  ===  ===  ===  ===  ===  ==============================================================================================================================================================================================================


Export
------

- ✓: Support to export this opset.
- △: Partially support to export this opset (e.g. some cases cannot be supported, or not completely tested).
- X: Supported, but test failed.
- Empty: Not support corresponding opset version.

:NNabla Version Info:
  - Version: 1.0.21

Total: 82/172

.. table:: Neural Network Layer

Count 10/14
 

    =========================  ===  ===  ===  ======================================================================================
         NNabla Function        6    7    9                                        Description                                      
    =========================  ===  ===  ===  ======================================================================================
      Affine                   ✓    ✓    ✓                                                                                          
      RNN                                                                                                                           
      LSTM                                                                                                                          
      GRU                                                                                                                           
      Convolution              ✓    ✓    ✓                                                                                          
      DepthwiseConvolution     ✓    ✓    ✓                                                                                          
      Deconvolution            △    △    △    Caffe2 and onnxruntime do not support dilations != 1.                                 
      DepthwiseDeconvolution   △    △    △    Caffe2 and onnxruntime do not support dilations != 1.                                 
      MaxPooling               ✓    ✓    ✓                                                                                          
      AveragePooling           △    △    △    Currently only supports the cases where both ignore_border and including_pad are True.
      GlobalAveragePooling     ✓    ✓    ✓                                                                                          
      SumPooling               X    ✓    ✓                                                                                          
      Unpooling                △    ✓    ✓    The kernel only supports 2d on opset 6.                                               
      Embed                                                                                                                         
    =========================  ===  ===  ===  ======================================================================================


.. table:: Neural Network Activation Functions

Count 8/21
 

    =================  ===  ===  ===  =================================================================================
     NNabla Function    6    7    9                                      Description                                   
    =================  ===  ===  ===  =================================================================================
      Sigmoid          ✓    ✓    ✓                                                                                     
      Swish                                                                                                            
      Tanh             ✓    ✓    ✓                                                                                     
      ReLU             ✓    ✓    ✓                                                                                     
      LeakyReLU        ✓    ✓    ✓                                                                                     
      Softmax          △    ✓    ✓    ONNX Add, Sub operator does not support multidirectional broadcasting on opset 6.
      LogSoftmax                                                                                                       
      ELU              ✓    ✓    ✓                                                                                     
      SELU             ✓    ✓    ✓                                                                                     
      CReLU                                                                                                            
      CELU                                                                                                             
      PReLU            ✓    ✓    ✓                                                                                     
      GELU                                                                                                             
      ReLU6                                                                                                            
      HardSigmoid                                                                                                      
      HardTanh                                                                                                         
      LogSigmoid                                                                                                       
      SoftPlus                                                                                                         
      SoftSign                                                                                                         
      TanhShrink                                                                                                       
      Sinc                                                                                                             
    =================  ===  ===  ===  =================================================================================


.. table:: Normalization

Count 1/6
 

    ==========================  ===  ===  ===  =======================================================================================================
         NNabla Function         6    7    9                                                 Description                                              
    ==========================  ===  ===  ===  =======================================================================================================
      FusedBatchNormalization                                                                                                                         
      BatchNormalization        △    △    △    In inferring stage, caffe2 mistmatch onnx 1.4.x's implementation, "in-place" feature cannot be applied.
      SyncBatchNormalization                                                                                                                          
      MeanSubtraction                                                                                                                                 
      ClipGradByValue                                                                                                                                 
      ClipGradByNorm                                                                                                                                  
    ==========================  ===  ===  ===  =======================================================================================================


.. table:: Reduction

Count 5/7
 

    =================  ===  ===  ===  =============
     NNabla Function    6    7    9    Description 
    =================  ===  ===  ===  =============
      Sum              ✓    ✓    ✓                 
      Mean             ✓    ✓    ✓                 
      Max              ✓    ✓    ✓                 
      Min              ✓    ✓    ✓                 
      Prod             ✓    ✓    ✓                 
      ReduceSum                                    
      ReduceMean                                   
    =================  ===  ===  ===  =============


.. table:: Arithmetic

Count 11/12
 

    =================  ===  ===  ===  ============================================================================
     NNabla Function    6    7    9                                   Description                                 
    =================  ===  ===  ===  ============================================================================
      Add2             △    ✓    ✓    ONNX Add operator does not support multidirectional broadcasting on opset 6.
      BcAdd2                                                                                                      
      Sub2             △    ✓    ✓    ONNX Sub operator does not support multidirectional broadcasting on opset 6.
      Mul2             △    ✓    ✓    ONNX Mul operator does not support multidirectional broadcasting on opset 6.
      Div2             △    ✓    ✓    ONNX Div operator does not support multidirectional broadcasting on opset 6.
      Pow2             △    ✓    ✓    ONNX Pow operator does not support multidirectional broadcasting on opset 6.
      AddScalar        ✓    ✓    ✓                                                                                
      MulScalar        ✓    ✓    ✓                                                                                
      PowScalar        ✓    ✓    ✓                                                                                
      RSubScalar       ✓    ✓    ✓                                                                                
      RDivScalar       ✓    ✓    ✓                                                                                
      RPowScalar       ✓    ✓    ✓                                                                                
    =================  ===  ===  ===  ============================================================================


.. table:: Logical

Count 12/29
 

    =====================  ===  ===  ===  ============================================================================
       NNabla Function      6    7    9                                   Description                                 
    =====================  ===  ===  ===  ============================================================================
      Sign                 X    X    ✓                                                                                
      Minimum2             △    ✓    ✓    ONNX Add operator does not support multidirectional broadcasting on opset 6.
      Maximum2             △    ✓    ✓    ONNX Add operator does not support multidirectional broadcasting on opset 6.
      MinimumScalar        ✓    ✓    ✓                                                                                
      MaximumScalar        ✓    ✓    ✓                                                                                
      LogicalAnd           ✓    ✓    ✓                                                                                
      LogicalOr            ✓    ✓    ✓                                                                                
      LogicalXor           ✓    ✓    ✓                                                                                
      Equal                ✓    ✓    ✓                                                                                
      NotEqual                                                                                                        
      GreaterEqual                                                                                                    
      Greater              ✓    ✓    ✓                                                                                
      LessEqual                                                                                                       
      Less                 ✓    ✓    ✓                                                                                
      LogicalAndScalar                                                                                                
      LogicalOrScalar                                                                                                 
      LogicalXorScalar                                                                                                
      EqualScalar                                                                                                     
      NotEqualScalar                                                                                                  
      GreaterEqualScalar                                                                                              
      GreaterScalar                                                                                                   
      LessEqualScalar                                                                                                 
      LessScalar                                                                                                      
      LogicalNot           ✓    ✓    ✓                                                                                
      IsNaN                                                                                                           
      IsInf                                                                                                           
      ResetNaN                                                                                                        
      ResetInf                                                                                                        
      Where                                                                                                           
    =====================  ===  ===  ===  ============================================================================


.. table:: Math

Count 19/22
 

    =================  ===  ===  ===  =============
     NNabla Function    6    7    9    Description 
    =================  ===  ===  ===  =============
      Constant                                     
      Arange                                       
      Abs              ✓    ✓    ✓                 
      Exp              ✓    ✓    ✓                 
      Log              ✓    ✓    ✓                 
      Identity         ✓    ✓    ✓                 
      BatchMatmul      ✓    ✓    ✓                 
      Round                                        
      Ceil             ✓    ✓    ✓                 
      Floor            ✓    ✓    ✓                 
      Sin              X    ✓    ✓                 
      Cos              X    ✓    ✓                 
      Tan              X    ✓    ✓                 
      Sinh             X    X    ✓                 
      Cosh             X    X    ✓                 
      ASin             X    ✓    ✓                 
      ACos             X    ✓    ✓                 
      ATan             X    ✓    ✓                 
      ATan2            X    ✓    ✓                 
      ASinh            X    X    ✓                 
      ACosh            X    X    ✓                 
      ATanh            X    X    ✓                 
    =================  ===  ===  ===  =============


.. table:: Array Manipulation

Count 9/19
 

    =================  ===  ===  ===  ============================================================================================================================
     NNabla Function    6    7    9                                                           Description                                                         
    =================  ===  ===  ===  ============================================================================================================================
      Concatenate      ✓    ✓    ✓                                                                                                                                
      Split            ✓    ✓    ✓                                                                                                                                
      Stack            ✓    ✓    ✓                                                                                                                                
      Slice            △    △    △    ONNX slice cannot support step != 1 on opset < 10.                                                                          
      Pad              △    △    △    When the mode of the pad is reflect, if the size of the pad exceeds the input size, caffe2 and onnxruntime cannot handle it.
      Transpose        ✓    ✓    ✓                                                                                                                                
      Broadcast                                                                                                                                                   
      BroadcastTo      ✓    ✓    ✓                                                                                                                                
      Tile                                                                                                                                                        
      OneHot                                                                                                                                                      
      Flip             ✓    ✓    ✓                                                                                                                                
      Shift                                                                                                                                                       
      Sort                                                                                                                                                        
      Reshape          ✓    ✓    ✓                                                                                                                                
      MatrixDiag                                                                                                                                                  
      MatrixDiagPart                                                                                                                                              
      Assign                                                                                                                                                      
      GatherNd                                                                                                                                                    
      ScatterNd                                                                                                                                                   
    =================  ===  ===  ===  ============================================================================================================================


.. table:: Signal Processing

Count 0/3
 

    =================  ===  ===  ===  =============
     NNabla Function    6    7    9    Description 
    =================  ===  ===  ===  =============
      Interpolate                                  
      FFT                                          
      IFFT                                         
    =================  ===  ===  ===  =============


.. table:: Stochasticity

Count 1/11
 

    ====================  ===  ===  ===  =============
      NNabla Function      6    7    9    Description 
    ====================  ===  ===  ===  =============
      Dropout             ✓    ✓    ✓                 
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
    ====================  ===  ===  ===  =============


.. table:: Loss Functions

Count 0/9
 

    ==========================  ===  ===  ===  =============
         NNabla Function         6    7    9    Description 
    ==========================  ===  ===  ===  =============
      SigmoidCrossEntropy                                   
      BinaryCrossEntropy                                    
      SoftmaxCrossEntropy                                   
      CategoricalCrossEntropy                               
      SquaredError                                          
      AbsoluteError                                         
      HuberLoss                                             
      EpsilonInsensitiveLoss                                
      KLMultinomial                                         
    ==========================  ===  ===  ===  =============


.. table:: Quantization Neural Network Layers

Count 6/11
 

    ===========================  ===  ===  ===  =============
          NNabla Function         6    7    9    Description 
    ===========================  ===  ===  ===  =============
      BinarySigmoid              X    X    ✓                 
      BinaryTanh                 X    X    ✓                 
      BinaryConnectAffine        ✓    ✓    ✓                 
      BinaryConnectConvolution   ✓    ✓    ✓                 
      BinaryWeightAffine         ✓    ✓    ✓                 
      BinaryWeightConvolution    ✓    ✓    ✓                 
      INQAffine                                              
      INQConvolution                                         
      FixedPointQuantize                                     
      Pow2Quantize                                           
      Prune                                                  
    ===========================  ===  ===  ===  =============


.. table:: Validation

Count 0/3
 

    ==================  ===  ===  ===  =============
     NNabla Function     6    7    9    Description 
    ==================  ===  ===  ===  =============
      TopNError                                     
      BinaryError                                   
      ConfusionMatrix                               
    ==================  ===  ===  ===  =============


.. table:: Unsupported, Special Use

Count 0/5
 

    =====================  ===  ===  ===  =============
       NNabla Function      6    7    9    Description 
    =====================  ===  ===  ===  =============
      VATNoise                                         
      Unlink                                           
      Sink                                             
      NmsDetection2d                                   
      MaxPoolingBackward                               
    =====================  ===  ===  ===  =============



