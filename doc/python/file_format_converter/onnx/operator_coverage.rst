ONNX Support Status Document
============================

:Note: In this document, the numbers in the header of all tables represent the version of onnx opset.

Import
------

- √: onnx specification defined, and supported.
- X: onnx specification defined, but not support yet.
- Empty: Not defined (Support status follows latest).

:ONNX Version Info:
  - Version: 1.4.1
  - Commit id: 3b0ecd5

Total: 53/129

.. table:: 

    ==============================  ===  ===  ===  ===  ===  ===  ===  ===  ===  ====  =============
            ONNX Operator            1    2    3    4    5    6    7    8    9    10    Description 
    ==============================  ===  ===  ===  ===  ===  ===  ===  ===  ===  ====  =============
     Abs                            X                        √                                      
     Acos                                                         X                                 
     Acosh                                                                  X                       
     Add                            X                        √    √                                 
     And                            √                             √                                 
     ArgMax                         X                                                               
     ArgMin                         X                                                               
     Asin                                                         X                                 
     Asinh                                                                  X                       
     Atan                                                         X                                 
     Atanh                                                                  X                       
     AveragePool                    √                             √              X                  
     BatchNormalization             X                        √    X         √                       
     Cast                           X                        X              X                       
     Ceil                           X                        X                                      
     Clip                           X                        √                                      
     Compress                                                               X                       
     Concat                         √              √                                                
     Constant                       √                                       √                       
     ConstantOfShape                                                        X                       
     Conv                           √                                                               
     ConvTranspose                  X                                                               
     Cos                                                          X                                 
     Cosh                                                                   X                       
     DepthToSpace                   X                                                               
     Div                            X                        √    √                                 
     Dropout                        √                        √    √              X                  
     Elu                            X                        √                                      
     Equal                          √                             √                                 
     Erf                                                                    X                       
     Exp                            X                        √                                      
     Expand                                                            X                            
     EyeLike                                                                X                       
     Flatten                        X                                       X                       
     Floor                          X                        X                                      
     GRU                            X         X                   X                                 
     Gather                         X                                                               
     Gemm                           √                        √    X         X                       
     GlobalAveragePool              √                                                               
     GlobalLpPool                   X    X                                                          
     GlobalMaxPool                  X                                                               
     Greater                        √                             √         √                       
     HardSigmoid                    X                        X                                      
     Hardmax                        X                                                               
     Identity                       √                                                               
     If                             X                                                               
     InstanceNormalization          X                        X                                      
     IsNaN                                                                  X                       
     LRN                            √                                                               
     LSTM                           X                             X                                 
     LeakyRelu                      X                        √                                      
     Less                           √                             √         √                       
     Log                            X                        √                                      
     LogSoftmax                     √                                                               
     Loop                           X                                                               
     LpNormalization                X                                                               
     LpPool                         X    X                                                          
     MatMul                         √                                       √                       
     Max                            X                        √         √                            
     MaxPool                        √                                  X         X                  
     MaxRoiPool                     X                                                               
     MaxUnpool                                                              X                       
     Mean                           X                        √         √                            
     Min                            X                        √         √                            
     Mul                            X                        √    √                                 
     Multinomial                                                  X                                 
     Neg                            X                        √                                      
     NonZero                                                                X                       
     Not                            √                                                               
     OneHot                                                                 X                       
     Or                             √                             √                                 
     PRelu                          X                        √    X         X                       
     Pad                            X    √                                                          
     Pow                            √                             √                                 
     RNN                            X                             X                                 
     RandomNormal                   X                                                               
     RandomNormalLike               X                                                               
     RandomUniform                  X                                                               
     RandomUniformLike              X                                                               
     Reciprocal                     X                        √                                      
     ReduceL1                       X                                                               
     ReduceL2                       X                                                               
     ReduceLogSum                   X                                                               
     ReduceLogSumExp                X                                                               
     ReduceMax                      √                                                               
     ReduceMean                     √                                                               
     ReduceMin                      √                                                               
     ReduceProd                     √                                                               
     ReduceSum                      √                                                               
     ReduceSumSquare                X                                                               
     Relu                           X                        √                                      
     Reshape                        X                   √                                           
     Resize                                                                      X                  
     Scan                                                              X    X                       
     Scatter                                                                X                       
     Selu                           X                        √                                      
     Shape                          X                                                               
     Shrink                                                                 X                       
     Sigmoid                        X                        √                                      
     Sign                                                                   X                       
     Sin                                                          X                                 
     Sinh                                                                   X                       
     Size                           X                                                               
     Slice                          X                                            X                  
     Softmax                        √                                                               
     Softplus                       √                                                               
     Softsign                       √                                                               
     SpaceToDepth                   X                                                               
     Split                          X    X                                                          
     Sqrt                           X                        X                                      
     Squeeze                        X                                                               
     StringNormalizer                                                            X                  
     Sub                            X                        √    √                                 
     Sum                            X                        √         √                            
     Tan                                                          X                                 
     Tanh                           X                        √                                      
     TfIdfVectorizer                                                        X                       
     ThresholdedRelu                                                             X                  
     Tile                           X                        X                                      
     TopK                           X                                            X                  
     Transpose                      √                                                               
     Unsqueeze                      X                                                               
     Upsample                       X                             X         X    X                  
     Where                                                                  X                       
     Xor                            √                             √                                 
     experimental ATen              X                                                               
     experimental GRUUnit           X                                                               
     experimental GivenTensorFill   X                                                               
     experimental Scale             X                                                               
    ==============================  ===  ===  ===  ===  ===  ===  ===  ===  ===  ====  =============


Export
------

- √: Support to export this opset.
- △: Partially support to export this opset (e.g. some cases cannot be supported, or not completely tested).
- Empty: Not support corresponding opset version.

:NNabla Version Info:
  - Version: 1.0.15.dev1
  - Commit id: 8a603de

Total: 63/155

.. table:: 

    ==========================  ===  ===  ===  ===  ===  ===  ===  ===  ===  ====  ===========================================
         NNabla Functions        1    2    3    4    5    6    7    8    9    10                   Description                
    ==========================  ===  ===  ===  ===  ===  ===  ===  ===  ===  ====  ===========================================
     ACos                                                                                                                     
     ACosh                                                                                                                    
     ASin                                                                                                                     
     ASinh                                                                                                                    
     ATan                                                                                                                     
     ATan2                                                                                                                    
     ATanh                                                                                                                    
     Abs                                                 √              √          By Abs                                     
     AbsoluteError                                                                                                            
     Add2                                                √              √          By Add                                     
     AddScalar                                           △              △          By Add                                     
     Affine                                              △              △          By Reshape,Flatten,Gemm                    
     Arange                                                                                                                   
     AveragePooling                                      √              √          By AveragePool                             
     BatchMatmul                                         √              √          By Matmul                                  
     BatchNormalization                                  √              √          By InstanceNormalization,BatchNormalization
     BcAdd2                                                                                                                   
     BinaryConnectAffine                                                                                                      
     BinaryConnectConvolution                            △              △          By Conv,Reshape                            
     BinaryCrossEntropy                                                                                                       
     BinaryError                                                                                                              
     BinarySigmoid                                       △              △          By HardSigmoid                             
     BinaryTanh                                                                                                               
     BinaryWeightAffine                                                                                                       
     BinaryWeightConvolution                                                                                                  
     Broadcast                                                                                                                
     BroadcastTo                                         √              √                                                     
     CELU                                                                                                                     
     CReLU                                                                                                                    
     CategoricalCrossEntropy                                                                                                  
     Ceil                                                                                                                     
     ClipGradByNorm                                                                                                           
     ClipGradByValue                                                                                                          
     Concatenate                                         √              √          By Concat                                  
     ConfusionMatrix                                                                                                          
     Constant                                                                                                                 
     Convolution                                         √              √          By Conv                                    
     Cos                                                                                                                      
     Cosh                                                                                                                     
     Deconvolution                                       △              △          By ConvTranspose,Add                       
     DepthwiseConvolution                                △              △          By Conv                                    
     DepthwiseDeconvolution                                                                                                   
     Div2                                                √              √          By Div                                     
     Dropout                                             △              △          By Dropout                                 
     ELU                                                 √              √          By ELU                                     
     Embed                                                                                                                    
     EpsilonInsensitiveLoss                                                                                                   
     Equal                                               √              √          By Equal                                   
     EqualScalar                                                                                                              
     Exp                                                 √              √          By Exp                                     
     FFT                                                                                                                      
     FixedPointQuantize                                                                                                       
     Flip                                                △              △          By Gather,Transpose,Identity               
     Floor                                                                                                                    
     GELU                                                                                                                     
     GRU                                                                                                                      
     GlobalAveragePooling                                √              √          By GlobalAveragePool                       
     Greater                                             √              √          By Greater                                 
     GreaterEqual                                                                                                             
     GreaterEqualScalar                                                                                                       
     GreaterScalar                                                                                                            
     HuberLoss                                                                                                                
     IFFT                                                                                                                     
     INQAffine                                                                                                                
     INQConvolution                                                                                                           
     Identity                                            √              √          By Identity                                
     ImageAugmentation                                                                                                        
     Interpolate                                                                                                              
     IsInf                                                                                                                    
     IsNaN                                                                                                                    
     KLMultinomial                                                                                                            
     LSTM                                                                                                                     
     LeakyReLU                                           √              √          By LeakyRelu                               
     Less                                                √              √          By Less                                    
     LessEqual                                                                                                                
     LessEqualScalar                                                                                                          
     LessScalar                                                                                                               
     Log                                                 √              √          By Log                                     
     LogicalAnd                                          √              √          By And                                     
     LogicalAndScalar                                                                                                         
     LogicalNot                                          √              √          By Not                                     
     LogicalOr                                           √              √          By Or                                      
     LogicalOrScalar                                                                                                          
     LogicalXor                                          √              √          By Xor                                     
     LogicalXorScalar                                                                                                         
     MatrixDiag                                                                                                               
     MatrixDiagPart                                                                                                           
     Max                                                 √              √          By ReduceMax                               
     MaxPooling                                          √              √          By MaxPool                                 
     Maximum2                                            √              √          By Max                                     
     MaximumScalar                                       √              √          By Clip                                    
     Mean                                                √              √          By ReduceMean                              
     MeanSubtraction                                                                                                          
     Min                                                 √              √          By ReduceMin                               
     Minimum2                                            √              √          By Min                                     
     MinimumScalar                                       √              √          By Clip                                    
     Mul2                                                √              √          By Mul                                     
     MulScalar                                           √              √          By Mul                                     
     NmsDetection2d                                                                                                           
     NotEqual                                                                                                                 
     NotEqualScalar                                                                                                           
     OneHot                                              △              △          By Flatten,Gather,Reshape                  
     PReLU                                               √              √          By PRelu                                   
     Pad                                                 √              √          By Pad                                     
     Pow2                                                √              √          By Pow                                     
     Pow2Quantize                                                                                                             
     PowScalar                                           √              △          By Pow                                     
     Prod                                                √              √          By ReduceProd                              
     Prune                                                                                                                    
     RDivScalar                                          √              √          By Div                                     
     RNN                                                                                                                      
     RPowScalar                                          △              △          By Pow                                     
     RSubScalar                                          △              △          By Sub                                     
     Rand                                                                                                                     
     Randint                                                                                                                  
     Randn                                                                                                                    
     RandomCrop                                                                                                               
     RandomFlip                                                                                                               
     RandomShift                                                                                                              
     ReLU                                                √              √          By Relu                                    
     ReduceMean                                                                                                               
     ReduceSum                                                                                                                
     ResetInf                                                                                                                 
     ResetNaN                                                                                                                 
     Reshape                                             △              △          By Reshape                                 
     Round                                                                                                                    
     SELU                                                √              √          By SELU                                    
     Shift                                                                                                                    
     Sigmoid                                             √              √          By Sigmoid                                 
     SigmoidCrossEntropy                                                                                                      
     Sign                                                                                                                     
     Sin                                                                                                                      
     Sinh                                                                                                                     
     Sink                                                                                                                     
     Slice                                               △              △          By Slice                                   
     Softmax                                             √              √          By Softmax                                 
     SoftmaxCrossEntropy                                                                                                      
     Sort                                                                                                                     
     Split                                               △              △          By Split,Squeeze                           
     SquaredError                                                                                                             
     Stack                                               △              △          By Unsqueeze,Concat                        
     Sub2                                                √              √          By Sub                                     
     Sum                                                 √              √          By ReduceSum                               
     SumPooling                                          △              △          By Mul                                     
     Swish                                                                                                                    
     Tan                                                                                                                      
     Tanh                                                √              √          By Tanh                                    
     TopKData                                                                                                                 
     TopKGrad                                                                                                                 
     TopNError                                                                                                                
     Transpose                                           √              √          By Transpose                               
     Unlink                                                                                                                   
     Unpooling                                           △              △          By Upsample                                
     VATNoise                                                                                                                 
     Where                                                                                                                    
    ==========================  ===  ===  ===  ===  ===  ===  ===  ===  ===  ====  ===========================================

