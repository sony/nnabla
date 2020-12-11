====================
Model Support Status
====================

.. contents::
   :local:
   :depth: 3

ONNX Support Status
===================

Import
------

- ✓: Support to convert
- X: Not support

Total: 11/12

ONNX Import Sample Test(onnx --> nnp)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Count 11/12
 

    ===================================  =======  ==========================================================
                   Name                  Support                             Memo                           
    ===================================  =======  ==========================================================
    bvlc_alexnet_model_                     ✓                                                               
    bvlc_reference_caffenet_model_          ✓                                                               
    resnet50_model_                         ✓                                                               
    bvlc_reference_rcnn_ilsvrc13_model_     ✓                                                               
    zfnet512_model_                         ✓                                                               
    shufflenet_model_                       ✓                                                               
    bvlc_googlenet_model_                   ✓                                                               
    densenet121_model_                      ✓                                                               
    squeezenet_model_                       ✓                                                               
    vgg19_model_                            ✓                                                               
    inception_v1_model_                     X     The `edge` mode of the `pad` in nnabla is not implemented.
    inception_v2_model_                     ✓                                                               
    ===================================  =======  ==========================================================





Export
------

- ✓: Support to convert
- X: Not support

Total: 59/65

ONNX Export Sample Test(nnp --> onnx)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Count 34/37
 

    ================================  =======  =======================================================
                  Name                Support                           Memo                          
    ================================  =======  =======================================================
    stacked_GRU_27_                      ✓                                                            
    LeNet_35_                            ✓                                                            
    01_logistic_regression_9_            ✓                                                            
    LSTM_auto_encoder_24_                ✓                                                            
    mnist_vae_3_                         ✓                                                            
    10_deep_mlp_13_                      ✓                                                            
    11_deconvolution_12_                 ✓                                                            
    01_logistic_regression_10_           ✓                                                            
    mnist_dcgan_with_label_1_            X     NNabla converter error, will be fixed in the future.   
    semi_supervised_learning_VAT_37_     X     NNP with only a single executor is currently supported.
    binary_net_mnist_LeNet_7_            ✓                                                            
    LSTM_auto_encoder_23_                ✓                                                            
    long_short_term_memoryLSTM_29_       ✓                                                            
    binary_connect_mnist_LeNet_5_        ✓                                                            
    long_short_term_memoryLSTM_30_       ✓                                                            
    binary_connect_mnist_MLP_8_          ✓                                                            
    elman_net_22_                        ✓                                                            
    02_binary_cnn_15_                    ✓                                                            
    gated_recurrent_unitGRU_31_          ✓                                                            
    bidirectional_elman_net_25_          ✓                                                            
    06_auto_encoder_18_                  ✓                                                            
    02_binary_cnn_16_                    ✓                                                            
    stacked_GRU_28_                      ✓                                                            
    elman_net_21_                        ✓                                                            
    LeNet_36_                            ✓                                                            
    binary_weight_mnist_MLP_6_           ✓                                                            
    06_auto_encoder_17_                  ✓                                                            
    10_deep_mlp_14_                      ✓                                                            
    binary_net_mnist_MLP_4_              ✓                                                            
    elman_net_with_attention_34_         ✓                                                            
    elman_net_with_attention_33_         ✓                                                            
    mnist_dcgan_with_label_2_            X     NNabla converter error, will be fixed in the future.   
    gated_recurrent_unitGRU_32_          ✓                                                            
    12_residual_learning_19_             ✓                                                            
    12_residual_learning_20_             ✓                                                            
    bidirectional_elman_net_26_          ✓                                                            
    11_deconvolution_11_                 ✓                                                            
    ================================  =======  =======================================================


ONNX Export Pretrained Model Test(nnp --> onnx)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Count 18/18
 

    ======================  =======  ====
             Name           Support  Memo
    ======================  =======  ====
    Resnet-50_4_178_           ✓         
    Resnet-34_4_128_           ✓         
    Resnet-101_4_348_          ✓         
    DenseNet-161_2_570_        ✓         
    ShuffleNet-0.5x_2_202_     ✓         
    GoogLeNet_4_142_           ✓         
    NIN_                       ✓         
    Resnet-18_3_71_            ✓         
    Xception_                  ✓         
    AlexNet_                   ✓         
    ShuffleNet_2_202_          ✓         
    MobileNet_1_86_            ✓         
    SqueezeNet-1.0_2_70_       ✓         
    VGG-13_                    ✓         
    Resnet-152_4_518_          ✓         
    VGG-11_                    ✓         
    SqueezeNet-1.1_2_70_       ✓         
    VGG-16_                    ✓         
    ======================  =======  ====


ONNX Export Example Model Test(nnp --> onnx)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Count 7/10
 

    ==================  =======  ====================================================
           Name         Support                          Memo                        
    ==================  =======  ====================================================
    word_embedding_        ✓                                                         
    classification_        ✓                                                         
    wavenet_               X     The `onehot` dimension != 2 is not supported.       
    capsules_              X     NNabla converter error, will be fixed in the future.
    cycle_gan_             ✓                                                         
    siamese_embedding_     ✓                                                         
    meta_learning_         X     Failed to compare inferring result.                 
    yolov2_                ✓                                                         
    pix2pix_               ✓                                                         
    deeplabv3plus_         ✓                                                         
    ==================  =======  ====================================================





Tensorflow Support Status
=========================

Import
------

- ✓: Support to convert
- X: Not support

Total: 9/10

Tensorflow Import Sample Test(pb --> nnp)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Count 9/10
 

    ======================================  =======  ===============================================================================
                     Name                   Support                                       Memo                                      
    ======================================  =======  ===============================================================================
    mobilenet_v1_1.0_224_                      ✓                                                                                    
    inception_resnet_v2_2016_08_30_frozen_     ✓                                                                                    
    inception_v4_2016_09_09_frozen_            ✓                                                                                    
    conv-layers_frozen_                        ✓     Failed to convert from tensorflow to onnx, `Bias` should be 1D, but actual n-D.
    lstm_frozen_                               X     The `Shape` is currently not supported to convert by nnabla.                   
    mobilenet_v1_0.75_192_                     ✓                                                                                    
    inception_v1_2016_08_28_frozen_            ✓                                                                                    
    inception_v3_2016_08_28_frozen_            ✓                                                                                    
    fc-layers_frozen_                          ✓                                                                                    
    ae0_frozen_                                ✓                                                                                    
    ======================================  =======  ===============================================================================





Export
------

- ✓: Support to convert
- X: Not support

Total: 81/130

Tensorflow Export Sample Test(Disable optimization)(nnp --> pb)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Count 34/37
 

    ================================  =======  =======================================================
                  Name                Support                           Memo                          
    ================================  =======  =======================================================
    stacked_GRU_27_                      ✓                                                            
    LeNet_35_                            ✓                                                            
    01_logistic_regression_9_            ✓                                                            
    LSTM_auto_encoder_24_                ✓                                                            
    mnist_vae_3_                         ✓                                                            
    10_deep_mlp_13_                      ✓                                                            
    11_deconvolution_12_                 ✓                                                            
    01_logistic_regression_10_           ✓                                                            
    mnist_dcgan_with_label_1_            X     NNabla converter error, will be fixed in the future.   
    semi_supervised_learning_VAT_37_     X     NNP with only a single executor is currently supported.
    binary_net_mnist_LeNet_7_            ✓                                                            
    LSTM_auto_encoder_23_                ✓                                                            
    long_short_term_memoryLSTM_29_       ✓                                                            
    binary_connect_mnist_LeNet_5_        ✓                                                            
    long_short_term_memoryLSTM_30_       ✓                                                            
    binary_connect_mnist_MLP_8_          ✓                                                            
    elman_net_22_                        ✓                                                            
    02_binary_cnn_15_                    ✓                                                            
    gated_recurrent_unitGRU_31_          ✓                                                            
    bidirectional_elman_net_25_          ✓                                                            
    06_auto_encoder_18_                  ✓                                                            
    02_binary_cnn_16_                    ✓                                                            
    stacked_GRU_28_                      ✓                                                            
    elman_net_21_                        ✓                                                            
    LeNet_36_                            ✓                                                            
    binary_weight_mnist_MLP_6_           ✓                                                            
    06_auto_encoder_17_                  ✓                                                            
    10_deep_mlp_14_                      ✓                                                            
    binary_net_mnist_MLP_4_              ✓                                                            
    elman_net_with_attention_34_         ✓                                                            
    elman_net_with_attention_33_         ✓                                                            
    mnist_dcgan_with_label_2_            X     NNabla converter error, will be fixed in the future.   
    gated_recurrent_unitGRU_32_          ✓                                                            
    12_residual_learning_19_             ✓                                                            
    12_residual_learning_20_             ✓                                                            
    bidirectional_elman_net_26_          ✓                                                            
    11_deconvolution_11_                 ✓                                                            
    ================================  =======  =======================================================


Tensorflow Export Pretrained Models(Disable optimization)(nnp --> pb)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Count 18/18
 

    ======================  =======  ====
             Name           Support  Memo
    ======================  =======  ====
    Resnet-50_4_178_           ✓         
    Resnet-34_4_128_           ✓         
    Resnet-101_4_348_          ✓         
    DenseNet-161_2_570_        ✓         
    ShuffleNet-0.5x_2_202_     ✓         
    GoogLeNet_4_142_           ✓         
    NIN_                       ✓         
    Resnet-18_3_71_            ✓         
    Xception_                  ✓         
    AlexNet_                   ✓         
    ShuffleNet_2_202_          ✓         
    MobileNet_1_86_            ✓         
    SqueezeNet-1.0_2_70_       ✓         
    VGG-13_                    ✓         
    Resnet-152_4_518_          ✓         
    VGG-11_                    ✓         
    SqueezeNet-1.1_2_70_       ✓         
    VGG-16_                    ✓         
    ======================  =======  ====


Tensorflow Export Example Models(Disable optimization)(nnp --> pb)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Count 7/10
 

    ==================  =======  =============================================
           Name         Support                      Memo                     
    ==================  =======  =============================================
    word_embedding_        ✓                                                  
    classification_        ✓                                                  
    wavenet_               X     The `onehot` dimension != 2 is not supported.
    capsules_              ✓                                                  
    cycle_gan_             ✓                                                  
    siamese_embedding_     ✓                                                  
    meta_learning_         X     Failed to compare inferring result.          
    yolov2_                ✓                                                  
    pix2pix_               ✓                                                  
    deeplabv3plus_         X     The `Interpolate` is currently not supported.
    ==================  =======  =============================================


Tensorflow Export Sample Test(Enable optimization)(nnp --> pb)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Count 20/37
 

    ================================  =======  ====
                  Name                Support  Memo
    ================================  =======  ====
    stacked_GRU_27_                      ✓         
    LeNet_35_                            X         
    01_logistic_regression_9_            ✓         
    LSTM_auto_encoder_24_                X         
    mnist_vae_3_                         ✓         
    10_deep_mlp_13_                      ✓         
    11_deconvolution_12_                 X         
    01_logistic_regression_10_           ✓         
    mnist_dcgan_with_label_1_            X         
    semi_supervised_learning_VAT_37_     X         
    binary_net_mnist_LeNet_7_            X         
    LSTM_auto_encoder_23_                X         
    long_short_term_memoryLSTM_29_       ✓         
    binary_connect_mnist_LeNet_5_        X         
    long_short_term_memoryLSTM_30_       ✓         
    binary_connect_mnist_MLP_8_          ✓         
    elman_net_22_                        ✓         
    02_binary_cnn_15_                    X         
    gated_recurrent_unitGRU_31_          ✓         
    bidirectional_elman_net_25_          ✓         
    06_auto_encoder_18_                  ✓         
    02_binary_cnn_16_                    X         
    stacked_GRU_28_                      ✓         
    elman_net_21_                        ✓         
    LeNet_36_                            X         
    binary_weight_mnist_MLP_6_           ✓         
    06_auto_encoder_17_                  ✓         
    10_deep_mlp_14_                      ✓         
    binary_net_mnist_MLP_4_              ✓         
    elman_net_with_attention_34_         X         
    elman_net_with_attention_33_         X         
    mnist_dcgan_with_label_2_            X         
    gated_recurrent_unitGRU_32_          ✓         
    12_residual_learning_19_             X         
    12_residual_learning_20_             X         
    bidirectional_elman_net_26_          ✓         
    11_deconvolution_11_                 X         
    ================================  =======  ====


Tensorflow Export Pretrained Models(Enable optimization)(nnp --> pb)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Count 0/18
 

    ======================  =======  ====
             Name           Support  Memo
    ======================  =======  ====
    Resnet-50_4_178_           X         
    Resnet-34_4_128_           X         
    Resnet-101_4_348_          X         
    DenseNet-161_2_570_        X         
    ShuffleNet-0.5x_2_202_     X         
    GoogLeNet_4_142_           X         
    NIN_                       X         
    Resnet-18_3_71_            X         
    Xception_                  X         
    AlexNet_                   X         
    ShuffleNet_2_202_          X         
    MobileNet_1_86_            X         
    SqueezeNet-1.0_2_70_       X         
    VGG-13_                    X         
    Resnet-152_4_518_          X         
    VGG-11_                    X         
    SqueezeNet-1.1_2_70_       X         
    VGG-16_                    X         
    ======================  =======  ====


Tensorflow Export Example Models(Enable optimization)(nnp --> pb)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Count 2/10
 

    ==================  =======  ====
           Name         Support  Memo
    ==================  =======  ====
    word_embedding_        ✓         
    classification_        X         
    wavenet_               X         
    capsules_              ✓         
    cycle_gan_             X         
    siamese_embedding_     X         
    meta_learning_         X         
    yolov2_                X         
    pix2pix_               X         
    deeplabv3plus_         X         
    ==================  =======  ====




Tensorflow Lite Support Status
==============================


Export
------

- ✓: Support to convert
- X: Not support

Total: 18/130

Tensorflow Lite Export Sample Test(Disable optimization)(nnp --> tflite)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Count 9/37
 

    ================================  =======  ====
                  Name                Support  Memo
    ================================  =======  ====
    stacked_GRU_27_                      X         
    LeNet_35_                            ✓         
    01_logistic_regression_9_            ✓         
    LSTM_auto_encoder_24_                X         
    mnist_vae_3_                         X         
    10_deep_mlp_13_                      ✓         
    11_deconvolution_12_                 X         
    01_logistic_regression_10_           ✓         
    mnist_dcgan_with_label_1_            X         
    semi_supervised_learning_VAT_37_     X         
    binary_net_mnist_LeNet_7_            X         
    LSTM_auto_encoder_23_                X         
    long_short_term_memoryLSTM_29_       X         
    binary_connect_mnist_LeNet_5_        X         
    long_short_term_memoryLSTM_30_       X         
    binary_connect_mnist_MLP_8_          X         
    elman_net_22_                        X         
    02_binary_cnn_15_                    ✓         
    gated_recurrent_unitGRU_31_          X         
    bidirectional_elman_net_25_          X         
    06_auto_encoder_18_                  X         
    02_binary_cnn_16_                    ✓         
    stacked_GRU_28_                      X         
    elman_net_21_                        X         
    LeNet_36_                            ✓         
    binary_weight_mnist_MLP_6_           X         
    06_auto_encoder_17_                  X         
    10_deep_mlp_14_                      ✓         
    binary_net_mnist_MLP_4_              X         
    elman_net_with_attention_34_         X         
    elman_net_with_attention_33_         X         
    mnist_dcgan_with_label_2_            X         
    gated_recurrent_unitGRU_32_          X         
    12_residual_learning_19_             X         
    12_residual_learning_20_             ✓         
    bidirectional_elman_net_26_          X         
    11_deconvolution_11_                 X         
    ================================  =======  ====


Tensorflow Lite Export Pretrained Models(Disable optimization)(nnp --> tflite)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Count 2/18
 

    ======================  =======  ====
             Name           Support  Memo
    ======================  =======  ====
    Resnet-50_4_178_           X         
    Resnet-34_4_128_           X         
    Resnet-101_4_348_          X         
    DenseNet-161_2_570_        X         
    ShuffleNet-0.5x_2_202_     X         
    GoogLeNet_4_142_           ✓         
    NIN_                       X         
    Resnet-18_3_71_            X         
    Xception_                  X         
    AlexNet_                   X         
    ShuffleNet_2_202_          X         
    MobileNet_1_86_            ✓         
    SqueezeNet-1.0_2_70_       X         
    VGG-13_                    X         
    Resnet-152_4_518_          X         
    VGG-11_                    X         
    SqueezeNet-1.1_2_70_       X         
    VGG-16_                    X         
    ======================  =======  ====


Tensorflow Lite Export Example Models(Disable optimization)(nnp --> tflite)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Count 2/10
 

    ==================  =======  ====
           Name         Support  Memo
    ==================  =======  ====
    word_embedding_        ✓         
    classification_        X         
    wavenet_               X         
    capsules_              X         
    cycle_gan_             X         
    siamese_embedding_     ✓         
    meta_learning_         X         
    yolov2_                X         
    pix2pix_               X         
    deeplabv3plus_         X         
    ==================  =======  ====


Tensorflow Lite Export Sample Test(Enable optimization)(nnp --> tflite)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Count 4/37
 

    ================================  =======  ====
                  Name                Support  Memo
    ================================  =======  ====
    stacked_GRU_27_                      X         
    LeNet_35_                            X         
    01_logistic_regression_9_            ✓         
    LSTM_auto_encoder_24_                X         
    mnist_vae_3_                         X         
    10_deep_mlp_13_                      ✓         
    11_deconvolution_12_                 X         
    01_logistic_regression_10_           ✓         
    mnist_dcgan_with_label_1_            X         
    semi_supervised_learning_VAT_37_     X         
    binary_net_mnist_LeNet_7_            X         
    LSTM_auto_encoder_23_                X         
    long_short_term_memoryLSTM_29_       X         
    binary_connect_mnist_LeNet_5_        X         
    long_short_term_memoryLSTM_30_       X         
    binary_connect_mnist_MLP_8_          X         
    elman_net_22_                        X         
    02_binary_cnn_15_                    X         
    gated_recurrent_unitGRU_31_          X         
    bidirectional_elman_net_25_          X         
    06_auto_encoder_18_                  X         
    02_binary_cnn_16_                    X         
    stacked_GRU_28_                      X         
    elman_net_21_                        X         
    LeNet_36_                            X         
    binary_weight_mnist_MLP_6_           X         
    06_auto_encoder_17_                  X         
    10_deep_mlp_14_                      ✓         
    binary_net_mnist_MLP_4_              X         
    elman_net_with_attention_34_         X         
    elman_net_with_attention_33_         X         
    mnist_dcgan_with_label_2_            X         
    gated_recurrent_unitGRU_32_          X         
    12_residual_learning_19_             X         
    12_residual_learning_20_             X         
    bidirectional_elman_net_26_          X         
    11_deconvolution_11_                 X         
    ================================  =======  ====


Tensorflow Lite Export Pretrained Models(Enable optimization)(nnp --> tflite)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Count 0/18
 

    ======================  =======  ====
             Name           Support  Memo
    ======================  =======  ====
    Resnet-50_4_178_           X         
    Resnet-34_4_128_           X         
    Resnet-101_4_348_          X         
    DenseNet-161_2_570_        X         
    ShuffleNet-0.5x_2_202_     X         
    GoogLeNet_4_142_           X         
    NIN_                       X         
    Resnet-18_3_71_            X         
    Xception_                  X         
    AlexNet_                   X         
    ShuffleNet_2_202_          X         
    MobileNet_1_86_            X         
    SqueezeNet-1.0_2_70_       X         
    VGG-13_                    X         
    Resnet-152_4_518_          X         
    VGG-11_                    X         
    SqueezeNet-1.1_2_70_       X         
    VGG-16_                    X         
    ======================  =======  ====


Tensorflow Lite Export Example Models(Enable optimization)(nnp --> tflite)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Count 1/10
 

    ==================  =======  ====
           Name         Support  Memo
    ==================  =======  ====
    word_embedding_        ✓         
    classification_        X         
    wavenet_               X         
    capsules_              X         
    cycle_gan_             X         
    siamese_embedding_     X         
    meta_learning_         X         
    yolov2_                X         
    pix2pix_               X         
    deeplabv3plus_         X         
    ==================  =======  ====




NNabla C Runtime Support Status
===============================

Export
------

- ✓: Support to convert
- X: Not support

Total: 34/37

NNC Export Sample Test(nnp --> nnb)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Count 34/37
 

    ================================  =======  ===================================
                  Name                Support                 Memo                
    ================================  =======  ===================================
    stacked_GRU_27_                      ✓                                        
    LeNet_35_                            ✓                                        
    01_logistic_regression_9_            ✓                                        
    LSTM_auto_encoder_24_                ✓                                        
    mnist_vae_3_                         ✓                                        
    10_deep_mlp_13_                      ✓                                        
    11_deconvolution_12_                 ✓                                        
    01_logistic_regression_10_           ✓                                        
    mnist_dcgan_with_label_1_            X     Failed to infer by nnabla.         
    semi_supervised_learning_VAT_37_     X     Failed to compare inferring result.
    binary_net_mnist_LeNet_7_            ✓                                        
    LSTM_auto_encoder_23_                ✓                                        
    long_short_term_memoryLSTM_29_       ✓                                        
    binary_connect_mnist_LeNet_5_        ✓                                        
    long_short_term_memoryLSTM_30_       ✓                                        
    binary_connect_mnist_MLP_8_          ✓                                        
    elman_net_22_                        ✓                                        
    02_binary_cnn_15_                    ✓                                        
    gated_recurrent_unitGRU_31_          ✓                                        
    bidirectional_elman_net_25_          ✓                                        
    06_auto_encoder_18_                  ✓                                        
    02_binary_cnn_16_                    ✓                                        
    stacked_GRU_28_                      ✓                                        
    elman_net_21_                        ✓                                        
    LeNet_36_                            ✓                                        
    binary_weight_mnist_MLP_6_           ✓                                        
    06_auto_encoder_17_                  ✓                                        
    10_deep_mlp_14_                      ✓                                        
    binary_net_mnist_MLP_4_              ✓                                        
    elman_net_with_attention_34_         ✓                                        
    elman_net_with_attention_33_         ✓                                        
    mnist_dcgan_with_label_2_            X     Failed to compare inferring result.
    gated_recurrent_unitGRU_32_          ✓                                        
    12_residual_learning_19_             ✓                                        
    12_residual_learning_20_             ✓                                        
    bidirectional_elman_net_26_          ✓                                        
    11_deconvolution_11_                 ✓                                        
    ================================  =======  ===================================





.. _bvlc_alexnet_model: https://s3.amazonaws.com/download.onnx/models/opset_6/bvlc_alexnet.tar.gz
.. _bvlc_reference_caffenet_model: https://s3.amazonaws.com/download.onnx/models/opset_6/bvlc_reference_caffenet.tar.gz
.. _resnet50_model: https://s3.amazonaws.com/download.onnx/models/opset_6/resnet50.tar.gz
.. _bvlc_reference_rcnn_ilsvrc13_model: https://s3.amazonaws.com/download.onnx/models/opset_6/bvlc_reference_rcnn_ilsvrc13.tar.gz
.. _zfnet512_model: https://s3.amazonaws.com/download.onnx/models/opset_6/zfnet512.tar.gz
.. _shufflenet_model: https://s3.amazonaws.com/download.onnx/models/opset_6/shufflenet.tar.gz
.. _bvlc_googlenet_model: https://s3.amazonaws.com/download.onnx/models/opset_6/bvlc_googlenet.tar.gz
.. _densenet121_model: https://s3.amazonaws.com/download.onnx/models/opset_6/densenet121.tar.gz
.. _squeezenet_model: https://s3.amazonaws.com/download.onnx/models/opset_6/squeezenet.tar.gz
.. _vgg19_model: https://s3.amazonaws.com/download.onnx/models/opset_6/vgg19.tar.gz
.. _inception_v1_model: https://s3.amazonaws.com/download.onnx/models/opset_6/inception_v1.tar.gz
.. _inception_v2_model: https://s3.amazonaws.com/download.onnx/models/opset_6/inception_v2.tar.gz
.. _stacked_GRU_27: https://dl.sony.com/assets/sdcproj/tutorial/recurrent_neural_networks/stacked_GRU.sdcproj
.. _LeNet_35: https://dl.sony.com/assets/sdcproj/image_recognition/MNIST/LeNet.sdcproj
.. _01_logistic_regression_9: https://dl.sony.com/assets/sdcproj/tutorial/basics/01_logistic_regression.sdcproj
.. _LSTM_auto_encoder_24: https://dl.sony.com/assets/sdcproj/tutorial/recurrent_neural_networks/LSTM_auto_encoder.sdcproj
.. _mnist_vae_3: https://dl.sony.com/assets/sdcproj/image_generation/mnist_vae.sdcproj
.. _10_deep_mlp_13: https://dl.sony.com/assets/sdcproj/tutorial/basics/10_deep_mlp.sdcproj
.. _11_deconvolution_12: https://dl.sony.com/assets/sdcproj/tutorial/basics/11_deconvolution.sdcproj
.. _01_logistic_regression_10: https://dl.sony.com/assets/sdcproj/tutorial/basics/01_logistic_regression.sdcproj
.. _mnist_dcgan_with_label_1: https://dl.sony.com/assets/sdcproj/image_generation/mnist_dcgan_with_label.sdcproj
.. _semi_supervised_learning_VAT_37: https://dl.sony.com/assets/sdcproj/image_recognition/MNIST/semi_supervised_learning_VAT.sdcproj
.. _binary_net_mnist_LeNet_7: https://dl.sony.com/assets/sdcproj/tutorial/binary_networks/binary_net_mnist_LeNet.sdcproj
.. _LSTM_auto_encoder_23: https://dl.sony.com/assets/sdcproj/tutorial/recurrent_neural_networks/LSTM_auto_encoder.sdcproj
.. _long_short_term_memoryLSTM_29: https://dl.sony.com/assets/sdcproj/tutorial/recurrent_neural_networks/long_short_term_memory(LSTM).sdcproj
.. _binary_connect_mnist_LeNet_5: https://dl.sony.com/assets/sdcproj/tutorial/binary_networks/binary_connect_mnist_LeNet.sdcproj
.. _long_short_term_memoryLSTM_30: https://dl.sony.com/assets/sdcproj/tutorial/recurrent_neural_networks/long_short_term_memory(LSTM).sdcproj
.. _binary_connect_mnist_MLP_8: https://dl.sony.com/assets/sdcproj/tutorial/binary_networks/binary_connect_mnist_MLP.sdcproj
.. _elman_net_22: https://dl.sony.com/assets/sdcproj/tutorial/recurrent_neural_networks/elman_net.sdcproj
.. _02_binary_cnn_15: https://dl.sony.com/assets/sdcproj/tutorial/basics/02_binary_cnn.sdcproj
.. _gated_recurrent_unitGRU_31: https://dl.sony.com/assets/sdcproj/tutorial/recurrent_neural_networks/gated_recurrent_unit(GRU).sdcproj
.. _bidirectional_elman_net_25: https://dl.sony.com/assets/sdcproj/tutorial/recurrent_neural_networks/bidirectional_elman_net.sdcproj
.. _06_auto_encoder_18: https://dl.sony.com/assets/sdcproj/tutorial/basics/06_auto_encoder.sdcproj
.. _02_binary_cnn_16: https://dl.sony.com/assets/sdcproj/tutorial/basics/02_binary_cnn.sdcproj
.. _stacked_GRU_28: https://dl.sony.com/assets/sdcproj/tutorial/recurrent_neural_networks/stacked_GRU.sdcproj
.. _elman_net_21: https://dl.sony.com/assets/sdcproj/tutorial/recurrent_neural_networks/elman_net.sdcproj
.. _LeNet_36: https://dl.sony.com/assets/sdcproj/image_recognition/MNIST/LeNet.sdcproj
.. _binary_weight_mnist_MLP_6: https://dl.sony.com/assets/sdcproj/tutorial/binary_networks/binary_weight_mnist_MLP.sdcproj
.. _06_auto_encoder_17: https://dl.sony.com/assets/sdcproj/tutorial/basics/06_auto_encoder.sdcproj
.. _10_deep_mlp_14: https://dl.sony.com/assets/sdcproj/tutorial/basics/10_deep_mlp.sdcproj
.. _binary_net_mnist_MLP_4: https://dl.sony.com/assets/sdcproj/tutorial/binary_networks/binary_net_mnist_MLP.sdcproj
.. _elman_net_with_attention_34: https://dl.sony.com/assets/sdcproj/tutorial/recurrent_neural_networks/elman_net_with_attention.sdcproj
.. _elman_net_with_attention_33: https://dl.sony.com/assets/sdcproj/tutorial/recurrent_neural_networks/elman_net_with_attention.sdcproj
.. _mnist_dcgan_with_label_2: https://dl.sony.com/assets/sdcproj/image_generation/mnist_dcgan_with_label.sdcproj
.. _gated_recurrent_unitGRU_32: https://dl.sony.com/assets/sdcproj/tutorial/recurrent_neural_networks/gated_recurrent_unit(GRU).sdcproj
.. _12_residual_learning_19: https://dl.sony.com/assets/sdcproj/tutorial/basics/12_residual_learning.sdcproj
.. _12_residual_learning_20: https://dl.sony.com/assets/sdcproj/tutorial/basics/12_residual_learning.sdcproj
.. _bidirectional_elman_net_26: https://dl.sony.com/assets/sdcproj/tutorial/recurrent_neural_networks/bidirectional_elman_net.sdcproj
.. _11_deconvolution_11: https://dl.sony.com/assets/sdcproj/tutorial/basics/11_deconvolution.sdcproj
.. _Resnet-50_4_178: https://nnabla.org/pretrained-models/nnp_models/imagenet/Resnet-50/Resnet-50.nnp
.. _Resnet-34_4_128: https://nnabla.org/pretrained-models/nnp_models/imagenet/Resnet-34/Resnet-34.nnp
.. _Resnet-101_4_348: https://nnabla.org/pretrained-models/nnp_models/imagenet/Resnet-101/Resnet-101.nnp
.. _DenseNet-161_2_570: https://nnabla.org/pretrained-models/nnp_models/imagenet/DenseNet-161/DenseNet-161.nnp
.. _ShuffleNet-0.5x_2_202: https://nnabla.org/pretrained-models/nnp_models/imagenet/ShuffleNet-0.5x/ShuffleNet-0.5x.nnp
.. _GoogLeNet_4_142: https://nnabla.org/pretrained-models/nnp_models/imagenet/GoogLeNet/GoogLeNet.nnp
.. _NIN: https://nnabla.org/pretrained-models/nnp_models/imagenet/NIN/NIN.nnp
.. _Resnet-18_3_71: https://nnabla.org/pretrained-models/nnp_models/imagenet/Resnet-18/Resnet-18.nnp
.. _Xception: https://nnabla.org/pretrained-models/nnp_models/imagenet/Xception/Xception.nnp
.. _AlexNet: https://notfound
.. _ShuffleNet_2_202: https://nnabla.org/pretrained-models/nnp_models/imagenet/ShuffleNet-2.0x/ShuffleNet-2.0x.nnp
.. _MobileNet_1_86: https://nnabla.org/pretrained-models/nnp_models/imagenet/MobileNet/MobileNet.nnp
.. _SqueezeNet-1.0_2_70: https://nnabla.org/pretrained-models/nnp_models/imagenet/SqueezeNet-1.0/SqueezeNet-1.0.nnp
.. _VGG-13: https://nnabla.org/pretrained-models/nnp_models/imagenet/VGG-13/VGG-13.nnp
.. _Resnet-152_4_518: https://nnabla.org/pretrained-models/nnp_models/imagenet/Resnet-152/Resnet-152.nnp
.. _VGG-11: https://nnabla.org/pretrained-models/nnp_models/imagenet/VGG-11/VGG-11.nnp
.. _SqueezeNet-1.1_2_70: https://nnabla.org/pretrained-models/nnp_models/imagenet/SqueezeNet-1.1/SqueezeNet-1.1.nnp
.. _VGG-16: https://nnabla.org/pretrained-models/nnp_models/imagenet/VGG-16/VGG-16.nnp
.. _word_embedding: https://github.com/sony/nnabla-examples
.. _classification: https://github.com/sony/nnabla-examples
.. _wavenet: https://github.com/sony/nnabla-examples
.. _capsules: https://github.com/sony/nnabla-examples
.. _cycle_gan: https://github.com/sony/nnabla-examples
.. _siamese_embedding: https://github.com/sony/nnabla-examples
.. _meta_learning: https://github.com/sony/nnabla-examples
.. _yolov2: https://github.com/sony/nnabla-examples
.. _pix2pix: https://github.com/sony/nnabla-examples
.. _deeplabv3plus: https://github.com/sony/nnabla-examples
.. _mobilenet_v1_1.0_224: https://storage.googleapis.com/download.tensorflow.org/models/mobilenet_v1_1.0_224_frozen.tgz
.. _inception_resnet_v2_2016_08_30_frozen: https://storage.googleapis.com/download.tensorflow.org/models/inception_resnet_v2_2016_08_30_frozen.pb.tar.gz
.. _inception_v4_2016_09_09_frozen: https://storage.googleapis.com/download.tensorflow.org/models/inception_v4_2016_09_09_frozen.pb.tar.gz
.. _conv-layers_frozen: https://github.com/onnx/tensorflow-onnx/blob/master/tests/models/conv-layers/frozen.pb
.. _lstm_frozen: https://github.com/onnx/tensorflow-onnx/blob/master/tests/models/lstm/frozen.pb
.. _mobilenet_v1_0.75_192: https://storage.googleapis.com/download.tensorflow.org/models/mobilenet_v1_0.75_192_frozen.tgz
.. _inception_v1_2016_08_28_frozen: https://storage.googleapis.com/download.tensorflow.org/models/inception_v1_2016_08_28_frozen.pb.tar.gz
.. _inception_v3_2016_08_28_frozen: https://storage.googleapis.com/download.tensorflow.org/models/inception_v3_2016_08_28_frozen.pb.tar.gz
.. _fc-layers_frozen: https://github.com/onnx/tensorflow-onnx/blob/master/tests/models/fc-layers/frozen.pb
.. _ae0_frozen: https://github.com/onnx/tensorflow-onnx/blob/master/tests/models/ae0/frozen.pb
