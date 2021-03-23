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
    bvlc_reference_caffenet_model_          ✓                                                               
    densenet121_model_                      ✓                                                               
    bvlc_googlenet_model_                   ✓                                                               
    inception_v2_model_                     ✓                                                               
    zfnet512_model_                         ✓                                                               
    bvlc_alexnet_model_                     ✓                                                               
    vgg19_model_                            ✓                                                               
    inception_v1_model_                     X     The `edge` mode of the `pad` in nnabla is not implemented.
    squeezenet_model_                       ✓                                                               
    bvlc_reference_rcnn_ilsvrc13_model_     ✓                                                               
    resnet50_model_                         ✓                                                               
    shufflenet_model_                       ✓                                                               
    ===================================  =======  ==========================================================





Export
------

- ✓: Support to convert
- X: Not support

Total: 58/65

ONNX Export Sample Test(nnp --> onnx)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Count 34/37
 

    ================================  =======  =======================================================
                  Name                Support                           Memo                          
    ================================  =======  =======================================================
    elman_net_with_attention_33_         ✓                                                            
    LeNet_36_                            ✓                                                            
    stacked_GRU_28_                      ✓                                                            
    01_logistic_regression_9_            ✓                                                            
    10_deep_mlp_13_                      ✓                                                            
    01_logistic_regression_10_           ✓                                                            
    binary_connect_mnist_LeNet_5_        ✓                                                            
    02_binary_cnn_16_                    ✓                                                            
    gated_recurrent_unitGRU_32_          ✓                                                            
    elman_net_22_                        ✓                                                            
    mnist_vae_3_                         ✓                                                            
    gated_recurrent_unitGRU_31_          ✓                                                            
    stacked_GRU_27_                      ✓                                                            
    10_deep_mlp_14_                      ✓                                                            
    mnist_dcgan_with_label_2_            X     NNabla converter error, will be fixed in the future.   
    elman_net_with_attention_34_         ✓                                                            
    binary_connect_mnist_MLP_8_          ✓                                                            
    elman_net_21_                        ✓                                                            
    mnist_dcgan_with_label_1_            X     NNabla converter error, will be fixed in the future.   
    LSTM_auto_encoder_23_                ✓                                                            
    binary_weight_mnist_MLP_6_           ✓                                                            
    06_auto_encoder_17_                  ✓                                                            
    bidirectional_elman_net_26_          ✓                                                            
    06_auto_encoder_18_                  ✓                                                            
    12_residual_learning_20_             ✓                                                            
    12_residual_learning_19_             ✓                                                            
    bidirectional_elman_net_25_          ✓                                                            
    long_short_term_memoryLSTM_29_       ✓                                                            
    binary_net_mnist_LeNet_7_            ✓                                                            
    02_binary_cnn_15_                    ✓                                                            
    semi_supervised_learning_VAT_37_     X     NNP with only a single executor is currently supported.
    11_deconvolution_12_                 ✓                                                            
    long_short_term_memoryLSTM_30_       ✓                                                            
    LeNet_35_                            ✓                                                            
    binary_net_mnist_MLP_4_              ✓                                                            
    11_deconvolution_11_                 ✓                                                            
    LSTM_auto_encoder_24_                ✓                                                            
    ================================  =======  =======================================================


ONNX Export Pretrained Model Test(nnp --> onnx)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Count 17/18
 

    ======================  =======  ====
             Name           Support  Memo
    ======================  =======  ====
    Resnet-101_4_348_          ✓         
    VGG-13_                    ✓         
    ShuffleNet_2_202_          ✓         
    SqueezeNet-1.1_2_70_       ✓         
    Resnet-18_3_71_            ✓         
    VGG-16_                    ✓         
    AlexNet_                   ✓         
    ShuffleNet-0.5x_2_202_     ✓         
    Resnet-34_4_128_           ✓         
    MobileNet_1_86_            ✓         
    Xception_                  ✓         
    DenseNet-161_2_570_        ✓         
    NIN_                       X         
    GoogLeNet_4_142_           ✓         
    VGG-11_                    ✓         
    Resnet-50_4_178_           ✓         
    Resnet-152_4_518_          ✓         
    SqueezeNet-1.0_2_70_       ✓         
    ======================  =======  ====


ONNX Export Example Model Test(nnp --> onnx)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Count 7/10
 

    ==================  =======  ====================================================
           Name         Support                          Memo                        
    ==================  =======  ====================================================
    word_embedding_        ✓                                                         
    yolov2_                ✓                                                         
    deeplabv3plus_         ✓                                                         
    pix2pix_               ✓                                                         
    siamese_embedding_     ✓                                                         
    wavenet_               X     The `onehot` dimension != 2 is not supported.       
    classification_        ✓                                                         
    meta_learning_         X     Failed to compare inferring result.                 
    cycle_gan_             ✓                                                         
    capsules_              X     NNabla converter error, will be fixed in the future.
    ==================  =======  ====================================================





Tensorflow Support Status
=========================

Import
------

- ✓: Support to convert
- X: Not support

Total: 15/16

Tensorflow Import Sample Test(tf --> nnp)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Count 15/16
 

    ======================================  =======  ============================================================
                     Name                   Support                              Memo                            
    ======================================  =======  ============================================================
    GoogLeNet_                                 ✓                                                                 
    ZFNet_                                     ✓                                                                 
    AlexNet_                                   ✓                                                                 
    ResNet50_                                  ✓                                                                 
    LeNet_5_                                   ✓                                                                 
    VGG16_                                     ✓                                                                 
    mobilenet_v1_0.75_192_                     ✓                                                                 
    ae0_frozen_                                ✓                                                                 
    conv-layers_frozen_                        ✓                                                                 
    lstm_frozen_                               X     The `Shape` is currently not supported to convert by nnabla.
    inception_v3_2016_08_28_frozen_            ✓                                                                 
    inception_v4_2016_09_09_frozen_            ✓                                                                 
    inception_v1_2016_08_28_frozen_            ✓                                                                 
    fc-layers_frozen_                          ✓                                                                 
    mobilenet_v1_1.0_224_                      ✓                                                                 
    inception_resnet_v2_2016_08_30_frozen_     ✓                                                                 
    ======================================  =======  ============================================================





Export
------

- ✓: Support to convert
- X: Not support

Total: 58/65

Tensorflow Export Sample Test(nnp --> tf)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Count 34/37
 

    ================================  =======  =======================================================
                  Name                Support                           Memo                          
    ================================  =======  =======================================================
    elman_net_with_attention_33_         ✓                                                            
    LeNet_36_                            ✓                                                            
    stacked_GRU_28_                      ✓                                                            
    01_logistic_regression_9_            ✓                                                            
    10_deep_mlp_13_                      ✓                                                            
    01_logistic_regression_10_           ✓                                                            
    binary_connect_mnist_LeNet_5_        ✓                                                            
    02_binary_cnn_16_                    ✓                                                            
    gated_recurrent_unitGRU_32_          ✓                                                            
    elman_net_22_                        ✓                                                            
    mnist_vae_3_                         ✓                                                            
    gated_recurrent_unitGRU_31_          ✓                                                            
    stacked_GRU_27_                      ✓                                                            
    10_deep_mlp_14_                      ✓                                                            
    mnist_dcgan_with_label_2_            X     NNabla converter error, will be fixed in the future.   
    elman_net_with_attention_34_         ✓                                                            
    binary_connect_mnist_MLP_8_          ✓                                                            
    elman_net_21_                        ✓                                                            
    mnist_dcgan_with_label_1_            X     NNabla converter error, will be fixed in the future.   
    LSTM_auto_encoder_23_                ✓                                                            
    binary_weight_mnist_MLP_6_           ✓                                                            
    06_auto_encoder_17_                  ✓                                                            
    bidirectional_elman_net_26_          ✓                                                            
    06_auto_encoder_18_                  ✓                                                            
    12_residual_learning_20_             ✓                                                            
    12_residual_learning_19_             ✓                                                            
    bidirectional_elman_net_25_          ✓                                                            
    long_short_term_memoryLSTM_29_       ✓                                                            
    binary_net_mnist_LeNet_7_            ✓                                                            
    02_binary_cnn_15_                    ✓                                                            
    semi_supervised_learning_VAT_37_     X     NNP with only a single executor is currently supported.
    11_deconvolution_12_                 ✓                                                            
    long_short_term_memoryLSTM_30_       ✓                                                            
    LeNet_35_                            ✓                                                            
    binary_net_mnist_MLP_4_              ✓                                                            
    11_deconvolution_11_                 ✓                                                            
    LSTM_auto_encoder_24_                ✓                                                            
    ================================  =======  =======================================================


Tensorflow Export Pretrained Models(nnp --> tf)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Count 17/18
 

    ======================  =======  ====
             Name           Support  Memo
    ======================  =======  ====
    Resnet-101_4_348_          ✓         
    VGG-13_                    ✓         
    ShuffleNet_2_202_          ✓         
    SqueezeNet-1.1_2_70_       ✓         
    Resnet-18_3_71_            ✓         
    VGG-16_                    ✓         
    AlexNet_                   ✓         
    ShuffleNet-0.5x_2_202_     ✓         
    Resnet-34_4_128_           ✓         
    MobileNet_1_86_            ✓         
    Xception_                  ✓         
    DenseNet-161_2_570_        ✓         
    NIN_                       X         
    GoogLeNet_4_142_           ✓         
    VGG-11_                    ✓         
    Resnet-50_4_178_           ✓         
    Resnet-152_4_518_          ✓         
    SqueezeNet-1.0_2_70_       ✓         
    ======================  =======  ====


Tensorflow Export Example Models(nnp --> tf)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Count 7/10
 

    ==================  =======  =============================================
           Name         Support                      Memo                     
    ==================  =======  =============================================
    word_embedding_        ✓                                                  
    yolov2_                ✓                                                  
    deeplabv3plus_         ✓     The `Interpolate` is currently not supported.
    pix2pix_               ✓                                                  
    siamese_embedding_     ✓                                                  
    wavenet_               X     The `onehot` dimension != 2 is not supported.
    classification_        ✓                                                  
    meta_learning_         X     Failed to compare inferring result.          
    cycle_gan_             ✓                                                  
    capsules_              X                                                  
    ==================  =======  =============================================




Tensorflow Lite Support Status
==============================


Export
------

- ✓: Support to convert
- X: Not support

Total: 44/65

Tensorflow Lite Export Sample Test(nnp --> tflite)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Count 32/37
 

    ================================  =======  ====
                  Name                Support  Memo
    ================================  =======  ====
    elman_net_with_attention_33_         ✓         
    LeNet_36_                            ✓         
    stacked_GRU_28_                      ✓         
    01_logistic_regression_9_            ✓         
    10_deep_mlp_13_                      ✓         
    01_logistic_regression_10_           ✓         
    binary_connect_mnist_LeNet_5_        ✓         
    02_binary_cnn_16_                    ✓         
    gated_recurrent_unitGRU_32_          ✓         
    elman_net_22_                        ✓         
    mnist_vae_3_                         ✓         
    gated_recurrent_unitGRU_31_          ✓         
    stacked_GRU_27_                      ✓         
    10_deep_mlp_14_                      ✓         
    mnist_dcgan_with_label_2_            X         
    elman_net_with_attention_34_         ✓         
    binary_connect_mnist_MLP_8_          ✓         
    elman_net_21_                        ✓         
    mnist_dcgan_with_label_1_            X         
    LSTM_auto_encoder_23_                ✓         
    binary_weight_mnist_MLP_6_           ✓         
    06_auto_encoder_17_                  ✓         
    bidirectional_elman_net_26_          ✓         
    06_auto_encoder_18_                  ✓         
    12_residual_learning_20_             ✓         
    12_residual_learning_19_             ✓         
    bidirectional_elman_net_25_          ✓         
    long_short_term_memoryLSTM_29_       ✓         
    binary_net_mnist_LeNet_7_            X         
    02_binary_cnn_15_                    ✓         
    semi_supervised_learning_VAT_37_     X         
    11_deconvolution_12_                 ✓         
    long_short_term_memoryLSTM_30_       ✓         
    LeNet_35_                            ✓         
    binary_net_mnist_MLP_4_              X         
    11_deconvolution_11_                 ✓         
    LSTM_auto_encoder_24_                ✓         
    ================================  =======  ====


Tensorflow Lite Export Pretrained Models(nnp --> tflite)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Count 6/18
 

    ======================  =======  ====
             Name           Support  Memo
    ======================  =======  ====
    Resnet-101_4_348_          X         
    VGG-13_                    ✓         
    ShuffleNet_2_202_          X         
    SqueezeNet-1.1_2_70_       ✓         
    Resnet-18_3_71_            X         
    VGG-16_                    ✓         
    AlexNet_                   X         
    ShuffleNet-0.5x_2_202_     X         
    Resnet-34_4_128_           X         
    MobileNet_1_86_            ✓         
    Xception_                  X         
    DenseNet-161_2_570_        X         
    NIN_                       X         
    GoogLeNet_4_142_           X         
    VGG-11_                    ✓         
    Resnet-50_4_178_           X         
    Resnet-152_4_518_          X         
    SqueezeNet-1.0_2_70_       ✓         
    ======================  =======  ====


Tensorflow Lite Export Example Models(nnp --> tflite)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Count 6/10
 

    ==================  =======  ====
           Name         Support  Memo
    ==================  =======  ====
    word_embedding_        ✓         
    yolov2_                ✓         
    deeplabv3plus_         X         
    pix2pix_               ✓         
    siamese_embedding_     ✓         
    wavenet_               X         
    classification_        ✓         
    meta_learning_         X         
    cycle_gan_             ✓         
    capsules_              X         
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
    elman_net_with_attention_33_         ✓                                        
    LeNet_36_                            ✓                                        
    stacked_GRU_28_                      ✓                                        
    01_logistic_regression_9_            ✓                                        
    10_deep_mlp_13_                      ✓                                        
    01_logistic_regression_10_           ✓                                        
    binary_connect_mnist_LeNet_5_        ✓                                        
    02_binary_cnn_16_                    ✓                                        
    gated_recurrent_unitGRU_32_          ✓                                        
    elman_net_22_                        ✓                                        
    mnist_vae_3_                         ✓                                        
    gated_recurrent_unitGRU_31_          ✓                                        
    stacked_GRU_27_                      ✓                                        
    10_deep_mlp_14_                      ✓                                        
    mnist_dcgan_with_label_2_            X     Failed to compare inferring result.
    elman_net_with_attention_34_         ✓                                        
    binary_connect_mnist_MLP_8_          ✓                                        
    elman_net_21_                        ✓                                        
    mnist_dcgan_with_label_1_            X     Failed to infer by nnabla.         
    LSTM_auto_encoder_23_                ✓                                        
    binary_weight_mnist_MLP_6_           ✓                                        
    06_auto_encoder_17_                  ✓                                        
    bidirectional_elman_net_26_          ✓                                        
    06_auto_encoder_18_                  ✓                                        
    12_residual_learning_20_             ✓                                        
    12_residual_learning_19_             ✓                                        
    bidirectional_elman_net_25_          ✓                                        
    long_short_term_memoryLSTM_29_       ✓                                        
    binary_net_mnist_LeNet_7_            ✓                                        
    02_binary_cnn_15_                    ✓                                        
    semi_supervised_learning_VAT_37_     X     Failed to compare inferring result.
    11_deconvolution_12_                 ✓                                        
    long_short_term_memoryLSTM_30_       ✓                                        
    LeNet_35_                            ✓                                        
    binary_net_mnist_MLP_4_              ✓                                        
    11_deconvolution_11_                 ✓                                        
    LSTM_auto_encoder_24_                ✓                                        
    ================================  =======  ===================================





.. _bvlc_reference_caffenet_model: https://media.githubusercontent.com/media/onnx/models/master/vision/classification/caffenet/model/caffenet-9.tar.gz
.. _densenet121_model: https://media.githubusercontent.com/media/onnx/models/master/vision/classification/densenet-121/model/densenet-9.tar.gz
.. _bvlc_googlenet_model: https://media.githubusercontent.com/media/onnx/models/master/vision/classification/inception_and_googlenet/googlenet/model/googlenet-9.tar.gz
.. _inception_v2_model: https://media.githubusercontent.com/media/onnx/models/master/vision/classification/inception_and_googlenet/inception_v2/model/inception-v2-9.tar.gz
.. _zfnet512_model: https://media.githubusercontent.com/media/onnx/models/master/vision/classification/zfnet-512/model/zfnet512-9.tar.gz
.. _bvlc_alexnet_model: https://media.githubusercontent.com/media/onnx/models/master/vision/classification/alexnet/model/bvlcalexnet-9.tar.gz
.. _vgg19_model: https://media.githubusercontent.com/media/onnx/models/master/vision/classification/vgg/model/vgg19-caffe2-9.tar.gz
.. _inception_v1_model: https://media.githubusercontent.com/media/onnx/models/master/vision/classification/inception_and_googlenet/inception_v1/model/inception-v1-9.tar.gz
.. _squeezenet_model: https://media.githubusercontent.com/media/onnx/models/master/vision/classification/squeezenet/model/squeezenet1.0-9.tar.gz
.. _bvlc_reference_rcnn_ilsvrc13_model: https://media.githubusercontent.com/media/onnx/models/master/vision/classification/rcnn_ilsvrc13/model/rcnn-ilsvrc13-9.tar.gz
.. _resnet50_model: https://media.githubusercontent.com/media/onnx/models/master/vision/classification/resnet/model/resnet50-caffe2-v1-9.tar.gz
.. _shufflenet_model: https://media.githubusercontent.com/media/onnx/models/master/vision/classification/shufflenet/model/shufflenet-9.tar.gz
.. _elman_net_with_attention_33: https://dl.sony.com/assets/sdcproj/tutorial/recurrent_neural_networks/elman_net_with_attention.sdcproj
.. _LeNet_36: https://dl.sony.com/assets/sdcproj/image_recognition/MNIST/LeNet.sdcproj
.. _stacked_GRU_28: https://dl.sony.com/assets/sdcproj/tutorial/recurrent_neural_networks/stacked_GRU.sdcproj
.. _01_logistic_regression_9: https://dl.sony.com/assets/sdcproj/tutorial/basics/01_logistic_regression.sdcproj
.. _10_deep_mlp_13: https://dl.sony.com/assets/sdcproj/tutorial/basics/10_deep_mlp.sdcproj
.. _01_logistic_regression_10: https://dl.sony.com/assets/sdcproj/tutorial/basics/01_logistic_regression.sdcproj
.. _binary_connect_mnist_LeNet_5: https://dl.sony.com/assets/sdcproj/tutorial/binary_networks/binary_connect_mnist_LeNet.sdcproj
.. _02_binary_cnn_16: https://dl.sony.com/assets/sdcproj/tutorial/basics/02_binary_cnn.sdcproj
.. _gated_recurrent_unitGRU_32: https://dl.sony.com/assets/sdcproj/tutorial/recurrent_neural_networks/gated_recurrent_unit(GRU).sdcproj
.. _elman_net_22: https://dl.sony.com/assets/sdcproj/tutorial/recurrent_neural_networks/elman_net.sdcproj
.. _mnist_vae_3: https://dl.sony.com/assets/sdcproj/image_generation/mnist_vae.sdcproj
.. _gated_recurrent_unitGRU_31: https://dl.sony.com/assets/sdcproj/tutorial/recurrent_neural_networks/gated_recurrent_unit(GRU).sdcproj
.. _stacked_GRU_27: https://dl.sony.com/assets/sdcproj/tutorial/recurrent_neural_networks/stacked_GRU.sdcproj
.. _10_deep_mlp_14: https://dl.sony.com/assets/sdcproj/tutorial/basics/10_deep_mlp.sdcproj
.. _mnist_dcgan_with_label_2: https://dl.sony.com/assets/sdcproj/image_generation/mnist_dcgan_with_label.sdcproj
.. _elman_net_with_attention_34: https://dl.sony.com/assets/sdcproj/tutorial/recurrent_neural_networks/elman_net_with_attention.sdcproj
.. _binary_connect_mnist_MLP_8: https://dl.sony.com/assets/sdcproj/tutorial/binary_networks/binary_connect_mnist_MLP.sdcproj
.. _elman_net_21: https://dl.sony.com/assets/sdcproj/tutorial/recurrent_neural_networks/elman_net.sdcproj
.. _mnist_dcgan_with_label_1: https://dl.sony.com/assets/sdcproj/image_generation/mnist_dcgan_with_label.sdcproj
.. _LSTM_auto_encoder_23: https://dl.sony.com/assets/sdcproj/tutorial/recurrent_neural_networks/LSTM_auto_encoder.sdcproj
.. _binary_weight_mnist_MLP_6: https://dl.sony.com/assets/sdcproj/tutorial/binary_networks/binary_weight_mnist_MLP.sdcproj
.. _06_auto_encoder_17: https://dl.sony.com/assets/sdcproj/tutorial/basics/06_auto_encoder.sdcproj
.. _bidirectional_elman_net_26: https://dl.sony.com/assets/sdcproj/tutorial/recurrent_neural_networks/bidirectional_elman_net.sdcproj
.. _06_auto_encoder_18: https://dl.sony.com/assets/sdcproj/tutorial/basics/06_auto_encoder.sdcproj
.. _12_residual_learning_20: https://dl.sony.com/assets/sdcproj/tutorial/basics/12_residual_learning.sdcproj
.. _12_residual_learning_19: https://dl.sony.com/assets/sdcproj/tutorial/basics/12_residual_learning.sdcproj
.. _bidirectional_elman_net_25: https://dl.sony.com/assets/sdcproj/tutorial/recurrent_neural_networks/bidirectional_elman_net.sdcproj
.. _long_short_term_memoryLSTM_29: https://dl.sony.com/assets/sdcproj/tutorial/recurrent_neural_networks/long_short_term_memory(LSTM).sdcproj
.. _binary_net_mnist_LeNet_7: https://dl.sony.com/assets/sdcproj/tutorial/binary_networks/binary_net_mnist_LeNet.sdcproj
.. _02_binary_cnn_15: https://dl.sony.com/assets/sdcproj/tutorial/basics/02_binary_cnn.sdcproj
.. _semi_supervised_learning_VAT_37: https://dl.sony.com/assets/sdcproj/image_recognition/MNIST/semi_supervised_learning_VAT.sdcproj
.. _11_deconvolution_12: https://dl.sony.com/assets/sdcproj/tutorial/basics/11_deconvolution.sdcproj
.. _long_short_term_memoryLSTM_30: https://dl.sony.com/assets/sdcproj/tutorial/recurrent_neural_networks/long_short_term_memory(LSTM).sdcproj
.. _LeNet_35: https://dl.sony.com/assets/sdcproj/image_recognition/MNIST/LeNet.sdcproj
.. _binary_net_mnist_MLP_4: https://dl.sony.com/assets/sdcproj/tutorial/binary_networks/binary_net_mnist_MLP.sdcproj
.. _11_deconvolution_11: https://dl.sony.com/assets/sdcproj/tutorial/basics/11_deconvolution.sdcproj
.. _LSTM_auto_encoder_24: https://dl.sony.com/assets/sdcproj/tutorial/recurrent_neural_networks/LSTM_auto_encoder.sdcproj
.. _Resnet-101_4_348: https://nnabla.org/pretrained-models/nnp_models/imagenet/Resnet-101/Resnet-101.nnp
.. _VGG-13: https://nnabla.org/pretrained-models/nnp_models/imagenet/VGG-13/VGG-13.nnp
.. _ShuffleNet_2_202: https://nnabla.org/pretrained-models/nnp_models/imagenet/ShuffleNet-2.0x/ShuffleNet-2.0x.nnp
.. _SqueezeNet-1.1_2_70: https://nnabla.org/pretrained-models/nnp_models/imagenet/SqueezeNet-1.1/SqueezeNet-1.1.nnp
.. _Resnet-18_3_71: https://nnabla.org/pretrained-models/nnp_models/imagenet/Resnet-18/Resnet-18.nnp
.. _VGG-16: https://nnabla.org/pretrained-models/nnp_models/imagenet/VGG-16/VGG-16.nnp
.. _AlexNet: https://notfound
.. _ShuffleNet-0.5x_2_202: https://nnabla.org/pretrained-models/nnp_models/imagenet/ShuffleNet-0.5x/ShuffleNet-0.5x.nnp
.. _Resnet-34_4_128: https://nnabla.org/pretrained-models/nnp_models/imagenet/Resnet-34/Resnet-34.nnp
.. _MobileNet_1_86: https://nnabla.org/pretrained-models/nnp_models/imagenet/MobileNet/MobileNet.nnp
.. _Xception: https://nnabla.org/pretrained-models/nnp_models/imagenet/Xception/Xception.nnp
.. _DenseNet-161_2_570: https://nnabla.org/pretrained-models/nnp_models/imagenet/DenseNet-161/DenseNet-161.nnp
.. _NIN: https://nnabla.org/pretrained-models/nnp_models/imagenet/NIN/NIN.nnp
.. _GoogLeNet_4_142: https://nnabla.org/pretrained-models/nnp_models/imagenet/GoogLeNet/GoogLeNet.nnp
.. _VGG-11: https://nnabla.org/pretrained-models/nnp_models/imagenet/VGG-11/VGG-11.nnp
.. _Resnet-50_4_178: https://nnabla.org/pretrained-models/nnp_models/imagenet/Resnet-50/Resnet-50.nnp
.. _Resnet-152_4_518: https://nnabla.org/pretrained-models/nnp_models/imagenet/Resnet-152/Resnet-152.nnp
.. _SqueezeNet-1.0_2_70: https://nnabla.org/pretrained-models/nnp_models/imagenet/SqueezeNet-1.0/SqueezeNet-1.0.nnp
.. _word_embedding: https://github.com/sony/nnabla-examples
.. _yolov2: https://github.com/sony/nnabla-examples
.. _deeplabv3plus: https://github.com/sony/nnabla-examples
.. _pix2pix: https://github.com/sony/nnabla-examples
.. _siamese_embedding: https://github.com/sony/nnabla-examples
.. _wavenet: https://github.com/sony/nnabla-examples
.. _classification: https://github.com/sony/nnabla-examples
.. _meta_learning: https://github.com/sony/nnabla-examples
.. _cycle_gan: https://github.com/sony/nnabla-examples
.. _capsules: https://github.com/sony/nnabla-examples
.. _GoogLeNet: https://notfound
.. _ZFNet: https://notfound
.. _AlexNet: https://notfound
.. _ResNet50: https://notfound
.. _LeNet_5: https://notfound
.. _VGG16: https://notfound
.. _mobilenet_v1_0.75_192: https://storage.googleapis.com/download.tensorflow.org/models/mobilenet_v1_0.75_192_frozen.tgz
.. _ae0_frozen: https://github.com/onnx/tensorflow-onnx/blob/master/tests/models/ae0/frozen.pb
.. _conv-layers_frozen: https://github.com/onnx/tensorflow-onnx/blob/master/tests/models/conv-layers/frozen.pb
.. _lstm_frozen: https://github.com/onnx/tensorflow-onnx/blob/master/tests/models/lstm/frozen.pb
.. _inception_v3_2016_08_28_frozen: https://storage.googleapis.com/download.tensorflow.org/models/inception_v3_2016_08_28_frozen.pb.tar.gz
.. _inception_v4_2016_09_09_frozen: https://storage.googleapis.com/download.tensorflow.org/models/inception_v4_2016_09_09_frozen.pb.tar.gz
.. _inception_v1_2016_08_28_frozen: https://storage.googleapis.com/download.tensorflow.org/models/inception_v1_2016_08_28_frozen.pb.tar.gz
.. _fc-layers_frozen: https://github.com/onnx/tensorflow-onnx/blob/master/tests/models/fc-layers/frozen.pb
.. _mobilenet_v1_1.0_224: https://storage.googleapis.com/download.tensorflow.org/models/mobilenet_v1_1.0_224_frozen.tgz
.. _inception_resnet_v2_2016_08_30_frozen: https://storage.googleapis.com/download.tensorflow.org/models/inception_resnet_v2_2016_08_30_frozen.pb.tar.gz
