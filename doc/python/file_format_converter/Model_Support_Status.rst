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
    bvlc_googlenet_model_                   ✓                                                               
    bvlc_reference_caffenet_model_          ✓                                                               
    bvlc_reference_rcnn_ilsvrc13_model_     ✓                                                               
    densenet121_model_                      ✓                                                               
    inception_v1_model_                     X     The `edge` mode of the `pad` in nnabla is not implemented.
    inception_v2_model_                     ✓                                                               
    resnet50_model_                         ✓                                                               
    shufflenet_model_                       ✓                                                               
    squeezenet_model_                       ✓                                                               
    vgg19_model_                            ✓                                                               
    zfnet512_model_                         ✓                                                               
    ===================================  =======  ==========================================================





Export
------

- ✓: Support to convert
- X: Not support

Total: 60/65

ONNX Export Sample Test(nnp --> onnx)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Count 34/37


    ================================  =======  =======================================================
                  Name                Support                           Memo                          
    ================================  =======  =======================================================
    01_logistic_regression_10_           ✓                                                            
    01_logistic_regression_9_            ✓                                                            
    02_binary_cnn_15_                    ✓                                                            
    02_binary_cnn_16_                    ✓                                                            
    06_auto_encoder_17_                  ✓                                                            
    06_auto_encoder_18_                  ✓                                                            
    10_deep_mlp_13_                      ✓                                                            
    10_deep_mlp_14_                      ✓                                                            
    11_deconvolution_11_                 ✓                                                            
    11_deconvolution_12_                 ✓                                                            
    12_residual_learning_19_             ✓                                                            
    12_residual_learning_20_             ✓                                                            
    LSTM_auto_encoder_23_                ✓                                                            
    LSTM_auto_encoder_24_                ✓                                                            
    LeNet_35_                            ✓                                                            
    LeNet_36_                            ✓                                                            
    bidirectional_elman_net_25_          ✓                                                            
    bidirectional_elman_net_26_          ✓                                                            
    binary_connect_mnist_LeNet_5_        ✓                                                            
    binary_connect_mnist_MLP_8_          ✓                                                            
    binary_net_mnist_LeNet_7_            ✓                                                            
    binary_net_mnist_MLP_4_              ✓                                                            
    binary_weight_mnist_MLP_6_           ✓                                                            
    elman_net_21_                        ✓                                                            
    elman_net_22_                        ✓                                                            
    elman_net_with_attention_33_         ✓                                                            
    elman_net_with_attention_34_         ✓                                                            
    gated_recurrent_unitGRU_31_          ✓                                                            
    gated_recurrent_unitGRU_32_          ✓                                                            
    long_short_term_memoryLSTM_29_       ✓                                                            
    long_short_term_memoryLSTM_30_       ✓                                                            
    mnist_dcgan_with_label_1_            X     NNabla converter error, will be fixed in the future.   
    mnist_dcgan_with_label_2_            X     NNabla converter error, will be fixed in the future.   
    mnist_vae_3_                         ✓                                                            
    semi_supervised_learning_VAT_37_     X     NNP with only a single executor is currently supported.
    stacked_GRU_27_                      ✓                                                            
    stacked_GRU_28_                      ✓                                                            
    ================================  =======  =======================================================


ONNX Export Pretrained Model Test(nnp --> onnx)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Count 17/18


    ======================  =======  ====
             Name           Support  Memo
    ======================  =======  ====
    AlexNet_                   ✓         
    DenseNet-161_2_570_        ✓         
    GoogLeNet_4_142_           ✓         
    MobileNet_1_86_            ✓         
    NIN_                       ✓         
    Resnet-101_4_348_          ✓         
    Resnet-152_4_518_          ✓         
    Resnet-18_3_71_            ✓         
    Resnet-34_4_128_           X         
    Resnet-50_4_178_           ✓         
    ShuffleNet-0.5x_2_202_     ✓         
    ShuffleNet_2_202_          ✓         
    SqueezeNet-1.0_2_70_       ✓         
    SqueezeNet-1.1_2_70_       ✓         
    VGG-11_                    ✓         
    VGG-13_                    ✓         
    VGG-16_                    ✓         
    Xception_                  ✓         
    ======================  =======  ====


ONNX Export Example Model Test(nnp --> onnx)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Count 9/10


    ==================  =======  =============================================
           Name         Support                      Memo                     
    ==================  =======  =============================================
    capsules_              ✓                                                  
    classification_        ✓                                                  
    cycle_gan_             ✓                                                  
    deeplabv3plus_         ✓                                                  
    meta_learning_         ✓                                                  
    pix2pix_               ✓                                                  
    siamese_embedding_     ✓                                                  
    wavenet_               X     The `onehot` dimension != 2 is not supported.
    word_embedding_        ✓                                                  
    yolov2_                ✓                                                  
    ==================  =======  =============================================





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
    AlexNet_                                   ✓                                                                 
    GoogLeNet_                                 ✓                                                                 
    LeNet_5_                                   ✓                                                                 
    ResNet50_                                  ✓                                                                 
    VGG16_                                     ✓                                                                 
    ZFNet_                                     ✓                                                                 
    ae0_frozen_                                ✓                                                                 
    conv-layers_frozen_                        ✓                                                                 
    fc-layers_frozen_                          ✓                                                                 
    inception_resnet_v2_2016_08_30_frozen_     ✓                                                                 
    inception_v1_2016_08_28_frozen_            ✓                                                                 
    inception_v3_2016_08_28_frozen_            ✓                                                                 
    inception_v4_2016_09_09_frozen_            ✓                                                                 
    lstm_frozen_                               X     The `Shape` is currently not supported to convert by nnabla.
    mobilenet_v1_0.75_192_                     ✓                                                                 
    mobilenet_v1_1.0_224_                      ✓                                                                 
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
    01_logistic_regression_10_           ✓                                                            
    01_logistic_regression_9_            ✓                                                            
    02_binary_cnn_15_                    ✓                                                            
    02_binary_cnn_16_                    ✓                                                            
    06_auto_encoder_17_                  ✓                                                            
    06_auto_encoder_18_                  ✓                                                            
    10_deep_mlp_13_                      ✓                                                            
    10_deep_mlp_14_                      ✓                                                            
    11_deconvolution_11_                 ✓                                                            
    11_deconvolution_12_                 ✓                                                            
    12_residual_learning_19_             ✓                                                            
    12_residual_learning_20_             ✓                                                            
    LSTM_auto_encoder_23_                ✓                                                            
    LSTM_auto_encoder_24_                ✓                                                            
    LeNet_35_                            ✓                                                            
    LeNet_36_                            ✓                                                            
    bidirectional_elman_net_25_          ✓                                                            
    bidirectional_elman_net_26_          ✓                                                            
    binary_connect_mnist_LeNet_5_        ✓                                                            
    binary_connect_mnist_MLP_8_          ✓                                                            
    binary_net_mnist_LeNet_7_            ✓                                                            
    binary_net_mnist_MLP_4_              ✓                                                            
    binary_weight_mnist_MLP_6_           ✓                                                            
    elman_net_21_                        ✓                                                            
    elman_net_22_                        ✓                                                            
    elman_net_with_attention_33_         ✓                                                            
    elman_net_with_attention_34_         ✓                                                            
    gated_recurrent_unitGRU_31_          ✓                                                            
    gated_recurrent_unitGRU_32_          ✓                                                            
    long_short_term_memoryLSTM_29_       ✓                                                            
    long_short_term_memoryLSTM_30_       ✓                                                            
    mnist_dcgan_with_label_1_            X     NNabla converter error, will be fixed in the future.   
    mnist_dcgan_with_label_2_            X     NNabla converter error, will be fixed in the future.   
    mnist_vae_3_                         ✓                                                            
    semi_supervised_learning_VAT_37_     X     NNP with only a single executor is currently supported.
    stacked_GRU_27_                      ✓                                                            
    stacked_GRU_28_                      ✓                                                            
    ================================  =======  =======================================================


Tensorflow Export Pretrained Models(nnp --> tf)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Count 15/18


    ======================  =======  ====
             Name           Support  Memo
    ======================  =======  ====
    AlexNet_                   X         
    DenseNet-161_2_570_        ✓         
    GoogLeNet_4_142_           ✓         
    MobileNet_1_86_            ✓         
    NIN_                       ✓         
    Resnet-101_4_348_          ✓         
    Resnet-152_4_518_          ✓         
    Resnet-18_3_71_            ✓         
    Resnet-34_4_128_           ✓         
    Resnet-50_4_178_           ✓         
    ShuffleNet-0.5x_2_202_     X         
    ShuffleNet_2_202_          X         
    SqueezeNet-1.0_2_70_       ✓         
    SqueezeNet-1.1_2_70_       ✓         
    VGG-11_                    ✓         
    VGG-13_                    ✓         
    VGG-16_                    ✓         
    Xception_                  ✓         
    ======================  =======  ====


Tensorflow Export Example Models(nnp --> tf)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Count 9/10


    ==================  =======  =============================================
           Name         Support                      Memo                     
    ==================  =======  =============================================
    capsules_              ✓                                                  
    classification_        ✓                                                  
    cycle_gan_             ✓                                                  
    deeplabv3plus_         ✓                                                  
    meta_learning_         ✓                                                  
    pix2pix_               ✓                                                  
    siamese_embedding_     ✓                                                  
    wavenet_               X     The `onehot` dimension != 2 is not supported.
    word_embedding_        ✓                                                  
    yolov2_                ✓                                                  
    ==================  =======  =============================================




Tensorflow Lite Support Status
==============================


Export
------

- ✓: Support to convert
- X: Not support

Total: 46/65

Tensorflow Lite Export Sample Test(nnp --> tflite)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Count 29/37


    ================================  =======  ====
                  Name                Support  Memo
    ================================  =======  ====
    01_logistic_regression_10_           ✓         
    01_logistic_regression_9_            ✓         
    02_binary_cnn_15_                    ✓         
    02_binary_cnn_16_                    ✓         
    06_auto_encoder_17_                  ✓         
    06_auto_encoder_18_                  ✓         
    10_deep_mlp_13_                      ✓         
    10_deep_mlp_14_                      ✓         
    11_deconvolution_11_                 ✓         
    11_deconvolution_12_                 ✓         
    12_residual_learning_19_             ✓         
    12_residual_learning_20_             ✓         
    LSTM_auto_encoder_23_                ✓         
    LSTM_auto_encoder_24_                ✓         
    LeNet_35_                            ✓         
    LeNet_36_                            ✓         
    bidirectional_elman_net_25_          ✓         
    bidirectional_elman_net_26_          ✓         
    binary_connect_mnist_LeNet_5_        X         
    binary_connect_mnist_MLP_8_          X         
    binary_net_mnist_LeNet_7_            X         
    binary_net_mnist_MLP_4_              X         
    binary_weight_mnist_MLP_6_           X         
    elman_net_21_                        ✓         
    elman_net_22_                        ✓         
    elman_net_with_attention_33_         ✓         
    elman_net_with_attention_34_         ✓         
    gated_recurrent_unitGRU_31_          ✓         
    gated_recurrent_unitGRU_32_          ✓         
    long_short_term_memoryLSTM_29_       ✓         
    long_short_term_memoryLSTM_30_       ✓         
    mnist_dcgan_with_label_1_            X         
    mnist_dcgan_with_label_2_            X         
    mnist_vae_3_                         ✓         
    semi_supervised_learning_VAT_37_     X         
    stacked_GRU_27_                      ✓         
    stacked_GRU_28_                      ✓         
    ================================  =======  ====


Tensorflow Lite Export Pretrained Models(nnp --> tflite)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Count 10/18


    ======================  =======  ====
             Name           Support  Memo
    ======================  =======  ====
    AlexNet_                   ✓         
    DenseNet-161_2_570_        X         
    GoogLeNet_4_142_           ✓         
    MobileNet_1_86_            ✓         
    NIN_                       X         
    Resnet-101_4_348_          ✓         
    Resnet-152_4_518_          ✓         
    Resnet-18_3_71_            ✓         
    Resnet-34_4_128_           ✓         
    Resnet-50_4_178_           ✓         
    ShuffleNet-0.5x_2_202_     ✓         
    ShuffleNet_2_202_          ✓         
    SqueezeNet-1.0_2_70_       X         
    SqueezeNet-1.1_2_70_       X         
    VGG-11_                    X         
    VGG-13_                    X         
    VGG-16_                    X         
    Xception_                  X         
    ======================  =======  ====


Tensorflow Lite Export Example Models(nnp --> tflite)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Count 7/10


    ==================  =======  ====
           Name         Support  Memo
    ==================  =======  ====
    capsules_              X         
    classification_        ✓         
    cycle_gan_             ✓         
    deeplabv3plus_         ✓         
    meta_learning_         ✓         
    pix2pix_               ✓         
    siamese_embedding_     ✓         
    wavenet_               X         
    word_embedding_        ✓         
    yolov2_                X         
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
    01_logistic_regression_10_           ✓                                        
    01_logistic_regression_9_            ✓                                        
    02_binary_cnn_15_                    ✓                                        
    02_binary_cnn_16_                    ✓                                        
    06_auto_encoder_17_                  ✓                                        
    06_auto_encoder_18_                  ✓                                        
    10_deep_mlp_13_                      ✓                                        
    10_deep_mlp_14_                      ✓                                        
    11_deconvolution_11_                 ✓                                        
    11_deconvolution_12_                 ✓                                        
    12_residual_learning_19_             ✓                                        
    12_residual_learning_20_             ✓                                        
    LSTM_auto_encoder_23_                ✓                                        
    LSTM_auto_encoder_24_                ✓                                        
    LeNet_35_                            ✓                                        
    LeNet_36_                            ✓                                        
    bidirectional_elman_net_25_          ✓                                        
    bidirectional_elman_net_26_          ✓                                        
    binary_connect_mnist_LeNet_5_        ✓                                        
    binary_connect_mnist_MLP_8_          ✓                                        
    binary_net_mnist_LeNet_7_            ✓                                        
    binary_net_mnist_MLP_4_              ✓                                        
    binary_weight_mnist_MLP_6_           ✓                                        
    elman_net_21_                        ✓                                        
    elman_net_22_                        ✓                                        
    elman_net_with_attention_33_         ✓                                        
    elman_net_with_attention_34_         ✓                                        
    gated_recurrent_unitGRU_31_          ✓                                        
    gated_recurrent_unitGRU_32_          ✓                                        
    long_short_term_memoryLSTM_29_       ✓                                        
    long_short_term_memoryLSTM_30_       ✓                                        
    mnist_dcgan_with_label_1_            X     Failed to infer by nnabla.         
    mnist_dcgan_with_label_2_            X     Failed to compare inferring result.
    mnist_vae_3_                         ✓                                        
    semi_supervised_learning_VAT_37_     X     Failed to compare inferring result.
    stacked_GRU_27_                      ✓                                        
    stacked_GRU_28_                      ✓                                        
    ================================  =======  ===================================





.. _shufflenet_model: https://media.githubusercontent.com/media/onnx/models/master/vision/classification/shufflenet/model/shufflenet-9.tar.gz
.. _bvlc_reference_rcnn_ilsvrc13_model: https://media.githubusercontent.com/media/onnx/models/master/vision/classification/rcnn_ilsvrc13/model/rcnn-ilsvrc13-9.tar.gz
.. _inception_v1_model: https://media.githubusercontent.com/media/onnx/models/master/vision/classification/inception_and_googlenet/inception_v1/model/inception-v1-9.tar.gz
.. _inception_v2_model: https://media.githubusercontent.com/media/onnx/models/master/vision/classification/inception_and_googlenet/inception_v2/model/inception-v2-9.tar.gz
.. _bvlc_reference_caffenet_model: https://media.githubusercontent.com/media/onnx/models/master/vision/classification/caffenet/model/caffenet-9.tar.gz
.. _bvlc_alexnet_model: https://media.githubusercontent.com/media/onnx/models/master/vision/classification/alexnet/model/bvlcalexnet-9.tar.gz
.. _resnet50_model: https://media.githubusercontent.com/media/onnx/models/master/vision/classification/resnet/model/resnet50-caffe2-v1-9.tar.gz
.. _zfnet512_model: https://media.githubusercontent.com/media/onnx/models/master/vision/classification/zfnet-512/model/zfnet512-9.tar.gz
.. _densenet121_model: https://media.githubusercontent.com/media/onnx/models/master/vision/classification/densenet-121/model/densenet-9.tar.gz
.. _bvlc_googlenet_model: https://media.githubusercontent.com/media/onnx/models/master/vision/classification/inception_and_googlenet/googlenet/model/googlenet-9.tar.gz
.. _vgg19_model: https://media.githubusercontent.com/media/onnx/models/master/vision/classification/vgg/model/vgg19-caffe2-9.tar.gz
.. _squeezenet_model: https://media.githubusercontent.com/media/onnx/models/master/vision/classification/squeezenet/model/squeezenet1.0-9.tar.gz
.. _binary_connect_mnist_LeNet_5: https://dl.sony.com/assets/sdcproj/tutorial/binary_networks/binary_connect_mnist_LeNet.sdcproj
.. _elman_net_with_attention_34: https://dl.sony.com/assets/sdcproj/tutorial/recurrent_neural_networks/elman_net_with_attention.sdcproj
.. _LSTM_auto_encoder_23: https://dl.sony.com/assets/sdcproj/tutorial/recurrent_neural_networks/LSTM_auto_encoder.sdcproj
.. _10_deep_mlp_14: https://dl.sony.com/assets/sdcproj/tutorial/basics/10_deep_mlp.sdcproj
.. _12_residual_learning_19: https://dl.sony.com/assets/sdcproj/tutorial/basics/12_residual_learning.sdcproj
.. _binary_weight_mnist_MLP_6: https://dl.sony.com/assets/sdcproj/tutorial/binary_networks/binary_weight_mnist_MLP.sdcproj
.. _LeNet_35: https://dl.sony.com/assets/sdcproj/image_recognition/MNIST/LeNet.sdcproj
.. _06_auto_encoder_18: https://dl.sony.com/assets/sdcproj/tutorial/basics/06_auto_encoder.sdcproj
.. _LSTM_auto_encoder_24: https://dl.sony.com/assets/sdcproj/tutorial/recurrent_neural_networks/LSTM_auto_encoder.sdcproj
.. _elman_net_with_attention_33: https://dl.sony.com/assets/sdcproj/tutorial/recurrent_neural_networks/elman_net_with_attention.sdcproj
.. _binary_connect_mnist_MLP_8: https://dl.sony.com/assets/sdcproj/tutorial/binary_networks/binary_connect_mnist_MLP.sdcproj
.. _bidirectional_elman_net_26: https://dl.sony.com/assets/sdcproj/tutorial/recurrent_neural_networks/bidirectional_elman_net.sdcproj
.. _mnist_vae_3: https://dl.sony.com/assets/sdcproj/image_generation/mnist_vae.sdcproj
.. _02_binary_cnn_16: https://dl.sony.com/assets/sdcproj/tutorial/basics/02_binary_cnn.sdcproj
.. _01_logistic_regression_9: https://dl.sony.com/assets/sdcproj/tutorial/basics/01_logistic_regression.sdcproj
.. _10_deep_mlp_13: https://dl.sony.com/assets/sdcproj/tutorial/basics/10_deep_mlp.sdcproj
.. _LeNet_36: https://dl.sony.com/assets/sdcproj/image_recognition/MNIST/LeNet.sdcproj
.. _12_residual_learning_20: https://dl.sony.com/assets/sdcproj/tutorial/basics/12_residual_learning.sdcproj
.. _elman_net_22: https://dl.sony.com/assets/sdcproj/tutorial/recurrent_neural_networks/elman_net.sdcproj
.. _binary_net_mnist_LeNet_7: https://dl.sony.com/assets/sdcproj/tutorial/binary_networks/binary_net_mnist_LeNet.sdcproj
.. _bidirectional_elman_net_25: https://dl.sony.com/assets/sdcproj/tutorial/recurrent_neural_networks/bidirectional_elman_net.sdcproj
.. _11_deconvolution_11: https://dl.sony.com/assets/sdcproj/tutorial/basics/11_deconvolution.sdcproj
.. _long_short_term_memoryLSTM_30: https://dl.sony.com/assets/sdcproj/tutorial/recurrent_neural_networks/long_short_term_memory(LSTM).sdcproj
.. _gated_recurrent_unitGRU_32: https://dl.sony.com/assets/sdcproj/tutorial/recurrent_neural_networks/gated_recurrent_unit(GRU).sdcproj
.. _binary_net_mnist_MLP_4: https://dl.sony.com/assets/sdcproj/tutorial/binary_networks/binary_net_mnist_MLP.sdcproj
.. _stacked_GRU_28: https://dl.sony.com/assets/sdcproj/tutorial/recurrent_neural_networks/stacked_GRU.sdcproj
.. _gated_recurrent_unitGRU_31: https://dl.sony.com/assets/sdcproj/tutorial/recurrent_neural_networks/gated_recurrent_unit(GRU).sdcproj
.. _elman_net_21: https://dl.sony.com/assets/sdcproj/tutorial/recurrent_neural_networks/elman_net.sdcproj
.. _02_binary_cnn_15: https://dl.sony.com/assets/sdcproj/tutorial/basics/02_binary_cnn.sdcproj
.. _stacked_GRU_27: https://dl.sony.com/assets/sdcproj/tutorial/recurrent_neural_networks/stacked_GRU.sdcproj
.. _mnist_dcgan_with_label_2: https://dl.sony.com/assets/sdcproj/image_generation/mnist_dcgan_with_label.sdcproj
.. _semi_supervised_learning_VAT_37: https://dl.sony.com/assets/sdcproj/image_recognition/MNIST/semi_supervised_learning_VAT.sdcproj
.. _11_deconvolution_12: https://dl.sony.com/assets/sdcproj/tutorial/basics/11_deconvolution.sdcproj
.. _06_auto_encoder_17: https://dl.sony.com/assets/sdcproj/tutorial/basics/06_auto_encoder.sdcproj
.. _long_short_term_memoryLSTM_29: https://dl.sony.com/assets/sdcproj/tutorial/recurrent_neural_networks/long_short_term_memory(LSTM).sdcproj
.. _mnist_dcgan_with_label_1: https://dl.sony.com/assets/sdcproj/image_generation/mnist_dcgan_with_label.sdcproj
.. _01_logistic_regression_10: https://dl.sony.com/assets/sdcproj/tutorial/basics/01_logistic_regression.sdcproj
.. _Resnet-50_4_178: https://nnabla.org/pretrained-models/nnp_models/imagenet/Resnet-50/Resnet-50.nnp
.. _Resnet-152_4_518: https://nnabla.org/pretrained-models/nnp_models/imagenet/Resnet-152/Resnet-152.nnp
.. _NIN: https://nnabla.org/pretrained-models/nnp_models/imagenet/NIN/NIN.nnp
.. _Xception: https://nnabla.org/pretrained-models/nnp_models/imagenet/Xception/Xception.nnp
.. _GoogLeNet_4_142: https://nnabla.org/pretrained-models/nnp_models/imagenet/GoogLeNet/GoogLeNet.nnp
.. _VGG-11: https://nnabla.org/pretrained-models/nnp_models/imagenet/VGG-11/VGG-11.nnp
.. _ShuffleNet_2_202: https://nnabla.org/pretrained-models/nnp_models/imagenet/ShuffleNet-2.0x/ShuffleNet-2.0x.nnp
.. _SqueezeNet-1.1_2_70: https://nnabla.org/pretrained-models/nnp_models/imagenet/SqueezeNet-1.1/SqueezeNet-1.1.nnp
.. _MobileNet_1_86: https://nnabla.org/pretrained-models/nnp_models/imagenet/MobileNet/MobileNet.nnp
.. _DenseNet-161_2_570: https://nnabla.org/pretrained-models/nnp_models/imagenet/DenseNet-161/DenseNet-161.nnp
.. _ShuffleNet-0.5x_2_202: https://nnabla.org/pretrained-models/nnp_models/imagenet/ShuffleNet-0.5x/ShuffleNet-0.5x.nnp
.. _VGG-13: https://nnabla.org/pretrained-models/nnp_models/imagenet/VGG-13/VGG-13.nnp
.. _Resnet-18_3_71: https://nnabla.org/pretrained-models/nnp_models/imagenet/Resnet-18/Resnet-18.nnp
.. _AlexNet: https://notfound
.. _Resnet-101_4_348: https://nnabla.org/pretrained-models/nnp_models/imagenet/Resnet-101/Resnet-101.nnp
.. _SqueezeNet-1.0_2_70: https://nnabla.org/pretrained-models/nnp_models/imagenet/SqueezeNet-1.0/SqueezeNet-1.0.nnp
.. _VGG-16: https://nnabla.org/pretrained-models/nnp_models/imagenet/VGG-16/VGG-16.nnp
.. _Resnet-34_4_128: https://nnabla.org/pretrained-models/nnp_models/imagenet/Resnet-34/Resnet-34.nnp
.. _siamese_embedding: https://github.com/sony/nnabla-examples
.. _word_embedding: https://github.com/sony/nnabla-examples
.. _deeplabv3plus: https://github.com/sony/nnabla-examples
.. _yolov2: https://github.com/sony/nnabla-examples
.. _classification: https://github.com/sony/nnabla-examples
.. _meta_learning: https://github.com/sony/nnabla-examples
.. _wavenet: https://github.com/sony/nnabla-examples
.. _pix2pix: https://github.com/sony/nnabla-examples
.. _capsules: https://github.com/sony/nnabla-examples
.. _cycle_gan: https://github.com/sony/nnabla-examples
.. _ZFNet: https://notfound
.. _LeNet_5: https://notfound
.. _ResNet50: https://notfound
.. _GoogLeNet: https://notfound
.. _AlexNet: https://notfound
.. _VGG16: https://notfound
.. _mobilenet_v1_0.75_192: https://storage.googleapis.com/download.tensorflow.org/models/mobilenet_v1_0.75_192_frozen.tgz
.. _inception_v4_2016_09_09_frozen: https://storage.googleapis.com/download.tensorflow.org/models/inception_v4_2016_09_09_frozen.pb.tar.gz
.. _conv-layers_frozen: https://github.com/onnx/tensorflow-onnx/blob/master/tests/models/conv-layers/frozen.pb
.. _mobilenet_v1_1.0_224: https://storage.googleapis.com/download.tensorflow.org/models/mobilenet_v1_1.0_224_frozen.tgz
.. _ae0_frozen: https://github.com/onnx/tensorflow-onnx/blob/master/tests/models/ae0/frozen.pb
.. _inception_v1_2016_08_28_frozen: https://storage.googleapis.com/download.tensorflow.org/models/inception_v1_2016_08_28_frozen.pb.tar.gz
.. _fc-layers_frozen: https://github.com/onnx/tensorflow-onnx/blob/master/tests/models/fc-layers/frozen.pb
.. _inception_v3_2016_08_28_frozen: https://storage.googleapis.com/download.tensorflow.org/models/inception_v3_2016_08_28_frozen.pb.tar.gz
.. _inception_resnet_v2_2016_08_30_frozen: https://storage.googleapis.com/download.tensorflow.org/models/inception_resnet_v2_2016_08_30_frozen.pb.tar.gz
.. _lstm_frozen: https://github.com/onnx/tensorflow-onnx/blob/master/tests/models/lstm/frozen.pb
