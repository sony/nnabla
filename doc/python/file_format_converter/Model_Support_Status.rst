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
    inception_v2_model_                     ✓                                                               
    inception_v1_model_                     X     The `edge` mode of the `pad` in nnabla is not implemented.
    zfnet512_model_                         ✓                                                               
    bvlc_reference_rcnn_ilsvrc13_model_     ✓                                                               
    densenet121_model_                      ✓                                                               
    shufflenet_model_                       ✓                                                               
    bvlc_alexnet_model_                     ✓                                                               
    vgg19_model_                            ✓                                                               
    bvlc_reference_caffenet_model_          ✓                                                               
    resnet50_model_                         ✓                                                               
    squeezenet_model_                       ✓                                                               
    bvlc_googlenet_model_                   ✓                                                               
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
    10_deep_mlp_13_                      ✓                                                            
    10_deep_mlp_14_                      ✓                                                            
    12_residual_learning_20_             ✓                                                            
    06_auto_encoder_17_                  ✓                                                            
    binary_connect_mnist_LeNet_5_        ✓                                                            
    elman_net_with_attention_33_         ✓                                                            
    11_deconvolution_12_                 ✓                                                            
    LeNet_36_                            ✓                                                            
    01_logistic_regression_10_           ✓                                                            
    binary_weight_mnist_MLP_6_           ✓                                                            
    binary_net_mnist_MLP_4_              ✓                                                            
    bidirectional_elman_net_25_          ✓                                                            
    gated_recurrent_unitGRU_32_          ✓                                                            
    mnist_vae_3_                         ✓                                                            
    stacked_GRU_27_                      ✓                                                            
    bidirectional_elman_net_26_          ✓                                                            
    12_residual_learning_19_             ✓                                                            
    mnist_dcgan_with_label_1_            X     NNabla converter error, will be fixed in the future.   
    semi_supervised_learning_VAT_37_     X     NNP with only a single executor is currently supported.
    elman_net_22_                        ✓                                                            
    gated_recurrent_unitGRU_31_          ✓                                                            
    elman_net_with_attention_34_         ✓                                                            
    11_deconvolution_11_                 ✓                                                            
    binary_net_mnist_LeNet_7_            ✓                                                            
    mnist_dcgan_with_label_2_            X     NNabla converter error, will be fixed in the future.   
    01_logistic_regression_9_            ✓                                                            
    02_binary_cnn_15_                    ✓                                                            
    02_binary_cnn_16_                    ✓                                                            
    LSTM_auto_encoder_24_                ✓                                                            
    LeNet_35_                            ✓                                                            
    LSTM_auto_encoder_23_                ✓                                                            
    binary_connect_mnist_MLP_8_          ✓                                                            
    long_short_term_memoryLSTM_30_       ✓                                                            
    06_auto_encoder_18_                  ✓                                                            
    stacked_GRU_28_                      ✓                                                            
    elman_net_21_                        ✓                                                            
    long_short_term_memoryLSTM_29_       ✓                                                            
    ================================  =======  =======================================================


ONNX Export Pretrained Model Test(nnp --> onnx)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Count 18/18
 

    ======================  =======  ====
             Name           Support  Memo
    ======================  =======  ====
    ShuffleNet_2_202_          ✓         
    VGG-11_                    ✓         
    DenseNet-161_2_570_        ✓         
    SqueezeNet-1.0_2_70_       ✓         
    SqueezeNet-1.1_2_70_       ✓         
    Resnet-18_3_71_            ✓         
    VGG-16_                    ✓         
    Resnet-50_4_178_           ✓         
    Xception_                  ✓         
    ShuffleNet-0.5x_2_202_     ✓         
    Resnet-34_4_128_           ✓         
    Resnet-152_4_518_          ✓         
    MobileNet_1_86_            ✓         
    Resnet-101_4_348_          ✓         
    GoogLeNet_4_142_           ✓         
    AlexNet_                   ✓         
    NIN_                       ✓         
    VGG-13_                    ✓         
    ======================  =======  ====


ONNX Export Example Model Test(nnp --> onnx)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Count 6/10
 

    ==================  =======  ====================================================
           Name         Support                          Memo                        
    ==================  =======  ====================================================
    pix2pix_               ✓                                                         
    yolov2_                ✓                                                         
    classification_        ✓                                                         
    capsules_              X     NNabla converter error, will be fixed in the future.
    cycle_gan_             ✓                                                         
    siamese_embedding_     ✓                                                         
    meta_learning_         X     Failed to compare inferring result.                 
    wavenet_               X     The `onehot` dimension != 2 is not supported.       
    deeplabv3plus_         X     The `Interpolate` is currently not supported.       
    word_embedding_        ✓                                                         
    ==================  =======  ====================================================





Tensorflow Support Status
=========================

Import
------

- ✓: Support to convert
- X: Not support

Total: 3/10

Tensorflow Import Sample Test(pb --> nnp)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Count 3/10
 

    ======================================  =======  ===============================================================================
                     Name                   Support                                       Memo                                      
    ======================================  =======  ===============================================================================
    inception_resnet_v2_2016_08_30_frozen_     X     Failed to convert from tensorflow to onnx, `Bias` should be 1D, but actual n-D.
    inception_v4_2016_09_09_frozen_            ✓                                                                                    
    ae0_frozen_                                ✓                                                                                    
    lstm_frozen_                               X     The `Shape` is currently not supported to convert by nnabla.                   
    inception_v1_2016_08_28_frozen_            X     the `edge` mode of the `pad` in nnabla is not implemented.                     
    mobilenet_v1_1.0_224_                      X     The `edge` mode of the pad function in nnabla is not implemented.              
    fc-layers_frozen_                          ✓                                                                                    
    mobilenet_v1_0.75_192_                     X     The `edge` mode of the pad function in nnabla is not implemented.              
    conv-layers_frozen_                        X     Failed to convert from tensorflow to onnx, `Bias` should be 1D, but actual n-D.
    inception_v3_2016_08_28_frozen_            X     Failed to convert from tensorflow to onnx, `Bias` should be 1D, but actual n-D.
    ======================================  =======  ===============================================================================





Export
------

- ✓: Support to convert
- X: Not support

Total: 58/65

Tensorflow Export Sample Test(nnp --> pb)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Count 34/37
 

    ================================  =======  =======================================================
                  Name                Support                           Memo                          
    ================================  =======  =======================================================
    10_deep_mlp_13_                      ✓                                                            
    10_deep_mlp_14_                      ✓                                                            
    12_residual_learning_20_             ✓                                                            
    06_auto_encoder_17_                  ✓                                                            
    binary_connect_mnist_LeNet_5_        ✓                                                            
    elman_net_with_attention_33_         ✓                                                            
    11_deconvolution_12_                 ✓                                                            
    LeNet_36_                            ✓                                                            
    01_logistic_regression_10_           ✓                                                            
    binary_weight_mnist_MLP_6_           ✓                                                            
    binary_net_mnist_MLP_4_              ✓                                                            
    bidirectional_elman_net_25_          ✓                                                            
    gated_recurrent_unitGRU_32_          ✓                                                            
    mnist_vae_3_                         ✓                                                            
    stacked_GRU_27_                      ✓                                                            
    bidirectional_elman_net_26_          ✓                                                            
    12_residual_learning_19_             ✓                                                            
    mnist_dcgan_with_label_1_            X     NNabla converter error, will be fixed in the future.   
    semi_supervised_learning_VAT_37_     X     NNP with only a single executor is currently supported.
    elman_net_22_                        ✓                                                            
    gated_recurrent_unitGRU_31_          ✓                                                            
    elman_net_with_attention_34_         ✓                                                            
    11_deconvolution_11_                 ✓                                                            
    binary_net_mnist_LeNet_7_            ✓                                                            
    mnist_dcgan_with_label_2_            X     NNabla converter error, will be fixed in the future.   
    01_logistic_regression_9_            ✓                                                            
    02_binary_cnn_15_                    ✓                                                            
    02_binary_cnn_16_                    ✓                                                            
    LSTM_auto_encoder_24_                ✓                                                            
    LeNet_35_                            ✓                                                            
    LSTM_auto_encoder_23_                ✓                                                            
    binary_connect_mnist_MLP_8_          ✓                                                            
    long_short_term_memoryLSTM_30_       ✓                                                            
    06_auto_encoder_18_                  ✓                                                            
    stacked_GRU_28_                      ✓                                                            
    elman_net_21_                        ✓                                                            
    long_short_term_memoryLSTM_29_       ✓                                                            
    ================================  =======  =======================================================


Tensorflow Export Pretrained Models(nnp --> pb)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Count 18/18
 

    ======================  =======  ====
             Name           Support  Memo
    ======================  =======  ====
    ShuffleNet_2_202_          ✓         
    VGG-11_                    ✓         
    DenseNet-161_2_570_        ✓         
    SqueezeNet-1.0_2_70_       ✓         
    SqueezeNet-1.1_2_70_       ✓         
    Resnet-18_3_71_            ✓         
    VGG-16_                    ✓         
    Resnet-50_4_178_           ✓         
    Xception_                  ✓         
    ShuffleNet-0.5x_2_202_     ✓         
    Resnet-34_4_128_           ✓         
    Resnet-152_4_518_          ✓         
    MobileNet_1_86_            ✓         
    Resnet-101_4_348_          ✓         
    GoogLeNet_4_142_           ✓         
    AlexNet_                   ✓         
    NIN_                       ✓         
    VGG-13_                    ✓         
    ======================  =======  ====


Tensorflow Export Example Models(nnp --> pb)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Count 6/10
 

    ==================  =======  ====================================================
           Name         Support                          Memo                        
    ==================  =======  ====================================================
    pix2pix_               ✓                                                         
    yolov2_                ✓                                                         
    classification_        ✓                                                         
    capsules_              X     NNabla converter error, will be fixed in the future.
    cycle_gan_             ✓                                                         
    siamese_embedding_     ✓                                                         
    meta_learning_         X     Failed to compare inferring result.                 
    wavenet_               X     The `onehot` dimension != 2 is not supported.       
    deeplabv3plus_         X     The `Interpolate` is currently not supported.       
    word_embedding_        ✓                                                         
    ==================  =======  ====================================================



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
    10_deep_mlp_13_                      ✓                                        
    10_deep_mlp_14_                      ✓                                        
    12_residual_learning_20_             ✓                                        
    06_auto_encoder_17_                  ✓                                        
    binary_connect_mnist_LeNet_5_        ✓                                        
    elman_net_with_attention_33_         ✓                                        
    11_deconvolution_12_                 ✓                                        
    LeNet_36_                            ✓                                        
    01_logistic_regression_10_           ✓                                        
    binary_weight_mnist_MLP_6_           ✓                                        
    binary_net_mnist_MLP_4_              ✓                                        
    bidirectional_elman_net_25_          ✓                                        
    gated_recurrent_unitGRU_32_          ✓                                        
    mnist_vae_3_                         ✓                                        
    stacked_GRU_27_                      ✓                                        
    bidirectional_elman_net_26_          ✓                                        
    12_residual_learning_19_             ✓                                        
    mnist_dcgan_with_label_1_            X     Failed to infer by nnabla.         
    semi_supervised_learning_VAT_37_     X     Failed to compare inferring result.
    elman_net_22_                        ✓                                        
    gated_recurrent_unitGRU_31_          ✓                                        
    elman_net_with_attention_34_         ✓                                        
    11_deconvolution_11_                 ✓                                        
    binary_net_mnist_LeNet_7_            ✓                                        
    mnist_dcgan_with_label_2_            X     Failed to compare inferring result.
    01_logistic_regression_9_            ✓                                        
    02_binary_cnn_15_                    ✓                                        
    02_binary_cnn_16_                    ✓                                        
    LSTM_auto_encoder_24_                ✓                                        
    LeNet_35_                            ✓                                        
    LSTM_auto_encoder_23_                ✓                                        
    binary_connect_mnist_MLP_8_          ✓                                        
    long_short_term_memoryLSTM_30_       ✓                                        
    06_auto_encoder_18_                  ✓                                        
    stacked_GRU_28_                      ✓                                        
    elman_net_21_                        ✓                                        
    long_short_term_memoryLSTM_29_       ✓                                        
    ================================  =======  ===================================





.. _inception_v2_model: https://s3.amazonaws.com/download.onnx/models/opset_6/inception_v2.tar.gz
.. _inception_v1_model: https://s3.amazonaws.com/download.onnx/models/opset_6/inception_v1.tar.gz
.. _zfnet512_model: https://s3.amazonaws.com/download.onnx/models/opset_6/zfnet512.tar.gz
.. _bvlc_reference_rcnn_ilsvrc13_model: https://s3.amazonaws.com/download.onnx/models/opset_6/bvlc_reference_rcnn_ilsvrc13.tar.gz
.. _densenet121_model: https://s3.amazonaws.com/download.onnx/models/opset_6/densenet121.tar.gz
.. _shufflenet_model: https://s3.amazonaws.com/download.onnx/models/opset_6/shufflenet.tar.gz
.. _bvlc_alexnet_model: https://s3.amazonaws.com/download.onnx/models/opset_6/bvlc_alexnet.tar.gz
.. _vgg19_model: https://s3.amazonaws.com/download.onnx/models/opset_6/vgg19.tar.gz
.. _bvlc_reference_caffenet_model: https://s3.amazonaws.com/download.onnx/models/opset_6/bvlc_reference_caffenet.tar.gz
.. _resnet50_model: https://s3.amazonaws.com/download.onnx/models/opset_6/resnet50.tar.gz
.. _squeezenet_model: https://s3.amazonaws.com/download.onnx/models/opset_6/squeezenet.tar.gz
.. _bvlc_googlenet_model: https://s3.amazonaws.com/download.onnx/models/opset_6/bvlc_googlenet.tar.gz
.. _10_deep_mlp_13: https://dl.sony.com/assets/sdcproj/tutorial/basics/10_deep_mlp.sdcproj
.. _10_deep_mlp_14: https://dl.sony.com/assets/sdcproj/tutorial/basics/10_deep_mlp.sdcproj
.. _12_residual_learning_20: https://dl.sony.com/assets/sdcproj/tutorial/basics/12_residual_learning.sdcproj
.. _06_auto_encoder_17: https://dl.sony.com/assets/sdcproj/tutorial/basics/06_auto_encoder.sdcproj
.. _binary_connect_mnist_LeNet_5: https://dl.sony.com/assets/sdcproj/tutorial/binary_networks/binary_connect_mnist_LeNet.sdcproj
.. _elman_net_with_attention_33: https://dl.sony.com/assets/sdcproj/tutorial/recurrent_neural_networks/elman_net_with_attention.sdcproj
.. _11_deconvolution_12: https://dl.sony.com/assets/sdcproj/tutorial/basics/11_deconvolution.sdcproj
.. _LeNet_36: https://dl.sony.com/assets/sdcproj/image_recognition/MNIST/LeNet.sdcproj
.. _01_logistic_regression_10: https://dl.sony.com/assets/sdcproj/tutorial/basics/01_logistic_regression.sdcproj
.. _binary_weight_mnist_MLP_6: https://dl.sony.com/assets/sdcproj/tutorial/binary_networks/binary_weight_mnist_MLP.sdcproj
.. _binary_net_mnist_MLP_4: https://dl.sony.com/assets/sdcproj/tutorial/binary_networks/binary_net_mnist_MLP.sdcproj
.. _bidirectional_elman_net_25: https://dl.sony.com/assets/sdcproj/tutorial/recurrent_neural_networks/bidirectional_elman_net.sdcproj
.. _gated_recurrent_unitGRU_32: https://dl.sony.com/assets/sdcproj/tutorial/recurrent_neural_networks/gated_recurrent_unit(GRU).sdcproj
.. _mnist_vae_3: https://dl.sony.com/assets/sdcproj/image_generation/mnist_vae.sdcproj
.. _stacked_GRU_27: https://dl.sony.com/assets/sdcproj/tutorial/recurrent_neural_networks/stacked_GRU.sdcproj
.. _bidirectional_elman_net_26: https://dl.sony.com/assets/sdcproj/tutorial/recurrent_neural_networks/bidirectional_elman_net.sdcproj
.. _12_residual_learning_19: https://dl.sony.com/assets/sdcproj/tutorial/basics/12_residual_learning.sdcproj
.. _mnist_dcgan_with_label_1: https://dl.sony.com/assets/sdcproj/image_generation/mnist_dcgan_with_label.sdcproj
.. _semi_supervised_learning_VAT_37: https://dl.sony.com/assets/sdcproj/image_recognition/MNIST/semi_supervised_learning_VAT.sdcproj
.. _elman_net_22: https://dl.sony.com/assets/sdcproj/tutorial/recurrent_neural_networks/elman_net.sdcproj
.. _gated_recurrent_unitGRU_31: https://dl.sony.com/assets/sdcproj/tutorial/recurrent_neural_networks/gated_recurrent_unit(GRU).sdcproj
.. _elman_net_with_attention_34: https://dl.sony.com/assets/sdcproj/tutorial/recurrent_neural_networks/elman_net_with_attention.sdcproj
.. _11_deconvolution_11: https://dl.sony.com/assets/sdcproj/tutorial/basics/11_deconvolution.sdcproj
.. _binary_net_mnist_LeNet_7: https://dl.sony.com/assets/sdcproj/tutorial/binary_networks/binary_net_mnist_LeNet.sdcproj
.. _mnist_dcgan_with_label_2: https://dl.sony.com/assets/sdcproj/image_generation/mnist_dcgan_with_label.sdcproj
.. _01_logistic_regression_9: https://dl.sony.com/assets/sdcproj/tutorial/basics/01_logistic_regression.sdcproj
.. _02_binary_cnn_15: https://dl.sony.com/assets/sdcproj/tutorial/basics/02_binary_cnn.sdcproj
.. _02_binary_cnn_16: https://dl.sony.com/assets/sdcproj/tutorial/basics/02_binary_cnn.sdcproj
.. _LSTM_auto_encoder_24: https://dl.sony.com/assets/sdcproj/tutorial/recurrent_neural_networks/LSTM_auto_encoder.sdcproj
.. _LeNet_35: https://dl.sony.com/assets/sdcproj/image_recognition/MNIST/LeNet.sdcproj
.. _LSTM_auto_encoder_23: https://dl.sony.com/assets/sdcproj/tutorial/recurrent_neural_networks/LSTM_auto_encoder.sdcproj
.. _binary_connect_mnist_MLP_8: https://dl.sony.com/assets/sdcproj/tutorial/binary_networks/binary_connect_mnist_MLP.sdcproj
.. _long_short_term_memoryLSTM_30: https://dl.sony.com/assets/sdcproj/tutorial/recurrent_neural_networks/long_short_term_memory(LSTM).sdcproj
.. _06_auto_encoder_18: https://dl.sony.com/assets/sdcproj/tutorial/basics/06_auto_encoder.sdcproj
.. _stacked_GRU_28: https://dl.sony.com/assets/sdcproj/tutorial/recurrent_neural_networks/stacked_GRU.sdcproj
.. _elman_net_21: https://dl.sony.com/assets/sdcproj/tutorial/recurrent_neural_networks/elman_net.sdcproj
.. _long_short_term_memoryLSTM_29: https://dl.sony.com/assets/sdcproj/tutorial/recurrent_neural_networks/long_short_term_memory(LSTM).sdcproj
.. _ShuffleNet_2_202: https://nnabla.org/pretrained-models/nnp_models/imagenet/ShuffleNet-2.0x/ShuffleNet-2.0x.nnp
.. _VGG-11: https://nnabla.org/pretrained-models/nnp_models/imagenet/VGG-11/VGG-11.nnp
.. _DenseNet-161_2_570: https://nnabla.org/pretrained-models/nnp_models/imagenet/DenseNet-161/DenseNet-161.nnp
.. _SqueezeNet-1.0_2_70: https://nnabla.org/pretrained-models/nnp_models/imagenet/SqueezeNet-1.0/SqueezeNet-1.0.nnp
.. _SqueezeNet-1.1_2_70: https://nnabla.org/pretrained-models/nnp_models/imagenet/SqueezeNet-1.1/SqueezeNet-1.1.nnp
.. _Resnet-18_3_71: https://nnabla.org/pretrained-models/nnp_models/imagenet/Resnet-18/Resnet-18.nnp
.. _VGG-16: https://nnabla.org/pretrained-models/nnp_models/imagenet/VGG-16/VGG-16.nnp
.. _Resnet-50_4_178: https://nnabla.org/pretrained-models/nnp_models/imagenet/Resnet-50/Resnet-50.nnp
.. _Xception: https://nnabla.org/pretrained-models/nnp_models/imagenet/Xception/Xception.nnp
.. _ShuffleNet-0.5x_2_202: https://nnabla.org/pretrained-models/nnp_models/imagenet/ShuffleNet-0.5x/ShuffleNet-0.5x.nnp
.. _Resnet-34_4_128: https://nnabla.org/pretrained-models/nnp_models/imagenet/Resnet-34/Resnet-34.nnp
.. _Resnet-152_4_518: https://nnabla.org/pretrained-models/nnp_models/imagenet/Resnet-152/Resnet-152.nnp
.. _MobileNet_1_86: https://nnabla.org/pretrained-models/nnp_models/imagenet/MobileNet/MobileNet.nnp
.. _Resnet-101_4_348: https://nnabla.org/pretrained-models/nnp_models/imagenet/Resnet-101/Resnet-101.nnp
.. _GoogLeNet_4_142: https://nnabla.org/pretrained-models/nnp_models/imagenet/GoogLeNet/GoogLeNet.nnp
.. _AlexNet: https://notfound
.. _NIN: https://nnabla.org/pretrained-models/nnp_models/imagenet/NIN/NIN.nnp
.. _VGG-13: https://nnabla.org/pretrained-models/nnp_models/imagenet/VGG-13/VGG-13.nnp
.. _pix2pix: https://github.com/sony/nnabla-examples
.. _yolov2: https://github.com/sony/nnabla-examples
.. _classification: https://github.com/sony/nnabla-examples
.. _capsules: https://github.com/sony/nnabla-examples
.. _cycle_gan: https://github.com/sony/nnabla-examples
.. _siamese_embedding: https://github.com/sony/nnabla-examples
.. _meta_learning: https://github.com/sony/nnabla-examples
.. _wavenet: https://github.com/sony/nnabla-examples
.. _deeplabv3plus: https://github.com/sony/nnabla-examples
.. _word_embedding: https://github.com/sony/nnabla-examples
.. _inception_resnet_v2_2016_08_30_frozen: https://storage.googleapis.com/download.tensorflow.org/models/inception_resnet_v2_2016_08_30_frozen.pb.tar.gz
.. _inception_v4_2016_09_09_frozen: https://storage.googleapis.com/download.tensorflow.org/models/inception_v4_2016_09_09_frozen.pb.tar.gz
.. _ae0_frozen: https://github.com/onnx/tensorflow-onnx/blob/master/tests/models/ae0/frozen.pb
.. _lstm_frozen: https://github.com/onnx/tensorflow-onnx/blob/master/tests/models/lstm/frozen.pb
.. _inception_v1_2016_08_28_frozen: https://storage.googleapis.com/download.tensorflow.org/models/inception_v1_2016_08_28_frozen.pb.tar.gz
.. _mobilenet_v1_1.0_224: https://storage.googleapis.com/download.tensorflow.org/models/mobilenet_v1_1.0_224_frozen.tgz
.. _fc-layers_frozen: https://github.com/onnx/tensorflow-onnx/blob/master/tests/models/fc-layers/frozen.pb
.. _mobilenet_v1_0.75_192: https://storage.googleapis.com/download.tensorflow.org/models/mobilenet_v1_0.75_192_frozen.tgz
.. _conv-layers_frozen: https://github.com/onnx/tensorflow-onnx/blob/master/tests/models/conv-layers/frozen.pb
.. _inception_v3_2016_08_28_frozen: https://storage.googleapis.com/download.tensorflow.org/models/inception_v3_2016_08_28_frozen.pb.tar.gz
