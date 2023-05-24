
Quantization-Aware-Training Tutorial
========================================

What is model Quantization-Aware-Training
--------------------------
In general, the weights and the activation of artificial neural networks are represented by float32; on the other hand, model quantization means use lower precision to represent numbers, such as float16, int8 and uint8.

By reducing the precision we can reduce model size and memory occupancy. What's more, on some devices we can also shorten the inference time.

However, using lower precision instead of float32 will introduces quantization error to the model. Quantization-Aware-Training (QAT) mitigates the quantization errors by simulating quantization effect at training time.

Quantization-Aware-Training with NNabla
-----------------------------------
In nnabla, we divide QAT into two stages, RERORDING and TRAINING.
In RECORDING stage, we will collect and record the dynamic range of each parameter and buffer.
In TRAINING stage, we will insert Quantization&Dequantization node to simulate the quantization effect.

We provide QATScheduler to support Quantization-Aware-Training.
Here is some sample code about QATScheduler.

Create QATScheduler
~~~~~~~~~~~~~~~~~~~~
Firstly, we need to create a QATScheduler. QATScheduler will convert the network automatically to make it support Quantization-Aware-Training.

.. code:: python

    from nnabla.utils.qnn import QATScheduler, QATConfig, PrecisionMode
    # Create training network
    pred = model(image, test=False)
    # Create validation network
    vpred = model(vimage, test=True)
    # configure of QATScheduler
    config = QATConfig()
    config.bn_folding = True
    config.bn_self_folding = True
    config.channel_last = False
    config.precision_mode = PrecisionMode.SIM_QNN
    config.skip_bias = True
    config.niter_to_recording = 1
    config.niter_to_training = steps_per_epoch * 2
    # Create a QATScheduler object
    qat_scheduler = QATScheduler(config=config, solver=solver)
    # register the training network to QATScheduler
    qat_scheduler(pred)
    # register the validation network to QATScheduler
    qat_scheduler(vpred, training=False)

Modify your training loop
~~~~~~~~~~~~~~~~~~~~~~~~~
In general, Your training loop should look like this:

.. code:: python

    for step in range(max_step):
        x, y = dataset.next()
        image.d, label.d = x, y
        loss.forward()
        solver.zero_grad()
        loss.backword()
        solver.weight_decay(weight_decay)
        solver.update()

Compare with the training loop above. You just need to insert a line of code.
Inside the 'step' function, it records your training step and converts the network when the step reaches 'niter_to_recording' or 'niter_to_recording'.

.. code:: python

    for step in range(max_step):
        # Run qat_scheduler step by step
        qat_scheduler.step()
        x, y = dataset.next()
        image.d, label.d = x, y
        loss.forward()
        solver.zero_grad()
        loss.backword()
        solver.weight_decay(weight_decay)
        solver.update()
    # Save the QAT model
    qat_scheduler.save('your_model.nnp', vimage, batch_size=1, deploy=False)

Performance of the quantized model
----------------------------------

================== =============== ================================= ==========================
Model              with BN-folding float32 model validation error(%) QAT model validation error(%)
================== =============== ================================= ==========================
Mobilenetv1        NO              28.05                             27.53
Resnet18           NO              29.66                             28.71
Resnet50           NO              23.46                             23.19
Mobilenetv1        YES             28.05                             27.92
Resnet18           YES             29.66                             28.63
Resnet50           YES             23.46                             23.16
================== =============== ================================= ==========================

Above is some Quantization-Aware-Training experimental results on imagenet dataset.

Deploy the model with TensorRT
----------------------------------

========= =============== ====================== ===================== ====================== =====================
with pow2 with BN-folding maximum absolute error mean absolute error   maximum relative error mean relative error
========= =============== ====================== ===================== ====================== =====================
NO        NO              0.0144                 0.0094                9.9737                 0.0907
NO        YES             4.639e-04              1.577e-04             13.3496                0.0151
YES       NO              0.0378                 0.0287                21.7627                0.0433
YES       YES             4.673e-07              3.465e-07             0.00069                2.588e-08
========= =============== ====================== ===================== ====================== =====================

You can also deploy the NNabla QAT model with other framework by using our :ref:`File_Format_Converter` to convert .nnp to other format.
Above is the comparison between the output of NNabla and the output of TensorRT. The model we used to compare is mobilenetv1.
If you want to deploy the model with TensorRT, we recommend that you enable these options in QATConfig: 'bn_folding', 'bn_self_folding', 'pow2', otherwise, the error between NNabla and TensorRT may become large. See also QATTensorRTConfig (lined).
