# Open question
# TODO: save in a nice way, the save should be of nnp format
# TODO: address the data format (since int8x32 convolution is normally fastest among large shape)
# TODO: Input data quantize, e.g, input= image do not update scale (min/max) in recording, use 1 / 255 and remove F.mul2_scalar(255)


"""
QNN-specific Functions
"""
import nnabla as nn
import nnabla.functions as F
import numpy as np
import nnabla_ext
import nnabla.experimental.graph_converters as GC

from enum import Enum
from nnabla.function import PythonFunction
from nnabla.initializer import ConstantInitializer

__round_methods__ = {
    'CEIL': np.ceil,
    'ROUND': np.round,
    'FLOOR': np.floor,
    'NOTROUND': None
}


def _param_exits(param):
    return True if nn.parameter.get_parameter(param) else False


class MinMaxMinMaxRecorder(PythonFunction):
    """
    MinMaxMinMaxRecorder records the min and max of the batch over the training iterations.
    """

    def __init__(self, ctx, training=True):
        super(MinMaxMinMaxRecorder, self).__init__(ctx)
        self.training = training

    @property
    def name(self):
        return self.__class__.__name__

    def min_outputs(self):
        return 1

    def setup_impl(self, inputs, outputs):
        assert len(inputs) == 3, "len(inputs) must be 3; data, min, max."
        x = inputs[0]
        m = inputs[1]
        M = inputs[2]
        assert m.ndim == x.ndim and M.ndim == x.ndim, \
            "ndim of min and max must be same as ndim of data."
        assert np.prod(m.shape) == 1 and np.prod(M.shape) == 1, \
            "Any dimenstion of the shape of min and max must be 1."
        y = outputs[0]
        y.reset_shape(x.shape, True)
        # inplace
        # y.data = x.data

    def forward_impl(self, inputs, outputs):
        x = inputs[0].data
        m = inputs[1].data
        M = inputs[2].data
        y = outputs[0].data
        y.copy_from(x)

        if not self.training:
            return
        mb = F.min(x, keepdims=True)
        Mb = F.max(x, keepdims=True)
        F.minimum2(m, mb, outputs=[m])
        F.maximum2(M, Mb, outputs=[M])

    def backward_impl(self, inputs, outputs, propagate_down, accum):
        dx = inputs[0].grad
        dy = outputs[0].grad
        dy.copy_from(dx)

        if propagate_down[0]:
            if accum[0]:
                dx += dy
            else:
                dx.copy_from(dy)

    def grad_depends_output_data(self, i, o):
        return False

    def grad_depends_input_data(self, i, j):
        return False


def minmax_minmax_recorder(x, m, M, training=True):
    ctx = nn.get_current_context()
    func = MinMaxMinMaxRecorder(ctx, training)
    return func(x, m, M)


class MinMaxMinMaxRecorderCallback(object):
    def __init__(self):
        self._function = minmax_minmax_recorder

    def name(self):
        n = self.__class__.__name__
        return n[:n.rfind('Callback')]

    def __call__(self, x, axes=[1], training=True, name=''):
        shape = [1] * x.ndim
        m = nn.parameter.get_parameter_or_create('m-{}'.format(name), shape,
                                                 ConstantInitializer())
        M = nn.parameter.get_parameter_or_create('M-{}'.format(name), shape,
                                                 ConstantInitializer())
        y = self._function(x, m, M, training=training)
        return y

    def get_scale_zeropoint(x, axes=[1], narrow_range=False, round_method='NOTROUND', name=''):
        # If recorder is not added before Q/DQ, return neither scale nor zp
        if not _param_exits('m-{}'.format(name)) and not _param_exits('M-{}'.format(name)):
            return None, None

        shape = [1] * x.ndim
        m = nn.parameter.get_parameter_or_create('m-{}'.format(name), shape)
        M = nn.parameter.get_parameter_or_create('M-{}'.format(name), shape)

        n_bits = 8
        im = - 2 ** (n_bits - 1)
        iM = 2 ** (n_bits - 1) - 1
        de = (iM - im) if not narrow_range else (iM - (im + 1))

        # MinMaxMinMax, 1e-24 is a small experience value to avoid zero scale
        scale = np.maximum((M.d - m.d), 1e-24) / de

        # round
        _round = __round_methods__[round_method]
        scale = 2 ** (_round(np.log(scale) / np.log(2))
                      ) if _round else scale  # pow2 scale

        # set zeropoint to zero
        zp = np.round((np.zeros_like(m.d) / scale).astype(np.int8))
        zp = nn.parameter.get_parameter_or_create('zeropoint-{}'.format(name),
                                                  zp.shape, zp, False)

        scale = nn.parameter.get_parameter_or_create('scale-{}'.format(name),
                                                     scale.shape, scale, False)

        return scale, zp


class AbsMaxRecorder(PythonFunction):
    """
    AbsMaxRecorder records the max of the aboslute of the batch over the training iterations.
    """

    def __init__(self, ctx, training=True):
        super(AbsMaxRecorder, self).__init__(ctx)
        self.training = training

    @property
    def name(self):
        return self.__class__.__name__

    def min_outputs(self):
        return 1

    def setup_impl(self, inputs, outputs):
        assert len(inputs) == 2, "len(inputs) must be 2; data, abs_max."
        x = inputs[0]
        M = inputs[1]
        assert M.ndim == x.ndim, \
            "ndim of abs_max must be same as ndim of data."
        assert np.prod(M.shape) == 1, \
            "Any dimenstion of the shape of min and max must be 1."
        y = outputs[0]
        y.reset_shape(x.shape, True)
        # inplace
        # y.data = x.data

    def forward_impl(self, inputs, outputs):
        x = inputs[0].data
        M = inputs[1].data
        y = outputs[0].data
        y.copy_from(x)

        if not self.training:
            return
        Mb = F.max(F.abs(x), keepdims=True)
        F.maximum2(M, Mb, outputs=[M])

    def backward_impl(self, inputs, outputs, propagate_down, accum):
        dx = inputs[0].grad
        dy = outputs[0].grad
        dy.copy_from(dx)

        if propagate_down[0]:
            if accum[0]:
                dx += dy
            else:
                dx.copy_from(dy)

    def grad_depends_output_data(self, i, o):
        return False

    def grad_depends_input_data(self, i, j):
        return False


def abs_max_recorder(x, M, training=True):
    ctx = nn.get_current_context()
    func = AbsMaxRecorder(ctx, training)
    return func(x, M)


class AbsMaxRecorderCallback(object):
    def __init__(self):
        self._function = abs_max_recorder

    def name(self):
        n = self.__class__.__name__
        return n[:n.rfind('Callback')]

    def __call__(self, x, axes=[1], training=True, name=''):
        shape = [1] * x.ndim
        M = nn.parameter.get_parameter_or_create('M-{}'.format(name), shape,
                                                 ConstantInitializer())

        y = self._function(x, M, training=training)
        return y

    def get_scale_zeropoint(x, axes=[1], narrow_range=False, round_method='NOTROUND', name=''):
        # If recorder is not added before Q/DQ, return neither scale nor zp
        if not _param_exits('M-{}'.format(name)):
            return None, None

        shape = [1] * x.ndim
        M = nn.parameter.get_parameter_or_create('M-{}'.format(name), shape)

        n_bits = 8
        im = - 2 ** (n_bits - 1)
        iM = 2 ** (n_bits - 1) - 1
        de = (iM - im) if not narrow_range else (iM - (im + 1))
        scale = (2 * M.d) / de  # AbsMax

        # round
        _round = __round_methods__[round_method]
        scale = 2 ** (_round(np.log(scale) / np.log(2))
                      ) if _round else scale  # pow2 scale

        # set zeropoint to zero
        zp = np.round((np.zeros_like(M.d) / scale).astype(np.int8))
        zp = nn.parameter.get_parameter_or_create('zeropoint-{}'.format(name),
                                                  zp.shape, zp, False)

        scale = nn.parameter.get_parameter_or_create('scale-{}'.format(name),
                                                     scale.shape, scale, False)

        return scale, zp


class MinMaxMvaRecorder(PythonFunction):
    """
    MinMaxMvaRecorder records the moving average of the min and max of the batch over the training iterations.
    """

    def __init__(self, ctx, decay=0.99, training=True):
        super(MinMaxMvaRecorder, self).__init__(ctx)

        self.decay = decay
        self.training = training

    @property
    def name(self):
        return self.__class__.__name__

    def min_outputs(self):
        return 1

    def setup_impl(self, inputs, outputs):
        assert len(inputs) == 3, "len(inputs) must be 3; data, min, max."
        x = inputs[0]
        m = inputs[1]
        M = inputs[2]
        assert m.ndim == x.ndim and M.ndim == x.ndim, \
            "ndim of min and max must be same as ndim of data."
        assert np.prod(m.shape) == 1 and np.prod(M.shape) == 1, \
            "Any dimenstion of the shape of min and max must be 1."
        y = outputs[0]
        y.reset_shape(x.shape, True)
        # inplace
        # y.data = x.data

    def forward_impl(self, inputs, outputs):
        x = inputs[0].data
        m = inputs[1].data
        M = inputs[2].data
        y = outputs[0].data
        y.copy_from(x)

        if not self.training:
            return
        mb = F.min(x, keepdims=True)
        Mb = F.max(x, keepdims=True)
        F.identity(self.decay * m + (1 - self.decay) * mb, outputs=[m])
        F.identity(self.decay * M + (1 - self.decay) * Mb, outputs=[M])

    def backward_impl(self, inputs, outputs, propagate_down, accum):
        dx = inputs[0].grad
        dy = outputs[0].grad
        dy.copy_from(dx)

        if propagate_down[0]:
            if accum[0]:
                dx += dy
            else:
                dx.copy_from(dy)

    def grad_depends_output_data(self, i, o):
        return False

    def grad_depends_input_data(self, i, j):
        return False


def minmax_mva_recorder(x, m, M, decay=0.99, training=True):
    ctx = nn.get_current_context()
    func = MinMaxMvaRecorder(ctx, decay, training)
    return func(x, m, M)


class MinMaxMvaRecorderCallback(object):
    def __init__(self):
        self._function = minmax_mva_recorder

    def name(self):
        n = self.__class__.__name__
        return n[:n.rfind('Callback')]

    def __call__(self, x, axes=[1], training=True, name=''):
        shape = [1] * x.ndim
        m = nn.parameter.get_parameter_or_create('m-{}'.format(name), shape,
                                                 ConstantInitializer())
        M = nn.parameter.get_parameter_or_create('M-{}'.format(name), shape,
                                                 ConstantInitializer())

        y = self._function(x, m, M, training=training)
        return y

    def get_scale_zeropoint(x, axes=[1], narrow_range=False, round_method='NOTROUND', name=''):
        # If recorder is not added before Q/DQ, return neither scale nor zp
        if not _param_exits('m-{}'.format(name)) and not _param_exits('M-{}'.format(name)):
            return None, None

        shape = [1] * x.ndim
        m = nn.parameter.get_parameter_or_create('m-{}'.format(name), shape)
        M = nn.parameter.get_parameter_or_create('M-{}'.format(name), shape)

        n_bits = 8
        im = - 2 ** (n_bits - 1)
        iM = 2 ** (n_bits - 1) - 1
        de = (iM - im) if not narrow_range else (iM - (im + 1))

        # MinMaxMva, 1e-24 is a small experience value to avoid zero scale
        scale = np.maximum((M.d - m.d), 1e-24) / de

        # round
        _round = __round_methods__[round_method]
        scale = 2 ** (_round(np.log(scale) / np.log(2))
                      ) if _round else scale  # pow2 scale

        # set zeropoint to zero
        zp = np.round((np.zeros_like(m.d) / scale).astype(np.int8))
        zp = nn.parameter.get_parameter_or_create('zeropoint-{}'.format(name),
                                                  zp.shape, zp, False)

        scale = nn.parameter.get_parameter_or_create('scale-{}'.format(name),
                                                     scale.shape, scale, False)

        return scale, zp


class MaxMaxRecorder(PythonFunction):
    """
    MaxMaxRecorder records the max of the batch over the training iterations.
    """

    def __init__(self, ctx, training=True):
        super(MaxMaxRecorder, self).__init__(ctx)
        self.training = training

    @property
    def name(self):
        return self.__class__.__name__

    def min_outputs(self):
        return 1

    def setup_impl(self, inputs, outputs):
        assert len(inputs) == 2, "len(inputs) must be 2; data, max."
        x = inputs[0]
        M = inputs[1]
        assert M.ndim == x.ndim, \
            "ndim of max must be same as ndim of data."
        assert np.prod(M.shape) == 1, \
            "Any dimenstion of the shape of max must be 1."
        y = outputs[0]
        y.reset_shape(x.shape, True)
        # inplace
        # y.data = x.data

    def forward_impl(self, inputs, outputs):
        x = inputs[0].data
        M = inputs[1].data
        y = outputs[0].data
        y.copy_from(x)

        if not self.training:
            return
        Mb = F.max(x, keepdims=True)
        F.maximum2(M, Mb, outputs=[M])

    def backward_impl(self, inputs, outputs, propagate_down, accum):
        dx = inputs[0].grad
        dy = outputs[0].grad
        dy.copy_from(dx)

        if propagate_down[0]:
            if accum[0]:
                dx += dy
            else:
                dx.copy_from(dy)

    def grad_depends_output_data(self, i, o):
        return False

    def grad_depends_input_data(self, i, j):
        return False


def max_max_recorder(x, M, training=True):
    ctx = nn.get_current_context()
    func = MaxMaxRecorder(ctx, training)
    return func(x, M)


class MaxMaxRecorderCallback(object):
    def __init__(self):
        self._function = max_max_recorder

    def name(self):
        n = self.__class__.__name__
        return n[:n.rfind('Callback')]

    def __call__(self, x, axes=[1], training=True, name=''):
        shape = [1] * x.ndim
        M = nn.parameter.get_parameter_or_create('M-{}'.format(name), shape,
                                                 ConstantInitializer())

        y = self._function(x, M, training=training)
        return y

    def get_scale_zeropoint(x, axes=[1], narrow_range=False, round_method='NOTROUND', name=''):
        # If recorder is not added before Q/DQ, return neither scale nor zp
        if not _param_exits('M-{}'.format(name)):
            return None, None

        shape = [1] * x.ndim
        M = nn.parameter.get_parameter_or_create('M-{}'.format(name), shape)

        n_bits = 8
        im = - 2 ** (n_bits - 1)
        iM = 2 ** (n_bits - 1) - 1
        de = (iM - im) if not narrow_range else (iM - (im + 1))
        scale = (2 * M.d) / de  # MaxMax

        # round
        _round = __round_methods__[round_method]
        scale = 2 ** (_round(np.log(scale) / np.log(2))
                      ) if _round else scale  # pow2 scale

        # set zeropoint to zero
        zp = np.round((np.zeros_like(M.d) / scale).astype(np.int8))
        zp = nn.parameter.get_parameter_or_create('zeropoint-{}'.format(name),
                                                  zp.shape, zp, False)

        scale = nn.parameter.get_parameter_or_create('scale-{}'.format(name),
                                                     scale.shape, scale, False)

        return scale, zp


class MaxMvaRecorder(PythonFunction):
    """
    MaxMvaRecorder records the moving average of the max of the batch over the training iterations.
    """

    def __init__(self, ctx, decay=0.99, training=True):
        super(MaxMvaRecorder, self).__init__(ctx)

        self.decay = decay
        self.training = training

    @property
    def name(self):
        return self.__class__.__name__

    def min_outputs(self):
        return 1

    def setup_impl(self, inputs, outputs):
        assert len(inputs) == 2, "len(inputs) must be 2; data, max."
        x = inputs[0]
        M = inputs[1]
        assert M.ndim == x.ndim, \
            "ndim of max must be same as ndim of data."
        assert np.prod(M.shape) == 1, \
            "Any dimenstion of the shape of max must be 1."
        y = outputs[0]
        y.reset_shape(x.shape, True)
        # inplace
        # y.data = x.data

    def forward_impl(self, inputs, outputs):
        x = inputs[0].data
        M = inputs[1].data
        y = outputs[0].data
        y.copy_from(x)

        if not self.training:
            return
        Mb = F.max(x, keepdims=True)
        F.identity(self.decay * M + (1 - self.decay) * Mb, outputs=[M])

    def backward_impl(self, inputs, outputs, propagate_down, accum):
        dx = inputs[0].grad
        dy = outputs[0].grad
        dy.copy_from(dx)

        if propagate_down[0]:
            if accum[0]:
                dx += dy
            else:
                dx.copy_from(dy)

    def grad_depends_output_data(self, i, o):
        return False

    def grad_depends_input_data(self, i, j):
        return False


def max_mva_recorder(x, M, decay=0.99, training=True):
    ctx = nn.get_current_context()
    func = MaxMvaRecorder(ctx, decay, training)
    return func(x, M)


class MaxMvaRecorderCallback(object):
    def __init__(self):
        self._function = max_mva_recorder

    def name(self):
        n = self.__class__.__name__
        return n[:n.rfind('Callback')]

    def __call__(self, x, axes=[1], training=True, name=''):
        shape = [1] * x.ndim
        M = nn.parameter.get_parameter_or_create('M-{}'.format(name), shape,
                                                 ConstantInitializer())

        y = self._function(x, M, training=training)
        return y

    def get_scale_zeropoint(x, axes=[1], narrow_range=False, round_method='NOTROUND', name=''):
        # If recorder is not added before Q/DQ, return neither scale nor zp
        if not _param_exits('M-{}'.format(name)):
            return None, None

        shape = [1] * x.ndim
        M = nn.parameter.get_parameter_or_create('M-{}'.format(name), shape)

        n_bits = 8
        im = - 2 ** (n_bits - 1)
        iM = 2 ** (n_bits - 1) - 1
        de = (iM - im) if not narrow_range else (iM - (im + 1))
        scale = (2 * M.d) / de  # MaxMva

        # round
        _round = __round_methods__[round_method]
        scale = 2 ** (_round(np.log(scale) / np.log(2))
                      ) if _round else scale  # pow2 scale

        # set zeropoint to zero
        zp = np.round((np.zeros_like(M.d) / scale).astype(np.int8))
        zp = nn.parameter.get_parameter_or_create('zeropoint-{}'.format(name),
                                                  zp.shape, zp, False)

        scale = nn.parameter.get_parameter_or_create('scale-{}'.format(name),
                                                     scale.shape, scale, False)

        return scale, zp


class PrecisionMode(Enum):
    # Quantized functions only
    QNN = 0
    # Quantize/Dequantize only
    SIM_QNN = 1
    # Quantized functions as much as possible;
    # otherwise Quantize/Dequantize (simulated quantization) are used
    MIXED_QNN = 2


class QNNState(Enum):
    # Floating network
    NON_QNN = 0
    # Recording min/max at some weights and activations
    RECORDING = 1
    # Quantized network, some layers are quantized, depending on layer configuration
    TRAINING = 2
    # TODO: Deployment network, quantized weights might be fused and saved as int8
    DEPLOYMENT = 3


# TODO : elaborate more


class QATConfig:
    #: Extension Context. 'cpu', 'cuda' or 'cudnn'
    ext_name = "cudnn"

    #: Use zero-point (asymmetric) or not use (symmetric)
    zero_point = False

    #: Precision
    dtype = np.int8
    precision_mode = PrecisionMode.SIM_QNN

    #: Enable channel last (channel_first is only supported now)
    channel_last = False

    # (TODO: near future, normally used for weights)
    #: Enable channel-wise quantization
    channel_wise = False

    #: Step start to record
    niter_to_recording = 0

    #: Step start to QAT.
    #: The number of steps between recording and training should be greater than the number of steps of one epoch training.
    niter_to_training = -1

    #: Coerce to power-of-2 scale when transiting from the recording graph
    class RoundingMethod(Enum):
        """
        Round method of scale
        """
        #: round up.
        #: e.g. ceil(9.4) = 10
        CEIL = 'CEIL'  # round up
        #: round.
        #: e.g. round(9.4) = 9, round(9.5) = 10
        ROUND = 'ROUND'
        #: round down.
        #: e.g. floor(9.5) = 9
        FLOOR = 'FLOOR'  # round down
        #: not round
        NOTROUND = 'NOTROUND'

    #: Member of :obj:`nnabla.utils.qnn.QATConfig.RoundingMethod`.
    #: Round the scale to power-of-2.
    #: If you want to deploy the model with tensorrt, please enable this.
    pow2 = RoundingMethod.ROUND

    #: Narrow the lower-bound (e.g., when in int8, -128 -> -127)
    narrow_range = False

    #: Round mode of quantize layer
    round_mode = 'HALF_TO_EVEN'

    # (TODO: decide by experiments for ImageNet classification and super resolution task)

    #: Enable Batch Normalization Folding.
    #: Note that sometimes this can cause the training become unstable.
    bn_folding = False

    #: Enable Batch Normalization Self-Folding.
    #: Note that sometimes this can cause the training become unstable.
    bn_self_folding = False

    #: One of :obj:`nnabla.utils.qnn.MinMaxMinMaxRecorderCallback`,
    #: :obj:`nnabla.utils.qnn.AbsMaxRecorderCallback`,
    #: :obj:`nnabla.utils.qnn.MinMaxMvaRecorderCallback`,
    #: :obj:`nnabla.utils.qnn.MaxMaxRecorderCallback`,
    #: :obj:`nnabla.utils.qnn.MaxMvaRecorderCallback`
    #: Recorder of weight
    recorder_weight = MinMaxMinMaxRecorderCallback

    #: One of :obj:`nnabla.utils.qnn.MinMaxMinMaxRecorderCallback`,
    #: :obj:`nnabla.utils.qnn.AbsMaxRecorderCallback`,
    #: :obj:`nnabla.utils.qnn.MinMaxMvaRecorderCallback`,
    #: :obj:`nnabla.utils.qnn.MaxMaxRecorderCallback`,
    #: :obj:`nnabla.utils.qnn.MaxMvaRecorderCallback`
    #: Recorder of activation
    recorder_activation = MaxMvaRecorderCallback

    #: list of nnabla function names.
    #: Recording layers.
    #: If empty, add recoders to all layers. Otherwise, only add recoders to functions in record_layers.
    record_layers = []

    class RecorderPosition(Enum):
        """
        Position to add recorder for function.
        """

        #: Add recoder only before a function
        BEFORE = 0

        #: Add recoder before/after a function
        BOTH = 1

    #: Member of :obj:`nnabla.utils.qnn.QATConfig.RecorderPosition`.
    #: Recorder position
    recorder_position = RecorderPosition.BEFORE

    #: List of nnabla function name.
    #: Skip quantizing inputs layers of network
    skip_inputs_layers = ['Convolution', 'Deconvolution']

    #: List of nnabla function name.
    #: Skip quantizing outputs layers of network
    skip_outputs_layers = ['Affine']

    #: QAT Learning_rate = NonQNN Learning_rate * learning_rate_scale.
    #: Recommend setting it to 0.1 or 0.01
    learning_rate_scale = 0.1

    #: Skip quantizing bias of Affine and bias of the Convolution function family
    skip_bias = False


class QATTensorRTConfig(QATConfig):
    pow2 = QATConfig.RoundingMethod.ROUND
    bn_folding = True
    bn_self_folding = True
    record_layers = ["Convolution", "Deconvolution",
                     "Affine", "BatchMatmul", "ReLU"]
    record_position = QATConfig.RecorderPosition.BEFORE


qat_default_config = QATTensorRTConfig()


class FunctionsRankRecorder(object):
    def __init__(self, functions_only_for_training=['ImageAugmentation', 'Dropout']):
        self.functions_only_for_training = functions_only_for_training
        self.rank = 0
        self.functions_ranks = {}

    def __call__(self, f):
        if f.info.type_name not in self.functions_only_for_training:
            self.functions_ranks[f] = self.rank
            self.rank += 1


class QATScheduler:
    """
    Scheduler for quantization aware training.

    Args:
      config (:obj:`QATConfig`): Quantization-Aware-Training Configuration
      solver (:obj:`nnabla.solver.Solver`): Neural Network Solver

    Example:

        .. code-block:: python

            from nnabla.utils.qnn import QATScheduler, QATConfig, PrecisionMode

            # Set configuration
            config = QATConfig()
            config.bn_folding = True
            config.bn_self_folding = True
            config.channel_last = False
            config.precision_mode = PrecisionMode.SIM_QNN
            config.niter_to_recording = 1
            config.niter_to_training = 500

            qat_scheduler = QATScheduler(config=config, solver=solver)

            # convert graph to enable quantization aware training.
            qat_scheduler(pred) # pred is the output variable of training network
            qat_scheduler(vpred, training=False) # vpred is the output variable of evaluation network

            # Training loop
            for i in range(training_step):
                qat_scheduler.step()

                # Your training code here

            # save quantized nnp
            qat_scheduler.save('qnn.np', vimage, deploy=False) # vimage is the input variable of network
    """

    # set default config
    def __init__(self, config=qat_default_config, solver=None):
        """
        Args:
          config (:obj:`QATConfig`):
          solver (:obj: 'nnabla.solvers.Solver'): Neural Network Solver

        """
        self.config = config
        if config.niter_to_training <= 0 or config.niter_to_recording < 0:
            raise ValueError(
                'Please set niter_to_recording and niter_to_training correctly! niter_to_recording should be greater than or equal to 0. niter_to_training should be greater than 0')
        if (config.niter_to_training - config.niter_to_recording) <= 0:
            raise ValueError(
                'Please set niter_to_recording and niter_to_training correctly! The number of steps between recording and training should be greater than the number of steps of one epoch training.')
        self.solver = solver
        self.counter = 0
        self.state = QNNState.NON_QNN
        self.registry = []  # [(nn.Variable, training)]

    def __call__(self, pred, training=True):
        """
        Wrap the network to quantized.

        Args:
          pred (:obj:`nnabla.Variable` or list of :obj:`nnabla.Variable`): Network output; the output of the original computation graph to be quantized.
        """
        # TODO: address list case (e.g., multiple outputs)
        self.registry.append((pred, training))

    def _register_params(self, solver):
        """
        Re-register parameters to `solver`
        """
        if not solver:
            return

        if (not self.config.bn_folding) and (not self.config.bn_self_folding):
            return

        solver.set_parameters(nn.get_parameters(grad_only=True))

    def _set_qat_learning_rate(self):
        self.solver.set_learning_rate(self.solver.learning_rate()
                                      * self.config.learning_rate_scale)

    def _fold_bn(self, pred):
        qpred_prev = pred
        # BN folding & BN self folding
        modifiers = [] if not self.config.bn_folding else [GC.BatchNormalizationFoldingModifier(
                opposite=False, channel_last=self.config.channel_last), GC.BatchNormalizationFoldingModifier(
                opposite=True, channel_last=self.config.channel_last)]
        modifiers = modifiers + \
            [GC.BatchNormalizationSelfFoldingModifier(
            )] if self.config.bn_self_folding else modifiers
        if len(modifiers) > 0:
            # expand fused_batch_normalization if BN folding or BN self folding is enabled.
            modifiers.insert(0, GC.UnfusedBatchNormalizationModifier())
            qpred_without_bn = GC.GraphConverter(
                modifiers).convert(qpred_prev)
            qpred_prev.rewire_on(qpred_without_bn)
        return qpred_prev

    def _clear_memory_cache(self):
        if self.config.ext_name in ["cuda", "cudnn"]:
            nnabla_ext.cuda.clear_memory_cache()

    def _schedule_to_recording(self):
        for i, elm in enumerate(self.registry):
            pred, training = elm
            qpred_prev = pred
            qpred_prev = self._fold_bn(qpred_prev)

            # Collect functions rank
            rank_recorder = FunctionsRankRecorder()
            qpred_prev.visit(rank_recorder)

            qpred_curr = GC.GraphConverter([
                GC.QuantizeNonQNNToRecordingModifier(
                    rank_recorder.functions_ranks, config=self.config, training=training)]).convert(qpred_prev)
            qpred_prev.rewire_on(qpred_curr)
            qpred_prev.need_grad = False
            self._register_params(self.solver)
            self.registry[i] = (qpred_prev, training)
            self.state = QNNState.RECORDING
            print(
                'QNNState.NON_QNN -> QNNState.RECORDING: graph={}'.format(qpred_prev))

    def _schedule_to_training(self):
        for i, elm in enumerate(self.registry):
            pred, training = elm
            qpred_prev = pred

            # Remove recorder
            modifiers = []
            modifiers.append(GC.RemoveFunctionModifier(
                rm_funcs=[self.config.recorder_activation().name(),
                          self.config.recorder_weight().name()]))
            qpred_noqnn = GC.GraphConverter(modifiers).convert(qpred_prev)
            qpred_prev.rewire_on(qpred_noqnn)

            # Collect functions rank
            rank_recorder = FunctionsRankRecorder()
            qpred_prev.visit(rank_recorder)

            # Recording to training
            qpred_curr = GC.GraphConverter([
                GC.QuantizeRecordingToTrainingModifier(rank_recorder.functions_ranks, config=self.config)]).convert(
                qpred_prev)
            qpred_prev.rewire_on(qpred_curr)
            self._register_params(self.solver)
            self._set_qat_learning_rate()
            self.registry[i] = (qpred_prev, training)
            self.state = QNNState.TRAINING
            print(
                'QNNState.RECORDING -> QNNState.TRAINING: graph={}'.format(qpred_prev))

    def step(self):
        """
        Step in the state of QNN. According to the number of iterations in config.
        """

        # TODO: there is other patterns
        # TODO: address list case (e.g., multiple outputs)
        if self.counter == self.config.niter_to_recording:
            self._clear_memory_cache()
            self._schedule_to_recording()

        elif self.counter == self.config.niter_to_training:
            self._clear_memory_cache()
            self._schedule_to_training()

        self.counter += 1

    def save(self, fname, inputs, batch_size=1, net_name='net', deploy=False):
        """
        Save QAT network model to NNP file as default.

        Args:
          fname (str): NNP file name.
          inputs (:obj:`nnabla.Variable` or list of :obj:`nnabla.Variable`): Network inputs variables.
          batch_size (int): batch size.
          net_name (str): network name.
          deploy (bool): Whether to apply QNN deployment conversion. deploy=True is not supported yet.

        Returns:
          None
        """

        def _force_list(o):
            if isinstance(o, (tuple)):
                return list(o)
            if not isinstance(o, (list)):
                return [o]
            return o

        for i, elm in enumerate(self.registry):
            pred, training = elm
            if deploy:
                assert self.state == QNNState.training
                # TODO: Convert the training graph to deployment graph
                # TODO: Save as nnp (we have to define nicely)
            else:
                if training:
                    continue

                from collections import defaultdict

                inps = defaultdict(list)
                otps = defaultdict(list)
                ec_data = []
                ec_otps = []

                inputs = _force_list(inputs)
                for i, inp in enumerate(inputs):
                    key = 'x{}'.format(i)
                    inps[key] = inp
                    ec_data.append(key)

                outputs = _force_list(pred)
                for i, otp in enumerate(outputs):
                    key = 'y{}'.format(i)
                    otps[key] = otp
                    ec_otps.append(key)

                contents = {
                    'networks': [
                        {'name': net_name,
                         'batch_size': batch_size,
                         'outputs': otps,
                         'names': inps
                         }],
                    'executors': [
                        {'name': 'runtime',
                         'network': net_name,
                         'data': ec_data,
                         'outputs': ec_otps
                         }]
                }

                from nnabla.utils.save import save
                save(fname, contents)
