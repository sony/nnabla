import nnabla as nn
import nnabla.functions as F
import numpy as np
import scipy


CHECK_COLUMNS = ['mean', 'stddev', 'variance', 'entropy']
SCIPY_FUNCS = ['mean', 'std', 'var', 'entropy']


def distribution_test_util(dist_fn,
                           scipy_fn,
                           param_fn,
                           skip_columns=[],
                           sample_shape=(10000, 10),
                           ref_sample_fn=None,
                           ref_columns={},
                           ref_prob_fn=None):
    # check mean and standard deviation of sampled values
    _check_sample(dist_fn, scipy_fn, param_fn, ref_sample_fn)

    # check each parameters
    _check_columns(dist_fn, scipy_fn, param_fn, skip_columns, sample_shape,
                   ref_columns)

    # check probability density function
    _check_prob(dist_fn, scipy_fn, param_fn, sample_shape, ref_sample_fn,
                ref_prob_fn)


def _check_sample(dist_fn, scipy_fn, param_fn, ref_sample_fn):
    params = param_fn(shape=(10000, 10))

    dist = dist_fn(*params)

    # nnabla sample
    sample = dist.sample()
    sample.forward(clear_buffer=True)

    if ref_sample_fn is None:
        # scipy sample
        scipy_dist = scipy_fn(*params)
        ref_sample = scipy_dist.rvs(size=(10000, 10))
    else:
        ref_sample = ref_sample_fn(*params, shape=(10000, 10))

    assert np.allclose(sample.d.mean(), ref_sample.mean(), atol=3e-2,
                       rtol=3e-2)
    assert np.allclose(sample.d.std(), ref_sample.std(), atol=3e-2,
                       rtol=3e-2)

    # nnabla sample_n
    sample_n = dist.sample_n(2)
    sample_n.forward(clear_buffer=True)
    assert sample_n.d.shape == (10000, 2, 10)


def _check_prob(dist_fn,
                scipy_fn,
                param_fn,
                sample_shape,
                ref_sample_fn,
                ref_prob_fn):
    params = param_fn(shape=sample_shape)
    dist = dist_fn(*params)

    if ref_sample_fn is None:
        scipy_dist = scipy_fn(*params)
        sample = scipy_dist.rvs(size=sample_shape)
    else:
        sample = ref_sample_fn(*params, shape=sample_shape)

    prob = dist.prob(nn.Variable.from_numpy_array(sample))
    prob.forward(clear_buffer=True)

    if ref_prob_fn is None:
        scipy_dist = scipy_fn(*params)
        ref_prob = scipy_dist.pdf(sample)
    else:
        ref_prob = ref_prob_fn(*params, sample=sample, shape=sample_shape)

    assert np.allclose(prob.d, ref_prob, atol=3e-2, rtol=3e-2)


def _check_columns(dist_fn,
                   scipy_fn,
                   param_fn,
                   skip_columns,
                   sample_shape,
                   ref_columns):
    param = param_fn(shape=sample_shape)
    dist = dist_fn(*param)
    scipy_dist = scipy_fn(*param)

    for i, column in enumerate(CHECK_COLUMNS):
        if column in skip_columns:
            continue

        v = getattr(dist, column)()
        v.forward()

        if column in ref_columns:
            ref_v = ref_columns[column](*param, shape=sample_shape)
        else:
            ref_v = getattr(scipy_dist, SCIPY_FUNCS[i])()

        assert np.allclose(v.d, ref_v, atol=3e-2, rtol=3e-2)
