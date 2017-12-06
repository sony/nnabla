
def test_save_load_parameters():
    import nnabla as nn
    import nnabla.functions as F
    import nnabla.parametric_functions as PF
    import nnabla.utils.save
    import nnabla.utils.load

    batch_size = 16
    x0 = nn.Variable([batch_size, 100])
    x1 = nn.Variable([batch_size, 100])
    h1_0 = PF.affine(x0, 100, name='affine1_0')
    h1_1 = PF.affine(x1, 100, name='affine1_0')
    h1 = F.tanh(h1_0 + h1_1)
    h2 = F.tanh(PF.affine(h1, 50, name='affine2'))
    y0 = PF.affine(h2, 10, name='affiney_0')
    y1 = PF.affine(h2, 10, name='affiney_1')

    contents = {
        'networks': [
            {'name': 'net1',
             'batch_size': batch_size,
             'outputs': {'y0': y0, 'y1': y1},
             'names': {'x0': x0, 'x1': x1}}],
        'executors': [
            {'name': 'runtime',
             'network': 'net1',
             'data': ['x0', 'x1'],
             'output': ['y0', 'y1']}]}
    nnabla.utils.save.save('tmp.nnp', contents)
    nnabla.utils.load.load('tmp.nnp')
