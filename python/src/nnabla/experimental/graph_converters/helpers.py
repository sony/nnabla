from collections import defaultdict


class PrintFuncName(object):

    def __init__(self, ):
        pass

    def __call__(self, nnabla_func):
        print(nnabla_func.name)


class GraphInfo(object):
    """Graph Information


    Args:
      pred (:obj:`Variable`): The end of `Variable`.

    Attributes: 
      funcs (list of :obj:`Function`): :obj:`Function` list to be call in the forward order.
      inputs (`list` of :obj:`Variable`): :obj:`Variable` list coming to :obj:`Function`.
      outputs (`list` of :obj:`Variable`): :obj:`Function` list to coming from :obj:`Function`.
      variable_to_funcs (`dict`): Dictionary from :obj:`Variable` to :obj:`Function`.

    """
    class Functor(object):

        def __init__(self, funcs, inputs, outputs,
                     variable_to_funcs):
            self.funcs = funcs
            self.inputs = inputs
            self.outputs = outputs
            self.variable_to_funcs = variable_to_funcs

        def __call__(self, func):
            self.funcs.append(func)
            self.inputs.append(func.inputs)
            self.outputs.append(func.outputs)
            for i in func.inputs:
                self.variable_to_funcs[i].append(func)

    def __init__(self, pred):
        self.funcs = []
        self.inputs = []
        self.outputs = []
        self.variable_to_funcs = defaultdict(list)

        functor = GraphInfo.Functor(self.funcs,
                                    self.inputs, self.outputs,
                                    self.variable_to_funcs)
        pred.visit(functor)
