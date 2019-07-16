import nnabla.communicators as C


class CommunicationWrapper(object):

    def __init__(self, ctx):
        try:
            comm = C.MultiProcessDataParallelCommunicator(ctx)
        except Exception as e:
            print(e)
            print("No communicator found. Running with a single process. If you run this with MPI processes, all processes will perform totally same.")
            self.n_procs = 1
            self.rank = 0
            self.ctx = ctx
            self.comm = None
            return

        comm.init()
        self.n_procs = comm.size
        self.rank = comm.rank
        self.ctx = ctx
        self.ctx.device_id = str(self.rank)
        self.comm = comm

    def all_reduce(self, params, division, inplace):
        if self.n_procs == 1:
            return
        self.comm.all_reduce(params, division=division, inplace=inplace)
