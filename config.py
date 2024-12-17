# Same config.py as done in the paper.
def zx_to_npx(zx, depth):
    '''
    Calculate size of output image given depth of network and input field size
    Inputs: 
    zx (int)—size of input field
    deptu (int)—depth of convolutional stack 

    Output: size of output image
    '''
    return (zx - 1) * 2**depth + 1

class Config:
    '''
    Hold PSGAN parameters.
    Taken from the paper — this is NOT a parameter tuning project.
    '''
    lr = 0.0002 # adam learning rate
    b1 = 0.5 # adam momentum term
    l2_fac = 1e-8 # L2 weight reg
    epoch_count = 100 # training epochs
    k = 1 # D updates vs G updates
    batch_size = 25
    epoch_iters = batch_size * 1000 # steps per epoch 

    def __init__(self):
        # sampling params from paper
        self.nz_local = 30
        self.nz_global = 60
        self.nz_periodic = 3
        self.nz_periodic_MLPnodes = 50
        self.nz = self.nz_local + self.nz_global + self.nz_periodic * 2 
        self.periodic_affine = False 
        self.zx = 6
        self.zx_sample = 32
        self.zx_sample_quilt = self.zx_sample / 4

        # network params from paper
        self.nc = 3
        self.gen_ks = ([(5,5)] * 5)[::-1]
        self.dis_ks = [(5,5)] * 5
        self.gen_ls = len(self.gen_ks)
        self.dis_ls = len(self.dis_ks)
        self.gen_fn = [self.nc] + [2 ** (n+6) for n in range(self.gen_ls - 1)]
        self.gen_fn = self.gen_fn[::-1]
        self.dis_fn = [2 ** (n+6) for n in range(self.dis_ls-1)] + [1]
        self.npx = zx_to_npx(self.zx, self.gen_ls)