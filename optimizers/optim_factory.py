from optimizers.mixop.gdas import GDASMixOp
from optimizers.sampler.gdas import GDASSampler
from optimizers.sampler.reinmax import ReinmaxSampler



def get_mixop(opt_name):
    return GDASMixOp()

def get_sampler(opt_name):
    if opt_name == "gdas":
        return GDASSampler()
    elif opt_name == "reinmax":
        return ReinmaxSampler()
    else:
        raise ValueError("Unknown sampler: {}".format(opt_name))
    
