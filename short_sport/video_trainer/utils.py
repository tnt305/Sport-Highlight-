import configparser
from pathlib import Path

def read_config():
    config = configparser.ConfigParser()
    # cur_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    config.read("D:/projects/v2v/v5/settings.cfg")#os.path.join(cur_path, 'location.cfg'))
    
#     2/17: Peter made cfg for 571 for ease of use
    return config['305']

def normalize_path(path: str):
    return Path(path).as_posix()

class AverageMeter(object):
    """Computes and stores the average and current value
       Imported from https://github.com/pytorch/examples/blob/master/imagenet/main.py#L247-L262
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0.0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
