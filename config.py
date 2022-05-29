import os 
from yacs.config import CfgNode as CN


_C = CN()

_C.SYSTEM = CN()
_C.SYSTEM.NUM_GPUS = 2
_C.SYSTEM.NUM_WORKERS = 8

_C.PROJECT.ROOT = "/home/johann/sonstiges/vision-transformer-pytorch"

_C.DATA = CN()
_C.DATA.ROOT = os.path.join(_C.PROJECT.ROOT, "data/imagenette2-320")
_C.DATA.ANNOS = os.path.join(_C.DATA.ROOT, "noisy_imagenette.csv")
_C.DATA.NUM_CLASSES = 10 
_C.DATA.IMG_SIZE = (320, 320)
_C.DATA.PATCH_SIZE = 32
_C.DATA.LABEL2ID = list(sorted(os.listdir(os.path.join(_C.DATA.ROOT, 'train'))))
_C.DATA.LABELNAMES = [
    'fish', 'dog', 'radio', 'chain saw', 'church', 
    'french horn', 'garbage truck', 'tank column', 
    'golf ball', 'paraglider',]

_C.TRAIN = CN()
_C.TRAIN.N_EPOCHS = 100
_C.TRAIN.BASE_LR = 0.001
_C.TRAIN.BATCH_SIZE = 128

_C.VAL = CN()
_C.VAL.BATCH_SIZE = 64

_C.MODEL = CN()
_C.MODEL.INPUT_DIM = int(_C.DATA.PATCH_SIZE**2 * 3)
_C.MODEL.MAX_SEQ_LEN = int(
    (_C.DATA.IMG_SIZE[0] / _C.DATA.PATCH_SIZE) *
    (_C.DATA.IMG_SIZE[1] / _C.DATA.PATCH_SIZE)
)
_C.MODEL.NUM_LAYERS = 4
_C.MODEL.D_MODEL = 1024
_C.MODEL.NUM_HEADS = 8
_C.MODEL.D_FF = 2048
_C.MODEL.DROPOUT = 0.1

_C.OPTIM = CN()
_C.OPTIM.BETAS = (0.9, 0.999)
_C.OPTIM.EPS = 0.000000001


def get_cfg_defaults():
  """Get a yacs CfgNode object with default values for my_project."""
  # Return a clone so that the defaults will not be altered
  # This is for the "local variable" use pattern
  return _C.clone()