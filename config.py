from typing import Any, List

from yacs.config import CfgNode as CN


class Config(object):
    def __init__(self, config_yaml: str, config_override: List[Any] = []):

        self._C = CN()
        self._C.GPU = [0]
        self._C.VERBOSE = False
        self._C.THREADS = 2
        self._C.SEED = 9487

        self._C.MODEL = CN()
        self._C.MODEL.MODE = 'global'
        self._C.MODEL.SESSION = 'ps128_bs1'

        self._C.OPTIM = CN()
        self._C.OPTIM.BATCH_SIZE = 2 
        self._C.OPTIM.TRAIN_EPOCH_SIZE = 1
        self._C.OPTIM.VALID_EPOCH_SIZE = 100
        self._C.OPTIM.EPOCH_MAX = 100
        self._C.OPTIM.NUM_EPOCHS = 100
        self._C.OPTIM.NEPOCH_DECAY = [100]
        self._C.OPTIM.LR_INITIAL = 0.0002
        self._C.OPTIM.LR_MIN = 0.0002
        self._C.OPTIM.BETA1 = 0.5
        self._C.OPTIM.ALPHA = 50.0
        self._C.OPTIM.W_ST = 0.0 
        self._C.OPTIM.W_LT = 0.0
        self._C.OPTIM.BLOCKS = 5
        self._C.OPTIM.NORM = 'IN'

        self._C.DATASET = CN()
        self._C.DATASET.DATA_DIR = '/home/disk1/zwz/derainCode/datasets'
        self._C.DATASET.LIST_DIR = '/home/disk1/zwz/derainCode/file_list'
        self._C.DATASET.CROP_SIZE = 128  
        self._C.DATASET.GEOMETRY_AUG = 0 
        self._C.DATASET.ORDER_AUG = 0 
        self._C.DATASET.SCALE_MIN = 0.5
        self._C.DATASET.SCALE_MAX = 2.0
        self._C.DATASET.SAMPLE_FRAMES = 5


        self._C.TRAINING = CN()
        self._C.TRAINING.VAL_AFTER_EVERY = 3
        self._C.TRAINING.RESUME = False
        self._C.TRAINING.SAVE_IMAGES = False
        self._C.TRAINING.TRAIN_DIR = 'images_dir/train'
        self._C.TRAINING.VAL_DIR = 'images_dir/val'
        self._C.TRAINING.SAVE_DIR = 'checkpoints'
        self._C.TRAINING.TRAIN_PS = 64
        self._C.TRAINING.VAL_PS = 64

        self._C.merge_from_file(config_yaml)
        self._C.merge_from_list(config_override)

        self._C.freeze()

    def dump(self, file_path: str):
        r"""Save config at the specified file path.

        Parameters
        ----------
        file_path: str
            (YAML) path to save config at.
        """
        self._C.dump(stream=open(file_path, "w"))

    def __getattr__(self, attr: str):
        return self._C.__getattr__(attr)

    def __repr__(self):
        return self._C.__repr__()
