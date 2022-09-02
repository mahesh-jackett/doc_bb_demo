# !python -m pip install 'git+https://github.com/facebookresearch/detectron2.git' --quiet


import numpy as np
import gdown
import os
import random
import torch

import warnings
warnings.filterwarnings('ignore') #Ignore "future" warnings and Data-Frame-Slicing warnings.

# detectron2
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import Visualizer

# -------------------------------

seed = 42
os.environ["PL_GLOBAL_SEED"] = str(seed)
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

# ----------------------------------

def build_model(THRESH:float = 0.6):

    cfg = get_cfg()

    config_name = "config.yml" # Using pre trained layout parser configs
    cfg.merge_from_file(config_name)


    if not torch.cuda.is_available(): cfg.MODEL.DEVICE = "cpu"

    cfg.DATALOADER.NUM_WORKERS: 2
    cfg.TEST.EVAL_PERIOD = 20 # Evaluate after N epochs

    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128 # Default 256 
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1 # in config file, it is written before weights

    if not os.path.exists('./model.pth'):
        url = "https://drive.google.com/uc?id=1S5LhdZiKS8dXqUeXYfDEOVkfBxVMuZsk"
        output = "./model.pth"
        gdown.download(url, output, quiet=True)

    cfg.MODEL.WEIGHTS = './model.pth' # layout parser Pre trained weights


    cfg.SOLVER.IMS_PER_BATCH = 4 # Batch size
    cfg.SOLVER.BASE_LR = 0.0025
    cfg.SOLVER.WARMUP_ITERS = 50
    cfg.SOLVER.MAX_ITER = 1000 # adjust up if val mAP is still rising, adjust down if overfit
    cfg.SOLVER.STEPS = (300, 800) # must be less than  MAX_ITER 
    cfg.SOLVER.GAMMA = 0.05
    cfg.SOLVER.CHECKPOINT_PERIOD = 20  # Save weights after these many epochs


    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = THRESH
    return DefaultPredictor(cfg)


def get_results(predictor, image):
    output = predictor(image)
    v = Visualizer(image[:,:,::-1]) # BGR to RGB format : https://github.com/youngwanLEE/vovnet-detectron2/issues/16
    v = v.draw_instance_predictions(output['instances'].to('cpu'))
    return v.get_image()



