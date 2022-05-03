from detectron2.utils.logger import setup_logger
setup_logger() # Setup detectron2 logger

from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.data.datasets import register_coco_instances, load_coco_json

from mask_rcnn.custom_trainer import MyTrainer
from mask_rcnn.collect_json import labelme2coco
from mask_rcnn.utils import select_best
from mask_rcnn.pt2weight import make_mask_weights

import os
import glob
import shutil
import datetime
import pickle


def training(args):

    print("Mask RCNN preprocessing...")
    

    if os.path.isfile('mask_classes.pkl'):
        with open('mask_classes.pkl', 'rb') as f:
            thing_classes = pickle.load(f)
            args.mask_classes = len(thing_classes)
            args.mask_id = thing_classes.index('wire')
    else:
        thing_classes = ['airplane',
                         'banana',
                         'baseball bat',
                         'carrot',
                         'dining table',
                         'fork',
                         'giraffe',
                         'hot dog',
                         'keyboard',
                         'knife',
                         # 'nipped',
                         'pen',
                         'pizza',
                         'scissors',
                         'skateboard',
                         'skis',
                         'snowboard',
                         'spoon',
                         'sports ball',
                         'stop sign',
                         'surfboard',
                         'tennis racket',
                         'tie',
                         'toothbrush',
                         'train',
                         'truck',
                         'wire']
    
        args.mask_classes = len(thing_classes)
        args.mask_id = thing_classes.index('wire')

    # train dataset path
    train_json_folder_orig = os.path.join(args.file_storage_path, "train_data", "json")
    train_json_folder_new = os.path.join(args.data_path, "train_data", "json")
    train_jsons = glob.glob(os.path.join(train_json_folder_orig, "*.json")) + glob.glob(os.path.join(train_json_folder_new, "*.json"))

    # make json file like coco dataset format
    train_collect = os.path.join(args.mask_output_dir, "train.json")
    labelme2coco(train_jsons, train_collect, thing_classes=thing_classes)

    # test (same with train)
    test_json_folder_orig = os.path.join(args.file_storage_path, "test_data", "json")
    test_json_folder_new = os.path.join(args.data_path, "test_data", "json")
    test_jsons = glob.glob(os.path.join(test_json_folder_orig, "*.json")) + glob.glob(os.path.join(test_json_folder_new, "*.json"))
    test_collect = os.path.join(args.mask_output_dir, "test.json")
    labelme2coco(test_jsons, test_collect, thing_classes=thing_classes)

    # register json file for detectron
    register_coco_instances("train", {}, train_collect, "")  # train_collect must be coco dataset style
    register_coco_instances("test", {}, test_collect, "")

    # hyperparameters tuning
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml"))
    cfg.DATASETS.TRAIN = ("train", )
    cfg.DATASETS.TEST = ("test", )
    # cfg.TEST.EVAL_PERIOD = args.mask_checkpoint
    cfg.DATALOADER.NUM_WORKERS = args.num_workers

    cfg.INPUT.MIN_SIZE_TRAIN = [640, 672, 704, 736, 768, 800]
    cfg.INPUT.MAX_SIZE_TRAIN = 1333
    cfg.INPUT.MIN_SIZE_TEST = 800
    cfg.INPUT.MAX_SIZE_TEST = 1333

    cfg.SOLVER.IMS_PER_BATCH = args.mask_batch
    cfg.SOLVER.BASE_LR = args.mask_lr  # pick a good LR
    cfg.SOLVER.BASE_LR_END = 1e-6   # use with WarmupCosineLr
    cfg.SOLVER.LR_SCHEDULER_NAME = "WarmupCosineLR" # WarmupCosineLR or WarmupMultiStepLR
    cfg.SOLVER.MAX_ITER = args.mask_iters
    cfg.SOLVER.STEPS = [5000, 10000, 15000]
    cfg.SOLVER.GAMMA = 0.5
    cfg.SOLVER.CHECKPOINT_PERIOD = args.mask_checkpoint
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = args.mask_batch_per_images
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = args.mask_classes  # num classes
    cfg.OUTPUT_DIR = args.mask_output_dir

    # load pretrained Mask RCNN
    try:
        cfg.MODEL.WEIGHTS = args.mask_path
    except:
        print("Cannot load the pretrained mask model")
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml")  # training initialize from model zoo

    # Call Custom Trainer
    trainer = MyTrainer(cfg)
    trainer.resume_or_load(resume=False)

    print(f"Training until {args.mask_iters} iterations")
    
    # Training
    try:
        trainer.train()
    except:
        raise

    # leave the best model
    args.mask_best_model, args.AP = select_best(args)
    print(f"Trained mask model path : {args.mask_best_model}")
    print("Mask RCNN Training has done.")

    del cfg, trainer


def mask_train(args):
    # now = datetime.datetime.now().strftime('%Y-%m-%d_%Hh%Mm%Ss')

    # Training
    training(args)

    # pt to weights
    make_mask_weights(args)
