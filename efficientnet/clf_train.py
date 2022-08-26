import torch
import random
import os
import numpy as np
import pandas as pd
import glob
import shutil
import time

from torch.utils.tensorboard import SummaryWriter

from efficientnet.model import WireClassifier
from efficientnet.clf_train_function import train_fn
from efficientnet.dataset import CustomDataset
from efficientnet.utils import get_transforms
from efficientnet.pt2weight import make_eff_weights

import torch.optim as optim
from torch.cuda.amp import GradScaler


def seed_torch(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def efficientnet_train(args):

    start_time = time.time()

    shutil.copy2(args.train_csv, os.path.join(args.eff_output_dir, "train_data.csv"))
    shutil.copy2(args.test_csv, os.path.join(args.eff_output_dir, "test_data.csv"))

    print(f"output directory : {args.eff_output_dir}")

    # random seed
    seed_torch(seed=args.seed)

    # logger
    def init_logger(log_file=args.eff_output_dir + '/train.log'):
        from logging import getLogger, INFO, FileHandler, Formatter, StreamHandler
        logger = getLogger(__name__)
        logger.setLevel(INFO)
        handler1 = StreamHandler()
        handler1.setFormatter(Formatter("%(message)s"))
        handler2 = FileHandler(filename=log_file)
        handler2.setFormatter(Formatter("%(message)s"))
        logger.addHandler(handler1)
        logger.addHandler(handler2)
        return logger

    LOGGER = init_logger()

    # Tensorboard writer
    tensorboard_dir = os.path.join(args.eff_output_dir, "tensorboard")
    writer = SummaryWriter(log_dir=tensorboard_dir)

    # model instance
    model = WireClassifier(num_classes=args.num_classes)
    if len(args.clf_path) > 0:
        try:
            model.load_state_dict(torch.load(args.clf_path))
        except:
            model = torch.jit.load(args.clf_path, map_location=args.device)

    model.to(args.device)

    # data read
    train_df = pd.read_csv(args.train_csv)
    val_df = pd.read_csv(args.test_csv)

    # remove original train data
    if len(train_df) > 25335: # original dataset length : 25535
        train_df = train_df.loc[25335:]
        train_df = train_df.reset_index(drop=True)


    # dataset length
    train_max = args.dataset_maximum  # default : 100000
    train_length = len(train_df)


    # validation dataset loader
    val_dataset = CustomDataset(args=args, df=val_df,  # df=val_df_sample,
                                transforms=get_transforms(img_size=args.test_img_size, data='test'))

    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.test_batch_size, shuffle=False,
                                             pin_memory=True, num_workers=args.num_workers)

    # scales to use in training
    train_img_sizes = [192, 256, 320, 352]
    n_sizes = len(train_img_sizes)

    ## epochs regularize
    # dataset 크기에 학습시간이 변하지 않도록 변수를 넣어줌
    # 학습 dataset의 크기가 기존 데이터셋의 크기인 25000개 보다 크다면 lambda epoch에 의해
    # epochs이 작아지고, 25000개 보다 작다면 epochs는 늘어남.
    lambda_epoch = 25000 / train_length  # if train_length >= 25000 else 1
    if lambda_epoch > 3: # 데이터셋이 작더라도 epochs가 3배 이상으로 많아지진 않도록 제약을 두어 과적합 방지
        lambda_epoch = 3
    args.epochs = int(args.epochs * lambda_epoch)

    # 4개의 scale로 학습하므로 epochs를 4로 나눠줌
    if args.epochs > n_sizes:
        args.epoch_per_stage = int(args.epochs / n_sizes)
    else:
        args.epoch_per_stage = args.epochs
        n_sizes = 1

    # training options
    scaler = GradScaler()
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.RMSprop(model.parameters(), lr=args.lr, alpha=0.9, eps=1e-8, weight_decay=args.weight_decay,
                              momentum=args.momentum)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50, eta_min=1e-6)
    # optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # training
    for stage in range(1, n_sizes + 1):
        LOGGER.info(f"\n========== stage: {stage}/{n_sizes} training ==========")

        # image scale
        args.train_img_size = train_img_sizes[stage - 1]

        # train dataset total size sampling
        if train_length > train_max:
            train_df_sample = train_df.sample(n=train_max).copy()
        else:
            train_df_sample = train_df.copy()

        # dataset
        train_dataset = CustomDataset(args=args, df=train_df_sample,
                                      transforms=get_transforms(img_size=args.train_img_size, data='train'))

        # data loader
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.train_batch_size, shuffle=True,
                                                   pin_memory=True, num_workers=args.num_workers)

        # train start
        train_fn(args=args, model=model, device=args.device, train_loader=train_loader, val_loader=val_loader,
                 writer=writer, LOGGER=LOGGER, scaler=scaler, criterion=criterion, optimizer=optimizer,
                 scheduler=scheduler, stage=stage)

        del train_loader

    end_time = time.time()
    LOGGER.info("\n Efficientnet Training Time : {}".format(end_time - start_time))
