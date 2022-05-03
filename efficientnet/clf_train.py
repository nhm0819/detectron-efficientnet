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
from efficientnet.scheduler import CosineAnnealingWarmUpRestarts


def seed_torch(seed=42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def efficientnet_train(args):
    # import datetime

    # now = datetime.datetime.now().strftime('%Y-%m-%d_%Hh%Mm%Ss')
    # args.eff_output_dir = os.path.join(args.output_dir, "efficientnet", f"{args.start_time}")
    # if not os.path.exists(args.eff_output_dir):
    # os.makedirs(args.eff_output_dir, exist_ok=True)

    start_time = time.time()

    shutil.copy2(args.train_csv, os.path.join(args.eff_output_dir, "train_data.csv"))
    shutil.copy2(args.test_csv, os.path.join(args.eff_output_dir, "test_data.csv"))

    print(f"output directory : {args.eff_output_dir}")

    # random seed
    seed_torch(seed=args.seed)

    # logger
    def init_logger(log_file=args.eff_output_dir + "/train.log"):
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

    # model create
    model = WireClassifier(num_classes=args.num_classes)
    try:
        model.load_state_dict(torch.load(args.clf_path))
    except:
        model = torch.jit.load(args.clf_path, map_location=args.device)

    model.to(args.device)

    # data read
    train_df = pd.read_csv(args.train_csv)
    val_df = pd.read_csv(args.test_csv)
    # indistinct_df = pd.read_csv(args.indistinct_csv)

    # dataset length
    train_max = args.dataset_size  # default : 100000
    val_max = int(train_max * 0.2)

    train_length = len(train_df)
    val_length = len(val_df)

    # # train dataset total size sampling
    # if train_length > train_max:
    #     train_cut_index = train_length - train_max
    #     train_df_sample = train_df[train_cut_index:] # train_df_sample = train_df_weighted[cut_index:]
    #     train_df_sample = pd.concat([train_df_sample, indistinct_df]).copy()

    # else:
    #     train_df_sample = pd.concat([train_df, indistinct_df]).copy()

    # validation setting
    if val_length > val_max:
        val_cut_index = val_length - val_max
        val_df_sample = val_df[val_cut_index:]
    else:
        val_df_sample = val_df

    val_dataset = CustomDataset(
        args=args,
        df=val_df_sample,
        transforms=get_transforms(img_size=args.test_img_size, data="test"),
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.test_batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=args.num_workers,
    )

    # # scale up training

    # # train image sizes
    # if args.train_img_sizes is not None:
    #     train_img_sizes = args.train_img_sizes
    # else:
    #     train_img_sizes = [args.train_img_size]

    train_img_sizes = [192, 256, 320, 352]
    n_sizes = len(train_img_sizes)

    # epochs regularize
    lambda_epoch = 25000 / train_length
    args.epochs = int(args.epochs * lambda_epoch)

    if args.epochs > n_sizes:
        args.epoch_per_stage = int(args.epochs / n_sizes)
    else:
        args.epoch_per_stage = args.epochs
        n_sizes = 1

    # training options
    scaler = GradScaler()
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.RMSprop(
        model.parameters(),
        lr=args.lr,
        alpha=0.9,
        eps=1e-8,
        weight_decay=args.weight_decay,
        momentum=args.momentum,
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50, eta_min=1e-6)
    # optimizer = optim.Adam(model.parameters(), lr=args.lr)
    # scheduler = CosineAnnealingWarmUpRestarts(optimizer, T_0=50, T_mult=1, eta_max=1e-3, T_up=15, gamma=0.5)

    # training
    for stage in range(1, n_sizes + 1):
        LOGGER.info(f"\n========== stage: {stage}/{n_sizes} training ==========")

        args.train_img_size = train_img_sizes[stage - 1]

        # # train dataset weighted sampling
        train_df_0 = train_df.loc[train_df.label == 0, :]
        train_df_1 = train_df.loc[train_df.label == 1, :]
        train_df_2 = train_df.loc[train_df.label == 2, :]

        label_length = 0
        if len(train_df_0) > len(train_df_1):
            label_length = len(train_df_1)
        else:
            label_length = len(train_df_0)

        train_df_0 = train_df_0.sample(n=label_length)
        train_df_1 = train_df_1.sample(n=label_length)
        train_df_weighted = pd.concat([train_df_0, train_df_1, train_df_2])

        # train dataset total size sampling
        if train_length > train_max:
            train_df_sample = train_df_weighted.sample(n=train_max).copy()
        else:
            train_df_sample = train_df_weighted.copy()

        del train_df_0, train_df_1, train_df_2, train_df_weighted

        # dataset
        train_dataset = CustomDataset(
            args=args,
            df=train_df_sample,
            transforms=get_transforms(img_size=args.train_img_size, data="train"),
        )

        # data loader
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=args.train_batch_size,
            shuffle=True,
            pin_memory=True,
            num_workers=args.num_workers,
        )

        # train start
        train_fn(
            args=args,
            model=model,
            device=args.device,
            train_loader=train_loader,
            val_loader=val_loader,
            writer=writer,
            LOGGER=LOGGER,
            scaler=scaler,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            stage=stage,
        )

        del train_loader

    end_time = time.time()
    LOGGER.info("\n Efficientnet Training Time : {}".format(end_time - start_time))
