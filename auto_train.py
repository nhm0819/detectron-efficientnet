import argparse
import os
import shutil
import datetime
import time
from mask_rcnn.mask_train import mask_train
from mask_rcnn.inference import mask_inference
from efficientnet.clf_train import efficientnet_train
from efficientnet.utils import eff_select_best
from efficientnet.inference import eff_inference

import torch


parser = argparse.ArgumentParser(description="PyTorch Classification")

# base
parser.add_argument(
    "--data_path", type=str, default="", metavar="S", help="absolute path of data path"
)
parser.add_argument(
    "--file_storage_path",
    type=str,
    default="Z:\\private\\traning\\original",
    metavar="S",
    help="absolute path of file storage",
)
parser.add_argument(
    "--mask_path",
    type=str,
    default="Z:\\private\\traning\\original\\weights\\mask_rcnn_26_AP_98.33.pt",
    metavar="S",
    help="pretrained mask rcnn X101 path",
)
parser.add_argument(
    "--clf_path",
    type=str,
    default="Z:\\private\\traning\\original\\weights\\efficientnet_over_current_acc_98.55.pt",
    metavar="S",
    help="pretrained efficientnet b4 path",
)
parser.add_argument(
    "--seed", type=int, default=42, metavar="S", help="random seed (default: 42)"
)
parser.add_argument(
    "--cuda", action="store_true", default=True, help="enables CUDA training"
)
parser.add_argument(
    "--num_workers",
    type=int,
    default=6,
    metavar="N",
    help="how many training processes to use (default: 6)",
)
parser.add_argument(
    "--testing", action="store_true", help="not make all image to mask image"
)


# mask rcnn
parser.add_argument(
    "--mask_classes",
    type=int,
    default=26,
    metavar="N",
    help="num classes for mask rcnn (default : 26)",
)
parser.add_argument(
    "--mask_id",
    type=int,
    default=25,
    metavar="N",
    help="number of wire id (default : 25)",
)
parser.add_argument(
    "--mask_lr",
    type=float,
    default=1e-4,
    metavar="LR",
    help="learning rate (default: 0.0001)",
)
parser.add_argument(
    "--mask_iters",
    type=int,
    default=20000,
    metavar="N",
    help="number of iterations to train (default: 20000)",
)
parser.add_argument(
    "--mask_batch",
    type=int,
    default=4,
    metavar="N",
    help="number of batch size in mask rcnn (default: 4)",
)
parser.add_argument(
    "--mask_batch_per_images",
    type=int,
    default=512,
    metavar="N",
    help="number of batch size per image (default: 512)",
)
parser.add_argument(
    "--mask_checkpoint",
    type=int,
    default=2000,
    metavar="N",
    help="number of iters per checkpoint (default: 2000)",
)


## efficientnet preproc
parser.add_argument(
    "--train_img_sizes", type=int, nargs="+", metavar="N", help="train image sizes"
)  # --train_img_sizes 192 256 320 380
parser.add_argument(
    "--train_img_size",
    type=int,
    default=320,
    metavar="N",
    help="train image size (default: 320)",
)
parser.add_argument(
    "--test_img_size",
    type=int,
    default=380,
    metavar="N",
    help="test image size (default: 380)",
)
parser.add_argument(
    "--dataset_size",
    type=int,
    default=100000,
    metavar="N",
    help="maximum train dataset size",
)
parser.add_argument(
    "--target_col",
    type=str,
    default="label",
    metavar="S",
    help="target column name (default: label)",
)


# efficientnet hyperparams
parser.add_argument(
    "--train_batch_size",
    type=int,
    default=32,
    metavar="N",
    help="input batch size for training (default: 32)",
)
parser.add_argument(
    "--test_batch_size",
    type=int,
    default=32,
    metavar="N",
    help="input batch size for validation (default: 32)",
)
parser.add_argument(
    "--epochs",
    type=int,
    default=300,
    metavar="N",
    help="number of epochs to train (default: 300)",
)
parser.add_argument(
    "--num_classes",
    type=int,
    default=3,
    metavar="N",
    help="number of classes in classification (default: 3)",
)
parser.add_argument(
    "--lr",
    type=float,
    default=1e-4,
    metavar="LR",
    help="learning rate (default: 0.0001)",
)
parser.add_argument(
    "--momentum",
    type=float,
    default=0.9,
    metavar="M",
    help="RMSprop momentum (default: 0.9)",
)
parser.add_argument(
    "--weight_decay",
    type=float,
    default=0.00001,
    metavar="D",
    help="RMSprop weight decay (default: 0.00001)",
)
parser.add_argument(
    "--no_cast",
    action="store_true",
    default=False,
    help="Autocasting off (default: False)",
)


# log
parser.add_argument(
    "--val_per_epochs",
    type=int,
    default=20,
    metavar="N",
    help="validation per epoch (default: 20)",
)
parser.add_argument(
    "--test_per_epochs",
    type=int,
    default=20,
    metavar="N",
    help="test per epoch (default: 20)",
)
parser.add_argument(
    "--log-interval",
    type=int,
    default=10,
    metavar="N",
    help="how many batches to wait before logging training status",
)
parser.add_argument(
    "--start_time",
    type=str,
    default="no_input",
    metavar="S",
    help="the time when training operation is started in nodejs",
)


def train_all(args):

    if len(args.data_path) == 0:
        args.data_path = args.file_storage_path

    args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # result folder path
    args.output_dir = os.path.join(args.file_storage_path, "train_results")

    # start time record
    if args.start_time == "no_input":
        args.start_time = datetime.datetime.now().strftime("%Y-%m-%d_%Hh%Mm%Ss")

    print("---------Mask RCNN Training---------")

    # mask rcnn output folder
    args.mask_output_dir = os.path.join(
        args.output_dir, "mask_rcnn", f"{args.start_time}"
    )
    if not os.path.exists(args.mask_output_dir):
        os.makedirs(args.mask_output_dir, exist_ok=True)

    ### mask rcnn training
    print(f'output directory : "{args.mask_output_dir}"')
    mask_train(args)

    # memory clear
    torch.cuda.empty_cache()

    ### mask image extracting
    print("--------Saving Mask Images--------")

    # csv file path
    args.train_csv = os.path.join(args.data_path, "train_data.csv")
    args.test_csv = os.path.join(args.data_path, "test_data.csv")
    args.indistinct_csv = os.path.join(args.file_storage_path, "indistinct_data.csv")

    # masked image folder path
    args.mask_folder = os.path.join(args.file_storage_path, "masked_image")

    # efficientnet output folder
    args.eff_output_dir = os.path.join(
        args.output_dir, "efficientnet", f"{args.start_time}"
    )
    if not os.path.exists(args.eff_output_dir):
        os.makedirs(args.eff_output_dir, exist_ok=True)

    if not args.testing:
        if (round(args.AP, 3) + 0.4) > float(
            args.mask_path.split("_")[-1][:-3]
        ):  # 'args.AP' is made in 'mask_train.py'
            args.mask_path = args.mask_best_model
            mask_inference(args, renewal=True)

        else:
            mask_inference(args)

    # memory clear
    torch.cuda.empty_cache()

    print("---------EfficientNet Training---------")

    ### efficientnet training
    # try:
    #     efficientnet_train(args)
    #     torch.cuda.empty_cache()
    # except:
    #     print("Error : EfficientNet Training Error")
    #     shutil.rmtree(args.mask_output_dir)
    #     shutil.rmtree(args.eff_output_dir)
    #     raise

    efficientnet_train(args)
    torch.cuda.empty_cache()

    eff_select_best(args)
    eff_inference(args)


if __name__ == "__main__":
    args = parser.parse_args()

    start_time = time.time()
    train_all(args)
    end_time = time.time()

    print(f"Total Training Time : {(end_time-start_time)/3600:.2f} hours")
