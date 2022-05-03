"""
crop mask -> save with white background
"""
import pandas as pd
from detectron2.utils.logger import setup_logger

setup_logger()
import os
import cv2
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
import numpy as np
import glob
from tqdm import tqdm
import torch
import torch.nn.functional as F
from efficientnet.model import WireClassifier

from torchvision.datasets.coco import CocoDetection
import torchvision

import csv

import argparse
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader
from dataset import InferDataset, InferDataset2


parser = argparse.ArgumentParser(description="PyTorch Classification")
parser.add_argument(
    "--model_name", type=str, default="efficientnet_b4", metavar="S", help="model name"
)
parser.add_argument(
    "--mask_path",
    type=str,
    default="E:\\work\\kesco\\file_storage\\weights\\mask_rcnn_27_AP_98.01.pt",
    metavar="S",
    help="efficientnet b4 weights path",
)
parser.add_argument(
    "--clf_path",
    type=str,
    default="E:\\work\\kesco\\file_storage\\weights\\efficientnet_b4_17_loss_0.70_acc_99.16.pt",
    metavar="S",
    help="efficientnet b4 weights path",
)  # efficientnet_b4_17_acc_99.16_loss_0.70
parser.add_argument(
    "--num_workers", type=int, default=4, metavar="N", help="num workers"
)
parser.add_argument(
    "--clf_classes", type=int, default=2, metavar="N", help="clf classes"
)


# parser.add_argument('--dataset_dir', type=str, default="E:\\work\\kesco\\raw_data\\20211008", metavar='S',
#                     help='model path')
# parser.add_argument('--df_path', type=str, default="test_data.csv", metavar='S',
#                     help='model path')
# parser.add_argument('--save_dir_name', type=str, default="test_all_result_4", metavar='S',
#                     help='model path')


def get_predictor(args):
    cfg = get_cfg()
    cfg.merge_from_file(
        model_zoo.get_config_file(
            "COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml"
        )
    )
    cfg.DATASETS.TRAIN = ()
    cfg.DATALOADER.NUM_WORKERS = args.num_workers
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 8
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 27

    cfg.MODEL.WEIGHTS = args.mask_path
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7

    cfg.INPUT.MIN_SIZE_TEST = 720
    cfg.INPUT.MAX_SIZE_TEST = 1280
    cfg.INPUT.MIN_SIZE_TRAIN = 720
    cfg.INPUT.MAX_SIZE_TRAIN = 1280

    predictor = DefaultPredictor(cfg)
    return predictor


def load_mask(predictor, image_path):
    img = cv2.imread(image_path)

    if img is None:
        img_array = np.fromfile(image_path, np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

    outputs = predictor(img)

    # mask = outputs["instances"].pred_masks[0].to("cpu")
    # if len(outputs["instances"].pred_classes) > 1:
    #     for i in range(1,len(outputs["instances"].pred_classes)):
    #         mask = np.bitwise_or(mask,outputs["instances"].pred_masks[i].to("cpu"))

    output = outputs["instances"].to("cpu")
    wire_idxs = (output.pred_classes == 26).nonzero().squeeze()

    if wire_idxs.nelement() == 0:
        return 0, 0
    elif wire_idxs.nelement() == 1:
        wire_idx = wire_idxs.item()
    else:
        wire_idx = wire_idxs.numpy()[0]

    wire_prob = output.scores[wire_idx].item()
    mask = output.pred_masks[wire_idx]

    mask = np.greater(mask, 0)  # get only non-zero positive pixels/labels
    mask = np.expand_dims(mask, axis=-1)  # (H, W) -> (H, W, 1)
    mask = np.concatenate((mask, mask, mask), axis=-1)
    bit_and = np.multiply(img, mask)
    mask_0 = bit_and
    where_0 = np.where(mask_0 == 0)
    mask_0[where_0] = 255

    return mask_0, wire_prob


def imwrite(filename, img, params=None):
    try:
        ext = os.path.splitext(filename)[1]
        result, n = cv2.imencode(ext, img, params)

        if result:
            with open(filename, mode="w+b") as f:
                n.tofile(f)
                return True
        else:
            return False
    except Exception as e:
        print(e)
        return False


if __name__ == "__main__":
    from coco_load import CustomCocoDetection
    from torch.utils.data import DataLoader
    from torchvision import transforms

    # setting
    args = parser.parse_args()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # classes
    dict = {"electric": 0, "flame": 1, "indistinct": 2}
    thing_classes = [
        "airplane",
        "banana",
        "baseball bat",
        "carrot",
        "dining table",
        "fork",
        "giraffe",
        "hot dog",
        "keyboard",
        "knife",
        "nipped",
        "pen",
        "pizza",
        "scissors",
        "skateboard",
        "skis",
        "snowboard",
        "spoon",
        "sports ball",
        "stop sign",
        "surfboard",
        "tennis racket",
        "tie",
        "toothbrush",
        "train",
        "truck",
        "wire",
    ]

    # transform
    transform = A.Compose(
        [
            A.Resize(height=380, width=380),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ]
    )

    # detectron init
    predictor = get_predictor(args)

    # efficientnet_b4 init
    classifier = WireClassifier(num_classes=args.clf_classes)
    classifier.to(device)
    classifier.load_state_dict(torch.load(args.clf_path))
    classifier.eval()

    # # test list
    # image_list1 = glob.glob("E:\\work\\kesco\\kesco_training\\TEST DATA SET 홍효진\\*\\*\\*\\*.jpg")
    # image_list2 = glob.glob("E:\\work\\kesco\\kesco_training\\TEST DATA SET 홍효진\\*\\*\\*\\*\\*.jpg")
    #
    # image_list3 = glob.glob("E:\\work\\kesco\\file_storage\\train_data\\electric\\*.jpg")
    # image_list4 = glob.glob("E:\\work\\kesco\\file_storage\\train_data\\flame\\*.jpg")
    #
    # image_list5 = glob.glob("E:\\work\\kesco\\file_storage\\test_data\\electric\\*.jpg")
    # image_list6 = glob.glob("E:\\work\\kesco\\file_storage\\test_data\\flame\\*.jpg")
    #
    # image_list = image_list1 + image_list2 + image_list3 + image_list4 + image_list5 + image_list6
    #
    # infer_dataset = InferDataset(image_list)

    f = open(f"res.csv", "w", encoding="utf8", newline="")
    wr = csv.writer(f)
    wr.writerow(
        [
            "path",
            "is_train",
            "label",
            "pred",
            "wire_score",
            "electric_output",
            "flame_output",
            "score",
        ]
    )

    # image_list = glob.glob("E:\\work\\kesco\\raw_data\\stationary\\*.jpg") # + glob.glob("E:\\work\\kesco\\raw_data\\things\\*.jpg")
    image_list = glob.glob("E:\\work\\kesco\\raw_data\\nipper\\*.jpg")

    # coco_train = CustomCocoDetection(root="E:\\data\\COCO\\train2017",
    #                                  annFile="E:\\data\\COCO\\annotations\\instances_train2017.json"
    #                                  )
    #
    # coco_loader = DataLoader(coco_train)

    # for type in ["train", "test"]:
    #     df = pd.read_csv(f"E:\\work\\kesco\\file_storage\\{type}_data.csv")
    #     infer_dataset = InferDataset2(df, is_train=type)
    #     infer_loader = DataLoader(infer_dataset, num_workers=args.num_workers)

    image_list = ["E:\\work\\kesco\\raw_data\\photo_2022-01-19_16-42-05.jpg"]

    for idx, line in tqdm(enumerate(image_list)):
        # image_path = os.path.join("E:\\work\\kesco\\file_storage", line[0][0])
        # label = line[1][0].item()
        # is_train = line[2][0]

        image_path = line
        label = "nipped"
        is_train = "test"

        # label = 1
        # if "단락" in image_path.split("\\")[-2][:2]:
        #     label = 0
        # elif "electric" in image_path.split("\\")[-2]:
        #     label = 0
        #
        # is_train = "test2"
        # if "train" in image_path.split("\\")[-3]:
        #     is_train = "train"
        # elif "test" in image_path.split("\\")[-3]:
        #     is_train = "test"

        # label = 2
        # is_train = "train"

        mask, wire_prob = load_mask(predictor, image_path)

        image = cv2.imread(image_path)
        file_name = os.path.basename(image_path)

        # masking
        # mask_path = os.path.join("E:\\work\\kesco\\file_storage\\masked_image", line[0][0])
        # mask_path = image_path.replace("raw_data", "raw_data\\masked_image")
        mask_path = os.path.join(
            "E:\\work\\kesco\\kesco_training\\result_dataset",
            os.path.basename(image_path),
        )
        # mask_path = mask_path.replace(os.path.basename(mask_path), f"things_{idx}.jpg")
        os.makedirs(os.path.dirname(mask_path), exist_ok=True)

        if isinstance(mask, int):
            mask = cv2.imread(image_path)
            pred_label = 0  # cannot find wire
            output_0 = 0
            output_1 = 0
            score = 0

            imwrite(mask_path, mask)
            wr.writerow(
                [
                    file_name,
                    is_train,
                    label,
                    pred_label,
                    wire_prob,
                    output_0,
                    output_1,
                    score,
                ]
            )
            continue

        else:
            # cv2.imwrite(mask_path, mask)
            # imwrite(mask_path, image)
            pass

        # classify
        input = transform(image=mask[:, :, ::-1])["image"].unsqueeze(0)
        input = input.to(device)

        with torch.no_grad():
            pred = classifier(input)

        pred_label = pred.max(1)[1].to("cpu").numpy().item()
        output_0 = pred[0][0].to("cpu").numpy().item()
        output_1 = pred[0][1].to("cpu").numpy().item()
        score = torch.sigmoid(pred.max()).to("cpu").numpy().item()
        # score = F.softmax(pred).max(1).values.to("cpu").numpy().item()

        wr.writerow(
            [
                file_name,
                is_train,
                label,
                pred_label,
                wire_prob,
                output_0,
                output_1,
                score,
            ]
        )

    f.close()
